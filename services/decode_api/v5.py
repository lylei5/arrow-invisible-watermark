"""Authenticated ARROW v5 photo decoder with bounded single-flight inference."""
import os
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '40000000')
import hashlib
import re
import secrets
import time
import sqlite3
from contextlib import asynccontextmanager
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from arrow_service import Decoder, FORMAT, MASTER_ID
from arrow_decode import reason_text
from .counts import VerificationStore, RequestConflict

cv2.setNumThreads(1)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
RELEASE = 'vector-v5-counts-20261108'
_decoder = None
_store = None
_busy = False

@asynccontextmanager
async def lifespan(app):
    global _decoder, _store
    if not os.environ.get('DECODE_TOKEN'):
        raise RuntimeError('DECODE_TOKEN must be configured')
    _store = await run_in_threadpool(VerificationStore, os.environ.get('ARROW_DB_PATH', '/var/lib/arrow/verifications.sqlite3'))
    _decoder = await run_in_threadpool(Decoder)
    sample = _decoder.carrier.embed('01088886666')['plate']
    result = await run_in_threadpool(_decoder.decode, sample)
    if not result.accepted or result.digits != '01088886666':
        raise RuntimeError('V5 startup self-test failed')
    yield
    _decoder = None
    _store = None

app = FastAPI(title='ARROW v5 decoder', version='5.1.0', lifespan=lifespan)

class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > MAX_IMAGE_BYTES + 1024 * 1024:
                raise HTTPException(413, '图像超过 20 MiB，请裁剪后重试')
            return message
        await self.app(scope, limited_receive, send)

app.add_middleware(BodyLimit)

@app.get('/health')
def health():
    if _decoder is None:
        raise HTTPException(503, 'warming up')
    return dict(status='ok', ready=True, format=FORMAT, master_id=MASTER_ID,
                release=RELEASE, layout_version=5, max_image_bytes=MAX_IMAGE_BYTES,
                max_pixels=MAX_PIXELS, verification_counting=True,
                counting_since=_store.counting_since)

def read_image(data):
    supported = (data.startswith(b'\x89PNG\r\n\x1a\n') or data.startswith(b'\xff\xd8\xff')
                 or (data.startswith(b'RIFF') and data[8:12] == b'WEBP'))
    if not supported:
        raise HTTPException(415, '请上传 JPEG、PNG 或 WebP 图像')
    try:
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        raise HTTPException(422, '图像无效或分辨率超过 4000 万像素')
    if gray is None or min(gray.shape) < 24:
        raise HTTPException(422, '无法读取有效图像')
    if gray.size > MAX_PIXELS:
        raise HTTPException(413, '图像超过 4000 万像素，请裁剪后重试')
    return gray

@app.post('/decode')
async def decode(request: Request):
    global _busy
    expected = os.environ.get('DECODE_TOKEN', '')
    supplied = request.headers.get('x-decode-token', '')
    if not expected or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, '解码服务认证失败')
    if request.headers.get('x-arrow-format') != FORMAT:
        raise HTTPException(409, '请使用 ARROW v5 小程序')
    digest = request.headers.get('x-image-sha256', '').lower()
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise HTTPException(400, '缺少有效的图像 SHA-256')
    request_id = request.headers.get('x-verification-id', '')
    if not re.fullmatch('[A-Za-z0-9_-]{20,100}', request_id):
        raise HTTPException(400, '缺少有效的验证请求编号，请更新小程序')
    if _decoder is None:
        raise HTTPException(503, '解码服务正在初始化')
    if _busy:
        raise HTTPException(429, '解码服务正在处理照片，请稍后重试', headers={'Retry-After':'2'})
    _busy = True
    start = time.perf_counter()
    try:
        kind = request.headers.get('content-type', '').split(';')[0].lower()
        if kind == 'multipart/form-data':
            async with request.form(max_files=1, max_fields=2) as form:
                upload = form.get('image')
                if not isinstance(upload, UploadFile):
                    raise HTTPException(400, '缺少 image 图像文件')
                data = await upload.read(MAX_IMAGE_BYTES + 1)
        elif kind in ('image/png', 'image/jpeg', 'image/webp', 'application/octet-stream'):
            data = await request.body()
        else:
            raise HTTPException(415, '请使用图像上传接口')
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, '图像为空或超过 20 MiB')
        actual = hashlib.sha256(data).hexdigest()
        if not secrets.compare_digest(digest, actual):
            raise HTTPException(422, '图像完整性校验失败，请重新拍摄')
        cached = await run_in_threadpool(_store.lookup, request_id, actual)
        if cached is not None:
            return cached
        gray = await run_in_threadpool(read_image, data)
        inference_start = time.perf_counter()
        result = await run_in_threadpool(_decoder.decode, gray)
        payload = result.public_dict()
        payload.pop('candidate_digits', None)
        payload.update(format=FORMAT, master_id=MASTER_ID, release=RELEASE,
                       layout_version=5, input_sha256=actual,
                       input_size={'width':gray.shape[1], 'height':gray.shape[0]},
                       reason_text=reason_text(result.reason),
                       decode_ms=round((time.perf_counter()-inference_start)*1000, 1),
                       server_ms=round((time.perf_counter()-start)*1000, 1))
        if result.accepted:
            payload = await run_in_threadpool(_store.record, request_id, actual, payload)
        else:
            payload.update(verification_count=None, verification_id=request_id,
                           verification_replayed=False, counting_since=_store.counting_since)
        return payload
    except RequestConflict as error:
        raise HTTPException(409, str(error))
    except sqlite3.Error:
        raise HTTPException(503, '验证记录暂时不可用，请稍后重试')
    finally:
        _busy = False
