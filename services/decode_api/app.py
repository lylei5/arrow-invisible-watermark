"""HTTP wrapper around ``arrow_decode`` for the mini-program and TI-ONE.

One :class:`arrow_decode.Decoder` is cached per carrier. Decode calls take a
process-wide lock so OpenCV's matcher is not used re-entrantly. The accept gate lives in ``arrow_decode``. This module maps it to JSON:
``status`` is ``accepted``, ``rejected``, or ``error``. ``ok`` is not a
success bit. ``candidate_digits`` is omitted unless ``debug`` is set and
``DECODE_TOKEN`` is configured (the request must already have matched it).

Listen on ``0.0.0.0:$PORT`` (default **8501**). ``POST /decode`` keeps
multipart, and also accepts JSON ``image_base64`` or a raw image body so
``wx.cloud.callContainer`` can send a photo without building a form.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

# 云托管实例是 0.5 CPU：OpenCV 默认按可见核数开线程，在这种小实例上线程争抢
# 反而更慢、峰值内存更高（容易把 1 GB 的实例顶掉）。进程内解码本来就被一把锁
# 串行化，所以限制成单线程没有任何吞吐损失。
try:
    cv2.setNumThreads(1)
except Exception:                      # pragma: no cover - 防御性
    pass

# Repo root holds arrow_decode.py / arrow_wm.py / arrow_inner.py. Insert it so
# `uvicorn app:app` from this directory and `uvicorn services.decode_api.app:app`
# from the repo root both import the library.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import arrow_decode as AD

REPO_ROOT = _ROOT

# 内孔只有一种版式：v4（NRZ，77 码元，每码元 0.328 mm）。旧 v2/v3 印版已停产，
# 服务端不再支持多版式共存，也不读任何版式环境变量——少一个能配错的地方。
INNER_LAYOUT = 4

# Preferred multipart field, then common client aliases. Any other file part
# is still accepted so a renamed form field does not 400.
IMAGE_FIELDS = ("image", "file", "img", "photo", "picture")

# Phone photos are a few megabytes. Caps are abuse limits, not decode knobs.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000

# Base64 expands by 4/3. Cap the text before decoding so a 1GB instance is
# not asked to materialize an unbounded body. Raw image bodies use
# MAX_IMAGE_BYTES directly.
_BASE64_CHAR_CAP = MAX_IMAGE_BYTES * 2

# callContainer can hand the photo over as an ArrayBuffer. These media types
# are the body itself (carrier stays on the query string).
_RAW_MEDIA_TYPES = frozenset({
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
})

_MS_SEGMENT = re.compile(r"^ms-[A-Za-z0-9_-]+$")

# RLock so decode_image can call get_decoder while already holding the lock.
# The lock stays held for the whole decode: OpenCV's matcher is not re-entrant.
_lock = threading.RLock()
# Edge sessions are keyed by layout version. Inner ignores layout.
_sessions: dict[tuple, AD.Decoder] = {}


def strip_ms_prefix(path: str) -> str:
    """Drop leading ``/ms-<id>`` path segments.

    The public gateway URL is ``https://<host>/ms-<id>/decode``. TI-ONE usually
    strips that prefix before the container. A few routes (some CLB examples)
    forward it once or twice. The app's own routes stay ``/decode`` and
    ``/health``.
    """
    rest = path or "/"
    while True:
        parts = rest.split("/", 2)
        if len(parts) < 2 or not _MS_SEGMENT.match(parts[1] or ""):
            return rest or "/"
        if len(parts) == 2 or not parts[2]:
            return "/"
        rest = "/" + parts[2]


class GatewayPrefixMiddleware:
    """Rewrite ``/ms-<id>/...`` onto the app path. Does not read the body."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path") or "/"
            stripped = strip_ms_prefix(path)
            if stripped != path:
                scope = dict(scope)
                scope["path"] = stripped
                scope["raw_path"] = stripped.encode("utf-8")
        await self.app(scope, receive, send)


def _warmup_carriers() -> list[str]:
    """Which sessions to build before serving.

    ``DECODE_WARMUP=1`` (default) loads inner and edge.
    ``DECODE_WARMUP=inner`` or ``default`` loads only the production default.
    ``DECODE_WARMUP=edge`` loads only the edge carrier (legacy plates).
    ``DECODE_WARMUP=0`` skips warmup (tests).
    """
    flag = os.environ.get("DECODE_WARMUP", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return []
    if flag in ("inner", "default"):
        return [AD.DEFAULT_CARRIER]
    if flag == "edge":
        return ["edge"]
    return [AD.DEFAULT_CARRIER, "edge"]


def get_decoder(carrier: str) -> AD.Decoder:
    """Return the cached session for ``carrier``, building it on first use.

    Inner always uses the v4 NRZ format; edge keeps its own legacy decoder.
    There is no layout selector any more.
    """
    if carrier not in AD.CARRIERS:
        raise ValueError(f"carrier must be one of {AD.CARRIERS}, got {carrier!r}")
    version = None if carrier == "edge" else INNER_LAYOUT
    key = (carrier, version)
    with _lock:
        session = _sessions.get(key)
        if session is None:
            session = AD.Decoder(carrier=carrier, layout_version=version)
            _sessions[key] = session
        return session


def decode_image(carrier: str, gray: np.ndarray) -> AD.DecodeResult:
    """Run one production decode under the process-wide lock.

    Inner plates are always the v4 NRZ format. Edge keeps its own legacy
    decoder — there is no layout selector any more.
    """
    with _lock:
        return get_decoder(carrier).decode(gray)


def candidate_debug_allowed(debug: bool) -> bool:
    """Candidate digits leave the process only for an authenticated debug call.

    ``DECODE_TOKEN`` unset means there is no valid token, so debug cannot
    reveal a candidate. ``check_token`` has already rejected a mismatch.
    """
    if not debug:
        return False
    return bool(os.environ.get("DECODE_TOKEN", "").strip())


def _reason_text_with_numbers(result: AD.DecodeResult,
                             input_size: Optional[tuple] = None) -> str:
    """把拒绝原因翻成**带数值**的中文，便于用户把屏幕上那句话直接报给支持。

    ``AD.reason_text`` 只给一句"画面中 Logo 过小"，看不出到底差多少；现场反馈
    时无法区分"拍太远"和"裕度不够"。这里把触发的那条判据和实测值都写进去，
    并附上服务端实际收到的图片尺寸——那是判断"客户端传的是紧凑裁剪还是整张照片"
    的唯一可靠线索（两家尺寸差一个数量级）。
    """
    reason = str(result.reason or "")
    if result.accepted:
        return AD.reason_text(reason)

    parts: list[str] = []
    ls = result.logo_scale
    if "low_logo_scale" in reason or "low_logo_scale_v2" in reason:
        need = AD.LOGO_SCALE_MIN
        if ls is not None:
            parts.append(f"画面中 Logo 过小（logo_scale={ls:.3f}，需 ≥{need:.2f}）")
        else:
            parts.append("画面中 Logo 过小")
    if "low_cc" in reason:
        parts.append(f"配准不足（cc={result.cc:.2f}，需 ≥{AD.CC_THRESHOLD:.2f}）")
    if "crc_mismatch" in reason:
        parts.append("校验码不符（CRC 未通过，多半是拍糊了或距离太远）")
    if "low_group_margin" in reason:
        parts.append(f"单组裕度不足（margin_min={result.margin_min:.2f}）")
    if "low_margin" in reason:
        parts.append(f"码字裕度不足（margin={result.margin:.2f}，需 ≥{result.margin_threshold:.1f}）")
    if "invalid_payload" in reason:
        parts.append("载荷不完整")

    # 诊断尾巴：图片尺寸 + logo_scale，现场报障时最有用
    diag = []
    if input_size:
        diag.append(f"图片 {int(input_size[0])}×{int(input_size[1])}")
    if ls is not None:
        diag.append(f"logo_scale={ls:.3f}")
    if diag:
        parts.append("〔" + "，".join(diag) + "〕")

    if parts:
        return "；".join(parts)
    return AD.reason_text(reason)


def result_payload(result: AD.DecodeResult, *,
                   include_candidate: bool = False,
                   input_sha256: Optional[str] = None,
                   input_size: Optional[tuple] = None) -> dict:
    """``DecodeResult.public_dict()`` plus Chinese ``reason_text``.

    ``candidate_digits`` is removed unless ``include_candidate``. Clients
    must treat only ``accepted === true`` (equivalently ``status ==
    "accepted"``) as a phone number. ``ok`` is the research decoder's sync
    flag and is true on many rejects.
    """
    body = result.public_dict()
    body["reason_text"] = _reason_text_with_numbers(result, input_size)
    # wx.uploadFile 直连分支会用这个字段核对"服务端收到的就是客户端发的"。
    body["input_sha256"] = input_sha256
    if input_size:
        body["input_size"] = [int(input_size[0]), int(input_size[1])]
    if not include_candidate:
        body.pop("candidate_digits", None)
    return body


class UTF8JSONResponse(JSONResponse):
    """UTF-8 JSON so ``reason_text`` is not ``\\u`` escaped. Rejects NaN."""

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


def check_token(provided: Optional[str]) -> None:
    """Require ``X-Decode-Token`` only when ``DECODE_TOKEN`` is set.

    Unset or blank means no auth (local dev). ``GET /health`` does not call
    this, so a platform probe does not need the secret.
    """
    expected = os.environ.get("DECODE_TOKEN", "").strip()
    if not expected:
        return
    got = "" if provided is None else provided.strip()
    if not secrets.compare_digest(got, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def ensure_upload_size(nbytes: int) -> None:
    if nbytes > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")


def fetch_url_bytes(url: str) -> bytes:
    """Fetch an image over http(s).

    ``callContainer`` 的请求体有 100KB 硬上限（官方 FAQ 错误码 -606001），
    所以客户端改成先把原图传到「对象存储」，再只把这个临时 URL 发过来。
    这里只允许 http(s)、限时限量，避免被当成任意 URL 的代理。
    """
    if not url.lower().startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail="url must be http(s)")
    request = urllib.request.Request(
        url, headers={"User-Agent": "arrow-decode/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"fetch failed: {exc}")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    return data


def bytes_to_gray(data: bytes) -> np.ndarray:
    """Grayscale image, the same conversion as ``arrow_wm.imread_u``.

    Undecodable bytes become an empty array. ``Decoder`` then returns
    ``reason=unreadable`` instead of raising.
    """
    if not data:
        return np.empty((0,), np.uint8)
    buf = np.frombuffer(data, dtype=np.uint8)
    gray = cv2.imdecode(np.ascontiguousarray(buf), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.size == 0:
        return np.empty((0,), np.uint8)
    if int(gray.size) > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="image too large")
    return gray


def media_type(content_type: str) -> str:
    """``Content-Type`` without parameters, lowercased."""
    return (content_type or "").split(";", 1)[0].strip().lower()


def carrier_from_value(raw: Optional[str]) -> str:
    """Normalize a carrier string. Blank means the production default."""
    text = "" if raw is None else str(raw).strip().lower()
    if not text:
        return AD.DEFAULT_CARRIER
    if text not in AD.CARRIERS:
        raise HTTPException(
            status_code=400, detail="carrier must be 'edge' or 'inner'")
    return text


def pick_carrier(query_value: Optional[str], form) -> str:
    """Query ``carrier`` wins over the form field. Default is ``inner``."""
    raw = query_value
    if raw is None or str(raw).strip() == "":
        item = form.get("carrier")
        if item is not None and not isinstance(item, UploadFile):
            raw = str(item)
    return carrier_from_value(None if raw is None else str(raw))


def decode_base64_image(text: str) -> bytes:
    """Decode a base64 image. A leading ``data:*;base64,`` prefix is stripped."""
    raw = text.strip()
    if raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[-1]
    raw = "".join(raw.split())
    if len(raw) > _BASE64_CHAR_CAP:
        raise HTTPException(status_code=413, detail="image too large")
    pad = (-len(raw)) % 4
    if pad:
        raw += "=" * pad
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_base64")
    return data


def declared_length(request: Request) -> Optional[int]:
    raw = request.headers.get("content-length")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def finish_decode(carrier: str, data: bytes, *,
                        include_candidate: bool = False):
    """Shared tail for every ``POST /decode`` body shape."""
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    ensure_upload_size(len(data))
    gray = bytes_to_gray(data)
    try:
        result = await run_in_threadpool(decode_image, carrier, gray)
    except HTTPException:
        raise
    except Exception:
        return UTF8JSONResponse(
            {"status": "error", "detail": "decode error"}, status_code=500)
    return UTF8JSONResponse(
        result_payload(result, include_candidate=include_candidate,
                       input_sha256=hashlib.sha256(data).hexdigest(),
                       input_size=(gray.shape[1], gray.shape[0])))


async def decode_json_body(request: Request, query_carrier: Optional[str], *,
                           include_candidate: bool = False):
    """``{"image_base64": "...", "carrier": "inner"}``. Query carrier wins."""
    length = declared_length(request)
    if length is not None and length > _BASE64_CHAR_CAP + 4096:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    raw_carrier = query_carrier
    if raw_carrier is None or str(raw_carrier).strip() == "":
        if payload.get("carrier") is not None:
            raw_carrier = payload.get("carrier")
    chosen = carrier_from_value(None if raw_carrier is None else str(raw_carrier))
    # 对象存储直链：客户端先 uploadFile，再只发这个临时 URL。
    # 用来绕开 callContainer 的 100KB 请求体上限（-606001），
    # 好处是原图不必压缩，服务端拿到的是完整分辨率。
    remote = payload.get("url")
    if isinstance(remote, str) and remote.strip():
        data = await run_in_threadpool(fetch_url_bytes, remote.strip())
        return await finish_decode(
            chosen, data, include_candidate=include_candidate)
    image = payload.get("image_base64")
    if not isinstance(image, str):
        image = payload.get("image")
    if not isinstance(image, str) or not image.strip():
        raise HTTPException(status_code=400, detail="missing image_base64")
    return await finish_decode(
        chosen, decode_base64_image(image), include_candidate=include_candidate)


async def decode_raw_body(request: Request, query_carrier: Optional[str], *,
                          include_candidate: bool = False):
    """Raw image bytes. ``carrier`` is the query param (default ``inner``)."""
    length = declared_length(request)
    if length is not None and length > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    data = await request.body()
    return await finish_decode(
        carrier_from_value(query_carrier), data,
        include_candidate=include_candidate)


def select_upload(form) -> Optional[UploadFile]:
    for name in IMAGE_FIELDS:
        item = form.get(name)
        if isinstance(item, UploadFile):
            return item
    for _key, item in form.multi_items():
        if isinstance(item, UploadFile):
            return item
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """先让 uvicorn 开始监听，再把解码器预热放到后台线程。

    云托管的小实例（0.5 CPU）就绪探针在容器启动后约 2 秒就开始打 /health，
    而 import cv2/numpy + 建两个解码器要好几秒。如果预热卡在 lifespan 里
    （uvicorn 在 yield 之前不监听端口），探针会连续失败、实例一直不 Ready，
    网关就返回 503。预热改成后台线程后，端口几乎立刻可用。
    """
    carriers = _warmup_carriers()

    def _warm() -> None:
        for carrier in carriers:
            print(f"[decode_api] loading {carrier} decoder", flush=True)
            try:
                get_decoder(carrier)
            except Exception as exc:          # 预热失败不该拖死整个进程
                print(f"[decode_api] warmup {carrier} failed: {exc!r}", flush=True)
        if carriers:
            print("[decode_api] ready", flush=True)

    if carriers:
        threading.Thread(target=_warm, name="warmup", daemon=True).start()
    yield


app = FastAPI(title="ARROW decode", version="1.0.0", lifespan=lifespan)
app.add_middleware(GatewayPrefixMiddleware)


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    """Bad requests and auth failures are ``status=error``, not a decode."""
    return UTF8JSONResponse(
        {"status": "error", "detail": exc.detail},
        status_code=exc.status_code,
        headers=exc.headers,
    )


@app.get("/health")
def health():
    # Liveness only. This "ok" is not a decode success.
    return {"status": "ok"}


@app.post("/decode", response_model=None)
async def decode_endpoint(
    request: Request,
    carrier: Optional[str] = Query(default=None),
    debug: bool = Query(default=False),
    x_decode_token: Optional[str] = Header(default=None, alias="X-Decode-Token"),
):
    check_token(x_decode_token)
    include_candidate = candidate_debug_allowed(debug)
    kind = media_type(request.headers.get("content-type", ""))
    if kind == "application/json":
        return await decode_json_body(
            request, carrier, include_candidate=include_candidate)
    if kind in _RAW_MEDIA_TYPES:
        return await decode_raw_body(
            request, carrier, include_candidate=include_candidate)
    if kind != "multipart/form-data":
        raise HTTPException(status_code=400, detail="expected multipart/form-data")
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="expected multipart/form-data")
    try:
        chosen = pick_carrier(carrier, form)
        upload = select_upload(form)
        if upload is None:
            raise HTTPException(
                status_code=400,
                detail="missing image; send multipart field 'image'",
            )
        data = await upload.read()
    finally:
        await form.close()
    return await finish_decode(
        chosen, data, include_candidate=include_candidate)
