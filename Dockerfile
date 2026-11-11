# ARROW v5 解码 HTTP 服务镜像
#
# 与官方 python:3.11-slim-bookworm 一致，只是把 Debian / PyPI 源换成腾讯云镜像
# （deb.debian.org 在国内服务器上经常慢到超时）。海外机器可把这两行镜像地址换回官方源。
#
# 构建（上下文 = 仓库根）：
#   docker build -t arrow-decode:v5 .
#
# 运行：
#   docker run -d --name arrow-decode-v5 --restart unless-stopped \
#     -p 127.0.0.1:8603:8501 -v arrow-v5-data:/var/lib/arrow \
#     -e DECODE_TOKEN=<token> arrow-decode:v5

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/arrow \
    PORT=8501 \
    ARROW_DB_PATH=/var/lib/arrow/verifications.sqlite3 \
    OPENCV_IO_MAX_IMAGE_PIXELS=40000000 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /opt/arrow

RUN sed -i 's|deb.debian.org|mirrors.cloud.tencent.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/decode_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://mirrors.cloud.tencent.com/pypi/simple -r /tmp/requirements.txt

COPY arrow_decode.py arrow_wm.py arrow_inner.py /opt/arrow/
COPY arrow_vector.py arrow_v5.py arrow_camera.py arrow_service.py /opt/arrow/
COPY services/__init__.py /opt/arrow/services/__init__.py
COPY services/decode_api /opt/arrow/services/decode_api

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/arrow \
    && chown -R appuser:appuser /opt/arrow /var/lib/arrow
USER appuser

EXPOSE 8501

# start-period covers SIFT template load. /health is intentionally unauthenticated.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8501'), timeout=4)"

# 同进程内解码由一把锁串行；要多核并行设 WEB_CONCURRENCY=N（每个 worker 各持一套模板）。
CMD ["sh", "-c", "exec python -m uvicorn services.decode_api.app:app --host 0.0.0.0 --port ${PORT:-8501} --workers ${WEB_CONCURRENCY:-1}"]
