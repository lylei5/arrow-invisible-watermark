# ARROW 隐形水印解码服务

ARROW 字标（2200×600）里嵌了一组亚像素几何水印，写在 O / R / A 四个字的内孔轮廓上，
字母外轮廓逐比特不变。这个仓库是**解码侧**的完整服务：FastAPI 包一层 HTTP，
给小程序 / 客户端调用。

线上入口（自建服务器，2026-11 起）：

```
POST https://arrow.liylei666.online/decode   # multipart，字段 image
GET  https://arrow.liylei666.online/health   # 存活探针，不鉴权
```

## 目录

```
arrow_decode.py            解码主流程（配准 → 采样 → 判决）
arrow_wm.py                水印编解码 / 版式 / 模板定义
arrow_inner.py             内孔载体的几何与模板
ARROW_clean_pro.png        母版字标（SIFT 模板）
services/decode_api/
  app.py                   FastAPI：POST /decode、GET /health
  requirements.txt         运行依赖（numpy / opencv-headless / fastapi / uvicorn）
samples/inner/             一张可用于冒烟测试的内孔样张
tools/smoke_test.py        命令行冒烟测试（只用标准库）
Dockerfile                 容器镜像
docs/DEPLOY.md             服务器部署：Docker + nginx + 证书
```

## 接口

### `POST /decode`

| 位置 | 名字 | 必填 | 说明 |
|---|---|---|---|
| multipart | `image` | 是 | 照片。也接受 `file` / `img` / `photo` / `picture` |
| 查询或表单 | `carrier` | 否 | `inner`（默认）或 `edge`。旧边缘印版才传 `edge` |
| 头 | `X-Decode-Token` | 服务端设了 `DECODE_TOKEN` 时 | 与变量值一致 |
| 头 | `X-Image-SHA256` | 否 | 客户端算的图片哈希；带上会回显并可用于核对 |

也接受 `application/json` 的 `{"image_base64": "..."}` 和原始字节体（`application/octet-stream`）。

### 响应

解码本身（包括「这张图没有可信号码」）是 HTTP 200，`status` 只会是 `accepted` / `rejected`：

| 字段 | 含义 |
|---|---|
| `status` | `accepted` / `rejected`（请求非法时才是 `error`，且 HTTP 非 200） |
| `accepted` | 与 `status === "accepted"` 同一件事，**成功只认这个** |
| `digits` | 采纳时的 11 位号码，拒绝时为 `null` |
| `margin` / `margin_min` | 码字裕度均值 / 最弱一组；采纳线 `1.8` / `0.55` |
| `cc` | 配准相关，采纳线 `0.85` |
| `logo_scale` | 字标宽度占画面宽度的比例；低于 `0.08` 会以 `low_logo_scale` 拒绝 |
| `reason` / `reason_text` | 机器可读 / 中文原因 |
| `input_sha256` | 服务端实际收到的字节哈希 |

不要用 `ok` 判断成功（无码图也会是 `true`），也不要构造 `digits` 之外的结果。

请求不合法：`400`（缺字段 / 载体非法）、`401`（令牌不符）、`413`（超过 20 MB 或 4000 万像素）。
上限：`MAX_IMAGE_BYTES = 20 MB`，`MAX_PIXELS = 40 MP`。

### `GET /health`

`{"status":"ok"}`，不校验 `X-Decode-Token`，给探针用。

## 本地跑

```bash
pip install -r services/decode_api/requirements.txt
python -m uvicorn services.decode_api.app:app --host 127.0.0.1 --port 8501
```

启动时加载 SIFT 模板，日志出现 `[decode_api] ready` 后 `/health` 才通。

## 冒烟测试

```bash
python tools/smoke_test.py \
  --url http://127.0.0.1:8501/decode \
  --image samples/inner/ARROW_13800138000.png \
  --token "$DECODE_TOKEN"
```

期望 `accepted=true`、`digits=13800138000`。

## 镜像

```bash
docker build -t arrow-decode:latest .
docker run -d --name arrow-decode --restart unless-stopped \
  -p 127.0.0.1:8601:8501 -e DECODE_TOKEN=... arrow-decode:latest
```

部署到公网（nginx 反代 + Let's Encrypt）见 [docs/DEPLOY.md](docs/DEPLOY.md)。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8501` | 监听端口 |
| `DECODE_TOKEN` | 空 | 非空时 `POST /decode` 校验 `X-Decode-Token`；`/health` 始终不校验 |
| `DECODE_WARMUP` | `1` | 启动预热：`1` 全部 / `inner` / `edge` / `0` 不预热 |
| `DECODE_INNER_LAYOUT` | `3` | 内孔版式（v4 生产版式） |
| `WEB_CONCURRENCY` | `1` | worker 数；每个 worker 各持一套模板 |

## 许可与来源

算法与模板来自 ARROW 隐形水印项目；本仓库只保留解码服务运行所需的代码。
