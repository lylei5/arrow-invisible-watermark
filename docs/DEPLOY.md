# 部署：Docker + nginx + Let's Encrypt

线上实例（2026-11 起）：

| 项 | 值 |
|---|---|
| 服务器 | 腾讯云轻量 `43.142.77.96`（Ubuntu 24.04，4C8G） |
| 容器 | `arrow-decode`，镜像 `arrow-decode:20261104`，绑 `127.0.0.1:8601` |
| 域名 | `arrow.liylei666.online`（A 记录指向服务器） |
| 证书 | Let's Encrypt，certbot 自动续期 |

同机还跑着景德镇 57-bit 陶瓷解码（`jingde.liylei666.online` → `127.0.0.1:8501`），
两条服务用不同容器、不同端口、不同 token，互不影响。

## 1. 构建镜像

```bash
# 在服务器上，仓库根目录
docker build -t arrow-decode:$(date +%Y%m%d) .
```

Dockerfile 已把 Debian / PyPI 源换成腾讯云镜像；海外机器可改回官方源。

国内服务器拉基础镜像可能被墙，可先走镜像再打 tag：

```bash
docker pull mirror.ccs.tencentyun.com/library/python:3.11-slim-bookworm
docker tag mirror.ccs.tencentyun.com/library/python:3.11-slim-bookworm python:3.11-slim-bookworm
```

## 2. 运行容器

token 只放在服务器上，不进仓库：

```bash
sudo install -d -m 700 /etc/arrow
sudo sh -c 'printf "DECODE_TOKEN=%s\n" "$(python3 -c "import secrets;print(secrets.token_hex(32))")" > /etc/arrow/decoder.env'
sudo chmod 600 /etc/arrow/decoder.env

docker run -d --name arrow-decode --restart unless-stopped \
  --cpus 2 --memory 3g --memory-swap 3g \
  --env-file /etc/arrow/decoder.env \
  -p 127.0.0.1:8601:8501 \
  --log-opt max-size=10m --log-opt max-file=3 \
  arrow-decode:latest
```

端口只绑 `127.0.0.1`，公网流量一律走 nginx。

## 3. nginx 反代

`/etc/nginx/sites-available/arrow.liylei666.online`：

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name arrow.liylei666.online;

    client_max_body_size 25m;          # 服务端上限 20 MB，留余量

    location / {
        proxy_pass http://127.0.0.1:8601;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        proxy_connect_timeout 10s;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/arrow.liylei666.online /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## 4. 证书

```bash
apt install -y certbot python3-certbot-nginx
certbot --nginx -d arrow.liylei666.online --redirect
certbot renew --dry-run          # 验证自动续期
```

云厂商防火墙需放行 TCP 80（HTTP-01 验证）和 443。

## 5. 验收

```bash
curl -fsS https://arrow.liylei666.online/health
python3 tools/smoke_test.py \
  --url https://arrow.liylei666.online/decode \
  --image samples/inner/ARROW_13800138000.png \
  --token "$(sudo sed -n 's/^DECODE_TOKEN=//p' /etc/arrow/decoder.env)"
```

期望：`/health` 返回 `{"status":"ok"}`，解码返回 `accepted=true`、`digits=13800138000`；
不带 token 返回 401，SHA 不符返回 422。

## 6. 更新镜像

```bash
docker build -t arrow-decode:$(date +%Y%m%d) .
docker rm -f arrow-decode
# 重新 docker run（env-file 不变）
```

容器启动要加载 SIFT 模板，`/health` 大约 10–20 秒后才通（healthcheck start-period 120s）。

## 7. 排障

| 现象 | 查什么 |
|---|---|
| 502 | `docker ps` 容器是否在跑；`docker logs arrow-decode` |
| 413 | nginx `client_max_body_size` 或服务端 20 MB 上限 |
| 401 | `X-Decode-Token` 与 `/etc/arrow/decoder.env` 不一致 |
| 一直 rejected / `low_logo_scale` | 字标宽度不到画面宽度 8%；让 Logo 占满画面重拍 |
| 上传超时 | 原图较大；看 nginx `proxy_read_timeout` 与网络 |
