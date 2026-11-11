# ARROW v5 部署

## 构建与运行

```bash
docker build -t arrow-decode:v5 .
sudo install -d -o 10001 -g 10001 -m 700 /opt/arrow/v5-data
docker run -d --name arrow-decode-v5 --restart unless-stopped \
  --cpus 1 --memory 2g --memory-swap 2g \
  --env-file /etc/arrow/decoder.env \
  -e ARROW_DB_PATH=/var/lib/arrow/verifications.sqlite3 \
  -v /opt/arrow/v5-data:/var/lib/arrow \
  -p 127.0.0.1:8603:8501 \
  --log-opt max-size=10m --log-opt max-file=3 \
  arrow-decode:v5
```

私有环境文件必须设置非空 `DECODE_TOKEN`，限制为管理员可读，禁止提交 Git。已有部署沿用现有令牌及数据库，不要因更新代码而生成新的计数数据库。以上命令适用于新实例；已有同名容器需按维护流程备份和替换。

镜像使用用户 10001，数据库目录需要可写。默认单 worker；`WEB_CONCURRENCY` 可调整 worker 数，但每个 worker 都会加载模板。CPU 和内存约束应根据目标机器实测延迟及并发负载设置。

## HTTPS 代理

v5 对外路径为 `/v5/decode`，容器内部为 `/decode`。在已有域名的 HTTPS server 块内配置：

```nginx
location /v5/ {
    client_max_body_size 21m;
    proxy_pass http://127.0.0.1:8603/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 120s;
}
```

检查 `nginx -t` 后再 reload。保留同机其他服务的容器、端口、配置和数据目录。健康检查为 `/v5/health`，就绪时返回 `format=arrow-vector-v5` 和 `layout_version=5`。

## 小程序接入

在微信后台的 request 和 uploadFile 合法域名中登记 HTTPS 域名。上传 `image` 字段，附带 README 列出的四个请求头。使用 v5 格式及母版标识校验返回值；网络重试沿用同一 `X-Verification-ID`，用户重新点击检测时创建新编号。

本仓库提供解码和制版代码；小程序 AppID、私有连接配置及发布状态在独立项目中维护。更新 GitHub 不会自动部署服务或发布小程序。

## 数据与验收

通过 SQLite backup API 备份运行中的数据库，保证 WAL 数据一致。替换容器后保留同一挂载路径，验证 `counting_since` 和计数连续性。计数历史包含编号及请求元数据，不保存上传照片。

验收包括健康检查、已知样本成功解码、无码图拒绝、前导零保留、同一请求重试不增次、新请求增次、照片哈希及母版标识核对。手机端另需验证首开拍摄、重复拍摄、权限、裁剪几何和返回图像尺寸。
