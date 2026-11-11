# v5 服务记录

以下为 2026-11-10 部署并验证的 v5 实例记录。仓库更新不等于服务器重新部署。

| 项目 | 配置 |
| --- | --- |
| 格式 | `arrow-vector-v5` |
| 发布标识 | `vector-v5-counts-20261108` |
| 母版 SHA-256 | `dc4bd045fb3c06204e32f41b548d30c0afc41403d4784bb3f2f18f331ad830a0` |
| 解码地址 | `https://arrow.liylei666.online/v5/decode` |
| 健康检查 | `https://arrow.liylei666.online/v5/health` |
| 镜像 | `arrow-decode:vector-v5-counts-20261108` |
| 容器 | `arrow-decode-v5` |
| 端口 | `127.0.0.1:8603` → 容器 `8501` |
| 资源上限 | 1 CPU、2 GiB RAM |
| 数据挂载 | `/opt/arrow/v5-data:/var/lib/arrow` |
| 计数起始时间 | `2026-11-10T05:24:07.520Z` |

已通过公网 HTTPS 验证成功解码及请求重放不重复计数；微信开发者工具运行时完成了测试照片原生上传和结果展示。真机调试包已成功上传并连接手机。真实印刷品的连续拍摄检测率与速度仍需现场验证。

更新服务器时应保留数据库、私有令牌和同机其他服务。部署操作独立于 Git 提交与推送。
