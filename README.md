# ARROW v5 隐形水印

本仓库提供 ARROW v5 矢量母版、内孔水印编码、手机照片解码、HTTP 服务和验证次数统计。默认入口、Docker 镜像和工具均使用 `arrow-vector-v5`。

样本画布为 **22 × 6 mm（2.2 × 0.6 cm）**，字标为 22 × 4.4 mm，上下各留白 0.8 mm。水印只改变 A、两个 R 和 O 的四个内孔轮廓；外轮廓保持不变，不添加定位条纹。母版依据设计图比例重新构建，曲线为人工几何重建，不是官方矢量原稿。

## 编码与解码

- 业务编号：11 位十进制字符串，保留前导零，占 37 bit。
- 完整性校验：CRC-7/MMC，37 + 7 = **44 位数据**。
- 纠错：11 组 Hamming(7,4)，得到 **77 位编码**，对应 77 个内孔码元。
- 布局：等角采样、按孔周长分配码元、跨孔交织；每组 7 位覆盖至少三个孔。
- 嵌入：平滑径向位移，峰值 ±0.08 mm，固定异或加扰。
- 拍照解码：缩略图定位，将单应矩阵映射回原图，在原始像素上矫正、测量边界，随后软判决、CRC 校验和质量门控。

CRC 用于差错检测，不是密码学防伪；验证次数属于编号，不能单独证明实物唯一性。检测率、可见度和延迟仍需在实际印刷材料与手机上测量，仿真不能代替实物测试。

## 快速开始

```bash
python -m pip install -r requirements-research.txt
python tools/arrow_research.py master --out outputs/master
python tools/arrow_research.py embed --digits 13800138000 --out outputs/ARROW_13800138000.png
python tools/arrow_research.py decode --image samples/v5/ARROW_13800138000.png
python tools/export_sample_sheet.py --out outputs/print-sheet
```

最后一条命令生成一页 72 个不同编号的 A4 PDF，每个样本均为 22 × 6 mm。打印选择“实际大小 / 100%”，关闭“适合页面”。PNG 母版为 2200 × 600 px，标记 2540 dpi。PDF 工具可用 `--font /path/to/chinese.ttf` 指定中文字体。

## HTTP 服务

```bash
python -m pip install -r services/decode_api/requirements.txt
export DECODE_TOKEN='replace-with-your-private-token'
export ARROW_DB_PATH='./runtime/verifications.sqlite3'
python -m uvicorn services.decode_api.app:app --host 127.0.0.1 --port 8501
```

PowerShell 用 `$env:DECODE_TOKEN` 和 `$env:ARROW_DB_PATH` 设置对应环境变量。服务启动时执行已知样本自检；令牌必须配置。

`GET /health` 返回就绪状态、格式、母版 SHA-256、布局版本和计数起始时间，无需认证。`POST /decode` 接受 multipart 的 `image` 字段，或 JPEG、PNG、WebP 的原始字节。

| 请求头 | 要求 |
| --- | --- |
| `X-Decode-Token` | 与服务端私有令牌一致 |
| `X-Arrow-Format` | `arrow-vector-v5` |
| `X-Image-SHA256` | 实际上传图像字节的 SHA-256，64 位十六进制 |
| `X-Verification-ID` | 一次检测的唯一编号，20–100 个字母、数字、下划线或连字符；同一照片的网络重试复用此编号 |

成功必须以 `accepted === true` 为准。成功时返回 `digits`、`verification_count`、`verified_at`、`verification_replayed`；拒绝时 `digits` 和 `verification_count` 为 `null`。客户端还应核对 `format`、`master_id` 和 `input_sha256`。

母版标识：`dc4bd045fb3c06204e32f41b548d30c0afc41403d4784bb3f2f18f331ad830a0`。

质量门限为平均码字裕度 ≥ 1.8、最弱组裕度 ≥ 0.20、配准相关 ≥ 0.85、`logo_scale` ≥ 0.08。`logo_scale` 是照片内字标尺度相对于 2200 px 母版的比例，不是占整个画面的比例。

请求错误包括 `400` 字段错误、`401` 认证失败、`409` 格式或请求编号冲突、`413` 超限、`415` 文件类型错误、`422` 哈希不一致或图像无效、`429` 解码繁忙、`503` 初始化或计数存储不可用。上限为 20 MiB 和 4000 万像素。每个 worker 同时只处理一次解码。

```bash
python tools/smoke_test.py --url http://127.0.0.1:8501/decode --image samples/v5/ARROW_13800138000.png --expect-digits 13800138000
```

冒烟测试从 `DECODE_TOKEN` 环境变量读取令牌，自动计算哈希和生成请求编号。每次成功调用会增加该编号的验证次数。

## 测试与目录

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

| 路径 | 内容 |
| --- | --- |
| `arrow_vector.py` | 矢量字形、比例和物理尺寸 |
| `arrow_v5.py` | CRC、交织、嵌入和软解码 |
| `arrow_inner.py` | 内孔分配及径向测量 |
| `arrow_wm.py` | 共用几何、Hamming 和配准函数 |
| `arrow_camera.py` | 缩略图定位和坐标映射 |
| `arrow_decode.py` | 质量门控和结果结构 |
| `arrow_service.py` | v5 固定生产参数 |
| `services/decode_api/` | 认证上传、健康检查、SQLite 验证计数 |
| `tools/` | 样本生成、仿真、性能测试和接口测试 |
| `samples/v5/` | 可直接解码的 v5 样本和母版 SVG |

详见 [方法与评估](docs/RECONSTRUCTION.md)、[部署指南](docs/DEPLOY.md)、[v5 服务记录](docs/DEPLOYMENT.md) 和 [验证次数](docs/VERIFICATION_COUNTS.md)。

本次自动化结果见 [v5 验证记录](docs/VALIDATION.md)。
