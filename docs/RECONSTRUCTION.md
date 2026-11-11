# ARROW v5 方法与评估

## 母版

`arrow_vector.py` 用多边形和 Bézier 曲线构造字形。总宽为 5X、字高为 X，四个字间距依次为 0.04X、0.03X、0.03X、0.01X；两个 R 使用相同几何。字标宽 22 mm、高 4.4 mm，居中放置于 22 × 6 mm 画布。以 4 倍超采样生成掩膜，再面积平均为 2200 × 600 灰度母版。

参考设计图未提供全部曲线控制点，因此这是按比例重建的字形，不宣称与官方原始矢量文件完全一致。

## 内孔载体

四个封闭内孔提供连续且可测量的局部边界。水印保持五个字母的外轮廓不变，在内孔分配共 77 个码元。每个码元以正弦窗产生径向位移，峰值 ±8 px，即 ±0.08 mm。径向位移在尖角处不等于轮廓法向位移。

37 位业务载荷与 7 位 CRC-7/MMC 拼接为 44 位数据，分成 11 组 Hamming(7,4)，输出 77 位。每个码元承载一个编码位；固定加扰翻转部分码元符号。按各孔配额交织，使每个纠错组分散到至少三个孔，减少局部模糊集中破坏一个纠错组的机会。

## 照片解码

1. 在长边不超过 900 px 的缩略图中，通过 SIFT、字形几何和候选卡片区域定位。
2. 按实际缩放尺寸及像素中心约定，将单应矩阵映射回原图。
3. 用原始像素进行透视矫正和 ECC 配准。
4. 按字标尺度选择径向扫描窗口，用梯度加权质心测量边界；减去未调制模板的参考半径。
5. 每孔减均值，去除共同偏置；反加扰、反交织后做 Hamming 软判决。
6. 校验编号范围、CRC、平均及最弱组裕度、配准相关和字标尺度。全部通过才输出可信编号。

默认使用等角窗。`v5_arc` 是等弧长窗口研究选项，需匹配生成与解码参数；HTTP 服务固定使用等角窗和 0.08 mm 深度。

## 可复现评估

```bash
python tools/benchmark_reconstruction.py --out outputs/standard --n 32 --seed 921047 --profiles clean,print1200_close,print1200_medium,print1200_small,print600_medium,local_O_blur,ink_spread
python tools/benchmark_reconstruction.py --out outputs/stress --n 40 --seed 590871 --profiles defocus_286,defocus_330,tiny_220,local_O_severe,print300
python tools/benchmark_reconstruction.py --out outputs/negative --n 40 --seed 298415 --negative --profiles print1200_close,print1200_small,defocus_286,tiny_220,print300
python tools/build_deployment_fixtures.py outputs/fixtures
python tools/benchmark_service.py outputs/fixtures outputs/service-benchmark
```

模拟信道包含打印网格采样、透视、失焦、噪声、JPEG、局部 O 模糊和油墨扩散近似。解码器不接收仿真的真实单应矩阵。输出按条件记录正确接受、错误接受、拒绝、耗时和比特错误率。

实物印刷的网点、釉面、烧制、非平面形变以及肉眼可见度仍需真实样本验证。全图 PSNR 会稀释局部轮廓变化，不能单独作为隐蔽性证据。CRC 和 Hamming 不提供密码学身份认证。
