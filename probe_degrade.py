"""真机照片的失效包络：在这张实拍照上逐个加恶化，看解到哪里断。

为什么要用真机照做这件事
-----------------------
仿真里的失焦 σ、噪声、光照都是我设的；`bench_realscale` 默认 σ=1.6 px，
结果推出「水线 logo_scale≈0.16」。而实际打印页的实拍在 **0.127** 就解出来了，
5/6 接受。差别不在分辨率，在**这张照片比我仿真的清晰**。

所以这里以真机裁块为基准，单独扫三个轴，把「分辨率」和「模糊」分开：
  1. 模糊 σ（相机域）——真正的主导变量
  2. 降采样后的印版像素数（每码元像素）
  3. JPEG 质量

单位换算（已用干净印版标定）：
    logo_scale = 0.8513 × (印版宽 px / 2200)
    每码元相机像素 = 16.4 × (印版宽 px / 2200) = 19.27 × logo_scale
    奈奎斯特极限 = 2.00 px/码元  <=>  印版宽 ≈ 268 px  <=>  logo_scale ≈ 0.104

用法:
    python probe_degrade.py user-photo-1.jpg
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import arrow_decode as AD

SCALE_K = 0.8513          # logo_scale = K * plate_px/2200
ELEM_PER_PLATE = 16.4     # 每码元的底版弧长


def plate_px(logo_scale: float) -> float:
    return (logo_scale / SCALE_K) * 2200.0


def elem_px(width: float) -> float:
    return ELEM_PER_PLATE * width / 2200.0


def decode(dec, img):
    r = dec.decode(img)
    return r


def job(img, w):
    pass


def show_row(tag, dec, img, extra=""):
    r = dec.decode(img)
    ls = r.logo_scale or 0.0
    print(f"  {tag:<22} {elem_px(plate_px(ls)) if ls else 0:>7.2f} "
          f"{str(r.accepted):>9} {str(r.digits):>12} {r.margin:>7.2f} "
          f"{r.cc:>6.3f} {(r.reason or '-'):>18} {extra}")
    return r


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    dec = AD.Decoder(carrier="inner")

    # 取一块清晰的印版：用整幅定位到的框（第二行左侧那块，margin 最高）
    import probe_user_photo as P
    boxes = P.find_marks(gray)
    if not boxes:
        print("没找到印版")
        return 1
    # 选面积最大的两块里 margin 更高的——这里直接取第 3 块（实测 margin 4.15）
    boxes.sort(key=lambda b: (b[1], b[0]))
    x, y, w, h = boxes[2] if len(boxes) > 2 else boxes[0]
    pad = int(round(0.12 * w))
    crop = gray[max(0, y - pad):min(H, y + h + pad),
                max(0, x - pad):min(W, x + w + pad)]
    print(f"基准裁块 {crop.shape[1]}×{crop.shape[0]} px（来自 {path}）")
    r0 = dec.decode(crop)
    print(f"基准: logo_scale={r0.logo_scale:.4f} 印版≈{plate_px(r0.logo_scale):.0f} px "
          f"每码元≈{elem_px(plate_px(r0.logo_scale)):.2f} px "
          f"accepted={r0.accepted} margin={r0.margin:.2f}\n")

    fmt = (f"  {{tag:<22}} {{elem:>7}} {{acc:>9}} {{dig:>12}} "
           f"{{mar:>7}} {{cc:>6}} {{rea:>18}} {{extra}}")

    print("① 模糊（相机域高斯 σ，同时看它怎么吃掉裕度）")
    print(f"  {'σ px':<22} {'px/码元':>7} {'accepted':>9} {'digits':>12} "
          f"{'margin':>7} {'cc':>6} {'reason':>18}")
    for sig in (0.0, 0.5, 0.8, 1.1, 1.4, 1.8, 2.2):
        img = crop if sig <= 0 else cv2.GaussianBlur(
            crop.astype(np.float32), (0, 0), sig)
        show_row(f"σ={sig:.1f}", dec,
                 np.clip(img, 0, 255).astype(np.uint8))

    print("\n② 分辨率（把裁块降到印版只有 N px，模糊不变）")
    print(f"  {'印版宽 px':<22} {'px/码元':>7} {'accepted':>9} {'digits':>12} "
          f"{'margin':>7} {'cc':>6} {'reason':>18}")
    base_plate = plate_px(r0.logo_scale)
    for target in (450, 400, 350, 320, 290, 260, 230):
        f = target / base_plate
        img = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        show_row(f"印版 {target} px", dec, img,
                 extra=f"(每码元 {elem_px(target):.2f})")

    print("\n③ JPEG 质量（分辨率不变）")
    print(f"  {'q':<22} {'px/码元':>7} {'accepted':>9} {'digits':>12} "
          f"{'margin':>7} {'cc':>6} {'reason':>18}")
    for q in (95, 88, 75, 60, 45, 30):
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        show_row(f"q={q}", dec, img, extra=f"({len(buf)//1024} KB)")

    print("\n④ 旋转（手持角度）")
    print(f"  {'角度':<22} {'px/码元':>7} {'accepted':>9} {'digits':>12} "
          f"{'margin':>7} {'cc':>6} {'reason':>18}")
    ch, cw = crop.shape
    for ang in (3.0, 8.0, 15.0, 25.0):
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
        img = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        show_row(f"旋转 {ang:.0f}°", dec, img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
