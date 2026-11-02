"""真实尺度压测：把印版缩到「手机实拍里 logo 只占几百像素」再解。

为什么需要这个压测
------------------
`arrow_inner.py --bench` / `arrow_wm.py --bench` 用的是
`CeramicCamSimulator`，它把 2200×600 的印版**铺满** 2600×1600 的画面，
也就是 logo_scale ≈ 1.2。那是「贴着拍、印版占满取景框」的理想工况，
和线上真实照片差一个数量级：实拍样张里 logo 只占底版宽度的 0.13–0.16，
即 2200 px 的印版在照片里只有 290–350 px。

两者的差别不是「清晰度」而是**采样率**：内孔载体的一个码元在底版上是
16–18 px 弧长，logo_scale=0.13 时落到相机上只有 2.1 个像素；差分对
（一宽一窄）的周期 4.2 px，正好在奈奎斯特边缘。再叠加 1.5–2 px 的失焦，
调制幅度会被压低到标称值的 30–50%。这一步是全幅压测完全测不到的，
而它恰恰是线上失败的主要来源。

所以本脚本按真实尺度缩小印版再走一遍完整生产门，作为回归基线。

用法
----
    python bench_realscale.py                        # 默认网格
    python bench_realscale.py -n 3 --scales 0.10 0.13 0.16 0.20
    python bench_realscale.py --layout 2             # 旧版内孔印版
    python bench_realscale.py --blur 2.0 --jpeg 60   # 更差的拍摄条件
"""
from __future__ import annotations

import argparse
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A

DEFAULT_PAYLOADS = ("13912345678", "13800138000", "18600001234",
                    "17701234567", "01088886666", "13500001111")
DEFAULT_SCALES = (0.10, 0.13, 0.16, 0.20)
FRAME_W, FRAME_H = 574, 185          # 实拍样张的实际尺寸


def render_at_scale(plate: np.ndarray, scale: float, *, rng: np.random.Generator,
                    frame: Tuple[int, int] = (FRAME_W, FRAME_H),
                    blur: float = 1.6, noise: float = 3.0, jpeg: int = 70,
                    paper: float = 168.0, ink: float = 48.0) -> np.ndarray:
    """印版 -> 按 scale 缩小 -> 贴到浅灰纸面 -> 失焦/噪声/JPEG。

    与 `CeramicCamSimulator` 的区别只有一条：不做「铺满画面」的归一化，
    而是严格按 scale 决定印版在照片里的像素宽度。这样 logo_scale 才是
    真实工况，而不是 1.2。
    """
    h, w = plate.shape
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    # 面积平均：真实相机的像素是积分，不是点采样
    small = cv2.resize(plate.astype(np.float32), (nw, nh),
                       interpolation=cv2.INTER_AREA)
    cov = np.clip((255.0 - small) / 255.0, 0.0, 1.0)
    small = paper * (1.0 - cov) + ink * cov

    fw, fh = frame
    canvas = np.full((fh, fw), paper, np.float32)
    ox = (fw - nw) // 2
    oy = (fh - nh) // 2
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + nw, fw), min(oy + nh, fh)
    canvas[y0:y1, x0:x1] = small[y0 - oy:y1 - oy, x0 - ox:x1 - ox]

    # 纸面/釉面纹理（实拍图里能看到一层细网点）
    tex = cv2.GaussianBlur(rng.normal(0, 6.0, (fh, fw)).astype(np.float32),
                           (0, 0), 0.6)
    canvas += tex
    if blur > 0.05:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas += rng.normal(0, noise, canvas.shape).astype(np.float32)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg)])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


def run(design, *, layout: int = I.LAYOUT_V3, payloads: Sequence[str] = DEFAULT_PAYLOADS,
        scales: Sequence[float] = DEFAULT_SCALES, seeds: Sequence[int] = (1, 2),
        blur: float = 1.6, noise: float = 3.0, jpeg: int = 70,
        allow_legacy_v2: bool = False) -> int:
    carrier = I.InnerCarrier(design, depth=I.DEPTH, layout_version=layout)
    session = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                         layout_version=layout, allow_legacy_v2=allow_legacy_v2)
    print("=" * 92)
    print(f"  真实尺度压测   layout=v{layout}  {carrier.n_cells} 码元  "
          f"depth={carrier.depth:g}px  失焦 σ={blur:g}px  噪声 σ={noise:g}  "
          f"JPEG q={jpeg}")
    print(f"  {len(payloads)} 号码 × {len(scales)} 尺度 × {len(seeds)} 种子")
    print("=" * 92)
    print(f"| {'尺度':>5} | {'研究正确':>9} | {'接受且对':>9} | {'接受且错':>9} "
          f"| {'拒绝':>6} | {'中位裕度':>9} | {'耗时':>7} |")
    print("|" + "-" * 7 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11
          + "|" + "-" * 8 + "|" + "-" * 11 + "|" + "-" * 9 + "|")

    total_wrong = 0
    for s in scales:
        n = ok = wrong = rej = 0
        margins: List[float] = []
        t0 = time.time()
        for dg in payloads:
            plate = carrier.embed(dg)["plate"]
            for sd in seeds:
                rng = np.random.default_rng(1000 * sd + int(round(s * 1000)))
                img = render_at_scale(plate, s, rng=rng, blur=blur,
                                      noise=noise, jpeg=jpeg)
                r = session.decode(img)
                n += 1
                if r.accepted and r.digits == dg:
                    ok += 1
                elif r.accepted:
                    wrong += 1
                else:
                    rej += 1
                if r.raw.get("ok"):
                    margins.append(r.margin)
        dt = time.time() - t0
        total_wrong += wrong
        med = float(np.median(margins)) if margins else 0.0
        print(f"| {s:>5.2f} | {ok:>4d}/{n:<4d} | {ok:>4d}/{n:<4d} | "
              f"{wrong:>4d}/{n:<4d} | {rej:>4d}/{n:<4d} | {med:>9.2f} | "
              f"{dt / max(n, 1):>6.2f}s |")
    print("=" * 92)
    if total_wrong:
        print(f"  ⚠ 接受且错号 {total_wrong} 次：这是最严重的失败，必须先修这个再谈召回率。")
    else:
        print("  接受且错号 0：所有接受的结果都是正确号码。")
    return total_wrong


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARROW 内孔载体真实尺度压测")
    ap.add_argument("-n", type=int, default=2, help="每个尺度的随机种子数")
    ap.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES),
                    help="logo_scale 网格（默认 0.10 0.13 0.16 0.20）")
    ap.add_argument("--layout", choices=("2", "3"), default="3",
                    help="内孔版式：3=交织+CRC-7（新印版，默认），2=旧印版")
    ap.add_argument("--blur", type=float, default=1.6, help="相机域失焦 σ（px）")
    ap.add_argument("--noise", type=float, default=3.0, help="传感器噪声 σ（灰度级）")
    ap.add_argument("--jpeg", type=int, default=70, help="JPEG 质量")
    ap.add_argument("--allow-legacy-v2", action="store_true",
                    help="允许旧版 v2 印版在小画面下被接受（不推荐）")
    args = ap.parse_args(argv)

    design = A.ArrowDesign()
    run(design, layout=int(args.layout), scales=args.scales,
        seeds=tuple(range(1, args.n + 1)), blur=args.blur, noise=args.noise,
        jpeg=args.jpeg, allow_legacy_v2=args.allow_legacy_v2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
