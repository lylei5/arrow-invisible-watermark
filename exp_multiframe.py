"""多帧策略对照：选最清晰 vs 矫正域对齐平均 vs 二者结合。

为什么值得单独测
---------------
我的诊断说「σ 是主导变量」，那么连拍 3 帧就有两条完全不同的用法：

  A. **选最清晰**：σ 是随机量（手抖/对焦跑），取 3 帧里最小的一帧。
     只利用「分布的最小值」，收益取决于 σ 的帧间波动有多大。
  B. **对齐后平均**：每帧的采样相位略有不同，配准到同一画布后做多帧平均，
     等效把随机噪声压到 1/√N，并把「被采样到的相位」补齐一点。
     即使 σ 完全相同，这一步也有增益——它动的是噪声/采样，不是 σ。

在仿真里可以让两者同时变好，也可以用「固定 σ、只换采样相位」把 B 单独隔出来。

用法:
    python exp_multiframe.py -n 3
    python exp_multiframe.py --sigma-fixed 1.6    # 只让相位/噪声变化，隔离 B 的效果
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from bench_client_pipeline import client_roi_ladder, expand_box

PAYLOADS = ("13800138000", "13912345678", "18600001234")


def render_frame(plate: np.ndarray, scale: float, *, rng, blur: float,
                 noise: float, jpeg: int, shift: Tuple[float, float],
                 frame_pad: float = 1.4) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """渲染一帧：印版按 scale 缩小 + 亚像素平移 + 失焦 + 噪声 + JPEG。"""
    h, w = plate.shape
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    small = cv2.resize(plate.astype(np.float32), (nw, nh),
                       interpolation=cv2.INTER_AREA)
    cov = np.clip((255.0 - small) / 255.0, 0.0, 1.0)
    small = 168.0 * (1.0 - cov) + 48.0 * cov

    margin = int(round(frame_pad * max(nw, nh)))
    fw, fh = nw + 2 * margin, nh + 2 * margin
    canvas = np.full((fh, fw), 168.0, np.float32)
    canvas[margin:margin + nh, margin:margin + nw] = small
    if blur > 0.05:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    # 亚像素平移：模拟手持时每帧的采样相位不同
    M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
    canvas = cv2.warpAffine(canvas, M, (fw, fh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    canvas += rng.normal(0, noise, canvas.shape).astype(np.float32)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img, (margin, margin, nw, nh)


def sharpen_score(img: np.ndarray) -> float:
    """与小程序一致的清晰度口径：拉普拉斯平方均值（越大越清晰）。"""
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    return float((lap * lap).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多帧策略对照")
    ap.add_argument("-n", type=int, default=3, help="每组的帧数")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.13, 0.16])
    ap.add_argument("--sigma", type=float, default=1.6, help="基础失焦 σ（px）")
    ap.add_argument("--sigma-jitter", type=float, default=0.5,
                    help="帧间 σ 的随机抖动幅度（0=σ 完全相同，隔离平均的贡献）")
    ap.add_argument("--noise", type=float, default=4.0)
    ap.add_argument("--jpeg", type=int, default=85)
    args = ap.parse_args(argv)

    design = A.ArrowDesign()
    carrier = I.InnerCarrier(design, depth=I.DEPTH, layout_version=I.LAYOUT_V3)
    dec = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                     layout_version=I.LAYOUT_V3)

    print("=" * 92)
    print(f"  多帧对照  n={args.n}  σ={args.sigma}±{args.sigma_jitter}  "
          f"噪声 σ={args.noise}  帧间亚像素平移随机（±0.5 px）")
    print("=" * 92)
    print(f"{'尺度':>5} {'单帧':>9} {'选最清晰':>10} {'对齐平均':>10} "
          f"{'平均后取最清晰':>14} {'帧数':>5}")
    print("-" * 92)

    for s in args.scales:
        stats: Dict[str, List[int]] = {"single": [0, 0], "best": [0, 0],
                                       "avg": [0, 0], "avg_best": [0, 0]}
        for dg in PAYLOADS:
            plate = carrier.embed(dg)["plate"]
            for sd in range(1, args.seeds + 1):
                rng = np.random.default_rng(7000 + sd * 31 + int(s * 1000))
                frames = []
                for k in range(args.n):
                    blur = max(0.05, args.sigma
                               + rng.normal(0, args.sigma_jitter))
                    shift = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
                    img, box = render_frame(plate, s, rng=rng, blur=blur,
                                            noise=args.noise, jpeg=args.jpeg,
                                            shift=shift)
                    # 走客户端链路：裁 ROI + 高质量 JPEG
                    roi = expand_box(box, 0.25, img.shape[1], img.shape[0])
                    x, y, bw, bh = roi
                    crop = img[y:y + bh, x:x + bw]
                    packed = client_roi_ladder(crop, mode="maxq")
                    sent = cv2.imdecode(
                        np.frombuffer(packed["data"], np.uint8),
                        cv2.IMREAD_GRAYSCALE)
                    frames.append(sent)

                def ok_of(res, dg=dg):
                    return 1 if (res.accepted and res.digits == dg) else 0

                # A. 单帧（取第 0 帧）
                stats["single"][0] += ok_of(dec.decode(frames[0]))
                stats["single"][1] += 1
                # B. 选最清晰
                best = max(frames, key=sharpen_score)
                stats["best"][0] += ok_of(dec.decode(best))
                stats["best"][1] += 1
                # C. 先在相机域对齐再平均（用 ECC 对齐到第一帧）
                base = frames[0].astype(np.float32)
                acc = base.copy()
                for f in frames[1:]:
                    g = f.astype(np.float32)
                    warp = np.eye(2, 3, dtype=np.float32)
                    try:
                        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                100, 1e-5)
                        _, warp = cv2.findTransformECC(
                            base, g, warp, cv2.MOTION_EUCLIDEAN, crit,
                            None, 5)
                        g = cv2.warpAffine(g, warp, (base.shape[1], base.shape[0]),
                                           flags=cv2.INTER_LINEAR
                                           | cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_REPLICATE)
                    except Exception:
                        pass
                    acc += g
                avg = np.clip(acc / len(frames), 0, 255).astype(np.uint8)
                stats["avg"][0] += ok_of(dec.decode(avg))
                stats["avg"][1] += 1
                # D. 平均后如果还不行，再在平均帧上做一次"取最清晰"式的补救
                res_avg = dec.decode(avg)
                if res_avg.accepted:
                    stats["avg_best"][0] += 1
                else:
                    stats["avg_best"][0] += ok_of(dec.decode(best))
                stats["avg_best"][1] += 1

        def fmt(k):
            a, b = stats[k]
            return f"{a}/{b}"
        print(f"{s:>5.2f} {fmt('single'):>9} {fmt('best'):>10} {fmt('avg'):>10} "
              f"{fmt('avg_best'):>14} {args.n:>5}")
    print("-" * 92)
    print("  说明：单帧=第 0 帧；选最清晰=按拉普拉斯平方均值挑；"
          "对齐平均=ECC 欧氏对齐后取均值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
