"""解码器变体实验：在小 logo 实拍链路下比较几处重采样/测窗设置。

为什么要单独做实验
-----------------
`bench_client_pipeline.py` 复现的是「线上这条链路现在有多差」。
本脚本是在**同一条链路**上换解码器的几个自由度，看哪一处能稳定改善
低尺度召回，而不是只在一个失焦值上碰运气。因此每个变体都在
blur ∈ {1.2, 1.6, 2.0} × scale ∈ {0.13, 0.16} 上跑同一批号码。

被判定的变体：
  interp       矫正重采样核（双线性 / 双三次 / Lanczos）
  sigmas       自适应采样窗半宽 = depth + k·σ_rectified
  weight       梯度质心的高斯权重（相对窗半宽）

用法
----
    python exp_decode_variants.py --variant interp
    python exp_decode_variants.py --variant sigmas
    python exp_decode_variants.py --baseline-only
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from bench_client_pipeline import (DEFAULT_PAYLOADS, ROI_EXPAND_RATIO,
                                   client_roi_ladder, expand_box, render_photo)


def build_cases(scales: Sequence[float], blurs: Sequence[float],
                seeds: Sequence[int], layout: int):
    """把「号码 × 尺度 × 失焦 × 种子」提前渲染好，供各变体复用。"""
    design = A.ArrowDesign()
    carrier = I.InnerCarrier(design, depth=I.DEPTH, layout_version=layout)
    cases = []
    for dg in DEFAULT_PAYLOADS:
        plate = carrier.embed(dg)["plate"]
        for s in scales:
            for b in blurs:
                for sd in seeds:
                    rng = np.random.default_rng(
                        1000 * sd + int(round(s * 1000)) + int(round(b * 10)))
                    photo, box = render_photo(plate, s, rng=rng, blur=b)
                    roi = expand_box(box, ROI_EXPAND_RATIO,
                                     photo.shape[1], photo.shape[0])
                    x, y, bw, bh = roi
                    packed = client_roi_ladder(photo[y:y + bh, x:x + bw],
                                               mode="maxq")
                    sent = cv2.imdecode(
                        np.frombuffer(packed["data"], np.uint8),
                        cv2.IMREAD_GRAYSCALE)
                    cases.append({"digits": dg, "scale": s, "blur": b,
                                  "seed": sd, "image": sent})
    return design, cases


def _gauss_psf(sigma: float, size: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.ones((1, 1), np.float32)
    k = size or int(max(3, 2 * round(3 * sigma) + 1))
    ax = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    psf = np.outer(g, g)
    return (psf / psf.sum()).astype(np.float32)


def richardson_lucy(img: np.ndarray, sigma: float, iters: int = 12) -> np.ndarray:
    """在相机域做 Richardson-Lucy 反卷积，把失焦糊掉的边缘抢回来一点。

    失焦是实测的主导失效因子（0.13 档 σ=1.2 时 9/12，σ=2.0 时 0/12），
    而失焦在数学上是可逆的卷积——只要 PSF 大致已知。手机主摄在 7–10 cm
    的工作距离上 σ≈1.6 px 是稳定的（README 用五张实拍标定过），所以可以
    用固定 PSF 去反卷积，不需要每张图估。
    """
    psf = _gauss_psf(sigma)
    psf_t = psf[::-1, ::-1]
    obs = np.clip(img.astype(np.float32) / 255.0, 1e-4, 1.0)
    est = obs.copy()
    for _ in range(iters):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE)
        ratio = obs / np.clip(conv, 1e-4, None)
        est *= cv2.filter2D(ratio, -1, psf_t, borderType=cv2.BORDER_REPLICATE)
        est = np.clip(est, 1e-4, 1.0)
    return np.clip(est * 255.0, 0, 255).astype(np.uint8)


def evaluate(design, cases, *, interp: int = cv2.INTER_LINEAR,
             sigmas: float = I.SCAN_SIGMAS,
             weight: float = I.SCAN_WEIGHT_FRAC,
             preproc=None,
             layout: int = I.LAYOUT_V3) -> Dict:
    """跑一个变体，返回按 (scale, blur) 分桶的准确率。"""
    old_interp, old_sigmas, old_weight = (
        A.RECTIFY_INTERP, I.SCAN_SIGMAS, I.SCAN_WEIGHT_FRAC)
    A.RECTIFY_INTERP = interp
    I.SCAN_SIGMAS = sigmas
    I.SCAN_WEIGHT_FRAC = weight
    try:
        session = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                             layout_version=layout)
        buckets: Dict[Tuple[float, float], List[int]] = {}
        wrong = 0
        t0 = time.time()
        for c in cases:
            img = c["image"] if preproc is None else preproc(c["image"])
            r = session.decode(img)
            key = (c["scale"], c["blur"])
            b = buckets.setdefault(key, [0, 0, 0])  # ok, wrong, rejected
            if r.accepted and r.digits == c["digits"]:
                b[0] += 1
            elif r.accepted:
                b[1] += 1
                wrong += 1
            else:
                b[2] += 1
        return {"buckets": buckets, "wrong": wrong,
                "sec": (time.time() - t0) / max(len(cases), 1)}
    finally:
        A.RECTIFY_INTERP, I.SCAN_SIGMAS, I.SCAN_WEIGHT_FRAC = (
            old_interp, old_sigmas, old_weight)


def show(label: str, res: Dict, scales: Sequence[float],
         blurs: Sequence[float]) -> None:
    total_ok = sum(v[0] for v in res["buckets"].values())
    total = sum(sum(v) for v in res["buckets"].values())
    cells = []
    for s in scales:
        row = []
        for b in blurs:
            ok, wr, rj = res["buckets"].get((s, b), (0, 0, 0))
            row.append(f"{ok}/{ok + wr + rj}")
        cells.append(f"s={s:.2f}:" + " ".join(row))
    print(f"  {label:<26} {total_ok}/{total}  错号 {res['wrong']}"
          f"   {res['sec']:.2f}s/张   " + "  ".join(cells))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARROW 解码器变体实验")
    ap.add_argument("--variant", choices=("interp", "sigmas", "weight", "preproc"),
                    default="interp")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.13, 0.16])
    ap.add_argument("--blurs", type=float, nargs="+", default=[1.2, 1.6, 2.0])
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args(argv)

    scales, blurs = args.scales, args.blurs
    seeds = tuple(range(1, args.n + 1))
    print(f"渲染 {len(DEFAULT_PAYLOADS)} 号码 × {len(scales)} 尺度 × "
          f"{len(blurs)} 失焦 × {len(seeds)} 种子 ...")
    design, cases = build_cases(scales, blurs, seeds, I.LAYOUT_V3)
    total = len(cases)
    print(f"共 {total} 张，客户端用修好后的质量阶梯（maxq）\n")

    grid = {
        "interp": [("LINEAR (现状)", dict(interp=cv2.INTER_LINEAR)),
                   ("CUBIC", dict(interp=cv2.INTER_CUBIC)),
                   ("LANCZOS4", dict(interp=cv2.INTER_LANCZOS4))],
        "sigmas": [("2.2 (现状)", dict(sigmas=2.2)),
                   ("1.8", dict(sigmas=1.8)),
                   ("2.6", dict(sigmas=2.6)),
                   ("3.0", dict(sigmas=3.0))],
        "weight": [("0.55 (现状)", dict(weight=0.55)),
                   ("0.40", dict(weight=0.40)),
                   ("0.70", dict(weight=0.70))],
        "preproc": [
            ("无（现状）", dict()),
            ("反卷积 σ=1.2", dict(preproc=lambda im: richardson_lucy(im, 1.2))),
            ("反卷积 σ=1.6", dict(preproc=lambda im: richardson_lucy(im, 1.6))),
            ("反卷积 σ=2.0", dict(preproc=lambda im: richardson_lucy(im, 2.0))),
            ("反卷积 σ=1.6 ×24", dict(preproc=lambda im: richardson_lucy(im, 1.6, 24))),
        ],
    }[args.variant]

    if args.baseline_only:
        grid = grid[:1]
    for label, kw in grid:
        res = evaluate(design, cases, **kw)
        show(f"{args.variant}={label}", res, scales, blurs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
