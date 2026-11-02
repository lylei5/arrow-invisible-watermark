"""每孔切向相位搜索的 A/B：真机照 + 注入已知切向误差。

验证两件事：
  1. 在真机照片上开/关相位搜索，召回和裕度有没有变化；
  2. **注入一个已知的切向误差**（把标块旋转 θ），看它能不能被救回来——
     这是验证机制本身，比看总成功率更有说服力。

为什么旋转等于切向偏移：码元窗沿 θ 排布，一个码元占 `2π/n_cells` 弧度
（n_cells≈40 时约 9°）。绕标块中心旋转 θ 度 ≈ 给每个孔的位移场加一个
`θ / dtheta` 个码元的循环平移。匹配滤波输出会乘 `cos(π·Δ)`，Δ=0.5 码元时归零。

用法:
    python exp_phase_search.py user-photo-1.jpg
    python exp_phase_search.py user-photo-1.jpg --synthetic    # 再跑合成边际集
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A


def find_blocks(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    import probe_user_photo as P
    boxes = P.find_marks(gray)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def crops_of(gray: np.ndarray, boxes, pad_frac=0.06):
    H, W = gray.shape
    out = []
    for (x, y, w, h) in boxes:
        pad = int(round(pad_frac * w))
        out.append(gray[max(0, y - pad):min(H, y + h + pad),
                        max(0, x - pad):min(W, x + w + pad)])
    return out


def run_on(imgs, expect: str, enabled: bool):
    I.PHASE_SEARCH = enabled
    dec = AD.Decoder(carrier="inner", layout_version=I.LAYOUT_V3)
    ok = 0
    margins = []
    shifts = []
    for img in imgs:
        r = dec.decode(img)
        if r.accepted and r.digits == expect:
            ok += 1
        margins.append(float(r.margin))
        impl = dec._impl
        impl = getattr(impl, "_v3", impl)
        sh = getattr(impl, "phase_shift", None)
        if sh:
            shifts.append(sh)
    I.PHASE_SEARCH = True
    return ok, margins, shifts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="切向相位搜索 A/B")
    ap.add_argument("photo")
    ap.add_argument("--expect", default="13800138000")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0.0, 1.5, 3.0, 4.5])
    args = ap.parse_args(argv)

    gray = cv2.imread(args.photo, cv2.IMREAD_GRAYSCALE)
    boxes = find_blocks(gray)
    base = crops_of(gray, boxes)
    print(f"照片 {gray.shape[1]}×{gray.shape[0]}，找到 {len(base)} 块\n")

    print("① 真机照（不注入误差）")
    for enabled in (False, True):
        ok, mg, sh = run_on(base, args.expect, enabled)
        tag = "开" if enabled else "关"
        print(f"   相位搜索{tag}: {ok}/{len(base)} 正确   "
              f"裕度中位 {np.median(mg):.2f}   "
              f"选中的相位 {[[round(v,2) for v in s] for s in sh][:3]}")

    print("\n② 注入已知切向误差（旋转整块，≈ θ/9° 个码元的切向偏移）")
    print(f"   {'角度':>6} {'≈码元':>7} {'关: 正确':>9} {'关: 中位裕度':>12} "
          f"{'开: 正确':>9} {'开: 中位裕度':>12}")
    for ang in args.angles:
        rots = []
        for img in base:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            rots.append(cv2.warpAffine(img, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE))
        res = {}
        for enabled in (False, True):
            ok, mg, _ = run_on(rots, args.expect, enabled)
            res[enabled] = (ok, float(np.median(mg)))
        print(f"   {ang:>6.1f} {ang/9.0:>7.2f} "
              f"{res[False][0]:>4}/{len(rots):<4} {res[False][1]:>12.2f} "
              f"{res[True][0]:>4}/{len(rots):<4} {res[True][1]:>12.2f}")

    if args.synthetic:
        from exp_decode_variants import build_cases
        print("\n③ 合成边际集（0.13/0.16 × 失焦 1.6/2.0）")
        design, cases = build_cases((0.13, 0.16), (1.6, 2.0), (1, 2),
                                    I.LAYOUT_V3)
        for enabled in (False, True):
            I.PHASE_SEARCH = enabled
            dec = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                             layout_version=I.LAYOUT_V3)
            ok = wrong = 0
            for c in cases:
                r = dec.decode(c["image"])
                if r.accepted and r.digits == c["digits"]:
                    ok += 1
                elif r.accepted:
                    wrong += 1
            I.PHASE_SEARCH = True
            tag = "开" if enabled else "关"
            print(f"   相位搜索{tag}: {ok}/{len(cases)} 正确，错号 {wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
