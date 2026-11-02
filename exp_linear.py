"""gamma 线性化输入的 A/B。

假设：相机输出是 sRGB gamma 编码，物理阶跃在 gamma 域不对称，梯度质心
系统性偏向亮侧；模糊越宽偏移越大。线性化后斜坡对称 → 质心无偏。

**注意**必须同时线性化参考模板（`_apply_linear`），否则配准和半径参考错位。

用法:
    python exp_linear.py user-photo-1.jpg
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A


def run(imgs, expect, enabled, layout=3):
    I.LINEAR_INPUT = enabled
    dec = AD.Decoder(carrier="inner", layout_version=layout)
    ok = wrong = 0
    margins = []
    for img in imgs:
        r = dec.decode(img)
        margins.append(float(r.margin))
        if r.accepted and r.digits == expect:
            ok += 1
        elif r.accepted:
            wrong += 1
    I.LINEAR_INPUT = False
    return ok, wrong, margins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gamma 线性化 A/B")
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--expect", default="13800138000")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)

    if args.photo:
        import probe_user_photo as P
        gray = cv2.imread(args.photo, cv2.IMREAD_GRAYSCALE)
        boxes = P.find_marks(gray)
        boxes.sort(key=lambda b: (b[1], b[0]))
        H, W = gray.shape
        imgs = []
        for (x, y, w, h) in boxes:
            pad = int(round(0.06 * w))
            imgs.append(gray[max(0, y - pad):min(H, y + h + pad),
                             max(0, x - pad):min(W, x + w + pad)])
        print(f"真机照 {W}×{H}，{len(imgs)} 块")
        for enabled in (False, True):
            ok, wrong, mg = run(imgs, args.expect, enabled)
            print(f"  gamma 线性化 {'开' if enabled else '关'}: "
                  f"{ok}/{len(imgs)} 正确，错号 {wrong}，"
                  f"裕度中位 {np.median(mg):.2f}，最低 {min(mg):.2f}")

    if args.synthetic:
        from exp_decode_variants import build_cases
        print("\n合成边际集（0.13/0.16 × 失焦 1.2/1.6/2.0）")
        design, cases = build_cases((0.13, 0.16), (1.2, 1.6, 2.0), (1, 2),
                                    I.LAYOUT_V3)
        for enabled in (False, True):
            I.LINEAR_INPUT = enabled
            dec = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                             layout_version=I.LAYOUT_V3)
            ok = wrong = 0
            for c in cases:
                r = dec.decode(c["image"])
                if r.accepted and r.digits == c["digits"]:
                    ok += 1
                elif r.accepted:
                    wrong += 1
            I.LINEAR_INPUT = False
            print(f"  gamma 线性化 {'开' if enabled else '关'}: "
                  f"{ok}/{len(cases)} 正确，错号 {wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
