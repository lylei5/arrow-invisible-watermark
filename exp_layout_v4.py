"""格式升级对照：v3（Manchester，154 码元）vs v4（NRZ，77 码元）。

只改「一个 bit 占几个码元」，编码/CRC-7/Hamming(7,4) 完全复用：
  v3: 77 bit × 2 码元 = 154 码元，每码元 16.4 底版 px = 0.164 mm
  v4: 77 bit × 1 码元 = 77 码元，每码元 32.8 底版 px = 0.328 mm
v4 的代价是失去直流免疫（用「每孔减均值」补回）与长游程问题（用固定加扰打散）。

理论预期：高斯模糊的基波衰减 MTF = exp(-(π²/2)(σ/L)²)。码元翻倍 → σ/L 减半 →
在 σ/L=0.8 处 MTF 从 4% 升到 30%，约 +9 dB。

用法:
    python exp_layout_v4.py -n 2
    python exp_layout_v4.py --scales 0.13 0.16 --blurs 1.6 2.0
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from bench_client_pipeline import (client_roi_ladder, expand_box, render_photo)

PAYLOADS = ("13800138000", "13912345678", "18600001234", "17701234567")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v3 vs v4 格式对照")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.13, 0.16, 0.20])
    ap.add_argument("--blurs", type=float, nargs="+", default=[1.2, 1.6, 2.0])
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--noise", type=float, default=3.0)
    args = ap.parse_args(argv)

    design = A.ArrowDesign()
    carriers = {lv: I.InnerCarrier(design, depth=I.DEPTH, layout_version=lv)
                for lv in (3, 4)}
    decoders = {lv: AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                               layout_version=lv) for lv in (3, 4)}

    print("=" * 104)
    print(f"  v3(Manchester,154码元,16.4px/码元) vs v4(NRZ,77码元,32.8px/码元)")
    print(f"  {len(PAYLOADS)} 号码 × {len(args.scales)} 尺度 × "
          f"{len(args.blurs)} 失焦 × {args.n} 种子；客户端走 maxq 质量阶梯")
    print("=" * 104)
    hdr = f"{'尺度':>5} {'失焦':>5} " + " ".join(
        f"{'v%d 对/错/拒' % lv:>16}" for lv in (3, 4)) + "   中位裕度 v3/v4"
    print(hdr)
    print("-" * 104)

    tot = {3: [0, 0, 0], 4: [0, 0, 0]}
    for s in args.scales:
        for b in args.blurs:
            cells = {3: [0, 0, 0], 4: [0, 0, 0]}
            marg = {3: [], 4: []}
            t0 = time.time()
            for dg in PAYLOADS:
                for sd in range(1, args.n + 1):
                    rng = np.random.default_rng(
                        9100 + sd * 17 + int(s * 1000) + int(b * 10))
                    for lv in (3, 4):
                        plate = carriers[lv].embed(dg)["plate"]
                        photo, box = render_photo(plate, s, rng=rng, blur=b,
                                                  noise=args.noise)
                        roi = expand_box(box, 0.25, photo.shape[1], photo.shape[0])
                        x, y, bw, bh = roi
                        packed = client_roi_ladder(
                            photo[y:y + bh, x:x + bw], mode="maxq")
                        sent = cv2.imdecode(
                            np.frombuffer(packed["data"], np.uint8),
                            cv2.IMREAD_GRAYSCALE)
                        r = decoders[lv].decode(sent)
                        marg[lv].append(float(r.margin))
                        if r.accepted and r.digits == dg:
                            cells[lv][0] += 1
                        elif r.accepted:
                            cells[lv][1] += 1
                        else:
                            cells[lv][2] += 1
            for lv in (3, 4):
                for k in range(3):
                    tot[lv][k] += cells[lv][k]
            row = f"{s:>5.2f} {b:>5.1f} "
            for lv in (3, 4):
                c = cells[lv]
                row += f"{c[0]:>6}/{c[1]:<2}/{c[2]:<2}      "
            row += f"   {np.median(marg[3]):.2f}/{np.median(marg[4]):.2f}"
            print(row + f"   ({time.time()-t0:.0f}s)")
    print("-" * 104)
    n = len(PAYLOADS) * len(args.scales) * len(args.blurs) * args.n
    for lv in (3, 4):
        c = tot[lv]
        print(f"  合计 v{lv}: 正确 {c[0]}/{n}  错号 {c[1]}  拒绝 {c[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
