"""depth ±10 vs ±8 的代价：同一条客户端链路上比召回与裕度。

v4 的码元周期本来就是 v3 的两倍（更显眼），如果再保持 ±10 的深度，可见度
是最差组合。depth 降到 ±8 能改善隐蔽性，但会直接削弱信号幅度——这里量化它。

用法: python bench_depth.py -n 2
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from bench_client_pipeline import client_roi_ladder, expand_box, render_photo

PAYLOADS = ("13800138000", "13912345678", "18600001234", "17701234567")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="depth 10 vs 8 对照（v4）")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.13, 0.16, 0.20])
    ap.add_argument("--blurs", type=float, nargs="+", default=[1.2, 1.6, 2.0])
    ap.add_argument("--depths", type=float, nargs="+", default=[10.0, 8.0])
    ap.add_argument("-n", type=int, default=2)
    args = ap.parse_args(argv)

    design = A.ArrowDesign()
    carriers = {d: I.InnerCarrier(design, depth=d, layout_version=I.LAYOUT_V4)
                for d in args.depths}
    decoders = {d: AD.Decoder(carrier="inner", design=design, depth=d,
                              layout_version=I.LAYOUT_V4) for d in args.depths}

    print("=" * 100)
    print(f"  v4 版式，depth 对照：" + " / ".join(f"±{d:g}" for d in args.depths))
    print(f"  {len(PAYLOADS)} 号码 × {len(args.scales)} 尺度 × {len(args.blurs)} 失焦 × {args.n} 种子")
    print("=" * 100)
    hdr = f"{'尺度':>5} {'失焦':>5} " + " ".join(f"{'±%g 对/拒/裕度' % d:>20}" for d in args.depths)
    print(hdr)
    print("-" * 100)
    tot = {d: [0, 0] for d in args.depths}
    for s in args.scales:
        for b in args.blurs:
            cells = {d: [0, 0] for d in args.depths}
            margs = {d: [] for d in args.depths}
            for dg in PAYLOADS:
                for sd in range(1, args.n + 1):
                    rng = np.random.default_rng(3100 + sd * 13 + int(s * 1000) + int(b * 10))
                    for d in args.depths:
                        plate = carriers[d].embed(dg)["plate"]
                        photo, box = render_photo(plate, s, rng=rng, blur=b, noise=3.0)
                        roi = expand_box(box, 0.25, photo.shape[1], photo.shape[0])
                        x, y, bw, bh = roi
                        packed = client_roi_ladder(photo[y:y + bh, x:x + bw], mode="maxq")
                        sent = cv2.imdecode(np.frombuffer(packed["data"], np.uint8),
                                            cv2.IMREAD_GRAYSCALE)
                        r = decoders[d].decode(sent)
                        margs[d].append(float(r.margin))
                        if r.accepted and r.digits == dg:
                            cells[d][0] += 1
                            tot[d][0] += 1
                        else:
                            cells[d][1] += 1
                            tot[d][1] += 1
            row = f"{s:>5.2f} {b:>5.1f} "
            for d in args.depths:
                ok, bad = cells[d]
                row += f"{ok:>4}/{bad:<3}  m={np.median(margs[d]):5.2f}   "
            print(row)
    print("-" * 100)
    for d in args.depths:
        ok, bad = tot[d]
        print(f"  ±{d:g}: 正确 {ok}/{ok+bad}  拒绝 {bad}"
              f"  （整组对照见上表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
