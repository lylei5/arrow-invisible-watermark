"""对照：反卷积重试（`RETRY_DECONV_SIGMAS`）开 / 关。

首解被拒时 `arrow_decode.Decoder.decode` 会依次做两次兜底：
  1. 换尺度重解（`RETRY_SCALES`，1.35x / 1.7x / 0.8x）
  2. 反卷积重解（`RETRY_DECONV_SIGMAS`）
两者共用同一条安全规则：必须是 v3、必须 CRC-7 通过、必须与首解候选同号。

本脚本在**同一条客户端链路**（裁 ROI + ≤96 KiB 高质量 JPEG）上，
只切 `RETRY_DECONV_SIGMAS` 这一个开关，看它到底挣回多少、有没有副作用。

    python exp_deconv_retry.py
    python exp_deconv_retry.py --scales 0.13 0.16 0.20 --blurs 1.6 2.0 -n 3
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Sequence, Tuple

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from exp_decode_variants import build_cases


def run(design, cases, enabled: bool) -> Tuple[Dict, int, float]:
    old = AD.RETRY_DECONV_SIGMAS
    AD.RETRY_DECONV_SIGMAS = (2.0,) if enabled else ()
    try:
        dec = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                         layout_version=I.LAYOUT_V3)
        buckets: Dict[Tuple[float, float], List[int]] = {}
        wrong = 0
        t0 = time.time()
        for c in cases:
            r = dec.decode(c["image"])
            b = buckets.setdefault((c["scale"], c["blur"]), [0, 0, 0])
            if r.accepted and r.digits == c["digits"]:
                b[0] += 1
            elif r.accepted:
                b[1] += 1
                wrong += 1
                print(f"    !! 错号 {c['digits']} -> {r.digits} "
                      f"(scale {c['scale']} blur {c['blur']})")
            else:
                b[2] += 1
        return buckets, wrong, (time.time() - t0) / max(len(cases), 1)
    finally:
        AD.RETRY_DECONV_SIGMAS = old


def line(label, buckets, wrong, sec, scales, blurs):
    total_ok = sum(v[0] for v in buckets.values())
    total = sum(sum(v) for v in buckets.values())
    cells = []
    for s in scales:
        cells.append(f"s={s:.2f}:" + " ".join(
            f"{buckets.get((s, b), [0,0,0])[0]}/{sum(buckets.get((s, b), [0,0,0]))}"
            for b in blurs))
    print(f"  {label:<18} {total_ok}/{total}  错号 {wrong}  "
          f"{sec:.2f}s/张   " + "  ".join(cells))
    return total_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="反卷积重试对照实验")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.13, 0.16])
    ap.add_argument("--blurs", type=float, nargs="+", default=[1.6, 2.0])
    ap.add_argument("-n", type=int, default=2)
    args = ap.parse_args(argv)

    design, cases = build_cases(tuple(args.scales), tuple(args.blurs),
                                tuple(range(1, args.n + 1)), I.LAYOUT_V3)
    print(f"共 {len(cases)} 张（客户端走修好的质量阶梯），只切 RETRY_DECONV_SIGMAS")
    print(f"  失焦列顺序：{args.blurs}")
    off_b, off_w, off_s = run(design, cases, enabled=False)
    on_b, on_w, on_s = run(design, cases, enabled=True)
    print()
    a = line("反卷积重试 关", off_b, off_w, off_s, args.scales, args.blurs)
    b = line("反卷积重试 开", on_b, on_w, on_s, args.scales, args.blurs)
    print(f"\n  净变化 {b - a:+d} 张；错号 {off_w} -> {on_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
