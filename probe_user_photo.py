"""用真机照片做诊断：逐块解码 + 折算每码元像素 + 走一遍客户端链路。

照片是「打印测试页」的实拍。它比仿真更有价值的地方在于：
仿真里的失焦/噪声/光照是我设的，这里全是真的。

用法:
    python probe_user_photo.py user-photo-1.jpg
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import cv2
import numpy as np

import arrow_decode as AD

ELEM_DESIGN_PX = 2525.0 / 154.0      # 每个码元的底版弧长
CANVAS_W = 2200


def find_marks(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """找「白底 + 黑框」的小片：取亮连通域，按宽高比过滤。

    印版是白色小卡贴在深色物体上；这里虽然背景是白纸，但黑框把每块隔开了，
    所以亮连通域仍然是每块的内腔。
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 40 or h < 12:
            continue
        aspect = w / max(h, 1)
        if not (1.8 <= aspect <= 6.5):
            continue
        if area / float(max(w * h, 1)) < 0.5:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    out.sort(key=lambda b: (b[1] // 60, b[0]))
    return out


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("读不到图")
        return 1
    H, W = gray.shape
    print(f"照片 {W}×{H} px")
    dec = AD.Decoder(carrier="inner")

    full = dec.decode(gray)
    print(f"\n[整幅] accepted={full.accepted} digits={full.digits} "
          f"margin={full.margin:.2f} margin_min={full.margin_min:.2f} "
          f"cc={full.cc:.3f} logo_scale={full.logo_scale:.3f} "
          f"reason={full.reason or '-'}")

    boxes = find_marks(gray)
    print(f"\n找到 {len(boxes)} 块候选\n")
    print(f"{'#':>2} {'框 x,y,w,h':>22} {'px/码元':>8} {'logo_scale':>11} "
          f"{'accepted':>9} {'digits':>12} {'margin':>7} {'cc':>6} {'reason':>16}")
    print("-" * 104)
    ok = 0
    for i, (x, y, w, h) in enumerate(boxes):
        pad = int(round(0.06 * w))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = gray[y0:y1, x0:x1]
        r = dec.decode(crop)
        per = CANVAS_W and (0.0074545 * ELEM_DESIGN_PX * 0) or 0  # 占位
        elem_px = (16.4 * (r.logo_scale or 0))
        ok += 1 if r.accepted else 0
        print(f"{i:>2} {f'{x},{y},{w},{h}':>22} {elem_px:>8.2f} "
              f"{(r.logo_scale or 0):>11.3f} {str(r.accepted):>9} "
              f"{str(r.digits):>12} {r.margin:>7.2f} {r.cc:>6.3f} "
              f"{(r.reason or '-'):>16}")
    print("-" * 104)
    print(f"逐块 {ok}/{len(boxes)} 接受；整幅 {'接受' if full.accepted else '拒绝'}")
    print("\n注：末块如果是「无码对照」，它被拒绝是**正确**行为。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
