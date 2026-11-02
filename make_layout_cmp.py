"""v3 / v4 印版的外观对照：同一尺寸、同一位置放大看。

格式升级的代价是**码元周期翻倍**（0.164 → 0.328 mm），空间频率从 6.1 降到
3.05 cyc/mm，更接近人眼敏感频段，所以同深度下更看得出来。这里把两者按
「实拍里 logo 约 520 px 宽」渲染，再各放大 4 倍看边缘。
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import arrow_wm as A
import arrow_inner as I

PAYLOADS = ("13800138000", "13912345678", "17701234567")


def main(argv=None) -> int:
    out = (argv or sys.argv[1:] or ["layout_v3_v4_cmp.png"])[0]
    design = A.ArrowDesign()
    c3 = I.InnerCarrier(design, depth=I.DEPTH, layout_version=3)
    c4 = I.InnerCarrier(design, depth=I.DEPTH, layout_version=4)

    print(f"{'号码':<13}{'v3 墨覆盖率':>12}{'v4 墨覆盖率':>12}")
    rows = []
    for dg in PAYLOADS:
        p3 = c3.embed(dg)["plate"]
        p4 = c4.embed(dg)["plate"]
        i3 = float((p3 < 200).mean()) * 100
        i4 = float((p4 < 200).mean()) * 100
        print(f"{dg:<13}{i3:>11.2f}%{i4:>11.2f}%")
        rows.append((dg, p3, p4))

    # 视觉对照：每行 "v3 全幅 | v4 全幅 | v3 放大 | v4 放大"
    scale_w = 520          # 相当于实拍里 logo 约 520 px 宽
    tile_h = int(round(scale_w * 600 / 2200))
    zoom = 4
    # 放大区域：O 内环附近的笔画边界（取中间偏左）
    zx, zy, zw, zh = 380, 180, 220, 130
    panels = []
    for dg, p3, p4 in rows:
        tile = []
        for p in (p3, p4):
            small = cv2.resize(p, (scale_w, tile_h), interpolation=cv2.INTER_AREA)
            tile.append(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR))
        for p in (p3, p4):
            crop = p[zy:zy + zh, zx:zx + zw]
            big = cv2.resize(crop, (zw * zoom, zh * zoom),
                             interpolation=cv2.INTER_NEAREST)
            big = cv2.resize(big, (scale_w, tile_h), interpolation=cv2.INTER_AREA)
            tile.append(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
        line = np.hstack(tile)
        cv2.putText(line, f"v3 {dg}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 180), 1, cv2.LINE_AA)
        cv2.putText(line, f"v4 {dg}", (scale_w + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 1, cv2.LINE_AA)
        panels.append(line)
        panels.append(np.full((10, line.shape[1], 3), 255, np.uint8))
    img = np.vstack(panels)
    cv2.imwrite(out, img)
    print(f"[输出] {out}  ({img.shape[1]}×{img.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
