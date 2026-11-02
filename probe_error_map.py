"""把失败样本的软值画回轮廓位置：错误是成簇（几何/打印）还是分散（噪声/模糊）。

外部评审（Gemini 与 GPT 各自独立）都指出：我用「给真机图叠高斯模糊」只能证明
**对 σ 敏感**，不能证明 **σ 就是那张失败块的病因**。真正的下一步是：载荷已知，
把逐码元软值映射回孔/角度位置，看错误空间分布。

判断口径：
  - 错误成簇（集中在某一个孔或某一个角度区间）→ 几何残差 / 曲率 / 打印缺陷
  - 错误分散（11 组每组都错 1 个）→ 噪声或模糊，且 Hamming 应当能纠

用法:
    python probe_error_map.py user-photo-1.jpg
"""
from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A


def main(argv=None) -> int:
    import cv2
    path = (argv or sys.argv[1:])[0]
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    import probe_user_photo as P
    boxes = P.find_marks(gray)
    boxes.sort(key=lambda b: (b[1], b[0]))

    design = A.ArrowDesign()
    carrier = I.InnerCarrier(design, depth=I.DEPTH, layout_version=I.LAYOUT_V3)
    dec = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                     layout_version=I.LAYOUT_V3)

    # 码元 -> 孔 的映射
    cell_hole: List[int] = []
    for hi, h in enumerate(carrier.holes):
        cell_hole.extend([hi] * h["n_cells"])
    hole_name = ["O内环", "R1字怀", "R2字怀", "A上三角"]

    # 逻辑 bit j -> 差分对的第 i 对 -> 码元 2i / 2i+1
    pair_order = list(carrier.pair_order)
    bit_cells = [(2 * pair_order[j], 2 * pair_order[j] + 1)
                 for j in range(len(pair_order))]

    H, W = gray.shape
    for idx, (x, y, w, h) in enumerate(boxes):
        pad = int(round(0.06 * w))
        crop = gray[max(0, y - pad):min(H, y + h + pad),
                    max(0, x - pad):min(W, x + w + pad)]
        r = dec.decode(crop, return_debug=True)
        raw = r.raw or {}
        conf = raw.get("conf")
        if conf is None:
            print(f"#{idx} 没有软值（{r.reason}）")
            continue
        conf = np.asarray(conf, np.float64)
        ms = np.asarray(raw.get("group_margins") or [], np.float64)
        print("=" * 96)
        print(f"#{idx} 框 ({x},{y},{w},{h})  accepted={r.accepted} digits={r.digits} "
              f"margin={r.margin:.2f} margin_min={r.margin_min:.2f} "
              f"crc_ok={raw.get('crc_ok')} reason={r.reason or '-'}")
        print("  11 组裕度: " + " ".join(f"{v:5.2f}" for v in ms))
        # 最弱的 8 个 bit 落在哪
        order = np.argsort(np.abs(conf))
        print("  最弱的 8 个 bit（组号/组内位/软值/所属孔）:")
        for j in order[:8]:
            cells = bit_cells[j]
            holes = sorted({cell_hole[c] for c in cells})
            print(f"     bit{j:>2} (组{j//7} 位{j%7})  conf={conf[j]:+7.3f}  "
                  f"孔={[hole_name[k] for k in holes]}")
        # 每组的孔构成
        print("  每组涉及的孔:")
        for g in range(len(conf) // 7):
            js = range(g * 7, g * 7 + 7)
            holes = sorted({cell_hole[c] for j in js for c in bit_cells[j]})
            weak = min(abs(conf[j]) for j in js)
            print(f"     组{g}: 孔 {[hole_name[k] for k in holes]}  "
                  f"组内最弱 |conf|={weak:.3f}  裕度={ms[g] if g < len(ms) else float('nan'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
