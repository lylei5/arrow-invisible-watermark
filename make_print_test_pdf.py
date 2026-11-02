"""把内孔 v3 印版按**出厂真实尺寸**排成可打印的 A4 测试页（PDF）。

尺寸只有一种：**22.0 × 6.0 mm**（2200×600 px 的 1:1 出厂规格，等效 2540 dpi）。
不做尺寸阶梯——要测的就是线上那一个工作点。

为什么需要这个脚本
-----------------
仓库里 `export_print_plates.py` 导出的是**边缘载体**的单张成品图，给制版厂用。
要拿打印件验证线上链路（小程序 → 云托管 → `arrow_decode`），需要的是另一件事：

* **内孔 v3** 印版（线上默认 `DECODE_INNER_LAYOUT=3`），不是边缘版；
* 每块外面套一圈 0.35 mm 细黑框：线上印版是贴在深色陶瓷上的白色小卡，
  客户端 `coarseLocateArrowRoi` 靠「亮块」找它；直接印在白纸上没有边界，
  会退回「整幅上传」那条会掉分辨率的路径；
* 一条 50 mm 校准尺 + 一条 22 mm 参照线：打印机默认「适合页面」会偷偷缩放，
  量一下就知道，不然会以为算法不行；
* 每块旁边标出**它编码的号码**（标签在框外，剪的时候只剪黑框）。

2.2 cm 是很小的东西：拍摄距离 6–8 cm、让白块占画面 1/3 以上。10 cm 以外
基本解不出，那不是 bug，是每个码元拿不到足够的相机像素（见
`docs/解码效果诊断与改进.md`）。

用法
----
    python make_print_test_pdf.py                     # 默认两页
    python make_print_test_pdf.py --out test.pdf
    python make_print_test_pdf.py --copies 14         # 第 1 页多印几份
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import arrow_inner as I
import arrow_wm as A

# ---- 出厂规格（不要改，这就是线上那一版）----------------------------------
PLATE_W_MM = 22.0
PLATE_H_MM = 6.0
PLATE_ASPECT = A.CANVAS_H / A.CANVAS_W          # 600/2200 = 3/11

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
PATCH_PAD_MM = 1.6            # 白边，给客户端一点余量去框住亮块
PATCH_BORDER_MM = 0.35
PATCH_W_MM = PLATE_W_MM + 2 * PATCH_PAD_MM
PATCH_H_MM = PLATE_H_MM + 2 * PATCH_PAD_MM

INK = Color(0.10, 0.10, 0.10)
GREY = Color(0.42, 0.42, 0.42)
RULE = Color(0.78, 0.78, 0.78)

DEFAULT_NUMBER = "13800138000"
DEFAULT_CHECK_NUMBERS = ("13800138000", "13912345678", "18600001234",
                         "17701234567", "13500001111")
# 想让白块占画面宽度 25%：手机主摄水平视场约 65°，画幅宽 ≈ 1.27×距离
_FILL, _FOV_W = 0.25, 1.27


def _top(mm_from_top: float) -> float:
    """从页面顶部量起的 mm -> PDF 的 y 坐标（pt）。"""
    return PAGE_H - mm_from_top * mm


def _register_fonts() -> Tuple[str, str]:
    regular = bold = None
    for path, idx in ((r"C:\Windows\Fonts\msyh.ttc", 0),
                      (r"C:\Windows\Fonts\simhei.ttf", 0),
                      (r"C:\Windows\Fonts\Deng.ttf", 0)):
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("CJK", path, subfontIndex=idx))
                regular = "CJK"
                break
            except Exception:
                continue
    for path, idx in ((r"C:\Windows\Fonts\msyhbd.ttc", 0),
                      (r"C:\Windows\Fonts\simhei.ttf", 0)):
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("CJKB", path, subfontIndex=idx))
                bold = "CJKB"
                break
            except Exception:
                continue
    if regular is None:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        regular = "STSong-Light"
    return regular, (bold or regular)


def _plate_png(digits: Optional[str], carrier: I.InnerCarrier, tmpdir: str) -> str:
    """写一张印版到 ASCII 路径的 PNG（reportlab 不吃非 ASCII 路径）。"""
    import cv2
    if digits is None:
        plate = carrier.design.template
        name = "blank"
    else:
        plate = carrier.embed(digits)["plate"]
        name = digits
    path = os.path.join(tmpdir, f"plate_{name}.png")
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(plate.astype(np.uint8)))
    if not ok:
        raise RuntimeError("PNG 编码失败")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def _draw_patch(c: canvas.Canvas, png: str, x: float, top_pt: float) -> None:
    """在 (x, top_pt) 画「白底 + 细黑框 + 22.0×6.0 mm 印版」。"""
    w = PLATE_W_MM * mm
    h = PLATE_H_MM * mm
    pad = PATCH_PAD_MM * mm
    y = top_pt - h
    c.setFillColor(Color(1, 1, 1))
    c.rect(x - pad, y - pad, w + 2 * pad, h + 2 * pad, stroke=0, fill=1)
    c.setStrokeColor(INK)
    c.setLineWidth(PATCH_BORDER_MM * mm)
    c.rect(x - pad, y - pad, w + 2 * pad, h + 2 * pad, stroke=1, fill=0)
    c.drawImage(png, x, y, width=w, height=h,
                preserveAspectRatio=True, mask=None)


def _text(c, font, size, x, y, text, color=INK):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def _wrap(c, font, size, x, y, lines, leading=None, color=INK):
    leading = leading or size * 1.55
    c.setFont(font, size)
    c.setFillColor(color)
    for i, line in enumerate(lines):
        c.drawString(x, y - i * leading, line)
    return y - (len(lines) - 1) * leading


def _ruler(c, font, x: float, y: float, length_mm: float) -> None:
    L = length_mm * mm
    c.setStrokeColor(INK)
    c.setLineWidth(0.4 * mm)
    c.line(x, y, x + L, y)
    for i in range(0, int(length_mm) + 1, 5):
        tall = 2.6 * mm if i % 10 == 0 else 1.5 * mm
        c.setLineWidth(0.3 * mm)
        c.line(x + i * mm, y - tall, x + i * mm, y + tall)
    _text(c, font, 7.5, x, y - 6.4 * mm,
          f"校准尺：这条线必须是 {length_mm:.0f} mm，短了就说明打印机缩放了")


def _page_footer(c, font, page: int, total: int, extra: str = "") -> None:
    c.setFont(font, 7)
    c.setFillColor(GREY)
    c.drawString(MARGIN, MARGIN - 4 * mm,
                 f"页 {page}/{total} · 生成于 {_dt.date.today():%Y-%m-%d} · "
                 f"印版 {PLATE_W_MM:.0f}×{PLATE_H_MM:.0f} mm（2200×600 px，"
                 f"等效 2540 dpi）{extra}")


def build(out_path: str, copies: int, ladder_number: str,
          check_numbers: Sequence[str], carrier: I.InnerCarrier,
          font: str, bold: str, layout_label: str = "v4") -> None:
    tmpdir = tempfile.mkdtemp(prefix="arrow_pdf_")
    c = canvas.Canvas(out_path, pagesize=A4)
    c.setTitle("ARROW 隐形水印打印测试页")
    c.setAuthor("arrow-invisible-watermark")

    ladder_png = _plate_png(ladder_number, carrier, tmpdir)
    blank_png = _plate_png(None, carrier, tmpdir)
    dist_cm = PLATE_W_MM / (_FILL * _FOV_W) / 10.0

    # ======================= 第 1 页：同号码多份 =======================
    _text(c, bold, 15, MARGIN, _top(15),
          "ARROW 隐形水印 · 打印测试页 · " + layout_label)
    _text(c, font, 8.5, MARGIN, _top(25),
          f"真实出厂尺寸：每块印版 {PLATE_W_MM:.1f} × {PLATE_H_MM:.1f} mm，"
          f"与产品上的完全一致（这块是 {ladder_number}）。", GREY)
    _text(c, font, 8.5, MARGIN, _top(34),
          "打印：A4 · 缩放 100%（务必关掉「适合页面 / 缩放以适合」）· "
          "≥600 dpi · 普通白纸", GREY)

    _ruler(c, font, MARGIN, _top(48), 50.0)

    # 22 mm 参照线：确认印版被按真实宽度输出
    ref_y = _top(72)
    c.setStrokeColor(INK)
    c.setLineWidth(0.4 * mm)
    c.line(MARGIN, ref_y, MARGIN + PLATE_W_MM * mm, ref_y)
    c.setLineWidth(0.3 * mm)
    c.line(MARGIN, ref_y - 2.4 * mm, MARGIN, ref_y + 2.4 * mm)
    c.line(MARGIN + PLATE_W_MM * mm, ref_y - 2.4 * mm,
           MARGIN + PLATE_W_MM * mm, ref_y + 2.4 * mm)
    _text(c, font, 7.5, MARGIN + PLATE_W_MM * mm + 4 * mm, ref_y - 2,
          f"参照：印版宽度应为 {PLATE_W_MM:.0f} mm（应与下面任意一块的内侧宽度相等）",
          GREY)

    # 网格：带码印版 + 1 块无码对照。行数按实际份数自适应：22 mm 的印版
    # 一片纸能放很多，多印几份方便贴到不同物件上重复试。
    cols, col_pitch, row_pitch = 3, 60.0, 25.0
    grid_top_mm = 92.0
    cells: List[Optional[str]] = [ladder_number] * copies + [None]
    n_rows_used = (len(cells) + cols - 1) // cols
    for idx, digits in enumerate(cells):
        r, col = divmod(idx, cols)
        x = MARGIN + col * col_pitch * mm
        top = _top(grid_top_mm + r * row_pitch)
        _draw_patch(c, blank_png if digits is None else ladder_png, x, top)
        label_x = x + PATCH_W_MM * mm + 3.5 * mm
        _text(c, bold, 8, label_x, top - 4.2 * mm,
              "无码对照" if digits is None else digits)
        _text(c, font, 7, label_x, top - 8.6 * mm,
              "应显示「未识别」" if digits is None else
              f"{PLATE_W_MM:.0f}×{PLATE_H_MM:.0f} mm", GREY)

    # 拍摄与剪裁说明
    box_top = _top(grid_top_mm + n_rows_used * row_pitch - 4)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.25 * mm)
    c.line(MARGIN, box_top, PAGE_W - MARGIN, box_top)
    _text(c, bold, 9, MARGIN, box_top - 7 * mm, "怎么用")
    _wrap(c, font, 7.8, MARGIN, box_top - 14 * mm, [
        "1  沿黑框剪下，贴在深色物体上（黑框外再留 5 mm 以上背景），或手持对着深色背景拍。",
        f"2  拍摄距离 {dist_cm:.0f}–{dist_cm + 2:.0f} cm，让白块占画面宽度的 1/3 以上。"
        f"2.2 cm 的印版在 10 cm 以外基本解不出。",
        "3  光从侧面斜着打，避开釉面/纸面反光——反光是仅次于分辨率的干扰源。",
        "4  剪的时候只剪黑框，框外的号码标签不要剪进去；贴之前记一下这块是哪个号码。",
        "5  全部「未识别」时：先量 50 mm 校准尺，再确认距离与白块占画比例。",
    ], leading=10.6)

    _page_footer(c, font, 1, 2)
    c.showPage()

    # ======================= 第 2 页：号码核对 =======================
    _text(c, bold, 15, MARGIN, _top(15), "号码核对页")
    _text(c, font, 8.5, MARGIN, _top(25),
          f"每个号码 2 块，都是 {PLATE_W_MM:.1f}×{PLATE_H_MM:.1f} mm。"
          f"逐块拍，看小程序显示的是不是标注号码。", GREY)
    _text(c, font, 8.5, MARGIN, _top(34),
          "只要出现「别的号码」就是误读——比识别失败严重得多，请记下是哪一块。", GREY)

    pngs = {d: _plate_png(d, carrier, tmpdir) for d in check_numbers}
    row_pitch2 = 18.0
    for i, digits in enumerate(check_numbers):
        top = _top(50 + i * row_pitch2)
        _draw_patch(c, pngs[digits], MARGIN + 4 * mm, top)
        _draw_patch(c, pngs[digits], MARGIN + 36 * mm, top)
        _text(c, bold, 11, MARGIN + 72 * mm, top - 4.6 * mm, digits)
        _text(c, font, 7.5, MARGIN + 72 * mm, top - 9.4 * mm,
              f"{PLATE_W_MM:.0f}×{PLATE_H_MM:.0f} mm · 约 {dist_cm:.0f} cm 拍摄",
              GREY)

    # 记录表
    tbl_top = _top(50 + len(check_numbers) * row_pitch2 + 10)
    cols_mm = (52.0, 34.0, 34.0, 60.0)
    row_h = 8.5 * mm
    n_rows = 1 + len(check_numbers)
    c.setStrokeColor(Color(0.55, 0.55, 0.55))
    c.setLineWidth(0.25 * mm)
    for r in range(n_rows + 1):
        c.line(MARGIN, tbl_top - r * row_h,
               MARGIN + sum(cols_mm) * mm, tbl_top - r * row_h)
    xx = MARGIN
    for width in cols_mm:
        c.line(xx, tbl_top, xx, tbl_top - n_rows * row_h)
        xx += width * mm
    c.line(xx, tbl_top, xx, tbl_top - n_rows * row_h)
    for i, label in enumerate(("印版号码", "第 1 块", "第 2 块",
                               "备注（通过 / 未识别 / 误读）")):
        _text(c, font, 8, MARGIN + sum(cols_mm[:i]) * mm + 2 * mm,
              tbl_top - 6.0 * mm, label)
    for i, digits in enumerate(check_numbers):
        _text(c, font, 8.5, MARGIN + 2 * mm,
              tbl_top - (i + 1) * row_h - 6.0 * mm, digits)

    note_y = tbl_top - n_rows * row_h - 14 * mm
    _wrap(c, font, 8, MARGIN, note_y, [
        "每块拍 3 张：近一点、远一点、换个角度。填进上面的表。",
        "如果全部「未识别」：先量 50 mm 校准尺，再确认拍摄距离 6–8 cm、白块占画面 1/3 以上。",
    ], leading=11, color=GREY)

    _page_footer(c, font, 2, 2, " · " + layout_label)
    c.showPage()
    c.save()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="生成 ARROW 打印测试页 PDF（22×6 mm 出厂尺寸）")
    ap.add_argument("--out", default="ARROW-watermark-A4-22mm.pdf")
    ap.add_argument("--copies", type=int, default=11,
                    help="第 1 页印多少份带码印版（默认 11，另加 1 块无码对照）")
    ap.add_argument("--ladder-number", default=DEFAULT_NUMBER,
                    help="第 1 页用的号码")
    ap.add_argument("--check-numbers", nargs="+",
                    default=list(DEFAULT_CHECK_NUMBERS),
                    help="第 2 页要核对的号码（各 2 块）")
    args = ap.parse_args(argv)

    design = A.ArrowDesign()
    carrier = I.InnerCarrier(design, depth=I.DEPTH, layout_version=I.LAYOUT_V4)
    font, bold = _register_fonts()
    out = os.path.abspath(args.out)
    label = "v4"
    build(out, max(1, args.copies), args.ladder_number, args.check_numbers,
          carrier, font, bold, layout_label=label)
    print(f"[输出] {out}")
    print(f"[规格] 每块印版 {PLATE_W_MM:.1f} × {PLATE_H_MM:.1f} mm"
          f"（含白边与黑框 {PATCH_W_MM:.1f} × {PATCH_H_MM:.1f} mm）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
