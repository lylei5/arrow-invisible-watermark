"""Produce a single A4 sheet of 72 production v5 plates at 22 x 6 mm."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
from pypdf.generic import ContentStream
from arrow_service import Decoder, FORMAT, MASTER_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--font', help='Optional path to a Chinese TrueType font')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    decoder = Decoder()
    carrier = decoder.carrier
    rng = random.Random(20261109)
    numbers = ['13800138000', '01088886666', '72510342187',
               '00000000000', '99999999999', '01234567890']
    while len(numbers) < 72:
        digits = f'{rng.randrange(10**11):011d}'
        if digits not in numbers:
            numbers.append(digits)
    font_path = Path(args.font) if args.font else Path('C:/Windows/Fonts/simhei.ttf')
    if font_path.is_file():
        pdfmetrics.registerFont(TTFont('Chinese', str(font_path)))
        chinese_font = 'Chinese'
    else:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        chinese_font = 'STSong-Light'
    target = out / 'ARROW_v5_72_samples_22x6mm_A4.pdf'
    doc = canvas.Canvas(str(target), pagesize=(210*mm, 297*mm), pageCompression=1)
    doc.setTitle('ARROW v5 | 72 samples | 22 x 6 mm | A4')
    doc.setAuthor('ARROW watermark project')
    doc.setViewerPreference('PrintScaling', 'None')

    def label(x, y, text, size=8, gray=0.15, font=None):
        doc.setFillGray(gray)
        doc.setFont(font or chinese_font, size)
        doc.drawString(x*mm, y*mm, text)

    label(19, 278, 'ARROW 水印打印样本', 19)
    label(19, 268, '72 个不同编号  /  v5  /  样本画布 22 × 6 mm', 10)
    label(19, 260, '打印选择“实际大小 / 100%”，关闭“适合页面”；使用 A4 纸。', 8)
    label(19, 254, '字标 22 × 4.4 mm，上下各留白 0.8 mm。核对号码位于样本下方。', 8)
    doc.setStrokeGray(.75)
    doc.setLineWidth(.3)
    doc.line(19*mm, 249*mm, 191*mm, 249*mm)
    rows = []
    for i, digits in enumerate(numbers):
        x, y = 19 + (i % 6)*30, 238 - (i // 6)*17
        plate = carrier.embed(digits)['plate']
        result = decoder.decode(plate)
        if not result.accepted or result.digits != digits:
            raise RuntimeError(f'Master decode failed at sample {i+1}: {result}')
        doc.drawImage(ImageReader(Image.fromarray(plate)), x*mm, y*mm,
                      width=22*mm, height=6*mm)
        label(x, y-4.2, f'{i+1:02d}  {digits}', 6.7, .25, 'Helvetica')
        rows.append(dict(index=i+1, digits=digits, x_mm=x, y_mm=y,
                         width_mm=22, height_mm=6, master_accepted=True,
                         margin=float(result.margin), cc=float(result.cc)))
        if (i+1) % 12 == 0:
            print(f'Generated and decoded {i+1}/72', flush=True)
    doc.setStrokeGray(.1)
    doc.setLineWidth(.6)
    doc.line(19*mm, 30*mm, 69*mm, 30*mm)
    for x in (19, 69):
        doc.line(x*mm, 28.5*mm, x*mm, 31.5*mm)
    label(19, 24, '校准线：打印后应为 50 mm', 8)
    label(82, 30, '拍摄时每次只取一个完整字标，保留四周少量白边。', 8)
    label(82, 24, '号码是解码核对值；同号样品共享服务端验证次数。', 8)
    label(19, 14, '2026-11-09  |  ARROW vector v5  |  径向调制幅度 ±0.08 mm  |  1 / 1', 7, .4)
    doc.showPage()
    doc.save()

    reader = PdfReader(target)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    assert abs(float(page.mediabox.width)/mm-210) < 1e-4
    assert abs(float(page.mediabox.height)/mm-297) < 1e-4
    # Inspect actual PDF image placement matrices, not just source parameters.
    operations = ContentStream(page.get_contents(), reader).operations
    image_transforms = []
    transform = None
    for operands, op in operations:
        if op == b'cm':
            transform = [float(n) for n in operands]
        if op == b'Do':
            assert transform is not None
            a, b, c, d, e, f = transform
            assert abs(a/mm-22) < 1e-5 and abs(d/mm-6) < 1e-5
            assert b == c == 0
            image_transforms.append(transform)
    assert len(image_transforms) == 72
    images = list(page.images)
    assert len(images) == 72
    assert all(im.image.size == (2200, 600) for im in images)
    with (out/'sample_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(format=FORMAT, master_id=MASTER_ID, page_mm=[210,297],
                    sample_mm=[22,6], wordmark_mm=[22,4.4], count=72,
                    source_pixels=[2200,600], source_dpi=2540,
                    depth_mm=.08, master_decode_passed=72,
                    pdf_image_sizes_verified=72, print_scaling='None',
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(), samples=rows,
                    holes=[dict(cells=h['n_cells'],
                                centroid_px=(h['c']/carrier.SS).tolist()) for h in carrier.holes])
    (out/'validation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k!='samples'}, indent=2))


if __name__ == '__main__':
    main()
