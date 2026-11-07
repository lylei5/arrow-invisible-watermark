"""Create a dimensioned A4 comparison sheet; print at actual size (100%)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
from io import BytesIO

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

import arrow_wm as A
import arrow_inner as I
from arrow_arc import ArcCarrier
from arrow_reference import ReferenceDesign


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--font',type=Path,default=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/simhei.ttf',
                   help='Chinese TrueType font; supply a local font path on Linux/macOS')
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    pdfmetrics.registerFont(TTFont('CN',str(args.font)))
    c=canvas.Canvas(str(args.output),pagesize=A4)
    c.setTitle('ARROW 22 x 6 mm print comparison')
    c.setAuthor('ARROW watermark research')
    def text(x,y,s,size=9):
        c.setFont('CN',size);c.drawString(x*mm,y*mm,s)
    def plate(image,x,y):
        b=BytesIO();Image.fromarray(image).save(b,format='PNG');b.seek(0)
        c.drawImage(ImageReader(b),x*mm,y*mm,width=22*mm,height=6*mm,mask='auto')
        c.setStrokeColorRGB(.65,.65,.65);c.setLineWidth(.2)
        # Crop marks lie outside the 22 x 6 mm tile.
        for xx,yy,sx,sy in [(x,y,-1,-1),(x+22,y,1,-1),(x,y+6,-1,1),(x+22,y+6,1,1)]:
            c.line((xx+sx*.6)*mm,yy*mm,(xx+sx*1.5)*mm,yy*mm)
            c.line(xx*mm,(yy+sy*.6)*mm,xx*mm,(yy+sy*1.5)*mm)
        c.setStrokeColorRGB(0,0,0)
    text(18,279,'ARROW 22 x 6 mm 打印对照',18)
    text(18,268,'画幅：22 x 6 mm；参考字标墨迹约 20 x 4.01 mm；径向峰值位移 0.08 mm。')
    text(18,261,'打印选择“实际大小 / 100%”，关闭“适应页面”。先用实体尺核对下方 50 mm 标尺。')
    c.setLineWidth(.5);c.line(18*mm,250*mm,68*mm,250*mm)
    for i in range(11):
        x=18+i*5;c.line(x*mm,250*mm,x*mm,(247 if i%2==0 else 248)*mm)
    text(73,248,'50 mm 校准尺')
    columns=[24,83,142]
    for x,title in zip(columns,['现有母版 / 生产 v4','参考母版 / v4','参考母版 / arc-v1']):
        text(x,233,title,10)
    d=ReferenceDesign();old=A.ArrowDesign()
    carriers=[I.InnerCarrier(old,layout_version=4),I.InnerCarrier(d,layout_version=4),ArcCarrier(d,layout_version=4)]
    payloads=['13800138000','13912345678','18600001234']
    for row,digits in enumerate(payloads):
        y=213-row*30
        for x,carrier in zip(columns,carriers):
            plate(carrier.embed(digits)['plate'],x,y)
            text(x,y-7,digits,9)
            text(x,y-13,'22 x 6 mm',7)
    for x,carrier in zip(columns,carriers):
        plate(carrier.design.template,x,120);text(x,113,'原图对照',9)
    text(18,96,'单色打印对照：reference-arc-v1 / 13800138000',10)
    image=carriers[-1].embed(payloads[0])['plate']
    mono=np.where(image>127,255,0).astype(np.uint8)
    for x in columns:plate(mono,x,82)
    text(18,65,'实拍记录：打印机 / 标称 DPI / 手机型号 / 字标相机像素宽度 / 光照 / 距离。')
    text(18,58,'每次拍摄一个字标。保留手机原始照片，并记录其格式和载荷；裁切时保留字标周围留白。')
    text(18,51,'参考版均用 tools/reference_watermark.py decode；中列加 --format reference-v4。')
    text(18,45,'左列使用原有 HTTP v4 接口；右列使用参考版解码命令的默认格式。',8)
    text(18,38,'本页用于实物验证。模拟实验结果不能替代真实打印、釉面反光与拍摄验证。',8)
    c.showPage();c.save()


if __name__=='__main__':main()
