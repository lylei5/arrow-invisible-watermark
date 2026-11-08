"""Export physical-size experimental plates, quality metrics and a print sheet."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,io
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import arrow_wm as A
import arrow_inner as I
import arrow_v5 as V
from arrow_vector import export,GLYPHS,PLACEMENT

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    cv2.setNumThreads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    design=export(out/'master');samples=out/'samples';samples.mkdir(exist_ok=True)
    carriers={'vector_v4_d8':I.InnerCarrier(design),
              'v5_d8':V.Carrier(design),
              'v5_d6':V.Carrier(design,depth=6),
              'v5_d4':V.Carrier(design,depth=4)}
    metrics={};digits='13800138000';plates={'master':design.template}
    for name,carrier in carriers.items():
        result=carrier.embed(digits);plate=result['plate'];plates[name]=plate
        for dpi in (600,1200,2540):
            rendered=cv2.resize(plate,(round(22*dpi/25.4),round(6*dpi/25.4)),interpolation=cv2.INTER_AREA)
            Image.fromarray(rendered).save(samples/f'{name}_{digits}_{dpi}dpi.png',dpi=(dpi,dpi))
        delta=plate.astype(float)-design.template.astype(float)
        metrics[name]=dict(depth_mm=carrier.depth/100,PSNR_dB=float(cv2.PSNR(plate,design.template)),
                           changed_pixel_fraction=float(np.mean(delta!=0)),
                           mean_absolute_difference=float(np.mean(np.abs(delta))),
                           maximum_absolute_difference=float(np.max(np.abs(delta))))
    # Perceptual metrics on a large white canvas underweight the inner boundary;
    # report explicit boundary amplitude and a magnified comparison as well.
    (out/'quality.json').write_text(json.dumps(metrics,indent=2))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',27)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    preview=Image.new('RGB',(1600,1440),'#f4f5f7');draw=ImageDraw.Draw(preview)
    labels={'master':'Vector master | 22 x 6 mm canvas',
            'vector_v4_d8':'v4 on vector master | radial amplitude 0.08 mm',
            'v5_d8':'v5 cross-hole interleaving | radial amplitude 0.08 mm',
            'v5_d6':'v5 cross-hole interleaving | radial amplitude 0.06 mm',
            'v5_d4':'v5 cross-hole interleaving | radial amplitude 0.04 mm'}
    for j,(name,img) in enumerate(plates.items()):
        y=20+j*280;draw.text((30,y),labels[name],fill='#17202a',font=font)
        preview.paste(Image.fromarray(img).convert('RGB').resize((800,218)),(30,y+45))
        # Enlarge the O inner boundary at the same scale across variants.
        patch=Image.fromarray(img[125:345,1330:1550]).convert('RGB').resize((220,220))
        preview.paste(patch,(1100,y+40));draw.text((1350,y+70),'O detail',fill='#333333',font=small)
    preview.save(out/'comparison.png')
    pdf=canvas.Canvas(str(out/'ARROW_print_test_22x6mm.pdf'),pagesize=(210*mm,297*mm))
    pdf.setTitle('ARROW 22 x 6 mm print evaluation');pdf.setAuthor('ARROW watermark research')
    pdf.setFont('Helvetica-Bold',18);pdf.drawString(18*mm,278*mm,'ARROW / Physical print evaluation')
    pdf.setFont('Helvetica',10)
    for y,text in [(269,'Print at 100% / Actual size. Disable Fit to page.'),
                   (263,'Every logo canvas is exactly 22 x 6 mm; the wordmark is 22 x 4.4 mm.'),
                   (257,'v4 and v5 require their matching vector master and decoder. Experimental plates.')]:
        pdf.drawString(18*mm,y*mm,text)
    columns=[('Clean control',None),('v4 / 0.08 mm','vector_v4_d8'),
             ('v5 / 0.08 mm','v5_d8'),('v5 / 0.06 mm','v5_d6')]
    for col,(label,_) in enumerate(columns):
        pdf.setFont('Helvetica-Bold',9);pdf.drawString((18+45*col)*mm,241*mm,label)
    ids=['13800138000','00000000000','99999999999','01088886666','72510342187']
    for row,payload in enumerate(ids):
        y=224-row*32
        for col,(label,key) in enumerate(columns):
            x=18+45*col
            img=design.template if key is None else carriers[key].embed(payload)['plate']
            pdf.drawImage(ImageReader(Image.fromarray(img)),x*mm,y*mm,width=22*mm,height=6*mm)
            pdf.setFont('Helvetica',8);pdf.drawString(x*mm,(y-6)*mm,'No payload' if key is None else payload)
    pdf.setStrokeColorRGB(.1,.1,.1);pdf.setLineWidth(.5)
    pdf.line(18*mm,63*mm,68*mm,63*mm)
    for x in (18,68):pdf.line(x*mm,61*mm,x*mm,65*mm)
    pdf.setFont('Helvetica',9);pdf.drawString(18*mm,56*mm,'Measure this line: 50 mm')
    for y,text in [(43,'Capture each logo with at least 440 pixels across its width; also test 330 and 220 pixels.'),
                   (37,'Record printer DPI, ink, substrate, firing process, phone, distance, focus and glare.'),
                   (31,'Report exact payload acceptance, rejection and wrong acceptance separately.'),
                   (25,'Synthetic benchmark results do not establish performance on fired ceramic.')]:
        pdf.drawString(18*mm,y*mm,text)
    pdf.showPage();pdf.save()
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
