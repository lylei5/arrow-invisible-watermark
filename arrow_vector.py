"""Parametric ARROW reconstruction; dimensions in cap-height units (X).

The supplied construction sheet specifies a 5X by X wordmark and four gaps.
Curves are manually reconstructed cubic Beziers, not raster-traced contours.
The screenshot does not specify Bezier handles: this is a reconstruction,
not a claim to possess the manufacturer's original vector artwork.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import arrow_wm as A

# Each contour begins with M, followed by L/C and an implicit closing edge.
GLYPHS = {
    'A': [
        [('M', 0,1), ('L',.405,0), ('L',.585,0), ('L',1,1),
         ('L',.818,1), ('L',.717,.733), ('L',.270,.733), ('L',.173,1)],
        [('M',.341,.592), ('L',.494,.201), ('L',.653,.592)]],
    'R': [
        [('M',0,1), ('L',0,0), ('L',.548,0),
         ('C',.713,0,.825,.077,.825,.255),
         ('C',.825,.407,.739,.496,.603,.534),
         ('C',.690,.573,.737,.658,.783,.758),
         ('L',.870,1), ('L',.683,1), ('L',.579,.778),
         ('C',.513,.627,.464,.554,.310,.554), ('L',.169,.554), ('L',.169,1)],
        [('M',.169,.139), ('L',.169,.415), ('L',.449,.415),
         ('C',.578,.415,.668,.375,.668,.268),
         ('C',.668,.178,.610,.139,.488,.139)]],
    'O': [
        [('M',.465,0), ('C',.180,0,0,.203,0,.500),
         ('C',0,.801,.180,1,.465,1), ('C',.753,1,.930,.801,.930,.500),
         ('C',.930,.203,.753,0,.465,0)],
        [('M',.465,.145), ('C',.269,.145,.167,.290,.167,.500),
         ('C',.167,.716,.269,.855,.465,.855),
         ('C',.663,.855,.763,.716,.763,.500),
         ('C',.763,.290,.663,.145,.465,.145)]],
    'W': [[('M',0,0), ('L',.157,0), ('L',.320,.726),
           ('L',.511,0), ('L',.697,0), ('L',.893,.726),
           ('L',1.065,0), ('L',1.220,0), ('L',.991,1),
           ('L',.809,1), ('L',.605,.239), ('L',.402,1), ('L',.220,1)]]
}
PLACEMENT = [('A',0.,1.), ('R',1.04,.87), ('R',1.94,.87),
             ('O',2.84,.93), ('W',3.78,1.22)]

def polygon(commands, samples=180):
    points=[]
    for command in commands:
        if command[0] in ('M','L'):
            p=np.array(command[1:],float); points.append(p)
        else:
            p0=p.copy(); p1,p2,p3=np.array(command[1:],float).reshape(3,2)
            t=np.linspace(0,1,samples+1)[1:,None]
            points.extend((1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3)
            p=p3
    return np.array(points)

def mask(width=8800, height=2400):
    out=np.zeros((height,width),np.uint8)
    scale=width/5.; y0=(height-scale)/2
    for name,x,_ in PLACEMENT:
        for j,cmd in enumerate(GLYPHS[name]):
            p=polygon(cmd)*scale+np.array([x*scale,y0])
            cv2.fillPoly(out,[np.rint(p).astype(np.int32)],255 if j==0 else 0)
    return out

def svg_text():
    paths=[]
    for name,x,_ in PLACEMENT:
        d=' '.join(c[0]+' '+' '.join(f'{v:g}' for v in c[1:])
                   for contour in GLYPHS[name] for c in contour+[('Z',)])
        paths.append(f'<path transform="translate({x:g} 0)" d="{d}"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="22mm" height="6mm" '
            'viewBox="0 0 5 1.363636363636">\n'
            '<g fill="#3a3a3a" fill-rule="evenodd" transform="translate(0 .181818181818)">\n'
            +'\n'.join(paths)+'\n</g>\n</svg>\n')

class VectorDesign(A.ArrowDesign):
    def _rasterize(self):
        return mask(A.CANVAS_W*A.SS,A.CANVAS_H*A.SS)
    def _place_letters(self):
        return [(n,x*A.CANVAS_W/5) for n,x,_ in PLACEMENT]
    def _compute_letter_centroids(self):
        out=[]
        for _,x,w in PLACEMENT:
            left=int(round(x*A.CANVAS_W/5)); right=int(round((x+w)*A.CANVAS_W/5))
            yy,xx=np.nonzero(self.mask1[:,left:right])
            out.append((float(left+xx.mean()),float(yy.mean())))
        return out

def export(directory):
    from PIL import Image
    d=Path(directory); d.mkdir(parents=True,exist_ok=True)
    (d/'ARROW_master.svg').write_text(svg_text(),encoding='utf-8')
    design=VectorDesign()
    Image.fromarray(design.template).save(d/'ARROW_master_2540dpi.png',dpi=(2540,2540))
    for dpi in (600,1200):
        w,h=round(22*dpi/25.4),round(6*dpi/25.4)
        coverage=cv2.resize(mask(w*4,h*4),(w,h),interpolation=cv2.INTER_AREA)
        image=(255-coverage.astype(float)*197/255).astype(np.uint8)
        Image.fromarray(image).save(d/f'ARROW_master_{dpi}dpi.png',dpi=(dpi,dpi))
    metadata={'canvas_mm':[22,6],'wordmark_mm':[22,4.4],
              'margins_mm':{'top':.8,'bottom':.8,'left':0,'right':0},
              'gap_X':[.04,.03,.03,.01], 'wordmark_aspect':5,
              'curve_source':'manual cubic Bezier reconstruction from construction sheet',
              'official_vector':False,'color':'#3a3a3a'}
    (d/'geometry.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return design

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    export(p.parse_args().out)
