"""Seeded physical-size print/camera benchmark; no capture truth in blind decode.

python experiments/channel_review.py --out experiments/dev --n 6 --seed 41000
Prints progress and writes each result immediately. Ctrl+C preserves completed rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from arrow_arc import ArcCarrier, ArcDecoder, NuisanceDecoder
from arrow_reference import ReferenceDesign


def capture(plate, *, width, blur, dpi, seed, local_defect=False):
    """Coverage raster at known DPI -> anti-aliased perspective -> camera PSF.

    width is the 22-mm canvas width in camera pixels. No physical print or
    ceramic firing is asserted. Geometry is returned only for oracle diagnosis.
    """
    rng = np.random.default_rng(seed)
    pw, ph = round(22/25.4*dpi), round(6/25.4*dpi)
    printed = cv2.resize(plate, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
    if local_defect:
        # Blur the A counter's physical neighborhood without changing outer
        # geometry elsewhere. Location is a channel parameter, never a label.
        x0, x1 = round(pw*.075), round(pw*.20)
        y0, y1 = round(ph*.22), round(ph*.72)
        b = cv2.GaussianBlur(printed, (0, 0), max(1.0, pw/2200*15))
        printed[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    # Pixel integration anti-alias before any camera-domain downsampling.
    s = width/pw
    aa = 0.45/s
    printed = cv2.GaussianBlur(printed, (0, 0), aa)
    h = width*6/22
    cx, cy = 500+rng.uniform(-55, 55), 300+rng.uniform(-35, 35)
    roll = rng.uniform(-6, 6)*np.pi/180
    q = np.array([[-width/2,-h/2],[width/2,-h/2],[width/2,h/2],[-width/2,h/2]],np.float32)
    q += rng.uniform(-1,1,(4,2)).astype(np.float32)*[width*.012,h*.045]
    rot = np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
    q = (q@rot.T+[cx,cy]).astype(np.float32)
    hp = cv2.getPerspectiveTransform(np.array([[0,0],[pw,0],[pw,ph],[0,ph]],np.float32), q)
    img = cv2.warpPerspective(printed, hp, (1000,600), flags=cv2.INTER_LINEAR,borderValue=255)
    img = cv2.GaussianBlur(img, (0,0), blur)
    yy, xx = np.mgrid[:600,:1000]
    img = img*(1+rng.uniform(-.035,.035)*(xx/1000-.5))+rng.normal(0,1.5,img.shape)
    img = np.clip(img,0,255).astype(np.uint8)
    _, jpg = cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,85])
    img = cv2.imdecode(jpg,cv2.IMREAD_GRAYSCALE)
    canonical_to_camera = hp@np.diag([pw/2200,ph/600,1])
    return img, canonical_to_camera


def summarize(rows):
    result = {}
    for row in rows:
        key = f"{row['method']}|{row['condition']}"
        group = result.setdefault(key,dict(n=0,positive_n=0,negative_n=0,correct=0,accepted_correct=0,accepted_wrong=0,false_accept=0,ber=[]))
        group['n'] += 1
        if row['negative']:
            group['negative_n'] += 1
            group['false_accept'] += int(row['accepted'])
        else:
            group['positive_n'] += 1
            group['correct'] += int(row['correct'])
            group['accepted_correct'] += int(row['accepted'] and row['correct'])
            group['accepted_wrong'] += int(row['accepted'] and not row['correct'])
            group['ber'].append(row['ber'])
    for group in result.values():
        b = group.pop('ber')
        group['mean_ber'] = float(np.mean(b)) if b else None
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=6)
    p.add_argument('--seed',type=int,default=41000)
    p.add_argument('--oracle',action='store_true')
    p.add_argument('--methods',default='legacy,reference,arc,arc_nuisance')
    args=p.parse_args()
    allowed={'legacy','reference','arc','arc_crc','arc_nuisance'}
    if args.n < 1 or not set(args.methods.split(',')) <= allowed:
        p.error('n must be positive and methods must name supported codecs')
    cv2.setNumThreads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    design=ReferenceDesign()
    setup={}
    for method in args.methods.split(','):
        d=A.ArrowDesign() if method=='legacy' else design
        if method.startswith('arc'):
            c=ArcCarrier(d,layout_version=4,checksum='mmc' if method=='arc_crc' else 'legacy')
            decoder=NuisanceDecoder(c) if method.endswith('nuisance') else ArcDecoder(c)
        else:
            c=I.InnerCarrier(d,layout_version=4)
            decoder=I.InnerDecoder(c)
        setup[method]=(c,decoder)
    conditions=[(440,.8,1200,False),(330,1.2,1200,False),(220,1.2,1200,False),
                (330,1.8,1200,False),(330,1.2,600,False),(330,1.2,1200,True)]
    rows=[]
    config=dict(seed=args.seed,n=args.n,conditions=conditions,methods=list(setup),oracle=args.oracle,
                physical_mm=[22,6],camera_canvas=[1000,600],noise_sigma=1.5,jpeg_quality=85,
                file_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             [Path(__file__),Path('arrow_arc.py'),Path('arrow_inner.py'),Path('assets/reference/mask4.png')]})
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    handle=(args.out/'rows.csv').open('w',newline='',encoding='utf-8')
    writer=None
    t0=time.monotonic()
    for ci,(width,blur,dpi,defect) in enumerate(conditions):
        condition=f'w{width}_b{blur}_dpi{dpi}'+('_local' if defect else '')
        for j in range(args.n):
            seed=args.seed+ci*1000+j
            digits=f'{int(np.random.default_rng(args.seed+j).integers(0,10**11)):011d}'
            # One unmarked control per payload/condition/method; same channel.
            for negative in (False,True):
                for name,(carrier,decoder) in setup.items():
                    wm=carrier.embed(digits) if not negative else None
                    plate=wm['plate'] if wm else carrier.design.template
                    image,H=capture(plate,width=width,blur=blur,dpi=dpi,seed=seed,local_defect=defect)
                    ts=time.monotonic()
                    if args.oracle:
                        warped=cv2.warpPerspective(image,np.linalg.inv(H),(2200,600),flags=cv2.INTER_LINEAR,borderValue=255)
                        raw=decoder.decode_rectified(dict(warped=warped,logo_scale=width/2200,ecc=dict(cc=1.0)))
                    else:
                        raw=decoder.decode(image)
                    gated=AD.apply_gate(raw,carrier='inner')
                    ber=float(np.mean(np.asarray(raw.get('bits',[])) != np.asarray(wm['bits']))) if wm and len(raw.get('bits') or [])==77 else 1.0
                    row=dict(method=name,condition=condition,seed=seed,digits=digits,negative=negative,
                             correct=(raw.get('digits')==digits and not negative),accepted=gated.accepted,
                             decoded=gated.digits or '',candidate=raw.get('digits') or '',crc=raw.get('crc_ok'),
                             cc=raw.get('cc',0),margin=raw.get('margin',0),margin_min=raw.get('margin_min',0),
                             logo_scale=raw.get('logo_scale',0),ber=ber,seconds=time.monotonic()-ts,reason=gated.reason)
                    rows.append(row)
                    if writer is None:
                        writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); handle.flush()
                    if j==0 and not negative:
                        cv2.imwrite(str(args.out/f'{name}_{condition}.png'),image)
        summary=summarize(rows)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(condition, {k:v for k,v in summary.items() if k.endswith(condition)},f'elapsed={time.monotonic()-t0:.1f}s',flush=True)
    handle.close()


if __name__=='__main__':
    main()
