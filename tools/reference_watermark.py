"""Export or decode the reference-arc-v1 research format.

python tools/reference_watermark.py export --out print-assets
python tools/reference_watermark.py decode photograph.jpg
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image

import arrow_decode as AD
import arrow_inner as I
from arrow_arc import ArcCarrier, ArcDecoder
from arrow_reference import ReferenceDesign


def vector_mask(mask4, path):
    contours, _ = cv2.findContours(mask4, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        pts = cv2.approxPolyDP(contour, .5, True).reshape(-1, 2)/4
        paths.append('M'+' L'.join(f'{x:.3f},{y:.3f}' for x,y in pts)+' Z')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="22mm" height="6mm" '
                    'viewBox="0 0 2200 600"><rect width="2200" height="600" fill="white"/>'
                    '<path fill="#3a3a3a" fill-rule="evenodd" d="'+' '.join(paths)+'"/></svg>\n',encoding='utf-8')


def make_codec(depth, format_id):
    design=ReferenceDesign()
    if format_id=='reference-arc-v1':
        carrier=ArcCarrier(design,depth=depth,layout_version=4)
        return carrier,ArcDecoder(carrier)
    carrier=I.InnerCarrier(design,depth=depth,layout_version=4)
    return carrier,I.InnerDecoder(carrier)


def export(out, payloads, depth=8, format_id='reference-arc-v1'):
    out.mkdir(parents=True, exist_ok=True)
    carrier,decoder=make_codec(depth,format_id)
    design=carrier.design
    records = []
    Image.fromarray(design.template).save(out/'ARROW_control.png', dpi=(2540,2540))
    vector_mask(design.mask4,out/'ARROW_control.svg')
    for digits in payloads:
        wm=carrier.embed(digits)
        raw=decoder.decode(wm['plate'])
        result=AD.apply_gate(raw,carrier='inner')
        if not result.accepted or result.digits != digits:
            raise RuntimeError(f'Plate self-check failed for {digits}: {result.public_dict()}')
        name=f'ARROW_{"arc" if format_id=="reference-arc-v1" else "refv4"}_{digits}'
        Image.fromarray(wm['plate']).save(out/(name+'.png'),dpi=(2540,2540))
        mono=Image.fromarray(np.where(wm['plate']>127,255,0).astype(np.uint8)).convert('1',dither=Image.Dither.NONE)
        mono.save(out/(name+'_1bit.png'),dpi=(2540,2540))
        vector_mask(wm['mask4'],out/(name+'.svg'))
        records.append(dict(digits=digits,accepted=result.accepted,margin=result.margin,
                            margin_min=result.margin_min,cc=result.cc,
                            png_sha256=hashlib.sha256((out/(name+'.png')).read_bytes()).hexdigest()))
    info=dict(format=format_id,checksum='CRC-7/MMC' if format_id=='reference-arc-v1' else 'legacy-v4',payload_bits=37,code_bits=77,
              physical_canvas_mm=[22,6],ink_width_mm=20,depth_px=depth,depth_mm=depth*.01,
              image_px=[2200,600],dpi=2540,template_sha256=hashlib.sha256(design.reference_path.read_bytes()).hexdigest(),
              plates=records,decoder=f'python tools/reference_watermark.py decode PHOTO --format {format_id}')
    (out/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--out',type=Path,required=True)
    e.add_argument('--payloads',nargs='+',default=['13800138000','13912345678','18600001234'])
    e.add_argument('--depth',type=float,default=8)
    e.add_argument('--format',choices=['reference-arc-v1','reference-v4'],default='reference-arc-v1')
    d=sub.add_parser('decode');d.add_argument('image',type=Path);d.add_argument('--depth',type=float,default=8)
    d.add_argument('--format',choices=['reference-arc-v1','reference-v4'],default='reference-arc-v1')
    args=p.parse_args();cv2.setNumThreads(2)
    if args.command=='export':
        print(json.dumps(export(args.out,args.payloads,args.depth,args.format),indent=2));return
    c,decoder=make_codec(args.depth,args.format)
    img=cv2.imdecode(np.frombuffer(args.image.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError('Unreadable image')
    result=AD.apply_gate(decoder.decode(img),carrier='inner')
    public=result.public_dict();public.pop('candidate_digits',None);public['format']=args.format
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
