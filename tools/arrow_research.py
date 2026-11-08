"""Explicit master/format selection for reproducible experimental encode/decode."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
import cv2
from PIL import Image
import arrow_wm as A
import arrow_inner as I
import arrow_v5 as V
from arrow_vector import VectorDesign,export
from arrow_camera import CameraDecoder,install_thumbnail_locator

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['master','embed','decode'])
    p.add_argument('--method',choices=['v4-legacy','v4-vector','v5'],default='v5')
    p.add_argument('--depth',type=float,default=8.,help='Radial displacement in 0.01 mm units')
    p.add_argument('--digits');p.add_argument('--image');p.add_argument('--out')
    args=p.parse_args();cv2.setNumThreads(1)
    if args.action=='master':
        if not args.out:p.error('master requires --out')
        export(args.out);return
    design=A.ArrowDesign() if args.method=='v4-legacy' else VectorDesign()
    carrier=V.Carrier(design,depth=args.depth) if args.method=='v5' else I.InnerCarrier(design,depth=args.depth)
    if args.action=='embed':
        if not args.digits or not args.out:p.error('embed requires --digits and --out')
        result=carrier.embed(args.digits);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(result['plate']).save(out,dpi=(2540,2540))
        out.with_suffix('.json').write_text(json.dumps(dict(method=args.method,depth=args.depth,
             digits=args.digits,canvas_mm=[22,6],master='repository bitmap' if args.method=='v4-legacy' else 'vector reconstruction'),indent=2))
        print(json.dumps({'image':str(out),'method':args.method}));return
    if not args.image:p.error('decode requires --image')
    if args.method=='v5':
        img=A.imread_u(args.image)
        if img is None:p.error('cannot read image')
        decoder=V.Decoder(carrier);install_thumbnail_locator(decoder._reg)
        result=V.gate(decoder.decode(img))
    else:
        result=CameraDecoder(design=design,depth=args.depth).decode(args.image)
    print(json.dumps(dict(accepted=result.accepted,digits=result.digits,reason=result.reason,
         margin=result.margin,margin_min=result.margin_min,cc=result.cc,method=args.method)))

if __name__=='__main__':main()
