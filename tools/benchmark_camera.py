"""Paired test of the production input resize and thumbnail-only localization."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,json,time,argparse
import arrow_wm as A
import arrow_inner as I
import arrow_decode as D
from arrow_camera import CameraDecoder
from benchmark_reconstruction import channel,PROFILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=16)
    a=p.parse_args();cv2.setNumThreads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    design=A.ArrowDesign();carrier=I.InnerCarrier(design)
    decoders={'production':D.Decoder(design=design),'camera':CameraDecoder(design=design)}
    rows=[]
    for j,digits in enumerate(A.make_payloads(a.n,88173)):
        for negative in (False,True):
            plate=design.template if negative else carrier.embed(digits)['plate']
            patch=channel(plate,PROFILES['print1200_medium'],87181+j)
            canvas=np.full((1800,2400),255,np.uint8)
            h,w=patch.shape;canvas[810:810+h,900:900+w]=patch
            if j==0:cv2.imwrite(str(out/('unmarked.png' if negative else 'marked.png')),canvas)
            for name,decoder in decoders.items():
                start=time.perf_counter();r=decoder.decode(canvas)
                rows.append(dict(method=name,negative=negative,truth=None if negative else digits,
                     digits=r.digits,accepted=r.accepted,reason=r.reason,
                     correct=r.accepted and r.digits==digits and not negative,
                     wrong=r.accepted and (negative or r.digits!=digits),
                     seconds=time.perf_counter()-start))
            (out/'rows.json').write_text(json.dumps(rows,indent=2))
        print(j+1,'/',a.n,flush=True)

if __name__=='__main__':main()
