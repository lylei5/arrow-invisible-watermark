"""Paired print/camera simulation in physical units; no ground-truth rectification.

Independent payload and channel seeds, fixed gates, per-condition accounting.
This is a synthetic benchmark, not evidence of real ceramic print performance.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,time
import cv2
import numpy as np
from PIL import Image
import arrow_wm as A
import arrow_inner as I
import arrow_decode as D
import arrow_v5 as V
from arrow_vector import VectorDesign

PROFILES={
 'clean':dict(dpi=2540,width=1100,blur=0.,noise=0.,jpeg=100,tilt=0.),
 'print1200_close':dict(dpi=1200,width=660,blur=.55,noise=1.5,jpeg=92,tilt=.025),
 'print1200_medium':dict(dpi=1200,width=440,blur=.75,noise=2.,jpeg=85,tilt=.04),
 'print1200_small':dict(dpi=1200,width=330,blur=1.1,noise=2.5,jpeg=80,tilt=.05),
 'print600_medium':dict(dpi=600,width=440,blur=.8,noise=2.,jpeg=85,tilt=.04),
 'local_O_blur':dict(dpi=1200,width=660,blur=.55,noise=1.5,jpeg=90,tilt=.03,local=2.5),
 'ink_spread':dict(dpi=1200,width=550,blur=.65,noise=2.,jpeg=88,tilt=.03,ink=.025),
 'defocus_286':dict(dpi=1200,width=286,blur=1.6,noise=3.,jpeg=80,tilt=.06),
 'defocus_330':dict(dpi=1200,width=330,blur=2.,noise=3.,jpeg=80,tilt=.06),
 'tiny_220':dict(dpi=1200,width=220,blur=1.6,noise=3.,jpeg=80,tilt=.06),
 'local_O_severe':dict(dpi=1200,width=440,blur=.8,noise=2.,jpeg=85,tilt=.04,local=4.),
 'print300':dict(dpi=300,width=440,blur=.8,noise=2.,jpeg=85,tilt=.04),
}

def channel(plate,p,seed):
    rng=np.random.default_rng(seed)
    # Print-grid sampling for a 22 x 6 mm canvas. Gray antialiasing is an ideal
    # coverage model; real halftone, glaze and printer MTF need physical tests.
    pw,ph=round(22*p['dpi']/25.4),round(6*p['dpi']/25.4)
    ink=p.get('ink',0)
    if ink:
        radius=max(1,round(ink*100))
        plate=cv2.erode(plate,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    printed=cv2.resize(plate,(pw,ph),interpolation=cv2.INTER_AREA)
    cw=p['width']; ch=cw*6/22
    width=round(cw*1.32); height=round(ch+cw*.30)
    x=(width-cw)/2; y=(height-ch)/2
    src=np.float32([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]])
    dst=np.float32([[x,y],[x+cw,y],[x+cw,y+ch],[x,y+ch]])
    jitter=rng.uniform(-1,1,(4,2))*np.array([cw,ch])*p['tilt']
    dst+=jitter.astype(np.float32)
    H=cv2.getPerspectiveTransform(src,dst)
    img=cv2.warpPerspective(printed,H,(width,height),borderValue=255).astype(float)
    if p['blur']:
        img=cv2.GaussianBlur(img,(0,0),p['blur'])
    if p.get('local'):
        blurred=cv2.GaussianBlur(img,(0,0),p['local'])
        yy,xx=np.mgrid[:height,:width]
        # O region at roughly 66% of the word width, with smooth transitions.
        alpha=np.exp(-((xx-(x+.66*cw))/(.095*cw))**8)
        img=img*(1-alpha)+blurred*alpha
    if p['noise']:
        gain=rng.uniform(.88,1.04); offset=rng.uniform(-5,5)
        img=img*gain+offset+rng.normal(0,p['noise'],img.shape)
    img=np.uint8(np.clip(img,0,255))
    if p['jpeg']<100:
        ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,p['jpeg']])
        assert ok; img=cv2.imdecode(buf,0)
    return img

def summarize(rows):
    groups={}
    for r in rows:
        key=r['method']+'/'+r['profile']
        g=groups.setdefault(key,dict(n=0,accepted_correct=0,accepted_wrong=0,rejected=0,
                                     candidate_correct=0,crc_pass=0,bit_errors=0,bit_total=0,seconds=0.))
        g['n']+=1
        for k in ('accepted_correct','accepted_wrong','rejected','candidate_correct','crc_pass','bit_errors','bit_total','seconds'):
            g[k]+=r[k]
    for g in groups.values():
        g['exact_accept_rate']=g['accepted_correct']/g['n']
        g['ber']=g['bit_errors']/g['bit_total'] if g['bit_total'] else None
    return groups

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=12)
    p.add_argument('--seed',type=int,default=82341)
    p.add_argument('--methods',default='legacy,vector_v4,v5_angle,v5_arc')
    p.add_argument('--profiles',default=','.join(PROFILES))
    p.add_argument('--depth',type=float,default=8.)
    p.add_argument('--negative',action='store_true')
    args=p.parse_args();cv2.setNumThreads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    methods={};vector=VectorDesign()
    for name in args.methods.split(','):
        if name=='legacy': c=I.InnerCarrier(A.ArrowDesign(),depth=args.depth)
        elif name=='vector_v4':c=I.InnerCarrier(vector,depth=args.depth)
        else:c=V.Carrier(vector,depth=args.depth,equal_arc=name=='v5_arc')
        methods[name]=(c,V.Decoder(c) if name.startswith('v5') else I.InnerDecoder(c))
    payloads=A.make_payloads(args.n,args.seed);rows=[]
    (out/'manifest.json').write_text(json.dumps(dict(seed=args.seed,n=args.n,
      depth_px=args.depth,depth_mm=args.depth*.01,profiles={k:PROFILES[k] for k in args.profiles.split(',')},
      payloads=payloads,methods=list(methods),negative=args.negative,
      gate=dict(mean=1.8,minimum=.2,cc=.85,scale=.08),
      resolution='22x6 mm; print-grid then camera warp; original camera pixels decoded',
      limitation='Synthetic gray-coverage printing; no physical capture or true halftoning'),indent=2))
    for name,(carrier,decoder) in methods.items():
        for j,payload in enumerate(payloads):
            encoded=carrier.embed(payload)
            plate=carrier.design.template if args.negative else encoded['plate']
            if j==0: Image.fromarray(plate).save(out/f'{name}_plate.png',dpi=(2540,2540))
            for pi,profile in enumerate(args.profiles.split(',')):
                img=channel(plate,PROFILES[profile],args.seed+100000+j*101+pi)
                if j==0:Image.fromarray(img).save(out/f'{name}_{profile}.png')
                start=time.perf_counter();raw=decoder.decode(img)
                result=V.gate(raw);elapsed=time.perf_counter()-start
                correct=raw.get('digits')==payload and not args.negative
                bits=raw.get('bits');bit_total=77 if bits is not None and not args.negative else 0
                errors=int(np.sum(np.asarray(bits)!=encoded['bits'])) if bit_total else 0
                rows.append(dict(method=name,profile=profile,payload=None if args.negative else payload,
                    candidate=raw.get('digits'),accepted_correct=int(result.accepted and correct),
                    accepted_wrong=int(result.accepted and not correct),rejected=int(not result.accepted),
                    candidate_correct=int(correct),crc_pass=int(bool(raw.get('crc_ok'))),
                    bit_errors=errors,bit_total=bit_total,seconds=elapsed,reason=result.reason,
                    margin=result.margin,margin_min=result.margin_min,cc=result.cc))
            print(name,j+1,'/',args.n,flush=True)
            (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
            with (out/'rows.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
