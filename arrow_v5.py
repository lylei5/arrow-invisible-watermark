"""V5 inner-contour watermark.

77 cells, 37 payload bits, proper CRC-7/MMC, Hamming(7,4), proportional
cross-hole interleaving and optional equal-arclength research windows.
"""
import cv2
import numpy as np
import arrow_wm as A
import arrow_inner as I
import arrow_decode as D
from arrow_vector import VectorDesign

def crc7(bits):
    reg=0
    for b in bits:
        feedback=((reg>>6)&1) ^ int(b)
        reg=(reg<<1)&127
        if feedback:
            reg ^= 0x09
    return [(reg>>i)&1 for i in range(6,-1,-1)]

def encode(digits):
    p=A.payload_to_bits(digits)
    return A.hamming74_encode(p+crc7(p))

def decode_soft(soft):
    soft=np.asarray(soft,float)
    if soft.shape!=(77,) or not np.isfinite(soft).all():
        return None,False,0
    cb=A.hamming_codebook(); signs=cb*2.-1
    data=[]; changes=0
    for start in range(0,77,7):
        code=cb[np.argmax(signs@soft[start:start+7])]
        data.extend(int(code[j]) for j in (2,4,5,6))
        changes+=int(np.sum(code != (soft[start:start+7]>0)))
    digits=A.bits_to_payload(data[:37])
    return digits,bool(digits is not None and crc7(data[:37])==data[37:]),changes

class Carrier(I.InnerCarrier):
    def __init__(self,design=None,depth=8.,equal_arc=False,interleave=True):
        self.equal_arc=equal_arc
        self.interleave=interleave
        super().__init__(design or VectorDesign(),depth=depth)
        if not equal_arc:
            return
        for h in self.holes:
            xy=np.stack([h['r_tab']*self.cos_t,h['r_tab']*self.sin_t],axis=1)
            # Smooth raster-derived radii only for the arclength clock. Geometry
            # remains unchanged; tiny staircase segments must not alter pitch.
            from scipy.ndimage import gaussian_filter1d
            xy=gaussian_filter1d(xy,2,axis=0,mode='wrap')
            ds=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
            h['arc_fraction']=np.r_[0,np.cumsum(ds[:-1])]/ds.sum()

    def _build_pair_order(self):
        if not self.interleave:
            return tuple(range(77))
        counts=[h['n_cells'] for h in self.holes]
        offsets=np.r_[0,np.cumsum(counts)[:-1]]
        used=[0]*len(counts); order=[]
        for _ in range(77):
            hi=min((j for j in range(len(counts)) if used[j]<counts[j]),
                   key=lambda j:((used[j]+1)/counts[j],j))
            order.append(int(offsets[hi]+used[hi])); used[hi]+=1
        return tuple(order)

    def _window(self,hi,k):
        if not self.equal_arc:
            return super()._window(hi,k)
        h=self.holes[hi]
        t=(h['arc_fraction']*h['n_cells']-k)%h['n_cells']
        return np.where(t<1,np.sin(np.pi*np.clip(t,0,1)),0.)

    def embed(self,digits):
        bits=encode(digits); arranged=np.zeros(77,dtype=int)
        arranged[list(self.pair_order)]=bits
        mask=self.design.mask4.copy()
        for hi,(h,delta) in enumerate(zip(self.holes,self._delta(arranged))):
            cv2.fillPoly(mask,[h['poly_orig']],255)
            r=np.maximum(h['r_tab']+delta*self.SS,4.)
            pts=np.stack([h['c'][0]+r*self.cos_t,h['c'][1]+r*self.sin_t],axis=1)
            cv2.fillPoly(mask,[np.rint(pts).astype(np.int32)],0)
        return {'digits':digits,'bits':bits,'plate':self.design.render(mask),
                'mask4':mask,'layout_version':5}

class Decoder(I.InnerDecoder):
    def decode_rectified(self,r,return_debug=False):
        raw=super().decode_rectified(r,return_debug)
        digits,ok,changes=decode_soft(raw['conf'])
        raw.update(digits=digits,payload_valid=digits is not None,crc_ok=ok,
                   corrected=changes,n_corrected=changes,layout_version=5)
        return raw

def gate(raw):
    return D.apply_gate(raw)

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    from PIL import Image
    p=argparse.ArgumentParser()
    p.add_argument('--digits',default='13800138000'); p.add_argument('--out',required=True)
    p.add_argument('--depth',type=float,default=8.)
    p.add_argument('--equal-arc',action='store_true',help='Use the arclength ablation')
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    c=Carrier(depth=args.depth,equal_arc=args.equal_arc); result=c.embed(args.digits)
    Image.fromarray(result['plate']).save(out/f'ARROW_v5_{args.digits}.png',dpi=(2540,2540))
    r=gate(Decoder(c).decode(result['plate']))
    print(json.dumps({'accepted':r.accepted,'digits':r.digits,'reason':r.reason}))
