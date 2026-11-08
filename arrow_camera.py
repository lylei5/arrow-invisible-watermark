"""Opt-in two-resolution camera decoder for existing v4 plates.

Locate on a thumbnail, lift the homography, and measure the original pixels.
The existing production Decoder and HTTP entry point are not replaced.
"""
import cv2
import numpy as np
import arrow_decode as D

def install_thumbnail_locator(registration,max_side=900):
    if max_side<128:
        raise ValueError('max_side must be at least 128')
    find=registration._find_transforms
    def locate(gray):
        h,w=gray.shape[:2]
        if max(h,w)<=max_side:
            return find(gray)
        factor=max_side/max(h,w)
        sw,sh=max(8,round(w*factor)),max(8,round(h*factor))
        small=cv2.resize(gray,(sw,sh),interpolation=cv2.INTER_AREA)
        # Exact resize dimensions, not the rounded ideal scale.
        sx,sy=sw/w,sh/h
        scale=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]],np.float32)
        return [H@scale for H in find(small)]
    registration._find_transforms=locate

class CameraDecoder(D.Decoder):
    def __init__(self,*,locate_max_side=900,**kwargs):
        super().__init__(carrier='inner',**kwargs)
        install_thumbnail_locator(self._impl._reg,locate_max_side)

    def decode(self,image,*,return_debug=False):
        img=D._load_image(image)
        if img is None:
            return D._unreadable(self.carrier,self.margin_threshold,
                                self.min_group_margin,self.cc_threshold,'unreadable')
        if img.ndim==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY if img.shape[2]==4 else cv2.COLOR_BGR2GRAY)
        raw=self._impl.decode(img,return_debug=return_debug)
        return D.apply_gate(raw,carrier=self.carrier,margin_threshold=self.margin_threshold,
                            min_group_margin=self.min_group_margin,
                            cc_threshold=self.cc_threshold,
                            allow_legacy_v2=self.allow_legacy_v2)
