"""Locate on a thumbnail and lift the homography to original camera pixels."""
import cv2
import numpy as np
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
