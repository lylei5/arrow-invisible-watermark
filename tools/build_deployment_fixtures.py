"""Reproducible production-path regression fixtures, including full phone frames."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
import arrow_wm as A
from arrow_service import Decoder
from benchmark_reconstruction import channel,PROFILES

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(1)
decoder=Decoder();rows=[]
profiles=['print1200_medium','print1200_small','defocus_286','defocus_330','tiny_220','local_O_severe','print300']
payloads=['01088886666','00000000000']+A.make_payloads(10,590871)
for j,digits in enumerate(payloads):
    plate=decoder.carrier.embed(digits)['plate']
    for pi,name in enumerate(profiles):
        img=channel(plate,PROFILES[name],590871+100000+j*101+pi)
        path=f'{j:02}_{name}.png';cv2.imwrite(str(out/path),img)
        rows.append(dict(file=path,truth=digits,profile=name))
    # Actual small mark in a high-resolution photograph, both landscape and portrait.
    for name,shape in [('portrait',(2400,1800)),('landscape',(1800,2400))]:
        img=np.full(shape,245,np.uint8)
        scene=channel(plate,PROFILES['print1200_medium'],98231+j)
        h,w=scene.shape;y=(shape[0]-h)//2;x=(shape[1]-w)//2
        img[y:y+h,x:x+w]=scene
        path=f'{j:02}_{name}.png';cv2.imwrite(str(out/path),img)
        rows.append(dict(file=path,truth=digits,profile=name))
for j in range(12):
    img=channel(decoder.carrier.design.template,PROFILES[profiles[j%len(profiles)]],71121+j)
    path=f'negative_{j:02}.png';cv2.imwrite(str(out/path),img)
    rows.append(dict(file=path,truth=None,profile='unmarked'))
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
print(json.dumps({'fixtures':len(rows)}))
