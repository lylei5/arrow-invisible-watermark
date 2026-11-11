"""Measure the production decoder against a fixture manifest on the deployment CPU."""
import sys,json,time,cProfile,pstats,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from arrow_service import Decoder
cv2.setNumThreads(1)
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
decoder=Decoder();rows=[];profiler=cProfile.Profile()
for i,sample in enumerate(json.loads((root/'manifest.json').read_text())):
    image=cv2.imread(str(root/sample['file']),0)
    start=time.perf_counter();profiler.enable();r=decoder.decode(image);profiler.disable()
    rows.append(dict(**sample,accepted=r.accepted,digits=r.digits,reason=r.reason,
                     correct=bool(r.accepted and r.digits==sample['truth']),
                     wrong=bool(r.accepted and r.digits!=sample['truth']),ms=1000*(time.perf_counter()-start)))
    if (i+1)%12==0:print(i+1,flush=True)
(out/'rows.json').write_text(json.dumps(rows,indent=2))
groups={}
for name in sorted(set(r['profile'] for r in rows)):
    rr=[r for r in rows if r['profile']==name]
    groups[name]=dict(n=len(rr),correct=sum(r['correct'] for r in rr),wrong=sum(r['wrong'] for r in rr),
                     accepted=sum(r['accepted'] for r in rr),p50_ms=float(np.median([r['ms'] for r in rr])),
                     p95_ms=float(np.percentile([r['ms'] for r in rr],95)))
(out/'summary.json').write_text(json.dumps(groups,indent=2))
s=io.StringIO();pstats.Stats(profiler,stream=s).sort_stats('cumtime').print_stats(35)
(out/'profile.txt').write_text(s.getvalue())
print(json.dumps(groups))
