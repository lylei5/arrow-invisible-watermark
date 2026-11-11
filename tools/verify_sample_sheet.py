"""Decode sample regions from the actual PDF's 600 dpi Poppler rendering."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from arrow_service import Decoder

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--rendered', required=True)
a = p.parse_args()
out = Path(a.out)
metadata = json.loads((out/'validation.json').read_text(encoding='utf-8'))
image = cv2.imread(a.rendered, cv2.IMREAD_GRAYSCALE)
assert image is not None
cv2.setNumThreads(1)
decoder = Decoder()
rows = []
scale = 600/25.4
for item in metadata['samples']:
    x, y = item['x_mm'], item['y_mm']
    left, right = round(x*scale), round((x+22)*scale)
    top, bottom = round((297-y-6)*scale), round((297-y)*scale)
    crop = image[top:bottom, left:right]
    result = decoder.decode(crop)
    correct = bool(result.accepted and result.digits == item['digits'])
    rows.append(dict(index=item['index'], expected=item['digits'],
                     decoded=result.digits, accepted=result.accepted,
                     correct=correct, reason=result.reason,
                     width_px=crop.shape[1], height_px=crop.shape[0]))
    if len(rows) % 12 == 0:
        print(f'PDF rendered samples checked: {len(rows)}/72', flush=True)
metadata['pdf_render_check'] = dict(dpi=600, total=len(rows),
                                  correct=sum(r['correct'] for r in rows), rows=rows)
(out/'validation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
assert all(r['correct'] for r in rows), [r for r in rows if not r['correct']]
print('PASS: all 72 PDF samples decode correctly after 600 dpi rendering.')
