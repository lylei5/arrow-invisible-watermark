"""Export the v5 master, embed a number, or decode a photograph."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import cv2
from PIL import Image
import arrow_wm as A
from arrow_vector import export
from arrow_service import Decoder, FORMAT, MASTER_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['master', 'embed', 'decode'])
    parser.add_argument('--digits')
    parser.add_argument('--image')
    parser.add_argument('--out')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.action == 'master':
        if not args.out:
            parser.error('master requires --out')
        export(args.out)
        return
    decoder = Decoder()
    if args.action == 'embed':
        if not args.digits or not args.out:
            parser.error('embed requires --digits and --out')
        result = decoder.carrier.embed(args.digits)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result['plate']).save(out, dpi=(2540, 2540))
        out.with_suffix('.json').write_text(json.dumps(dict(
            format=FORMAT, master_id=MASTER_ID, digits=args.digits,
            canvas_mm=[22, 6], depth_mm=.08), indent=2), encoding='utf-8')
        print(json.dumps({'image': str(out), 'format': FORMAT}))
        return
    if not args.image:
        parser.error('decode requires --image')
    image = A.imread_u(args.image)
    if image is None:
        parser.error('cannot read image')
    result = decoder.decode(image)
    response = result.public_dict()
    response.pop('candidate_digits', None)
    response.update(format=FORMAT, master_id=MASTER_ID)
    print(json.dumps(response))


if __name__ == '__main__':
    main()
