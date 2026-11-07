"""Extract the supplied design screenshot with deterministic color segmentation.

This reconstructs a raster silhouette, not the unavailable original vector file.
Run: python tools/extract_reference.py SOURCE.jpg --out assets/reference
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract(source: Path, out: Path) -> dict:
    raw = source.read_bytes()
    rgb = np.array(Image.open(source).convert("RGB"))
    if rgb.shape[:2] != (570, 1110):
        raise ValueError("This measured ROI applies only to the supplied 1110 x 570 screenshot")
    # Boundaries follow the design's magenta guides. Max-channel filtering
    # removes grid lines and retains dark ink even where a guide crosses it.
    x0, y0, x1, y1 = 223, 232, 931, 374
    crop = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = ((gray < 125) & (crop.max(axis=2) < 130)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 100]
    mask = np.isin(labels, keep).astype(np.uint8) * 255
    if len(keep) != 5:
        raise ValueError(f"Expected five letter components, found {len(keep)}")
    # Smooth less than one source pixel before upscaling. This suppresses
    # JPEG stair steps while retaining the observed R leg and W vertices.
    scale = 4
    smooth = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    hi = cv2.resize(smooth, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    hi = (hi >= 127.5).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(hi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    holes = sum(int(h[3] >= 0) for h in hierarchy[0])
    if holes != 4:
        raise ValueError(f"Expected four enclosed counters, found {holes}")
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out / "source_mask.png")
    # 20 mm ink width inside a 22 x 6 mm plate, matching the repository's
    # layout convention. The screenshot's approximately 5:1 aspect is kept.
    tw = 8000
    th = round(tw * hi.shape[0] / hi.shape[1])
    canvas = np.zeros((2400, 8800), np.uint8)
    top = (2400 - th) // 2
    big = cv2.resize(hi, (tw, th), interpolation=cv2.INTER_CUBIC)
    canvas[top:top + th, 400:8400] = (big >= 128).astype(np.uint8) * 255
    Image.fromarray(canvas).save(out / "mask4.png")
    coverage = cv2.resize(canvas, (2200, 600), interpolation=cv2.INTER_AREA) / 255.0
    plate = np.rint(255 - coverage * (255 - 58)).astype(np.uint8)
    Image.fromarray(plate).save(out / "ARROW_reference.png", dpi=(2540, 2540))
    paths = []
    for contour in contours:
        points = cv2.approxPolyDP(contour, 0.65, True).reshape(-1, 2) / scale
        points[:, 0] = 100 + points[:, 0] * 2000 / (x1 - x0)
        points[:, 1] = top / 4 + points[:, 1] * 2000 / (x1 - x0)
        paths.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in points) + " Z")
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="22mm" height="6mm" '
           'viewBox="0 0 2200 600"><rect width="2200" height="600" fill="white"/>'
           '<path fill="#3a3a3a" fill-rule="evenodd" d="' + " ".join(paths) + '"/></svg>\n')
    (out / "ARROW_reference.svg").write_text(svg, encoding="utf-8")
    overlay = crop.copy()
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    overlay[edge] = [0, 180, 120]
    Image.fromarray(cv2.resize(overlay, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)).save(out / "source_overlay.png")
    info = dict(source_sha256=hashlib.sha256(raw).hexdigest(), source_size=[1110, 570],
                roi_xyxy=[x0, y0, x1, y1], source_ink_size_px=[x1-x0, y1-y0],
                letters=5, counters=holes, canvas_mm=[22, 6], ink_width_mm=20,
                ink_height_mm=20*(y1-y0)/(x1-x0), dpi=2540,
                uncertainty="Raster reconstruction; guide intersections and JPEG edges have approximately 1 source pixel uncertainty.",
                extraction=dict(gray_max_exclusive=125, channel_max_exclusive=130, gaussian_sigma_source_px=0.8))
    (out / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, default=Path("assets/reference"))
    args = p.parse_args()
    print(json.dumps(extract(args.source, args.out), indent=2))
