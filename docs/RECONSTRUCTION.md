# ARROW vector master and camera experiments

The research entry point is `tools/arrow_research.py`. Its default format is v5:
37 payload bits, CRC-7/MMC, Hamming(7,4), 77 equal-angle radial windows and
proportional cross-hole interleaving. Each Hamming group spans at least three
holes. Default radial amplitude is 0.08 mm. The master has a 22 x 6 mm canvas
and a centered 22 x 4.4 mm wordmark. Cubic Bezier curves are manually
reconstructed from the supplied construction sheet, with gaps 0.04X, 0.03X,
0.03X and 0.01X. This is a reconstruction, not official manufacturer artwork.

## Usage

Install `requirements-research.txt`, then run commands from the repository root:

```sh
python tools/arrow_research.py master --out generated/master
python tools/arrow_research.py embed --method v5 --digits 13800138000 --out generated/plate.png
python tools/arrow_research.py decode --method v5 --image generated/plate.png
python tools/arrow_research.py decode --method v4-legacy --image samples/inner/ARROW_13800138000.png
python tools/export_reconstruction.py --out generated/review
python -m unittest discover -s tests -v
```

`--depth 6` selects 0.06 mm displacement; use the same depth for encode and
decode. Always consume only `accepted` and `digits`. A rejected result returns
`digits: null`. Crop a single logo and retain its aspect ratio when photographing
the test sheet. The original HTTP service continues to use its existing master
and v4 format. The experimental master and v5 format require explicit selection;
the current production service cannot decode the new print sheet reliably.

## Resolution-preserving decoding

`arrow_camera.CameraDecoder` supports existing v4 plates. It locates on a
900-pixel thumbnail, lifts the homography with exact resize pixel-center
coordinates, then measures the original pixels. The CLI applies the same
locator to v5. No artificial square-to-landscape aspect conversion is applied.
This is an opt-in implementation, not an update to the running HTTP service.

## Reproducing the comparison

All methods use paired payloads, identical channel seeds and fixed acceptance
gates: mean margin 1.8, minimum group margin 0.20, registration CC 0.85 and
logo scale 0.08. Print raster sampling uses a 22 x 6 mm physical canvas.
The camera width is the width of that canvas, including the original master's
margins when evaluating the repository master. Geometric decoding does not
receive the simulator's ground-truth homography.

```sh
python tools/benchmark_reconstruction.py --out results/standard --n 32 --seed 921047 --profiles clean,print1200_close,print1200_medium,print1200_small,print600_medium,local_O_blur,ink_spread
python tools/benchmark_reconstruction.py --out results/selection --n 24 --seed 725103 --profiles defocus_286,defocus_330,tiny_220,local_O_severe,print300
python tools/benchmark_reconstruction.py --out results/validation --n 40 --seed 590871 --methods legacy,vector_v4,v5_angle --profiles defocus_286,defocus_330,tiny_220,local_O_severe,print300
python tools/benchmark_reconstruction.py --out results/negative --n 40 --seed 298415 --negative --methods v5_angle --profiles print1200_close,print1200_small,defocus_286,tiny_220,print300
python tools/benchmark_reconstruction.py --out results/depth6 --n 24 --seed 725103 --depth 6 --methods v5_angle,v5_arc --profiles defocus_286,defocus_330,tiny_220,local_O_severe,print300
python tools/benchmark_camera.py --out results/camera --n 32
```

`legacy` means the repository bitmap master and current v4 inner decoder;
`vector_v4` holds the master fixed when comparing coding/layout changes;
`v5_angle` is the candidate; `v5_arc` is the equal-arclength ablation.
The main watermark comparison calls the core decoder at native camera
resolution. The separate camera benchmark compares the production wrapper's
resize against thumbnail-only localization on 2400 x 1800 images.

## Interpretation and limits

The legacy checksum and CRC-7/MMC both have rank 7 over the full 37-bit payload
space. They are incompatible checksum rules, not different checksum capacities.
CRC-7/MMC passes the standard ASCII `123456789` check value 0x75; the legacy
function yields 0x25. Neither CRC is cryptographic authentication. The benchmark
changes checksum and interleaver together, so it does not isolate their effects.

Gray-coverage print sampling, Gaussian blur, JPEG, noise, mild perspective,
local O blur and an ink-spread proxy are simulated. Actual printer halftoning,
glaze, firing, nonplanar surfaces and physical human visibility require printed
samples. The delivered PDF uses exact 22 x 6 mm image rectangles; print at 100%.
The PNG's 2540 dpi is a digital master resolution, not a requirement or claim
about printer hardware. Radial displacement is not the same as contour-normal
displacement at sharp corners. Full-image PSNR underweights the changed inner
contour; use the detail comparison and actual-size print in visual assessment.
