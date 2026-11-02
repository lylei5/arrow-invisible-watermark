"""端到端压测：模拟小程序真实上传链路，再走生产解码门。

为什么需要这个压测
-----------------
`bench_realscale.py` 只放大/缩小印版，直接把「干净的灰度图」交给解码器。
线上小程序不是这样：照片先被**裁到 Logo ROI**，再被压到 callContainer 的
100 KiB 上限以内（`pages/capture/index.js` 的 `exportRectJpeg` 阶梯），
最后服务端 `arrow_decode` 才拿到这张 JPEG。

这条链路上有两处会吃掉亚像素调制：
  1. ROI 导出 JPEG 质量阶梯从 q0.70 起，**只要体积没过 96 KiB 就立刻返回**，
     于是绝大多数裁块都停在 q0.70，剩下 3–4 倍的字节预算完全没用；
  2. 定位失败退回整幅时，`shrinkForCallContainer` 在 [8,70] 里二分质量，
     一张 4000 px 手机照被压到 q≈10，几何调制基本归零。

本脚本把这条链路写成可复现的模拟器（裁块 + 质量阶梯 + 体积上限），
用它衡量「客户端压缩」和「解码器采样」各自贡献了多少失败。

用法
----
    python bench_client_pipeline.py                  # 当前客户端逻辑
    python bench_client_pipeline.py --client maxq    # 修好质量阶梯之后
    python bench_client_pipeline.py --client lossless --scales 0.13 0.20
    python bench_client_pipeline.py --full-frame     # 定位失败、整幅上传
"""
from __future__ import annotations

import argparse
import io
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A

DEFAULT_PAYLOADS = ("13912345678", "13800138000", "18600001234",
                    "17701234567", "01088886666", "13500001111")

# 与 pages/capture/index.js 对齐的客户端常量
ROI_JPEG_LIMIT_BYTES = 96 * 1024
CALLCONTAINER_MAX_BYTES = 100 * 1024
ROI_LONG_SIDE_MAX = 2000
ROI_EXPAND_RATIO = 0.25
# exportRectJpeg 的阶梯（质量是 (0,1] 的小数）
ROI_QUALITIES = (0.70, 0.64, 0.58, 0.52, 0.46, 0.40, 0.32, 0.25)
# shrinkForCallContainer 的宽度阶梯（compressedWidth）
FULLFRAME_WIDTHS = (None, 1800, 1600, 1400, 1280, 1100, 1024, 900, 800, 720, 640)


# ---------------------------------------------------------------------------
# 相机域：印版 -> 照片（尺度、倾斜、失焦、噪声、机内 JPEG）
# ---------------------------------------------------------------------------
def render_photo(plate: np.ndarray, scale: float, *,
                 rng: np.random.Generator,
                 blur: float = 1.6, noise: float = 3.0, cam_jpeg: int = 88,
                 tilt_deg: float = 0.0, paper: float = 168.0,
                 ink: float = 48.0,
                 frame_pad: float = 0.9) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """按 ``scale`` 把印版放进相机画面，返回 (照片, 印版包围框)。

    ``frame_pad`` 是印版四周留白相对印版长边的倍数。默认 0.9 相当于一张
    「贴着印版拍」的小图；真实手机照是 12 MP，logo 只占画面一小块，这时
    要把 frame_pad 调大（例如 6）才能复现「整幅上传被压到极限」的场景。
    """
    h, w = plate.shape
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    small = cv2.resize(plate.astype(np.float32), (nw, nh),
                       interpolation=cv2.INTER_AREA)
    cov = np.clip((255.0 - small) / 255.0, 0.0, 1.0)
    small = paper * (1.0 - cov) + ink * cov

    margin = int(round(frame_pad * max(nw, nh)))
    fw, fh = nw + 2 * margin, nh + 2 * margin
    canvas = np.full((fh, fw), paper, np.float32)
    canvas[margin:margin + nh, margin:margin + nw] = small
    # 纸面/釉面细纹
    canvas += cv2.GaussianBlur(
        rng.normal(0, 6.0, (fh, fw)).astype(np.float32), (0, 0), 0.6)
    if tilt_deg:
        M = cv2.getRotationMatrix2D((fw / 2, fh / 2), tilt_deg, 1.0)
        canvas = cv2.warpAffine(canvas, M, (fw, fh),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        rot_corners = cv2.transform(
            np.array([[[margin, margin], [margin + nw, margin],
                       [margin + nw, margin + nh], [margin, margin + nh]]],
                     np.float32), M).reshape(-1, 2)
    else:
        rot_corners = np.array([[margin, margin], [margin + nw, margin],
                                [margin + nw, margin + nh], [margin, margin + nh]],
                               np.float32)
    if blur > 0.05:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas += rng.normal(0, noise, canvas.shape).astype(np.float32)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(cam_jpeg)])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    x0, y0 = np.floor(rot_corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(rot_corners.max(axis=0)).astype(int)
    return img, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


# ---------------------------------------------------------------------------
# 客户端：ROI 裁块 + JPEG 质量阶梯（与 exportRectJpeg / shrinkForCallContainer 一致）
# ---------------------------------------------------------------------------
def expand_box(box, ratio: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh = box
    ex = int(round(bw * ratio))
    ey = int(round(bh * ratio))
    x0 = max(0, x - ex)
    y0 = max(0, y - ey)
    x1 = min(w, x + bw + ex)
    y1 = min(h, y + bh + ey)
    return x0, y0, x1 - x0, y1 - y0


def encode_jpeg(img: np.ndarray, quality: float) -> Tuple[bytes, int, int]:
    q = int(round(float(np.clip(quality, 0.01, 1.0)) * 100))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return enc.tobytes(), img.shape[1], img.shape[0]


def client_roi_ladder(crop: np.ndarray, *, mode: str = "current",
                      limit: int = ROI_JPEG_LIMIT_BYTES,
                      quality: Optional[float] = None) -> Dict:
    """模拟 exportRectJpeg：先在最大边长上过质量阶梯，不够再缩像素。

    ``mode="fixed"`` 时按 ``quality`` 在原始像素上编一次（剂量-反应实验用）。
    """
    h, w = crop.shape
    long_side = max(w, h)
    start = min(long_side, ROI_LONG_SIDE_MAX)
    sides: List[int] = [start]
    for side in (2000, 1600, 1280, 1024, 800, 640):
        if side < start:
            sides.append(side)
    seen = set()
    sides = [s for s in sides if not (s in seen or seen.add(s))]

    def emit(side: int, quality: float) -> Dict:
        sc = side / long_side
        dw = max(2, int(round(w * sc)))
        dh = max(2, int(round(h * sc)))
        if sc != 1.0:
            canvas = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            canvas = crop
        data, ew, eh = encode_jpeg(canvas, quality)
        return {"data": data, "bytes": len(data), "width": ew, "height": eh,
                "quality": quality, "side": side}

    if mode == "lossless":
        ok, enc = cv2.imencode(".png", crop)
        return {"data": enc.tobytes(), "bytes": len(enc), "width": w,
                "height": h, "quality": None, "side": long_side}

    if mode == "fixed":
        q = 0.85 if quality is None else float(quality)
        data, ew, eh = encode_jpeg(crop, q)
        return {"data": data, "bytes": len(data), "width": ew, "height": eh,
                "quality": q, "side": long_side}

    if mode == "current":
        # 原逻辑：阶梯顺序 first-fit
        for side in sides:
            for quality in ROI_QUALITIES:
                cand = emit(side, quality)
                if cand["bytes"] <= limit:
                    return cand
        return cand  # 全部超限时返回最后一档（原实现会抛错/回退）

    if mode == "maxq":
        # 修正后的阶梯：先在原像素上把质量顶到上限，再考虑缩像素。
        for side in sides:
            budget: Optional[Dict] = None
            q = 0.95
            while q >= 0.70 - 1e-9:
                cand = emit(side, q)
                if cand["bytes"] <= limit:
                    budget = cand
                    break
                q -= 0.05
            if budget is None:
                budget = emit(side, 0.70)
            if budget["bytes"] <= limit:
                return budget
        return budget  # 仍未达标则交给上层缩像素

    raise ValueError(f"unknown client mode {mode!r}")


def client_fullframe_ladder(photo: np.ndarray, *, mode: str = "current",
                            limit: int = CALLCONTAINER_MAX_BYTES) -> Dict:
    """模拟 shrinkForCallContainer：整幅上传时 [8,70] 二分质量 + 宽度阶梯。"""
    h, w = photo.shape
    if mode == "lossless":
        ok, enc = cv2.imencode(".png", photo)
        return {"data": enc.tobytes(), "bytes": len(enc), "width": w,
                "height": h, "quality": None, "side": max(w, h)}
    for width in FULLFRAME_WIDTHS:
        if width is not None and width >= w:
            continue
        out = photo if width is None else cv2.resize(
            photo, (width, max(2, int(round(h * width / w)))),
            interpolation=cv2.INTER_AREA)
        best = None
        lo, hi = 8, 70
        while lo <= hi:
            mid = (lo + hi) // 2
            data, ew, eh = encode_jpeg(out, mid / 100.0)
            cand = {"data": data, "bytes": len(data), "width": ew,
                    "height": eh, "quality": mid / 100.0,
                    "side": max(ew, eh)}
            if len(data) <= limit:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best
    return cand


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def blank_plate() -> Optional[np.ndarray]:
    """无码头像（同一套字形，不含调制），用来量误接受。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "新印版_v3_10px", "ARROW_无码对照.png")
    if not os.path.exists(path):
        return None
    img = A.imread_u(path)
    return img if img is not None and img.size else None


def run(design, *, layout: int = I.LAYOUT_V3,
        payloads: Sequence[str] = DEFAULT_PAYLOADS,
        scales: Sequence[float] = (0.13, 0.16, 0.20, 0.30),
        seeds: Sequence[int] = (1, 2), client: str = "current",
        full_frame: bool = False, blur: float = 1.6, noise: float = 3.0,
        cam_jpeg: int = 88, tilt: float = 0.0, frame_pad: float = 0.9,
        unmarked: bool = False, quality: Optional[float] = None,
        verbose: bool = False) -> int:
    carrier = I.InnerCarrier(design, depth=I.DEPTH, layout_version=layout)
    session = AD.Decoder(carrier="inner", design=design, depth=I.DEPTH,
                         layout_version=layout)
    items: List[Tuple[Optional[str], np.ndarray]] = [
        (dg, carrier.embed(dg)["plate"]) for dg in payloads]
    if unmarked:
        blank = blank_plate()
        if blank is None:
            print("  ! 找不到无码对照图，跳过误接受测量")
        else:
            items.append((None, blank))

    print("=" * 104)
    print(f"  客户端链路压测  layout=v{layout}  client={client}"
          f"{'  [整幅上传]' if full_frame else '  [ROI 上传]'}"
          f"  留白/印版长边={frame_pad:g}")
    print(f"  {len(payloads)} 号码{' + 无码对照' if unmarked else ''} × "
          f"{len(scales)} 尺度 × {len(seeds)} 种子"
          f"   相机失焦 σ={blur:g}  噪声 σ={noise:g}  机内 JPEG q{cam_jpeg}")
    print("=" * 104)
    header = (f"| {'尺度':>5} | {'接受且对':>9} | {'接受且错':>9} "
              f"| {'拒绝':>6} | {'中位裕度':>9} | {'上传KB':>7} | {'JPEG q':>7} "
              f"| {'上传px':>10} | {'中位cc':>7} |")
    print(header)
    print("|" + "-" * 7 + "|" + "-" * 11 + "|" + "-" * 11
          + "|" + "-" * 8 + "|" + "-" * 11 + "|" + "-" * 9 + "|" + "-" * 9
          + "|" + "-" * 12 + "|" + "-" * 9 + "|")

    wrong_total = 0
    for s in scales:
        n = ok = wrong = 0
        margins: List[float] = []
        kbs: List[float] = []
        qs: List[float] = []
        pxs: List[int] = []
        ccs: List[float] = []
        reasons: Dict[str, int] = {}
        t0 = time.time()
        for dg, plate in items:
            for sd in seeds:
                rng = np.random.default_rng(1000 * sd + int(round(s * 1000)))
                photo, box = render_photo(plate, s, rng=rng, blur=blur,
                                          noise=noise, cam_jpeg=cam_jpeg,
                                          tilt_deg=tilt, frame_pad=frame_pad)
                if full_frame:
                    packed = client_fullframe_ladder(photo, mode=client)
                else:
                    roi = expand_box(box, ROI_EXPAND_RATIO,
                                     photo.shape[1], photo.shape[0])
                    x, y, bw, bh = roi
                    crop = photo[y:y + bh, x:x + bw]
                    packed = client_roi_ladder(crop, mode=client,
                                               quality=quality)
                sent = cv2.imdecode(np.frombuffer(packed["data"], np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
                r = session.decode(sent)
                n += 1
                kbs.append(packed["bytes"] / 1024.0)
                if packed.get("quality") is not None:
                    qs.append(float(packed["quality"]))
                pxs.append(max(packed["width"], packed["height"]))
                ccs.append(float(r.cc))
                margins.append(float(r.margin))
                if dg is None:
                    # 无码对照：任何接受都是误接受
                    if r.accepted:
                        wrong += 1
                        if verbose:
                            print(f"    !! 无码图被接受为 {r.digits} (scale {s})")
                    else:
                        ok += 1
                elif r.accepted and r.digits == dg:
                    ok += 1
                elif r.accepted:
                    wrong += 1
                    if verbose:
                        print(f"    !! 错号 {dg} -> {r.digits} (scale {s})")
                else:
                    key = r.reason.split(",")[0] or "unknown"
                    reasons[key] = reasons.get(key, 0) + 1
        wrong_total += wrong
        dt = (time.time() - t0) / max(n, 1)
        med = lambda v: float(np.median(v)) if v else 0.0
        qtext = f"{med(qs):.2f}" if qs else " png "
        print(f"| {s:>5.2f} | {ok:>4d}/{n:<4d} | "
              f"{wrong:>4d}/{n:<4d} | "
              f"{n - ok - wrong:>4d}/{n:<4d} | "
              f"{med(margins):>9.2f} | {med(kbs):>7.1f} | {qtext:>7} | "
              f"{int(np.median(pxs)) if pxs else 0:>10d} | {med(ccs):>7.3f} |")
        if reasons:
            top = ", ".join(f"{k}×{v}" for k, v in
                            sorted(reasons.items(), key=lambda kv: -kv[1]))
            print(f"        拒绝原因：{top}   ({dt:.2f}s/张)")
        else:
            print(f"        ({dt:.2f}s/张)")
    print("=" * 104)
    if wrong_total:
        print(f"  [!] 接受且错号/误接受 {wrong_total} 次 —— 必须先把这些压到 0，"
              f"再谈召回率。")
    else:
        print("  错号/误接受 0：所有接受的结果都是正确号码。")
    return wrong_total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARROW 小程序上传链路端到端压测")
    ap.add_argument("--client", choices=("current", "maxq", "lossless", "fixed"),
                    default="current", help="客户端压缩策略")
    ap.add_argument("--roi-quality", type=float, default=None,
                    help="配合 --client fixed：在原始像素上按该质量编一次")
    ap.add_argument("--full-frame", action="store_true",
                    help="模拟定位失败、整幅上传（shrinkForCallContainer）")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.13, 0.16, 0.20, 0.30])
    ap.add_argument("-n", type=int, default=2, help="每个尺度的随机种子数")
    ap.add_argument("--layout", choices=("2", "3"), default="3")
    ap.add_argument("--blur", type=float, default=1.6)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--cam-jpeg", type=int, default=88)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--frame-pad", type=float, default=0.9,
                    help="印版四周留白（相对印版长边）。0.9=贴着拍；"
                         "6≈12 MP 手机照里 logo 只占一小块")
    ap.add_argument("--unmarked", action="store_true",
                    help="加一张无码对照图，量误接受")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    design = A.ArrowDesign()
    run(design, layout=int(args.layout), scales=args.scales,
        seeds=tuple(range(1, args.n + 1)), client=args.client,
        full_frame=args.full_frame, blur=args.blur, noise=args.noise,
        cam_jpeg=args.cam_jpeg, tilt=args.tilt, frame_pad=args.frame_pad,
        unmarked=args.unmarked, quality=args.roi_quality,
        verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
