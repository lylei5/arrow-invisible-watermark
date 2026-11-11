"""Inner-hole geometry and original-resolution measurements for ARROW v5."""
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import arrow_wm as A

DEPTH = 8.0
N_THETA = 1440
SCAN_HALF = 26.0
SCAN_HALF_MIN = 16.0
SCAN_HALF_MAX = 40.0
SCAN_STEP = 0.25
BLUR_SIGMA0 = 1.6
SCAN_SIGMAS = 2.2
SCAN_WEIGHT_FRAC = 0.55
SCRAMBLE = tuple(int(b) for b in np.random.default_rng(20260311).integers(0, 2, 77))

class InnerCarrier:
    """Allocate 77 smooth radial cells among the four enclosed letter holes."""
    def __init__(self, design, depth=DEPTH, verbose=False):
        self.design = design
        self.depth = float(depth)
        self.layout_version = 5
        self.n_cells = self.n_bits = 77
        self.SS = A.SS
        self.theta = np.linspace(0., 2. * np.pi, N_THETA, endpoint=False)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
        self.holes = self._extract_holes()
        self._alloc_cells()
        self.pair_order = self._build_pair_order()

    def _build_pair_order(self):
        raise NotImplementedError('Use arrow_v5.Carrier')


    def _extract_holes(self) -> List[Dict]:
        ink = (self.design.mask4 > 127).astype(np.uint8)
        bg = (1 - ink).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bg, 4)
        H, W = ink.shape
        out: List[Dict] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                continue                     # 与画幅相连 -> 外部背景，不是孔
            if area < 800:
                continue                     # 4x 域，约等于 1x 的 50 px^2
            m = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = cnts[0].reshape(-1, 2).astype(np.float64)
            peri4 = float(cv2.arcLength(cnts[0], True))
            cen = c.mean(axis=0)
            d = c - cen
            r = np.hypot(d[:, 0], d[:, 1])
            th = np.arctan2(d[:, 1], d[:, 0]) % (2.0 * np.pi)
            order = np.argsort(th)
            th_s, r_s = th[order], r[order]
            # 周期延拓后线性插值 -> 稠密的 r(theta) 表
            th_ext = np.concatenate([th_s - 2 * np.pi, th_s, th_s + 2 * np.pi])
            r_ext = np.tile(r_s, 3)
            r_tab = np.interp(self.theta, th_ext, r_ext)
            out.append(dict(c=cen, r_tab=r_tab, peri=peri4,
                            poly_orig=np.round(c).astype(np.int32)))
        # 按周长降序，保证分配稳定
        out.sort(key=lambda h: -h["peri"])
        return out

    def _alloc_cells(self) -> None:
        n_tot = self.n_cells
        per = np.array([h["peri"] for h in self.holes], np.float64)
        raw = per / per.sum() * n_tot
        alloc = np.maximum(1, np.floor(raw).astype(int))
        while alloc.sum() < n_tot:
            score = raw - alloc
            alloc[int(np.argmax(score))] += 1
        while alloc.sum() > n_tot:
            cand = [i for i in range(len(alloc)) if alloc[i] > 1]
            if not cand:
                break
            score = np.array([alloc[i] - raw[i] if alloc[i] > 1 else -1e9
                              for i in range(len(alloc))])
            alloc[int(np.argmax(score))] -= 1
        for h, k in zip(self.holes, alloc):
            h["n_cells"] = int(k)
            h["dtheta"] = 2.0 * np.pi / int(k)
        self.cells = []
        for hi, h in enumerate(self.holes):
            for k in range(h["n_cells"]):
                self.cells.append((hi, k))
        assert len(self.cells) == n_tot, \
            f"码元数 {len(self.cells)} != {n_tot}"
        return

    def _window(self, hi: int, k: int) -> np.ndarray:
        h = self.holes[hi]
        th = self.theta
        t = ((th - k * h["dtheta"]) % (2.0 * np.pi)) / h["dtheta"]
        inside = t < 1.0
        return np.where(inside, np.sin(np.pi * np.clip(t, 0.0, 1.0)), 0.0)

    def _delta(self, bits):
        fields = [np.zeros(N_THETA) for _ in self.holes]
        for i, b in enumerate(bits):
            hi, k = self.cells[i]
            sign = 1.0 if (int(b) ^ SCRAMBLE[i]) else -1.0
            fields[hi] += sign * self.depth * self._window(hi, k)
        return fields

class InnerDecoder:
    """Measure radial displacements after self-registration of the wordmark."""

    def __init__(self, carrier: InnerCarrier, use_ecc_refine: bool = True):
        self.carrier = carrier
        self.design = carrier.design
        # 字形自同步的配准/矫正链路（自同步 + ECC 精配 + 粗搜抢救）
        # 真实尺寸优化（跳过高分 SIFT、粗搜两阶段、小 logo 边缘恢复）也在这条链路上。
        self._reg = A.WatermarkDecoder(self.design, use_ecc_refine=use_ecc_refine)
        self._tmpl_f = self.design.template.astype(np.float32)
        self._half = 0.0
        self._scale_key = None
        # 梯度质心用的沿 s 平滑核（抑制 JPEG 块效应与网点，σ=1 个采样点=0.25 px）
        n = 4
        k = np.exp(-0.5 * (np.arange(-n, n + 1) / 1.0) ** 2)
        self._k_smooth = (k / k.sum()).astype(np.float32).reshape(1, -1)
        self._build_maps(SCAN_HALF)
        # 参考半径用「未调制模板」实测得到：这样 4x 掩膜轮廓与 1x 灰度跳变沿
        # 之间随角度变化的系统偏差会被逐点抵消，只剩真正的调制位移。
        self._r_ref = self._ref_radii()
        # 预计算 77 个码元窗，供照片测量复用
        self._win = [self.carrier._window(hi, k) for hi, k in self.carrier.cells]
        self._win_den = np.array([max(float((w * w).sum()), 1e-6) for w in self._win],
                                 np.float32)



    @staticmethod
    def scan_half_for_scale(scale: Optional[float], depth: float = DEPTH) -> float:
        """按画面尺度选采样半宽（矫正域 px）。

        ``scale`` 是照片里 logo 宽度 / 底版宽度（``rectify`` 返回的
        ``logo_scale``）。矫正回 2200 px 把相机域模糊放大 1/scale 倍，
        所以斜坡宽度随 scale 反比增长。scale 未知时用默认窗。
        """
        if scale is None or not np.isfinite(scale) or scale <= 0:
            return float(SCAN_HALF)
        sigma = BLUR_SIGMA0 / max(float(scale), 1e-3)
        half = float(depth) + SCAN_SIGMAS * sigma
        return float(np.clip(half, SCAN_HALF_MIN, SCAN_HALF_MAX))

    def _build_maps(self, half: float) -> None:
        """预先把「沿法向扫描」的采样坐标建成固定网格（几何与解码无关）。"""
        half = float(half)
        ss = self.carrier.SS
        n_s = int(round(2 * half / SCAN_STEP)) + 1
        s_grid = np.linspace(-half, half, n_s)
        self._n_s = n_s
        self._half = half
        self._s_grid = s_grid.astype(np.float32)
        # 梯度质心的空间权重：高斯衰减，保证窗口边缘的第二条边不会把质心拉走
        self._scan_w = np.exp(
            -(s_grid / max(SCAN_WEIGHT_FRAC * half, 1e-6)) ** 2).astype(np.float32)
        self._maps: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for h in self.carrier.holes:
            r0 = h["r_tab"] / ss                       # 1x 尺度
            cx, cy = h["c"][0] / ss, h["c"][1] / ss
            rad = r0[:, None] + s_grid[None, :]        # (N_THETA, n_s)
            mx = (cx + rad * self.carrier.cos_t[:, None]).astype(np.float32)
            my = (cy + rad * self.carrier.sin_t[:, None]).astype(np.float32)
            self._maps.append((mx, my, r0))

    def _ensure_scale(self, scale: Optional[float]) -> None:
        """按画面尺度切换采样窗；同一尺度只重建一次。"""
        half = self.scan_half_for_scale(scale, self.carrier.depth)
        # 5% 的量化：相邻尺度不必重建，跨档才重建
        key = int(round(half / max(SCAN_HALF * 0.05, 1e-6)))
        if self._scale_key == key and self._maps:
            return
        self._build_maps(half)
        self._scale_key = key
        # 参考半径必须跟着同一套网格重算：r_ref 是「未调制模板」在同一窗口、
        # 同一估计器下的读数，两者必须同源，否则系统偏差不会被抵消。
        self._r_ref = self._ref_radii()

    def _ref_radii(self) -> List[np.ndarray]:
        profs = self._profiles(self._tmpl_f)
        return [self._radius(p, r0) for p, (_, _, r0) in zip(profs, self._maps)]

    def _radius(self, prof: np.ndarray, r0: np.ndarray) -> np.ndarray:
        """沿每条射线求边缘半径的亚像素位置：梯度加权质心。

        为什么不是阈值过零点：见文件头「自适应采样窗」。质心估计对任意宽度的
        对称模糊斜坡都无偏（模糊只改变斜坡宽度，不改变质心），并且用满整条
        斜坡的样本——相当于对边缘做匹配滤波，噪声按斜坡样本数平均下去。
        阈值法只用斜坡上的 2 个样本，且在斜坡比窗口宽时阈值本身会跟着调制漂移。
        """
        sm = cv2.filter2D(prof, -1, self._k_smooth,
                          borderType=cv2.BORDER_REPLICATE)
        g = np.maximum(-np.gradient(sm, axis=1), 0.0).astype(np.float32)
        w = g * self._scan_w[None, :]
        den = w.sum(axis=1)
        num = (w * self._s_grid[None, :]).sum(axis=1)
        # 对比度门槛：平坡（无边缘）时不输出质心，退回参考半径
        # Partition each scan once for both endpoints; estimator is unchanged.
        low, high = np.percentile(sm, [5, 95], axis=1)
        contrast = high - low
        ok = (contrast > 3.0) & (den > 0.02 * np.maximum(contrast, 1e-6))
        out = r0 + num / np.maximum(den, 1e-6)
        return np.where(ok, out, r0)

    def _profiles(self, img: np.ndarray) -> List[np.ndarray]:
        out = []
        for mx, my, _ in self._maps:
            out.append(cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE))
        return out

    def measure(self, warped: np.ndarray) -> np.ndarray:
        """返回每个码元的径向位移投影 M（单位 px）。"""
        img = warped.astype(np.float32)
        profs = self._profiles(img)
        feats = np.zeros(len(self.carrier.cells), np.float32)
        # 按孔聚合位移，再一次性投影到该孔的全部码元窗
        off = 0
        for hi, (prof, (_, _, r0)) in enumerate(zip(profs, self._maps)):
            r_meas = self._radius(prof, r0)
            delta = r_meas - self._r_ref[hi]
            # 物理上限：单侧位移不可能超过扫描半宽，越界视为测量失败
            delta = np.where(np.abs(delta) <= self._half, delta, 0.0)
            n = self.carrier.holes[hi]["n_cells"]
            for j in range(n):
                feats[off + j] = float(
                    (delta * self._win[off + j]).sum() / self._win_den[off + j])
            off += n
        return feats

    def _detrend(self, feat: np.ndarray) -> np.ndarray:
        """按孔去掉直流偏置（NRZ 的直流免疫替代品）。

        差分调制天然免疫油墨扩散/曝光这类直流变化；NRZ 没有这个性质，
        一个正的全局偏移会让所有 bit 都判成 1。

        **别用滑动平均**：3 点均值会把长度 ≥3 的游程整体减掉——实测干净印版上
        就有 7 个 bit 被压到 |conf|≈0.02–0.12，margin_min 直接掉到 0.01。
        **也别用中位数**：双极信号的 median 非 +depth 即 −depth，码字稍不平衡
        就会把一半码元抹成 0（实测 margin_min 仍是 0.01）。
        正确做法是**每孔减均值**：只去常数偏置，游程不受影响；加扰保证了
        ±1 序列大致平衡，所以扣掉均值只会削去调制自身那一点点直流。
        """
        out = np.array(feat, dtype=np.float64, copy=True)
        off = 0
        for h in self.carrier.holes:
            n = h["n_cells"]
            if n >= 3:
                seg = out[off:off + n]
                out[off:off + n] = seg - float(seg.mean())
            off += n
        return out

    def _decode_gray(self, gray: np.ndarray, return_debug: bool = False) -> Dict:
        """单幅灰度图：矫正 → 内孔半径测量 → 软判决。"""
        r = self._reg.rectify(gray)
        if not r.get("ok"):
            return self._rectify_failure(r)
        return self.decode_rectified(r, return_debug)

    @staticmethod
    def _rectify_failure(r: Dict) -> Dict:
        return {"ok": False, "reason": r.get("reason", "rectify_failed"),
                "digits": None, "bits": None, "mean_conf": 0.0,
                "margin": 0.0, "cc": 0.0,
                "layout_version": None, "crc_ok": None}

    def decode_rectified(self, r: Dict, return_debug: bool = False) -> Dict:
        """Measure de-scrambled soft bits in payload order; the subclass applies CRC."""
        self._ensure_scale(r.get("logo_scale"))
        feat = self.measure(r["warped"])
        ecc = dict(r.get("ecc", {}))
        ecc.setdefault("cc", 0.0)
        conf = self._detrend(feat)
        conf_a = np.array(
            [conf[pi] * (-1.0 if SCRAMBLE[pi] else 1.0)
             for pi in self.carrier.pair_order], np.float32)
        bits = [1 if c > 0 else 0 for c in conf_a]
        digits, crc_ok, n_chg = (None, False, 0)
        ms = A.group_codeword_margins(conf_a)
        margin = float(ms.mean()) if ms.size else 0.0
        return {"ok": True, "digits": digits,
                "payload_valid": digits is not None,
                "bits": bits, "feat": feat, "conf": conf_a,
                "mean_conf": float(np.mean(np.abs(conf_a))) * 0.5,
                "corrected": n_chg, "n_corrected": n_chg,
                "ecc": ecc, "cc": float(ecc.get("cc", 0.0)),
                "logo_scale": float(r.get("logo_scale", 1.0)),
                "warped": r["warped"] if return_debug else None,
                "layout_version": 5,
                "crc_ok": bool(crc_ok), "pad_bits": None,
                "margin": margin, "margin_mean": margin,
                "margin_min": float(ms.min()) if ms.size else 0.0,
                "group_margins": [float(x) for x in ms]}

    def decode(self, image: np.ndarray, return_debug: bool = False) -> Dict:
        """内孔解码入口：整幅照片和候选卡片区域定位。

          路径 A `full`     ：整幅直接解
          路径 B `plate#k`  ：先隔离亮底卡片再解（2.2 cm 实物贴在深色产品上时）

        大画幅 + 小白卡时优先走 B，避免整幅暗背景拖垮二值化/配准；
        择优仍按 (cc≥0.85, mean_conf) 字典序，防止裁错图时 conf 虚高。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        CC_GATE = 0.85

        def _rank(d: Dict):
            return (1 if d.get("cc", 0.0) >= CC_GATE else 0,
                    d.get("mean_conf", 0.0))

        plates = A.WatermarkDecoder._plate_candidates(gray, k=3)
        plate_first = False
        if plates:
            frac = plates[0].size / float(max(gray.size, 1))
            plate_first = frac < 0.45 and max(gray.shape[:2]) >= 1600

        best: Optional[Dict] = None

        def _consider(res: Dict, path: str, isolated: bool = False) -> None:
            nonlocal best
            res = dict(res)
            res["path"] = path
            if isolated:
                res["isolated"] = True
            if best is None or _rank(res) > _rank(best):
                best = res

        def _good_enough(gate_cc: float, gate_conf: float) -> bool:
            # 内孔版 mean_conf 标定与边缘不同（约 ×0.5 的差分尺度），
            # 干净印版约 2.0+；这里用相对宽松的 0.020 作短路门限。
            return (best is not None
                    and best.get("cc", 0.0) >= gate_cc
                    and best.get("mean_conf", 0.0) >= gate_conf)

        def _try_decode(img: np.ndarray) -> Optional[Dict]:
            """跑一次 _decode_gray，把退化画布上的 OpenCV 异常吞成失败。

            实测过：极小/极扁的候选块会让 `_background_diff` 里的 GaussianBlur
            抛 "Unknown C++ exception"，那会直接掀掉整次解码（生产里变成 500）。
            一个坏候选不该毁掉整张图，跳过去就是。
            """
            hh, ww = img.shape[:2]
            if hh < 24 or ww < 48:
                return None
            try:
                return self._decode_gray(img, return_debug)
            except Exception:
                return None

        if not plate_first:
            res = _try_decode(gray)
            if res is not None:
                _consider(res, "full")
                if _good_enough(0.95, 0.020):
                    return best

        for k, cand in enumerate(plates):
            if cand.size / float(max(gray.size, 1)) >= 0.85:
                continue
            res = _try_decode(cand)
            if res is None:
                continue
            _consider(res, f"plate#{k}", isolated=True)
            if _good_enough(CC_GATE, 0.020):
                return best

        if plate_first:
            res = _try_decode(gray)
            if res is not None:
                _consider(res, "full")

        if best is not None:
            return best
        return {"ok": False, "reason": "decode_failed", "digits": None,
                "bits": None, "mean_conf": 0.0, "margin": 0.0, "cc": 0.0,
                "path": "none"}
