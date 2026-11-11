"""Geometry, registration and Hamming primitives for ARROW v5."""
from __future__ import annotations
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

CANVAS_W = 2200
CANVAS_H = 600
SS = 4
INK_LEVEL = 58
BG_LEVEL = 255
PAYLOAD_DIGITS = 11
PAYLOAD_BITS = 37
RECTIFY_INTERP = cv2.INTER_LINEAR
_HAMMING74_CODEBOOK = None


def hamming74_encode(data_bits: Sequence[int]) -> List[int]:
    """每 4 bit -> 7 bit。要求 len(data_bits) % 4 == 0。"""
    if len(data_bits) % 4 != 0:
        raise ValueError("Hamming(7,4) 需要 4 的整数倍长度")
    out: List[int] = []
    for i in range(0, len(data_bits), 4):
        d1, d2, d3, d4 = (int(b) & 1 for b in data_bits[i:i + 4])
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4
        out += [p1, p2, d1, p4, d2, d3, d4]
    return out

def payload_to_bits(digits: str) -> List[int]:
    """11 位十进制 -> 37 bit（MSB first）。"""
    if len(digits) != PAYLOAD_DIGITS or not digits.isdigit():
        raise ValueError("载荷必须是 11 位十进制数字")
    v = int(digits)
    if v >= (1 << PAYLOAD_BITS):
        raise ValueError("载荷超出 37 bit 表示范围")
    return [(v >> (PAYLOAD_BITS - 1 - i)) & 1 for i in range(PAYLOAD_BITS)]

def bits_to_payload(bits37: Sequence[int]) -> Optional[str]:
    """37 bit -> 11 位号码字符串。

    37 bit 能表示到 1.37e11，比 11 位号码的上限 1e11 大，因此约 27% 的
    码字空间对应"不存在的号码"。原实现用取模把它们折回合法区间，
    结果是**任何垃圾输入都能吐出一个人模人样的 11 位号码**，
    只能靠置信度区分——这对验真场景是危险的。
    现在越界直接返回 None，让调用方明确知道"这段码字不合法"。
    """
    v = 0
    for b in bits37:
        v = (v << 1) | (int(b) & 1)
    if v >= 10 ** PAYLOAD_DIGITS:
        return None
    return f"{v:011d}"

def hamming_codebook() -> np.ndarray:
    """16 个合法码字（各 7 bit），用于软判决最大似然译码。"""
    global _HAMMING74_CODEBOOK
    if _HAMMING74_CODEBOOK is None:
        rows = []
        for v in range(16):
            d = [(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1]
            rows.append(hamming74_encode(d))
        _HAMMING74_CODEBOOK = np.array(rows, np.int8)
    return _HAMMING74_CODEBOOK

def group_codeword_margins(soft: Sequence[float]) -> np.ndarray:
    """每组 Hamming 码字的 best-vs-2nd 间隙，再除以整段软信息的标准差。

    返回长度 = 组数（77 bit 为 11 组）。平均值是「有没有水印」的门；
    最小值才会暴露「其中一组选得很勉强」。单组选错或打平时，平均值
    仍可能被其余组抬过 1.8。
    """
    cb = hamming_codebook()
    sgn = (cb.astype(np.float64) * 2.0 - 1.0)
    soft = np.asarray(soft, np.float64).ravel()
    n = int(soft.size) // 7
    if n <= 0:
        return np.zeros(0, np.float64)
    sd = float(np.std(soft)) or 1e-9
    ms = np.empty(n, np.float64)
    for g in range(n):
        v = sgn @ soft[g * 7:(g + 1) * 7]
        # partition 只保证这两位是最大的两个，彼此无序，必须再取 max-min。
        top = np.partition(v, len(v) - 2)[-2:]
        ms[g] = float(np.max(top) - np.min(top)) / sd
    return ms

def imread_u(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> Optional[np.ndarray]:
    """Unicode 安全的读图（Windows 下 cv2.imread 不支持中文路径）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None

def imwrite_u(path: str, img: np.ndarray) -> bool:
    """Unicode 安全的写图（Windows 下 cv2.imwrite 不支持中文路径）。"""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False



def order_quad(q: np.ndarray) -> np.ndarray:
    """把 4 个点整理为 TL / TR / BR / BL 顺序（图像坐标，y 向下）。"""
    q = np.asarray(q, np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    s = q[:, 0] + q[:, 1]
    q = np.roll(q, -int(np.argmin(s)), axis=0)
    if q[1, 0] < q[3, 0]:
        q = q[[0, 3, 2, 1]]
    return q.astype(np.float32)

def make_payloads(n_payloads: int, seed: int) -> List[str]:
    """压测号码。首尾不用 0，同一 seed 在各消融配置之间可以对照。"""
    rng = np.random.default_rng(seed)
    digits_list: List[str] = []
    while len(digits_list) < n_payloads:
        d = "".join(str(x) for x in rng.integers(0, 10, size=PAYLOAD_DIGITS))
        if d[0] == "0":
            d = "1" + d[1:]
        digits_list.append(d)
    return digits_list

class ArrowDesign:
    """Raster geometry and registration references for a supplied vector design."""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.text_mask4 = self._rasterize()
        self.mask4 = self.text_mask4
        small = cv2.resize(self.mask4, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
        self.mask1 = (small > 127).astype(np.uint8)
        self.ref_quad = self._compute_ref_quad()
        self.letter_centroids = self._compute_letter_centroids()
        self.hole_centroids = self._compute_hole_centroids()
        ys, xs = np.nonzero(self.mask1)
        self.ink_bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        p = np.column_stack(np.nonzero(self.mask1)[::-1]).astype(np.float64)
        self.mom_mu = p.mean(axis=0)
        self.mom_C = np.cov((p - self.mom_mu).T)

    def _rasterize(self):
        raise NotImplementedError('Use arrow_vector.VectorDesign')

    def _compute_letter_centroids(self):
        raise NotImplementedError('Use arrow_vector.VectorDesign')


    def _compute_hole_centroids(self) -> List[Tuple[float, float]]:
        """4 个内孔（O 内环 / 两个 R 字怀 / A 上三角）的质心，按 x 排序。

        内孔是**被墨迹完全包围**的背景区域，其质心不受外轮廓模糊、油墨胀缩影响，
        在极端透视下比"字母墨迹质心"更稳，是 4 点精确单应的理想基准。
        """
        bg = (1 - self.mask1).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(bg, 4)
        H, W = self.mask1.shape
        hs: List[Tuple[float, float, int]] = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                continue
            if a < 200:
                continue
            hs.append((float(cents[i][0]), float(cents[i][1]), int(a)))
        hs.sort(key=lambda t: t[0])
        return [(x, y) for x, y, _ in hs]

    def _compute_ref_quad(self) -> np.ndarray:
        """由字形凸包提取 4 个自同步基准角点（与拍摄端同一算法）。"""
        pts = np.column_stack(np.nonzero(self.mask1)[::-1]).astype(np.int32)  # x,y
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        for eps in (0.03, 0.02, 0.04, 0.05, 0.06, 0.015, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                return order_quad(ap.reshape(4, 2).astype(np.float32))
        rect = cv2.minAreaRect(pts)
        return order_quad(cv2.boxPoints(rect).astype(np.float32))

    def render(self, mask4: np.ndarray) -> np.ndarray:
        """把 4x 二值掩膜渲染成 8bit 灰度底版（白底深灰字）。"""
        cov = cv2.resize(mask4, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
        cov_f = cov.astype(np.float32) / 255.0
        return np.clip(BG_LEVEL - cov_f * (BG_LEVEL - INK_LEVEL), 0, 255).astype(np.uint8)

    @property
    def template(self) -> np.ndarray:
        """未加水印的原始底版（解码器用作配准模板）。

        每次访问都会重新渲染整张底版（8800×2400 → 2200×600 的 INTER_AREA），
        而一个解码器会取好几次。首次结果缓存在 ``_template_cache``，之后直接复用。
        """
        cached = getattr(self, "_template_cache", None)
        if cached is None:
            cached = self.render(self.mask4)
            self._template_cache = cached
        return cached

class WatermarkDecoder:
    """Locate and rectify the vector wordmark using SIFT, contours and ECC."""

    def __init__(self, design: ArrowDesign, use_ecc_refine: bool = True,
                 verbose: bool = False,
                 use_sift: bool = True):
        self.design = design
        self.use_sift = use_sift
        self.ref_quad = design.ref_quad.astype(np.float32)
        self.tmpl = design.template.astype(np.float32) / 255.0
        self.use_ecc_refine = use_ecc_refine
        self.verbose = verbose
        # 半分辨率模板（ECC 精配用）
        self._ecc_scale = 0.5
        self._ecc_iters = 30
        self._ecc_eps = 1e-6
        tw = int(round(CANVAS_W * self._ecc_scale))
        th = int(round(CANVAS_H * self._ecc_scale))
        self._tmpl_half = cv2.resize(self.tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        # 1/4 分辨率模板向量（候选四边形 NCC 打分用）
        qs = 0.25
        self._quad_scale = qs
        self._tmpl_q = cv2.resize(self.tmpl, (int(round(CANVAS_W * qs)),
                                              int(round(CANVAS_H * qs))),
                                  interpolation=cv2.INTER_AREA)
        _v = self._tmpl_q.ravel() - self._tmpl_q.mean()
        self._tmpl_vec = _v
        self._tmpl_norm = float(np.linalg.norm(_v))
        # 特征点定位（SIFT + RANSAC）：与亮度、背景、构图完全无关，
        # 是最稳的一条定位路径。模板描述子只算一次。
        self._sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01) \
            if use_sift else None
        self._sift_tpl = (self._sift.detectAndCompute(design.template, None)
                          if self._sift is not None else ([], None))
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        # 粗搜抢救用的 1/8 模板只算一次（每次 recover 再 resize 是纯浪费）
        _rsc = 0.125
        self._recover_sc = _rsc
        self._recover_tmpl = cv2.resize(
            self.tmpl, (int(round(CANVAS_W * _rsc)), int(round(CANVAS_H * _rsc))),
            interpolation=cv2.INTER_AREA).astype(np.float32)
        # 参考四边形的宽高比（用于候选合理性检查）
        w, h = self._quad_wh(self.ref_quad)
        self._ref_ar = w / max(h, 1e-6)

    @staticmethod
    def _quad_wh(q: np.ndarray) -> Tuple[float, float]:
        w = 0.5 * (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3]))
        h = 0.5 * (np.linalg.norm(q[1] - q[2]) + np.linalg.norm(q[3] - q[0]))
        return float(w), float(h)

    def _plausible(self, q: np.ndarray) -> bool:
        w, h = self._quad_wh(q)
        if h < 2.0 or w < 8.0:
            return False
        ar = w / h
        # 透视会把 5:1 压到 ~3:1，放宽到 [0.5x, 2.2x] 参考宽高比
        return 0.45 * self._ref_ar < ar < 2.2 * self._ref_ar

    @staticmethod
    def _background_diff(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """形态学闭运算估计"无墨背景"，返回 (bg-g, 缩放系数)。

        闭运算会填平所有比结构元窄的暗结构（笔画），于是 bg 就是
        "假如这里没有印字会是什么样"。差分 bg-g 即纯墨量，对釉面渐变、
        局部高光、整体曝光都不敏感。

        这一步与阈值无关，却被多阈值循环调用了三遍——缩图、模糊、两次
        大核形态学是全流程最贵的算子之一。拆出来只算一次。
        """
        sc = 0.5
        sm = cv2.resize(gray, (int(gray.shape[1] * sc), int(gray.shape[0] * sc)),
                        interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (3, 3), 0)
        k = int(min(sm.shape) * 0.15) | 1
        k = max(31, min(201, k))
        bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
        return cv2.subtract(bg, sm), sc

    @staticmethod
    def _thresh_ink(diff: np.ndarray, thr: float) -> np.ndarray:
        """把背景差分图按阈值二值化成墨迹掩膜（与原实现同序同参）。"""
        ink = (diff > thr).astype(np.uint8) * 255
        return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def _binarize(gray: np.ndarray, thr: float = 38.0) -> Tuple[np.ndarray, float]:
        """单次二值化（保留给需要独立调用的地方）。"""
        diff, sc = WatermarkDecoder._background_diff(gray)
        return WatermarkDecoder._thresh_ink(diff, thr), sc

    def _ncc_H(self, g_small: np.ndarray, Hm: np.ndarray) -> float:
        """用给定单应把 1/4 分辨率图矫正到标准坐标系，与模板算归一化互相关。"""
        qs = self._quad_scale
        th, tw = self._tmpl_q.shape
        S = np.array([[qs, 0, 0], [0, qs, 0], [0, 0, 1]], np.float32)
        Hs = (S @ Hm.astype(np.float32) @ np.linalg.inv(S)).astype(np.float32)
        w = cv2.warpPerspective(g_small, Hs, (tw, th), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        a = w.ravel() - w.mean()
        na = float(np.linalg.norm(a))
        if na < 1e-6:
            return -1.0
        return float(a.dot(self._tmpl_vec) / (na * self._tmpl_norm + 1e-9))

    def _sift_homography(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """用 SIFT 特征点 + RANSAC 直接估出 image->plate 的单应。

        这条路不依赖二值化、不依赖背景明暗、不依赖 logo 是否占满画面，
        对缩放/旋转/透视/画面杂物都免疫，是最可靠的定位方式。
        返回 None 表示特征点不够（画面太小、过糊或被大面积遮挡）。
        """
        if self._sift is None:
            return None
        kp1, des1 = self._sift_tpl
        if des1 is None or len(kp1) < 8:
            return None
        # SIFT 本身尺度无关，大图先降采样再跑。目标长边实测：1400 最好；
        # 降到 1000 内点数看着更多，但极限档成功率 74%→66%，故维持 1400。
        sc = 1.0
        small = gray
        long_side = max(gray.shape[:2])
        if long_side > 1400:
            sc = 1400.0 / long_side
            small = cv2.resize(gray, (int(round(gray.shape[1] * sc)),
                                      int(round(gray.shape[0] * sc))),
                               interpolation=cv2.INTER_AREA)
        kp2, des2 = self._sift.detectAndCompute(small, None)
        if des2 is None or len(kp2) < 8:
            return None
        try:
            matches = self._bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            return None
        good = [a for a, b in matches if a.distance < 0.75 * b.distance]
        if len(good) < 8:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hpi, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if Hpi is None or mask is None or int(mask.sum()) < 8:
            return None
        # 把"plate->缩小图"换算回"plate->原图"，再取逆得到 image->plate
        if sc != 1.0:
            D = np.diag([1.0 / sc, 1.0 / sc, 1.0])
            Hpi = (D @ Hpi.astype(np.float64)).astype(np.float32)
        try:
            Hm = np.linalg.inv(Hpi.astype(np.float64)).astype(np.float32)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(Hm)):
            return None
        return Hm

    def _find_transforms(self, gray: np.ndarray) -> List[np.ndarray]:
        """无标记自同步：产生若干候选单应，按 1/4 分辨率 NCC 排序。

        首选基准是**五个字母各自的墨迹质心**（5 点超定单应，最稳）：
        质心是全局积分量，对模糊、油墨胀缩、局部高光都不敏感。
        字母被糊成一团时，退回"整排字母凸包四角"（4 点）方案，
        且凸包简化在主轴坐标系里进行，避免滚转导致四点对应错位。

        顺序：先跑几何候选（便宜）。若最佳 NCC≥0.92，说明字形自同步已经
        足够准，跳过 SIFT（省 ~60–120 ms）。只有几何候选偏弱时才启用 SIFT——
        这与"SIFT 命中后跳过其他候选"不同，后者会掉样本。
        """
        qs = self._quad_scale
        g_small = cv2.resize(gray, (int(gray.shape[1] * qs), int(gray.shape[0] * qs)),
                             interpolation=cv2.INTER_AREA)
        pool: List[Tuple[float, np.ndarray]] = []
        dst_c = np.array(self.design.letter_centroids, np.float32)

        diff, sc = self._background_diff(gray)
        for thr in (38.0, 26.0, 18.0):
            ink = self._thresh_ink(diff, thr)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, 8)
            if n < 3:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
            amax = float(areas.max()) if areas.size else 0.0
            if amax < 200:
                continue
            ids = [i for i in range(1, n) if areas[i - 1] >= 0.03 * amax]

            # ---- 候选 A：五个字母质心（5 点超定单应） ----
            if len(ids) == 5:
                ids_s = sorted(ids, key=lambda i: cents[i][0])
                src = np.array([cents[i] for i in ids_s], np.float32) / sc
                Hm, _ = cv2.findHomography(src, dst_c, cv2.RANSAC, 4.0)
                if Hm is not None and np.all(np.isfinite(Hm)):
                    pool.append((self._ncc_H(g_small, Hm), Hm.astype(np.float32)))

            # ---- 候选 E：内孔质心（4 点精确单应，极端透视下最稳） ----
            if len(self.design.hole_centroids) == 4:
                bgm = (ink < 128).astype(np.uint8)
                nb, labb, sb, cb = cv2.connectedComponentsWithStats(bgm, 4)
                Hb, Wb = bgm.shape
                hb: List[Tuple[float, float, int]] = []
                for j in range(1, nb):
                    xb, yb, wb, hbb, ab = sb[j]
                    if xb == 0 or yb == 0 or xb + wb >= Wb or yb + hbb >= Hb:
                        continue
                    if ab < 800:
                        continue
                    hb.append((float(cb[j][0]), float(cb[j][1]), int(ab)))
                if len(hb) == 4:
                    hb.sort(key=lambda t: t[0])
                    src = np.array([[t[0], t[1]] for t in hb], np.float32) / sc
                    dst_h = np.array(self.design.hole_centroids, np.float32)
                    Hm, _ = cv2.findHomography(src, dst_h, 0)
                    if Hm is not None and np.all(np.isfinite(Hm)):
                        pool.append((self._ncc_H(g_small, Hm), Hm.astype(np.float32)))

            # ---- 候选 C：二阶矩仿射（天然含剪切，抗局部形变） ----
            lutc = np.zeros(n, np.uint8)
            for i in ids:
                lutc[i] = 255
            yc_, xc_ = np.nonzero(lutc[lab])
            if len(xc_) > 500:
                po = np.column_stack([xc_, yc_]).astype(np.float64) / sc
                mu_o = po.mean(axis=0)
                C_o = np.cov((po - mu_o).T)
                Ld, Vd = np.linalg.eigh(self.design.mom_C)
                Lo, Vo = np.linalg.eigh(C_o)
                Ld = np.maximum(Ld, 1e-6)
                Lo = np.maximum(Lo, 1e-6)
                Dd = np.diag(1.0 / np.sqrt(Ld))
                Do = np.diag(np.sqrt(Lo))
                for sgn in (1.0, -1.0):
                    Qm = np.diag([1.0, sgn])
                    A = Vo @ Do @ Qm @ Dd @ Vd.T
                    if abs(np.linalg.det(A)) < 1e-6:
                        continue
                    # 上式解出的是 模板->照片（p_obs = A·p_design + tt），
                    # 而候选池约定的是 照片->模板，必须取逆再入池。
                    tt = mu_o - A @ self.design.mom_mu
                    Hpi = np.array([[A[0, 0], A[0, 1], tt[0]],
                                    [A[1, 0], A[1, 1], tt[1]],
                                    [0.0, 0.0, 1.0]], np.float64)
                    if abs(np.linalg.det(Hpi)) < 1e-9:
                        continue
                    Hm = np.linalg.inv(Hpi).astype(np.float32)
                    if not np.all(np.isfinite(Hm)):
                        continue
                    pool.append((self._ncc_H(g_small, Hm), Hm))

            # ---- 候选 D：墨迹包围盒 -> 设计墨迹包围盒 ----
            # 对干净印版而言这就是恒等变换；对拍摄图是一个稳健的粗对齐。
            lutb = np.zeros(n, np.uint8)
            for i in ids:
                lutb[i] = 255
            yb_, xb_ = np.nonzero(lutb[lab])
            if len(xb_) > 500:
                ox0, ox1 = xb_.min() / sc, xb_.max() / sc
                oy0, oy1 = yb_.min() / sc, yb_.max() / sc
                bx0, by0, bx1, by1 = self.design.ink_bbox
                if (ox1 - ox0) > 10 and (oy1 - oy0) > 5:
                    sx = (bx1 - bx0) / (ox1 - ox0)
                    sy = (by1 - by0) / (oy1 - oy0)
                    Hm = np.array([[sx, 0.0, bx0 - sx * ox0],
                                   [0.0, sy, by0 - sy * oy0],
                                   [0.0, 0.0, 1.0]], np.float32)
                    pool.append((self._ncc_H(g_small, Hm), Hm))

            # ---- 候选 B：整排字母凸包四角 ----
            lut = np.zeros(n, np.uint8)
            for i in ids:
                lut[i] = 255
            ys, xs = np.nonzero(lut[lab])
            if len(xs) < 800:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(pts.astype(np.int32))
            ang = float(rect[2])
            if ang < -45.0:
                ang += 90.0
            elif ang > 45.0:
                ang -= 90.0
            M = cv2.getRotationMatrix2D((float(rect[0][0]), float(rect[0][1])), ang, 1.0)
            pts_r = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
            hull = cv2.convexHull(pts_r.astype(np.int32))
            peri = cv2.arcLength(hull, True)
            Minv = cv2.invertAffineTransform(M)
            for eps in (0.03, 0.02, 0.04, 0.05, 0.06, 0.08):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) != 4:
                    continue
                qr = cv2.transform(ap.reshape(1, -1, 2).astype(np.float32), Minv).reshape(4, 2)
                q = order_quad(qr / sc)
                if not self._plausible(q):
                    continue
                Hm = cv2.getPerspectiveTransform(q.astype(np.float32), self.ref_quad)
                pool.append((self._ncc_H(g_small, Hm), Hm.astype(np.float32)))

        best_geo = max((s for s, _ in pool), default=-1.0)
        # ---- 候选 F：SIFT + RANSAC（几何候选不够准时才跑） ----
        # 干净印版/轻中度扰动下几何 NCC 经常 ≥0.92，此时 SIFT 是纯开销。
        # 重度/极限或构图杂乱时几何偏弱，SIFT 才是主力。
        if best_geo < 0.92:
            Hs = self._sift_homography(gray)
            if Hs is not None:
                pool.append((self._ncc_H(g_small, Hs), Hs))

        if not pool:
            return []
        pool.sort(key=lambda t: -t[0])
        out: List[np.ndarray] = []
        for _, Hm in pool:
            if all(float(np.abs(Hm - o).max()) > 1e-3 for o in out):
                out.append(Hm)
            if len(out) >= 4:
                break
        return out

    def _ecc_refine(self, warped: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """用模板做 ECC 精配，先试 8 自由度单应，失败回退欧氏变换。"""
        sc = self._ecc_scale
        th, tw = self._tmpl_half.shape
        w_small = cv2.resize(warped.astype(np.float32) / 255.0, (tw, th),
                             interpolation=cv2.INTER_AREA)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self._ecc_iters, self._ecc_eps)
        S = np.array([[sc, 0, 0], [0, sc, 0], [0, 0, 1]], np.float32)
        Si = np.linalg.inv(S)
        # 单应已经拿到可用解（cc≥0.55）就不必再跑欧氏——欧氏几乎不会更好，
        # 却要再付一次 findTransformECC。
        motions = ((cv2.MOTION_HOMOGRAPHY, (3, 3)),
                   (cv2.MOTION_EUCLIDEAN, (2, 3)))
        for motion, wshape in motions:
            try:
                wm = np.eye(wshape[0], wshape[1], dtype=np.float32)
                cc, wm = cv2.findTransformECC(self._tmpl_half, w_small, wm,
                                              motion, crit, None, 5)
            except Exception:
                continue
            if not np.isfinite(cc) or cc < 0.40:
                continue
            if motion == cv2.MOTION_HOMOGRAPHY:
                corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]],
                                   np.float32).reshape(1, -1, 2)
                moved = cv2.perspectiveTransform(corners, wm).reshape(-1, 2)
                if float(np.abs(moved - corners.reshape(-1, 2)).max()) > 24.0:
                    continue
                Hf = (Si @ wm.astype(np.float32) @ S).astype(np.float32)
                warped = cv2.warpPerspective(
                    warped, Hf, (CANVAS_W, CANVAS_H),
                    flags=RECTIFY_INTERP | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE)
                shift = float(np.abs(moved - corners.reshape(-1, 2)).max()) / float(sc)
                return warped, {"cc": float(cc), "motion": int(motion),
                                "residual": shift}
            else:
                wf = wm.copy()
                wf[0, 2] /= sc
                wf[1, 2] /= sc
                warped = cv2.warpAffine(
                    warped, wf, (CANVAS_W, CANVAS_H),
                    flags=RECTIFY_INTERP | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE)
                residual = float(math.hypot(float(wf[0, 2]), float(wf[1, 2])))
                return warped, {"cc": float(cc), "motion": int(motion),
                                "residual": residual}
        return warped, {"rejected": True}

    def _recover_alignment(self, warped: np.ndarray) -> Tuple[np.ndarray, float]:
        """粗对齐抢救：1/8 分辨率上由粗到细搜索，扩大 ECC 收敛域。

        只在 ECC 相关性偏低时启用。原实现是 7×5×5=175 次全组合；
        改为两阶段：先搜旋转×缩放（剪切=0），再在最优点附近扫剪切与
        微调角度——同等覆盖、约 1/3 代价。
        """
        sc = self._recover_sc
        t = self._recover_tmpl
        th, tw = t.shape
        w = cv2.resize(warped.astype(np.float32) / 255.0, (tw, th),
                       interpolation=cv2.INTER_AREA)
        pad = 26
        cx, cy = tw * 0.5, th * 0.5
        best_s, best_T = -2.0, None
        best_ang, best_scl = 0.0, 1.0

        def _score(M3: np.ndarray) -> Tuple[float, np.ndarray]:
            w2 = cv2.warpAffine(w, M3[:2], (tw, th), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            w2p = cv2.copyMakeBorder(w2, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            res = cv2.matchTemplate(w2p, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            T = np.array([[1, 0, mloc[0] - pad], [0, 1, mloc[1] - pad],
                          [0, 0, 1]], np.float32)
            return float(mx), T @ M3

        # 阶段 1：旋转 × 缩放（35 次）
        for ang in (-9, -6, -3, 0, 3, 6, 9):
            for scl in (0.90, 0.95, 1.0, 1.05, 1.10):
                M0 = cv2.getRotationMatrix2D((cx, cy), ang, scl)
                M3 = np.vstack([M0, [0, 0, 1]]).astype(np.float32)
                mx, T = _score(M3)
                if mx > best_s:
                    best_s, best_T, best_ang, best_scl = mx, T, float(ang), float(scl)
        # 阶段 2：在最优点附近扫剪切 + 微角度（5×5=25 次）
        for sh in (-0.14, -0.07, 0.0, 0.07, 0.14):
            for dang in (-2.0, -1.0, 0.0, 1.0, 2.0):
                M0 = cv2.getRotationMatrix2D((cx, cy), best_ang + dang, best_scl)
                Sh = np.array([[1.0, sh, -sh * cy], [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]], np.float32)
                M3 = Sh @ np.vstack([M0, [0, 0, 1]]).astype(np.float32)
                mx, T = _score(M3)
                if mx > best_s:
                    best_s, best_T = mx, T
        if best_T is None:
            return warped, -1.0
        S = np.array([[sc, 0, 0], [0, sc, 0], [0, 0, 1]], np.float32)
        Hf = (np.linalg.inv(S) @ best_T @ S).astype(np.float32)
        out = cv2.warpPerspective(warped, Hf, (CANVAS_W, CANVAS_H),
                                  flags=RECTIFY_INTERP | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
        return out, best_s

    def _logo_scale(self, H: np.ndarray) -> float:
        """由 image→plate 单应反推照片里 logo 相对底版的尺度（宽比）。"""
        try:
            Hi = np.linalg.inv(H.astype(np.float64)).astype(np.float32)
            q = cv2.perspectiveTransform(
                self.ref_quad.reshape(1, -1, 2), Hi).reshape(-1, 2)
            w = 0.5 * (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2]))
            return float(w / CANVAS_W)
        except Exception:
            return 1.0

    @staticmethod
    def _restore_edges(warped: np.ndarray, scale: float) -> np.ndarray:
        """真实尺寸拍摄时 logo 往往只占画面一小块，矫正回 2200×600 等于强放大，
        双线性会抹掉亚像素调制（幅度见 MOD_DEPTH）。轻度反锐化遮罩把边缘对比度拉回来。
        scale≥0.45（接近半幅以上）时不处理，避免动干净印版的数值路径。
        """
        if scale >= 0.45:
            return warped
        up = 1.0 / max(scale, 1e-6)
        amt = min(1.35, 0.30 + 0.22 * (up - 2.0))
        if amt <= 0.05:
            return warped
        sig = min(1.7, 0.75 + 0.10 * up)
        blur = cv2.GaussianBlur(warped, (0, 0), sig)
        return np.clip(cv2.addWeighted(warped, 1.0 + amt, blur, -amt, 0),
                       0, 255).astype(np.uint8)

    def rectify(self, gray: np.ndarray) -> Dict:
        """多起点矫正 + 收敛域抢救，取相关性最高的结果。"""
        Hs = self._find_transforms(gray)
        if not Hs:
            return {"ok": False, "reason": "sync_failed"}
        best = None
        for k, Hm in enumerate(Hs):
            warped = cv2.warpPerspective(gray, Hm, (CANVAS_W, CANVAS_H),
                                         flags=RECTIFY_INTERP,
                                         borderMode=cv2.BORDER_REPLICATE)
            ecc: Dict = {}
            if self.use_ecc_refine:
                warped, ecc = self._ecc_refine(warped)
                # 粗搜只给排名前两的起点：后面的候选就算拉回来，
                # 也很少超过已经试过的更好起点，却会再付 1–2 次 ECC。
                if float(ecc.get("cc", 0.0)) < 0.80 and k < 2:
                    # ECC 没收敛好：先用粗搜把它拉回正确盆地，再精配一次。
                    # 注意图像与分数必须成对替换——曾经只换了图不换分，
                    # 导致候选之间拿"旧图的分数"比"新图的画质"。
                    base_warped, base_cc = warped, float(ecc.get("cc", 0.0))
                    rec_warped, ms = self._recover_alignment(warped)
                    if ms > 0.3:
                        rec_warped, ecc2 = self._ecc_refine(rec_warped)
                        if float(ecc2.get("cc", 0.0)) > base_cc:
                            warped, ecc = rec_warped, ecc2
                        else:
                            warped, ecc = base_warped, ecc
            cc = float(ecc.get("cc", 0.0))
            if best is None or cc > best["cc"]:
                best = {"ok": True, "warped": warped, "H": Hm, "ecc": ecc,
                        "cc": cc, "tries": k + 1}
            # 与解码入口的 CC_GATE=0.85 对齐：到门限即可停，不必追到 0.88
            if cc >= 0.85:
                break
        # 小尺寸实拍：边缘恢复后再交给宽度测量
        sc = self._logo_scale(best["H"])
        best["logo_scale"] = sc
        best["warped"] = self._restore_edges(best["warped"], sc)
        best.pop("cc", None)
        return best

    @staticmethod
    def _plate_candidates(gray: np.ndarray, k: int = 3) -> List[np.ndarray]:
        """定位候选：把"亮底卡片"的几种可能都裁出来（按面积从大到小）。

        真实构图里 logo 可能是：整幅就是底版、白底标签贴在深色产品/屏幕上、
        画面里还有别的亮区（窗口、台面）。这里取最大的 k 个亮连通域，
        各自填掉内孔、域外置白、裁出包围盒，交给解码端逐个尝试。
        """
        m = (gray > 170).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            return []
        order = sorted(range(1, n),
                       key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max(k, 1)]
        out: List[np.ndarray] = []
        for i in order:
            x, y, w, h, a = stats[i]
            if a < 1500 or h < 20:
                continue
            ar = w / float(h)
            if not (2.0 < ar < 6.0):       # 底版宽高比 3.67
                continue
            plate = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(plate, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            filled = np.zeros_like(plate)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            img = gray.copy()
            img[filled == 0] = 255
            out.append(img[y0:y1, x0:x1])
        return out
