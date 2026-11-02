"""ARROW 隐形水印 —— 换载体实验：以「内孔轮廓」为信息载体。

与 `arrow_wm.py`（边缘宽度调制）的区别只有一处：信息写在哪里。

    边缘版：调制字母**外轮廓**的笔画宽度 -> 外轮廓出现 ±MOD_DEPTH（默认 6px）波浪
    内孔版：外轮廓**一个像素都不动**，只调制 4 个内孔（O 内环、两个 R 字怀、
            A 上三角）的边界半径 -> 波浪被关进字母内部

容量：4 个内孔周长合计 2525 px，140 个码元 -> 平均每格 18 px。

复用 `arrow_wm` 的：码字编码（Hamming 7,4）、仿真器、配准/矫正链路。
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import arrow_wm as A

DEPTH: float = 8.0          # 内孔边界径向峰值位移（px @ 底版尺度，约 0.080 mm）
# 深度选择记录（早期 v3 数据，同批载荷/扰动）：
#   6.0 px：extreme 正常 6/12、BER 4.55%；A 三角局部模糊 5/12
#   8.0 px：extreme 正常 8/12、BER 3.25%；模糊 5/12
#  10.0 px：extreme 正常 8/12、BER 2.60%；模糊 7/12
# 2026-11-03 定为 **±8**（±0.080 mm）。依据（docs 10.8）：
#   v4 的码元周期是 v3 的两倍（0.328 mm），本来就比原来显眼；±8 把起伏幅度
#   降 20%，而代价只有 1/72：`bench_depth.py` 上 ±10 = 66/72、±8 = 65/72，
#   真机打印件实测幅度比 0.86（设计 0.80），两块 margin 仍有 3.88 / 4.02。
#   也就是说可见度明显下降，裕度几乎没损失。
# 打印 1200 dpi 时 ±8 ≈ 3.8 个墨点；凑近细看仍能看出边界微波。
N_THETA: int = 1440         # 每个内孔的圆周采样数
SCAN_HALF: float = 26.0     # 半径搜索半宽（px，矫正域）。见下面的「自适应采样窗」
SCAN_HALF_MIN: float = 16.0
SCAN_HALF_MAX: float = 40.0
SCAN_STEP: float = 0.25     # 半径搜索步长（px）

# ---- 自适应采样窗 --------------------------------------------------------
# 旧实现固定 SCAN_HALF=14，并按「窗口首 3 个样本 / 末 3 个样本」的中位数估计
# 背景与墨色电平。这只在「边缘比窗口窄」时成立，也就是干净印版（logo 占满
# 画面）。实拍时 logo 只占底版宽度的 10–20%，矫正回 2200 px 等于放大 5–10 倍，
# 相机域的失焦 σ 被同步放大 1/scale 倍：scale=0.13、σ=1.6 px 时矫正域的
# 边缘斜坡宽达 ±40 px，整条斜坡把 ±14 的窗口铺满。
# 后果：bright/dark 取到的是斜坡上的两个点而不是背景/墨色电平，阈值随调制
# 位置一起漂移 -> 幅度被压缩到标称值的 20–40%，且位移越大压得越狠（系统性
# 偏小）。真实尺度压测里 53% 的通过率主要丢在这里。
# 现在：窗宽按 logo_scale 自适应（覆盖 depth + 2.6σ 的完整斜坡），边缘位置
# 改用梯度加权质心（匹配滤波），不再依赖阈值。
BLUR_SIGMA0: float = 1.6    # 相机域典型失焦 σ（px），由五张实拍图 10–90% 边宽标定
SCAN_SIGMAS: float = 2.2    # 窗半宽 = depth + SCAN_SIGMAS × 矫正域 σ
# 质心权重的高斯宽度（相对窗半宽）。权重比窗窄，才能把窗口边缘的第二条边
# （细笔画处的外轮廓）挡在质心之外；窗本身要够宽，否则整条斜坡进不来。
SCAN_WEIGHT_FRAC: float = 0.55

# ---- 线性光输入 ----------------------------------------------------------
# 实测动机（`exp_linear.py`）：边缘质心假设斜坡对称，而相机输出是 gamma 编码的，
# 物理阶跃在 gamma 域不对称，质心系统性偏亮侧。线性化后斜坡对称。
# 注意必须**输入与模板同时线性化**，否则配准与半径参考错位。
LINEAR_INPUT: bool = False

# ---- 每孔独立切向相位恢复 ------------------------------------------------
# 假设：全局单应（SIFT+RANSAC）在四个孔之间留下各自的切向残差，偏半个码元
# 时匹配滤波输出乘 cos(π·Δ) 而归零。
# **实测证否**（`exp_phase_search.py`，真机打印页 6 块）：
#   · 开/关都是 5/6 正确，但裕度中位 3.85 → 3.74，**略差**；
#   · 选中的相位全部卡在搜索边界 ±0.40，说明能量准则在低信噪比下追的是噪声，
#     不是在找真峰；
#   · 把标块整体旋转 1.5°/3°/4.5°（≈0.17/0.33/0.50 个码元）几乎不影响结果
#     （5/6、裕度 3.72–3.77），说明**全局配准已经把旋转/平移吸收掉了**，
#     没有独立的切向时钟误差可恢复。
# 结论：这个机制在当前解码器里不成立（"码元时钟"和几何同源）。
# 保留代码与开关是为了让后来的人不必重做这个实验——默认关闭。
PHASE_SEARCH: bool = False
PHASE_SEARCH_HALF: float = 0.40     # 搜索范围 ±0.40 码元
PHASE_SEARCH_STEPS: int = 17        # 步数（步长 0.05 码元）

# ---- 版式 ----------------------------------------------------------------
# v2：140 码元 / 70 bit（37 bit 载荷 + 3 bit 固定补零），组内码元按孔顺序
#     连续排列，一组往往整组落在同一个孔上。
# v3：154 码元 / 77 bit（37 bit 载荷 + 7 bit CRC），11 组 (7,4)，码元按孔
#     轮流交织，每组覆盖 3–4 个孔；CRC 取代固定补零。
# 交织的意义：局部缺陷（A 三角失焦、纸张压痕、反光）不再让某一组 7 位全废，
# 而是摊成"每组错 1 位"，落在 Hamming 的纠错能力内。CRC-7 同时把"自信地
# 选错"的读数挡在门外（漏检率 1/128）。
LAYOUT_V2 = 2
LAYOUT_V3 = 3
# v4（实验）：**NRZ 调制**——一个码元直接承载一个 bit，不再用「一宽一窄」差分。
# 动机：v3 的差分对把信号能量全部搬到最高空间频率 1/(2L) 上，而高斯模糊的
# MTF = exp(-(π²/2)(σ/L)²) 在这个频率上衰减最狠。NRZ 让码元数从 154 降到 77、
# 每码元弧长翻倍（0.164 → 0.328 mm），σ/L 减半 → MTF 从 ~4% 升到 ~30%（+9 dB）。
# 代价与对策（见各常量注释）：
#   · 失去直流免疫 → 沿周长高通的去趋势（`V4_TREND_WIN`）
#   · 长游程会落在去趋势的阻带里 → 固定加扰（`V4_SCRAMBLE`）
# 编码、CRC-7、Hamming(7,4) **完全复用 v3**（37+7=44 bit → 11 组 → 77 bit = 77 码元）。
LAYOUT_V4 = 4
# 生产唯一版式。v2/v3 印版已停产，保留其构造代码只是为了让历史回归脚本还能跑，
# 任何产品路径（服务端、出图、默认值）都只走 v4。
DEFAULT_LAYOUT = LAYOUT_V4
N_CELLS_V2 = A.N_CELLS          # 140
N_BITS_V2 = A.N_BITS            # 70
N_CELLS_V3 = 154                # 77 bit = 11 组 × 7
N_BITS_V3 = 77
N_BITS_V4 = 77                  # NRZ：1 bit = 1 码元
N_CELLS_V4 = N_BITS_V4
# 去趋势的滑动平均窗口（码元）。3 点均值在 f=1/(2L) 处增益 1.33、在 f→0 处归零，
# 正好是高通且不伤信号带；5 点会开始削低 1010 这类最坏码型，默认 3。
V4_TREND_WIN = 3
# 固定加扰序列：打散长游程（否则连续同号码元会被去趋势当成趋势抹掉）。
# 只翻转「哪个码元画成正偏移」，解码端按同一位反号，CRC 仍然兜底。
_V4_RNG = np.random.default_rng(20260311)
V4_SCRAMBLE: Tuple[int, ...] = tuple(
    int(b) for b in _V4_RNG.integers(0, 2, N_BITS_V4))
CRC7_POLY = 0x09                # x^7 + x^3 + 1


# ==============================================================================
# 0. v3 码字：37 bit 载荷 + 7 bit CRC -> 44 数据位 -> 11 组 Hamming(7,4)
# ==============================================================================
def crc7(bits: Sequence[int]) -> List[int]:
    """CRC-7（多项式 x^7 + x^3 + 1），返回 7 个校验位（高位在前）。"""
    reg = 0
    mask = (1 << 7) - 1
    for b in bits:
        reg = ((reg << 1) | (int(b) & 1)) & mask
        if reg & (1 << 6):
            reg ^= CRC7_POLY
    return [(reg >> (6 - i)) & 1 for i in range(7)]


def payload_bits_v3(digits: str) -> List[int]:
    """11 位号码 -> 44 个数据位（37 载荷 + 7 CRC）。"""
    p37 = A.payload_to_bits(digits)
    return p37 + crc7(p37)


def encode_payload_v3(digits: str) -> List[int]:
    """11 位号码 -> 77 bit 码字。"""
    code = A.hamming74_encode(payload_bits_v3(digits))
    assert len(code) == N_BITS_V3, len(code)
    return code


def decode_payload_v3_soft(soft: Sequence[float]) -> Tuple[Optional[str], bool, int]:
    """软判决逐组最大似然 + CRC 校验。

    返回 ``(号码, CRC 是否通过, 与硬判决相差的位数)``。
    CRC 不过时号码仍是研究读数，生产门不得把它当号码放出去。
    """
    cb = A.hamming_codebook()
    sgn = cb.astype(np.float64) * 2.0 - 1.0
    soft = np.asarray(soft, np.float64).ravel()
    if soft.size != N_BITS_V3:
        return None, False, 0
    hard = (soft > 0).astype(np.int8)
    data: List[int] = []
    n_chg = 0
    for g in range(N_BITS_V3 // 7):
        sl = slice(g * 7, (g + 1) * 7)
        code = cb[int(np.argmax(sgn @ soft[sl]))]
        data += [int(code[2]), int(code[4]), int(code[5]), int(code[6])]
        n_chg += int(np.sum(code != hard[sl]))
    p37, crc = data[:37], data[37:44]
    return A.bits_to_payload(p37), crc7(p37) == crc, n_chg


# ==============================================================================
# 1. 载体：内孔轮廓
# ==============================================================================
class InnerCarrier:
    """提取内孔、按周长分配码元、把 70 bit 写成内孔边界的径向微波。"""

    def __init__(self, design: Optional[A.ArrowDesign] = None,
                 depth: float = DEPTH, verbose: bool = False,
                 layout_version: int = DEFAULT_LAYOUT):
        self.design = design or A.ArrowDesign()
        self.depth = float(depth)
        layout_version = int(layout_version)
        if layout_version not in (LAYOUT_V2, LAYOUT_V3, LAYOUT_V4):
            raise ValueError(f"layout_version 只能是 2 / 3 / 4，收到 {layout_version}")
        self.layout_version = layout_version
        if layout_version == LAYOUT_V4:
            self.n_cells, self.n_bits = N_CELLS_V4, N_BITS_V4
        elif layout_version == LAYOUT_V3:
            self.n_cells, self.n_bits = N_CELLS_V3, N_BITS_V3
        else:
            self.n_cells, self.n_bits = N_CELLS_V2, N_BITS_V2
        self.SS = A.SS
        self.theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
        self.holes = self._extract_holes()
        self._alloc_cells()
        self.pair_order = self._build_pair_order()
        if verbose:
            tot = sum(h["peri"] for h in self.holes)
            print(f"[InnerCarrier] {len(self.holes)} 个内孔，总周长 {tot:.0f} px @1x "
                  f"layout=v{self.layout_version} {self.n_cells} 码元")
            for i, h in enumerate(self.holes):
                print(f"  #{i} 周长 {h['peri'] / self.SS:6.1f}px  "
                      f"{h['n_cells']:2d} 码元  "
                      f"每格 {h['peri'] / self.SS / h['n_cells']:4.1f}px  "
                      f"圆心 ({h['c'][0] / self.SS:.0f},{h['c'][1] / self.SS:.0f})")

    # ---------------- 内孔提取 ----------------
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

    # ---------------- 码元分配 ----------------
    def _alloc_cells(self) -> None:
        n_tot = self.n_cells
        per = np.array([h["peri"] for h in self.holes], np.float64)
        raw = per / per.sum() * n_tot
        if self.layout_version == LAYOUT_V4:
            # NRZ 是 1 码元/bit，不必成对，允许奇数配额
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

        alloc = np.maximum(2, (np.floor(raw / 2.0) * 2).astype(int))
        # 用最大余数法补齐到 n_tot（每次 +2）
        while alloc.sum() < n_tot:
            score = raw - alloc
            alloc[int(np.argmax(score))] += 2
        while alloc.sum() > n_tot:
            cand = [i for i in range(len(alloc)) if alloc[i] > 2]
            if not cand:
                break
            score = np.array([alloc[i] - raw[i] for i in range(len(alloc))])
            for i in range(len(alloc)):
                if alloc[i] <= 2:
                    score[i] = -1e9
            alloc[int(np.argmax(score))] -= 2
        for h, k in zip(self.holes, alloc):
            h["n_cells"] = int(k)
            h["dtheta"] = 2.0 * np.pi / k

        # 全局码元表：先孔、后角度
        self.cells: List[Tuple[int, int]] = []
        for hi, h in enumerate(self.holes):
            for k in range(h["n_cells"]):
                self.cells.append((hi, k))
        assert len(self.cells) == n_tot, f"码元数 {len(self.cells)} != {n_tot}"

    def _build_pair_order(self) -> Tuple[int, ...]:
        """bit -> 码元对（扁平索引）的映射。

        v2 为恒等映射（组内连续，容易整组落在同一个孔）。
        v3 按孔轮流取对，使每组 7 bit 覆盖 3–4 个孔，把局部缺陷摊开。
        v4 是 NRZ，没有「对」的概念，恒等映射；跨孔交织靠 `_alloc_cells`
        的配额均分 + `V4_SCRAMBLE` 打散。
        """
        if self.layout_version == LAYOUT_V4:
            return tuple(range(N_BITS_V4))
        if self.layout_version == LAYOUT_V2:
            return tuple(range(N_BITS_V2))
        per: List[List[Tuple[int, int]]] = [
            [(hi, k) for k in range(h["n_cells"] // 2)]
            for hi, h in enumerate(self.holes)
        ]
        # 按「配额均衡」排孔序：每次选 (已用+1)/配额 最小的孔。
        # 这比简单轮转更均匀——轮转会把长孔的余量全堆到序列尾部，
        # 末尾几组又变回只覆盖一个孔。
        counts = [len(p) for p in per]
        used = [0] * len(per)
        seq: List[int] = []
        for _ in range(sum(counts)):
            best, best_key = -1, None
            for hi in range(len(per)):
                if used[hi] >= counts[hi]:
                    continue
                key = ((used[hi] + 1.0) / counts[hi], hi)
                if best_key is None or key < best_key:
                    best, best_key = hi, key
            seq.append(best)
            used[best] += 1
        cursor = [0] * len(per)
        order: List[Tuple[int, int]] = []
        for hi in seq:
            order.append(per[hi][cursor[hi]])
            cursor[hi] += 1
        index: Dict[Tuple[int, int], int] = {}
        off = 0
        for hi, h in enumerate(self.holes):
            for k in range(h["n_cells"] // 2):
                index[(hi, k)] = off + k
            off += h["n_cells"] // 2
        perm = tuple(index[p] for p in order)
        if sorted(perm) != list(range(self.n_bits)):
            raise RuntimeError("PAIR_ORDER 不是 77 长置换")
        return perm

    # ---------------- 每个码元的平滑窗 ----------------
    def _window(self, hi: int, k: int) -> np.ndarray:
        h = self.holes[hi]
        th = self.theta
        t = ((th - k * h["dtheta"]) % (2.0 * np.pi)) / h["dtheta"]
        inside = t < 1.0
        return np.where(inside, np.sin(np.pi * np.clip(t, 0.0, 1.0)), 0.0)

    def _delta(self, bits: List[int]) -> List[np.ndarray]:
        """返回每个内孔的径向位移场 delta(theta)。"""
        fields = [np.zeros(N_THETA) for _ in self.holes]
        if self.layout_version == LAYOUT_V4:
            # NRZ：一个码元一个 bit，直接 ±depth；加扰在这里生效
            for i, b in enumerate(bits):
                hi, k = self.cells[i]
                sgn = 1.0 if (int(b) ^ V4_SCRAMBLE[i]) else -1.0
                fields[hi] += sgn * self.depth * self._window(hi, k)
            return fields
        for i, b in enumerate(bits):
            ka, kb = 2 * i, 2 * i + 1
            hi_a, k_a = self.cells[ka]
            hi_b, k_b = self.cells[kb]
            sgn = 1.0 if b == 1 else -1.0
            fields[hi_a] += sgn * self.depth * self._window(hi_a, k_a)
            fields[hi_b] -= sgn * self.depth * self._window(hi_b, k_b)
        return fields

    def solid_mask4(self) -> np.ndarray:
        """把内孔填满后的 4x 掩膜 —— 只保留字母外轮廓。"""
        m = self.design.mask4.copy()
        for h in self.holes:
            cv2.fillPoly(m, [h["poly_orig"]], 255)
        return m

    def solid_template(self) -> np.ndarray:
        return self.design.render(self.solid_mask4())

    # ---------------- 嵌入 ----------------
    def embed(self, digits: str) -> Dict:
        if self.layout_version in (LAYOUT_V3, LAYOUT_V4):
            bits = encode_payload_v3(digits)
            arranged = [0] * self.n_bits
            for b, pi in enumerate(self.pair_order):
                arranged[pi] = bits[b]
        else:
            bits = A.encode_payload(digits)
            arranged = bits
        mask4 = self.design.mask4.copy()
        fields = self._delta(arranged)
        for hi, h in enumerate(self.holes):
            # 先把原孔整块填回墨色，再按调制后的多边形重新挖空
            cv2.fillPoly(mask4, [h["poly_orig"]], 255)
            rr = h["r_tab"] + fields[hi] * self.SS
            rr = np.maximum(rr, 4.0)
            pts = np.stack([h["c"][0] + rr * self.cos_t,
                            h["c"][1] + rr * self.sin_t], axis=1)
            cv2.fillPoly(mask4, [np.round(pts).astype(np.int32)], 0)
        plate = self.design.render(mask4)
        return {"digits": digits, "bits": bits, "plate": plate, "mask4": mask4,
                "layout_version": self.layout_version}


# ==============================================================================
# 2. 解码器：内孔边界亚像素半径 + 匹配滤波
# ==============================================================================
class InnerDecoder:
    def __init__(self, carrier: InnerCarrier, use_ecc_refine: bool = True,
                 solid_reg: bool = False):
        self.carrier = carrier
        self.design = carrier.design
        # 复用边缘版的配准/矫正链路（自同步 + ECC 精配 + 粗搜抢救）
        # 真实尺寸优化（跳过高分 SIFT、粗搜两阶段、小 logo 边缘恢复）也在这条链路上。
        self._reg = A.WatermarkDecoder(self.design, use_ecc_refine=use_ecc_refine)
        if solid_reg:
            # 实测：填孔后 ECC 反而更差（17/20 -> 14/20）。内孔是锐利且分布均匀的
            # 强特征，填掉等于自废配准基准。默认关闭，仅留作对照开关。
            self._use_solid_tmpl()
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
        # 码元窗预计算：measure 里原先每个码元每次现场算 sin 窗，140 次重复
        self._win = [self.carrier._window(hi, k) for hi, k in self.carrier.cells]
        self._win_den = np.array([max(float((w * w).sum()), 1e-6) for w in self._win],
                                 np.float32)
        self._build_phase_search()
        # 每孔最近一次选中的相位偏移（单位：码元），供诊断输出
        self.phase_shift = [0.0] * len(self.carrier.holes)
        if LINEAR_INPUT:
            self._apply_linear()

    def _apply_linear(self) -> None:
        """把参考模板与配准模板都换成线性光版本。

        只线性化输入、不线性化模板，会让 ECC 配准与 `_ref_radii` 都错位——
        参考必须和测量在同一个光度量纲里。
        """
        lin = A.linearize_u8(self.design.template)
        self._tmpl_f = lin.astype(np.float32)
        r = self._reg
        t = lin.astype(np.float32) / 255.0
        r.tmpl = t
        th, tw = r._tmpl_half.shape
        r._tmpl_half = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        qs = r._quad_scale
        tq = cv2.resize(t, (int(round(A.CANVAS_W * qs)),
                            int(round(A.CANVAS_H * qs))),
                        interpolation=cv2.INTER_AREA)
        r._tmpl_q = tq
        v = tq.ravel() - tq.mean()
        r._tmpl_vec = v
        r._tmpl_norm = float(np.linalg.norm(v))
        self._r_ref = self._ref_radii()

    def _build_phase_search(self) -> None:
        """把每个孔的码元窗堆成一个矩阵，供相位搜索一次矩阵乘算完。

        投影 ``(delta · w_j) / ||w_j||`` 对每孔是一个 n×N_THETA 的矩阵乘；
        除以 ``||w_j||`` 是让长窗短窗贡献可比（能量准则要在归一化后比较）。
        """
        self._phase_win: List[np.ndarray] = []
        off = 0
        for h in self.carrier.holes:
            n = h["n_cells"]
            w = np.stack([self._win[off + j] for j in range(n)]).astype(np.float32)
            w /= np.sqrt(np.maximum(self._win_den[off:off + n], 1e-6))[:, None]
            self._phase_win.append(w)
            off += n
        # 位移量单位：码元 -> 射线数。每个码元占 N_THETA/n 条射线。
        self._phase_ray_per_cell = [N_THETA / float(h["n_cells"])
                                    for h in self.carrier.holes]
        if PHASE_SEARCH_STEPS > 1:
            self._phase_grid = np.linspace(-PHASE_SEARCH_HALF, PHASE_SEARCH_HALF,
                                           PHASE_SEARCH_STEPS)
        else:
            self._phase_grid = np.zeros(1)

    @staticmethod
    def _circ_shift(v: np.ndarray, shift: float) -> np.ndarray:
        """沿 θ 做**分数**循环平移（线性插值，端点环绕）。"""
        n = v.size
        if abs(shift) < 1e-9:
            return v
        idx = (np.arange(n, dtype=np.float64) - shift) % n
        i0 = np.floor(idx).astype(np.int64) % n
        frac = (idx - np.floor(idx)).astype(np.float32)
        return v[i0] * (1.0 - frac) + v[(i0 + 1) % n] * frac

    def _best_phase(self, hi: int, delta: np.ndarray) -> np.ndarray:
        """在该孔上搜一个让匹配滤波能量最大的切向偏移。"""
        if not PHASE_SEARCH or PHASE_SEARCH_STEPS <= 1:
            self.phase_shift[hi] = 0.0
            return delta
        w = self._phase_win[hi]
        per_cell = self._phase_ray_per_cell[hi]
        best_d, best_e, best_s = delta, -1.0, 0.0
        for s in self._phase_grid:
            d = self._circ_shift(delta, float(s) * per_cell)
            e = float(np.square(w @ d).sum())
            if e > best_e:
                best_d, best_e, best_s = d, e, float(s)
        self.phase_shift[hi] = best_s
        return best_d

    def _use_solid_tmpl(self) -> None:
        r = self._reg
        solid = self.carrier.solid_template().astype(np.float32) / 255.0
        r.tmpl = solid
        th, tw = r._tmpl_half.shape
        r._tmpl_half = cv2.resize(solid, (tw, th), interpolation=cv2.INTER_AREA)
        qs = r._quad_scale
        tq = cv2.resize(solid, (int(round(A.CANVAS_W * qs)),
                                int(round(A.CANVAS_H * qs))),
                        interpolation=cv2.INTER_AREA)
        r._tmpl_q = tq
        v = tq.ravel() - tq.mean()
        r._tmpl_vec = v
        r._tmpl_norm = float(np.linalg.norm(v))

    # ---------------- 采样网格 ----------------
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
        contrast = (np.percentile(sm, 95, axis=1) -
                    np.percentile(sm, 5, axis=1))
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
            # 每孔独立切向相位恢复：偏半个码元信号会归零，必须先对齐再投影
            delta = self._best_phase(hi, delta)
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
        if LINEAR_INPUT:
            gray = A.linearize_u8(gray)
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
        """在已矫正的画面上测量与判决。

        自动版式解码时 v2/v3 共用同一次 rectify：配准只取决于设计稿，
        与码元版式无关。把这一步抽出来，旧印版不必再付一次配准。
        """
        # 采样窗按画面尺度自适应：小 logo 的照片矫正后模糊被放大，窗口要跟着变宽
        self._ensure_scale(r.get("logo_scale"))
        feat = self.measure(r["warped"])
        ecc = dict(r.get("ecc", {}))
        ecc.setdefault("cc", 0.0)
        if self.carrier.layout_version == LAYOUT_V4:
            # NRZ：feat 直接就是每个 bit 的软值（无差分）；先去掉低频趋势（补回
            # 直流免疫），再按加扰序列反号，最后交给与 v3 完全相同的 Hamming+CRC。
            conf = self._detrend(feat)
            conf_a = np.array(
                [conf[pi] * (-1.0 if V4_SCRAMBLE[pi] else 1.0)
                 for pi in self.carrier.pair_order], np.float32)
            bits = [1 if c > 0 else 0 for c in conf_a]
            digits, crc_ok, n_chg = decode_payload_v3_soft(conf_a)
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
                    "layout_version": LAYOUT_V4,
                    "crc_ok": bool(crc_ok), "pad_bits": None,
                    "margin": margin, "margin_mean": margin,
                    "margin_min": float(ms.min()) if ms.size else 0.0,
                    "group_margins": [float(x) for x in ms]}
        if self.carrier.layout_version == LAYOUT_V3:
            # 码元对 -> 逻辑 bit 顺序（交织的逆映射）
            dif = np.array([feat[2 * i] - feat[2 * i + 1]
                            for i in range(N_BITS_V3)], np.float64)
            conf_a = np.array([dif[pi] for pi in self.carrier.pair_order],
                              np.float32)
            bits = [1 if c > 0 else 0 for c in conf_a]
            digits, crc_ok, n_chg = decode_payload_v3_soft(conf_a)
            hard40, _, _ = A.hamming74_decode(bits)
            digits_hard = A.bits_to_payload(hard40[:37])
            ms = A.group_codeword_margins(conf_a)
            margin = float(ms.mean()) if ms.size else 0.0
            out = {"ok": True, "digits": digits, "digits_hard": digits_hard,
                   "payload_valid": digits is not None,
                   "bits": bits, "feat": feat, "conf": conf_a,
                   "mean_conf": float(np.mean(np.abs(conf_a))) * 0.5,
                   "corrected": n_chg, "n_corrected": n_chg,
                   "ecc": ecc, "cc": float(ecc.get("cc", 0.0)),
                   "logo_scale": float(r.get("logo_scale", 1.0)),
                   "warped": r["warped"] if return_debug else None,
                   "layout_version": LAYOUT_V3,
                   "crc_ok": bool(crc_ok),
                   "pad_bits": None,
                   "margin": margin, "margin_mean": margin,
                   "margin_min": float(ms.min()) if ms.size else 0.0,
                   "group_margins": [float(x) for x in ms]}
        else:
            bits = []
            conf = []
            for i in range(N_BITS_V2):
                d = float(feat[2 * i] - feat[2 * i + 1])
                bits.append(1 if d > 0 else 0)
                conf.append(d)
            conf_a = np.array(conf, np.float32)
            # 软判决最大似然：硬判决在一组内出现 2 位错时会误纠，软判决能救回
            digits, n_chg = A.decode_payload_soft(conf_a)
            hard, corrected, _ = A.decode_payload(bits)
            out = {"ok": True, "digits": digits, "digits_hard": hard,
                   "payload_valid": digits is not None,
                   "bits": bits, "feat": feat, "conf": conf_a,
                   "mean_conf": float(np.mean(np.abs(conf_a))) * 0.5,
                   "corrected": n_chg, "n_corrected": n_chg,
                   "ecc": ecc, "cc": float(ecc.get("cc", 0.0)),
                   "logo_scale": float(r.get("logo_scale", 1.0)),
                   "warped": r["warped"] if return_debug else None,
                   "layout_version": LAYOUT_V2, "crc_ok": None}
            out.update(A.readout_qc(conf_a, bits, soft=True))
        if return_debug:
            out["H"] = r.get("H")
        return out

    def decode(self, image: np.ndarray, return_debug: bool = False) -> Dict:
        """内孔版解码入口：与边缘版同一套真实构图策略。

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


# ==============================================================================
# 3. 命令行：压测 / 导出印版 / 解码实拍
# ==============================================================================
def run_bench(carrier: InnerCarrier, n: int = 20, seed: int = 6000,
              levels=("clean", "light", "moderate", "heavy", "extreme"),
              stress: bool = False) -> None:
    dec = InnerDecoder(carrier)
    sim = A.CeramicCamSimulator()
    import arrow_decode as AD
    print("=" * 88)
    print(f"  内孔载体压测（生产门）  layout=v{carrier.layout_version}  "
          f"{carrier.n_cells} 码元  depth={carrier.depth:g}px  "
          f"{'压力模式  ' if stress else ''}每组 {n} 样本")
    print("=" * 88)
    print(f"| {'档位':<12} | {'研究正确':>8} | {'接受且对':>8} | {'接受且错':>8} | "
          f"{'BER':>8} | {'耗时':>7} |")
    print("|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|"
          + "-" * 10 + "|" + "-" * 9 + "|")
    cc_thr = AD.effective_cc_threshold(
        AD.CC_THRESHOLD, bool(dec._reg.use_ecc_refine))
    for lv in levels:
        research_ok = accept_ok = accept_wrong = 0
        bers, ts = [], []
        for sd in range(n):
            dg = "1%010d" % ((sd + seed) * 7919 % 10 ** 10)
            wm = carrier.embed(dg)
            tb = np.array(wm["bits"])
            cam = sim.simulate(wm["plate"], lv, np.random.default_rng(seed + sd),
                               stress=stress)
            t0 = __import__("time").time()
            r = dec.decode(cam["image"])
            ts.append(__import__("time").time() - t0)
            b = np.array(r["bits"]) if r["ok"] else np.zeros(carrier.n_bits, int)
            bers.append(float(np.mean(b != tb)) * 100.0)
            research_ok += int(r.get("digits") == dg)
            gated = AD.apply_gate(r, carrier="inner", cc_threshold=cc_thr)
            if gated.accepted and gated.digits == dg:
                accept_ok += 1
            elif gated.accepted:
                accept_wrong += 1
        print(f"| {lv:<12} | {research_ok:>3d}/{n:<3d} | {accept_ok:>3d}/{n:<3d} | "
              f"{accept_wrong:>3d}/{n:<3d} | {np.mean(bers):>7.2f}% | "
              f"{np.mean(ts):>6.2f}s |")
    print("=" * 88)


def export_plates(carrier: InnerCarrier, out_dir: str,
                  payloads=("13800138000", "13912345678", "18600001234",
                            "01088886666", "17701234567")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dec = InnerDecoder(carrier)
    A.imwrite_u(os.path.join(out_dir, "ARROW_无码对照.png"), carrier.design.template)
    print(f"{'号码':<14}{'文件':<30}{'自检':>8}{'置信度':>9}")
    print("-" * 64)
    for dg in payloads:
        wm = carrier.embed(dg)
        name = f"ARROW_{dg}.png"
        A.imwrite_u(os.path.join(out_dir, name), wm["plate"])
        A.imwrite_u(os.path.join(out_dir, f"ARROW_{dg}_1bit.png"),
                    np.where(wm["plate"] > 127, 255, 0).astype(np.uint8))
        r = dec.decode(wm["plate"])
        ok = "OK" if r.get("digits") == dg else "FAIL"
        print(f"{dg:<14}{name:<30}{ok:>8}{r.get('mean_conf', 0):>9.4f}")
    print("-" * 64)


def main(argv=None) -> int:
    import argparse
    import os as _os
    ap = argparse.ArgumentParser(description="ARROW 内孔载体隐形水印")
    ap.add_argument("--bench", action="store_true", help="跑扰动压测")
    ap.add_argument("--stress", action="store_true",
                    help="压测时叠加晕染/网点/镜头畸变/非平面（更接近实物）")
    ap.add_argument("-n", type=int, default=20, help="每档样本数")
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--depth", type=float, default=DEPTH, help="径向峰值位移 (px)")
    ap.add_argument("--export", metavar="DIR", default=None, help="导出送印图到目录")
    ap.add_argument("--decode", nargs="+", metavar="IMG", default=None,
                    help="解码实拍照片")
    ap.add_argument("--layout", choices=("2", "3", "auto"), default="auto",
                    help="内孔版式：3=交织+CRC-7（新印版），2=旧印版，"
                         "auto=解码时自动（压测/导出按 3）")
    args = ap.parse_args(argv)

    make_layout = LAYOUT_V3 if args.layout == "auto" else int(args.layout)
    carrier = InnerCarrier(depth=args.depth, verbose=True,
                           layout_version=make_layout)
    if args.bench:
        run_bench(carrier, n=args.n, seed=args.seed, stress=args.stress)
    if args.export:
        export_plates(carrier, args.export)
    if args.decode:
        import time as _time
        import arrow_decode as AD
        # 与边缘版同一套生产门：裕度均值、单组最小裕度、补零、cc。
        # 压测 run_bench 对 InnerDecoder 的结果再过这道门，并统计接受且错号。
        session = AD.Decoder(
            carrier="inner", design=carrier.design, depth=args.depth,
            verbose=False,
            layout_version=None if args.layout == "auto" else int(args.layout))
        print("=" * 78)
        print("  实拍照片解码（内孔载体，生产门限）")
        print("=" * 78)
        for path in args.decode:
            img = A.imread_u(path)
            if img is None:
                print(f"  {_os.path.basename(path):<30} -> 读不到文件")
                continue
            t0 = _time.time()
            r = session.decode(img)
            dt = (_time.time() - t0) * 1000
            path_tag = r.raw.get("path", "?")
            scale = f"{r.logo_scale:.2f}" if r.logo_scale is not None else "—"
            tail = (f"裕度均值 {r.margin_mean:.2f}  最小 {r.margin_min:.2f}  "
                    f"cc {r.cc:.3f}  scale {scale}  "
                    f"置信度 {r.mean_conf:.4f}  {dt:.0f}ms  path={path_tag}")
            if r.accepted:
                print(f"  {_os.path.basename(path):<30} -> {r.digits}   {tail}  接受")
            else:
                if r.candidate_digits:
                    extra = f"候选 {r.candidate_digits}（不可信）"
                elif r.ok:
                    extra = "候选 码字越界（不可信）"
                else:
                    extra = "无候选"
                print(f"  {_os.path.basename(path):<30} -> 拒绝（{AD.reason_text(r.reason)}）  "
                      f"{extra}   {tail}")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
