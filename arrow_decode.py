"""Production decode API for the ARROW watermark.

The research decoders in ``arrow_wm`` / ``arrow_inner`` always emit a digit
string once sync succeeds. This module is the integrator-facing wrapper:

* default carrier is the inner-hole watermark (``carrier="inner"``)
* ``carrier="edge"`` selects the edge watermark for legacy plates; its algorithm is unchanged
* inner layout defaults to ``auto``: decode v3 (interleaved, CRC-7) first and
  stop when the CRC passes; otherwise decode v2 (legacy plates, group-0 pad).
  Both layouts share one registration pass. ``layout_version=2`` or ``3``
  pins one layout.
* a hard gate sets ``accepted=False`` and ``digits=None`` unless the readout
  clears the mean codeword-margin threshold, the per-group minimum margin,
  the integrity check (v3 CRC-7, or the v2 group-0 pad ``000``), and, when
  ECC refinement is on, the registration ``cc`` gate.
  ``use_ecc_refine=False`` (``--no-ecc``) skips only the cc gate: that path
  does not produce a registration score (``cc`` stays 0). The margin gates
  and the other accept rules still apply. Default remains ECC on.

``margin`` / ``margin_mean`` is the average of the ten group gaps. One
wrong or tied group is hidden by that average, so ``margin_min`` is a
separate hard gate.

``candidate_digits`` on the Python object is the research decoder's raw
read. It is not a trusted phone number. The HTTP service omits it unless
debug mode and a valid token are both present.

Clients decide success only with ``accepted is True`` or
``status == "accepted"``. ``ok`` means the research decoder finished a
sync; it is true for unmarked plates that this gate rejects. ``status``
is ``accepted``, ``rejected``, or (HTTP layer only) ``error``.

Batch callers should reuse :class:`Decoder`. :func:`decode` builds a new
session on every call (template + SIFT init).
"""
from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Production default. Edge stays selectable for legacy plates and is not tuned here.
DEFAULT_CARRIER = "inner"
CARRIERS = ("edge", "inner")

# These cuts are z-scores of amplitude-normalized soft values (soft_n),
# not pixel offsets and not the old 1.8 px modulation depth.
# Edge decode divides the matched-filter differential by π·depth before
# the gap/std margin is computed. A global scale does not move the z-score,
# so the numeric cuts stay put when depth changes; per-bit depth_scale would.
# v2@6 was checked against the 1.8 mean cut (see README ablation).
# True codes still sit at or above ~2.0 and unmarked plates at or below ~1.4.
# The 1.8 cut is a watermark-present gate, not a digit-correct gate.
MARGIN_THRESHOLD = 1.8

# Hard gate on the worst Hamming group (best-vs-2nd gap / std of soft_n).
# A single near-tie group can leave the mean above 1.8. Confidently wrong
# groups (large gap on the wrong codeword) are not caught here; that needs
# a payload CRC, which is out of scope.
# The old 1.0 cut sat between clean-plate accepts (margin_min >= 1.24 on the
# pre-v2 ±6 px probe) and two wrong reads at 0.17 and 0.32. A real inner
# photo of 17701234567 reads that number with margin_min ≈ 0.604. The
# near-tie smoke (one group at amplitude 0.04) stays near 0.325, so 0.55
# still rejects that tie and the 0.17 / 0.32 wrong reads. Not a pixel cut.
MIN_GROUP_MARGIN = 0.55

# v3 有 CRC-7 兜底，单组裕度只作为弱先验：低尺度真机照常见「其余九组很强、
# 一组接近打平」的情况。实测 0.55 → 0.20 在 extreme 仿真里不减少正确数、
# 也不新增错号；真机 5 张里多救回 1 张（最弱组 0.21、CRC 通过）。
# 只在调用方没有显式改写阈值时生效。
MIN_GROUP_MARGIN_V3 = 0.20

# --suspect: absolute z is not enough. The closest one-digit neighbor is
# Hamming distance 3, so ideal soft info still scores that neighbor at
# about z - 0.72. Strong evidence needs the gap as well as z.
SUSPECT_Z_STRONG = 4.5
SUSPECT_Z_MAYBE = 3.0
SUSPECT_GAP_STRONG = 0.45
SUSPECT_GAP_MAYBE = 0.15
PAD_BITS = (0, 0, 0)

# Both research decoders already refuse to *prefer* a path whose ECC
# correlation is below 0.85, because a bad crop can inflate mean |conf|.
# The production gate treats that same cut as a hard reject.
CC_THRESHOLD = 0.85

# At ~0.08× the plate width the wave is a fraction of a camera pixel
# (±6 px at scale 0.077 is ±0.46 px); that documented tiny frame stays
# rejected. Extreme sims in this repo sit around 0.7–0.9, so the cut does
# not touch that bench. Two real inner photos of 17701234567 bracket the cut:
#   scale ≈ 0.160 -> correct, margin 6.07 (original 0.160 sample)
#   scale ≈ 0.131 -> correct, margin 2.82, margin_min 0.66, cc 0.976
# The 0.131 frame was rejected only by this cut, with every signal gate
# comfortably green. The product owner set the cut to 0.10 (from 0.15):
# at 0.10 the ±6 px modulation is ±0.60 camera px per side (1.20 px
# differential), and the margin / pad / cc gates still decide readability.
# 0.077 remains under the cut, so the documented tiny frame is still rejected.
# A frame under the cut is rejected even when registration cc is high.
# It does not relax the margin gates, and a larger digital upscale of the
# same pixels is a different image that still has to clear margin / pad.
#
# 2026-11-03 下调 0.10 → 0.08（仅内孔 v3/v4），依据是真机打印件实测
# （docs 第 10.6 节）：同一张纸 v3/v4 交替、22×6 mm、手机直出，
# `logo_scale`≈0.093 时 v3 margin 中位 3.39（6/8）、v4 3.91（7/8），CRC 全过；
# 受控降采样到 144 px 印版（ls≈0.056）时 v4 仍有 3.49、7/7。
#
# **为什么不像最初设想的那样一刀切降到 0.06**：ci_smoke 里有两个哨兵测试，
# 其中一个记录了 0.077 的**「高裕度但读错」**样本（margin 4.2、margin_min 3.0、
# 任何一个信号门都拦不住）。那道门就是为它设的。但注意那是 **edge 载体**——
# edge 只有 3 bit 固定补零（漏检 1/8），而 inner v3/v4 有 CRC-7 兜底。
# 所以按载体分开：
#   inner（有 CRC）：0.08 —— 既保住 0.077 那条拒收，又放行实测过的 0.088–0.093 段
#   edge（无真校验）：0.10 —— 维持原样，宁可少解不可错号
# v2 内孔仍走 LOGO_SCALE_MIN_V2=0.20（没有 CRC，尺度必须更大）。
LOGO_SCALE_MIN = 0.08
# 边缘载体沿用原来的 0.10：它没有真正的完整性校验，尺度门是主要的防错号手段。
LOGO_SCALE_MIN_EDGE = 0.10

# 旧内孔版式 v2（140 码元、3 bit 固定补零）没有真正的完整性校验：补零只有
# 3 bit 且位置固定，漏检率 1/8。实测两张 v2 印版实拍（logo_scale 0.13/0.16）
# 都把 17701234567 读成 17701234564——低位 2 bit 一起翻，裕度 2.1、最弱组
# 0.27–0.90，和正确读数完全无法区分，任何门限都拦不住。
# v2 的码元组是「整组落在同一个孔」的块交织，A 三角（最弱孔）独占第 8–9 组，
# 尺度一小那两组的 7 bit 就一起翻。所以 v2 只在画面足够大、码元采样远高于
# 奈奎斯特时才是可信的：仿真里 0.20 以上（每码元 3.3 相机像素）两种模糊都读对，
# 0.12–0.16 在 blur=1.1 时稳定读错。低于该尺度一律拒绝，宁可少解不错解。
# 需要强行解旧印版时传 allow_legacy_v2=True。
LOGO_SCALE_MIN_V2 = 0.20

# 首解被拒时的兜底：换尺度再解一次。只对 v3（CRC 可裁决）启用；
# 成功路径不付代价，只有拒绝路径多付最多三次解码。
# 实测：一张 logo_scale=0.135 的 v3 实拍在 1.0× 下 CRC 差 2 bit（读对了号码
# 但 CRC 位本身被纠错），1.35×–2.2× 全部 CRC 通过且号码相同。多尺度是
# 唯一不带统计风险的救回手段——裁决权始终在 CRC 手里。
RETRY_SCALES = (1.35, 1.7, 0.8)
RETRY_MAX_SIDE = 6000

# 首解被拒时的第二类兜底：**反卷积重试**。
# 缩放重试只改采样网格，改不了模糊；而失焦是低尺度下的主导失效因子
# （相机 σ=1.2 px 时 0.13 档 9/12，σ=2.0 px 时 0/12）。失焦就是一次已知
# 形式的卷积，所以用固定 PSF 做 Richardson-Lucy 再解一次。
# 为什么只挂在「拒绝」路径上：无条件预处理会在轻失焦时过冲（实测把 0.16 档
# 11/12 压到 9/12）。挂在重试上就只增不减，且裁决权仍归 CRC-7——
# 与缩放重试同一条安全规则：必须 v3、必须 CRC 通过、必须与首解候选同号。
RETRY_DECONV_SIGMAS = (2.0,)
RETRY_DECONV_ITERS = 12

ImageLike = Union[np.ndarray, str, os.PathLike]

# ---- 共享构造缓存 ----------------------------------------------------------
# ArrowDesign 构造一次约 200 ms（栅格化 + 布局推导），Decoder 还要再花
# 250–500 ms 建模板 / SIFT / 内孔映射。一次性接口 decode() 原先每次调用都
# 重建一遍，单次调用里 3/4 的时间花在这上面。设计对象构造后只读，可以安全
# 复用；会话本身有内部状态（OpenCV matcher 不做重入保证），所以缓存里同时
# 存一把锁，同一会话的并发调用仍然串行。
_DESIGN_CACHE_MAX = 8
_SESSION_CACHE_MAX = 8
_design_cache: "OrderedDict[tuple, object]" = OrderedDict()
_session_cache: "OrderedDict[tuple, Tuple[object, threading.RLock]]" = OrderedDict()
# RLock：_shared_session 持锁构造 Decoder，而 Decoder 内部又会调
# _shared_design 取同一把锁。普通 Lock 会在这里自锁。
_cache_lock = threading.RLock()


def _design_key(kwargs: Dict) -> tuple:
    import arrow_wm as A
    return (int(kwargs.get("layout_version", A.LAYOUT_VERSION)),
            float(kwargs.get("depth", A.MOD_DEPTH)),
            float(kwargs.get("window_pow", A.WINDOW_POW)))


def _shared_design(**kwargs):
    """Return a process-wide ArrowDesign for this geometry.

    ``verbose=True`` always builds a fresh one so the layout prints still
    happen; it is a debugging path, not the hot one.
    """
    import arrow_wm as A
    if kwargs.get("verbose"):
        return A.ArrowDesign(**kwargs)
    key = _design_key(kwargs)
    with _cache_lock:
        design = _design_cache.get(key)
        if design is None:
            design = A.ArrowDesign(**kwargs)
            _design_cache[key] = design
            while len(_design_cache) > _DESIGN_CACHE_MAX:
                _design_cache.popitem(last=False)
        else:
            _design_cache.move_to_end(key)
        return design


def _session_key(carrier: str, margin_threshold: float,
                 min_group_margin: float, cc_threshold: Optional[float],
                 design, depth: Optional[float], layout_version: Optional[int],
                 window_pow: Optional[float], use_ecc_refine: bool,
                 verbose: bool, allow_legacy_v2: bool = False) -> tuple:
    return (carrier, float(margin_threshold), float(min_group_margin),
            None if cc_threshold is None else float(cc_threshold),
            None if design is None else id(design),
            None if depth is None else float(depth),
            None if layout_version is None else int(layout_version),
            None if window_pow is None else float(window_pow),
            bool(use_ecc_refine), bool(verbose), bool(allow_legacy_v2))


def _shared_session(**kwargs) -> Tuple["Decoder", threading.RLock]:
    """A cached Decoder plus the lock that serializes its use.

    Keyed on every constructor argument that changes behavior, thresholds
    included, so a call can never inherit another call's gate settings.
    """
    key = _session_key(
        kwargs.get("carrier", DEFAULT_CARRIER),
        kwargs.get("margin_threshold", MARGIN_THRESHOLD),
        kwargs.get("min_group_margin", MIN_GROUP_MARGIN),
        kwargs.get("cc_threshold", CC_THRESHOLD),
        kwargs.get("design"), kwargs.get("depth"), kwargs.get("layout_version"),
        kwargs.get("window_pow"), kwargs.get("use_ecc_refine", True),
        kwargs.get("verbose", False), kwargs.get("allow_legacy_v2", False))
    with _cache_lock:
        entry = _session_cache.get(key)
        if entry is None:
            session = Decoder(**kwargs)
            entry = (session, threading.RLock())
            _session_cache[key] = entry
            while len(_session_cache) > _SESSION_CACHE_MAX:
                _session_cache.popitem(last=False)
        else:
            _session_cache.move_to_end(key)
        return entry

REASON_TEXT = {
    "low_margin": "码字裕度不足",
    "low_group_margin": "单组裕度不足",
    "pad_mismatch": "补零不符",
    "crc_mismatch": "校验码不符",
    "low_cc": "配准 cc 不足",
    "low_logo_scale": "画面中 Logo 过小",
    "low_logo_scale_v2": "旧版印版且画面过小（无校验位，不予采信）",
    "invalid_payload": "码字越界",
    "unreadable": "读不到图像",
    "sync_failed": "同步失败",
    "decode_failed": "解码失败",
    "rectify_failed": "矫正失败",
}


def reason_text(reason: str) -> str:
    """Chinese label for a machine-readable ``reason`` code (or comma list)."""
    if not reason or reason == "accepted":
        return "接受"
    parts = [REASON_TEXT.get(p, p) for p in reason.split(",") if p]
    return "，".join(parts) if parts else "拒绝"


@dataclass(frozen=True)
class DecodeResult:
    """One production decode.

    ``digits`` is set only when ``accepted`` is True. On reject it is None
    even if the research decoder proposed a number (see ``candidate_digits``).
    """

    accepted: bool
    digits: Optional[str]
    candidate_digits: Optional[str]
    carrier: str
    margin: float
    margin_mean: float
    margin_min: float
    cc: float
    logo_scale: Optional[float]
    mean_conf: float
    payload_valid: bool
    ok: bool
    reason: str
    status: str
    margin_threshold: float
    min_group_margin_threshold: float
    cc_threshold: Optional[float]
    raw: Dict = field(repr=False, compare=False)

    def public_dict(self) -> Dict:
        """JSON-safe fields. Omits the research decoder dict (``raw``).

        ``candidate_digits`` is included here for in-process diagnostics.
        The HTTP service strips it unless debug and a valid token are set.
        Success is ``accepted`` / ``status == "accepted"`` only.
        """
        return {
            "accepted": self.accepted,
            "status": self.status,
            "digits": self.digits,
            "candidate_digits": self.candidate_digits,
            "carrier": self.carrier,
            "margin": self.margin,
            "margin_mean": self.margin_mean,
            "margin_min": self.margin_min,
            "cc": self.cc,
            "logo_scale": self.logo_scale,
            "mean_conf": self.mean_conf,
            "payload_valid": self.payload_valid,
            "ok": self.ok,
            "reason": self.reason,
            "margin_threshold": self.margin_threshold,
            "min_group_margin_threshold": self.min_group_margin_threshold,
            "cc_threshold": self.cc_threshold,
        }


def _finite(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _optional_finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _candidate_digits(raw: Dict) -> Optional[str]:
    digits = raw.get("digits")
    if digits is None:
        return None
    text = str(digits).strip()
    return text or None


def _legal_payload(digits: Optional[str]) -> bool:
    return digits is not None and len(digits) == 11 and digits.isdigit()


def _resolve_margins(raw: Dict) -> Tuple[float, float]:
    """Return ``(margin_mean, margin_min)``.

    When the research decoder did not report a per-group minimum, the mean
    is reused so older raw dicts keep the mean gate's behavior. A real
    decode always fills ``margin_min`` (or ``conf``).
    """
    mean = _finite(raw.get("margin_mean", raw.get("margin", 0.0)), 0.0)
    if raw.get("margin_min") is not None:
        return mean, _finite(raw.get("margin_min"), 0.0)
    groups = raw.get("group_margins")
    if groups:
        vals = [_finite(v, 0.0) for v in groups]
        if vals:
            return mean, min(vals)
    conf = raw.get("conf")
    if conf is not None:
        try:
            import arrow_wm as A
            arr = np.asarray(conf, np.float64).ravel()
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.size >= 7:
            ms = A.group_codeword_margins(arr)
            if ms.size:
                if "margin" not in raw and "margin_mean" not in raw:
                    mean = float(ms.mean())
                return mean, float(ms.min())
    return mean, mean


def _resolve_pad(raw: Dict) -> Optional[Tuple[int, int, int]]:
    """Group-0 pad bits, or None when this readout has nothing to check."""
    if "pad_bits" in raw and raw.get("pad_bits") is not None:
        try:
            bits = [int(b) for b in list(raw["pad_bits"])]
        except (TypeError, ValueError):
            return (-1, -1, -1)
        if len(bits) < 3:
            return (-1, -1, -1)
        return (bits[0] & 1, bits[1] & 1, bits[2] & 1)
    conf = raw.get("conf")
    if conf is None:
        return None
    import arrow_wm as A
    return A.group0_pad_bits(conf)


@dataclass(frozen=True)
class SuspectScore:
    """Blind check of one candidate number against measured soft bits.

    ``z`` is the absolute correlation in units of ``std(conf)``. ``gap`` is
    ``z`` minus the best one-digit neighbor. A neighbor can clear the
    absolute cut while losing the gap; that is not strong evidence.
    """

    suspect: str
    z: float
    score: float
    best_neighbor: Optional[str]
    neighbor_z: float
    gap: float
    verdict: str


def _suspect_verdict(z: float, gap: float) -> str:
    if z >= SUSPECT_Z_STRONG and gap >= SUSPECT_GAP_STRONG:
        return "包含（强证据）"
    if z >= SUSPECT_Z_MAYBE and gap >= SUSPECT_GAP_MAYBE:
        return "疑似"
    return "不包含"


def score_suspect(conf: Sequence[float], suspect: str) -> SuspectScore:
    """Score ``suspect`` against one-digit neighbors. Does not run ECC."""
    import arrow_wm as A
    suspect = str(suspect).strip()
    if len(suspect) != A.PAYLOAD_DIGITS or not suspect.isdigit():
        raise ValueError("suspect 必须是 11 位数字")
    soft = np.asarray(conf, np.float64).ravel()
    if soft.size != A.N_BITS:
        return SuspectScore(suspect, 0.0, 0.0, None, 0.0, 0.0, "不包含")
    sd = float(np.std(soft)) or 1e-9

    def z_of(digits: str) -> Tuple[float, float]:
        cw = np.asarray(A.encode_payload(digits), np.float64)
        sgn = cw * 2.0 - 1.0
        sc = float(np.dot(soft, sgn) / math.sqrt(float(sgn.size)))
        return sc / sd, sc

    z, sc = z_of(suspect)
    best_n: Optional[str] = None
    best_z = -1e30
    for i, ch in enumerate(suspect):
        for digit in "0123456789":
            if digit == ch:
                continue
            neighbor = suspect[:i] + digit + suspect[i + 1:]
            nz, _ = z_of(neighbor)
            if nz > best_z:
                best_z = nz
                best_n = neighbor
    gap = float(z - best_z)
    return SuspectScore(
        suspect=suspect,
        z=float(z),
        score=float(sc),
        best_neighbor=best_n,
        neighbor_z=float(best_z),
        gap=gap,
        verdict=_suspect_verdict(float(z), gap),
    )


def effective_cc_threshold(cc_threshold: Optional[float],
                           use_ecc_refine: bool) -> Optional[float]:
    """Registration gate for one session, or None to skip it.

    ``None`` always skips. With ECC refinement off the decoder does not
    compute a registration score (``cc`` stays 0), so the default cut of
    ``CC_THRESHOLD`` (0.85) is not applied. That is the ``--no-ecc`` rule.
    Any other numeric threshold is kept. ECC on keeps the given threshold,
    including the default 0.85.
    """
    if cc_threshold is not None:
        cc_threshold = float(cc_threshold)
    if not use_ecc_refine and (cc_threshold is None or cc_threshold == CC_THRESHOLD):
        return None
    return cc_threshold


def apply_gate(raw: Dict, *, carrier: str = DEFAULT_CARRIER,
               margin_threshold: float = MARGIN_THRESHOLD,
               min_group_margin: float = MIN_GROUP_MARGIN,
               cc_threshold: Optional[float] = CC_THRESHOLD,
               allow_legacy_v2: bool = False) -> DecodeResult:
    """Apply the production accept/reject gate to a research decoder dict.

    ``cc_threshold=None`` skips the registration gate. ``Decoder`` passes
    None for the default 0.85 cut when ECC refinement is off, because that
    path does not compute ``cc``. The mean-margin gate, the per-group
    minimum, and the group-0 pad check still apply.

    ``allow_legacy_v2`` lifts the extra ``logo_scale`` requirement that
    legacy inner v2 plates (no CRC) carry; see ``LOGO_SCALE_MIN_V2``.
    """
    if carrier not in CARRIERS:
        raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}")
    margin_threshold = float(margin_threshold)
    min_group_margin = float(min_group_margin)
    # v3/v4 的完整性由 CRC-7 负责；单组裕度可以比 v2/edge 松。
    if raw.get("layout_version") in (3, 4) and min_group_margin == MIN_GROUP_MARGIN:
        min_group_margin = MIN_GROUP_MARGIN_V3
    if cc_threshold is not None:
        cc_threshold = float(cc_threshold)

    ok = bool(raw.get("ok"))
    candidate = _candidate_digits(raw)
    legal = _legal_payload(candidate)
    payload_flag = raw.get("payload_valid", legal)
    payload_valid = bool(payload_flag) and legal
    margin, margin_min = _resolve_margins(raw)
    cc = _finite(raw.get("cc", 0.0), 0.0)
    logo_scale = _optional_finite(raw.get("logo_scale"))
    mean_conf = _finite(raw.get("mean_conf", 0.0), 0.0)

    reasons = []
    if not ok:
        reasons.append(str(raw.get("reason") or "decode_failed"))
    else:
        if not payload_valid:
            reasons.append("invalid_payload")
        if cc_threshold is not None and cc < cc_threshold:
            reasons.append("low_cc")
        # 尺度门按载体分：inner 有 CRC-7 兜底，可以比没有真校验的 edge 更松。
        scale_min = LOGO_SCALE_MIN_EDGE if carrier == "edge" else LOGO_SCALE_MIN
        if logo_scale is not None and logo_scale < scale_min:
            reasons.append("low_logo_scale")
        # 旧内孔 v2 没有 CRC，只能靠「画面足够大」保证码元采样率
        if (carrier == "inner" and raw.get("layout_version") == 2
                and not allow_legacy_v2 and logo_scale is not None
                and logo_scale < max(scale_min, LOGO_SCALE_MIN_V2)):
            reasons.append("low_logo_scale_v2")
        if margin < margin_threshold:
            reasons.append("low_margin")
        if margin_min < min_group_margin:
            reasons.append("low_group_margin")
        # v3 用 CRC-7 取代固定补零：crc_ok 存在时以它为准，不再看补零。
        crc_ok = raw.get("crc_ok")
        if crc_ok is not None:
            if not bool(crc_ok):
                reasons.append("crc_mismatch")
        else:
            pad = _resolve_pad(raw)
            if pad is not None and pad != PAD_BITS:
                reasons.append("pad_mismatch")

    accepted = not reasons
    return DecodeResult(
        accepted=accepted,
        digits=candidate if accepted else None,
        candidate_digits=candidate,
        carrier=carrier,
        margin=margin,
        margin_mean=margin,
        margin_min=margin_min,
        cc=cc,
        logo_scale=logo_scale,
        mean_conf=mean_conf,
        payload_valid=payload_valid,
        ok=ok,
        reason="" if accepted else ",".join(reasons),
        status="accepted" if accepted else "rejected",
        margin_threshold=margin_threshold,
        min_group_margin_threshold=min_group_margin,
        cc_threshold=cc_threshold,
        raw=raw,
    )


def _load_image(image: ImageLike) -> Optional[np.ndarray]:
    """Return a decoded array, or None when a path cannot be read.

    Empty arrays are treated as unreadable. A non-array, non-path value
    raises TypeError so a programming error is not reported as a reject.
    """
    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None
        return image
    if isinstance(image, (str, os.PathLike)):
        import arrow_wm as A
        return A.imread_u(os.fspath(image))
    raise TypeError(
        "image must be a numpy array or a filesystem path, "
        f"got {type(image).__name__}")


def _unreadable(carrier: str, margin_threshold: float,
                min_group_margin: float,
                cc_threshold: Optional[float], detail: str) -> DecodeResult:
    raw = {"ok": False, "reason": "unreadable", "digits": None, "detail": detail}
    return apply_gate(raw, carrier=carrier, margin_threshold=margin_threshold,
                      min_group_margin=min_group_margin,
                      cc_threshold=cc_threshold)


def _normalize_inner_layout(value) -> int:
    """内孔版式。**默认就是 v4**（生产唯一版式）。

    以前 ``None`` / ``"auto"`` 表示「先解 v3，不过再解 v2」；v2/v3 印版停产之后
    这套多版式兜底已经没有意义，默认直接落到 v4。
    显式传 2 / 3 仍然可用，只为了让仓库里的历史回归脚本能复现旧数字，
    产品路径不会走到它们。
    """
    if value is None:
        return 4
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto"):
            return 4
        value = text
    layout = int(value)
    if layout not in (2, 3, 4):
        raise ValueError(f"inner layout 只能是 2、3、4，收到 {value!r}")
    return layout



class Decoder:
    """Reusable decode session for one carrier.

    Constructing the session loads the ARROW master and the SIFT template.
    Call :meth:`decode` once per image.

    ``use_ecc_refine=False`` skips the default cc gate (registration ``cc``
    stays 0). The margin gates, the pad check, and the other accept rules
    still apply.
    """

    def __init__(self, carrier: str = DEFAULT_CARRIER, *,
                 margin_threshold: float = MARGIN_THRESHOLD,
                 min_group_margin: float = MIN_GROUP_MARGIN,
                 cc_threshold: Optional[float] = CC_THRESHOLD,
                 design=None,
                 depth: Optional[float] = None,
                 layout_version: Optional[int] = None,
                 window_pow: Optional[float] = None,
                 use_ecc_refine: bool = True,
                 allow_legacy_v2: bool = False,
                 verbose: bool = False):
        if carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}")
        self.carrier = carrier
        self.margin_threshold = float(margin_threshold)
        self.min_group_margin = float(min_group_margin)
        self.use_ecc_refine = bool(use_ecc_refine)
        self.allow_legacy_v2 = bool(allow_legacy_v2)
        # ECC off leaves cc at 0. Skip the default 0.85 gate; keep an
        # explicit non-default threshold, and always keep the margin gate.
        self.cc_threshold = effective_cc_threshold(
            cc_threshold, self.use_ecc_refine)

        import arrow_wm as A
        if design is not None:
            self.design = design
        elif carrier == "edge":
            kw = {"verbose": verbose}
            if layout_version is not None:
                kw["layout_version"] = int(layout_version)
            if depth is not None:
                kw["depth"] = float(depth)
            elif layout_version == 1:
                # Old plates were printed at 1.8. v1@6 passes depth explicitly.
                kw["depth"] = A.V1_DEPTH
            if window_pow is not None:
                kw["window_pow"] = float(window_pow)
            self.design = _shared_design(**kw)
        else:
            self.design = _shared_design(verbose=verbose)
        if carrier == "edge":
            self._impl = A.WatermarkDecoder(
                self.design, use_ecc_refine=self.use_ecc_refine, verbose=verbose)
            self._inner = None
        else:
            import arrow_inner as I
            # Default radial depth stays at arrow_inner.DEPTH. This wrapper
            # does not retune inner robustness.
            used_depth = I.DEPTH if depth is None else float(depth)
            self.inner_layout = _normalize_inner_layout(layout_version)
            self._inner = I.InnerCarrier(
                self.design, depth=used_depth, verbose=verbose,
                layout_version=self.inner_layout)
            self._impl = I.InnerDecoder(
                self._inner, use_ecc_refine=self.use_ecc_refine)

    def decode(self, image: ImageLike, *, return_debug: bool = False) -> DecodeResult:
        """Decode one image or file path.

        ``return_debug=True`` asks the research decoder for the rectified
        plate and homography (used by the verification figure). Those arrays
        live on ``result.raw`` and are not part of the stable field set.
        """
        loaded = _load_image(image)
        if loaded is None:
            detail = os.fspath(image) if isinstance(image, (str, os.PathLike)) else "empty array"
            return _unreadable(self.carrier, self.margin_threshold,
                               self.min_group_margin, self.cc_threshold, detail)
        # 统一的输入尺寸上限：内孔解码要在矫正域上做 1440 射线 × 4 孔的法向扫描，
        # 耗时随像素数近似线性增长。实测 1280×725（0.93 MP）本地就要 4.09 s，
        # 云托管 0.5 CPU 上会放大到十几秒，直接撞网关超时（客户端只看到 102002）。
        # 长边压到 900 后，22×6 mm 字标通常仍有 300+ px（logo_scale≈0.12，门槛 0.08），
        # 而耗时回到 0.3 s 量级。本来就是紧凑裁剪的小图不受影响。
        if (self.carrier == "inner" and loaded.ndim == 2
                and loaded.shape[0] > 8 and loaded.shape[1] > 8):
            _h0, _w0 = loaded.shape[:2]
            _long0 = max(_w0, _h0)
            if _long0 > 900:
                _k0 = 900.0 / _long0
                loaded = cv2.resize(
                    loaded, (max(8, int(round(_w0 * _k0))),
                             max(8, int(round(_h0 * _k0)))),
                    interpolation=cv2.INTER_AREA)

        # 近正方形输入的快路径：这类图几乎都是客户端把 22×6 mm 的长条塞进正方形
        # 画布造成的（实测 1536×1536）。直接按印版比例还原后解一次——命中就直接
        # 返回，省掉「先解一遍正方形、失败再解一遍」的双倍耗时（本地 4.0 s → 0.3 s，
        # 云托管 0.5 CPU 上这一步正是 callContainer 15 s 超时的来源）。
        if (self.carrier == "inner" and not return_debug and loaded.ndim == 2
                and loaded.shape[0] > 8 and loaded.shape[1] > 8):
            _h, _w = loaded.shape[:2]
            if 0.85 <= _w / float(_h) <= 1.18:
                _long = max(_w, _h)
                _fixed = cv2.resize(
                    loaded,
                    (_long, max(8, int(round(_long * 600.0 / 2200.0)))),
                    interpolation=cv2.INTER_AREA)
                _raw = self._impl.decode(_fixed, return_debug=False)
                if isinstance(_raw, dict) and _raw.get("crc_ok"):
                    _res = apply_gate(
                        _raw, carrier=self.carrier,
                        margin_threshold=self.margin_threshold,
                        min_group_margin=self.min_group_margin,
                        cc_threshold=self.cc_threshold,
                        allow_legacy_v2=self.allow_legacy_v2)
                    if _res.accepted:
                        return _res
                # 还原后仍不行 → 说明它可能本来就是正方形构图，回落到原图，
                # 但把长边压到 800 以锁住耗时：0.5 CPU 上 1280² 走完整候选搜索要几十秒。
                if _long > 800:
                    _k = 800.0 / _long
                    loaded = cv2.resize(
                        loaded, (max(8, int(round(_w * _k))),
                                 max(8, int(round(_h * _k)))),
                        interpolation=cv2.INTER_AREA)
        raw = self._impl.decode(loaded, return_debug=return_debug)
        if not isinstance(raw, dict):
            raw = {"ok": False, "reason": "decode_failed", "digits": None}

        def _gate(r: Dict) -> DecodeResult:
            return apply_gate(
                r, carrier=self.carrier,
                margin_threshold=self.margin_threshold,
                min_group_margin=self.min_group_margin,
                cc_threshold=self.cc_threshold,
                allow_legacy_v2=self.allow_legacy_v2)

        result = _gate(raw)
        # 多尺度兜底：仅内孔 v3 且首解被拒时。裁决权只在 CRC 手里，所以
        # 重试结果必须是 v3 且 CRC 通过——绝不能让放大后的 v2 读数（无校验位）
        # 借重试溜进来。return_debug 时跳过，避免调试图错位。
        # 注意不要把 margin_min 放进这个前置条件：最常见的失败恰恰是
        # 「其余组都很强、只有一组近打平」，此时 margin_min 接近 0，旧条件
        # 会把最该重试的样本挡在门外。
        # logo_scale 也必须沿用原图的：把同一张图放大不会多出光学采样，
        # 用放大后的 scale 去过「旧版印版要够大」这道门等于自欺。
        plausibly_near = (
            bool(raw.get("ok"))
            and raw.get("layout_version") == 3
            and float(raw.get("cc", 0.0)) >= 0.60
            and float(raw.get("margin", 0.0)) >= 1.2)
        orig_scale = raw.get("logo_scale")
        first_candidate = _candidate_digits(raw)
        if (not result.accepted and not return_debug and plausibly_near
                and self.carrier == "inner"
                and getattr(self, "inner_layout", None) in (3, "auto")):
            for scale in RETRY_SCALES:
                if max(loaded.shape[:2]) * scale > RETRY_MAX_SIDE:
                    continue
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                scaled = cv2.resize(loaded, None, fx=scale, fy=scale,
                                    interpolation=interp)
                raw2 = self._impl.decode(scaled, return_debug=False)
                if not isinstance(raw2, dict):
                    continue
                # 有 CRC 的版式都能进重试：v3/v4 都靠 crc_ok 裁决，写死 3
                # 会让 v4（当前生产版式）的重试永远不生效。
                if raw2.get("layout_version") not in (3, 4) or not raw2.get("crc_ok"):
                    continue
                # 只允许重试「确认」首解已经读到的号码，不允许它另起一个新号码。
                # CRC-7 的漏检率是 1/128，多解一次就多一次撞车机会；实测在
                # logo_scale=0.13 的压测里，无约束重试把误接受从 0 抬到 1/12。
                # 加上这条之后，重试的证据是「两次独立测量 + CRC」三者一致。
                if first_candidate is not None and raw2.get("digits") != first_candidate:
                    continue
                if orig_scale is not None:
                    raw2 = dict(raw2, logo_scale=float(orig_scale))
                res2 = _gate(raw2)
                if res2.accepted:
                    return res2
        # 反卷积重试：同一道闸（v3 + CRC + 与首解同号），只换信号处理方式。
        dec_sigmas = tuple(RETRY_DECONV_SIGMAS or ())
        if (not result.accepted and not return_debug and plausibly_near
                and dec_sigmas
                and self.carrier == "inner"
                and getattr(self, "inner_layout", None) in (3, 4)):
            import arrow_wm as _A
            for sigma in dec_sigmas:
                sharp = _A.richardson_lucy(loaded, float(sigma),
                                           RETRY_DECONV_ITERS)
                raw2 = self._impl.decode(sharp, return_debug=False)
                if not isinstance(raw2, dict):
                    continue
                if raw2.get("layout_version") not in (3, 4) or not raw2.get("crc_ok"):
                    continue
                if first_candidate is not None and raw2.get("digits") != first_candidate:
                    continue
                if orig_scale is not None:
                    raw2 = dict(raw2, logo_scale=float(orig_scale))
                res2 = _gate(raw2)
                if res2.accepted:
                    return res2
        return result


def decode(image: ImageLike, carrier: str = DEFAULT_CARRIER, *,
           margin_threshold: float = MARGIN_THRESHOLD,
           min_group_margin: float = MIN_GROUP_MARGIN,
           cc_threshold: Optional[float] = CC_THRESHOLD,
           return_debug: bool = False,
           design=None,
           depth: Optional[float] = None,
           layout_version: Optional[int] = None,
           window_pow: Optional[float] = None,
           use_ecc_refine: bool = True,
           allow_legacy_v2: bool = False,
           verbose: bool = False) -> DecodeResult:
    """Decode one image. Carrier defaults to ``"inner"``; pass ``"edge"`` for legacy plates.

    On reject, ``accepted`` is False, ``status`` is ``"rejected"``, and
    ``digits`` is None. Do not promote ``candidate_digits`` or ``ok``.

    ``use_ecc_refine=False`` matches ``--no-ecc``: the default cc ≥ 0.85
    check is skipped (``cc`` stays 0). Mean margin, minimum group margin,
    and the pad check still apply.
    """
    session, lock = _shared_session(
        carrier=carrier,
        margin_threshold=margin_threshold,
        min_group_margin=min_group_margin,
        cc_threshold=cc_threshold,
        design=design,
        depth=depth,
        layout_version=layout_version,
        window_pow=window_pow,
        use_ecc_refine=use_ecc_refine,
        allow_legacy_v2=allow_legacy_v2,
        verbose=verbose,
    )
    with lock:
        return session.decode(image, return_debug=return_debug)
