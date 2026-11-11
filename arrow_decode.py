"""Acceptance policy and public results for ARROW v5."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math
import numpy as np

MARGIN_THRESHOLD = 1.8
MIN_GROUP_MARGIN = 0.20
CC_THRESHOLD = 0.85
LOGO_SCALE_MIN = 0.08
REASON_TEXT = {
    'low_margin': '码字裕度不足', 'low_group_margin': '单组裕度不足',
    'crc_mismatch': '校验码不符', 'low_cc': '配准 cc 不足',
    'low_logo_scale': '画面中 Logo 过小', 'invalid_payload': '码字越界',
    'unreadable': '读不到图像', 'sync_failed': '同步失败',
    'decode_failed': '解码失败', 'rectify_failed': '矫正失败',
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
        The HTTP service always removes it.
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
    is used as a fallback for diagnostic dictionaries. A real
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

def apply_gate(raw: Dict, *, margin_threshold=MARGIN_THRESHOLD,
               min_group_margin=MIN_GROUP_MARGIN, cc_threshold=CC_THRESHOLD) -> DecodeResult:
    """Require a legal payload, CRC and all configured image-quality thresholds."""
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
        if logo_scale is not None and logo_scale < LOGO_SCALE_MIN:
            reasons.append("low_logo_scale")
        if margin < margin_threshold:
            reasons.append("low_margin")
        if margin_min < min_group_margin:
            reasons.append("low_group_margin")
        if not raw.get("crc_ok", False):
            reasons.append("crc_mismatch")

    accepted = not reasons
    return DecodeResult(
        accepted=accepted,
        digits=candidate if accepted else None,
        candidate_digits=candidate,
        carrier="inner",
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
