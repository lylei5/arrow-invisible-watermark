"""Research carrier: equal arc-length NRZ cells and cross-counter interleaving.

Format identity: reference-arc-v1. This geometry requires its matching decoder;
the production v4 HTTP endpoint uses a different master and cell map.
"""
import numpy as np

import arrow_inner as I
import arrow_wm as A


def crc7_mmc(bits):
    """CRC-7/MMC, MSB first, polynomial 0x09, init/xorout 0."""
    reg = 0
    for bit in bits:
        feedback = ((reg >> 6) & 1) ^ (int(bit) & 1)
        reg = (reg << 1) & 0x7f
        if feedback:
            reg ^= 0x09
    return [(reg >> i) & 1 for i in range(6, -1, -1)]


def encode_mmc(digits):
    payload = A.payload_to_bits(digits)
    return A.hamming74_encode(payload+crc7_mmc(payload))


def decode_mmc(soft):
    cb = A.hamming_codebook()
    signs = cb.astype(np.float64)*2-1
    soft = np.asarray(soft, np.float64).ravel()
    if soft.size != 77 or not np.all(np.isfinite(soft)):
        return None, False, 0
    data, changes = [], 0
    hard = soft > 0
    for g in range(11):
        sl = slice(7*g, 7*g+7)
        code = cb[np.argmax(signs@soft[sl])]
        data.extend(int(code[k]) for k in (2,4,5,6))
        changes += int(np.sum(code != hard[sl]))
    return A.bits_to_payload(data[:37]), crc7_mmc(data[:37])==data[37:], changes


class ArcCarrier(I.InnerCarrier):
    format_id = "reference-arc-v1"

    def __init__(self, *args, checksum="mmc", **kwargs):
        if checksum not in ("mmc", "legacy"):
            raise ValueError("checksum must be mmc or legacy")
        self.checksum = checksum
        super().__init__(*args, **kwargs)
        if self.layout_version != 4:
            raise ValueError("ArcCarrier requires NRZ layout 4")

    def _encode_payload(self, digits):
        return encode_mmc(digits) if self.checksum == "mmc" else I.encode_payload_v3(digits)

    def _alloc_cells(self):
        super()._alloc_cells()
        for h in self.holes:
            r = h["r_tab"] / self.SS
            xy = np.column_stack((r*self.cos_t, r*self.sin_t))
            lengths = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
            h["arc_phase"] = np.r_[0.0, np.cumsum(lengths[:-1])] / lengths.sum()

    def _window(self, hi, k):
        h = self.holes[hi]
        t = (h["arc_phase"]*h["n_cells"] - k) % h["n_cells"]
        return np.where(t < 1.0, np.sin(np.pi*np.clip(t, 0, 1)), 0.0)

    def _build_pair_order(self):
        counts = [h["n_cells"] for h in self.holes]
        used = [0]*len(counts)
        offsets = np.r_[0, np.cumsum(counts[:-1])]
        order = []
        for _ in range(sum(counts)):
            hi = min((i for i in range(len(counts)) if used[i] < counts[i]),
                     key=lambda i: ((used[i]+1)/counts[i], i))
            order.append(int(offsets[hi]+used[hi]))
            used[hi] += 1
        return tuple(order)


class ArcDecoder(I.InnerDecoder):
    def _decode_payload(self, soft):
        if self.carrier.checksum == "mmc":
            return decode_mmc(soft)
        return I.decode_payload_v3_soft(soft)


class NuisanceDecoder(ArcDecoder):
    """Jointly fit cell displacements and smooth geometric nuisance terms.

    The constant and first angular harmonics absorb residual registration and
    ink drift. A small ridge bounds amplification of weakly identified modes.
    This is an experimental readout, selected only on held-out measurements.
    """

    def __init__(self, carrier, **kwargs):
        super().__init__(carrier, **kwargs)
        self._projectors = []
        off = 0
        for hi, h in enumerate(carrier.holes):
            n = h["n_cells"]
            w = np.column_stack(self._win[off:off+n]).astype(np.float64)
            nuisance = np.column_stack((np.ones(I.N_THETA), carrier.cos_t, carrier.sin_t))
            x = np.column_stack((w, nuisance))
            ridge = np.eye(n+3)*0.01
            ridge[n:, n:] = np.eye(3)*1e-6
            self._projectors.append(np.linalg.solve(x.T@x+ridge, x.T)[:n])
            off += n

    def measure(self, warped):
        profiles = self._profiles(warped.astype(np.float32))
        features = []
        for hi, (profile, (_, _, r0)) in enumerate(zip(profiles, self._maps)):
            delta = self._radius(profile, r0)-self._r_ref[hi]
            delta = np.where(np.abs(delta) <= self._half, delta, 0.0)
            features.extend(self._projectors[hi]@delta)
        return np.asarray(features, np.float32)
