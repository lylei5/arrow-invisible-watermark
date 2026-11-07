"""Design from the supplied ARROW screenshot, used by research experiments."""
from pathlib import Path

import cv2
import numpy as np

import arrow_wm as A


class ReferenceDesign(A.ArrowDesign):
    """Explicit reference geometry with a separate template identity."""

    def __init__(self, mask_path=None):
        self.reference_path = Path(mask_path or Path(__file__).parent / "assets/reference/mask4.png")
        super().__init__()

    def _rasterize(self):
        data = np.frombuffer(self.reference_path.read_bytes(), np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (A.CANVAS_H*A.SS, A.CANVAS_W*A.SS):
            raise ValueError("Expected an 8800 x 2400 reference mask")
        return mask

    def _build_cells(self):
        # Inner-carrier registration needs geometry, not legacy outer-edge
        # cells whose frozen permutation belongs to a different master.
        self.cells = []
        self.pair_perm = tuple(range(A.N_BITS))
        self.pair_perm_id = "reference-inner"

    def _compute_letter_centroids(self):
        n, _, stats, centers = cv2.connectedComponentsWithStats(self.mask1, connectivity=8)
        ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 100]
        ids.sort(key=lambda i: centers[i, 0])
        return [tuple(map(float, centers[i])) for i in ids]
