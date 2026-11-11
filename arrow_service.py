"""Production v5: thumbnail localization and original-pixel measurement."""
import hashlib
import arrow_v5 as V
from arrow_vector import svg_text
from arrow_camera import install_thumbnail_locator

FORMAT = 'arrow-vector-v5'
MASTER_ID = hashlib.sha256(svg_text().encode('utf-8')).hexdigest()

class Decoder:
    def __init__(self):
        self.carrier = V.Carrier(depth=8., equal_arc=False, interleave=True)
        self.core = V.Decoder(self.carrier)
        install_thumbnail_locator(self.core._reg, 900)

    def decode(self, gray):
        return V.gate(self.core.decode(gray))
