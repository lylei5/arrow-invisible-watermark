"""Freeze the v5 print geometry and default HTTP entry point."""
import hashlib
import unittest
from pathlib import Path
import cv2
import numpy as np
from arrow_service import Decoder, FORMAT, MASTER_ID
from services.decode_api.app import app
from services.decode_api.v5 import app as v5_app


class ProductionContractTests(unittest.TestCase):
    def test_default_http_entry_and_master(self):
        self.assertIs(app, v5_app)
        self.assertEqual(FORMAT, 'arrow-vector-v5')
        self.assertEqual(MASTER_ID, 'dc4bd045fb3c06204e32f41b548d30c0afc41403d4784bb3f2f18f331ad830a0')

    def test_print_pixels_and_bundled_sample(self):
        expected = {
            '00000000000': '8f08dab08ece1f97b2986e496c2136c906ed5362b2cd2e010262156d26a9a704',
            '13800138000': '3f8f6a790faa27106b03ee488d7364300ee32218d1b7468e9bb0747e6dd5a924',
            '99999999999': 'db5e149b5608fad200a36f9e09d85812152a462e37b515f092c20680c221d8d7',
        }
        cv2.setNumThreads(1)
        decoder = Decoder()
        for digits, digest in expected.items():
            with self.subTest(digits=digits):
                plate = decoder.carrier.embed(digits)['plate']
                self.assertEqual(plate.shape, (600, 2200))
                self.assertEqual(hashlib.sha256(plate.tobytes()).hexdigest(), digest)
        path = Path(__file__).resolve().parents[1] / 'samples/v5/ARROW_13800138000.png'
        plate = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        result = decoder.decode(plate)
        self.assertTrue(result.accepted)
        self.assertEqual(result.digits, '13800138000')


if __name__ == '__main__':
    unittest.main()
