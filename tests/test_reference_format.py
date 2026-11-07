import unittest

import cv2
import numpy as np

import arrow_decode as AD
import arrow_inner as I
import arrow_wm as A
from arrow_arc import ArcCarrier, ArcDecoder, crc7_mmc, encode_mmc, decode_mmc
from arrow_reference import ReferenceDesign


class CodecTests(unittest.TestCase):
    def test_crc_mmc_published_check(self):
        bits=[(c>>i)&1 for c in b'123456789' for i in range(7,-1,-1)]
        self.assertEqual(int(''.join(map(str,crc7_mmc(bits))),2),0x75)

    def test_all_payload_one_and_two_bit_errors_are_detected(self):
        syndromes=[]
        for i in range(37):
            bits=[0]*37;bits[i]=1;syndromes.append(tuple(crc7_mmc(bits)))
        self.assertNotIn((0,)*7,syndromes)
        self.assertEqual(len(set(syndromes)),37)

    def test_legacy_checksum_compatibility(self):
        p=A.payload_to_bits('13800138000')
        q=p.copy();q[10]^=1;q[19]^=1
        self.assertEqual(A.bits_to_payload(q),'13733160208')
        self.assertEqual(I.crc7(p),I.crc7(q))
        self.assertNotEqual(crc7_mmc(p),crc7_mmc(q))

    def test_soft_codec_corrects_one_error_per_hamming_group(self):
        for digits in ('00000000000','99999999999','13800138000','01088886666'):
            bits=np.array(encode_mmc(digits))
            soft=(bits*2-1).astype(float)
            soft[np.arange(0,77,7)] *= -.4
            decoded,passed,_=decode_mmc(soft)
            self.assertEqual(decoded,digits);self.assertTrue(passed)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        cls.design=ReferenceDesign()
        cls.carrier=ArcCarrier(cls.design,layout_version=4)
        cls.decoder=ArcDecoder(cls.carrier)

    def test_topology_and_interleaving(self):
        self.assertEqual(len(self.design.letter_centroids),5)
        self.assertEqual(len(self.carrier.holes),4)
        order=self.carrier.pair_order
        self.assertEqual(sorted(order),list(range(77)))
        for g in range(11):
            self.assertGreaterEqual(len({self.carrier.cells[i][0] for i in order[g*7:g*7+7]}),3)

    def test_outer_contour_is_preserved(self):
        solid=self.carrier.solid_mask4()>0
        for digits in ('13800138000','99999999999','00000000000'):
            wm=self.carrier.embed(digits)
            self.assertEqual(np.count_nonzero(wm['mask4'][~solid]),0)

    def test_blind_clean_roundtrip_and_null_rejection(self):
        for digits in ('13800138000','99999999999','00000000000'):
            wm=self.carrier.embed(digits)
            result=AD.apply_gate(self.decoder.decode(wm['plate']),carrier='inner')
            self.assertTrue(result.accepted);self.assertEqual(result.digits,digits)
        null=AD.apply_gate(self.decoder.decode(self.design.template),carrier='inner')
        self.assertFalse(null.accepted)

    def test_production_sample_still_decodes(self):
        result=AD.decode('samples/inner/ARROW_13800138000.png')
        self.assertTrue(result.accepted);self.assertEqual(result.digits,'13800138000')


if __name__=='__main__':unittest.main()
