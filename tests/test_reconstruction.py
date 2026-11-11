import unittest
import cv2
import numpy as np
import arrow_wm as A
import arrow_v5 as V
from arrow_vector import VectorDesign,PLACEMENT,svg_text
from arrow_camera import install_thumbnail_locator

class CodingTests(unittest.TestCase):
    def test_crc7_mmc_check_vector(self):
        bits=[int(x) for byte in b'123456789' for x in f'{byte:08b}']
        self.assertEqual(sum(b<<(6-i) for i,b in enumerate(V.crc7(bits))),0x75)

    def test_checksum_output_dimension(self):
        def binary_rank(matrix):
            a=matrix.copy();r=0
            for col in range(a.shape[1]):
                candidates=np.flatnonzero(a[r:,col])
                if not len(candidates):continue
                pivot=r+int(candidates[0]);a[[r,pivot]]=a[[pivot,r]]
                for row in range(a.shape[0]):
                    if row!=r and a[row,col]:a[row]^=a[r]
                r+=1
                if r==a.shape[0]:break
            return r
        # Full 37-bit domain. Fixing suffix bits can restrict a checksum's
        # output image and is not a valid test of its total information rank.
        for checksum in (V.crc7,):
            columns=np.array([checksum(b) for b in np.eye(37,dtype=int)]).T
            self.assertEqual(binary_rank(columns),7)

    def test_many_payloads_and_one_error_per_codeword(self):
        for digits in A.make_payloads(100,998231)+['00000000000','99999999999']:
            code=np.array(V.encode(digits))
            for position in range(7):
                soft=code*2.-1
                soft[position::7]*=-.25
                decoded,ok,_=V.decode_soft(soft)
                self.assertTrue(ok);self.assertEqual(decoded,digits)

    def test_integrity_detects_modified_payload(self):
        bits=A.payload_to_bits('13800138000');checksum=V.crc7(bits)
        bits[12]^=1
        decoded,ok,_=V.decode_soft(np.array(A.hamming74_encode(bits+checksum))*2.-1)
        self.assertFalse(ok)

class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.design=VectorDesign();cls.carrier=V.Carrier(cls.design,equal_arc=True)

    def test_dimensions_and_topology(self):
        self.assertIn('width="22mm" height="6mm"',svg_text())
        self.assertAlmostEqual(PLACEMENT[-1][1]+PLACEMENT[-1][2],5)
        n,_=cv2.connectedComponents(self.design.mask1)
        self.assertEqual(n-1,5)
        self.assertEqual(len(self.carrier.holes),4)
        for i,gap in enumerate([.04,.03,.03,.01]):
            self.assertAlmostEqual(PLACEMENT[i+1][1]-sum(PLACEMENT[i][1:]),gap)

    def test_same_r_geometry(self):
        self.assertTrue(np.array_equal(self.design.mask1[:,458:841],self.design.mask1[:,854:1237]))

    def test_interleaver_spreads_each_group(self):
        c=self.carrier
        self.assertEqual(sorted(c.pair_order),list(range(77)))
        for start in range(0,77,7):
            holes={c.cells[i][0] for i in c.pair_order[start:start+7]}
            self.assertGreaterEqual(len(holes),3)

    def test_window_partition(self):
        for hi,h in enumerate(self.carrier.holes):
            windows=np.array([self.carrier._window(hi,k) for k in range(h['n_cells'])])
            self.assertTrue(np.all(windows>=0))
            self.assertTrue(np.all(windows.sum(0)<=1+1e-9))

    def test_outer_silhouette_unchanged(self):
        def silhouette(mask):
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            out=np.zeros_like(mask);cv2.drawContours(out,contours,-1,255,-1)
            return out
        expected=silhouette(self.design.mask4)
        for digits in A.make_payloads(8,1951):
            actual=silhouette(self.carrier.embed(digits)['mask4'])
            self.assertTrue(np.array_equal(expected,actual))

    def test_clean_roundtrip_and_unmarked_rejection(self):
        decoder=V.Decoder(self.carrier)
        for digits in ['00000000000','13800138000','99999999999']:
            result=V.gate(decoder.decode(self.carrier.embed(digits)['plate']))
            self.assertTrue(result.accepted);self.assertEqual(result.digits,digits)
        self.assertFalse(V.gate(decoder.decode(self.design.template)).accepted)

class CameraTests(unittest.TestCase):
    def test_homography_is_lifted_with_exact_dimensions(self):
        class Registration:
            def _find_transforms(self,gray):
                self.shape=gray.shape
                return [np.eye(3)]
        r=Registration();install_thumbnail_locator(r,900)
        H=r._find_transforms(np.zeros((1801,2401),np.uint8))[0]
        self.assertEqual(r.shape,(675,900))
        sx,sy=900/2401,675/1801
        self.assertTrue(np.allclose(H,[[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]]))

if __name__=='__main__':unittest.main()
