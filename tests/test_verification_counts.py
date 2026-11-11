import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from services.decode_api.counts import VerificationStore, RequestConflict

class CountTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=self.temp.name+'/records.sqlite3'
        self.store=VerificationStore(self.path)
        self.payload={'accepted':True,'digits':'00000000001'}

    def tearDown(self):self.temp.cleanup()

    def test_concurrent_unique_ordinals(self):
        def record(i):return self.store.record(str(i),'a'*64,self.payload)['verification_count']
        with ThreadPoolExecutor(max_workers=8) as pool:
            counts=list(pool.map(record,range(50)))
        self.assertEqual(sorted(counts),list(range(1,51)))

    def test_concurrent_retry_counts_once_and_survives_reopen(self):
        def record(i):return self.store.record('same-request','b'*64,self.payload)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(record,range(30)))
        self.assertEqual({r['verification_count'] for r in results},{1})
        self.assertEqual(sum(not r['verification_replayed'] for r in results),1)
        reopened=VerificationStore(self.path)
        self.assertEqual(reopened.lookup('same-request','b'*64)['verification_count'],1)
        self.assertEqual(reopened.record('new-request','b'*64,self.payload)['verification_count'],2)

    def test_codes_are_separate_and_keep_leading_zeros(self):
        a=self.store.record('first','a'*64,self.payload)
        b=self.store.record('second','a'*64,{'accepted':True,'digits':'10000000000'})
        self.assertEqual((a['verification_count'],b['verification_count']),(1,1))
        self.assertEqual(a['digits'],'00000000001')

    def test_image_conflict_and_failed_commit_do_not_increment(self):
        self.store.record('first','a'*64,self.payload)
        with self.assertRaises(RequestConflict):self.store.record('first','b'*64,self.payload)
        with self.assertRaises(TypeError):self.store.record('broken','a'*64,dict(self.payload,bad=object()))
        self.assertEqual(self.store.record('next','a'*64,self.payload)['verification_count'],2)
        self.assertIsNone(self.store.lookup('broken','a'*64))

    def test_rejected_decode_cannot_be_recorded(self):
        with self.assertRaises(ValueError):self.store.record('rejected','a'*64,{'accepted':False,'digits':None})

if __name__=='__main__':unittest.main()
