import hashlib
import os
import unittest
import tempfile
import uuid
from unittest.mock import patch
import cv2
from fastapi.testclient import TestClient
from services.decode_api import v5
from arrow_service import FORMAT, MASTER_ID

class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['DECODE_TOKEN']='local-test-token'
        cls.temp=tempfile.TemporaryDirectory()
        os.environ['ARROW_DB_PATH']=cls.temp.name+'/counts.sqlite3'
        cls.client=TestClient(v5.app)
        cls.client.__enter__()
        cls.image=cv2.imencode('.png',v5._decoder.carrier.embed('01088886666')['plate'])[1].tobytes()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None)
        cls.temp.cleanup()

    def send(self,data=None,headers=None):
        data=self.image if data is None else data
        h={'X-Decode-Token':'local-test-token','X-Arrow-Format':FORMAT,
           'X-Image-SHA256':hashlib.sha256(data).hexdigest(),
           'X-Verification-ID':uuid.uuid4().hex}
        h.update(headers or {})
        return self.client.post('/decode',files={'image':('photo.png',data,'image/png')},headers=h)

    def test_contract_and_leading_zero(self):
        r=self.send(); self.assertEqual(r.status_code,200)
        d=r.json();self.assertTrue(d['accepted']);self.assertEqual(d['digits'],'01088886666')
        self.assertEqual(d['format'],FORMAT);self.assertEqual(d['master_id'],MASTER_ID)
        self.assertNotIn('candidate_digits',d)
        self.assertEqual(d['input_sha256'],hashlib.sha256(self.image).hexdigest())
        self.assertTrue(self.client.get('/health').json()['ready'])
        self.assertGreaterEqual(d['verification_count'],1)

    def test_idempotent_success_and_next_attempt(self):
        request_id=uuid.uuid4().hex
        first=self.send(headers={'X-Verification-ID':request_id}).json()
        with patch.object(v5._decoder,'decode',side_effect=AssertionError('Replay must use cached result')):
            replay=self.send(headers={'X-Verification-ID':request_id}).json()
        self.assertEqual(first['verification_count'],replay['verification_count'])
        self.assertEqual(first['verified_at'],replay['verified_at'])
        self.assertTrue(replay['verification_replayed'])
        following=self.send().json()
        self.assertEqual(following['verification_count'],first['verification_count']+1)
        self.assertEqual(self.send(b'other',{'X-Verification-ID':request_id}).status_code,409)

    def test_validation(self):
        for headers,status in [({'X-Decode-Token':'wrong'},401),({'X-Arrow-Format':'unsupported-format'},409),
                               ({'X-Image-SHA256':'0'*64},422),({'X-Image-SHA256':''},400),
                               ({'X-Verification-ID':''},400)]:
            self.assertEqual(self.send(headers=headers).status_code,status)
        self.assertEqual(self.send(b'not an image').status_code,415)
        self.assertEqual(self.send(b'\x89PNG\r\n\x1a\n').status_code,422)

    def test_unmarked(self):
        image=cv2.imencode('.png',v5._decoder.carrier.design.template)[1].tobytes()
        d=self.send(image).json();self.assertFalse(d['accepted']);self.assertIsNone(d['digits'])
        self.assertIsNone(d['verification_count'])
        self.assertIsNone(v5._store.lookup(d['verification_id'],d['input_sha256']))

    def test_busy_rejects_without_queue(self):
        v5._busy=True
        try:self.assertEqual(self.send().status_code,429)
        finally:v5._busy=False

    def test_upload_limit(self):
        self.assertEqual(self.send(b'x'*(v5.MAX_IMAGE_BYTES+1)).status_code,413)

if __name__=='__main__':unittest.main()
