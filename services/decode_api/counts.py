"""Durable, transactional verification ordinals and idempotent successful responses."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class RequestConflict(ValueError):
    pass


class VerificationStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS codes (
                    digits TEXT PRIMARY KEY,
                    total INTEGER NOT NULL CHECK(total >= 1),
                    first_at TEXT NOT NULL,
                    last_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verifications (
                    request_id TEXT PRIMARY KEY,
                    image_sha256 TEXT NOT NULL,
                    digits TEXT NOT NULL REFERENCES codes(digits),
                    ordinal INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    UNIQUE(digits, ordinal)
                );
            ''')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('counting_since', self.now()))
            self.counting_since = db.execute('SELECT value FROM metadata WHERE key=?', ('counting_since',)).fetchone()[0]
        db.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        return db

    @staticmethod
    def cached(row, image_sha256):
        if row is None:
            return None
        if row[0] != image_sha256:
            raise RequestConflict('同一验证请求不能使用不同照片，请重新开始检测')
        result = json.loads(row[1])
        result['verification_replayed'] = True
        return result

    def lookup(self, request_id, image_sha256):
        db = self.connect()
        try:
            return self.cached(db.execute(
                'SELECT image_sha256,response_json FROM verifications WHERE request_id=?',
                (request_id,)).fetchone(), image_sha256)
        finally:
            db.close()

    def record(self, request_id, image_sha256, payload):
        if payload.get('accepted') is not True:
            raise ValueError('Only accepted decodes can be counted')
        digits = payload['digits']
        if not isinstance(digits, str) or len(digits) != 11 or not digits.isascii() or not digits.isdigit():
            raise ValueError('Expected an eleven-digit code')
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            existing = self.cached(db.execute(
                'SELECT image_sha256,response_json FROM verifications WHERE request_id=?',
                (request_id,)).fetchone(), image_sha256)
            if existing is not None:
                db.rollback()
                return existing
            now = self.now()
            db.execute('''INSERT INTO codes(digits,total,first_at,last_at) VALUES (?,1,?,?)
                ON CONFLICT(digits) DO UPDATE SET total=total+1,last_at=excluded.last_at''',
                (digits, now, now))
            total, first = db.execute('SELECT total,first_at FROM codes WHERE digits=?', (digits,)).fetchone()
            result = dict(payload, verification_count=total, first_verified_at=first,
                          verified_at=now, verification_id=request_id, verification_replayed=False,
                          counting_since=self.counting_since)
            db.execute('INSERT INTO verifications VALUES (?,?,?,?,?)',
                       (request_id, image_sha256, digits, total, json.dumps(result, ensure_ascii=False)))
            db.commit()
            return result
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
