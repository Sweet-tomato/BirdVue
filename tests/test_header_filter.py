# -*- coding: utf-8 -*-
"""_sensitive_header contract: credential-looking headers are dropped from
the echoed request headers; ordinary device/debug headers survive."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import bird_frames_service as api  # noqa: E402


class SensitiveHeaderTest(unittest.TestCase):
    def test_credential_headers_flagged(self):
        for name in (
            "x-api-key", "x-auth-token", "x-session-id", "x-access-token",
            "api_key", "x-client-secret", "x-signature", "x-auth",
        ):
            with self.subTest(name=name):
                self.assertTrue(api._sensitive_header(name), name)

    def test_ordinary_headers_kept(self):
        for name in (
            "x-device-id", "x-request-id", "user-agent", "x-forwarded-for",
            "content-type", "x-event-id", "x-custom-trace",
        ):
            with self.subTest(name=name):
                self.assertFalse(api._sensitive_header(name), name)

    def test_header_meta_drops_secrets_keeps_device(self):
        class FakeHeaders:
            def items(self):
                return [
                    ("X-Device-Id", "cam-9"),
                    ("X-Api-Key", "sk-live-123"),
                    ("Authorization", "Bearer k"),
                    ("X-Auth-Token", "tok"),
                    ("User-Agent", "birdctl/1.0"),
                ]

        class FakeRequest:
            headers = FakeHeaders()

        meta = api._header_meta(FakeRequest())
        lower = {k.lower(): v for k, v in meta.items()}
        self.assertEqual(lower.get("x-device-id"), "cam-9")
        self.assertEqual(lower.get("user-agent"), "birdctl/1.0")
        self.assertNotIn("x-api-key", lower)
        self.assertNotIn("authorization", lower)
        self.assertNotIn("x-auth-token", lower)


if __name__ == "__main__":
    unittest.main()
