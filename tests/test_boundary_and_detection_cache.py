# -*- coding: utf-8 -*-
"""边界安全与检测缓存的轻量回归测试（不加载模型权重）。"""
from __future__ import annotations

import contextlib
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipeline"))

import bird_cli  # noqa: E402
from api import _common  # noqa: E402


class BoundarySecurityTest(unittest.TestCase):
    def test_authorize_uses_constant_time_compare(self):
        with mock.patch.object(_common.hmac, "compare_digest", return_value=True) as compare:
            _common.authorize("Bearer secret", "secret")
        compare.assert_called_once_with("Bearer secret", "Bearer secret")

        with mock.patch.object(_common.hmac, "compare_digest", return_value=False):
            with self.assertRaises(HTTPException) as ctx:
                _common.authorize("Bearer wrong", "secret")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_resolve_public_ip_returns_global_address(self):
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        with mock.patch.object(_common.socket, "getaddrinfo", return_value=infos):
            ip = _common.resolve_public_ip("http://example.com/a.jpg", "image")
        self.assertEqual(ip, "93.184.216.34")

    def test_resolve_public_ip_rejects_private_address(self):
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        with mock.patch.object(_common.socket, "getaddrinfo", return_value=infos):
            with self.assertRaises(HTTPException) as ctx:
                _common.resolve_public_ip("http://example.com/a.jpg", "image")
        self.assertEqual(ctx.exception.detail, "private_image_host_not_allowed")


class DetectionCacheTest(unittest.TestCase):
    def test_detect_frame_scores_returns_reusable_focus(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
            calls = []

            def fake_create_focus_view(image, focus_net):
                calls.append(1)
                return image.crop((0, 0, 4, 4)), {"score": 0.5, "box": [0, 0, 4, 4]}

            with mock.patch.object(bird_cli, "create_focus_view", fake_create_focus_view), \
                    mock.patch.object(bird_cli, "_MODEL_LOCK", contextlib.nullcontext()):
                rows = bird_cli.detect_frame_scores([path], [None], include_focus=True)

            self.assertEqual(len(calls), 1)
            self.assertIsNotNone(rows[0]["focus"])
            self.assertEqual(rows[0]["focusMeta"]["score"], 0.5)

            key = str(path.resolve())
            cached = {key: {"focus": rows[0]["focus"], "focusMeta": rows[0]["focusMeta"]}}
            focus, meta, hit = bird_cli._cached_focus(path, cached)
            self.assertTrue(hit)
            self.assertIs(focus, rows[0]["focus"])
            self.assertIs(meta, rows[0]["focusMeta"])


if __name__ == "__main__":
    unittest.main()
