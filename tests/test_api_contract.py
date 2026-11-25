# -*- coding: utf-8 -*-
"""API 层契约测试：字段顺序、候选统一入口、回调投递与终态分发。

不加载模型权重，不启动服务，可直接 `python -m unittest` 运行。
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import bird_frames_service as api  # noqa: E402
import bird_cli  # noqa: E402
import doubao_report  # noqa: E402
from doubao_report import REQUIRED_FIELDS  # noqa: E402

MISSING = object()


class ResultFieldOrderTest(unittest.TestCase):
    """28 字段业务契约只允许一处定义，bestImageIndex 仅做插入。"""

    def test_field_order_matches_business_schema(self):
        expected = list(REQUIRED_FIELDS)
        expected.insert(expected.index("confidence"), "bestImageIndex")
        self.assertEqual(api.RESULT_FIELD_ORDER, expected)

    def test_order_result_drops_unknown_and_fills_defaults(self):
        report = {key: key for key in REQUIRED_FIELDS}
        report["unknownField"] = "drop-me"
        ordered = api._order_result(report, best_index=2)
        self.assertEqual(list(ordered), api.RESULT_FIELD_ORDER)
        self.assertEqual(ordered["bestImageIndex"], 2)
        self.assertNotIn("unknownField", ordered)


class FinalResultEnvelopeTest(unittest.TestCase):
    """完成态必须是 顶层关联字段 + report(28) + meta 的三段结构。"""

    TOP_KEYS = ["requestId", "status", "deviceId", "eventId", "capturedAt",
                "frameCount", "latencyMs", "canIdentify", "bird",
                "scientificName", "commonName", "confidence", "report", "meta"]
    META_KEYS = ["decisionSource", "reviewMode", "reviewTrigger", "latencyMs",
                 "qualityScore", "sceneRoute", "top5", "review", "traits",
                 "geographicPrior", "knowledgeGraph", "usage", "vlmStatus", "error"]

    def build(self):
        report = {key: key for key in REQUIRED_FIELDS}
        report.update({"categories": "Acridotheres tristis", "name": "Common Myna",
                       "canIdentify": True, "bird": True, "confidence": 0.95,
                       "qualityScore": 100})
        output = {
            "report": report,
            "meta": {
                "decisionSource": "doubao_forced_review", "reviewMode": "on",
                "reviewTrigger": "forced",
                "local": {"qualityScore": 100, "sceneRouter": {"route": "bird"},
                          "globalBioCLIP": {"reviewTop5": [
                              {"scientificName": "Acridotheres tristis"}]}},
                "doubao": {"status": "ok", "usage": {"total_tokens": 1},
                           "review": {"decision": "candidate", "candidateRank": 1},
                           "candidateTraits": {"status": "applied"},
                           "knowledgeGraph": {"applied": False}},
            },
        }
        return api._final_result(output, request_id="rid", submit={"deviceId": "feeder-001",
                                "eventId": "probe-002", "capturedAt": None},
                                frame_count=1, best_index=0, latency_ms=32601)

    def test_top_level_shape(self):
        result = self.build()
        self.assertEqual(list(result), self.TOP_KEYS)
        self.assertEqual(result["scientificName"], "Acridotheres tristis")
        self.assertEqual(result["commonName"], "Common Myna")
        self.assertEqual(result["deviceId"], "feeder-001")
        self.assertIsNone(result["capturedAt"])

    def test_report_and_meta_shape(self):
        result = self.build()
        self.assertEqual(list(result["report"]), api.RESULT_FIELD_ORDER)
        self.assertEqual(list(result["meta"]), self.META_KEYS)
        self.assertEqual(result["meta"]["top5"], ["Acridotheres tristis"])
        self.assertEqual(result["meta"]["knowledgeGraph"], {"applied": False})

    def test_local_only_output_uses_same_envelope(self):
        local = {
            "qualityScore": 90,
            "birdFocus": {"box": [1, 2, 3, 4]},
            "sceneRouter": {"route": "bird"},
            "selectedCandidate": {"scientificName": "Spinus tristis",
                                  "commonName": "American Goldfinch"},
            "risk": {"needsReview": False, "calibration": {"selectedPrecision": 0.9},
                     "blockingReasons": []},
        }
        output = api._local_only_output({"localResult": local})
        result = api._final_result(output, request_id="rid", latency_ms=10)
        self.assertEqual(list(result), self.TOP_KEYS)
        self.assertEqual(result["scientificName"], "Spinus tristis")
        self.assertEqual(result["meta"]["decisionSource"], "local_only")
        self.assertEqual(result["meta"]["vlmStatus"], "skipped")

    def test_device_id_from_header(self):
        output = {"report": {}, "meta": {}}
        result = api._final_result(output, request_id="rid",
                                   headers={"X-Device-Id": "cam-9"})
        self.assertEqual(result["deviceId"], "cam-9")

    def test_geographic_prior_summary_is_allowlisted(self):
        status = {"enabled": True, "applied": True, "reason": "support_available",
                  "cellId": "W114_S21", "supportedSpeciesCount": 12,
                  "bonuses": {"a": 0.02}}
        self.assertEqual(api._geographic_summary(status), {
            "enabled": True, "applied": True, "reason": "support_available",
            "cellId": "W114_S21", "supportedSpeciesCount": 12,
        })
        self.assertIsNone(api._geographic_summary(None))


class ReviewCandidateTest(unittest.TestCase):
    """候选兜底链只有 select_review_candidates 一处实现。"""

    def test_precedence(self):
        cases = [
            ({"reviewTop5": [{"id": "a"}], "reviewCandidates": [{"id": "b"}]}, "a"),
            ({"reviewCandidates": [{"id": "b"}], "rawTop5": [{"id": "c"}]}, "b"),
            ({"reviewTop3": [{"id": "d"}], "rawTop5": [{"id": "c"}]}, "d"),
            ({"rawTop5": [{"id": "c"}], "regionAdjustedTop5": [{"id": "e"}]}, "c"),
            ({"regionAdjustedTop5": [{"id": "e"}]}, "e"),
        ]
        for clip, expected in cases:
            with self.subTest(clip=list(clip)):
                rows = bird_cli.select_review_candidates({"globalBioCLIP": clip})
                self.assertEqual(rows[0]["id"], expected)

    def test_empty_local_result(self):
        self.assertEqual(bird_cli.select_review_candidates(None), [])


class CallbackServerTest(unittest.TestCase):
    """回调投递：首投 500、重试成功，签名与请求头可校验。"""

    def setUp(self):
        self.hits = []
        hits = self.hits

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                hits.append({"headers": dict(self.headers), "body": body})
                self.send_response(200 if len(hits) >= 2 else 500)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/cb"

        self.saved = (api.CALLBACK_ALLOW_PRIVATE, api.CALLBACK_SECRET, api.CALLBACK_MAX_ATTEMPTS)
        api.CALLBACK_ALLOW_PRIVATE = True
        api.CALLBACK_SECRET = "testsecret"
        api.CALLBACK_MAX_ATTEMPTS = 3

    def tearDown(self):
        api.CALLBACK_ALLOW_PRIVATE, api.CALLBACK_SECRET, api.CALLBACK_MAX_ATTEMPTS = self.saved
        self.server.shutdown()
        self.server.server_close()

    def test_retry_and_signature(self):
        payload = {"status": "ok", "qualityScore": 81}
        asyncio.run(api._deliver_callback("req123", self.url, payload))

        self.assertEqual(len(self.hits), 2)
        first = self.hits[0]
        digest = hmac.new(b"testsecret", first["body"], hashlib.sha256).hexdigest()
        self.assertEqual(first["headers"]["X-Bird-Signature"], "sha256=" + digest)
        self.assertEqual(first["headers"]["X-Bird-Request-Id"], "req123")
        self.assertEqual(first["headers"]["X-Bird-Event"], "job.done")
        self.assertEqual(json.loads(first["body"]), payload)

    def test_reject_invalid_callback_url(self):
        api.CALLBACK_ALLOW_PRIVATE = False
        for bad in ("http://127.0.0.1/cb", "ftp://example.com/cb", "not-a-url"):
            with self.subTest(url=bad), self.assertRaises(Exception):
                api._assert_callback_url(bad)


class CallbackDispatchTest(unittest.TestCase):
    """终态分发：done/error 各自 payload，无 callbackUrl 不投递。"""

    def run_dispatch(self, job):
        sent = []
        saved = api._deliver_callback

        async def fake_deliver(request_id, url, payload):
            sent.append((request_id, url, payload))

        api._deliver_callback = fake_deliver
        app = types.SimpleNamespace(state=types.SimpleNamespace(callback_tasks=set()))
        try:
            async def run():
                api._dispatch_callback(app, "req1", job)
                await asyncio.sleep(0)

            asyncio.run(run())
        finally:
            api._deliver_callback = saved
        return sent

    def test_done_payload(self):
        sent = self.run_dispatch({
            "callbackUrl": "https://example.com/cb", "status": "done",
            "result": {"status": "ok", "qualityScore": 90},
        })
        self.assertEqual(sent, [("req1", "https://example.com/cb",
                                 {"status": "ok", "qualityScore": 90})])

    def test_error_payload(self):
        sent = self.run_dispatch({
            "callbackUrl": "https://example.com/cb", "status": "error", "error": "boom",
        })
        self.assertEqual(sent, [("req1", "https://example.com/cb",
                                 {"requestId": "req1", "status": "error", "error": "boom"})])

    def test_no_callback_url(self):
        self.assertEqual(self.run_dispatch({"status": "done", "result": {}}), [])


class BuildObservationTest(unittest.TestCase):
    """location 入参到地理先验 observation 的薄透传与中性回退。"""

    def test_missing_location_returns_none(self):
        self.assertIsNone(api._build_observation({"capturedAt": "2026-05-01T08:00:00+08:00"}))
        self.assertIsNone(api._build_observation(None))

    def test_non_dict_location_returns_none(self):
        self.assertIsNone(api._build_observation({"location": "22.5,114.0"}))

    def test_valid_location_builds_observation(self):
        observation = api._build_observation({
            "capturedAt": "2026-05-01T08:00:00+08:00",
            "location": {"latitude": 22.5, "longitude": 114.0},
        })
        self.assertEqual(observation, {
            "enabled": True, "crs": "WGS84", "source": "device",
            "latitude": 22.5, "longitude": 114.0,
            "observedAt": "2026-05-01T08:00:00+08:00",
        })


class OpenSetSinglePassTest(unittest.TestCase):
    """single_pass_shortlist：单链路内注入 1 个 Top5 外扩展候选，不改时延。"""

    SETTINGS = {
        "enabled": True,
        "confirmationMode": "single_pass_shortlist",
        "singlePassMaximumInjectedCandidates": 1,
        "maximumBioCLIPRank": 50,
        "maximumChallengeCandidates": 5,
    }

    def local_result(self, pool):
        rows = lambda names: [
            {"rank": index, "stableId": name, "scientificName": name.title(),
             "commonName": name.title(), "family": "F", "genus": name[0].upper()}
            for index, name in enumerate(names, 1)
        ]
        return {
            "pipelineConfig": {"reviewMaxCandidates": 5, "openSetEscape": self.SETTINGS},
            "globalBioCLIP": {"reviewTop5": rows(["spiza", "passer", "pycno", "turdus"]),
                              "openSetCandidatePool": pool},
        }

    def test_injects_one_pool_candidate(self):
        pool = [
            {"rank": 5, "stableId": "mystery", "scientificName": "Mystery Bird",
             "commonName": "Mystery Bird", "family": "F", "genus": "M"},
        ]
        second, status = doubao_report.build_open_set_single_pass_local_result(
            self.local_result(pool), self.SETTINGS)
        slots = second["globalBioCLIP"]["reviewTop5"]
        self.assertEqual(len(slots), 5)
        self.assertIn("mystery", {row["stableId"] for row in slots})
        self.assertEqual([row["rank"] for row in slots], [1, 2, 3, 4, 5])
        self.assertTrue(second["_openSetEscapeSinglePass"])

    def test_empty_pool_keeps_shortlist_unchanged(self):
        second, status = doubao_report.build_open_set_single_pass_local_result(
            self.local_result([]), self.SETTINGS)
        self.assertEqual(len(second["globalBioCLIP"]["reviewTop5"]), 4)

    def test_injection_displaces_uncited_traitless_candidate_first(self):
        """挤占应优先踢掉无 trait 且不被引用的候选，保住 trait 交叉引用链。"""
        pool = [
            {"rank": 6, "stableId": "mystery", "scientificName": "Mystery Bird",
             "commonName": "Mystery Bird", "family": "F", "genus": "M"},
        ]
        local = self.local_result(pool)
        local["globalBioCLIP"]["reviewTop5"] = [
            {"rank": index, "stableId": name, "scientificName": name.title(),
             "commonName": name.title(), "family": "F", "genus": name[0].upper()}
            for index, name in enumerate(["spiza", "passer", "pycno", "turdus", "strepta"], 1)
        ]
        local["candidateTraits"] = [
            {"stableId": "passer", "scientificName": "Passer", "rank": 2,
             "traitText": "Passer, notes: smaller bill (as in Turdus)"},
            {"stableId": "turdus", "scientificName": "Turdus", "rank": 4,
             "traitText": "Turdus, notes: plain grey body"},
            {"stableId": "strepta", "scientificName": "Strepta", "rank": 5,
             "traitText": "Strepta, notes: long tail"},
        ]
        second, _ = doubao_report.build_open_set_single_pass_local_result(
            local, self.SETTINGS)
        kept = {row["stableId"] for row in second["globalBioCLIP"]["reviewTop5"]}
        self.assertIn("mystery", kept)
        self.assertIn("turdus", kept)     # 被 passer 的语料子句引用
        self.assertIn("strepta", kept)    # rank 最低但带 trait 行
        self.assertNotIn("pycno", kept)   # 无 trait 且无人引用 → 优先挤占
        self.assertEqual(
            [row["rank"] for row in second["globalBioCLIP"]["reviewTop5"]],
            [1, 2, 3, 4, 5])


class GeographicPriorTest(unittest.TestCase):
    """geographic_support 全链路：加分制、月度权重、缺失即中性、指纹校验。"""

    def setUp(self):
        import tempfile
        from geographic_prior import geographic_support
        self.geographic_support = geographic_support
        self.tmp = tempfile.TemporaryDirectory()
        asset = {
            "schemaVersion": 1, "crs": "EPSG:4326", "taxonomyVersion": "AviList-v2025b",
            "source": {"name": "unit-test", "license": "test", "commercialUseApproved": True},
            "cells": [{"id": "cell-a", "bounds": [113.9, 22.4, 114.1, 22.6],
                       "species": {"spiza-americana": {"yearRound": 0.4, "months": {"5": 0.9}}}}],
        }
        path = Path(self.tmp.name) / "grid.json"
        path.write_text(json.dumps(asset), encoding="utf-8")
        self.settings = {
            "enabled": True, "maxBonus": 0.02,
            "dataFile": str(path),
            "expectedSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self.observation = {
            "enabled": True, "crs": "WGS84", "source": "device",
            "latitude": 22.5, "longitude": 114.0,
            "observedAt": "2026-05-01T08:00:00+08:00",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_seasonal_bonus_applied(self):
        bonuses, status = self.geographic_support(
            self.settings, self.observation, root=self.tmp.name,
            taxonomy_version="AviList-v2025b")
        self.assertEqual(bonuses, {"spiza-americana": 0.02 * 0.9})
        self.assertTrue(status["applied"])
        self.assertTrue(status["seasonalApplied"])
        self.assertFalse(status["hardFiltering"])

    def test_missing_location_is_neutral(self):
        bonuses, status = self.geographic_support(
            self.settings, {"enabled": True}, root=self.tmp.name,
            taxonomy_version="AviList-v2025b")
        self.assertEqual(bonuses, {})
        self.assertEqual(status["reason"], "missing_or_invalid_wgs84_location")

    def test_sha_mismatch_is_neutral(self):
        settings = {**self.settings, "expectedSha256": "deadbeef"}
        bonuses, status = self.geographic_support(
            settings, self.observation, root=self.tmp.name,
            taxonomy_version="AviList-v2025b")
        self.assertEqual(bonuses, {})
        self.assertEqual(status["reason"], "invalid_or_unavailable_asset")

    def test_disabled_returns_neutral(self):
        bonuses, status = self.geographic_support(
            {"enabled": False}, self.observation, root=self.tmp.name,
            taxonomy_version="AviList-v2025b")
        self.assertEqual(bonuses, {})
        self.assertEqual(status["reason"], "disabled")


if __name__ == "__main__":
    unittest.main()
