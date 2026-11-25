# -*- coding: utf-8 -*-
"""Tests for cross-process job state (no model weights or server needed)."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from state_store import (  # noqa: E402
    EPHEMERAL_KEYS, MemoryJobStore, SQLiteJobStore, build_store,
)


def sample_job(request_id="abc", status="reviewing", updated=None):
    return {
        "requestId": request_id,
        "status": status,
        "createdAt": 100.0,
        "updatedAt": time.time() if updated is None else updated,
        "paths": ["/tmp/a.jpg"],
        "deviceId": "dev-1",
        "eventId": "",
        "capturedAt": "",
        "frameCount": 1,
        "local": {"top5": ["Turdus merula"]},
        "prepared": {"original": object()},  # deliberately NOT serialisable
        "result": None,
        "error": None,
    }


class MemoryBackendTests(unittest.TestCase):
    def test_write_through_on_in_place_mutation(self):
        store = MemoryJobStore()
        store["abc"] = sample_job()
        # This mirrors the baseline _worker: mutate the dict in place.
        job = store.get("abc")
        job["status"] = "processing"
        job["status"] = "done"
        job["result"] = {"report": {"bird": True}}
        fresh = store.get("abc")
        self.assertEqual(fresh["status"], "done")
        self.assertEqual(fresh["result"], {"report": {"bird": True}})

    def test_prepared_stays_local_but_is_readable_by_owner(self):
        store = MemoryJobStore()
        prepared = {"original": "PIL-image-placeholder"}
        store["abc"] = sample_job()
        store.get("abc")["prepared"] = prepared
        self.assertIs(store.get("abc")["prepared"], prepared)
        self.assertNotIn("prepared", store.get("abc").persistent())
        self.assertIn("prepared", EPHEMERAL_KEYS)

    def test_items_pop_and_len(self):
        store = MemoryJobStore()
        store["a"] = sample_job("a")
        store["b"] = sample_job("b")
        self.assertEqual(len(store), 2)
        self.assertEqual(sorted(k for k, _ in store.items()), ["a", "b"])
        self.assertIsNotNone(store.pop("a"))
        self.assertEqual(len(store), 1)
        self.assertIsNone(store.get("missing"))
        self.assertEqual(store.pop("missing", "fallback"), "fallback")

    def test_reap_stale_only_marks_non_terminal(self):
        store = MemoryJobStore()
        store["old"] = sample_job("old", "reviewing", updated=time.time() - 7200)
        store["done"] = sample_job("done", "done", updated=time.time() - 7200)
        store["fresh"] = sample_job("fresh", "reviewing", updated=time.time())
        changed = store.reap_stale(3600)
        self.assertEqual(changed, 1)
        self.assertEqual(store.get("old")["status"], "error")
        self.assertEqual(store.get("old")["error"], "worker_lost_or_review_timeout")
        self.assertEqual(store.get("done")["status"], "done")
        self.assertEqual(store.get("fresh")["status"], "reviewing")


class SQLiteBackendTests(unittest.TestCase):
    def test_cross_worker_result_visibility(self):
        """The exact multi-worker 404 scenario: A creates, B polls."""
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "jobs.sqlite3")
            worker_a = SQLiteJobStore(db, owner="worker-A")
            worker_b = SQLiteJobStore(db, owner="worker-B")
            worker_a["job-1"] = sample_job("job-1")

            # B can see the job immediately after A submits it.
            self.assertIsNotNone(worker_b.get("job-1"))
            self.assertEqual(worker_b.get("job-1")["status"], "reviewing")
            # ... but B must not see A's in-memory local-stage object.
            self.assertNotIn("prepared", worker_b.get("job-1").persistent())

            # A finishes the review and writes through.
            job = worker_a.get("job-1")
            job["status"] = "done"
            job["result"] = {"report": {"bird": True}}
            job["prepared"] = None
            job["updatedAt"] = time.time()
            polled = worker_b.get("job-1")
            self.assertEqual(polled["status"], "done")
            self.assertEqual(polled["result"], {"report": {"bird": True}})
            worker_a.close()
            worker_b.close()

    def test_reap_and_cleanup_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "jobs.sqlite3")
            a = SQLiteJobStore(db, owner="worker-A")
            a["stale"] = sample_job("stale", "reviewing", updated=time.time() - 7200)
            b = SQLiteJobStore(db, owner="worker-B")
            self.assertEqual(b.reap_stale(3600), 1)
            self.assertEqual(a.get("stale")["status"], "error")
            b.pop("stale")
            self.assertIsNone(a.get("stale"))
            a.close()
            b.close()

    def test_non_serialisable_payload_is_rejected_loudly(self):
        store = MemoryJobStore()
        from state_store import _encode
        with self.assertRaises(TypeError):
            _encode({"image": object()})


class FactoryTests(unittest.TestCase):
    def test_build_store_backends(self):
        self.assertIsInstance(build_store("memory"), MemoryJobStore)
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store("sqlite", sqlite_path=str(Path(tmp) / "j.db"))
            self.assertIsInstance(store, SQLiteJobStore)
            self.assertEqual(store.describe()["backend"], "SQLiteJobStore")
            store.close()
        with self.assertRaises(ValueError):
            build_store("unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
