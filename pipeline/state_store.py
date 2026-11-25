# -*- coding: utf-8 -*-
"""Cross-process job state for a multi-worker deployment.

Baseline keeps jobs in ``app.state.jobs`` (a plain dict) and the review queue
in an ``asyncio.Queue``; both are process-local. With ``uvicorn --workers N``,
a job created in worker A cannot be polled from worker B, so
``GET /v1/results/{id}`` returns 404.

This module externalises only the *serialisable* job metadata. ``prepared``
(the local-stage result, carrying PIL images / numpy arrays and the pipeline
config) deliberately stays in the creating worker's memory: that worker also
owns the local review queue, so no heavy object ever crosses a process
boundary. Two consequences are handled explicitly:

* The baseline ``_worker`` mutates the dict in place (``job["status"]=...``).
  A plain JSON round-trip would silently lose those mutations, so values are
  returned as :class:`TrackedJob` which writes through on every assignment.
* If a worker dies, its in-memory ``prepared`` is gone and the job would sit
  in ``reviewing`` forever. :meth:`JobStore.reap_stale` converts jobs that
  have been non-terminal for too long into an explicit error.

Backends
--------
``memory``  baseline behaviour (default), single worker only.
``sqlite``  shared file, standard library only, WAL mode, a few workers.
``redis``   shared Redis, needs the optional ``redis`` package.

Set ``BIRD_STATE_BACKEND`` to select one. Nothing here changes the public
HTTP contract.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Callable

# Local-stage products are process-local and must never be serialised.
EPHEMERAL_KEYS = frozenset({"prepared"})
TERMINAL_STATES = frozenset({"done", "error"})


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalar / 0-d array
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    raise TypeError("job payload contains a non-serialisable value: %r" % (type(value),))


def _encode(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      default=_json_default)


class TrackedJob(dict):
    """A dict that persists every mutation of a serialisable key.

    Assignments to :data:`EPHEMERAL_KEYS` stay in the local process only.
    Nested dicts are NOT tracked: `job["result"]["x"] = 1` mutates in place
    without persisting; use update_fields() or replace the nested value.
    """

    def __init__(self, store: "JobStore", request_id: str,
                 data: dict | None = None, ephemeral: dict | None = None):
        super().__init__(data or {})
        self._store = store
        self._id = request_id
        self._ephemeral = dict(ephemeral or {})

    def persistent(self) -> dict:
        # dict.copy on a subclass returns a plain dict of the underlying items.
        return dict.copy(self)

    def __setitem__(self, key, value):
        if key in EPHEMERAL_KEYS:
            self._ephemeral[key] = value
            # Keep this worker's local cache in sync too.
            with self._store._lock:
                local = self._store._ephemeral.setdefault(self._id, {})
                local[key] = value
            return
        super().__setitem__(key, value)
        self._store.save(self._id, self.persistent())

    def update_fields(self, **fields):
        """Batch write: one persistence call for all fields (single commit).

        Preferred over repeated assignments for status+timestamp updates.
        Nested dict mutation (`job["result"]["x"] = 1`) is never persisted;
        replace nested values wholesale or pass them here.
        """
        persistent_changed = False
        for key, value in fields.items():
            if key in EPHEMERAL_KEYS:
                self._ephemeral[key] = value
                with self._store._lock:
                    local = self._store._ephemeral.setdefault(self._id, {})
                    local[key] = value
            else:
                super().__setitem__(key, value)
                persistent_changed = True
        if persistent_changed:
            self._store.save(self._id, self.persistent())

    def __getitem__(self, key):
        if key in EPHEMERAL_KEYS:
            return self._ephemeral[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in EPHEMERAL_KEYS:
            return self._ephemeral.get(key, default)
        return super().get(key, default)

    def __contains__(self, key):
        if key in EPHEMERAL_KEYS:
            return key in self._ephemeral
        return super().__contains__(key)


class JobStore(MutableMapping):
    """dict-like view over a shared backend; values are :class:`TrackedJob`."""

    def __init__(self, owner: str | None = None):
        self.owner = owner or ("%d:%s" % (os.getpid(), uuid.uuid4().hex[:8]))
        self._lock = threading.RLock()
        self._ephemeral: dict[str, dict] = {}

    # ---- backend hooks -------------------------------------------------
    def _read(self, request_id: str) -> dict | None:
        raise NotImplementedError

    def _write(self, request_id: str, payload: dict, owner: str) -> None:
        raise NotImplementedError

    def _delete(self, request_id: str) -> None:
        raise NotImplementedError

    def _all(self) -> list[tuple[str, dict, str]]:
        """Return [(id, payload, owner)] for every stored job."""
        raise NotImplementedError

    # ---- Mapping interface ---------------------------------------------
    def _wrap(self, request_id: str, payload: dict) -> TrackedJob:
        with self._lock:
            ephemeral = dict(self._ephemeral.get(request_id, {}))
        return TrackedJob(self, request_id, payload, ephemeral)

    def __getitem__(self, request_id: str) -> TrackedJob:
        payload = self._read(request_id)
        if payload is None:
            raise KeyError(request_id)
        return self._wrap(request_id, payload)

    def __setitem__(self, request_id: str, job: dict) -> None:
        raw = dict(job)
        payload = {k: v for k, v in raw.items() if k not in EPHEMERAL_KEYS}
        ephemeral = {k: v for k, v in raw.items() if k in EPHEMERAL_KEYS}
        self._write(request_id, payload, self.owner)
        with self._lock:
            self._ephemeral[request_id] = ephemeral

    def __delitem__(self, request_id: str) -> None:
        self._delete(request_id)
        with self._lock:
            self._ephemeral.pop(request_id, None)

    def __iter__(self):
        return iter([key for key, _, _ in self._all()])

    def __len__(self) -> int:
        return len(self._all())

    def get(self, request_id: str, default=None):
        payload = self._read(request_id)
        return default if payload is None else self._wrap(request_id, payload)

    def pop(self, request_id: str, default=None):
        payload = self._read(request_id)
        if payload is None:
            return default
        job = self._wrap(request_id, payload)
        self._delete(request_id)
        with self._lock:
            self._ephemeral.pop(request_id, None)
        return job

    def items(self):
        return [(key, self._wrap(key, payload))
                for key, payload, _ in self._all()]

    def save(self, request_id: str, payload: dict) -> None:
        """Write-through used by :class:`TrackedJob`."""
        self._write(request_id, payload, self.owner)

    def reap_stale(self, max_age_s: float, now: float | None = None) -> int:
        """Mark non-terminal jobs that lost their worker as explicit errors.

        Multi-worker deployments keep ``prepared`` in the creating process. If
        that process exits, nothing will ever finish the job. Call this on
        startup (and optionally periodically) so polling gets ``error`` rather
        than an endless ``reviewing``.
        """
        now = time.time() if now is None else now
        cutoff = now - max_age_s
        changed = 0
        for request_id, payload, _ in self._all():
            if payload.get("status") in TERMINAL_STATES:
                continue
            if float(payload.get("updatedAt", 0)) >= cutoff:
                continue
            payload["status"] = "error"
            payload["error"] = "worker_lost_or_review_timeout"
            payload["updatedAt"] = now
            self._write(request_id, payload, self.owner)
            changed += 1
        return changed

    def close(self) -> None:
        """Release backend resources. No-op for in-memory stores."""
        return None

    def describe(self) -> dict:
        return {"backend": type(self).__name__, "owner": self.owner,
                "jobs": len(self)}


class MemoryJobStore(JobStore):
    """Single-process backend, behaviourally identical to the baseline dict."""

    def __init__(self, owner: str | None = None):
        super().__init__(owner)
        self._data: dict[str, tuple[dict, str]] = {}

    def _read(self, request_id):
        with self._lock:
            row = self._data.get(request_id)
        return None if row is None else dict(row[0])

    def _write(self, request_id, payload, owner):
        with self._lock:
            self._data[request_id] = (dict(payload), owner)

    def _delete(self, request_id):
        with self._lock:
            self._data.pop(request_id, None)

    def _all(self):
        with self._lock:
            return [(k, dict(v), o) for k, (v, o) in self._data.items()]


class SQLiteJobStore(JobStore):
    """Shared-file backend. WAL + busy_timeout make it safe across workers."""

    def __init__(self, path, owner: str | None = None, timeout: float = 10.0):
        super().__init__(owner)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False,
                                   timeout=timeout)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=%d" % int(timeout * 1000))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "id TEXT PRIMARY KEY, payload TEXT NOT NULL, owner TEXT NOT NULL, "
            "status TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        self._db.commit()

    def _read(self, request_id):
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM jobs WHERE id=?", (request_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def _write(self, request_id, payload, owner):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jobs "
                "(id,payload,owner,status,updated_at) VALUES (?,?,?,?,?)",
                (request_id, _encode(payload), owner,
                 str(payload.get("status", "")),
                 float(payload.get("updatedAt", time.time()))),
            )
            self._db.commit()

    def _delete(self, request_id):
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id=?", (request_id,))
            self._db.commit()

    def _all(self):
        with self._lock:
            rows = self._db.execute("SELECT id,payload,owner FROM jobs").fetchall()
        return [(k, json.loads(p), o) for k, p, o in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def describe(self):
        info = super().describe()
        info["path"] = str(self.path)
        return info


class RedisJobStore(JobStore):
    """Shared Redis backend (optional ``redis`` dependency)."""

    PREFIX = "bird:job:"
    INDEX = "bird:job:index"

    def __init__(self, url: str, owner: str | None = None, **kwargs):
        super().__init__(owner)
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "BIRD_STATE_BACKEND=redis requires the 'redis' package"
            ) from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True, **kwargs)

    def _read(self, request_id):
        raw = self._redis.get(self.PREFIX + request_id)
        return None if raw is None else json.loads(raw)

    def _write(self, request_id, payload, owner):
        payload = {**payload, "owner": owner}
        pipe = self._redis.pipeline()
        pipe.set(self.PREFIX + request_id, _encode(payload))
        pipe.sadd(self.INDEX, request_id)
        pipe.execute()

    def _delete(self, request_id):
        pipe = self._redis.pipeline()
        pipe.delete(self.PREFIX + request_id)
        pipe.srem(self.INDEX, request_id)
        pipe.execute()

    def _all(self):
        out = []
        for request_id in self._redis.smembers(self.INDEX):
            raw = self._redis.get(self.PREFIX + request_id)
            if raw is None:
                self._redis.srem(self.INDEX, request_id)
                continue
            payload = json.loads(raw)
            out.append((request_id, payload, payload.get("owner", "")))
        return out

    def close(self) -> None:
        self._redis.close()

    def describe(self):
        info = super().describe()
        info["index"] = self.INDEX
        return info


def build_store(backend: str | None = None, *, sqlite_path: str | None = None,
                redis_url: str | None = None, owner: str | None = None,
                data_dir: str | Path | None = None) -> JobStore:
    """Build the configured store. ``memory`` keeps the baseline behaviour."""
    backend = (backend or os.environ.get("BIRD_STATE_BACKEND", "memory")).strip().lower()
    if backend == "memory":
        return MemoryJobStore(owner)
    if backend == "sqlite":
        path = sqlite_path or os.environ.get("BIRD_STATE_SQLITE_PATH")
        if not path:
            root = Path(data_dir) if data_dir else Path(
                os.environ.get("BIRD_DATA_DIR", "."))
            path = str(root / "jobs.sqlite3")
        return SQLiteJobStore(path, owner)
    if backend == "redis":
        return RedisJobStore(
            redis_url or os.environ.get("BIRD_STATE_REDIS_URL", "redis://127.0.0.1:6379/0"),
            owner,
        )
    raise ValueError("BIRD_STATE_BACKEND must be memory, sqlite or redis")
