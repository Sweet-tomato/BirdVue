# -*- coding: utf-8 -*-
"""Standalone acoustic bird identification service.

Completely independent from the image pipeline: the image API never accepts
or reads audio, and this service never accepts images. The only shared assets
are the BirdNET/YAMNet models and the taxonomy file.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "pipeline"))

import audio_cli  # noqa: E402
from api._common import authorize as _common_authorize, assert_public_host as _common_assert_public_host  # noqa: E402

CONFIG = Path(os.environ.get("BIRD_CONFIG", SERVICE_ROOT / "config" / "pipeline_config.json"))
API_KEY = os.environ.get("BIRD_API_KEY", "")
AUDIO_ENABLED = os.environ.get("BIRD_AUDIO_ENABLED", "1").strip().lower() not in ("0", "false", "no")
MAX_AUDIO_BYTES = int(os.environ.get("BIRD_MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
CONCURRENCY = max(1, int(os.environ.get("BIRD_AUDIO_CONCURRENCY", "2")))
PERSIST_ENABLED = os.environ.get("BIRD_PERSIST_ENABLED", "1").strip().lower() not in ("0", "false", "no")
RETENTION_DAYS = int(os.environ.get("BIRD_RETENTION_DAYS", "30"))
DATA_DIR = Path(os.environ.get("BIRD_AUDIO_DATA_DIR", SERVICE_ROOT / "data" / "audio"))
LOG_DIR = Path(os.environ.get("BIRD_LOG_DIR", SERVICE_ROOT / "logs"))
LOG_RETENTION_DAYS = int(os.environ.get("BIRD_LOG_RETENTION_DAYS", "30"))
log = logging.getLogger("bird_audio")


def _authorize(authorization: str | None) -> None:
    return _common_authorize(authorization, API_KEY)


def _setup_file_logging() -> None:
    if any(isinstance(h, TimedRotatingFileHandler) for h in log.handlers):
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        LOG_DIR / "audio.log", when="midnight", backupCount=LOG_RETENTION_DAYS, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def _assert_public_host(url: str) -> None:
    return _common_assert_public_host(url, 'audio')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


AUDIO_MAGIC = (
    (b"RIFF", b"WAVE", ".wav"),
    (b"fLaC", None, ".flac"),
    (b"OggS", None, ".ogg"),
    (b"ID3", None, ".mp3"),
)


def _sniff_audio(path: Path) -> str:
    with path.open("rb") as handle:
        head = handle.read(16)
    for prefix, suffix, ext in AUDIO_MAGIC:
        if head.startswith(prefix) and (suffix is None or head[8:12] == suffix):
            return ext
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return ".mp3"
    raise ValueError("invalid_audio_format")


def _download_audio(url: str, directory: Path) -> Path:
    """20MB 流式上限 + 魔数校验（wav/flac/ogg/mp3）。"""
    _assert_public_host(url)
    tmp = directory / ".audio.part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "bird-audio/1.0"})
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(request, timeout=60) as response, tmp.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise RuntimeError("audio_too_large")
                out.write(chunk)
        path = directory / f"audio0{_sniff_audio(tmp)}"
        os.replace(tmp, path)
        return path
    except HTTPException:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"audio_download_failed: {error}") from error
    finally:
        tmp.unlink(missing_ok=True)


def _request_dir(request_id: str) -> Path:
    stamp = time.localtime()
    return DATA_DIR / time.strftime("%Y", stamp) / time.strftime("%m%d", stamp) / request_id


def _persist(directory: Path, name: str, payload: Any) -> None:
    if not PERSIST_ENABLED:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as error:
        log.warning("persist_failed file=%s error=%s", name, error)


def _prune_expired() -> int:
    if not DATA_DIR.exists():
        return 0
    cutoff = time.time() - RETENTION_DAYS * 86400
    removed = 0
    for job in DATA_DIR.glob("*/*/*"):
        if job.is_dir() and job.stat().st_mtime < cutoff:
            shutil.rmtree(job, ignore_errors=True)
            removed += 1
    for day in DATA_DIR.glob("*/*"):
        if day.is_dir() and day.stat().st_mtime < cutoff:
            shutil.rmtree(day, ignore_errors=True)
    if removed:
        log.info("retention_pruned jobs=%d retentionDays=%d", removed, RETENTION_DAYS)
    return removed


async def _retention_loop() -> None:
    while True:
        await run_in_threadpool(_prune_expired)
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_file_logging()
    app.state.engine = None
    app.state.ready = False
    app.state.semaphore = asyncio.Semaphore(CONCURRENCY)
    if AUDIO_ENABLED:
        try:
            pipeline_config = json.loads(CONFIG.read_text(encoding="utf-8"))
            pipeline_config["_path"] = str(CONFIG)
            audio_config = dict(pipeline_config.get("audio") or {})
            audio_config["_path"] = pipeline_config["_path"]
            if audio_config.get("enabled", True):
                app.state.engine = await run_in_threadpool(audio_cli.load_engine, audio_config)
                app.state.ready = True
                log.info("audio_engine_ready model=%s noiseGate=%s",
                         app.state.engine.config.get("modelFile"),
                         app.state.engine.noise is not None)
        except Exception as error:  # noqa: BLE001
            log.error("audio_engine_unavailable error=%s", error)
    else:
        log.warning("audio_disabled by BIRD_AUDIO_ENABLED")
    app.state.retention_task = asyncio.create_task(_retention_loop())
    log.info("audio_service_ready=%s concurrency=%d", app.state.ready, CONCURRENCY)
    yield
    app.state.retention_task.cancel()
    await asyncio.gather(app.state.retention_task, return_exceptions=True)


app = FastAPI(title="Bird Audio Identification", version="1.0", lifespan=lifespan)


@app.get("/health/live")
async def live():
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    return JSONResponse(status_code=200 if app.state.ready else 503,
                        content={"status": "ready" if app.state.ready else "loading"})


@app.get("/health/state")
async def state_health():
    return JSONResponse(status_code=200, content={
        "audioReady": bool(app.state.engine),
        "enabled": AUDIO_ENABLED,
        "concurrency": CONCURRENCY,
        "dataDir": str(DATA_DIR),
    })


def _extract_audio_urls(payload: Any) -> list[str]:
    urls = payload.get("audioUrls") if isinstance(payload, dict) else None
    if not isinstance(urls, list) or len(urls) != 1 or not isinstance(urls[0], str) or not urls[0].strip():
        raise HTTPException(status_code=400, detail="expect_1_audio_url")
    return [urls[0].strip()]


@app.post("/v1/audio/identify")
async def identify_audio(
    request: Request,
    payload: Any = Body(...),
    authorization: str | None = Header(default=None),
):
    """One audio URL in, acoustic result out. No image data is accepted."""
    _authorize(authorization)
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="service_loading")
    url = _extract_audio_urls(payload)[0]
    request_id = uuid.uuid4().hex
    remote = request.client.host if request.client else "-"
    log.info("audio_request requestId=%s remote=%s url=%s", request_id, remote, url)
    directory = _request_dir(request_id)
    _persist(directory, "request.json", {
        "requestId": request_id, "audioUrls": [url],
        "receivedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    })
    started = time.time()
    try:
        async with app.state.semaphore:
            path = await run_in_threadpool(_download_audio, url, directory)
            result = await run_in_threadpool(app.state.engine.analyze, path)
            result = {"requestId": request_id, **result}
            _persist(directory, "result.json", result)
            log.info("audio_result requestId=%s status=%s top=%s latencyMs=%d",
                     request_id, result.get("status"),
                     (result.get("top5") or [{}])[0].get("scientificName"),
                     int((time.time() - started) * 1000))
            return result
    finally:
        if not PERSIST_ENABLED:
            shutil.rmtree(directory, ignore_errors=True)
