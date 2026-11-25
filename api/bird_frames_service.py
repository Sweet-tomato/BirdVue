# -*- coding: utf-8 -*-
"""Bird identification API: synchronous + async job queue.

Public endpoints:
  POST /v1/identify            sync: 1-3 frames -> result JSON
  POST /v1/jobs                async: 1-3 frames -> 202 + requestId
  GET  /v1/results/{id}        poll async job
  GET  /health/live|ready

Only this file is the public interface. The pipeline stays in ../pipeline.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
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
from pydantic import BaseModel, ConfigDict
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "pipeline"))

import bird_cli  # noqa: E402
from api._common import (  # noqa: E402
    authorize as _common_authorize,
    assert_public_host as _common_assert_public_host,
    build_public_opener as _common_build_public_opener,
)
from bioclip25_runtime import load_runtime as load_bundle_runtime, bundle_stats  # noqa: E402
from doubao_report import REQUIRED_FIELDS as BUSINESS_FIELDS  # noqa: E402
from state_store import build_store  # noqa: E402

CONFIG = Path(os.environ.get("BIRD_CONFIG", SERVICE_ROOT / "config" / "pipeline_config.json"))
MAX_CONCURRENCY = int(os.environ.get("BIRD_MAX_CONCURRENCY", "8"))
MAX_UPLOAD_BYTES = int(os.environ.get("BIRD_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
QUEUE_WORKERS = int(os.environ.get("BIRD_QUEUE_WORKERS", "8"))
MAX_PENDING = int(os.environ.get("BIRD_MAX_PENDING", "64"))
LOCAL_CONCURRENCY = int(os.environ.get("BIRD_LOCAL_CONCURRENCY", "2"))
REVIEW_ENABLED = os.environ.get("BIRD_REVIEW_ENABLED", "1").strip().lower() not in ("0", "false", "no")
JOB_TTL_SECONDS = int(os.environ.get("BIRD_JOB_TTL_SECONDS", "3600"))
DATA_DIR = Path(os.environ.get("BIRD_DATA_DIR", SERVICE_ROOT / "data"))
IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
API_KEY = os.environ.get("BIRD_API_KEY", "")
STATE_BACKEND = os.environ.get("BIRD_STATE_BACKEND", "memory").strip().lower()
PERSIST_ENABLED = os.environ.get("BIRD_PERSIST_ENABLED", "1").strip().lower() not in ("0", "false", "no")
RETENTION_DAYS = int(os.environ.get("BIRD_RETENTION_DAYS", "30"))
LOG_DIR = Path(os.environ.get("BIRD_LOG_DIR", SERVICE_ROOT / "logs"))
LOG_RETENTION_DAYS = int(os.environ.get("BIRD_LOG_RETENTION_DAYS", "30"))
log = logging.getLogger("bird_service")
CPU_THREADS = int(os.environ.get("BIRD_CPU_THREADS", "4"))
os.environ.setdefault("BIRD_CPU_THREADS", str(CPU_THREADS))
CALLBACK_TIMEOUT_SECONDS = float(os.environ.get("BIRD_CALLBACK_TIMEOUT_SECONDS", "10"))
CALLBACK_MAX_ATTEMPTS = max(1, int(os.environ.get("BIRD_CALLBACK_MAX_ATTEMPTS", "3")))
CALLBACK_SECRET = os.environ.get("BIRD_CALLBACK_SECRET", "")
CALLBACK_ALLOW_PRIVATE = os.environ.get("BIRD_CALLBACK_ALLOW_PRIVATE", "0").strip().lower() in ("1", "true", "yes")


def _authorize(authorization: str | None) -> None:
    return _common_authorize(authorization, API_KEY)


def _top5_names(local: dict[str, Any] | None) -> list[str]:
    """本地 BioCLIP 候选前 5 的学名，跟随最终结果一起返回并落日志。"""
    candidates = bird_cli.select_review_candidates(local)
    return [c.get("scientificName") for c in candidates[:5] if c.get("scientificName")]


# 业务字段顺序与豆包复核 schema 同源；bestImageIndex 不属于业务 schema，仅在此插入。
RESULT_FIELD_ORDER = list(BUSINESS_FIELDS)
RESULT_FIELD_ORDER.insert(RESULT_FIELD_ORDER.index("confidence"), "bestImageIndex")

# 回显请求头时必须剔除鉴权类字段，避免把 API Key 原样返回给调用方。
HEADER_IGNORE = {"authorization", "cookie", "proxy-authorization"}
# 头名按分隔符拆词后命中这些词的也视为敏感（x-api-key、x-auth-token 等），
# 否则调用方自带的凭证会随结果回传并写进 request.json/result.json。
HEADER_SENSITIVE_TOKENS = {"key", "token", "secret", "auth", "credential", "session", "signature"}


def _sensitive_header(name: str) -> bool:
    parts = {part for part in re.split(r"[^a-z0-9]+", name.lower()) if part}
    return bool(parts & HEADER_SENSITIVE_TOKENS)


def _order_result(report: dict[str, Any], best_index: int = 0) -> dict[str, Any]:
    """按业务契约固定顺序输出，bestImageIndex 插在 animal 与 confidence 之间。"""
    ordered: dict[str, Any] = {}
    for key in RESULT_FIELD_ORDER:
        if key == "bestImageIndex":
            ordered[key] = int(report.get("bestImageIndex", best_index) or 0)
        elif key in report:
            ordered[key] = report[key]
    return ordered


def _device_from_headers(headers: dict[str, str] | None) -> str | None:
    """请求头兜底取设备号，兼容多种写法。"""
    lower = {key.lower(): value for key, value in (headers or {}).items()}
    return (lower.get("x-device-id") or lower.get("device-id")
            or lower.get("x-deviceid") or lower.get("x-device"))


def _final_result(output: dict[str, Any], *, request_id: str,
                  submit: dict[str, Any] | None = None,
                  headers: dict[str, str] | None = None,
                  frame_count: int = 1, best_index: int = 0,
                  latency_ms: int | None = None) -> dict[str, Any]:
    """完成态按业务契约输出：顶层关联字段 + report（28 字段）+ meta。"""
    meta = output.get("meta") or {}
    local = meta.get("local") or {}
    doubao = meta.get("doubao") or {}
    review = doubao.get("review") or {}
    report = _order_result(output.get("report") or {}, best_index)
    submit = submit or {}
    return {
        "requestId": request_id,
        "status": "ok",
        "deviceId": submit.get("deviceId") or _device_from_headers(headers),
        "eventId": submit.get("eventId"),
        "capturedAt": submit.get("capturedAt"),
        "frameCount": frame_count,
        "latencyMs": latency_ms,
        "canIdentify": report.get("canIdentify"),
        "bird": report.get("bird"),
        "scientificName": report.get("categories"),
        "commonName": report.get("name"),
        "confidence": report.get("confidence"),
        "report": report,
        "meta": {
            "decisionSource": meta.get("decisionSource"),
            "reviewMode": meta.get("reviewMode"),
            "reviewTrigger": meta.get("reviewTrigger"),
            "latencyMs": latency_ms,
            "qualityScore": report.get("qualityScore"),
            "sceneRoute": (local.get("sceneRouter") or {}).get("route"),
            "top5": _top5_names(local),
            "review": {
                "decision": review.get("decision"),
                "candidateRank": review.get("candidateRank"),
                "confidence": review.get("confidence"),
            },
            "traits": doubao.get("candidateTraits"),
            "geographicPrior": _geographic_summary(local.get("geographicPrior")),
            "knowledgeGraph": {
                "applied": bool((doubao.get("knowledgeGraph") or {}).get("applied"))
            },
            "usage": doubao.get("usage"),
            "vlmStatus": doubao.get("status"),
            "error": doubao.get("error"),
        },
    }


GEOGRAPHIC_STATUS_FIELDS = (
    "enabled", "applied", "reason", "mode", "cellId", "observationMonth",
    "supportedSpeciesCount", "seasonalApplied", "hardFiltering", "source",
)


def _geographic_summary(status: dict[str, Any] | None) -> dict[str, Any] | None:
    """仅回传地理先验判定摘要，完整候选加分不出现在响应中。"""
    if not isinstance(status, dict):
        return None
    return {key: status.get(key) for key in GEOGRAPHIC_STATUS_FIELDS if key in status}


def _submit_meta(payload: list[str] | JobSubmit) -> dict[str, Any]:
    """收集调用方提交的原始字段（files/deviceId/eventId/capturedAt 及任意扩展字段）。"""
    if isinstance(payload, list):
        return {}
    return payload.model_dump(exclude_none=True)


def _header_meta(request: Request) -> dict[str, str]:
    """回显调用方请求头，剔除鉴权/凭证类字段。"""
    return {key: value for key, value in request.headers.items()
            if key.lower() not in HEADER_IGNORE and not _sensitive_header(key)}


def _attach_submit(result: dict[str, Any], meta: dict[str, Any],
                   urls: list[str], headers: dict[str, str] | None = None) -> dict[str, Any]:
    """在标准结果之后追加提交字段、图片地址与请求头，供业务侧关联与排查。"""
    for key, value in meta.items():
        if key not in result:
            result[key] = value
    result.setdefault("imageUrls", list(urls))
    if headers:
        result["headers"] = headers
        lower = {key.lower(): value for key, value in headers.items()}
        device = (lower.get("x-device-id") or lower.get("device-id")
                  or lower.get("x-deviceid") or lower.get("x-device"))
        if device and "deviceId" not in result:
            result["deviceId"] = device
    return result


def _frame_sharpness(path: Path) -> float:
    """单帧清晰度（Laplacian 方差），与本地质量分使用同一度量。"""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    if max(image.shape) > 512:
        scale = 512 / max(image.shape)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _select_best_frame(paths: list[Path], request_id: str,
                       runtime: Any | None = None) -> tuple[int, list[Path], dict[str, Any]]:
    """选主帧：检出鸟的帧优先（检测分、检出面积），全部未检出时退回清晰度。

    仅按清晰度选帧会被纹理丰富的空镜/屏幕画面骗到，把没有鸟的帧当作主帧，
    后面检测失败会连带压低质量分、走错提示词模式。

    同时返回逐帧检测缓存，供本地阶段复用，避免同一帧重复跑 YOLO。
    """
    detection: list[dict[str, Any]] = []
    if runtime is not None:
        try:
            detection = bird_cli.detect_frame_scores(paths, runtime, include_focus=True)
        except Exception as error:  # noqa: BLE001 - 检测失败退回清晰度
            log.warning("best_frame_detect_failed requestId=%s error=%s", request_id, error)
            detection = []
    sharpness = [round(_frame_sharpness(path), 1) for path in paths]
    detected = [item for item in detection if item.get("box")]
    if detected:
        best = max(
            detected,
            key=lambda item: (float(item.get("score") or 0), float(item.get("area") or 0)),
        )
        best_index = int(best.get("index") or 0)
        reason = "detection"
    else:
        best_index = max(range(len(sharpness)), key=lambda index: sharpness[index]) if sharpness else 0
        reason = "sharpness"
    ordered = [paths[best_index], *[path for index, path in enumerate(paths) if index != best_index]]
    detection_cache: dict[str, Any] = {}
    for item in detection:
        index = int(item.get("index") or 0)
        # focusMeta 为 None 说明该帧检测抛过异常；不缓存，让本地阶段走原路重算，
        # 否则 analyze_local 会拿到 focus_meta=None 直接 AttributeError。
        if 0 <= index < len(paths) and item.get("focusMeta") is not None:
            detection_cache[str(paths[index].resolve())] = {
                "focus": item.get("focus"),
                "focusMeta": item.get("focusMeta"),
            }
    log.info("best_frame requestId=%s reason=%s detection=%s sharpness=%s bestIndex=%d order=%s",
             request_id, reason,
             [{"i": item.get("index"), "score": round(float(item.get("score") or 0), 4),
               "area": int(item.get("area") or 0)} for item in detection],
             sharpness, best_index, [path.name for path in ordered])
    return best_index, ordered, detection_cache


def _run_frames(runtime, paths: list[Path],
                observation: dict[str, Any] | None = None,
                frame_detections: dict[str, Any] | None = None) -> dict[str, Any]:
    return bird_cli.classify(
        paths[0], runtime, review_mode="on" if REVIEW_ENABLED else "off",
        extra_frames=tuple(paths[1:]), observation=observation,
        frame_detections=frame_detections,
    )


def _scene_brief(local: dict[str, Any]) -> dict[str, Any]:
    """场景分流：空镜/松鼠在本地阶段即可判出，供后端提前决策。"""
    scene = local.get("sceneRouter") or {}
    route = scene.get("route")
    focus = local.get("birdFocus") or {}
    return {
        "route": route,
        "reason": scene.get("reason"),
        "birdDetected": route == "bird",
        "birdScore": scene.get("birdScore"),
        "squirrelScore": scene.get("squirrelScore"),
        "detectionScore": focus.get("score"),
        "box": focus.get("box"),
    }


def _local_summary(prepared: dict[str, Any]) -> dict[str, Any]:
    local = prepared["localResult"]
    candidates = bird_cli.select_review_candidates(local)
    return {
        "top5": [c.get("scientificName") for c in candidates[:5]],
        "provisionalSpecies": (local.get("provisionalCandidate") or {}).get("scientificName"),
        "qualityScore": local.get("qualityScore"),
        "needsReview": (local.get("risk") or {}).get("needsReview"),
        "riskReasons": (local.get("risk") or {}).get("blockingReasons"),
        "scene": _scene_brief(local),
    }


def _local_only_output(prepared: dict[str, Any]) -> dict[str, Any]:
    """BIRD_REVIEW_ENABLED=0 时伪造与豆包一致的 output，供统一 envelope 组装。"""
    local = prepared["localResult"]
    risk = local.get("risk") or {}
    if risk.get("needsReview"):
        reasons = risk.get("blockingReasons") or ["local_risk_refusal"]
        report = bird_cli.uncertain_report(local, reasons[0])
    else:
        report = bird_cli.local_identified_report(local)
    return {
        "report": report,
        "meta": {
            "decisionSource": "local_only",
            "reviewMode": "off",
            "reviewTrigger": None,
            "local": local,
            "doubao": {
                "status": "skipped",
                "candidateTraits": {"status": "not_applicable"},
                "knowledgeGraph": {"applied": False},
            },
        },
    }


def _request_dir(kind: str, request_id: str) -> Path:
    """按提交时间分层：data/<kind>/YYYY/MMDD/<requestId>，避免单目录文件数膨胀。"""
    stamp = time.localtime()
    return (DATA_DIR / kind / time.strftime("%Y", stamp)
            / time.strftime("%m%d", stamp) / request_id)


def _prune_empty_parents(directory: Path) -> None:
    """回收 requestId 目录后顺带清掉空的日期目录，保留 DATA_DIR 本身。"""
    parent = directory.parent
    while parent != DATA_DIR and DATA_DIR in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _discard_dir(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    _prune_empty_parents(directory)


def _drop_request_dir(paths: list[Path]) -> None:
    if paths:
        _discard_dir(paths[0].parent)


def _setup_file_logging() -> None:
    """运行日志按日切分到 LOG_DIR/api.log，journald 继续保留一份。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    target = LOG_DIR / "api.log"
    for name in ("", "uvicorn.access"):
        logger = logging.getLogger(name)
        if any(getattr(handler, "_bird_file_log", False) for handler in logger.handlers):
            continue
        handler = TimedRotatingFileHandler(
            target, when="midnight", backupCount=LOG_RETENTION_DAYS, encoding="utf-8",
        )
        handler._bird_file_log = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        if name == "":
            logger.setLevel(logging.INFO)


def _persist(request_dir: Path, name: str, payload: dict[str, Any]) -> None:
    """把每单 JSON 原子写入 requestId 目录；失败只告警，不影响主流程。"""
    if not PERSIST_ENABLED:
        return
    tmp = request_dir / (name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, request_dir / name)
    except OSError as error:
        log.warning("persist_failed file=%s error=%s", name, error)


def _prune_expired_requests() -> None:
    """按天整批删除超过 BIRD_RETENTION_DAYS 的 YYYY/MMDD 目录。"""
    if not PERSIST_ENABLED or RETENTION_DAYS <= 0:
        return
    cutoff = time.time() - RETENTION_DAYS * 86400
    removed = 0
    for year_dir in (DATA_DIR / "jobs").glob("[0-9][0-9][0-9][0-9]"):
        if not year_dir.is_dir():
            continue
        for day_dir in year_dir.glob("[0-9][0-9][0-9][0-9]"):
            try:
                day_start = time.mktime(time.strptime(year_dir.name + day_dir.name, "%Y%m%d"))
            except ValueError:
                continue
            if day_start + 86400 < cutoff:
                shutil.rmtree(day_dir, ignore_errors=True)
                removed += 1
        try:
            year_dir.rmdir()
        except OSError:
            pass
    if removed:
        log.info("retention_pruned days=%d retentionDays=%d", removed, RETENTION_DAYS)


async def _retention_loop(app: FastAPI) -> None:
    while True:
        await run_in_threadpool(_prune_expired_requests)
        # 周期回收：worker 中途崩溃的 job 会永远卡在非终态，reap 成显式 error。
        try:
            reaped = app.state.jobs.reap_stale(JOB_TTL_SECONDS)
            if reaped:
                log.warning("stale_jobs_reaped count=%d ttlSeconds=%d", reaped, JOB_TTL_SECONDS)
        except Exception as error:  # noqa: BLE001
            log.warning("reap_stale_failed error=%s", error)
        await asyncio.sleep(3600)


class JobSubmit(BaseModel):
    """调用方 Java 契约：`files` 必填；其余字段原样保留并随结果回传。"""
    model_config = ConfigDict(extra="allow")

    files: list[str]
    callbackUrl: str | None = None
    deviceId: Any = None
    eventId: Any = None
    capturedAt: Any = None
    location: Any = None


def _build_observation(submit: dict[str, Any] | None) -> dict[str, Any] | None:
    """从提交的 location/capturedAt 构造地理先验 observation。

    缺失或非法时返回 None，pipeline 走中性回退（不发地理加分）。
    坐标合法性由 geographic_prior.valid_location 二次校验，此处保持薄透传。
    observedAt 用 capturedAt；非 ISO 字符串时月份权重不生效，仅 yearRound 兜底。
    """
    location = (submit or {}).get("location")
    if not isinstance(location, dict):
        return None
    return {
        "enabled": True,
        "crs": "WGS84",
        "source": "device",
        **location,
        "observedAt": (submit or {}).get("capturedAt"),
    }


def _extract_urls(payload: list[str] | JobSubmit) -> list[str]:
    urls = payload if isinstance(payload, list) else payload.files
    if not urls or len(urls) > 3:
        raise HTTPException(status_code=400, detail="expect_1_to_3_urls")
    return urls


def _assert_public_host(url: str) -> None:
    return _common_assert_public_host(url, 'image')


def _download_image(url: str, directory: Path, index: int) -> Path:
    """下载单帧：公网 IP 固定 + 10MB 流式上限 + PIL 格式探测 + verify 后落盘。"""
    tmp = directory / f".dl{index}.part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "bird-service/1.0"})
        opener = _common_build_public_opener(url, 'image', no_redirect=True)
        with opener.open(request, timeout=30) as response, tmp.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise RuntimeError("image_too_large")
                out.write(chunk)
        with Image.open(tmp) as image:
            fmt = image.format
            image.verify()
        if fmt not in IMAGE_FORMATS:
            raise RuntimeError("invalid_image")
        path = directory / f"frame{index}{IMAGE_FORMATS[fmt]}"
        os.replace(tmp, path)
        return path
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
            UnidentifiedImageError, OSError, ValueError) as error:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"frame{index} download failed: {error}") from error


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """回调不跟随 3xx，避免被重定向到内网地址绕过 SSRF 防护。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _assert_callback_url(url: str) -> None:
    """回调地址校验：仅 http/https 公网地址，防 SSRF。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail="invalid_callback_url")
    if CALLBACK_ALLOW_PRIVATE:
        return
    try:
        _assert_public_host(url)
    except HTTPException as error:
        raise HTTPException(status_code=400, detail="invalid_callback_url") from error


def _resolve_callback_url(payload: list[str] | JobSubmit, request: Request) -> str | None:
    """对象体用 callbackUrl 字段，纯数组用 ?callbackUrl= 查询参数；均可选。"""
    url = payload.callbackUrl if isinstance(payload, JobSubmit) else request.query_params.get("callbackUrl")
    if not url or not url.strip():
        return None
    url = url.strip()
    _assert_callback_url(url)
    return url


def _post_callback(url: str, body: bytes, headers: dict[str, str]) -> int:
    """同步 POST 回调；非 2xx 抛出异常，由上层重试。"""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    if CALLBACK_ALLOW_PRIVATE:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    else:
        opener = _common_build_public_opener(url, 'callback', no_redirect=True)
    with opener.open(request, timeout=CALLBACK_TIMEOUT_SECONDS) as response:
        if response.status >= 300:
            raise RuntimeError(f"callback_http_{response.status}")
        return response.status


async def _deliver_callback(request_id: str, url: str, payload: dict[str, Any]) -> None:
    """推送最终结果，指数退避重试；业务侧以 requestId 去重。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "bird-service/1.0",
        "X-Bird-Request-Id": request_id,
        "X-Bird-Event": "job.error" if payload.get("status") == "error" else "job.done",
    }
    if CALLBACK_SECRET:
        digest = hmac.new(CALLBACK_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers["X-Bird-Signature"] = "sha256=" + digest
    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        started = time.time()
        try:
            status = await run_in_threadpool(_post_callback, url, body, headers)
            log.info("callback_delivered requestId=%s status=%d attempt=%d latencyMs=%d",
                     request_id, status, attempt, int((time.time() - started) * 1000))
            return
        except Exception as error:  # noqa: BLE001
            log.warning("callback_attempt_failed requestId=%s attempt=%d error=%s",
                        request_id, attempt, error)
            if attempt < CALLBACK_MAX_ATTEMPTS:
                await asyncio.sleep(min(5 * 2 ** (attempt - 1), 60))
    log.error("callback_gave_up requestId=%s attempts=%d", request_id, CALLBACK_MAX_ATTEMPTS)


def _dispatch_callback(app: FastAPI, request_id: str, job: dict[str, Any]) -> None:
    """终态后异步推送，独立任务执行，不阻塞 worker。"""
    url = job.get("callbackUrl")
    if not url:
        return
    if job.get("status") == "done":
        payload = job.get("result") or {}
    else:
        payload = {"requestId": request_id, "status": "error", "error": job.get("error")}
    try:
        task = asyncio.create_task(_deliver_callback(request_id, url, payload))
    except RuntimeError:
        return
    app.state.callback_tasks.add(task)
    task.add_done_callback(app.state.callback_tasks.discard)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_file_logging()
    app.state.runtime = None
    app.state.ready = False
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    app.state.local_semaphore = asyncio.Semaphore(LOCAL_CONCURRENCY)
    # Job metadata is shared when BIRD_STATE_BACKEND=sqlite|redis; the
    # local-stage `prepared` object stays in this worker's memory and is
    # consumed by this worker's own queue. Default `memory` is identical
    # to the baseline dict for single-worker deployments.
    app.state.jobs = build_store(STATE_BACKEND, data_dir=DATA_DIR)
    app.state.jobs.reap_stale(JOB_TTL_SECONDS)
    app.state.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_PENDING)
    app.state.worker_tasks = []
    app.state.callback_tasks = set()
    app.state.runtime = await run_in_threadpool(load_bundle_runtime, CONFIG)
    app.state.ready = True
    for _ in range(max(1, QUEUE_WORKERS)):
        app.state.worker_tasks.append(asyncio.create_task(_worker(app)))
    app.state.retention_task = asyncio.create_task(_retention_loop(app))
    log.info("service_ready workers=%d persist=%s retentionDays=%d logDir=%s",
             max(1, QUEUE_WORKERS), PERSIST_ENABLED, RETENTION_DAYS, LOG_DIR)
    yield
    for task in app.state.worker_tasks:
        task.cancel()
    app.state.retention_task.cancel()
    await asyncio.gather(*app.state.worker_tasks, app.state.retention_task,
                         return_exceptions=True)
    for task in list(app.state.callback_tasks):
        task.cancel()
    await asyncio.gather(*app.state.callback_tasks, return_exceptions=True)
    app.state.jobs.close()


async def _worker(app: FastAPI):
    while True:
        request_id = await app.state.queue.get()
        job = app.state.jobs.get(request_id)
        try:
            if not job:
                continue
            job["status"] = "processing"
            job["updatedAt"] = time.time()
            # 后台下载阶段：URL 由后端提供，此处才拉图（公网校验 + 10MB + 格式探测）。
            directory = Path(job["dir"])
            paths = list(await asyncio.gather(*[
                run_in_threadpool(_download_image, url, directory, index)
                for index, url in enumerate(job["imageUrls"])
            ]))
            job["paths"] = [str(p) for p in paths]
            # _select_best_frame 内含逐帧 ONNX 检测，必须走线程池，否则阻塞事件循环。
            best_index, frames, frame_detections = await run_in_threadpool(
                _select_best_frame, paths, request_id, app.state.runtime)
            job["bestImageIndex"] = best_index
            # 本地阶段（检测/裁剪/质量评估）放在后台执行，提交接口不再等待它。
            async with app.state.local_semaphore:
                prepared = await run_in_threadpool(
                    bird_cli.prepare_local, frames[0], app.state.runtime,
                    extra_frames=tuple(frames[1:]),
                    observation=_build_observation(job.get("submit")),
                    frame_detections=frame_detections,
                )
            job["prepared"] = prepared
            job["local"] = _local_summary(prepared)
            log.info("local_ready requestId=%s top5=%s scene=%s", request_id,
                     json.dumps(job["local"].get("top5"), ensure_ascii=False),
                     json.dumps(job["local"].get("scene"), ensure_ascii=False))
            job["updatedAt"] = time.time()
            if REVIEW_ENABLED:
                output = await run_in_threadpool(
                    bird_cli.review_prepared, prepared, app.state.runtime
                )
                job["result"] = _final_result(
                    output, request_id=request_id,
                    submit=job.get("submit"), headers=job.get("headers"),
                    frame_count=len(paths),
                    best_index=job.get("bestImageIndex") or 0,
                    latency_ms=int((time.time() - job.get("createdAt", time.time())) * 1000),
                )
            else:
                # 模型部署端模式：BIRD_REVIEW_ENABLED=0，只产出本地 Top5 与性状子句。
                job["result"] = _final_result(
                    _local_only_output(prepared), request_id=request_id,
                    submit=job.get("submit"), headers=job.get("headers"),
                    frame_count=len(paths),
                    best_index=job.get("bestImageIndex") or 0,
                    latency_ms=int((time.time() - job.get("createdAt", time.time())) * 1000),
                )
            log.info("result_ready requestId=%s kind=jobs result=%s",
                     request_id, json.dumps(job["result"], ensure_ascii=False))
            job["status"] = "done"
            job["prepared"] = None
            job["updatedAt"] = time.time()
            _persist(paths[0].parent, "result.json", job["result"])
            _dispatch_callback(app, request_id, job)
        except Exception as error:  # noqa: BLE001
            if job:
                job["status"] = "error"
                job["error"] = f"{type(error).__name__}: {error}"
                job["updatedAt"] = time.time()
                _persist(Path(job["dir"]), "error.json", {
                    "requestId": request_id,
                    "status": "error",
                    "error": job["error"],
                    "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                })
                log.error("job_failed requestId=%s error=%s", request_id, job["error"])
                _dispatch_callback(app, request_id, job)
        finally:
            if job:
                log.info("job_finished requestId=%s status=%s totalMs=%d",
                         request_id, job.get("status"),
                         int((time.time() - job.get("createdAt", time.time())) * 1000))
                if not PERSIST_ENABLED:
                    _discard_dir(Path(job["dir"]))
            app.state.queue.task_done()


def _cleanup_jobs(app: FastAPI) -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for request_id, job in list(app.state.jobs.items()):
        if job.get("updatedAt", 0) < cutoff:
            if not PERSIST_ENABLED:
                _discard_dir(Path(job["dir"]))
            app.state.jobs.pop(request_id, None)


app = FastAPI(title="Bird Identification Service", version="1", lifespan=lifespan)

# 鉴权必须在中间件做：FastAPI 在进入 handler 前就完成 body 校验，
# 放在 handler 里会让未鉴权的非法 body 先得到 422（泄露字段结构）。
AUTH_PREFIXES = ("/v1/",)
SCHEMA_PATHS = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}


@app.middleware("http")
async def require_auth(request, call_next):
    path = request.url.path
    if API_KEY and (path.startswith(AUTH_PREFIXES) or path in SCHEMA_PATHS):
        try:
            _authorize(request.headers.get("authorization"))
        except HTTPException:
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


@app.get("/health/live")
async def live():
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    return JSONResponse(status_code=200 if app.state.ready else 503,
                        content={"status": "ready" if app.state.ready else "loading"})


@app.get("/health/state")
async def state_health():
    """Read-only production diagnostics (no contract change)."""
    return JSONResponse(status_code=200, content={
        "store": app.state.jobs.describe(),
        "cpuThreads": CPU_THREADS,
        "detectDedup": bool(bird_cli.DETECT_DEDUP),
        "sharedBundles": bundle_stats(),
    })


@app.post("/v1/identify")
async def identify(
    request: Request,
    payload: list[str] | JobSubmit = Body(...),
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="service_loading")
    urls = _extract_urls(payload)
    submit_meta = _submit_meta(payload)
    header_meta = _header_meta(request)
    request_id = uuid.uuid4().hex
    log.info("request_received requestId=%s kind=identify remote=%s urls=%s",
             request_id, request.client.host if request.client else "-",
             json.dumps(urls, ensure_ascii=False))
    directory = _request_dir("sync", request_id)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        paths = []
        for index, url in enumerate(urls):
            paths.append(await run_in_threadpool(_download_image, url, directory, index))
        # _select_best_frame 内含逐帧 ONNX 检测，走线程池避免阻塞事件循环。
        best_index, frames, frame_detections = await run_in_threadpool(
            _select_best_frame, paths, request_id, app.state.runtime)
    except HTTPException:
        _discard_dir(directory)
        raise
    except Exception as error:
        _discard_dir(directory)
        raise HTTPException(status_code=400, detail=f"image_download_failed: {error}") from error
    started = time.time()
    try:
        async with app.state.semaphore:
            output = await run_in_threadpool(
                _run_frames, app.state.runtime, frames,
                _build_observation(submit_meta), frame_detections
            )
            result = _final_result(
                output, request_id=request_id, submit=submit_meta,
                headers=header_meta, frame_count=len(paths), best_index=best_index,
                latency_ms=int((time.time() - started) * 1000),
            )
            log.info("result_ready requestId=%s kind=identify result=%s",
                     request_id, json.dumps(result, ensure_ascii=False))
            return result
    finally:
        _drop_request_dir(paths)


@app.post("/v1/jobs", status_code=202)
async def create_job_async(
    request: Request,
    payload: list[str] | JobSubmit = Body(...),
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="service_loading")
    if app.state.queue.full():
        raise HTTPException(status_code=429, detail="queue_full")
    urls = _extract_urls(payload)
    callback_url = _resolve_callback_url(payload, request)
    submit_meta = _submit_meta(payload)
    submit_meta.pop("callbackUrl", None)
    header_meta = _header_meta(request)
    _cleanup_jobs(app)
    request_id = uuid.uuid4().hex
    directory = _request_dir("jobs", request_id)
    directory.mkdir(parents=True, exist_ok=True)
    _persist(directory, "request.json", {
        "requestId": request_id, "imageUrls": list(urls),
        "frameCount": len(urls), "submit": submit_meta, "headers": header_meta,
        "receivedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    })
    app.state.jobs[request_id] = {
        "requestId": request_id, "status": "reviewing",
        "createdAt": time.time(), "updatedAt": time.time(),
        "dir": str(directory), "imageUrls": list(urls),
        "paths": [], "frameCount": len(urls), "local": None, "prepared": None,
        "callbackUrl": callback_url,
        "submit": submit_meta, "headers": header_meta, "result": None, "error": None,
    }
    try:
        app.state.queue.put_nowait(request_id)
    except asyncio.QueueFull:
        app.state.jobs.pop(request_id, None)
        _discard_dir(directory)
        raise HTTPException(status_code=429, detail="queue_full")
    return {"requestId": request_id, "status": "reviewing", "local": None,
            "resultUrl": f"/v1/results/{request_id}", "pollAfterMs": 2000}


_LONGPOLL_SECONDS = float(os.getenv("BIRD_RESULT_LONGPOLL_SECONDS", "15"))


@app.get("/v1/results/{request_id}")
async def get_result(request: Request, request_id: str,
                     authorization: str | None = Header(default=None)):
    _authorize(authorization)
    started = time.time()
    remote = request.client.host if request.client else "-"
    job = app.state.jobs.get(request_id)
    if not job:
        log.info("result_fetch requestId=%s status=not_found bytes=0 waitMs=%d remote=%s",
                 request_id, int((time.time() - started) * 1000), remote)
        raise HTTPException(status_code=404, detail="job_not_found")
    # 短时轮询：未完成时等待一小段时间后返回当前状态；调用方按 pollAfterMs 继续轮询
    # 或使用 callbackUrl 接收终态，避免单个请求长期占用连接。
    deadline = time.time() + _LONGPOLL_SECONDS
    while job["status"] not in ("done", "error") and time.time() < deadline:
        await asyncio.sleep(0.3)
        # TrackedJob 是快照，必须重新取一次才能看到后台写入的状态。
        job = app.state.jobs.get(request_id) or job
    if job["status"] == "done":
        # 完成态直接返回最终结果，结构与 /v1/identify 完全一致（status=ok）。
        response = JSONResponse(status_code=200, content=job["result"])
    else:
        response = JSONResponse(status_code=200, content=_attach_submit({
            "requestId": request_id,
            "status": job["status"],
            "frameCount": job.get("frameCount"),
            "local": job.get("local"),
            "error": job.get("error"),
        }, job.get("submit") or {}, job.get("imageUrls") or [], job.get("headers") or {}))
    # 读取埋点：确认调用方是否真的取走了结果、拿到时任务是否已完成。
    log.info("result_fetch requestId=%s status=%s bytes=%d waitMs=%d remote=%s",
             request_id, job["status"], len(response.body),
             int((time.time() - started) * 1000), remote)
    return response
