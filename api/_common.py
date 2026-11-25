"""Shared API boundary helpers: bearer auth + SSRF public-host guard.

Both bird_frames_service and bird_audio_service delegate here; error detail
strings are parameterized by resource ("image"/"audio") and frozen by tests.
"""
from __future__ import annotations

import ipaddress
import json
import math
import socket
import urllib.parse

from fastapi import HTTPException


def authorize(authorization: str | None, api_key: str) -> None:
    if not api_key:
        return
    if authorization != "Bearer " + api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


def _json_object(value, depth: int = 0):
    """Parse a dict, JSON string, or double-encoded JSON string."""
    if depth > 3:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        return _json_object(parsed, depth + 1)
    return None


def _coordinate(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _location_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("location", "gpsFence", "gps_fence", "attachment"):
        if payload.get(key) is None:
            continue
        data = _json_object(payload.get(key))
        if not isinstance(data, dict):
            continue
        latitude = _coordinate(data.get("latitude", data.get("lat")))
        longitude = _coordinate(data.get("longitude", data.get("lng", data.get("lon"))))
        if latitude is None or longitude is None:
            continue
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            continue
        return {"latitude": latitude, "longitude": longitude, "source": key}
    return None


def normalize_location(payload):
    """Extract WGS84 coordinates from the device payload shapes used in the field.

    Accepted containers (first hit wins): ``location`` (``{latitude, longitude}``),
    ``gpsFence`` (``{latitude, longitude, radius}``), ``attachment``
    (``{moonphase, batteryPower, latitude, longitude, timestamp}``). Values may be
    numbers, numeric strings, JSON strings, or double-encoded JSON strings. The
    same keys are also checked inside a ``deviceInfo`` object.
    """
    found = _location_from_payload(payload)
    if found is not None:
        return found
    if isinstance(payload, dict):
        device_info = _json_object(payload.get("deviceInfo"))
        if isinstance(device_info, dict):
            return _location_from_payload(device_info)
    return None


def assert_public_host(url: str, resource: str) -> None:
    """SSRF 防护：解析 DNS，拒绝私网/环回/链路本地等非公网地址。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail=f"invalid_{resource}_url")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail=f"unresolvable_{resource}_host")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise HTTPException(status_code=400, detail=f"private_{resource}_host_not_allowed")
