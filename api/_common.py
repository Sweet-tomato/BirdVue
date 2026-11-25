"""Shared API boundary helpers: bearer auth, SSRF guard, location normalization.

Both bird_frames_service and bird_audio_service delegate here; error detail
strings are parameterized by resource ("image"/"audio") and frozen by tests.
"""
from __future__ import annotations

import http.client
import hmac
import ipaddress
import json
import math
import socket
import urllib.parse
import urllib.request

from fastapi import HTTPException


def authorize(authorization: str | None, api_key: str) -> None:
    if not api_key:
        return
    expected = "Bearer " + api_key
    if not hmac.compare_digest(str(authorization or ""), expected):
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


def resolve_public_ip(url: str, resource: str) -> str:
    """解析并校验公网 IP；返回后续连接必须固定的 IP，防 DNS rebinding。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail=f"invalid_{resource}_url")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail=f"unresolvable_{resource}_host")
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_global:
            return str(address)
    raise HTTPException(status_code=400, detail=f"private_{resource}_host_not_allowed")


def assert_public_host(url: str, resource: str) -> None:
    """兼容旧调用：只做校验，不返回 IP。"""
    resolve_public_ip(url, resource)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, *, pinned_ip, port=None, timeout=None,
                 source_address=None, blocksize=8192):
        super().__init__(host, port=port, timeout=timeout,
                         source_address=source_address, blocksize=blocksize)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, *, pinned_ip, port=None, timeout=None,
                 source_address=None, context=None, blocksize=8192):
        super().__init__(host, port=port, timeout=timeout,
                         source_address=source_address, context=context,
                         blocksize=blocksize)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection(
            (self._pinned_ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            sock = self.sock
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, pinned_ip: str):
        super().__init__()
        self._pinned_ip = pinned_ip

    def http_open(self, req):
        return self.do_open(self._connection, req)

    def _connection(self, host, port=None, timeout=None, **kwargs):
        return _PinnedHTTPConnection(
            host, pinned_ip=self._pinned_ip, port=port, timeout=timeout, **kwargs)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, pinned_ip: str):
        super().__init__()
        self._pinned_ip = pinned_ip

    def https_open(self, req):
        return self.do_open(self._connection, req, context=self._context)

    def _connection(self, host, port=None, timeout=None, context=None, **kwargs):
        kwargs.pop("check_hostname", None)
        return _PinnedHTTPSConnection(
            host, pinned_ip=self._pinned_ip, port=port, timeout=timeout,
            context=context, **kwargs)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_public_opener(url: str, resource: str, *, no_redirect: bool = False,
                        extra_handlers: tuple = ()) -> urllib.request.OpenerDirector:
    """构建固定已校验 IP 的 opener，Host/SNI 仍使用原域名。

    调用方必须在校验后使用本 opener 发请求，避免二次 DNS 解析导致 rebinding。
    """
    pinned_ip = resolve_public_ip(url, resource)
    handlers = [urllib.request.ProxyHandler({}),
                _PinnedHTTPHandler(pinned_ip), _PinnedHTTPSHandler(pinned_ip)]
    if no_redirect:
        handlers.append(_NoRedirect())
    handlers.extend(extra_handlers)
    return urllib.request.build_opener(*handlers)
