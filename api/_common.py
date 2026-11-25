"""Shared API boundary helpers: bearer auth + SSRF public-host guard.

Both bird_frames_service and bird_audio_service delegate here; error detail
strings are parameterized by resource ("image"/"audio") and frozen by tests.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.parse

from fastapi import HTTPException


def authorize(authorization: str | None, api_key: str) -> None:
    if not api_key:
        return
    if authorization != "Bearer " + api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


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
