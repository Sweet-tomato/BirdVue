"""Offline, opt-in geographic support for review retrieval, never visual evidence.

Asset schema: {schemaVersion: 1, crs: "EPSG:4326", taxonomyVersion: str,
 source: {name: str, license: str, commercialUseApproved: true},
 cells: [{id: str, bounds: [west, south, east, north],
          species: {stableId: {yearRound: 0..1, months: {"1": 0..1}}}}]}.
Values are curated support weights, NOT calibrated occurrence probabilities.
Missing species/cells/dates are unknown, never absences. No network requests.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


def valid_location(location):
    if not isinstance(location, dict) or location.get("enabled") is not True:
        return None
    if location.get("crs") not in ("WGS84", "EPSG:4326"):
        return None
    if not isinstance(location.get("source"), str) or not location["source"].strip():
        return None
    values = []
    for key, limit in (("latitude", 90), ("longitude", 180)):
        value = location.get(key)
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or not -limit <= value <= limit:
            return None
        values.append(float(value))
    return tuple(values)


def observation_month(observed_at):
    """Use the event's ISO date/offset, never the processing date or file mtime."""
    if not isinstance(observed_at, str) or not observed_at.strip():
        return None
    try:
        text = observed_at.strip()
        if len(text) == 10:
            return date.fromisoformat(text).month
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        # A naive timestamp is ambiguous; a date-only value is intentionally local.
        return stamp.month if stamp.utcoffset() is not None else None
    except ValueError:
        return None


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Support weight must be a finite number in [0, 1].")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Support weight must be a finite number in [0, 1].")
    return float(value)


def validate_asset(asset, taxonomy_version):
    if asset.get("schemaVersion") != 1 or asset.get("crs") != "EPSG:4326":
        raise ValueError("Unsupported geographic asset schema or CRS.")
    if not taxonomy_version or asset.get("taxonomyVersion") != taxonomy_version:
        raise ValueError("Geographic asset taxonomy mismatch.")
    source = asset.get("source") or {}
    if not all(isinstance(source.get(k), str) and source[k].strip() for k in ("name", "license")):
        raise ValueError("Geographic asset requires source name and license.")
    if source.get("commercialUseApproved") is not True:
        raise ValueError("Geographic asset has not been approved for company use.")
    cells = asset.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Geographic asset requires a cells list.")
    ids = set()
    for cell in cells:
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id.strip() or cell_id in ids:
            raise ValueError("Cell ids must be nonempty and unique.")
        ids.add(cell_id)
        bounds = cell.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError("Cell bounds must be [west, south, east, north].")
        for value in bounds:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Cell bounds must be finite numbers.")
        west, south, east, north = bounds
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("Invalid cell bounds; split antimeridian cells explicitly.")
        species = cell.get("species")
        if not isinstance(species, dict):
            raise ValueError("Cell species must be a stableId-keyed object.")
        for stable_id, record in species.items():
            if not isinstance(stable_id, str) or not stable_id.strip() or not isinstance(record, dict):
                raise ValueError("Invalid species support record.")
            if "yearRound" in record:
                _weight(record["yearRound"])
            months = record.get("months", {})
            if not isinstance(months, dict) or any(str(m) not in {str(n) for n in range(1, 13)} for m in months):
                raise ValueError("Month keys must be 1..12.")
            for value in months.values():
                _weight(value)
    return asset


@lru_cache(maxsize=4)
def _load_asset_cached(path_str, modified_ns, size, expected_hash, taxonomy_version):
    """Cache signed immutable assets by (path, mtime, size); sha still verified."""
    content = Path(path_str).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_hash:
        raise ValueError("Geographic asset SHA-256 mismatch.")
    asset = validate_asset(json.loads(content.decode("utf-8")), taxonomy_version)
    return content, digest, asset


def geographic_support(settings, observation, *, root, taxonomy_version):
    """Return finite bounded bonuses and an audit status; failures are neutral.

    Used ONLY by the review shortlist. Pure visual rankings, margins, consensus
    and local automatic-acceptance gates remain unchanged. The magnitude is an
    experimental cosine-score offset, not a posterior probability.
    """
    status = {"enabled": bool(settings.get("enabled", False)), "applied": False,
              "reason": "disabled", "mode": "review_retrieval_soft_support",
              "calibrationStatus": "not_calibrated", "missingRecordPolicy": "neutral",
              "hardFiltering": False, "raritySupported": False}
    if not status["enabled"]:
        return {}, status
    point = valid_location(observation)
    if point is None:
        status["reason"] = "missing_or_invalid_wgs84_location"
        return {}, status
    try:
        max_bonus = settings.get("maxBonus", 0.02)
        if isinstance(max_bonus, bool) or not isinstance(max_bonus, (int, float)) or not math.isfinite(max_bonus) or not 0 <= max_bonus <= 0.05:
            raise ValueError("maxBonus must be finite and in [0, 0.05].")
        value = settings.get("dataFile")
        expected_hash = settings.get("expectedSha256")
        if not value or not expected_hash:
            status["reason"] = "distribution_asset_not_configured"
            return {}, status
        path = Path(value)
        if not path.is_absolute():
            path = Path(root) / path
        content, digest, asset = _load_asset_cached(
            str(path), path.stat().st_mtime_ns, path.stat().st_size,
            expected_hash, taxonomy_version,
        )
        latitude, longitude = point
        cells = [c for c in asset["cells"]
                 if c["bounds"][0] <= longitude <= c["bounds"][2]
                 and c["bounds"][1] <= latitude <= c["bounds"][3]]
        status.update({"assetSha256": digest, "source": asset["source"]["name"],
                       "license": asset["source"]["license"], "maxBonus": max_bonus})
        if len(cells) != 1:
            status["reason"] = "outside_coverage" if not cells else "ambiguous_cell_coverage"
            return {}, status
        cell = cells[0]
        month = observation_month(observation.get("observedAt"))
        bonuses = {}
        seasonal_used = False
        for stable_id, record in cell["species"].items():
            support = record.get("months", {}).get(str(month)) if month is not None else None
            if support is not None:
                seasonal_used = True
            if support is None:
                support = record.get("yearRound")
            if support is not None and support > 0 and max_bonus > 0:
                bonuses[stable_id] = float(max_bonus) * float(support)
        status.update({"applied": bool(bonuses), "reason": "support_available" if bonuses else "no_applicable_support",
                       "cellId": cell["id"], "observationMonth": month,
                       "supportedSpeciesCount": len(bonuses), "seasonalApplied": seasonal_used})
        return bonuses, status
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        status.update({"reason": "invalid_or_unavailable_asset", "error": f"{type(error).__name__}: {error}"})
        return {}, status


def allowed_species(settings, observation, *, root, taxonomy_version):
    """Species ids supported at the observation location (hard filter).

    Used by the local retrieval stage: when ``filterMode`` is ``hard`` and a
    valid device location falls inside the distribution asset, the caller
    suppresses BioCLIP similarity for every species without support in that
    raster cell, so Top-5 candidates can only come from the local avifauna.

    Missing/invalid location, coverage gaps, or a pool smaller than
    ``minPoolSpecies`` return ``(None, status)`` so the caller keeps the full
    taxonomy (neutral fallback).
    """
    status = {
        "enabled": bool(settings.get("enabled", False)),
        "filterMode": str(settings.get("filterMode", "off")),
        "applied": False,
        "reason": "disabled",
        "poolSpecies": None,
        "minPoolSpecies": int(settings.get("minPoolSpecies", 20)),
        "cellId": None,
        "hardFiltering": True,
    }
    if not status["enabled"] or status["filterMode"] != "hard":
        return None, status
    point = valid_location(observation)
    if point is None:
        status["reason"] = "missing_or_invalid_wgs84_location"
        return None, status
    try:
        value = settings.get("dataFile")
        expected_hash = settings.get("expectedSha256")
        if not value or not expected_hash:
            status["reason"] = "distribution_asset_not_configured"
            return None, status
        path = Path(value)
        if not path.is_absolute():
            path = Path(root) / path
        content, digest, asset = _load_asset_cached(
            str(path), path.stat().st_mtime_ns, path.stat().st_size,
            expected_hash, taxonomy_version,
        )
        latitude, longitude = point
        cells = [c for c in asset["cells"]
                 if c["bounds"][0] <= longitude <= c["bounds"][2]
                 and c["bounds"][1] <= latitude <= c["bounds"][3]]
        if len(cells) != 1:
            status["reason"] = "outside_coverage" if not cells else "ambiguous_cell_coverage"
            return None, status
        cell = cells[0]
        month = observation_month(observation.get("observedAt"))
        ids = set()
        for stable_id, record in cell["species"].items():
            support = record.get("months", {}).get(str(month)) if month is not None else None
            if support is None:
                support = record.get("yearRound")
            if support is not None and support > 0:
                ids.add(str(stable_id))
        status.update({"cellId": cell["id"], "poolSpecies": len(ids),
                       "assetSha256": digest, "source": asset["source"]["name"]})
        if len(ids) < int(status["minPoolSpecies"]):
            status["reason"] = "pool_too_small"
            return None, status
        status.update({"applied": True, "reason": "hard_filter"})
        return ids, status
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        status["reason"] = "invalid_or_unavailable_asset"
        status["error"] = f"{type(error).__name__}: {error}"
        return None, status
