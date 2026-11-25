"""Optional BioCLIP image-prototype retrieval for the bird CLI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class PrototypeGallery:
    embeddings: np.ndarray
    species: tuple[dict, ...]
    counts: np.ndarray
    metadata: dict


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(root: Path, entry: dict) -> Path:
    path = root / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["sizeBytes"]):
        raise ValueError(f"Prototype asset size mismatch: {path}")
    actual = _sha256_path(path)
    if actual != entry["sha256"]:
        raise ValueError(f"Prototype asset SHA-256 mismatch: {path}: {actual}")
    return path


def load_prototype_gallery(
    manifest_path,
    registry_by_id,
    *,
    expected_architecture,
    expected_dimension,
    expected_model_sha256,
    expected_taxonomy_version,
    optional=False,
):
    """Load and validate a species-centroid gallery.

    A missing optional gallery disables only this enhancement. Once a manifest
    exists, every identity and integrity check is strict so stale vectors never
    enter the production candidate stream.
    """

    started = time.perf_counter()
    path = Path(manifest_path)
    if not path.is_file():
        if optional:
            return None, {
                "available": False,
                "reason": "gallery_missing",
                "manifest": str(path),
                "loadLatencyMs": round((time.perf_counter() - started) * 1000),
            }
        raise FileNotFoundError(path)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schemaVersion", -1)) != 1:
        raise ValueError("Unsupported image prototype manifest schema")
    model = manifest.get("model") or {}
    if model.get("architecture") != expected_architecture:
        raise ValueError("Prototype/model architectures differ")
    if int(model.get("embeddingDimension", -1)) != int(expected_dimension):
        raise ValueError("Prototype/model embedding dimensions differ")
    if model.get("weightsSha256") != expected_model_sha256:
        raise ValueError("Prototype/model weight identities differ")
    taxonomy = manifest.get("taxonomy") or {}
    if taxonomy.get("version") != expected_taxonomy_version:
        raise ValueError("Prototype/runtime taxonomy versions differ")
    if manifest.get("aggregation") != "mean_then_l2_normalize":
        raise ValueError("Unsupported image prototype aggregation")

    files = manifest.get("files") or {}
    required = {"prototypes", "counts", "species"}
    if not required.issubset(files):
        raise ValueError("Prototype manifest is missing required files")
    root = path.parent
    prototypes_path = _verified_file(root, files["prototypes"])
    counts_path = _verified_file(root, files["counts"])
    species_path = _verified_file(root, files["species"])

    embeddings = np.load(prototypes_path, allow_pickle=False).astype(np.float32)
    counts = np.load(counts_path, allow_pickle=False)
    rows = [
        json.loads(line)
        for line in species_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    species_count = int(manifest.get("speciesCount", -1))
    expected_shape = (species_count, int(expected_dimension))
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Prototype vector shape mismatch: expected={expected_shape}, actual={embeddings.shape}"
        )
    if counts.shape != (species_count,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Prototype support counts have the wrong shape or dtype")
    if len(rows) != species_count:
        raise ValueError("Prototype species rows do not match the vector count")
    if not np.isfinite(embeddings).all():
        raise ValueError("Prototype vectors contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("Prototype vectors are not L2-normalized")
    if np.any(counts <= 0) or int(counts.sum()) != int(manifest.get("trainSampleCount", -1)):
        raise ValueError("Prototype support counts are invalid")

    canonical_rows = []
    stable_ids = []
    for index, row in enumerate(rows):
        if int(row.get("prototypeIndex", -1)) != index:
            raise ValueError("Prototype indices are not contiguous")
        stable_id = row.get("stableId")
        if stable_id not in registry_by_id:
            raise ValueError(f"Prototype gallery contains unknown stableId: {stable_id}")
        stable_ids.append(stable_id)
        canonical_rows.append(
            {
                **registry_by_id[stable_id],
                "prototypeIndex": index,
                "trainCount": int(counts[index]),
            }
        )
    if stable_ids != sorted(stable_ids) or len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Prototype stableIds must be unique and lexically sorted")

    gallery = PrototypeGallery(
        embeddings=np.ascontiguousarray(embeddings),
        species=tuple(canonical_rows),
        counts=counts.astype(np.int32, copy=False),
        metadata=manifest,
    )
    return gallery, {
        "available": True,
        "reason": "loaded",
        "manifest": str(path.resolve()),
        "speciesCount": species_count,
        "trainSampleCount": int(counts.sum()),
        "loadLatencyMs": round((time.perf_counter() - started) * 1000),
    }


def score_prototype_candidates(query, gallery, top_k=5):
    if gallery is None:
        return []
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if vector.shape != (gallery.embeddings.shape[1],) or not np.isfinite(vector).all():
        raise ValueError("Query embedding is invalid for the prototype gallery")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Query embedding has zero norm")
    scores = gallery.embeddings @ (vector / norm)
    order = np.argsort(-scores, kind="stable")[: max(0, int(top_k))]
    rows = []
    for index in order:
        item = gallery.species[int(index)]
        rows.append(
            {
                "rank": len(rows) + 1,
                "stableId": item["stableId"],
                "scientificName": item["scientificName"],
                "commonName": item.get("commonName") or item["scientificName"],
                "order": item.get("order", ""),
                "family": item.get("family", ""),
                "familyEnglishName": item.get("familyEnglishName", ""),
                "genus": item.get("genus", ""),
                "modelScientificName": item.get("modelScientificName", item["scientificName"]),
                "taxonomyMatchMethod": "image_prototype_stable_id",
                "candidateSource": "image_prototype_gallery",
                "scoreSpace": "bioclip25_image_image_cosine",
                "prototypeRank": len(rows) + 1,
                "prototypeSimilarity": round(float(scores[int(index)]), 6),
                "prototypeSupportCount": int(gallery.counts[int(index)]),
            }
        )
    return rows


def fuse_review_candidates(
    text_rows,
    prototype_rows,
    *,
    max_candidates=5,
    mode="diagnostic",
    maximum_injected=1,
    calibration_status="not_calibrated",
    allowed_stable_ids=None,
    minimum_similarity=None,
    minimum_margin=None,
    requires_bird_presence=False,
    bird_present=None,
):
    """Conservatively add at most one calibrated gallery candidate.

    Text-image and image-image cosine values are deliberately never mixed.
    Production candidate injection requires explicit calibration approval, a
    species allowlist, and calibrated thresholds. The separate experimental
    mode is visibly marked and can only add a review candidate; it never
    approves a final species answer. Diagnostic mode preserves the existing
    BioCLIP shortlist byte-for-byte.
    """

    text_rows = list(text_rows or [])[:max_candidates]
    prototype_rows = list(prototype_rows or [])
    text_ids = [row["stableId"] for row in text_rows]
    prototype_top = prototype_rows[0] if prototype_rows else None
    prototype_id = prototype_top.get("stableId") if prototype_top else None
    allowed_ids = {
        str(stable_id) for stable_id in (allowed_stable_ids or []) if stable_id
    }
    margin = None
    if len(prototype_rows) >= 2:
        margin = round(float(
            prototype_rows[0]["prototypeSimilarity"]
            - prototype_rows[1]["prototypeSimilarity"]
        ), 6)
    meta = {
        "mode": mode,
        "calibrationStatus": calibration_status,
        "allowedStableIds": sorted(allowed_ids),
        "applied": False,
        "reason": "prototype_unavailable" if not prototype_rows else "diagnostic_only",
        "textTop1StableId": text_ids[0] if text_ids else None,
        "prototypeTop1StableId": prototype_id,
        "top1Agreement": bool(text_ids and prototype_id == text_ids[0]),
        "prototypeTop1InTextCandidates": prototype_id in text_ids if prototype_id else False,
        "prototypeMargin": round(margin, 6) if margin is not None else None,
        "requiresBirdPresence": bool(requires_bird_presence),
        "birdPresent": bird_present,
        "deploymentStatus": (
            "experimental_not_runtime_approved"
            if mode == "experimental_candidate_injection"
            else "production_approved"
            if mode == "candidate_injection" and calibration_status == "approved"
            else "inactive"
        ),
    }
    if not prototype_top or mode == "diagnostic":
        return text_rows, meta
    if mode not in {"candidate_injection", "experimental_candidate_injection"}:
        raise ValueError(f"Unsupported image prototype fusion mode: {mode}")
    required_status = (
        "experimental_positive_only"
        if mode == "experimental_candidate_injection"
        else "approved"
    )
    if calibration_status != required_status:
        meta["reason"] = "calibration_not_approved"
        return text_rows, meta
    if requires_bird_presence and bird_present is not True:
        meta["reason"] = "bird_presence_not_confirmed"
        return text_rows, meta
    if not allowed_ids:
        meta["reason"] = "allowlist_empty"
        return text_rows, meta
    if prototype_id not in allowed_ids:
        meta["reason"] = "prototype_not_allowlisted"
        return text_rows, meta
    if maximum_injected < 1:
        meta["reason"] = "injection_disabled"
        return text_rows, meta
    if minimum_similarity is None or minimum_margin is None:
        meta["reason"] = "thresholds_not_calibrated"
        return text_rows, meta
    if float(prototype_top["prototypeSimilarity"]) < float(minimum_similarity):
        meta["reason"] = "prototype_similarity_below_threshold"
        return text_rows, meta
    if margin is None or margin < float(minimum_margin):
        meta["reason"] = "prototype_margin_below_threshold"
        return text_rows, meta
    if prototype_id in text_ids:
        meta["reason"] = "prototype_already_in_text_candidates"
        return text_rows, meta

    kept = [dict(row) for row in text_rows[: max(0, max_candidates - 1)]]
    injected = dict(prototype_top)
    if mode == "experimental_candidate_injection":
        injected.update(
            {
                "reviewCandidateSource": "experimental_camera_prototype",
                "candidateEvidenceRole": "candidate_recall_only",
                "notFinalSpeciesEvidence": True,
            }
        )
    kept.append(injected)
    for rank, row in enumerate(kept, 1):
        row["rank"] = rank
    meta.update(
        {
            "applied": True,
            "reason": (
                "experimental_prototype_candidate_injected"
                if mode == "experimental_candidate_injection"
                else "calibrated_prototype_candidate_injected"
            ),
            "injectedStableIds": [prototype_id],
        }
    )
    return kept, meta
