"""Production runtime loader for the versioned BioCLIP 2.5 bird bundle."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

from image_prototype_retrieval import load_prototype_gallery
from pipeline_config import load_pipeline_config
from scene_prompts import ROUTER_PROMPTS
from taxonomy_registry import (
    load_region_priors,
    load_taxonomy_registry,
    model_scientific_name,
    validate_taxonomy_alignment,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline_config_bioclip25.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, config_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_root / path


def verify_manifest_file(entry: dict, manifest_root: Path) -> Path:
    path = resolve_path(entry["path"], manifest_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["sizeBytes"]):
        raise ValueError(f"Asset size mismatch: {path}")
    actual = sha256_path(path)
    if actual != entry["sha256"]:
        raise ValueError(f"Asset SHA-256 mismatch: {path}: {actual}")
    return path


def load_text_overrides(manifest_path, *, asset_manifest, embedding_dimension, taxonomy_by_index):
    """Verify a text overlay manifest and return the rows it replaces.

    Overlays keep ``primary_embeddings.npy`` and its SHA-256 published, and pin
    the base embeddings, weights, taxonomy version and vector SHA they were
    built against so a stale overlay cannot be applied silently.
    """
    manifest_path = Path(manifest_path)
    override = json.loads(manifest_path.read_text(encoding="utf-8"))
    if override.get("bundle") != asset_manifest.get("bundle"):
        raise ValueError(f"Text override bundle mismatch: {manifest_path}")
    if override.get("taxonomyVersion") != asset_manifest["catalog"]["taxonomyVersion"]:
        raise ValueError(f"Text override taxonomy version mismatch: {manifest_path}")
    if int(override["embeddingDimension"]) != embedding_dimension:
        raise ValueError(f"Text override embedding dimension mismatch: {manifest_path}")
    if (
        override["basePrimaryEmbeddingsSha256"]
        != asset_manifest["files"]["primaryEmbeddings"]["sha256"]
    ):
        raise ValueError(
            f"Text override was built for different primary embeddings: {manifest_path}"
        )
    if override["modelWeightsSha256"] != asset_manifest["model"]["weightsSha256"]:
        raise ValueError(f"Text override was built for different weights: {manifest_path}")

    entry = override["vectors"]
    vector_path = verify_manifest_file(entry, manifest_path.parent)
    blob = np.load(vector_path, allow_pickle=False)
    vectors = blob[entry["key"]].astype(np.float32)
    stable_ids = [str(value) for value in blob["stableIds"]]
    blob.close()

    rows = override["rows"]
    if vectors.shape != (len(rows), embedding_dimension):
        raise ValueError(f"Text override vector shape mismatch: {vectors.shape}")
    if stable_ids != [row["stableId"] for row in rows]:
        raise ValueError(f"Text override stableId order mismatch: {manifest_path}")

    index_by_stable_id = {
        str(item.get("stableId")): int(index) for index, item in taxonomy_by_index.items()
    }
    indices = []
    for row in rows:
        index = index_by_stable_id.get(row["stableId"])
        if index is None:
            raise ValueError(
                f"Text override stableId is not in the taxonomy registry: {row['stableId']}"
            )
        expected = " ".join(
            str(taxonomy_by_index[index].get("modelScientificName") or "").split()
        ).casefold()
        actual = " ".join(str(row["scientificName"]).split()).casefold()
        if expected != actual:
            raise ValueError(f"Text override row targets another species: {row['stableId']}")
        indices.append(index)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or float(norms.min()) < 1e-6:
        raise ValueError(f"Text override contains unusable vectors: {manifest_path}")
    return {
        "name": override.get("name", manifest_path.stem),
        "sha256": entry["sha256"],
        "stableIds": stable_ids,
        "indices": indices,
        "vectors": (vectors / norms).astype(np.float32),
        "traitsByStableId": {str(row["stableId"]): str(row.get("traitText") or "").strip()
                             for row in rows if str(row.get("traitText") or "").strip()},
    }


def apply_text_geometry(matrix, remove_components):
    """Drop the dominant directions of a text matrix, in place.

    The leading components of the taxonomy-wide text matrix describe "a bird"
    rather than any species, so they consume similarity budget without adding
    discrimination. Removing them keeps every row on the unit sphere, which
    means open-set thresholds stay on the shipped score scale.
    """

    count = int(remove_components or 0)
    if count <= 0:
        return
    if count >= min(matrix.shape):
        raise ValueError(
            f"textGeometry.removePrincipalComponents={count} is too large for {matrix.shape}"
        )
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    # eigendecomposition of the 1024x1024 covariance gives the same leading
    # directions as a full SVD but in about a second instead of a minute
    covariance = centered.T @ centered
    _, vectors = np.linalg.eigh(covariance.astype(np.float64))
    basis = vectors[:, ::-1][:, :count].T.astype(np.float32)
    centered -= centered @ basis.T @ basis
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    matrix[:] = centered.astype(np.float32)



_BUNDLE_CACHE: dict = {}
_BUNDLE_LOCK = threading.Lock()


def cached_bundle(key, factory):
    """Build a shared read-only bundle at most once per key, process-wide."""
    with _BUNDLE_LOCK:
        if key not in _BUNDLE_CACHE:
            _BUNDLE_CACHE[key] = factory()
        return _BUNDLE_CACHE[key]


def bundle_stats():
    """Observability helper: how many bundles are cached in this process."""
    return {"entries": len(_BUNDLE_CACHE),
            "keys": [str(k) for k in _BUNDLE_CACHE]}

def validate_config_dependencies(config):
    """启动期配置预检：跨模块开关依赖必须显式满足，失败即拒绝启动。"""
    guard = config.get("visualConsensusGuard") or {}
    if guard.get("enabled") and guard.get("requirePrototypeAgreement", True):
        if not (config.get("imagePrototypeGallery") or {}).get("enabled"):
            raise ValueError(
                "visualConsensusGuard.requirePrototypeAgreement requires "
                "imagePrototypeGallery.enabled; otherwise every review "
                "silently degrades to visual_branch_unavailable."
            )


def _build_shared_bundle(config_path, verify_model_hash):
    """Load and validate everything identical for every runtime instance.

    Runs once per process. All returned objects are read-only afterwards;
    the two large text matrices are marked non-writable so a downstream
    in-place edit fails loudly instead of corrupting every instance.
    """
    config = load_pipeline_config(config_path)
    validate_config_dependencies(config)
    config_root = Path(config["_path"]).parent
    model_config = config["model"]
    manifest_path = resolve_path(model_config["assetManifestFile"], config_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle") != "bioclip25-avilist-v2025b":
        raise ValueError(f"Unexpected BioCLIP asset bundle: {manifest.get('bundle')}")

    model_identity = manifest["model"]
    embedding_dimension = int(model_config["embeddingDimension"])
    if model_identity["architecture"] != model_config["architecture"]:
        raise ValueError("Configured/model-manifest architectures differ")
    if int(model_identity["embeddingDimension"]) != embedding_dimension:
        raise ValueError("Configured/model-manifest embedding dimensions differ")

    manifest_root = manifest_path.parent
    asset_paths = {
        name: verify_manifest_file(entry, manifest_root)
        for name, entry in manifest["files"].items()
    }
    model_path = resolve_path(model_identity["weightsPath"], manifest_root)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if model_path.stat().st_size != int(model_identity["weightsSizeBytes"]):
        raise ValueError("BioCLIP model size does not match the asset manifest")
    if verify_model_hash:
        actual_model_hash = sha256_path(model_path)
        if actual_model_hash != model_identity["weightsSha256"]:
            raise ValueError(
                f"BioCLIP model SHA-256 mismatch: {actual_model_hash}"
            )

    open_clip_value = model_config.get("openClipPythonPath")
    if open_clip_value:
        open_clip_path = resolve_path(open_clip_value, config_root)
        sys.path.insert(0, str(open_clip_path))
    import cv2
    import open_clip
    import torch

    requested_device = str(model_config.get("device", "auto")).lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        device = requested_device
    else:
        raise ValueError("model.device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    # 8 vCPU host: 4 threads measured fastest for single-view encoding
    # (659 ms vs 953 ms at the old default of 12). Override via env.
    cpu_threads = int(os.environ.get("BIRD_CPU_THREADS", "4") or 4)
    if device == "cpu" and cpu_threads > 0:
        torch.set_num_threads(cpu_threads)

    taxa = json.loads(asset_paths["birdTaxa"].read_text(encoding="utf-8"))
    global_labels = [model_scientific_name(entry[0]) for entry in taxa]
    common_names = {
        scientific_name: entry[1] or scientific_name
        for scientific_name, entry in zip(global_labels, taxa)
    }
    global_text = np.load(asset_paths["primaryEmbeddings"], allow_pickle=False).astype(
        np.float32
    )
    if global_text.shape != (len(global_labels), embedding_dimension):
        raise ValueError(
            "Primary taxonomy/embedding shape mismatch: "
            f"labels={len(global_labels)}, embeddings={global_text.shape}"
        )
    global_text /= np.maximum(
        np.linalg.norm(global_text, axis=1, keepdims=True), 1e-12
    )

    registry, taxonomy_by_index, valid_taxonomy_indices = (
        load_taxonomy_registry(asset_paths["primaryRegistry"])
    )
    validate_taxonomy_alignment(
        global_labels, global_text, taxonomy_by_index
    )

    text_overrides = None
    override_config = config.get("textOverrides") or {}
    if override_config.get("enabled"):
        text_overrides = load_text_overrides(
            resolve_path(override_config["manifestFile"], config_root),
            asset_manifest=manifest,
            embedding_dimension=embedding_dimension,
            taxonomy_by_index=taxonomy_by_index,
        )
        global_text[text_overrides["indices"]] = text_overrides["vectors"]

    supplemental_registry = json.loads(
        asset_paths["supplementalRegistry"].read_text(encoding="utf-8")
    )
    supplemental_text = np.load(
        asset_paths["supplementalEmbeddings"], allow_pickle=False
    ).astype(np.float32)
    supplemental_species = supplemental_registry.get("species") or []
    if supplemental_text.shape != (len(supplemental_species), embedding_dimension):
        raise ValueError(
            "Supplemental registry/embedding shape mismatch: "
            f"species={len(supplemental_species)}, embeddings={supplemental_text.shape}"
        )
    if int(supplemental_registry.get("embeddingDimension", -1)) != embedding_dimension:
        raise ValueError("Supplemental registry has the wrong embedding dimension")
    supplemental_text /= np.maximum(
        np.linalg.norm(supplemental_text, axis=1, keepdims=True), 1e-12
    )

    geometry_config = config.get("textGeometry") or {}
    removed_components = int(geometry_config.get("removePrincipalComponents", 0) or 0)
    apply_text_geometry(global_text, removed_components)
    apply_text_geometry(supplemental_text, removed_components)

    if registry["version"] != config["taxonomy"]["version"]:
        raise ValueError("Primary registry taxonomy version differs from config")
    if supplemental_registry["version"] != config["taxonomy"]["version"]:
        raise ValueError("Supplemental registry taxonomy version differs from config")
    expected_method = config["supplementalCandidates"]["embeddingMethod"]
    if supplemental_registry.get("embeddingMethod") != expected_method:
        raise ValueError("Supplemental registry embedding method differs from config")

    primary_ids = {item["stableId"] for item in registry["species"]}
    supplemental_ids = [item.get("stableId") for item in supplemental_species]
    if any(not stable_id for stable_id in supplemental_ids):
        raise ValueError("Supplemental registry contains an empty stableId")
    if len(set(supplemental_ids)) != len(supplemental_ids):
        raise ValueError("Supplemental registry contains duplicate stableIds")
    overlap = primary_ids.intersection(supplemental_ids)
    if overlap:
        raise ValueError(f"Primary/supplemental stableIds overlap: {len(overlap)}")
    catalog = manifest["catalog"]
    if len(primary_ids) != int(catalog["primaryUniqueStableIds"]):
        raise ValueError("Primary unique stableId count differs from manifest")
    if len(supplemental_ids) != int(catalog["supplementalStableIds"]):
        raise ValueError("Supplemental stableId count differs from manifest")
    if len(primary_ids) + len(supplemental_ids) != int(
        catalog["totalUniqueStableIds"]
    ):
        raise ValueError("Unified stableId count differs from manifest")

    metadata = json.loads(asset_paths["embeddingMetadata"].read_text(encoding="utf-8"))
    if metadata["model"]["weightsSha256"] != model_identity["weightsSha256"]:
        raise ValueError("Embedding/model identities differ")
    if metadata["model"]["embeddingDimension"] != embedding_dimension:
        raise ValueError("Embedding metadata has the wrong dimension")
    if metadata.get("primaryRows") != len(global_labels):
        raise ValueError("Embedding metadata has the wrong primary row count")
    if metadata.get("supplementalRows") != len(supplemental_species):
        raise ValueError("Embedding metadata has the wrong supplemental row count")

    unified_species_by_id = {
        item["stableId"]: item for item in registry["species"]
    }
    unified_species_by_id.update(
        {item["stableId"]: item for item in supplemental_species}
    )
    prototype_config = config.get("imagePrototypeGallery") or {}
    prototype_gallery = None
    prototype_status = {"available": False, "reason": "disabled"}
    if prototype_config.get("enabled"):
        prototype_manifest = resolve_path(
            prototype_config["manifestFile"], config_root
        )
        prototype_gallery, prototype_status = load_prototype_gallery(
            prototype_manifest,
            unified_species_by_id,
            expected_architecture=model_config["architecture"],
            expected_dimension=embedding_dimension,
            expected_model_sha256=model_identity["weightsSha256"],
            expected_taxonomy_version=config["taxonomy"]["version"],
            optional=bool(prototype_config.get("optional", False)),
        )

    bundle = {
        "torch": torch, "cv2": cv2, "open_clip": open_clip,
        "config": config, "config_root": config_root,
        "model_config": model_config, "manifest": manifest,
        "manifest_root": manifest_root, "asset_paths": asset_paths,
        "model_identity": model_identity, "model_path": model_path,
        "embedding_dimension": embedding_dimension, "device": device,
        "global_labels": global_labels, "common_names": common_names,
        "global_text": global_text, "registry": registry,
        "taxonomy_by_index": taxonomy_by_index,
        "valid_taxonomy_indices": valid_taxonomy_indices,
        "supplemental_registry": supplemental_registry,
        "supplemental_text": supplemental_text,
        "supplemental_species": supplemental_species,
        "unified_species_by_id": unified_species_by_id,
        "text_overrides": text_overrides,
        "removed_components": removed_components,
        "prototype_gallery": prototype_gallery,
        "prototype_status": prototype_status,
        "prototype_config": prototype_config,
        "metadata": metadata,
    }
    for name in ("global_text", "supplemental_text"):
        bundle[name].setflags(write=False)
    return bundle


def load_shared_bundle(config_path=DEFAULT_CONFIG, verify_model_hash=False):
    """Return the process-wide read-only bundle (loaded exactly once per key)."""
    key = (str(config_path), bool(verify_model_hash))
    return cached_bundle(
        key, lambda: _build_shared_bundle(config_path, verify_model_hash)
    )

def load_runtime(config_path=DEFAULT_CONFIG, verify_model_hash=False):
    """Build one independent runtime that shares the read-only bundle.

    Only the model weights, ONNX session, scene-router text and the
    OpenCV focus net are rebuilt per instance. Taxonomy, text embeddings,
    prototype gallery and all SHA-256 validation happen once.
    """
    bundle = load_shared_bundle(config_path, verify_model_hash)
    torch = bundle["torch"]
    cv2 = bundle["cv2"]
    open_clip = bundle["open_clip"]
    config = bundle["config"]
    config_root = bundle["config_root"]
    model_config = bundle["model_config"]
    model_path = bundle["model_path"]
    device = bundle["device"]
    global_labels = bundle["global_labels"]
    common_names = bundle["common_names"]
    global_text = bundle["global_text"]
    registry = bundle["registry"]
    taxonomy_by_index = bundle["taxonomy_by_index"]
    valid_taxonomy_indices = bundle["valid_taxonomy_indices"]
    supplemental_registry = bundle["supplemental_registry"]
    supplemental_text = bundle["supplemental_text"]
    unified_species_by_id = bundle["unified_species_by_id"]
    prototype_gallery = bundle["prototype_gallery"]
    prototype_status = bundle["prototype_status"]
    prototype_config = bundle["prototype_config"]
    manifest = bundle["manifest"]
    text_overrides = bundle["text_overrides"]
    removed_components = bundle["removed_components"]

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config["architecture"], pretrained=str(model_path), device=device
    )
    model.eval()
    encoder_backend = os.environ.get("BIRD_ENCODER", "torch").strip().lower()
    if encoder_backend in {"onnx_fp32", "onnx_int8"}:
        from onnx_encoder import install_onnx_encoder
        onnx_dir = Path(os.environ.get(
            "BIRD_ONNX_DIR",
            str(Path(__file__).resolve().parents[1] / "models" / "onnx" / "e19"),
        ))
        install_onnx_encoder(
            model,
            encoder_backend,
            onnx_dir,
            threads=int(os.environ.get("BIRD_CPU_THREADS", "4")),
        )
    tokenizer = open_clip.get_tokenizer(model_config["architecture"])
    prompt_rows = [
        (category, prompt)
        for category, prompts in ROUTER_PROMPTS.items()
        for prompt in prompts
    ]
    with torch.inference_mode():
        tokens = tokenizer([prompt for _, prompt in prompt_rows]).to(device)
        model._scene_router_text = torch.nn.functional.normalize(
            model.encode_text(tokens), dim=-1
        )
    model._scene_router_groups = {
        category: [index for index, (name, _) in enumerate(prompt_rows) if name == category]
        for category in ROUTER_PROMPTS
    }

    priors_path = resolve_path(config["region"]["priorsFile"], config_root)
    model._pipeline_config = config
    model._taxonomy_registry = registry
    model._taxonomy_by_index = taxonomy_by_index
    model._valid_taxonomy_indices = valid_taxonomy_indices
    model._region_priors = load_region_priors(priors_path)
    model._supplemental_registry = supplemental_registry
    model._supplemental_text = supplemental_text
    model._unified_species_by_id = unified_species_by_id
    model._image_prototype_gallery = prototype_gallery
    model._image_prototype_status = prototype_status
    model._image_prototype_config = prototype_config
    model._inference_device = device
    model._asset_identity = manifest
    model._text_overrides = text_overrides
    model._text_override_traits = dict((text_overrides or {}).get("traitsByStableId", {}))
    model._text_geometry = {"removePrincipalComponents": removed_components}

    crop_model_path = resolve_path(model_config["cropModelFile"], config_root)
    focus_net = (
        cv2.dnn.readNetFromONNX(np.fromfile(crop_model_path, dtype=np.uint8))
        if crop_model_path.is_file()
        else None
    )
    return (
        torch,
        model,
        preprocess,
        None,
        1.0,
        [],
        global_labels,
        common_names,
        global_text,
        focus_net,
    )
