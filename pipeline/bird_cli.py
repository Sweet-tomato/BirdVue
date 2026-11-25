import argparse
import json
import os
import sys
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np

from image_prototype_retrieval import (
    fuse_review_candidates,
    score_prototype_candidates,
)
from single_image_fine_grained import (
    aggregate_scores,
    apply_visual_consensus_guard,
    build_detail_views,
    parse_settings as parse_fine_grained_settings,
)
from species_identification_cards import resolve_card_settings
from knowledge_graph_runtime import resolve_knowledge_graph_settings

from doubao_report import create_boss_report
from pipeline_config import load_pipeline_config
from scene_prompts import ROUTER_PROMPTS
from taxonomy_registry import (
    candidate_rows,
    load_region_priors,
    load_taxonomy_registry,
    model_scientific_name,
    validate_region,
    validate_taxonomy_alignment,
)


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
HEAD_PATH = ROOT / "models" / "unified_linear_head_406_full.pt"
CONFIG_PATH = ROOT / "pipeline_config.json"
BIOCLIP_MODEL_PATH = ROOT / "models" / "bioclip-2.5-vit-h14" / "open_clip_model.safetensors"
BIRD_DATA_PATH = ROOT / "models" / "bioclip25" / "birds"
BIOCLIP_PYTHON_PATH = str(ROOT)
CROP_MODEL_PATH = ROOT.parent / "models" / "bird-crop-yolox-tiny.onnx"
SUPPLEMENTAL_SPECIES_PATH = ROOT / "avilist_v2025b_supplemental_species.json"
SUPPLEMENTAL_EMBEDDINGS_PATH = ROOT / "avilist_v2025b_supplemental_embeddings.npy"
_MODEL_LOCK = threading.Lock()
TOP_K = 5
HEAD_RERANK_THRESHOLD = 0.5
DETECTION_INPUT_SIZE = 416
BIRD_CLASS_ID = 14
DETECTION_THRESHOLD = 0.05
# ZOOM_LEVELS can collapse to identical tile geometry (e.g. 640x480 ->
# 416x416 tiles in both levels). Reusing those forwards keeps the candidate
# list, `passes`, the break condition and the final best bit-identical.
DETECT_DEDUP = os.environ.get("BIRD_DETECT_DEDUP", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
FULL_FRAME_ACCEPT = 0.35
NMS_THRESHOLD = 0.45
FOCUS_MAX_OCCUPANCY = 0.35
FOCUS_CONTEXT_FACTOR = 2.0
ZOOM_LEVELS = ((0.55, 0.20), (0.34, 0.20))
ROUTER_SQUIRREL_MARGIN = 0.06
ROUTER_SQUIRREL_SCORE = 0.50
ROUTER_WEAK_BIRD_DETECTION = 0.10





def load_supplemental_catalog(species_path=SUPPLEMENTAL_SPECIES_PATH, embeddings_path=SUPPLEMENTAL_EMBEDDINGS_PATH):
    species_path = Path(species_path)
    embeddings_path = Path(embeddings_path)
    if not species_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            "AviList 补充候选资产不完整："
            f"species={species_path.is_file()}, embeddings={embeddings_path.is_file()}"
        )
    payload = json.loads(species_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, allow_pickle=False).astype(np.float32)
    species = payload.get("species") or []
    expected_count = int(payload.get("count", -1))
    expected_dimension = int(payload.get("embeddingDimension", -1))
    if embeddings.ndim != 2 or embeddings.shape != (expected_count, expected_dimension):
        raise ValueError(
            "AviList 补充候选元数据/向量不对齐："
            f"metadata=({expected_count}, {expected_dimension}), embeddings={embeddings.shape}."
        )
    if len(species) != expected_count:
        raise ValueError(
            f"AviList 补充候选条目数不对齐：metadata={expected_count}, species={len(species)}."
        )
    stable_ids = [item.get("stableId") for item in species]
    if any(not stable_id for stable_id in stable_ids) or len(set(stable_ids)) != len(stable_ids):
        raise ValueError("AviList 补充候选包含空或重复 stableId。")
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return payload, embeddings





def clean_path(value):
    cleaned = "".join(character for character in value if unicodedata.category(character) != "Cf")
    return Path(cleaned.strip().strip('"').strip("'"))


def load_asset_manifest(config):
    """Load the asset manifest declared by the config, if any.

    Returns ``(manifest, manifest_path)`` or ``(None, None)`` for legacy
    manifest-less configs.
    """
    manifest_value = (config.get("model") or {}).get("assetManifestFile")
    if not manifest_value:
        return None, None
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = Path(config["_path"]).parent / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"资产清单不存在：{manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def resolve_manifest_file(manifest, manifest_path, key):
    """Resolve one ``files.<key>`` entry of the asset manifest to a path."""
    entry = (manifest.get("files") or {}).get(key)
    if not entry or not entry.get("path"):
        return None
    path = Path(entry["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


_SPECIES_NAME_INDEX = None


def resolve_species_entry(name):
    """Resolve an eBird common name or binomial to the AviList registry row.

    Used to validate species names proposed by the review model when they are
    outside the supplied candidate shortlist. Unknown names return ``None`` so
    the caller can reject them instead of emitting an unverified species.
    """
    global _SPECIES_NAME_INDEX

    key = " ".join(str(name or "").split())
    if not key:
        return None
    if _SPECIES_NAME_INDEX is None:
        candidates = [
            os.environ.get("BIRD_CONFIG"),
            str(ROOT.parent / "config" / "pipeline_config.json"),
            str(CONFIG_PATH),
        ]
        config_path = next((item for item in candidates if item and Path(item).is_file()), None)
        if config_path is None:
            return None
        config = load_pipeline_config(config_path)
        manifest, manifest_path = load_asset_manifest(config)
        species_path = resolve_manifest_file(manifest, manifest_path, "primaryRegistry")
        payload = json.loads(Path(species_path).read_text(encoding="utf-8"))
        index = {}
        for item in payload.get("species") or []:
            for field in ("scientificName", "modelScientificName", "commonName", "englishName"):
                value = item.get(field)
                if value:
                    index.setdefault(" ".join(str(value).split()).casefold(), item)
        _SPECIES_NAME_INDEX = index
    return _SPECIES_NAME_INDEX.get(key.casefold())


def resolve_model_weights(config, manifest=None, manifest_path=None):
    """Resolve the BioCLIP checkpoint the same way the runtime does.

    The asset manifest is the single source of truth for the weight path and
    architecture; the hard-coded path is only a fallback for manifest-less
    configs.
    """
    model_config = config.get("model") or {}
    fallback_architecture = str(model_config.get("architecture") or "ViT-H-14")
    if manifest is None:
        manifest, manifest_path = load_asset_manifest(config)
    if manifest is None:
        return BIOCLIP_MODEL_PATH, fallback_architecture
    model_entry = manifest.get("model") or {}
    weights_value = model_entry.get("weightsPath")
    if not weights_value:
        raise ValueError(f"资产清单缺少 model.weightsPath：{manifest_path}")
    weights_path = Path(weights_value)
    if not weights_path.is_absolute():
        weights_path = manifest_path.parent / weights_path
    architecture = str(model_entry.get("architecture") or fallback_architecture)
    return weights_path, architecture


def load_runtime(config_path=CONFIG_PATH):
    config = load_pipeline_config(config_path)
    from bioclip25_runtime import validate_config_dependencies
    validate_config_dependencies(config)
    open_clip_python = (config.get("model") or {}).get("openClipPythonPath")
    if open_clip_python:
        open_clip_python = Path(open_clip_python)
        if not open_clip_python.is_absolute():
            open_clip_python = Path(config["_path"]).parent / open_clip_python
    sys.path.insert(0, str(open_clip_python or BIOCLIP_PYTHON_PATH))
    import cv2
    import open_clip
    import torch

    asset_manifest, manifest_path = load_asset_manifest(config)
    model_weights_path, model_architecture = resolve_model_weights(
        config, asset_manifest, manifest_path
    )

    if not model_weights_path.exists():
        raise FileNotFoundError(f"BioCLIP模型不存在：{model_weights_path}")

    head_config = config.get("experimentalClassifierHead") or {}
    head_data = None
    head = None
    classes = []
    if head_config.get("enabled"):
        if not HEAD_PATH.exists():
            raise FileNotFoundError(f"分类头不存在：{HEAD_PATH}")
        head_data = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
        classes = [str(name) for name in head_data["classes"]]
        head = torch.nn.Linear(int(head_data["embedding_dimension"]), len(classes))
        head.load_state_dict(head_data["state_dict"])
        head.eval()

    config_root = Path(config["_path"]).parent
    registry_path = resolve_manifest_file(asset_manifest, manifest_path, "primaryRegistry")
    if registry_path is None:
        registry_path = config_root / config["taxonomy"]["registryFile"]
    priors_path = config_root / config["region"]["priorsFile"]
    if not registry_path.is_file():
        raise FileNotFoundError(f"AviList 注册表不存在：{registry_path}")
    if not priors_path.is_file():
        raise FileNotFoundError(f"地域先验配置不存在：{priors_path}")

    taxa_path = resolve_manifest_file(asset_manifest, manifest_path, "birdTaxa")
    primary_embeddings_path = resolve_manifest_file(
        asset_manifest, manifest_path, "primaryEmbeddings"
    )
    supplemental_species_path = resolve_manifest_file(
        asset_manifest, manifest_path, "supplementalRegistry"
    )
    supplemental_embeddings_path = resolve_manifest_file(
        asset_manifest, manifest_path, "supplementalEmbeddings"
    )
    supplemental_from_manifest = (
        supplemental_species_path is not None and supplemental_embeddings_path is not None
    )
    taxa_path = taxa_path or BIRD_DATA_PATH / "bird_taxa.json"
    primary_embeddings_path = primary_embeddings_path or BIRD_DATA_PATH / "bird_embeddings.npy"
    supplemental_species_path = supplemental_species_path or SUPPLEMENTAL_SPECIES_PATH
    supplemental_embeddings_path = supplemental_embeddings_path or SUPPLEMENTAL_EMBEDDINGS_PATH

    taxa = json.loads(taxa_path.read_text(encoding="utf-8"))
    global_labels = [model_scientific_name(entry[0]) for entry in taxa]
    common_names = {
        scientific_name: entry[1] or scientific_name
        for scientific_name, entry in zip(global_labels, taxa)
    }
    global_text = np.load(primary_embeddings_path, allow_pickle=False).astype(np.float32)
    global_text /= np.maximum(np.linalg.norm(global_text, axis=1, keepdims=True), 1e-12)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_architecture, pretrained=str(model_weights_path), device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_architecture)
    prompt_rows = [(category, prompt) for category, prompts in ROUTER_PROMPTS.items() for prompt in prompts]
    with torch.inference_mode():
        model._scene_router_text = torch.nn.functional.normalize(
            model.encode_text(tokenizer([prompt for _, prompt in prompt_rows])), dim=-1
        )
    model._scene_router_groups = {
        category: [index for index, (name, _) in enumerate(prompt_rows) if name == category]
        for category in ROUTER_PROMPTS
    }
    registry, taxonomy_by_index, valid_taxonomy_indices = load_taxonomy_registry(registry_path)
    validate_taxonomy_alignment(global_labels, global_text, taxonomy_by_index)
    if registry["version"] != config["taxonomy"]["version"]:
        raise ValueError(
            f"分类配置要求 {config['taxonomy']['version']}，但注册表是 {registry['version']}。"
        )
    supplemental_registry, supplemental_text = load_supplemental_catalog(
        supplemental_species_path, supplemental_embeddings_path
    )
    if supplemental_registry["version"] != config["taxonomy"]["version"]:
        raise ValueError(
            f"分类配置要求 {config['taxonomy']['version']}，"
            f"但补充候选库是 {supplemental_registry['version']}。"
        )
    expected_supplemental_method = config["supplementalCandidates"]["embeddingMethod"]
    if (
        not supplemental_from_manifest
        and supplemental_registry.get("embeddingMethod") != expected_supplemental_method
    ):
        raise ValueError(
            f"补充候选配置要求 {expected_supplemental_method}，"
            f"但资产使用 {supplemental_registry.get('embeddingMethod')}。"
        )
    if supplemental_text.shape[1] != global_text.shape[1]:
        raise ValueError(
            "BioCLIP 主候选/补充候选向量维度不一致："
            f"primary={global_text.shape[1]}, supplemental={supplemental_text.shape[1]}."
        )
    primary_ids = {item["stableId"] for item in registry["species"]}
    overlapping_ids = primary_ids.intersection(
        item["stableId"] for item in supplemental_registry["species"]
    )
    if overlapping_ids:
        raise ValueError(f"AviList 主候选/补充候选 stableId 重叠：{len(overlapping_ids)} 条。")

    from bioclip25_runtime import apply_text_geometry, load_text_overrides

    text_overrides = None
    override_config = config.get("textOverrides") or {}
    if override_config.get("enabled") and asset_manifest is not None:
        override_path = Path(override_config["manifestFile"])
        if not override_path.is_absolute():
            override_path = config_root / override_path
        text_overrides = load_text_overrides(
            override_path,
            asset_manifest=asset_manifest,
            embedding_dimension=global_text.shape[1],
            taxonomy_by_index=taxonomy_by_index,
        )
        global_text[text_overrides["indices"]] = text_overrides["vectors"]
        model._text_override_traits = dict(text_overrides["traitsByStableId"])
    geometry_config = config.get("textGeometry") or {}
    removed_components = int(geometry_config.get("removePrincipalComponents", 0) or 0)
    apply_text_geometry(global_text, removed_components)
    apply_text_geometry(supplemental_text, removed_components)

    model._pipeline_config = config
    model._taxonomy_registry = registry
    model._taxonomy_by_index = taxonomy_by_index
    model._valid_taxonomy_indices = valid_taxonomy_indices
    model._region_priors = load_region_priors(priors_path)
    model._supplemental_registry = supplemental_registry
    model._supplemental_text = supplemental_text
    model._text_overrides = text_overrides
    model._text_geometry = {"removePrincipalComponents": removed_components}
    focus_net = cv2.dnn.readNetFromONNX(np.fromfile(CROP_MODEL_PATH, dtype=np.uint8)) if CROP_MODEL_PATH.exists() else None
    logit_scale = float(head_data["logit_scale"]) if head_data else 1.0
    return torch, model, preprocess, head, logit_scale, classes, global_labels, common_names, global_text, focus_net


def overlap(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def positions(full, tile, overlap_ratio):
    if tile >= full:
        return [0]
    step = max(1, round(tile * (1 - overlap_ratio)))
    values = list(range(0, full - tile + 1, step))
    if values[-1] != full - tile:
        values.append(full - tile)
    return values


def infer_birds(image, focus_net):
    import cv2

    width, height = image.size
    scale = min(DETECTION_INPUT_SIZE / width, DETECTION_INPUT_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    source = np.asarray(image)
    resized = cv2.resize(source, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized[:, :, ::-1]
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0, size=(DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE), swapRB=False, crop=False)
    focus_net.setInput(blob)
    rows = np.asarray(focus_net.forward()).reshape(-1, 85)

    candidates = []
    index = 0
    for stride in (8, 16, 32):
        size = DETECTION_INPUT_SIZE // stride
        for grid_y in range(size):
            for grid_x in range(size):
                row = rows[index]
                index += 1
                score = float(row[4] * row[5 + BIRD_CLASS_ID])
                if score < DETECTION_THRESHOLD:
                    continue
                center_x = (float(row[0]) + grid_x) * stride
                center_y = (float(row[1]) + grid_y) * stride
                box_width = np.exp(np.clip(float(row[2]), -20, 20)) * stride
                box_height = np.exp(np.clip(float(row[3]), -20, 20)) * stride
                candidates.append({
                    "score": score,
                    "box": [
                        max(0.0, (center_x - box_width / 2) / scale),
                        max(0.0, (center_y - box_height / 2) / scale),
                        min(float(width), (center_x + box_width / 2) / scale),
                        min(float(height), (center_y + box_height / 2) / scale),
                    ],
                })

    kept = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(overlap(candidate["box"], item["box"]) <= NMS_THRESHOLD for item in kept):
            kept.append(candidate)
    return kept


def primary_detection(candidates, width, height):
    frame_area = max(1, width * height)

    def priority(candidate):
        box = candidate["box"]
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        return candidate["score"] + min(0.18, np.sqrt(box_area / frame_area) * 1.2)

    return max(candidates, key=priority, default=None)


def detect_bird(image, focus_net):
    if focus_net is None:
        return None, 0, "model_unavailable"

    width, height = image.size
    passes = 1
    full_candidates = infer_birds(image, focus_net)
    full_best = primary_detection(full_candidates, width, height)
    if full_best and full_best["score"] >= FULL_FRAME_ACCEPT:
        return full_best, passes, "full"

    tiled = []
    seen_tiles = {}
    for fraction, tile_overlap in ZOOM_LEVELS:
        tile_width = min(width, max(DETECTION_INPUT_SIZE, round(width * fraction)))
        tile_height = min(height, max(DETECTION_INPUT_SIZE, round(height * fraction)))
        for top in positions(height, tile_height, tile_overlap):
            for left in positions(width, tile_width, tile_overlap):
                if left == 0 and top == 0 and tile_width == width and tile_height == height:
                    continue
                key = (left, top, tile_width, tile_height)
                cached = seen_tiles.get(key) if DETECT_DEDUP else None
                if cached is not None:
                    # Same pixels -> same network output. Extend with the very
                    # same translated candidates so `tiled`, `passes`, `break`
                    # and the returned best/source stay bit-identical.
                    tiled.extend(cached)
                    passes += 1
                    continue
                tile = image.crop((left, top, left + tile_width, top + tile_height))
                tile_candidates = infer_birds(tile, focus_net)
                passes += 1
                produced = []
                for candidate in tile_candidates:
                    box = candidate["box"]
                    produced.append({
                        "score": candidate["score"],
                        "box": [box[0] + left, box[1] + top, box[2] + left, box[3] + top],
                    })
                tiled.extend(produced)
                if DETECT_DEDUP:
                    seen_tiles[key] = produced
        if any(candidate["score"] >= 0.15 for candidate in tiled):
            break

    best = primary_detection(([full_best] if full_best else []) + tiled, width, height)
    return best, passes, "zoom_tiles" if best in tiled else "full" if best else "none"


def create_focus_view(image, focus_net):
    started = time.perf_counter()
    detection, passes, source = detect_bird(image, focus_net)
    width, height = image.size
    meta = {
        "modelAvailable": focus_net is not None,
        "source": source,
        "passes": passes,
        "score": None,
        "box": None,
        "birdOccupancy": 0.0,
        "focusUsed": False,
        "focusReason": "bird_not_detected",
    }
    if not detection:
        meta["latencyMs"] = round((time.perf_counter() - started) * 1000)
        return None, meta

    box = detection["box"]
    box_width = max(1.0, box[2] - box[0])
    box_height = max(1.0, box[3] - box[1])
    occupancy = box_width * box_height / max(1, width * height)
    edge_margin_x = width * 0.02
    edge_margin_y = height * 0.02
    clipped_edges = int(sum((
        box[0] <= edge_margin_x,
        box[1] <= edge_margin_y,
        box[2] >= width - edge_margin_x,
        box[3] >= height - edge_margin_y,
    )))
    meta.update({
        "score": round(float(detection["score"]), 4),
        "box": [round(value) for value in box],
        "birdOccupancy": round(float(occupancy), 4),
        "clippedEdges": clipped_edges,
    })

    if occupancy >= FOCUS_MAX_OCCUPANCY or clipped_edges >= 2:
        meta["focusReason"] = "bird_already_large_or_clipped"
        meta["latencyMs"] = round((time.perf_counter() - started) * 1000)
        return None, meta

    side = max(box_width, box_height) * FOCUS_CONTEXT_FACTOR
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    left = max(0.0, min(width - side, center_x - side / 2))
    top = max(0.0, min(height - side, center_y - side / 2))
    right = min(float(width), left + side)
    bottom = min(float(height), top + side)
    focus = image.crop((round(left), round(top), round(right), round(bottom)))
    meta["focusUsed"] = True
    meta["focusReason"] = "small_or_distant_bird"
    meta["focusBox"] = [round(left), round(top), round(right), round(bottom)]
    meta["latencyMs"] = round((time.perf_counter() - started) * 1000)
    return focus, meta


def local_quality_score(image, focus_meta):
    import cv2

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    if max(gray.shape) > 512:
        scale = 512 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luminance = float(gray.mean())
    dark_ratio = float((gray <= 18).mean())
    bright_ratio = float((gray >= 237).mean())

    detection_score = float(focus_meta.get("score") or 0)
    detection_factor = max(0.0, min(1.0, (detection_score - DETECTION_THRESHOLD) / (FULL_FRAME_ACCEPT - DETECTION_THRESHOLD)))
    clarity_factor = max(0.0, min(1.0, (sharpness - 20) / 180))
    mean_penalty = max(0.0, min(1.0, (abs(mean_luminance - 128) - 25) / 90))
    clipping_penalty = max(0.0, min(1.0, (dark_ratio + bright_ratio) / 0.35))
    light_factor = max(0.0, min(1.0, 1 - mean_penalty * 0.65 - clipping_penalty * 0.55))
    clipped_edges = int(focus_meta.get("clippedEdges", 4))
    occupancy = float(focus_meta.get("birdOccupancy") or 0)

    completeness = round(30 * detection_factor * max(0.0, min(1.0, 1 - clipped_edges * 0.18)))
    clarity = round(25 * clarity_factor)
    angle = round(20 * detection_factor)
    light = round(15 * light_factor)
    distance = round(10 * max(0.0, min(1.0, occupancy / 0.3)))
    return max(0, min(100, completeness + clarity + angle + light + distance))


def top_rows(scores, labels, common_names, probability=False):
    order = np.argsort(scores)[::-1][:TOP_K]
    rows = []
    for rank, index in enumerate(order, 1):
        scientific_name = labels[int(index)]
        row = {
            "rank": rank,
            "scientificName": scientific_name,
            "commonName": common_names.get(scientific_name, scientific_name),
        }
        row["relativeProbability" if probability else "similarity"] = round(float(scores[index]), 6)
        rows.append(row)
    return rows


def supplemental_candidate_rows(scores, registry, top_k=TOP_K, region=None, region_priors=None, max_bonus=0.0, geographic_bonuses=None):
    geographic_bonuses = {k: v for k, v in (geographic_bonuses or {}).items() if v != 0}
    scores = np.asarray(scores, dtype=np.float32)
    species = registry["species"]
    if scores.ndim != 1 or len(scores) != len(species):
        raise ValueError(
            f"补充候选分数/物种数不对齐：scores={scores.shape}, species={len(species)}."
        )
    bonuses = {}
    if region and region_priors:
        bonuses = region_priors["regions"][region].get("species", {})
    adjusted = scores.copy()
    if bonuses:
        for index, item in enumerate(species):
            adjusted[index] += max_bonus * float(bonuses.get(item["stableId"], 0.0))

    if geographic_bonuses:
        for index, item in enumerate(species):
            adjusted[index] += geographic_bonuses.get(item["stableId"], 0.0)
    rows = []
    for index in np.argsort(adjusted)[::-1][:top_k]:
        item = species[int(index)]
        raw_score = float(scores[int(index)])
        bonus = max_bonus * float(bonuses.get(item["stableId"], 0.0)) if bonuses else 0.0
        rows.append({
            "rank": len(rows) + 1,
            "stableId": item["stableId"],
            "scientificName": item["scientificName"],
            "commonName": item["commonName"],
            "order": item["order"],
            "family": item["family"],
            "familyEnglishName": item["familyEnglishName"],
            "genus": item["genus"],
            "modelScientificName": item["scientificName"],
            "taxonomyMatchMethod": "avilist_missing_text_prompt_ensemble",
            "candidateSource": "avilist_supplemental_text",
            "scoreSpace": "bioclip_prompt_ensemble_cosine",
            "similarity": round(raw_score, 6),
            "regionBonus": round(bonus, 6),
            "adjustedSimilarity": round(raw_score + bonus + (geographic_bonuses or {}).get(item["stableId"], 0.0), 6),
            **({"geographicBonus": round(geographic_bonuses.get(item["stableId"], 0.0), 6)} if geographic_bonuses else {}),
        })
    return rows


def select_review_candidates(local_result):
    """候选统一入口：所有模块都从这里取复核候选，避免兜底链各写一套。"""
    global_result = (local_result or {}).get("globalBioCLIP") or {}
    return (
        global_result.get("reviewTop5")
        or global_result.get("reviewCandidates")
        or global_result.get("reviewTop3")
        or global_result.get("rawTop5")
        or global_result.get("regionAdjustedTop5")
        or []
    )


def merge_candidate_rows(primary_rows, supplemental_rows, score_key="similarity", limit=TOP_K):
    merged = []
    seen = set()
    for row in sorted(
        [
            *(
                {**item, "candidateSource": item.get("candidateSource", "tree_of_life_precomputed")}
                for item in (primary_rows or [])
            ),
            *(dict(item) for item in (supplemental_rows or [])),
        ],
        key=lambda item: (-float(item[score_key]), item["stableId"]),
    ):
        if row["stableId"] in seen:
            continue
        seen.add(row["stableId"])
        merged.append({**row, "rank": len(merged) + 1})
        if len(merged) == limit:
            break
    return merged


def build_review_shortlist(
    primary_rows,
    supplemental_rows,
    max_candidates=3,
    merge_mode="legacy_conflict_gate",
):
    primary_rows = list(primary_rows or [])
    supplemental_rows = list(supplemental_rows or [])
    default_rows = [
        {**row, "candidateSource": row.get("candidateSource", "tree_of_life_precomputed")}
        for row in primary_rows[:max_candidates]
    ]
    if not primary_rows or not supplemental_rows:
        return default_rows, {
            "conflict": False,
            "reason": "candidate_stream_unavailable",
            "gateStatus": "conservative_unvalidated",
            "autoRoutingEnabled": False,
        }

    primary_scores = [float(row["similarity"]) for row in primary_rows]
    if len(primary_scores) >= 2:
        mean_rank_step = (primary_scores[0] - primary_scores[-1]) / (len(primary_scores) - 1)
        extrapolated_floor = primary_scores[-1] - max(0.0, mean_rank_step)
    else:
        mean_rank_step = 0.0
        extrapolated_floor = primary_scores[0]
    supplemental_top = supplemental_rows[0]
    cross_family = bool(
        primary_rows[0].get("family")
        and supplemental_top.get("family")
        and primary_rows[0]["family"] != supplemental_top["family"]
    )
    competitive = float(supplemental_top["similarity"]) >= extrapolated_floor
    conflict = cross_family and competitive

    if merge_mode == "unified_score_topk":
        selected = merge_candidate_rows(
            primary_rows,
            supplemental_rows,
            score_key="similarity",
            limit=max_candidates,
        )
    elif merge_mode == "legacy_conflict_gate" and conflict:
        selected = [
            {**primary_rows[0], "candidateSource": primary_rows[0].get("candidateSource", "tree_of_life_precomputed")},
            *supplemental_rows[:2],
        ]
    elif merge_mode == "legacy_conflict_gate":
        selected = default_rows
    else:
        raise ValueError(f"Unsupported supplemental merge mode: {merge_mode}")
    seen = set()
    shortlist = []
    for row in selected:
        if row["stableId"] in seen:
            continue
        seen.add(row["stableId"])
        shortlist.append({**row, "rank": len(shortlist) + 1})
        if len(shortlist) == max_candidates:
            break

    return shortlist, {
        "conflict": conflict,
        "reason": (
            "unified_score_topk"
            if merge_mode == "unified_score_topk"
            else "cross_family_competitive_supplement"
            if conflict
            else "no_competitive_cross_family_supplement"
        ),
        "mergeMode": merge_mode,
        "gateStatus": "conservative_unvalidated",
        "autoRoutingEnabled": False,
        "primaryTop1Similarity": round(primary_scores[0], 6),
        "primaryExtrapolatedNextRankFloor": round(extrapolated_floor, 6),
        "primaryMeanRankStep": round(mean_rank_step, 6),
        "supplementalTop1Similarity": round(float(supplemental_top["similarity"]), 6),
        "supplementalTop1Family": supplemental_top.get("family"),
        "primaryTop1Family": primary_rows[0].get("family"),
    }


def route_scene(feature, focus_meta, model):
    prompt_scores = (feature @ model._scene_router_text.T)[0]
    scores = {
        category: float(prompt_scores[indices].mean())
        for category, indices in model._scene_router_groups.items()
    }
    bird_score = scores["bird"]
    squirrel_score = scores["squirrel"]
    bird_advantage = bird_score - squirrel_score
    detection_score = float(focus_meta.get("score") or 0)

    if squirrel_score - bird_score >= ROUTER_SQUIRREL_MARGIN:
        if squirrel_score >= ROUTER_SQUIRREL_SCORE:
            route = "non_bird"
            reason = "strong_squirrel_semantic_evidence"
        else:
            route = "uncertain"
            reason = "weak_squirrel_semantic_evidence"
    elif detection_score >= FULL_FRAME_ACCEPT:
        route = "bird"
        reason = "strong_bird_detection"
    elif detection_score >= ROUTER_WEAK_BIRD_DETECTION and bird_advantage >= -0.015:
        route = "bird"
        reason = "bird_detection_and_semantic_support"
    else:
        route = "uncertain"
        reason = "insufficient_bird_evidence"

    return {
        "route": route,
        "reason": reason,
        "birdScore": round(bird_score, 6),
        "squirrelScore": round(squirrel_score, 6),
        "birdAdvantage": round(bird_advantage, 6),
        "birdDetectionScore": round(detection_score, 6),
        "thresholds": {
            "squirrelMargin": ROUTER_SQUIRREL_MARGIN,
            "squirrelScore": ROUTER_SQUIRREL_SCORE,
            "strongBirdDetection": FULL_FRAME_ACCEPT,
            "weakBirdDetection": ROUTER_WEAK_BIRD_DETECTION,
        },
    }


def assess_risk(local_result, config):
    policy = config["localDecision"]
    focus = local_result["birdFocus"]
    rows = local_result["globalBioCLIP"]["regionAdjustedTop5"]
    blocking = []
    auxiliary = []
    route = local_result["sceneRouter"]["route"]
    if route != "bird":
        blocking.append("scene_" + route)
    if focus.get("box") is None:
        blocking.append("target_not_detected")
    elif float(focus.get("score") or 0) < float(policy["minimumDetectionScore"]):
        blocking.append("detection_unreliable")
    if int(focus.get("clippedEdges") or 0) > int(policy["maximumClippedEdges"]):
        blocking.append("bird_incomplete_or_clipped")
    if policy.get("requireViewAgreement") and focus.get("viewConsistency") == "conflict":
        blocking.append("original_focus_conflict")
    supplemental_coverage = local_result["globalBioCLIP"].get("supplementalCoverage") or {}
    if supplemental_coverage.get("conflict") and config.get("supplementalCandidates", {}).get(
        "blockOnCrossFamilyConflict", True
    ):
        blocking.append("taxonomy_coverage_cross_family_conflict")

    margin_threshold = policy.get("marginThreshold")
    margin = None
    if len(rows) >= 2:
        margin = float(rows[0]["adjustedSimilarity"] - rows[1]["adjustedSimilarity"])
    if margin_threshold is None:
        blocking.append("margin_threshold_not_calibrated")
    elif margin is None or margin < float(margin_threshold):
        blocking.append("candidate_margin_below_calibrated_threshold")

    top3_families = {row["family"] for row in rows[:3] if row.get("family")}
    if len(top3_families) > 1:
        auxiliary.append("top3_cross_family")
    if float(focus.get("birdOccupancy") or 0) >= 0.55:
        auxiliary.append("bird_large_in_frame")

    return {
        "needsReview": bool(blocking),
        "blockingReasons": blocking,
        "auxiliarySignals": auxiliary,
        "metrics": {
            "candidateMargin": round(margin, 6) if margin is not None else None,
            "marginThreshold": margin_threshold,
            "qualityScore": local_result["qualityScore"],
            "detectionScore": focus.get("score"),
            "birdOccupancy": focus.get("birdOccupancy"),
            "clippedEdges": focus.get("clippedEdges"),
            "supplementalTop1Similarity": supplemental_coverage.get("supplementalTop1Similarity"),
            "supplementalConflictFloor": supplemental_coverage.get("primaryExtrapolatedNextRankFloor"),
        },
        "policyVersion": config["version"],
        "calibration": policy.get("calibration"),
    }


def finalize_local_candidate(local_result, provisional_candidate, config):
    local_result["risk"] = assess_risk(local_result, config)
    if local_result["risk"]["needsReview"]:
        local_result["selectedCandidate"] = None
        local_result["candidateDisposition"] = "diagnostic_only"
    else:
        local_result["selectedCandidate"] = provisional_candidate
        local_result["candidateDisposition"] = "eligible_for_local_decision"
    return local_result


def base_boss_report(quality_score):
    return {
        "qualityScore": int(quality_score),
        "canIdentify": False,
        "describe": "",
        "categories": "Unknown",
        "alias": "",
        "name": "Unknown",
        "gender": "Unknown",
        "age": "Unknown",
        "plumage": "Unknown",
        "healthStatus": 8000,
        "healthIssues": [],
        "healthDescription": "No health conclusion was made from the local classifier.",
        "rarity": "Unknown",
        "rarityReason": "Location was not used for rarity assessment.",
        "colorTags": [],
        "patternTags": [],
        "shapeTags": [],
        "behaviorTags": [],
        "flock": False,
        "bird": False,
        "person": False,
        "squirrel": False,
        "raccoon": False,
        "rattus": False,
        "snake": False,
        "opossum": False,
        "animal": False,
        "confidence": 0.0,
    }


def uncertain_report(local_result, reason):
    report = base_boss_report(local_result["qualityScore"])
    focus = local_result["birdFocus"]
    route = local_result["sceneRouter"]["route"]
    if route == "bird":
        report["name"] = "Unknown Bird"
        report["bird"] = True
        report["animal"] = True
    if reason == "target_not_detected" or focus.get("box") is None:
        report["describe"] = "Target presence could not be verified; no species conclusion was made."
    elif route == "non_bird":
        report["describe"] = "The local router indicated a possible non-bird, but presence was not independently confirmed."
    else:
        report["describe"] = "Image evidence was insufficient to confirm presence or species."
    return report


def local_identified_report(local_result):
    candidate = local_result["selectedCandidate"]
    report = base_boss_report(local_result["qualityScore"])
    calibration = local_result["risk"]["calibration"] or {}
    empirical_precision = calibration.get("selectedPrecision")
    report.update({
        "canIdentify": True,
        "describe": "Original and focused BioCLIP views passed the calibrated local decision policy.",
        "categories": candidate["scientificName"],
        "name": candidate["commonName"],
        "bird": True,
        "animal": True,
        "confidence": round(float(empirical_precision or 0.0), 4),
    })
    return report


def _cached_focus(image_path, frame_detections):
    """取选帧阶段已算好的 focus，避免同一帧重复检测。"""
    if not frame_detections:
        return None, None, False
    entry = frame_detections.get(str(Path(image_path).resolve()))
    if not entry:
        return None, None, False
    return entry.get("focus"), entry.get("focusMeta"), True


def encode_frame_embedding(image_path, torch, preprocess, model, focus_net,
                           use_crop, frame_detections=None):
    """Encode one extra frame of the same event the same way the main frame is."""

    from PIL import Image, ImageOps

    with Image.open(image_path) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    cached_focus, cached_meta, cached = _cached_focus(image_path, frame_detections)
    if cached:
        detected_focus, focus_meta = cached_focus, cached_meta
    else:
        with _MODEL_LOCK:
            detected_focus, focus_meta = create_focus_view(original, focus_net)
    focus = detected_focus if use_crop else None
    views = [original] + ([focus] if focus is not None else [])
    tensor = torch.stack([preprocess(view) for view in views]).to(
        getattr(model, "_inference_device", "cpu")
    )
    with _MODEL_LOCK, torch.inference_mode():
        features = torch.nn.functional.normalize(model.encode_image(tensor), dim=-1)
        if focus is not None:
            detection_score = float(focus_meta["score"] or 0)
            focus_weight = 0.65 if detection_score >= FULL_FRAME_ACCEPT else 0.55
            feature = torch.nn.functional.normalize(
                features[0] * (1 - focus_weight) + features[1] * focus_weight,
                dim=-1,
            )
        else:
            feature = features[0]
    return feature.cpu().numpy(), focus_meta


def analyze_local(image_path, runtime, region=None, use_crop=True,
                  extra_frames=(), observation=None, frame_detections=None):
    from PIL import Image, ImageOps

    torch, model, preprocess, head, logit_scale, classes, global_labels, common_names, global_text, focus_net = runtime
    config = model._pipeline_config
    region = validate_region(region, model._region_priors)
    started = time.perf_counter()
    with Image.open(image_path) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    cached_focus, cached_meta, cached = _cached_focus(image_path, frame_detections)
    if cached:
        detected_focus, focus_meta = cached_focus, cached_meta
    else:
        with _MODEL_LOCK:
            detected_focus, focus_meta = create_focus_view(original, focus_net)
    focus = detected_focus if use_crop else None
    if not use_crop and detected_focus is not None:
        focus_meta = {**focus_meta, "focusUsed": False, "focusReason": "disabled_by_option"}
    quality_score = local_quality_score(original, focus_meta)
    fine_grained_settings = parse_fine_grained_settings(config)
    views = [original] + ([focus] if focus is not None else [])
    tensor = torch.stack([preprocess(view) for view in views]).to(
        getattr(model, "_inference_device", "cpu")
    )
    with _MODEL_LOCK, torch.inference_mode():
        features = torch.nn.functional.normalize(model.encode_image(tensor), dim=-1)
        if focus is not None:
            detection_score = float(focus_meta["score"] or 0)
            focus_weight = 0.65 if detection_score >= FULL_FRAME_ACCEPT else 0.55
            feature = torch.nn.functional.normalize(
                features[0] * (1 - focus_weight) + features[1] * focus_weight,
                dim=-1,
            ).unsqueeze(0)
        else:
            focus_weight = 0.0
            feature = features[0].unsqueeze(0)
        head_probabilities = None
        if head is not None:
            head_logits = logit_scale * head(feature)
            head_probabilities = torch.softmax(head_logits, dim=-1).cpu().numpy()[0]

    image_embedding = feature.cpu().numpy()[0]
    frame_embeddings = [image_embedding]
    frame_focus = [focus_meta]
    for extra in extra_frames:
        if extra is None:
            continue
        extra_feature, extra_focus_meta = encode_frame_embedding(
            extra, torch, preprocess, model, focus_net, use_crop, frame_detections
        )
        frame_embeddings.append(extra_feature)
        frame_focus.append(extra_focus_meta)
    if len(frame_embeddings) > 1:
        # elementwise max keeps the strongest evidence from any frame instead of
        # averaging a good frame with a blurred or rear-view one
        image_embedding = np.max(np.stack(frame_embeddings), axis=0)
        image_embedding = image_embedding / max(
            float(np.linalg.norm(image_embedding)), 1e-12
        )
    fusion_mode = config.get("multiFrame", {}).get("candidateFusion", "embedding_max")
    if fusion_mode not in {"embedding_max", "score_mean"}:
        raise ValueError("multiFrame.candidateFusion must be embedding_max or score_mean")
    if len(frame_embeddings) > 1 and fusion_mode == "score_mean":
        frame_matrix = np.stack(frame_embeddings)
        baseline_global_scores = (global_text @ frame_matrix.T).mean(axis=1)
        baseline_supplemental_scores = (model._supplemental_text @ frame_matrix.T).mean(axis=1)
    else:
        baseline_global_scores = global_text @ image_embedding
        baseline_supplemental_scores = model._supplemental_text @ image_embedding
    original_scores = global_text @ features[0].cpu().numpy()
    supplemental_original_scores = model._supplemental_text @ features[0].cpu().numpy()
    focus_scores = global_text @ features[1].cpu().numpy() if focus is not None else None
    supplemental_focus_scores = (
        model._supplemental_text @ features[1].cpu().numpy()
        if focus is not None
        else None
    )

    # Hard geographic filter: when the device supplies a valid location, mask
    # every species without support in the containing distribution cell so the
    # local retrieval stage can only rank the regional avifauna. Missing
    # location / coverage gaps / tiny pools fall back to the full taxonomy.
    from geographic_prior import allowed_species as geographic_allowed_species

    geo_filter_observation = dict(observation or config.get("location") or {})
    geo_filter_ids, geo_filter_status = geographic_allowed_species(
        config.get("geographicPrior") or {},
        geo_filter_observation,
        root=Path(config["_path"]).parent,
        taxonomy_version=config.get("taxonomy", {}).get("version"),
    )
    if geo_filter_ids is not None:
        taxonomy_by_index = model._taxonomy_by_index
        valid_taxonomy_indices = model._valid_taxonomy_indices
        row_count = int(np.asarray(baseline_global_scores).shape[0])
        tax_mask = np.zeros(row_count, dtype=bool)
        for index in valid_taxonomy_indices:
            item = taxonomy_by_index.get(int(index))
            if item and str(item.get("stableId")) in geo_filter_ids:
                tax_mask[int(index)] = True
        supplemental_mask = np.asarray([
            str(item.get("stableId")) in geo_filter_ids
            for item in (model._supplemental_registry.get("species") or [])
        ], dtype=bool)
        baseline_global_scores = np.where(tax_mask, baseline_global_scores, -1.0)
        original_scores = np.where(tax_mask, original_scores, -1.0)
        if focus_scores is not None:
            focus_scores = np.where(tax_mask, focus_scores, -1.0)
        baseline_supplemental_scores = np.where(
            supplemental_mask, baseline_supplemental_scores, -1.0
        )
        supplemental_original_scores = np.where(
            supplemental_mask, supplemental_original_scores, -1.0
        )
        if supplemental_focus_scores is not None:
            supplemental_focus_scores = np.where(
                supplemental_mask, supplemental_focus_scores, -1.0
            )
        geo_filter_status["maskedSpecies"] = int(
            len(valid_taxonomy_indices) - int(tax_mask.sum())
        )
        geo_filter_status["allowedInRegistry"] = int(tax_mask.sum())
        geo_filter_status["supplementalAllowed"] = int(supplemental_mask.sum())

    detail_trigger_reason = "disabled"
    should_build_details = False
    if fine_grained_settings["enabled"] and focus is not None:
        if fine_grained_settings["triggerMode"] == "always":
            should_build_details = True
            detail_trigger_reason = "always"
        else:
            original_top_index = int(np.argmax(original_scores))
            focus_top_index = int(np.argmax(focus_scores))
            should_build_details = original_top_index != focus_top_index
            detail_trigger_reason = (
                "view_conflict" if should_build_details else "view_agreement"
            )
    elif fine_grained_settings["enabled"]:
        detail_trigger_reason = "focus_unavailable"

    detail_views = (
        build_detail_views(
            focus,
            crop_fraction=fine_grained_settings["cropFraction"],
            minimum_side=fine_grained_settings["minimumSide"],
        )
        if should_build_details
        else []
    )
    fine_global_scores = None
    fine_supplemental_scores = None
    if detail_views:
        detail_tensor = torch.stack(
            [preprocess(item["image"]) for item in detail_views]
        ).to(getattr(model, "_inference_device", "cpu"))
        with _MODEL_LOCK, torch.inference_mode():
            detail_embeddings = torch.nn.functional.normalize(
                model.encode_image(detail_tensor), dim=-1
            ).cpu().numpy()
        fine_global_scores = aggregate_scores(
            baseline_global_scores,
            global_text @ detail_embeddings.T,
            detail_weight=fine_grained_settings["detailWeight"],
            top_views=fine_grained_settings["topViews"],
        )
        fine_supplemental_scores = aggregate_scores(
            baseline_supplemental_scores,
            model._supplemental_text @ detail_embeddings.T,
            detail_weight=fine_grained_settings["detailWeight"],
            top_views=fine_grained_settings["topViews"],
        )
    fine_grained_active = bool(
        detail_views and fine_grained_settings["mode"] == "rerank"
    )
    global_scores = fine_global_scores if fine_grained_active else baseline_global_scores
    supplemental_scores = (
        fine_supplemental_scores
        if fine_grained_active
        else baseline_supplemental_scores
    )
    prototype_started = time.perf_counter()
    prototype_gallery = getattr(model, "_image_prototype_gallery", None)
    prototype_config = getattr(model, "_image_prototype_config", {}) or {}
    prototype_query_view = prototype_config.get("queryView", "combined")
    if prototype_query_view == "combined":
        prototype_query = image_embedding
    elif prototype_query_view == "original":
        prototype_query = features[0].cpu().numpy()
    else:
        raise ValueError(
            f"Unsupported image prototype query view: {prototype_query_view}"
        )
    prototype_rows = score_prototype_candidates(
        prototype_query,
        prototype_gallery,
        top_k=int(prototype_config.get("maxCandidates", TOP_K)),
    )
    prototype_status = dict(
        getattr(
            model,
            "_image_prototype_status",
            {"available": False, "reason": "not_configured"},
        )
    )
    prototype_status["scoreLatencyMs"] = round(
        (time.perf_counter() - prototype_started) * 1000
    )
    prototype_status["queryView"] = prototype_query_view
    max_bonus = float(config["region"]["maxBonus"])
    primary_raw_rows = candidate_rows(
        global_scores, model._taxonomy_by_index, model._valid_taxonomy_indices, TOP_K
    )
    primary_adjusted_rows = candidate_rows(
        global_scores, model._taxonomy_by_index, model._valid_taxonomy_indices, TOP_K,
        region=region, region_priors=model._region_priors, max_bonus=max_bonus,
    )
    primary_original_rows = candidate_rows(
        original_scores, model._taxonomy_by_index, model._valid_taxonomy_indices, TOP_K,
        region=region, region_priors=model._region_priors, max_bonus=max_bonus,
    )
    primary_original_raw_rows = candidate_rows(
        original_scores, model._taxonomy_by_index, model._valid_taxonomy_indices, TOP_K
    )
    primary_focus_rows = candidate_rows(
        focus_scores, model._taxonomy_by_index, model._valid_taxonomy_indices, TOP_K,
        region=region, region_priors=model._region_priors, max_bonus=max_bonus,
    ) if focus_scores is not None else None
    primary_focus_raw_rows = candidate_rows(
        focus_scores, model._taxonomy_by_index, model._valid_taxonomy_indices, TOP_K
    ) if focus_scores is not None else None
    supplemental_raw_rows = supplemental_candidate_rows(
        supplemental_scores, model._supplemental_registry, TOP_K
    )
    supplemental_rows = supplemental_candidate_rows(
        supplemental_scores, model._supplemental_registry, TOP_K,
        region=region, region_priors=model._region_priors, max_bonus=max_bonus,
    )
    supplemental_original_raw_rows = supplemental_candidate_rows(
        supplemental_original_scores, model._supplemental_registry, TOP_K
    )
    supplemental_original_rows = supplemental_candidate_rows(
        supplemental_original_scores, model._supplemental_registry, TOP_K,
        region=region, region_priors=model._region_priors, max_bonus=max_bonus,
    )
    supplemental_focus_raw_rows = supplemental_candidate_rows(
        supplemental_focus_scores, model._supplemental_registry, TOP_K
    ) if supplemental_focus_scores is not None else None
    supplemental_focus_rows = supplemental_candidate_rows(
        supplemental_focus_scores, model._supplemental_registry, TOP_K,
        region=region, region_priors=model._region_priors, max_bonus=max_bonus,
    ) if supplemental_focus_scores is not None else None

    fine_grained_rows = None
    if fine_global_scores is not None:
        fine_primary_rows = candidate_rows(
            fine_global_scores,
            model._taxonomy_by_index,
            model._valid_taxonomy_indices,
            TOP_K,
            region=region,
            region_priors=model._region_priors,
            max_bonus=max_bonus,
        )
        fine_supplemental_rows = supplemental_candidate_rows(
            fine_supplemental_scores,
            model._supplemental_registry,
            TOP_K,
            region=region,
            region_priors=model._region_priors,
            max_bonus=max_bonus,
        )
        fine_grained_rows = merge_candidate_rows(
            fine_primary_rows,
            fine_supplemental_rows,
            score_key="adjustedSimilarity",
        )

    merge_mode = config["supplementalCandidates"].get("mergeMode", "legacy_conflict_gate")
    if merge_mode == "unified_score_topk":
        raw_rows = merge_candidate_rows(primary_raw_rows, supplemental_raw_rows)
        adjusted_rows = merge_candidate_rows(
            primary_adjusted_rows,
            supplemental_rows,
            score_key="adjustedSimilarity",
        )
        original_raw_rows = merge_candidate_rows(
            primary_original_raw_rows, supplemental_original_raw_rows
        )
        original_rows = merge_candidate_rows(
            primary_original_rows,
            supplemental_original_rows,
            score_key="adjustedSimilarity",
        )
        focus_raw_rows = (
            merge_candidate_rows(primary_focus_raw_rows, supplemental_focus_raw_rows)
            if primary_focus_raw_rows is not None
            else None
        )
        focus_rows = (
            merge_candidate_rows(
                primary_focus_rows,
                supplemental_focus_rows,
                score_key="adjustedSimilarity",
            )
            if primary_focus_rows is not None
            else None
        )
    elif merge_mode == "legacy_conflict_gate":
        raw_rows = primary_raw_rows
        adjusted_rows = primary_adjusted_rows
        original_raw_rows = primary_original_raw_rows
        original_rows = primary_original_rows
        focus_raw_rows = primary_focus_raw_rows
        focus_rows = primary_focus_rows
    else:
        raise ValueError(f"Unsupported supplemental merge mode: {merge_mode}")

    open_set_config = dict(config.get("openSetEscape") or {})
    open_set_pool = []
    if bool(open_set_config.get("enabled", False)):
        open_set_pool_size = int(open_set_config.get("maximumBioCLIPRank", 50))
        if not 6 <= open_set_pool_size <= 100:
            raise ValueError(
                "openSetEscape.maximumBioCLIPRank must be between 6 and 100"
            )
        primary_open_set_rows = candidate_rows(
            global_scores,
            model._taxonomy_by_index,
            model._valid_taxonomy_indices,
            open_set_pool_size,
        )
        supplemental_open_set_rows = supplemental_candidate_rows(
            supplemental_scores,
            model._supplemental_registry,
            open_set_pool_size,
        )
        open_set_pool = (
            merge_candidate_rows(
                primary_open_set_rows,
                supplemental_open_set_rows,
                limit=open_set_pool_size,
            )
            if merge_mode == "unified_score_topk"
            else primary_open_set_rows
        )

    review_max_candidates = int(config["review"].get("maxCandidates", 3))
    if not 1 <= review_max_candidates <= TOP_K:
        raise ValueError(f"review.maxCandidates must be between 1 and {TOP_K}")
    review_candidates, supplemental_coverage = build_review_shortlist(
        primary_raw_rows,
        supplemental_raw_rows,
        max_candidates=review_max_candidates,
        merge_mode=merge_mode,
    )
    from geographic_prior import geographic_support
    # Request overrides replace, never mutate or partially merge shared location.
    observation = dict(config.get("location") or {}) if observation is None else dict(observation)
    geo_bonuses, geo_status = geographic_support(
        config.get("geographicPrior") or {}, observation,
        root=Path(config["_path"]).parent,
        taxonomy_version=config.get("taxonomy", {}).get("version"),
    )
    if geo_bonuses:
        geo_primary = candidate_rows(
            global_scores, model._taxonomy_by_index, model._valid_taxonomy_indices,
            review_max_candidates, geographic_bonuses=geo_bonuses,
        )
        geo_supplemental = supplemental_candidate_rows(
            supplemental_scores, model._supplemental_registry,
            review_max_candidates, geographic_bonuses=geo_bonuses,
        )
        # Before-TopK support can recall a local species outside raw Top5.
        # Keep pure visual rankings/margins and all automatic gates untouched.
        review_candidates = merge_candidate_rows(
            geo_primary, geo_supplemental if merge_mode == "unified_score_topk" else [],
            score_key="adjustedSimilarity", limit=review_max_candidates,
        )
    geo_status["retrievalScope"] = (
        "global_candidates" if geo_filter_status.get("applied") else "review_shortlist_only"
    )
    geo_status["visualScoresUnchanged"] = not bool(geo_filter_status.get("applied"))
    geo_status["filterMode"] = geo_filter_status.get("filterMode")
    geo_status["filter"] = {
        "applied": bool(geo_filter_status.get("applied")),
        "reason": geo_filter_status.get("reason"),
        "cellId": geo_filter_status.get("cellId"),
        "poolSpecies": geo_filter_status.get("poolSpecies"),
        "minPoolSpecies": geo_filter_status.get("minPoolSpecies"),
        "maskedSpecies": geo_filter_status.get("maskedSpecies"),
        "allowedInRegistry": geo_filter_status.get("allowedInRegistry"),
    }
    geo_status["hardFiltering"] = bool(geo_filter_status.get("applied"))
    scene_router = route_scene(feature, focus_meta, model)
    prototype_bird_gate = prototype_config.get(
        "birdPresenceGate", "scene_router"
    )
    if prototype_bird_gate == "scene_router":
        prototype_bird_present = scene_router.get("route") == "bird"
    elif prototype_bird_gate == "detector":
        prototype_bird_present = (
            float(focus_meta.get("score") or 0) >= DETECTION_THRESHOLD
        )
    elif prototype_bird_gate == "not_non_bird":
        prototype_bird_present = scene_router.get("route") != "non_bird"
    else:
        raise ValueError(
            f"Unsupported image prototype bird-presence gate: {prototype_bird_gate}"
        )
    review_candidates, prototype_fusion = fuse_review_candidates(
        review_candidates,
        prototype_rows,
        max_candidates=review_max_candidates,
        mode=prototype_config.get("mode", "diagnostic"),
        maximum_injected=int(
            prototype_config.get("maximumInjectedCandidates", 1)
        ),
        calibration_status=prototype_config.get(
            "calibrationStatus", "not_calibrated"
        ),
        allowed_stable_ids=prototype_config.get("allowedStableIds") or [],
        minimum_similarity=prototype_config.get("minimumSimilarity"),
        minimum_margin=prototype_config.get("minimumMargin"),
        requires_bird_presence=bool(
            prototype_config.get("requiresBirdPresence", False)
        ),
        bird_present=prototype_bird_present,
    )
    prototype_fusion["birdPresenceGate"] = prototype_bird_gate
    provisional_candidate = {
        **adjusted_rows[0],
        "reason": "region_soft_adjusted_top1" if region and region != "GLOBAL" else "global_bioclip_top1",
        "classifierHeadChangedGlobalTop1": False,
    }
    original_id = original_rows[0]["stableId"]
    focus_id = focus_rows[0]["stableId"] if focus_rows else None
    view_consistency = "agree" if focus_id == original_id else "conflict" if focus_id else "single_view"
    local_result = {
        "image": str(Path(image_path).resolve()),
        "qualityScore": quality_score,
        "sceneRouter": scene_router,
        "provisionalCandidate": provisional_candidate,
        "selectedCandidate": None,
        "candidateDisposition": "pending_risk_assessment",
        "birdFocus": {
            **focus_meta,
            "focusWeight": focus_weight,
            "viewConsistency": view_consistency,
            "originalTop1": original_rows[0],
            "focusTop1": focus_rows[0] if focus_rows else None,
        },
        # Per-frame detection boxes (best frame first). The review stage reuses
        # them to send background-removed close-ups instead of full frames.
        "frameFocus": frame_focus,
        "experimentalClassifierHead": (
            {
                "enabled": True,
                "scope": f"closed_set_{len(classes)}_species",
                "decisionRole": "baseline_only",
                "top5": top_rows(head_probabilities, classes, common_names, probability=True),
                "warning": "闭集分类头仅保留为对照基线，不参与最终候选重排。",
            }
            if head_probabilities is not None
            else {
                "enabled": False,
                "decisionRole": "disabled",
                "warning": "当前模型未加载兼容的闭集分类头。",
            }
        ),
        "globalBioCLIP": {
            "scope": (
                "avilist_unified_"
                f"{len({item['stableId'] for item in model._taxonomy_registry['species']}) + model._supplemental_registry['count']}"
                "_species"
                if merge_mode == "unified_score_topk"
                else f"avilist_mapped_{model._taxonomy_registry['matchedCount']}_model_rows"
            ),
            "taxonomyVersion": model._taxonomy_registry["version"],
            "rawTop5": raw_rows,
            "regionAdjustedTop5": adjusted_rows,
            "originalRawTop5": original_raw_rows,
            "originalTop5": original_rows,
            "focusRawTop5": focus_raw_rows,
            "focusTop5": focus_rows,
            "supplementalScope": f"avilist_missing_{model._supplemental_registry['count']}_species",
            "primaryRawTop5": primary_raw_rows,
            "primaryRegionAdjustedTop5": primary_adjusted_rows,
            "supplementalRawTop5": supplemental_raw_rows,
            "supplementalRegionAdjustedTop5": supplemental_rows,
            "reviewCandidates": review_candidates,
            "reviewTop5": review_candidates,
            "reviewTop3": review_candidates[:3],
            "openSetCandidatePool": open_set_pool,
            "supplementalCoverage": supplemental_coverage,
        },
        "imagePrototypeRetrieval": {
            **prototype_status,
            "birdPresenceGate": prototype_bird_gate,
            "decisionRole": (
                "review_candidate_expansion"
                if prototype_config.get("mode")
                in {"candidate_injection", "experimental_candidate_injection"}
                else "diagnostic_only"
            ),
            "topCandidates": prototype_rows,
            "fusion": prototype_fusion,
        },
        "singleImageFineGrained": {
            **fine_grained_settings,
            "active": fine_grained_active,
            "reason": (
                "detail_views_reranked_candidates"
                if fine_grained_active
                else "shadow_mode"
                if detail_views
                else "focus_unavailable_or_too_small"
                if detail_trigger_reason == "focus_unavailable" or should_build_details
                else "view_agreement_skipped"
                if detail_trigger_reason == "view_agreement"
                else "disabled"
            ),
            "triggerReason": detail_trigger_reason,
            "detailViews": [
                {"name": item["name"], "box": item["box"]}
                for item in detail_views
            ],
            "top5": fine_grained_rows,
        },
        "region": {
            "enabled": region is not None,
            "code": region,
            "mode": config["region"]["mode"] if region else "off",
            "maxBonus": max_bonus if region else 0.0,
        },
        "pipelineConfig": {
            "version": config["version"],
            "path": config["_path"],
            "reviewMinimumSupportingFeatures": config["review"]["minimumSupportingFeatures"],
            "reviewForceCandidate": bool(config["review"].get("forceCandidate", False)),
            "reviewForceAllowsUnknown": bool(config["review"].get("forceAllowsUnknown", False)),
            "reviewMaxCandidates": review_max_candidates,
            "identificationCards": resolve_card_settings(config),
            "knowledgeGraph": resolve_knowledge_graph_settings(config),
            "openSetEscape": open_set_config,
        },
        "location": observation,
        "geographicPrior": geo_status,
        "candidateTraits": [
            {
                "stableId": row.get("stableId"),
                "rank": index + 1,
                "scientificName": row.get("scientificName"),
                "traitText": trait,
            }
            for index, row in enumerate(review_candidates)
            if (trait := getattr(model, "_text_override_traits", {}).get(str(row.get("stableId"))))
        ],
        "latencyMs": round((time.perf_counter() - started) * 1000),
    }
    finalize_local_candidate(local_result, provisional_candidate, config)
    if len(frame_embeddings) > 1:
        local_result["frameFusion"] = {
            "mode": fusion_mode,
            "frames": len(frame_embeddings),
            "extraFrames": [str(path) for path in extra_frames if path is not None],
        }
    return local_result, original, focus


def prepare_local(image_path, runtime, region=None, use_crop=True,
                  extra_frames=(), observation=None, frame_detections=None):
    """Run the local BioCLIP chain only. The returned object is handed to
    review_prepared() later; Doubao review is the only async stage."""
    started = time.perf_counter()
    observation_kwargs = {"observation": observation} if observation is not None else {}
    local_result, original, focus = analyze_local(
        image_path, runtime, region=region, use_crop=use_crop, extra_frames=extra_frames,
        frame_detections=frame_detections, **observation_kwargs,
    )
    config = runtime[1]._pipeline_config
    event_images = None
    if config.get("multiFrame", {}).get("sendAllFramesToVlm", False) and extra_frames:
        from PIL import Image, ImageOps
        paths = [image_path, *[p for p in extra_frames if p is not None]]
        if len(paths) != 3 or len({str(Path(p).resolve()) for p in paths}) != 3:
            raise ValueError("Three-frame VLM input requires exactly three distinct frame paths")
        event_images = []
        for path in paths:
            with Image.open(path) as raw:
                event_images.append(ImageOps.exif_transpose(raw).convert("RGB"))
    return {
        "localResult": local_result,
        "original": original,
        "focus": focus,
        "eventImages": event_images,
        "eventFocusCrops": _event_frame_crops(event_images, local_result),
        "config": config,
        "started": started,
    }


def _tight_crop(image, box, minimum_side=120, pad_ratio=0.18):
    """Crop around the detection box with generous context margin.

    The box only covers what the detector fired on (often just a patch of
    plumage), so a tight crop can cut off bill, tail or head. 18% padding keeps
    the diagnostic parts in frame. Thin subjects (a bird on a twig) can be
    narrow but still tall, so the usable-size check uses the longer side. Small
    crops are upscaled to a legible square-ish size; that adds no detail but
    stops the close-up from occupying only a few visual tokens.
    """
    if image is None or not box or len(box) != 4:
        return None
    from PIL import Image

    width, height = image.size
    left, top, right, bottom = (float(value) for value in box)
    pad_x = max(2.0, (right - left) * pad_ratio)
    pad_y = max(2.0, (bottom - top) * pad_ratio)
    left = max(0, int(left - pad_x))
    top = max(0, int(top - pad_y))
    right = min(width, int(right + pad_x))
    bottom = min(height, int(bottom + pad_y))
    crop_width, crop_height = right - left, bottom - top
    if max(crop_width, crop_height) < minimum_side or min(crop_width, crop_height) < 32:
        return None
    crop = image.crop((left, top, right, bottom))
    longest = max(crop.size)
    if longest < 448:
        scale = 448 / longest
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    return crop


EVENT_CROP_MIN_SCORE = 0.35


def detect_frame_scores(paths, runtime, include_focus=False):
    """Detect the bird in every frame of one event.

    Returns per-frame ``{index, name, box, score, area}``. Used to pick the main
    frame (a detector hit beats a merely sharp frame) and to judge whether a
    per-frame crop is trustworthy enough to send to the review model.

    With ``include_focus=True`` the focus crop and its full metadata are also
    returned so the local stage can reuse the detection instead of running
    YOLO a second time on the same frame.
    """
    from PIL import Image, ImageOps

    focus_net = runtime[-1]
    results = []
    for index, path in enumerate(paths):
        box = None
        score = None
        focus = None
        focus_meta = None
        try:
            with Image.open(path) as image:
                original = ImageOps.exif_transpose(image).convert("RGB")
            with _MODEL_LOCK:
                focus, focus_meta = create_focus_view(original, focus_net)
            box = (focus_meta or {}).get("box")
            score = (focus_meta or {}).get("score")
        except Exception:  # noqa: BLE001 - a bad frame must not kill the request
            box, score, focus, focus_meta = None, None, None, None
        area = 0.0
        if box:
            area = max(0.0, float(box[2]) - float(box[0])) * max(
                0.0, float(box[3]) - float(box[1])
            )
        item = {
            "index": index,
            "name": Path(path).name,
            "box": box,
            "score": float(score or 0.0),
            "area": area,
        }
        if include_focus:
            item["focus"] = focus
            item["focusMeta"] = focus_meta
        results.append(item)
    return results


def _event_frame_crops(event_images, local_result, limit=2):
    """Close-ups from different frames of one event.

    Reuses the per-frame detection boxes already produced by the local pass
    (`frameFocus`), so no extra inference runs here. Frames without a usable
    box, or with a low-confidence detector hit, are skipped: a box that only
    covers a patch of plumage would hide the head/bill and mislead the review.
    """
    if not event_images:
        return []
    boxes = (local_result or {}).get("frameFocus") or []
    crops = []
    for image, meta in zip(event_images, boxes):
        if len(crops) >= limit:
            break
        meta = meta or {}
        if float(meta.get("score") or 0) < EVENT_CROP_MIN_SCORE:
            continue
        crop = _tight_crop(image, meta.get("box"))
        if crop is not None:
            crops.append(crop)
    return crops


def review_prepared(prepared, runtime):
    """Run exactly one Doubao forced review on a prepared local result."""
    local_result = prepared["localResult"]
    config = prepared["config"]
    started = prepared["started"]
    event_kwargs = {"event_images": prepared["eventImages"]} if prepared.get("eventImages") else {}
    if event_kwargs and prepared.get("eventFocusCrops"):
        event_kwargs["event_focus_crops"] = prepared["eventFocusCrops"]
    review_started = time.perf_counter()
    try:
        report, doubao_meta = create_boss_report(
            prepared["original"], prepared["focus"], local_result,
            local_result["qualityScore"], review_trigger="forced", **event_kwargs,
        )
        doubao_meta["status"] = "ok"
        decision_source = (
            "doubao_open_set_escape_review"
            if (doubao_meta.get("openSetEscape") or {}).get("approved")
            else "doubao_forced_review"
        )
        report, doubao_meta, guard_status = apply_visual_consensus_guard(
            report, doubao_meta, local_result, config,
        )
        doubao_meta["visualConsensusGuard"] = guard_status
        if guard_status.get("applied"):
            decision_source = "visual_consensus_guard_after_forced_review"
    except Exception as error:  # noqa: BLE001
        report = uncertain_report(local_result, "api_failure")
        doubao_meta = {
            "status": "error",
            "reviewTrigger": "forced",
            "errorType": type(error).__name__,
            "error": str(error),
            "latencyMs": round((time.perf_counter() - review_started) * 1000),
        }
        decision_source = "uncertain_after_forced_review_failure"

    output = finalize_decision(report, doubao_meta, local_result, decision_source)
    output["meta"].update({
        "reviewMode": "on",
        "reviewTrigger": "forced",
        "totalLatencyMs": round((time.perf_counter() - started) * 1000),
    })
    return output


def finalize_decision(report, doubao_meta, local_result, decision_source):
    """决策唯一收口：classify 与 review_prepared 共用，禁止再开第三份。

    presence_status 五分支语义为冻结契约，tests/test_golden_decision.py 锁定。
    """
    if decision_source == "local_direct":
        presence_status = "bird_confirmed"
    elif doubao_meta.get("status") == "ok" and doubao_meta.get("review", {}).get("decision") == "no_bird":
        presence_status = "non_bird_confirmed"
    elif doubao_meta.get("status") == "ok" and report.get("bird") is True:
        presence_status = "bird_confirmed"
    else:
        presence_status = {
            "bird": "bird_hint_unconfirmed",
            "non_bird": "non_bird_hint_unconfirmed",
        }.get(local_result["sceneRouter"]["route"], "uncertain")
    return {
        "report": report,
        "meta": {
            "decisionSource": decision_source,
            "presenceStatus": presence_status,
            "local": local_result,
            "doubao": doubao_meta,
        },
    }


def classify(image_path, runtime, region=None, use_crop=True, review_mode=None,
             extra_frames=(), observation=None, frame_detections=None):
    started = time.perf_counter()
    observation_kwargs = {"observation": observation} if observation is not None else {}
    local_result, original, focus = analyze_local(
        image_path, runtime, region=region, use_crop=use_crop, extra_frames=extra_frames,
        frame_detections=frame_detections, **observation_kwargs,
    )
    config = runtime[1]._pipeline_config
    event_kwargs = {}
    if config.get("multiFrame", {}).get("sendAllFramesToVlm", False) and extra_frames:
        from PIL import Image, ImageOps
        paths = [image_path, *[p for p in extra_frames if p is not None]]
        if len(paths) != 3 or len({str(Path(p).resolve()) for p in paths}) != 3:
            raise ValueError("Three-frame VLM input requires exactly three distinct frame paths")
        images = []
        for path in paths:
            with Image.open(path) as raw:
                images.append(ImageOps.exif_transpose(raw).convert("RGB"))
        event_kwargs["event_images"] = images
        crops = _event_frame_crops(images, local_result)
        if crops:
            event_kwargs["event_focus_crops"] = crops
    review_mode = review_mode or config["review"]["defaultMode"]
    if review_mode not in {"auto", "on", "off"}:
        raise ValueError("review_mode 必须是 auto、on 或 off。")
    needs_review = local_result["risk"]["needsReview"]
    should_review = review_mode == "on" or (review_mode == "auto" and needs_review)
    review_trigger = "forced" if review_mode == "on" else "risk" if should_review else None

    if should_review:
        review_started = time.perf_counter()
        try:
            report, doubao_meta = create_boss_report(
                original,
                focus,
                local_result,
                local_result["qualityScore"],
                review_trigger=review_trigger,
                **event_kwargs,
            )
            doubao_meta["status"] = "ok"
            decision_source = (
                "doubao_open_set_escape_review"
                if (doubao_meta.get("openSetEscape") or {}).get("approved")
                else f"doubao_{review_trigger}_review"
            )
            report, doubao_meta, guard_status = apply_visual_consensus_guard(
                report,
                doubao_meta,
                local_result,
                config,
            )
            doubao_meta["visualConsensusGuard"] = guard_status
            if guard_status.get("applied"):
                decision_source = f"visual_consensus_guard_after_{review_trigger}_review"
        except Exception as error:
            report = uncertain_report(local_result, "api_failure")
            doubao_meta = {
                "status": "error",
                "reviewTrigger": review_trigger,
                "errorType": type(error).__name__,
                "error": str(error),
                "latencyMs": round((time.perf_counter() - review_started) * 1000),
            }
            decision_source = f"uncertain_after_{review_trigger}_review_failure"
    elif needs_review:
        reason = local_result["risk"]["blockingReasons"][0]
        report = uncertain_report(local_result, reason)
        doubao_meta = {"status": "skipped", "reason": "review_disabled", "latencyMs": 0}
        decision_source = "local_risk_refusal"
    else:
        report = local_identified_report(local_result)
        doubao_meta = {"status": "skipped", "reason": "local_policy_passed", "latencyMs": 0}
        decision_source = "local_direct"

    output = finalize_decision(report, doubao_meta, local_result, decision_source)
    output["meta"].update({
        "reviewMode": review_mode,
        "reviewTrigger": review_trigger,
        "totalLatencyMs": round((time.perf_counter() - started) * 1000),
    })
    return output


def print_output(output):
    meta = output["meta"]
    print(
        f"[meta] local={meta['local']['latencyMs']}ms, "
        f"doubao={meta['doubao'].get('latencyMs', 0)}ms, "
        f"decision={meta['decisionSource']}, "
        f"review={meta['reviewMode']}",
        file=sys.stderr,
    )
    print(json.dumps(output["report"], ensure_ascii=False, indent=2))


def append_record(path, output):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="喂鸟器图像识别实验 CLI")
    parser.add_argument("image", nargs="?", help="图片文件地址；省略后进入连续输入模式")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="流程配置 JSON")
    parser.add_argument("--region", help="地域代码；OFF 关闭。非法代码会明确报错")
    parser.add_argument("--no-crop", action="store_true", help="关闭聚焦裁剪，保留检测框")
    parser.add_argument("--review", choices=("auto", "on", "off"), help="豆包复核策略")
    parser.add_argument(
        "--frames",
        nargs="*",
        default=[],
        help="同一事件的其他帧；与主图逐维最大值融合，其他字段仍以主图为准",
    )
    parser.add_argument("--record-jsonl", help="追加保存完整实验记录")
    args = parser.parse_args()

    print("正在加载 BioCLIP 2、YOLO 和实验分类头……", flush=True)
    runtime = load_runtime(args.config)
    config = runtime[1]._pipeline_config
    region = args.region if args.region is not None else config["region"]["default"]
    region = validate_region(region, runtime[1]._region_priors)
    review_mode = args.review or config["review"]["defaultMode"]
    if args.image:
        image_path = clean_path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        output = classify(
            image_path,
            runtime,
            region=region,
            use_crop=not args.no_crop,
            review_mode=review_mode,
            extra_frames=[clean_path(path) for path in args.frames],
        )
        append_record(args.record_jsonl, output)
        print_output(output)
        return

    print("模型已就绪。可连续输入图片；直接回车或输入 q 退出。")
    while True:
        raw_path = input("\n请输入鸟类图片文件地址：\n> ").strip()
        if not raw_path or raw_path.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            return
        image_path = clean_path(raw_path)
        if not image_path.is_file():
            print(f"图片不存在：{image_path}", file=sys.stderr)
            continue
        try:
            output = classify(image_path, runtime, region=region, use_crop=not args.no_crop, review_mode=review_mode)
            append_record(args.record_jsonl, output)
            print_output(output)
        except Exception as error:
            print(f"识别失败：{error}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"执行失败：{error}", file=sys.stderr)
        raise SystemExit(1)
