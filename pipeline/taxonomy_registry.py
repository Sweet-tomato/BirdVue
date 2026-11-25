import json
from pathlib import Path

import numpy as np


def load_taxonomy_registry(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_index = {int(item["index"]): item for item in payload["species"]}
    valid_indices = np.asarray(sorted(by_index), dtype=np.int64)
    return payload, by_index, valid_indices


def validate_taxonomy_alignment(model_labels, embeddings, by_index):
    if embeddings.ndim != 2 or embeddings.shape[0] != len(model_labels):
        raise ValueError(
            "BioCLIP taxonomy/embedding row mismatch: "
            f"labels={len(model_labels)}, embeddings={embeddings.shape}."
        )
    for model_index, item in by_index.items():
        if not 0 <= model_index < len(model_labels):
            raise ValueError(f"AviList registry index out of range: {model_index}.")
        expected = " ".join(str(model_labels[model_index]).split()).casefold()
        actual = " ".join(str(item.get("modelScientificName") or "").split()).casefold()
        if actual != expected:
            raise ValueError(
                "AviList registry/model row mismatch at index "
                f"{model_index}: registry={item.get('modelScientificName')!r}, "
                f"model={model_labels[model_index]!r}."
            )


def load_region_priors(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    regions = payload.get("regions")
    if not isinstance(regions, dict):
        raise ValueError("region_priors.json 缺少 regions 对象。")
    return payload


def validate_region(region, priors):
    if region in (None, "", "OFF"):
        return None
    normalized = str(region).strip().upper()
    if normalized not in priors["regions"]:
        allowed = ", ".join(sorted(priors["regions"])) or "无"
        raise ValueError(f"非法 region：{region}。当前可用：{allowed}；关闭地域辅助请使用 OFF。")
    return normalized


def candidate_rows(scores, by_index, valid_indices, top_k=5, region=None, region_priors=None, max_bonus=0.0, geographic_bonuses=None):
    geographic_bonuses = {k: v for k, v in (geographic_bonuses or {}).items() if v != 0}
    scores = np.asarray(scores, dtype=np.float32)
    bonuses = {}
    if region and region_priors:
        bonuses = region_priors["regions"][region].get("species", {})

    adjusted = scores[valid_indices].copy()
    if bonuses:
        for position, model_index in enumerate(valid_indices):
            stable_id = by_index[int(model_index)]["stableId"]
            adjusted[position] += max_bonus * float(bonuses.get(stable_id, 0.0))

    if geographic_bonuses:
        for position, model_index in enumerate(valid_indices):
            adjusted[position] += geographic_bonuses.get(by_index[int(model_index)]["stableId"], 0.0)
    order = np.argsort(adjusted)[::-1]
    rows = []
    seen = set()
    for position in order:
        model_index = int(valid_indices[int(position)])
        item = by_index[model_index]
        stable_id = item["stableId"]
        if stable_id in seen:
            continue
        seen.add(stable_id)
        raw_score = float(scores[model_index])
        bonus = max_bonus * float(bonuses.get(stable_id, 0.0)) if bonuses else 0.0
        rows.append({
            "rank": len(rows) + 1,
            "stableId": stable_id,
            "scientificName": item["scientificName"],
            "commonName": item["commonName"],
            "order": item["order"],
            "family": item["family"],
            "familyEnglishName": item["familyEnglishName"],
            "genus": item["genus"],
            "modelScientificName": item["modelScientificName"],
            "taxonomyMatchMethod": item["matchMethod"],
            "similarity": round(raw_score, 6),
            "regionBonus": round(bonus, 6),
            "adjustedSimilarity": round(raw_score + bonus + (geographic_bonuses or {}).get(stable_id, 0.0), 6),
            **({"geographicBonus": round(geographic_bonuses.get(stable_id, 0.0), 6)} if geographic_bonuses else {}),
        })
        if len(rows) >= top_k:
            break
    return rows


def model_scientific_name(taxonomy):
    genus = str(taxonomy[5] or "").strip()
    species = str(taxonomy[6] or "").strip()
    if species.casefold().startswith(genus.casefold() + " "):
        return " ".join(species.split())
    return " ".join(f"{genus} {species}".split())
