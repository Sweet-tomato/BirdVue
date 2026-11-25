import json
from functools import lru_cache
from pathlib import Path


FEATURE_ORDER = (
    "bill",
    "eye_ring",
    "crest",
    "face_head_pattern",
    "wing_marks",
    "tail_shape",
)


def resolve_knowledge_graph_settings(config):
    settings = dict(config.get("knowledgeGraph") or {"mode": "off"})
    settings["mode"] = str(settings.get("mode", "off"))
    config_root = Path(config["_path"]).parent if config.get("_path") else Path.cwd()
    for key in ("edgesFile", "runtimeOverridesFile"):
        value = settings.get(key)
        if value:
            path = Path(value)
            settings[key] = str(path if path.is_absolute() else config_root / path)
    return settings


@lru_cache(maxsize=4)
def _load_runtime_edges(path_value, modified_ns, size):
    del modified_ns, size
    latest = {}
    with Path(path_value).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            edge_id = str(row.get("edgeId") or "").strip()
            if not edge_id:
                raise ValueError(f"knowledge graph row {line_number} has no edgeId")
            latest[edge_id] = row
    return tuple(
        row for row in latest.values()
        if row.get("humanApproved") is True and row.get("runtimeEligible") is True
    )


def load_runtime_edges(path_value):
    path = Path(path_value)
    stat = path.stat()
    return _load_runtime_edges(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_runtime_overrides(path_value, modified_ns, size):
    del modified_ns, size
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "bird-knowledge-runtime-overrides-v1":
        raise ValueError("unsupported knowledge graph runtime override schema")
    overrides = {}
    for edge_id, row in (payload.get("overrides") or {}).items():
        if not isinstance(row, dict) or row.get("manualApproved") is not True:
            continue
        differences = []
        for difference in row.get("acceptedDifferences") or []:
            feature = str(difference.get("feature") or "")
            species_a = str(difference.get("speciesA") or "").strip()
            species_b = str(difference.get("speciesB") or "").strip()
            if feature not in FEATURE_ORDER or not species_a or not species_b:
                raise ValueError(f"invalid runtime override difference for {edge_id}")
            differences.append({
                "feature": feature,
                "speciesA": species_a,
                "speciesB": species_b,
                "visualStatus": "human_approved_runtime_override",
                "diagnosticStrength": str(
                    difference.get("diagnosticStrength") or "high"
                ),
                "decisionRule": str(difference.get("decisionRule") or ""),
            })
        if differences:
            overrides[str(edge_id)] = {
                "acceptedDifferences": differences,
                "reason": str(row.get("reason") or "manual runtime correction"),
            }
    return overrides


def load_runtime_overrides(path_value):
    if not path_value:
        return {}
    path = Path(path_value)
    stat = path.stat()
    return _load_runtime_overrides(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size
    )


def _compact_species(row, candidate_ranks):
    stable_id = str(row.get("stableId") or "")
    return {
        "stableId": stable_id,
        "scientificName": str(row.get("scientificName") or ""),
        "commonName": str(row.get("commonName") or ""),
        "candidateRank": candidate_ranks.get(stable_id),
        "outputAllowed": stable_id in candidate_ranks,
    }


def _compact_edge(edge, candidate_ranks, maximum_differences, override=None):
    accepted = set((edge.get("judge") or {}).get("acceptedFeatures") or [])
    visual_status = {
        row.get("feature"): row.get("status")
        for row in (edge.get("visualReview") or {}).get("checks") or []
        if row.get("feature")
    }
    by_feature = {
        row.get("feature"): row
        for row in (edge.get("draftCard") or {}).get("differences") or []
        if row.get("feature") in accepted
    }
    if override:
        differences = override["acceptedDifferences"][:maximum_differences]
    else:
        differences = []
        for feature in FEATURE_ORDER:
            row = by_feature.get(feature)
            if not row or visual_status.get(feature) != "consistent":
                continue
            differences.append({
                "feature": feature,
                "speciesA": str(row.get("speciesA") or ""),
                "speciesB": str(row.get("speciesB") or ""),
                "visualStatus": visual_status.get(feature, "not_recorded"),
            })
            if len(differences) >= maximum_differences:
                break
    species_a = _compact_species(edge.get("speciesA") or {}, candidate_ranks)
    species_b = _compact_species(edge.get("speciesB") or {}, candidate_ranks)
    candidate_endpoint_count = int(species_a["outputAllowed"]) + int(species_b["outputAllowed"])
    return {
        "edgeId": str(edge.get("edgeId") or ""),
        "relation": "human_approved_visual_comparison",
        "candidateEndpointCount": candidate_endpoint_count,
        "speciesA": species_a,
        "speciesB": species_b,
        "acceptedDifferences": differences,
        "runtimeOverrideApplied": bool(override),
    }


def build_knowledge_graph_context(candidate_slots, settings):
    mode = str((settings or {}).get("mode", "off"))
    if mode == "off":
        return None, {"applied": False, "reason": "disabled"}
    if mode != "approved_one_hop":
        return None, {"applied": False, "reason": "unsupported_mode", "mode": mode}
    path_value = str((settings or {}).get("edgesFile") or "").strip()
    if not path_value or not Path(path_value).is_file():
        return None, {"applied": False, "reason": "edges_file_missing", "mode": mode}

    candidate_ranks = {
        str(row.get("stableId") or ""): int(row.get("rank") or index)
        for index, row in enumerate(candidate_slots or [], 1)
        if row.get("stableId")
    }
    try:
        edges = load_runtime_edges(path_value)
        override_path = str(
            (settings or {}).get("runtimeOverridesFile") or ""
        ).strip()
        overrides = load_runtime_overrides(override_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return None, {
            "applied": False,
            "reason": "invalid_edges_file",
            "mode": mode,
            "errorType": type(error).__name__,
        }

    maximum_edges = max(1, min(20, int(settings.get("maxEdges", 8))))
    maximum_differences = max(1, min(6, int(settings.get("maxDifferencesPerEdge", 4))))
    maximum_chars = max(400, min(10000, int(settings.get("maxTotalChars", 3000))))
    relevant = []
    skipped_without_consistent_differences = 0
    for edge in edges:
        endpoint_ids = {
            str((edge.get("speciesA") or {}).get("stableId") or ""),
            str((edge.get("speciesB") or {}).get("stableId") or ""),
        }
        matched = endpoint_ids & set(candidate_ranks)
        if not matched:
            continue
        compact = _compact_edge(
            edge,
            candidate_ranks,
            maximum_differences,
            overrides.get(str(edge.get("edgeId") or "")),
        )
        if not compact["acceptedDifferences"]:
            skipped_without_consistent_differences += 1
            continue
        relevant.append(compact)
    relevant.sort(key=lambda row: (
        -int(row.get("runtimeOverrideApplied", False)),
        -row["candidateEndpointCount"],
        -len(row["acceptedDifferences"]),
        sum(
            species["candidateRank"] or 999
            for species in (row["speciesA"], row["speciesB"])
        ),
        row["edgeId"],
    ))

    selected = []
    remaining = list(relevant)

    def endpoint_ids(row):
        return {
            species["stableId"]
            for species in (row["speciesA"], row["speciesB"])
            if species["outputAllowed"]
        }

    def fits(row):
        candidate_payload = {
            "schemaVersion": "bird-knowledge-one-hop.v1",
            "policy": "reference_only_top5_closed_shortlist",
            "relations": selected + [row],
        }
        serialized = json.dumps(candidate_payload, ensure_ascii=False, separators=(",", ":"))
        return len(serialized) <= maximum_chars

    uncovered = set(candidate_ranks)
    while uncovered and remaining and len(selected) < maximum_edges:
        ordered = sorted(remaining, key=lambda row: (
            -int(row.get("runtimeOverrideApplied", False)),
            -len(endpoint_ids(row) & uncovered),
            -row["candidateEndpointCount"],
            -len(row["acceptedDifferences"]),
            sum(
                species["candidateRank"] or 999
                for species in (row["speciesA"], row["speciesB"])
            ),
            row["edgeId"],
        ))
        chosen = next(
            (row for row in ordered if endpoint_ids(row) & uncovered and fits(row)),
            None,
        )
        if chosen is None:
            break
        selected.append(chosen)
        remaining.remove(chosen)
        uncovered -= endpoint_ids(chosen)

    for row in remaining:
        if len(selected) >= maximum_edges:
            break
        if fits(row):
            selected.append(row)
    if not selected:
        return None, {
            "applied": False,
            "reason": "no_matching_approved_edges",
            "mode": mode,
            "availableRuntimeEdges": len(edges),
            "skippedWithoutConsistentDifferences": skipped_without_consistent_differences,
        }

    payload = {
        "schemaVersion": "bird-knowledge-one-hop.v1",
        "policy": "reference_only_top5_closed_shortlist",
        "relations": selected,
    }
    prompt_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    matched_ranks = sorted({
        species["candidateRank"]
        for row in selected
        for species in (row["speciesA"], row["speciesB"])
        if species["candidateRank"] is not None
    })
    outside_neighbors = {
        species["stableId"]
        for row in selected
        for species in (row["speciesA"], row["speciesB"])
        if not species["outputAllowed"]
    }
    return payload, {
        "applied": True,
        "mode": mode,
        "edgeCount": len(selected),
        "matchedCandidateRanks": matched_ranks,
        "outsideShortlistNeighborCount": len(outside_neighbors),
        "availableRuntimeEdges": len(edges),
        "skippedWithoutConsistentDifferences": skipped_without_consistent_differences,
        "promptChars": prompt_chars,
        "source": str(Path(path_value).resolve()),
        "runtimeOverrideCount": sum(
            bool(row.get("runtimeOverrideApplied")) for row in selected
        ),
        "runtimeOverridesSource": (
            str(Path(override_path).resolve()) if override_path else None
        ),
    }


def knowledge_graph_hint_section(candidate_slots, settings):
    payload, status = build_knowledge_graph_context(candidate_slots, settings)
    if payload is None:
        return "", status
    section = (
        "Human-approved one-hop comparison graph for the supplied Top-5 candidates "
        "(reference guidance, not image evidence):\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\nUse these relations only to decide which visible contrasts to inspect. "
        "A connected species with outputAllowed=false is context only and must never be returned. "
        "Do not select a candidate merely because it has a graph edge. Ignore any graph feature that "
        "is not directly observable in the current image. The allowed output shortlist remains unchanged.\n"
    )
    return section, status
