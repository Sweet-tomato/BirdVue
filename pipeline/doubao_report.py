import base64
import copy
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from species_identification_cards import build_reference_hints
from knowledge_graph_runtime import (
    build_knowledge_graph_context,
    knowledge_graph_hint_section,
)


ENV_SOURCE = Path(os.environ.get("BIRD_ENV_FILE", "/nonexistent"))
DEFAULT_MODEL = "doubao-seed-2-0-lite-260428"

REQUIRED_FIELDS = [
    "qualityScore", "canIdentify", "describe", "categories", "alias", "name",
    "gender", "age", "plumage", "healthStatus", "healthIssues",
    "healthDescription", "rarity", "rarityReason", "colorTags", "patternTags",
    "shapeTags", "behaviorTags", "flock", "bird", "person", "squirrel",
    "raccoon", "rattus", "snake", "opossum", "animal", "confidence",
]

TAG_GUIDE = """Tag codes:
Color: 2000 red, 2001 blue, 2002 yellow, 2003 green, 2004 orange, 2005 purple,
2006 black, 2007 white, 2008 gray, 2009 brown, 2010 beige, 2011 multicolor,
2012 iridescent, 2013 light, 2014 dark, 2015 bright, 2016 muted,
2017 high contrast, 2018 color patches.
Pattern: 5000 striped, 5001 fine striped, 5002 vertical striped, 5003 horizontal striped,
5004 spotted, 5005 speckled, 5006 mottled, 5007 color blocks, 5008 wing bars,
5009 tail bands, 5010 mask, 5011 eye stripe, 5012 eyebrow stripe,
5013 dark face, 5014 pale face, 5015 plain breast, 5016 spotted breast,
5017 streaked breast, 5018 gradient, 5019 two-tone, 5020 solid, 5021 edged feathers.
Shape: 3000 crest, 3001 tuft, 3002 flat head, 3003 large head, 3004 thick bill,
3005 thin bill, 3006 long bill, 3007 short bill, 3008 curved bill, 3009 hooked bill,
3010 needle bill, 3011 long tail, 3012 short tail, 3013 forked tail,
3014 fan tail, 3015 rounded tail, 3016 pointed tail, 3017 pointed wings,
3018 rounded wings, 3019 broad wings, 3020 narrow wings, 3021 small bird,
3022 medium bird, 3023 large bird, 3024 slender, 3025 rounded,
3026 long legs, 3027 short legs, 3028 perching, 3029 climbing.
Behavior: 4000 hovering, 4001 gliding, 4002 circling, 4003 ground feeding,
4004 tree feeding, 4005 nectar feeding, 4006 hunting, 4007 pecking,
4008 hopping, 4009 running, 4010 diving, 4011 solitary, 4012 flocking,
4013 calling, 4014 singing."""


def boss_schema(quality_score):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": REQUIRED_FIELDS,
        "properties": {
            "qualityScore": {"type": "integer", "enum": [int(quality_score)]},
            "canIdentify": {"type": "boolean"},
            "describe": {"type": "string"},
            "categories": {"type": "string"},
            "alias": {"type": "string"},
            "name": {"type": "string"},
            "gender": {"type": "string", "enum": ["Male", "Female", "Unknown"]},
            "age": {"type": "string", "enum": ["Adult", "Juvenile", "Chick", "Unknown"]},
            "plumage": {"type": "string", "enum": ["Breeding", "Non-breeding", "Molting", "Unknown"]},
            "healthStatus": {"type": "integer", "enum": [8000, 8001, 8002, 8003]},
            "healthIssues": {
                "type": "array", "maxItems": 6,
                "items": {"type": "integer", "enum": [8010, 8011, 8012, 8013, 8014, 8015]},
            },
            "healthDescription": {"type": "string"},
            "rarity": {"type": "string", "enum": ["Common", "Uncommon", "Rare", "Vagrant", "Migratory", "Unknown"]},
            "rarityReason": {"type": "string"},
            "colorTags": {"type": "array", "maxItems": 6, "items": {"type": "integer", "minimum": 2000, "maximum": 2018}},
            "patternTags": {"type": "array", "maxItems": 6, "items": {"type": "integer", "minimum": 5000, "maximum": 5021}},
            "shapeTags": {"type": "array", "maxItems": 8, "items": {"type": "integer", "minimum": 3000, "maximum": 3029}},
            "behaviorTags": {"type": "array", "maxItems": 5, "items": {"type": "integer", "minimum": 4000, "maximum": 4014}},
            "flock": {"type": "boolean"},
            "bird": {"type": "boolean"},
            "person": {"type": "boolean"},
            "squirrel": {"type": "boolean"},
            "raccoon": {"type": "boolean"},
            "rattus": {"type": "boolean"},
            "snake": {"type": "boolean"},
            "opossum": {"type": "boolean"},
            "animal": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def review_schema(
    quality_score,
    max_candidates=3,
    allow_open_set_proposals=False,
    outside_scientific_names=None,
    force_candidate=False,
    force_allows_unknown=False,
):
    if not 1 <= int(max_candidates) <= 5:
        raise ValueError("max_candidates must be between 1 and 5.")
    feature_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["feature", "status", "diagnosticity", "distinguishesFromCandidateRanks", "detail"],
        "properties": {
            "feature": {"type": "string", "enum": ["bill", "eye_ring", "crest", "face_head_pattern", "wing_marks", "tail_shape", "body_color", "eye_skin"]},
            "status": {"type": "string", "enum": ["supports", "conflicts", "unobservable"]},
            "diagnosticity": {"type": "string", "enum": ["species_specific", "shared_or_generic", "unobservable"]},
            "distinguishesFromCandidateRanks": {
                "type": "array",
                "maxItems": max_candidates - 1,
                "items": {"type": "integer", "minimum": 1, "maximum": max_candidates},
            },
            "detail": {"type": "string"},
        },
    }
    review_required = ["decision", "candidateRank", "observedFeatures", "notes"]
    review_properties = {
        "decision": {
            "type": "string",
            "enum": (
                ["candidate", "reject_all", "no_bird"]
                if force_candidate and force_allows_unknown
                else ["candidate", "no_bird"]
                if force_candidate
                else ["candidate", "genus", "family", "reject_all", "no_bird"]
            ),
        },
        "candidateRank": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": max_candidates,
        },
        "observedFeatures": {"type": "array", "minItems": 8, "maxItems": 8, "items": feature_schema},
        "notes": {"type": "string"},
    }
    if allow_open_set_proposals:
        scientific_name_schema = {"type": "string"}
        if outside_scientific_names:
            scientific_name_schema["enum"] = list(outside_scientific_names)
        review_required.append("outsideCandidates")
        review_properties["outsideCandidates"] = {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "scientificName",
                    "commonName",
                    "observedFeatureEvidence",
                    "contradictedCandidateRanks",
                ],
                "properties": {
                    "scientificName": scientific_name_schema,
                    "commonName": {"type": "string"},
                    "observedFeatureEvidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": {"type": "string"},
                    },
                    "contradictedCandidateRanks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_candidates,
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_candidates,
                        },
                    },
                },
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["report", "review"],
        "properties": {
            "report": boss_schema(quality_score),
            "review": {
                "type": "object",
                "additionalProperties": False,
                "required": review_required,
                "properties": review_properties,
            },
        },
    }


def parse_env_file(path):
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = re.match(r"^\s*([^#=\s]+)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(2).strip().strip('"').strip("'")
        markdown_link = re.match(r"^\[[^\]]+\]\((https?://[^)]+)\)$", value)
        values[match.group(1)] = markdown_link.group(1) if markdown_link else value
    return values


def load_config():
    file_values = parse_env_file(ENV_SOURCE)
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY1") or file_values.get("OPENAI_API_KEY1") or file_values.get("OPENAI_API_KEY4") or file_values.get("ARK_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL1") or file_values.get("OPENAI_BASE_URL1") or file_values.get("OPENAI_BASE_URL4")
    model = os.environ.get("OPENAI_VLM_MODEL1") or file_values.get("OPENAI_VLM_MODEL1") or DEFAULT_MODEL
    if not api_key or not base_url:
        raise RuntimeError("Doubao configuration is incomplete. Check the linked .env file.")
    return api_key, base_url, model


def responses_endpoint(base_url):
    value = str(base_url).strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    if value.endswith("/chat/completions"):
        value = value[:-17]
    return value if value.endswith("/responses") else value + "/responses"


def image_data_url(image, max_side, quality):
    resized = image.copy()
    resized.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def build_head_detail_board(image):
    if image is None:
        return None
    width, height = image.size
    if width < 64 or height < 64:
        return None
    crop_width = max(48, round(width * 0.62))
    crop_height = max(48, round(height * 0.62))
    boxes = [
        (0, 0, crop_width, crop_height),
        (width - crop_width, 0, width, crop_height),
        (0, height - crop_height, crop_width, height),
        (width - crop_width, height - crop_height, width, height),
    ]
    tile_size = 384
    gutter = 8
    board_size = tile_size * 2 + gutter
    board = Image.new("RGB", (board_size, board_size), (34, 39, 36))
    for index, box in enumerate(boxes):
        tile = image.crop(box).convert("RGB")
        tile.thumbnail(
            (tile_size, tile_size),
            Image.Resampling.LANCZOS,
        )
        column = index % 2
        row = index // 2
        x = column * (tile_size + gutter) + (tile_size - tile.width) // 2
        y = row * (tile_size + gutter) + (tile_size - tile.height) // 2
        board.paste(tile, (x, y))
    return board


def response_text(payload):
    if isinstance(payload.get("output_text"), str) and payload["output_text"]:
        return payload["output_text"]
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return content["text"]
    raise RuntimeError("Doubao returned no text output.")


def neutral_candidate_slots(local_result):
    from bird_cli import select_review_candidates

    max_candidates = int(
        local_result.get("pipelineConfig", {}).get("reviewMaxCandidates", 3)
    )
    if not 1 <= max_candidates <= 5:
        raise ValueError("reviewMaxCandidates must be between 1 and 5.")
    rows = select_review_candidates(local_result)[:max_candidates]

    def stable_hash(row):
        value = 2166136261
        for byte in row["stableId"].encode("utf-8"):
            value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
        return value

    slots = []
    for row in sorted(rows, key=lambda item: (stable_hash(item), item["stableId"])):
        slots.append({
            "rank": len(slots) + 1,
            "stableId": row["stableId"],
            "scientificName": row["scientificName"],
            "commonName": row["commonName"],
            "family": row["family"],
            "familyEnglishName": row.get("familyEnglishName", ""),
            "genus": row["genus"],
        })
    return slots


def candidate_context(local_result):
    slots = neutral_candidate_slots(local_result)
    return {
        "candidateSlots": slots,
        "candidateCount": len(slots),
        "ordering": "neutral_stable_id_hash",
        "scoresProvided": False,
    }


def open_set_escape_settings(local_result):
    settings = dict(
        local_result.get("pipelineConfig", {}).get("openSetEscape") or {}
    )
    enabled = bool(settings.get("enabled", False))
    maximum_proposals = int(settings.get("maximumProposals", 2))
    minimum_features = int(settings.get("minimumProposalFeatures", 3))
    maximum_bioclip_rank = int(settings.get("maximumBioCLIPRank", 50))
    second_pass_minimum = int(
        settings.get("secondPassMinimumSupportingFeatures", 3)
    )
    maximum_challenges = int(settings.get("maximumChallengeCandidates", 5))
    prototype_maximum_rank = int(settings.get("prototypeMaximumRank", 3))
    prototype_minimum_similarity = float(
        settings.get("prototypeMinimumSimilarity", 0.5)
    )
    prototype_minimum_support = int(settings.get("prototypeMinimumSupport", 10))
    confirmation_mode = str(
        settings.get("confirmationMode", "second_pass")
    ).strip()
    single_pass_maximum = int(
        settings.get("singlePassMaximumInjectedCandidates", 1)
    )
    minimum_graph_differences = int(settings.get("minimumGraphDifferences", 1))
    single_pass_minimum_original_contrasts = int(
        settings.get("singlePassMinimumOriginalCandidateContrasts", 0)
    )
    if not 1 <= maximum_proposals <= 2:
        raise ValueError("openSetEscape.maximumProposals must be between 1 and 2.")
    if not 1 <= minimum_features <= 6:
        raise ValueError(
            "openSetEscape.minimumProposalFeatures must be between 1 and 6."
        )
    if not 6 <= maximum_bioclip_rank <= 100:
        raise ValueError(
            "openSetEscape.maximumBioCLIPRank must be between 6 and 100."
        )
    if not 3 <= second_pass_minimum <= 6:
        raise ValueError(
            "openSetEscape.secondPassMinimumSupportingFeatures must be between 3 and 6."
        )
    if not 1 <= maximum_challenges <= 8:
        raise ValueError(
            "openSetEscape.maximumChallengeCandidates must be between 1 and 8."
        )
    if not 1 <= prototype_maximum_rank <= 5:
        raise ValueError(
            "openSetEscape.prototypeMaximumRank must be between 1 and 5."
        )
    if not -1.0 <= prototype_minimum_similarity <= 1.0:
        raise ValueError(
            "openSetEscape.prototypeMinimumSimilarity must be between -1 and 1."
        )
    if prototype_minimum_support < 1:
        raise ValueError("openSetEscape.prototypeMinimumSupport must be positive.")
    if confirmation_mode not in {"second_pass", "single_pass_shortlist"}:
        raise ValueError(
            "openSetEscape.confirmationMode must be second_pass or "
            "single_pass_shortlist."
        )
    if not 1 <= single_pass_maximum <= 2:
        raise ValueError(
            "openSetEscape.singlePassMaximumInjectedCandidates must be between 1 and 2."
        )
    if not 1 <= minimum_graph_differences <= 6:
        raise ValueError(
            "openSetEscape.minimumGraphDifferences must be between 1 and 6."
        )
    if not 0 <= single_pass_minimum_original_contrasts <= 6:
        raise ValueError(
            "openSetEscape.singlePassMinimumOriginalCandidateContrasts must be between 0 and 6."
        )
    return {
        **settings,
        "enabled": enabled,
        "maximumProposals": maximum_proposals,
        "minimumProposalFeatures": minimum_features,
        "maximumBioCLIPRank": maximum_bioclip_rank,
        "secondPassMinimumSupportingFeatures": second_pass_minimum,
        "maximumChallengeCandidates": maximum_challenges,
        "prototypeMaximumRank": prototype_maximum_rank,
        "prototypeMinimumSimilarity": prototype_minimum_similarity,
        "prototypeMinimumSupport": prototype_minimum_support,
        "confirmationMode": confirmation_mode,
        "singlePassMaximumInjectedCandidates": single_pass_maximum,
        "singlePassGraphNeighborsOnly": bool(
            settings.get("singlePassGraphNeighborsOnly", False)
        ),
        "graphNeighborRequiresBioCLIPSupport": bool(
            settings.get("graphNeighborRequiresBioCLIPSupport", False)
        ),
        "minimumGraphDifferences": minimum_graph_differences,
        "singlePassMinimumOriginalCandidateContrasts": (
            single_pass_minimum_original_contrasts
        ),
        "allowChallengeAfterCandidateDecision": bool(
            settings.get("allowChallengeAfterCandidateDecision", True)
        ),
    }


def challenge_candidate_context(local_result, existing_candidates, settings):
    maximum_challenges = int(settings.get("maximumChallengeCandidates", 5))
    maximum_bioclip_rank = int(settings.get("maximumBioCLIPRank", 50))
    minimum_graph_differences = int(settings.get("minimumGraphDifferences", 1))
    require_graph_bioclip_support = bool(
        settings.get("graphNeighborRequiresBioCLIPSupport", False)
    )
    prototype_maximum_rank = int(settings.get("prototypeMaximumRank", 3))
    prototype_minimum_similarity = float(
        settings.get("prototypeMinimumSimilarity", 0.5)
    )
    prototype_minimum_support = int(
        settings.get("prototypeMinimumSupport", 10)
    )
    existing_ids = {row.get("stableId") for row in existing_candidates}
    pool_rows = local_result.get("globalBioCLIP", {}).get(
        "openSetCandidatePool", []
    )
    pool_by_id = {row.get("stableId"): row for row in pool_rows}
    prototype_rows = local_result.get("imagePrototypeRetrieval", {}).get(
        "topCandidates", []
    )
    prototype_by_id = {row.get("stableId"): row for row in prototype_rows}
    selected = []
    by_id = {}

    def add(row, source):
        stable_id = str(row.get("stableId") or "")
        if not stable_id or stable_id in existing_ids:
            return
        if stable_id in by_id:
            if source not in by_id[stable_id]["challengeSources"]:
                by_id[stable_id]["challengeSources"].append(source)
            return
        scientific_name = " ".join(
            str(row.get("scientificName") or "").split()
        )
        if len(scientific_name.split()) != 2:
            return
        item = {
            **row,
            "stableId": stable_id,
            "scientificName": scientific_name,
            "commonName": str(row.get("commonName") or scientific_name),
            "family": str(row.get("family") or ""),
            "familyEnglishName": str(row.get("familyEnglishName") or ""),
            "genus": str(row.get("genus") or scientific_name.split()[0]),
            "challengeSources": [source],
            "candidateEvidenceRole": "challenge_only_not_final_evidence",
            "notFinalSpeciesEvidence": True,
        }
        by_id[stable_id] = item
        selected.append(item)

    graph_payload, graph_status = build_knowledge_graph_context(
        existing_candidates,
        local_result.get("pipelineConfig", {}).get("knowledgeGraph") or {},
    )
    for relation in (graph_payload or {}).get("relations", []):
        if len(relation.get("acceptedDifferences") or []) < minimum_graph_differences:
            continue
        for species in (relation.get("speciesA") or {}, relation.get("speciesB") or {}):
            if species.get("outputAllowed") is False:
                pool_row = pool_by_id.get(species.get("stableId"))
                if require_graph_bioclip_support and pool_row is None:
                    continue
                row = (
                    pool_row
                    or prototype_by_id.get(species.get("stableId"))
                    or species
                )
                add(row, "human_approved_graph_neighbor")

    for row in prototype_rows:
        if int(row.get("prototypeRank") or row.get("rank") or 999) > int(
            prototype_maximum_rank
        ):
            continue
        if float(row.get("prototypeSimilarity") or -2.0) < float(
            prototype_minimum_similarity
        ):
            continue
        if int(row.get("prototypeSupportCount") or 0) < int(
            prototype_minimum_support
        ):
            continue
        add(row, "image_prototype_top_candidate")

    for row in pool_rows:
        if int(row.get("rank") or 999) > maximum_bioclip_rank:
            continue
        add(row, "bioclip_extended_rank_pool")
        if len(selected) >= maximum_challenges:
            break

    selected = selected[:maximum_challenges]
    for rank, row in enumerate(selected, 1):
        row["challengeRank"] = rank
        pool_row = pool_by_id.get(row["stableId"])
        row["bioCLIPRank"] = pool_row.get("rank") if pool_row else None
    prompt_rows = [
        {
            "challengeRank": row["challengeRank"],
            "stableId": row["stableId"],
            "scientificName": row["scientificName"],
            "commonName": row["commonName"],
            "genus": row["genus"],
            "sources": row["challengeSources"],
        }
        for row in selected
    ]
    return selected, {
        "available": bool(selected),
        "candidateCount": len(selected),
        "candidateSlots": prompt_rows,
        "scoresProvided": False,
        "graph": graph_status,
    }


def select_open_set_candidates(review, local_result, existing_candidates, settings):
    status = {
        "enabled": bool(settings.get("enabled", False)),
        "eligible": False,
        "reason": "disabled",
        "proposals": copy.deepcopy(review.get("outsideCandidates") or []),
        "acceptedCandidates": [],
        "rejections": [],
    }
    if not status["enabled"]:
        return [], status
    if local_result.get("_openSetEscapeSecondPass"):
        status["reason"] = "already_in_second_pass"
        return [], status
    if int(local_result.get("qualityScore") or 0) < 50:
        status["reason"] = "image_quality_below_species_threshold"
        return [], status
    decision = review.get("decision")
    eligible_decisions = {"reject_all", "genus", "family"}
    if settings.get("allowChallengeAfterCandidateDecision"):
        eligible_decisions.add("candidate")
    if decision not in eligible_decisions:
        status["reason"] = "first_pass_not_uncertain"
        return [], status
    status["eligible"] = True
    challenge_rows, challenge_status = challenge_candidate_context(
        local_result, existing_candidates, settings
    )
    status["challengePool"] = challenge_status
    proposals = review.get("outsideCandidates") or []
    if not proposals:
        status["reason"] = "no_outside_candidate_proposed"
        return [], status
    existing_ids = {row.get("stableId") for row in existing_candidates}
    existing_names = {
        " ".join(str(row.get("scientificName") or "").split()).casefold()
        for row in existing_candidates
    }
    challenge_by_name = {
        " ".join(str(row.get("scientificName") or "").split()).casefold(): row
        for row in challenge_rows
    }
    accepted = []
    for proposal in proposals[: int(settings["maximumProposals"])]:
        scientific_name = " ".join(
            str(proposal.get("scientificName") or "").split()
        )
        normalized_name = scientific_name.casefold()
        reason = None
        if len(scientific_name.split()) != 2:
            reason = "not_a_binomial_scientific_name"
        elif normalized_name in existing_names:
            reason = "already_in_first_pass_candidates"
        elif normalized_name not in challenge_by_name:
            reason = "not_in_supplied_challenge_pool"
        elif len(proposal.get("observedFeatureEvidence") or []) < int(
            settings["minimumProposalFeatures"]
        ):
            reason = "insufficient_visible_feature_evidence"
        elif not proposal.get("contradictedCandidateRanks"):
            reason = "no_first_pass_candidate_contradiction"

        row = challenge_by_name.get(normalized_name)
        if reason is None and row.get("stableId") in existing_ids:
            reason = "already_in_first_pass_candidates"
        if reason is not None:
            status["rejections"].append(
                {"scientificName": scientific_name, "reason": reason}
            )
            continue
        accepted_row = {
            **row,
            "candidateSource": "vlm_open_set_proposal_bioclip_validated",
            "reviewCandidateSource": "vlm_open_set_proposal_bioclip_validated",
            "candidateEvidenceRole": "candidate_recall_only",
            "notFinalSpeciesEvidence": True,
            "openSetProposal": copy.deepcopy(proposal),
        }
        accepted.append(accepted_row)
        existing_ids.add(row["stableId"])
        existing_names.add(normalized_name)

    status["acceptedCandidates"] = [
        {
            "stableId": row["stableId"],
            "scientificName": row["scientificName"],
            "bioCLIPRank": row.get("bioCLIPRank"),
            "challengeSources": row.get("challengeSources") or [],
        }
        for row in accepted
    ]
    status["reason"] = (
        "validated_candidates_ready_for_second_pass"
        if accepted
        else "all_proposals_rejected"
    )
    return accepted, status


def build_open_set_second_pass_local_result(
    local_result, existing_candidates, accepted_candidates, settings
):
    maximum_candidates = int(
        local_result.get("pipelineConfig", {}).get("reviewMaxCandidates", 5)
    )
    keep_count = max(0, maximum_candidates - len(accepted_candidates))
    expanded = [
        *copy.deepcopy(existing_candidates[:keep_count]),
        *copy.deepcopy(accepted_candidates),
    ]
    for rank, row in enumerate(expanded, 1):
        row["rank"] = rank
    second = copy.deepcopy(local_result)
    second["_openSetEscapeSecondPass"] = True
    second["globalBioCLIP"]["reviewCandidates"] = expanded
    second["globalBioCLIP"]["reviewTop5"] = expanded
    second["globalBioCLIP"]["reviewTop3"] = expanded[:3]
    second["pipelineConfig"]["reviewMaxCandidates"] = len(expanded)
    second["pipelineConfig"]["reviewMinimumSupportingFeatures"] = int(
        settings["secondPassMinimumSupportingFeatures"]
    )
    second["pipelineConfig"]["openSetEscape"] = {
        **settings,
        "enabled": False,
    }
    return second


def _escape_keep_trait_aware(existing, keep_count, local_result):
    """Pick which existing candidates survive an escape injection.

    Displacing the tail by rank can sever contrast notes: a kept candidate's
    traitText may cite the displaced species, which then fails the review
    pool filter and empties the whole trait section. Prefer displacing
    candidates that carry no trait row and are not cited by any trait note;
    fall back to lowest rank first.
    """
    rows = list(existing)
    drop_needed = len(rows) - keep_count
    if drop_needed <= 0:
        return rows
    traits = [
        item for item in (local_result.get("candidateTraits") or [])
        if isinstance(item, dict) and str(item.get("traitText") or "").strip()
    ]
    trait_ids = {str(item.get("stableId")) for item in traits}
    cited = set()
    for item in traits:
        for clause in re.split(r"[|;]", str(item.get("traitText") or "")):
            cited |= trait_species_references(clause)

    def rank_key(row):
        return int(row.get("rank") or 999)

    free = [
        row for row in rows
        if str(row.get("stableId")) not in trait_ids
        and str(row.get("scientificName") or "").casefold() not in cited
    ]
    dropped = set()
    for row in sorted(free, key=rank_key, reverse=True):
        if len(dropped) >= drop_needed:
            break
        dropped.add(str(row.get("stableId")))
    if len(dropped) < drop_needed:
        for row in sorted(
            (r for r in rows if str(r.get("stableId")) not in dropped),
            key=rank_key,
            reverse=True,
        ):
            if len(dropped) >= drop_needed:
                break
            dropped.add(str(row.get("stableId")))
    return [row for row in rows if str(row.get("stableId")) not in dropped][:keep_count]


def build_open_set_single_pass_local_result(local_result, settings):
    from bird_cli import select_review_candidates

    maximum_candidates = int(
        local_result.get("pipelineConfig", {}).get("reviewMaxCandidates", 5)
    )
    existing = select_review_candidates(local_result)[:maximum_candidates]
    challenge_rows, challenge_status = challenge_candidate_context(
        local_result, existing, settings
    )
    if settings.get("singlePassGraphNeighborsOnly"):
        challenge_rows = [
            row
            for row in challenge_rows
            if "human_approved_graph_neighbor" in set(
                row.get("challengeSources") or []
            )
        ]

    def priority(row):
        sources = set(row.get("challengeSources") or [])
        return (
            0 if "image_prototype_top_candidate" in sources else 1,
            0 if "human_approved_graph_neighbor" in sources else 1,
            int(row.get("bioCLIPRank") or 999),
            str(row.get("stableId") or ""),
        )

    maximum_injected = min(
        int(settings["singlePassMaximumInjectedCandidates"]),
        max(0, maximum_candidates - 1),
    )
    injected = sorted(challenge_rows, key=priority)[:maximum_injected]
    keep_count = max(0, maximum_candidates - len(injected))
    expanded = [
        *copy.deepcopy(_escape_keep_trait_aware(existing, keep_count, local_result)),
        *copy.deepcopy(injected),
    ]
    for rank, row in enumerate(expanded, 1):
        row["rank"] = rank

    single = copy.deepcopy(local_result)
    single["_openSetEscapeSinglePass"] = True
    single["globalBioCLIP"]["reviewCandidates"] = expanded
    single["globalBioCLIP"]["reviewTop5"] = expanded
    single["globalBioCLIP"]["reviewTop3"] = expanded[:3]
    single["pipelineConfig"]["reviewMaxCandidates"] = len(expanded)
    single["pipelineConfig"]["openSetEscape"] = {
        **copy.deepcopy(settings),
        "enabled": False,
    }
    status = {
        "enabled": True,
        "confirmationMode": "single_pass_shortlist",
        "eligible": bool(injected),
        "reason": (
            "single_pass_candidates_injected"
            if injected
            else "no_challenge_candidate_available"
        ),
        "approved": False,
        "proposals": [],
        "acceptedCandidates": [],
        "originalCandidateStableIds": [
            row.get("stableId") for row in existing if row.get("stableId")
        ],
        "injectedCandidates": [
            {
                "stableId": row.get("stableId"),
                "scientificName": row.get("scientificName"),
                "bioCLIPRank": row.get("bioCLIPRank"),
                "challengeSources": row.get("challengeSources") or [],
            }
            for row in injected
        ],
        "challengePool": challenge_status,
    }
    return single, status


def valid_wgs84_location(location):
    from geographic_prior import valid_location
    return valid_location(location)


_TRAIT_SPECIES_REFERENCE = re.compile(r"as in ([A-Z][a-z]+ [a-z]+)")


def split_trait_clauses(text):
    """Split a contrastive trait note into ' | '-separated groups of '; '-separated clauses."""
    groups = []
    for group in str(text or "").split(" | "):
        clauses = [part.strip() for part in re.split(r"(?<=;)\s+", group.strip()) if part.strip()]
        if clauses:
            groups.append(clauses)
    return groups


def trait_species_references(clause):
    return {match.casefold() for match in _TRAIT_SPECIES_REFERENCE.findall(clause)}


def filter_trait_text(text, pool_names):
    """Trim contrast clauses that cite species outside the supplied candidate pool.

    The model may not answer with unlisted species, so naming an absent
    reference species is noise. For "X is A, not B as in <off-pool species>"
    clauses the candidate's own description ("X is A") is still useful visual
    guidance, so only the contrast tail is removed. Clauses without a salvageable
    self-description are dropped; the caller omits a slot whose note is empty.
    """
    pool = {str(name).casefold() for name in pool_names if name}
    kept_groups, kept, dropped = [], 0, 0
    for clauses in split_trait_clauses(text):
        keep = []
        for clause in clauses:
            clause = clause.strip().rstrip(";").strip()
            references = trait_species_references(clause)
            if references and not (references & pool):
                self_part = re.split(r",\s+not\s+", clause, maxsplit=1)[0].strip()
                if " is " in self_part and len(self_part) >= 20:
                    keep.append(self_part)
                    kept += 1
                else:
                    dropped += 1
                continue
            keep.append(clause)
            kept += 1
        if keep:
            kept_groups.append("; ".join(keep))
    if not kept_groups:
        return "", 0, dropped
    return " | ".join(kept_groups), kept, dropped


def trait_covered_pairs(candidate_traits, pool_names):
    """Species pairs already contrasted inside the candidate pool's trait notes."""
    pool = {str(name).casefold() for name in pool_names if name}
    covered = set()
    for item in candidate_traits or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("scientificName") or "").casefold()
        for clause in re.split(r"[|;]", str(item.get("traitText") or "")):
            for reference in trait_species_references(clause):
                if name and reference and reference != name and reference in pool:
                    covered.add(frozenset((name, reference)))
    return covered


def top_pair_structured_fields(slots):
    """True when both rank-1 and rank-2 candidates hit the curated field table."""
    if len(slots) < 2:
        return False
    by_id, _ = load_structured_fields()
    for slot in slots[:2]:
        entry = by_id.get(str(slot.get("stableId") or ""))
        fields = (entry or {}).get("fields") or {}
        if sum(1 for key in _FIELD_ORDER if fields.get(key)) < 2:
            return False
    return True


RELAXED_EVIDENCE_CONFIDENCE_CAP = 0.45


def review_evidence_policy(force_candidate, slots):
    """Pick the review evidence policy: strict only when both top-2 candidates
    hit the curated morphology field table, otherwise relaxed."""
    strict_pair_fields = top_pair_structured_fields(slots)
    relaxed = bool(force_candidate and not strict_pair_fields)
    return {
        "mode": "relaxed_no_top_pair_fields" if relaxed else "strict",
        "top1Top2StructuredFields": strict_pair_fields,
        "confidenceCap": RELAXED_EVIDENCE_CONFIDENCE_CAP if relaxed else None,
        "relaxed": relaxed,
    }


_FIELDS_ASSET_PATH = Path(
    "/root/bird_service/models/bioclip25/birds/text_overrides/common_birds_fields_v1.json"
)
_FIELDS_CACHE = None
_FIELD_ORDER = ("crest", "head_pattern", "bill", "iris", "wing_marks", "body", "tail_shape", "bare_skin")


def load_structured_fields():
    """Load the curated per-species morphology field asset (review-stage only).

    Returns (by_stable_id, status). Missing or malformed files degrade to an
    empty map so the prompt falls back to contrast-note traitText.
    """
    global _FIELDS_CACHE
    if _FIELDS_CACHE is None:
        try:
            payload = json.loads(_FIELDS_ASSET_PATH.read_text(encoding="utf-8"))
            entries = payload.get("species") or {}
            by_id = {}
            for sci, entry in entries.items():
                if isinstance(entry, dict) and entry.get("stableId"):
                    by_id[str(entry["stableId"])] = entry
            _FIELDS_CACHE = (by_id, "ok")
        except (OSError, json.JSONDecodeError) as error:
            _FIELDS_CACHE = ({}, f"error:{type(error).__name__}")
    return _FIELDS_CACHE[0], _FIELDS_CACHE[1]


def candidate_trait_section(local_result, candidates):
    traits = local_result.get("candidateTraits") or []
    if not isinstance(traits, list):
        return "", {"status": "invalid", "count": 0}
    slots = candidates["candidateSlots"]
    by_id = {row["stableId"]: row for row in slots}
    by_name = {str(row.get("scientificName")).casefold(): row for row in slots}
    pool_names = [row.get("scientificName") for row in slots]
    structured_by_id, structured_status = load_structured_fields()
    lines = []
    seen = set()
    clauses_kept = clauses_dropped = slots_omitted = structured_slots = 0
    for item in traits:
        if not isinstance(item, dict):
            continue
        slot = (by_id.get(item["stableId"]) if item.get("stableId")
                else by_name.get(str(item.get("scientificName") or "").casefold()))
        if not slot or slot["stableId"] in seen:
            continue
        seen.add(slot["stableId"])
        entry = structured_by_id.get(str(slot["stableId"]))
        field_parts = []
        if entry:
            fields = entry.get("fields") or {}
            field_parts = [f"{key}: {fields[key]}" for key in _FIELD_ORDER if fields.get(key)]
        if len(field_parts) >= 2:
            lines.append(
                f"- candidate slot {slot['rank']} ({slot.get('scientificName')}): "
                + "; ".join(field_parts)
            )
            structured_slots += 1
            continue
        text = str(item.get("traitText") or "").strip()
        if not text:
            slots_omitted += 1
            continue
        text, kept, dropped = filter_trait_text(text, pool_names)
        clauses_kept += kept
        clauses_dropped += dropped
        if not text:
            slots_omitted += 1
            continue
        lines.append(f"- candidate slot {slot['rank']}: {text}")
    status = {"status": "applied" if lines else "empty", "count": len(lines),
              "structuredSlots": structured_slots,
              "clausesKept": clauses_kept, "clausesDroppedOffPool": clauses_dropped,
              "slotsOmittedOffPool": slots_omitted}
    if not lines:
        return "", status
    if structured_slots:
        header = (
            "Curated per-species morphology fields (guidance only; never image "
            "evidence; compare the same field across candidate slots to decide "
            "which visible contrast to inspect, and treat any feature not visible "
            "in the image as unknown):\n"
        )
    else:
        header = ("Relative-difference notes derived from the text tower (guidance only; "
                  "never image evidence; use them only to decide which visible contrast to "
                  "inspect, and treat any feature not visible in the image as unknown):\n")
    return header + "\n".join(lines) + "\n", status


def reference_hint_section(local_result, candidates, prompt_mode):
    hints, status = build_reference_hints(
        candidates["candidateSlots"],
        local_result.get("pipelineConfig", {}).get("identificationCards") or {},
        prompt_mode,
    )
    if hints is None:
        return "", status
    section = (
        "Optional identification reference excerpts (not image evidence):\n"
        + json.dumps(hints, ensure_ascii=False, separators=(",", ":"))
        + "\nThese excerpts only suggest details to inspect. They can be incomplete, "
        "out of context, or unreviewed. Count support or conflict only when the "
        "corresponding feature is directly visible in this image. Missing text is "
        "never evidence that a feature is absent.\n"
    )
    return section, status


def graph_hint_section(local_result, candidates, allow_open_set_proposals=False):
    settings = local_result.get("pipelineConfig", {}).get("knowledgeGraph") or {}
    payload, status = build_knowledge_graph_context(candidates["candidateSlots"], settings)
    if payload is None:
        return "", status
    status = dict(status or {})
    pool_names = [row.get("scientificName") for row in candidates["candidateSlots"]]
    covered = trait_covered_pairs(local_result.get("candidateTraits"), pool_names)
    relations = list(payload.get("relations") or [])
    kept = [
        relation for relation in relations
        if frozenset((
            str((relation.get("speciesA") or {}).get("scientificName") or "").casefold(),
            str((relation.get("speciesB") or {}).get("scientificName") or "").casefold(),
        )) not in covered
    ]
    status["traitCoveredPairsSkipped"] = len(relations) - len(kept)
    if not kept:
        status["applied"] = False
        status["reason"] = "all_pairs_covered_by_traits" if relations else "no_relations"
        return "", status
    payload = dict(payload)
    payload["relations"] = kept
    if allow_open_set_proposals:
        section = (
            "Human-approved one-hop comparison graph for the supplied candidates "
            "(reference guidance, not image evidence):\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\nPairs already contrasted in the trait notes are omitted. Use these "
            "relations to inspect directly visible contrasts. A connected "
            "species with outputAllowed=false is not a current decision candidate, but "
            "may be named only in outsideCandidates when it also appears in the supplied "
            "challenge pool. It still requires a separate confirming review.\n"
        )
    else:
        section = (
            "Human-approved one-hop comparison graph for the supplied Top-5 candidates "
            "(reference guidance, not image evidence):\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\nPairs already contrasted in the trait notes are omitted. Use these relations only to decide "
            "which visible contrasts to inspect. "
            "A connected species with outputAllowed=false is context only and must never be returned. "
            "Do not select a candidate merely because it has a graph edge. Ignore any graph feature that "
            "is not directly observable in the current image. The allowed output shortlist remains unchanged.\n"
        )
    return section, status


def choose_prompt_mode(local_result, review_trigger=None):
    card_settings = local_result.get("pipelineConfig", {}).get("identificationCards") or {}
    if (
        review_trigger == "forced"
        and card_settings.get("forceEvidenceFullOnForcedReview")
    ):
        return "evidence_full"
    scene_route = local_result.get("sceneRouter", {}).get("route")
    if scene_route in {"non_bird", "uncertain"}:
        return "scene_only"
    if review_trigger == "forced":
        return "evidence_full"
    return "evidence_full" if local_result.get("risk", {}).get("needsReview") else "fast"


def sanitize_report(report):
    ranges = {
        "colorTags": (2000, 2018, 6),
        "patternTags": (5000, 5021, 6),
        "shapeTags": (3000, 3029, 8),
        "behaviorTags": (4000, 4014, 5),
    }
    for field, (minimum, maximum, limit) in ranges.items():
        values = report.get(field)
        if isinstance(values, list):
            report[field] = list(dict.fromkeys(
                value for value in values
                if isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum
            ))[:limit]
    issues = report.get("healthIssues")
    if isinstance(issues, list):
        report["healthIssues"] = list(dict.fromkeys(
            value for value in issues
            if isinstance(value, int) and not isinstance(value, bool) and 8010 <= value <= 8015
        ))[:6]
    if any(report.get(field) is True for field in ("bird", "squirrel", "raccoon", "rattus", "snake", "opossum")):
        report["animal"] = True
    if report.get("bird") is False:
        report["canIdentify"] = False
        report["name"] = "No Bird"
        report["categories"] = "Non-bird"
        report["alias"] = ""
        report["confidence"] = min(0.3, float(report.get("confidence") or 0))
    return report


def _visual_candidate_support(local_result, stable_id, minimum_votes=2):
    global_bioclip = (local_result or {}).get("globalBioCLIP") or {}
    enhanced_rows = global_bioclip.get("regionAdjustedTop5") or []
    focus = (local_result or {}).get("birdFocus") or {}
    branch_ids = {
        "enhanced": enhanced_rows[0].get("stableId") if enhanced_rows else None,
        "original": (focus.get("originalTop1") or {}).get("stableId"),
        "focus": (focus.get("focusTop1") or {}).get("stableId"),
    }
    available = [name for name, value in branch_ids.items() if value]
    matching = [
        name for name, value in branch_ids.items()
        if stable_id and value == stable_id
    ]
    return {
        "candidateStableId": stable_id,
        "availableBranches": available,
        "matchingBranches": matching,
        "minimumMatchingBranches": int(minimum_votes),
        "passed": len(matching) >= int(minimum_votes),
    }


def enforce_candidate_decision(
    report,
    review,
    candidates,
    quality_score,
    minimum_supporting_features=3,
    local_result=None,
    force_candidate=False,
):
    if not isinstance(review, dict):
        raise ValueError("Doubao review metadata is missing.")
    features = review.get("observedFeatures")
    expected_features = {
        "bill",
        "eye_ring",
        "crest",
        "face_head_pattern",
        "wing_marks",
        "tail_shape",
        "body_color",
        "eye_skin",
    }
    actual_features = {item.get("feature") for item in features or [] if isinstance(item, dict)}
    if (not isinstance(features, list) or len(features) != 8
            or any(not isinstance(item, dict) for item in features)
            or actual_features != expected_features):
        raise ValueError("Doubao did not evaluate all eight required visible features.")

    decision = review.get("decision")
    rank = review.get("candidateRank")
    review["originalDecision"] = decision
    review["originalCandidateRank"] = rank
    review["originalSelectedStableId"] = (
        candidates[rank - 1]["stableId"]
        if decision == "candidate" and isinstance(rank, int) and not isinstance(rank, bool)
        and 1 <= rank <= min(5, len(candidates)) else None
    )
    candidate_limit = min(5, len(candidates))
    candidate = None
    if decision in {"candidate", "genus", "family"}:
        if rank is None:
            decision = "reject_all"
            review["decision"] = decision
            review["notes"] = (
                (review.get("notes") or "")
                + " Program validation rejected a candidate-based decision without candidateRank."
            )
        elif not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= candidate_limit:
            raise ValueError(
                f"Candidate-based decision requires candidateRank 1-{candidate_limit}."
            )
        else:
            candidate = candidates[rank - 1]
    elif rank is not None:
        raise ValueError("reject_all/no_bird must use candidateRank null.")

    support_count = sum(item.get("status") == "supports" for item in features)
    distinguishing_support_count = sum(
        item.get("status") == "supports"
        and isinstance(item.get("distinguishesFromCandidateRanks"), list)
        and any(
            isinstance(other_rank, int)
            and not isinstance(other_rank, bool)
            and 1 <= other_rank <= candidate_limit
            and other_rank != rank
            for other_rank in item["distinguishesFromCandidateRanks"]
        )
        for item in features
    )
    diagnostic_support_count = sum(
        item.get("status") == "supports"
        and item.get("diagnosticity") == "species_specific"
        and isinstance(item.get("distinguishesFromCandidateRanks"), list)
        and any(
            isinstance(other_rank, int)
            and not isinstance(other_rank, bool)
            and 1 <= other_rank <= candidate_limit
            and other_rank != rank
            for other_rank in item["distinguishesFromCandidateRanks"]
        )
        for item in features
    )
    conflict_count = sum(item.get("status") == "conflicts" for item in features)
    minimum_diagnostic_supports = int(minimum_supporting_features)
    visual_support = _visual_candidate_support(
        local_result,
        candidate["stableId"] if candidate and decision == "candidate" else None,
    )
    consensus_exception_applied = bool(
        decision == "candidate"
        and minimum_diagnostic_supports >= 3
        and diagnostic_support_count >= minimum_diagnostic_supports - 1
        and conflict_count == 0
        and visual_support["passed"]
    )
    review["evidenceValidation"] = {
        "supportCount": support_count,
        "distinguishingSupportCount": distinguishing_support_count,
        "diagnosticSupportCount": diagnostic_support_count,
        "conflictCount": conflict_count,
        "minimumDiagnosticSupports": minimum_diagnostic_supports,
        "visualConsensusException": {
            **visual_support,
            "applied": consensus_exception_applied,
        },
    }
    evidence_policy = review_evidence_policy(force_candidate, candidates)
    strict_pair_fields = evidence_policy["top1Top2StructuredFields"]
    relaxed_evidence = evidence_policy["relaxed"]
    if decision == "candidate" and (
        (not force_candidate and (
            quality_score < 50
            or (
                diagnostic_support_count < minimum_diagnostic_supports
                and not consensus_exception_applied
            )
            or conflict_count > support_count
        ))
        or (
            force_candidate
            and local_result
            and local_result.get("pipelineConfig", {}).get("reviewForceAllowsUnknown")
            and (
                (strict_pair_fields
                 and sum(item.get("status") == "unobservable" for item in features) >= 5)
                or conflict_count > support_count
                or (strict_pair_fields and diagnostic_support_count == 0)
                or (relaxed_evidence and support_count == 0)
            )
        )
    ):
        decision = "reject_all"
        review["decision"] = decision
        review["candidateRank"] = None
        review["notes"] = (review.get("notes") or "") + " Program validation rejected insufficient species evidence."
    elif decision in {"genus", "family"} and (
        quality_score < 50
        or distinguishing_support_count < max(2, int(minimum_supporting_features) - 1)
        or conflict_count > support_count
    ):
        decision = "reject_all"
        review["decision"] = decision
        review["candidateRank"] = None
        review["notes"] = (review.get("notes") or "") + " Program validation rejected insufficient higher-taxon evidence."

    if decision == "candidate":
        report["canIdentify"] = True
        report["bird"] = True
        report["animal"] = True
        report["categories"] = candidate["scientificName"]
        report["name"] = candidate["commonName"]
        report["alias"] = report.get("alias") or ""
        review["selectedStableId"] = candidate["stableId"]
        if relaxed_evidence:
            report["confidence"] = min(
                evidence_policy["confidenceCap"],
                float(report.get("confidence") or 0),
            )
    elif decision == "genus":
        report["canIdentify"] = False
        report["bird"] = True
        report["animal"] = True
        report["categories"] = candidate["genus"]
        report["name"] = f"Uncertain {candidate['genus']} species"
        report["alias"] = ""
        report["confidence"] = min(0.6, float(report.get("confidence") or 0))
        review["selectedStableId"] = None
        review["evidenceCandidateStableId"] = candidate["stableId"]
    elif decision == "family":
        report["canIdentify"] = False
        report["bird"] = True
        report["animal"] = True
        report["categories"] = candidate["family"]
        report["name"] = candidate.get("familyEnglishName") or f"Uncertain {candidate['family']} bird"
        report["alias"] = ""
        report["confidence"] = min(0.5, float(report.get("confidence") or 0))
        review["selectedStableId"] = None
        review["evidenceCandidateStableId"] = candidate["stableId"]
    elif decision == "no_bird":
        review["selectedStableId"] = None
        report["canIdentify"] = False
        report["bird"] = False
        report["animal"] = any(report.get(field) is True for field in (
            "squirrel", "raccoon", "rattus", "snake", "opossum", "person"
        ))
        report["name"] = "No Bird"
        report["categories"] = "Non-bird"
        report["alias"] = ""
        report["confidence"] = min(0.3, float(report.get("confidence") or 0))
    elif decision == "reject_all":
        review["selectedStableId"] = None
        report["canIdentify"] = False
        report["bird"] = True
        report["animal"] = True
        report["name"] = "Unknown Bird"
        report["categories"] = "Unknown"
        report["alias"] = ""
        report["confidence"] = min(0.3, float(report.get("confidence") or 0))
    else:
        raise ValueError(f"Unsupported review decision: {decision}")
    review["evidencePolicy"] = {
        "mode": evidence_policy["mode"],
        "top1Top2StructuredFields": evidence_policy["top1Top2StructuredFields"],
        "confidenceCap": evidence_policy["confidenceCap"],
    }
    return report, review


def validate_report(report, quality_score, allow_low_quality=False, allow_location=False):
    if not isinstance(report, dict):
        raise ValueError("Doubao output is not a JSON object.")
    missing = [field for field in REQUIRED_FIELDS if field not in report]
    extra = [field for field in report if field not in REQUIRED_FIELDS]
    if missing or extra:
        raise ValueError(f"Boss JSON fields mismatch. missing={missing}, extra={extra}")
    if report["qualityScore"] != quality_score:
        raise ValueError("Doubao changed the locally computed qualityScore.")
    if not isinstance(report["canIdentify"], bool):
        raise ValueError("canIdentify must be boolean.")
    if not 0 <= float(report["confidence"]) <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if quality_score < 50 and report["canIdentify"] and not allow_low_quality:
        raise ValueError("canIdentify must be false when qualityScore is below 50.")
    if report["bird"] is False and report["canIdentify"]:
        raise ValueError("canIdentify must be false when no bird is present.")
    if report["rarity"] != "Unknown" and not allow_location:
        raise ValueError("rarity must be Unknown when no location is supplied.")
    for field in ("describe", "categories", "alias", "name", "healthDescription", "rarityReason"):
        if not isinstance(report[field], str):
            raise ValueError(f"{field} must be a string.")
    for field in ("colorTags", "patternTags", "shapeTags", "behaviorTags", "healthIssues"):
        if not isinstance(report[field], list) or any(isinstance(value, bool) or not isinstance(value, int) for value in report[field]):
            raise ValueError(f"{field} must be an integer array.")
    for field in ("flock", "bird", "person", "squirrel", "raccoon", "rattus", "snake", "opossum", "animal"):
        if not isinstance(report[field], bool):
            raise ValueError(f"{field} must be boolean.")


def _original_candidate_contrast_count(review, candidate_slots, original_ids):
    original_ids = set(original_ids or [])
    original_ranks = {
        row.get("rank")
        for row in candidate_slots or []
        if row.get("stableId") in original_ids
    }
    return sum(
        item.get("status") == "supports"
        and item.get("diagnosticity") == "species_specific"
        and bool(
            original_ranks
            & set(item.get("distinguishesFromCandidateRanks") or [])
        )
        for item in review.get("observedFeatures") or []
    )


def accept_confident_proposal(report, review, candidate_slots, quality_score, settings,
                              *, geo_filter_applied=False):
    """Accept a model-proposed species outside the supplied candidate shortlist.

    Only used when ``openSetEscape.allowConfidentProposal`` is enabled. The
    proposal must resolve to an AviList registry row and carry explicit visible
    evidence plus at least one contradicted candidate rank; otherwise the first
    pass result (usually reject_all) is kept. The accepted species is capped in
    confidence and recorded under ``review.openSetProposal`` for audit.

    When the hard geographic filter restricted the candidate pool, a proposal is
    a deliberate bypass (non-local bird, e.g. screen/remote stream); it needs at
    least ``proposalGeoBypassMinEvidence`` visible evidence and is marked with
    ``geoFilterBypassed`` in the audit trail.
    """
    proposals = review.get("outsideCandidates") or []
    status = {
        "enabled": True,
        "confirmationMode": "confident_proposal",
        "approved": False,
        "reason": "no_outside_proposal",
        "geoFilterBypassed": bool(geo_filter_applied),
        "proposals": copy.deepcopy(proposals),
        "rejections": [],
    }
    if not proposals:
        return status
    decision = review.get("decision")
    overriding_candidate = decision == "candidate"
    selected_row = None
    selected_rank = None
    if overriding_candidate:
        if not bool(settings.get("proposalOverrideCandidate", False)):
            status["reason"] = "first_pass_candidate_kept"
            return status
        selected_rank = review.get("candidateRank")
        if (
            not isinstance(selected_rank, int)
            or isinstance(selected_rank, bool)
            or not 1 <= selected_rank <= len(candidate_slots)
        ):
            status["reason"] = "candidate_rank_unusable_for_override"
            return status
        selected_row = candidate_slots[selected_rank - 1]
    elif decision not in {"reject_all", "genus", "family"}:
        status["reason"] = "first_pass_decision_kept"
        return status
    if int(quality_score) < int(settings.get("proposalMinimumQuality", 50)):
        status["reason"] = "image_quality_below_proposal_threshold"
        return status
    minimum_features = int(
        settings.get("proposalOverrideMinEvidence", 3)
        if overriding_candidate
        else settings.get("proposalMinimumVisibleFeatures", 2)
    )
    if geo_filter_applied:
        minimum_features = max(
            minimum_features, int(settings.get("proposalGeoBypassMinEvidence", 3))
        )
    cap = float(
        settings.get("proposalOverrideConfidenceCap", 0.4)
        if overriding_candidate
        else settings.get("proposalConfidenceCap", 0.5)
    )
    existing_names = {
        " ".join(str(row.get("scientificName") or "").split()).casefold()
        for row in candidate_slots
    }
    from bird_cli import resolve_species_entry  # lazy import avoids a cycle

    for proposal in proposals[: int(settings.get("maximumProposals", 1))]:
        raw_name = str(proposal.get("scientificName") or "").strip()
        entry = resolve_species_entry(raw_name) or resolve_species_entry(
            proposal.get("commonName")
        )
        evidence = list(proposal.get("observedFeatureEvidence") or [])
        contradicted = list(proposal.get("contradictedCandidateRanks") or [])
        contradicted_ranks = {
            int(value)
            for value in contradicted
            if isinstance(value, int) and not isinstance(value, bool)
        }
        reason = None
        if not raw_name:
            reason = "missing_scientific_name"
        elif entry is None:
            reason = "species_not_in_registry"
        elif (
            " ".join(str(entry.get("scientificName") or "").split()).casefold()
            in existing_names
        ):
            reason = "already_in_candidate_slots"
        elif len(evidence) < minimum_features:
            reason = "insufficient_visible_feature_evidence"
        elif not contradicted:
            reason = "no_contradicted_candidate_rank"
        elif overriding_candidate and int(selected_rank) not in contradicted_ranks:
            reason = "selected_candidate_not_contradicted"
        elif overriding_candidate and (
            " ".join(str(entry.get("family") or "").split()).casefold()
            != " ".join(str(selected_row.get("family") or "").split()).casefold()
        ):
            reason = "proposal_family_differs_from_selected_candidate"
        if reason is not None:
            status["rejections"].append({"scientificName": raw_name, "reason": reason})
            continue

        scientific_name = str(entry.get("scientificName") or raw_name)
        common_name = str(
            entry.get("commonName") or proposal.get("commonName") or scientific_name
        )
        report["canIdentify"] = True
        report["bird"] = True
        report["animal"] = True
        report["categories"] = scientific_name
        report["name"] = common_name
        report["alias"] = ""
        report["confidence"] = round(
            min(cap, max(0.3, float(report.get("confidence") or 0))), 4
        )
        review["originalDecision"] = review.get("originalDecision") or review.get("decision")
        review["decision"] = "candidate"
        review["selectedStableId"] = entry.get("stableId")
        review["openSetProposal"] = {
            "scientificName": scientific_name,
            "commonName": common_name,
            "sourceName": raw_name,
            "observedFeatureEvidence": evidence,
            "contradictedCandidateRanks": contradicted,
            "resolvedStableId": entry.get("stableId"),
            "geoFilterBypassed": bool(geo_filter_applied),
            "overrodeCandidate": (
                {
                    "rank": int(selected_rank),
                    "stableId": selected_row.get("stableId"),
                    "scientificName": selected_row.get("scientificName"),
                }
                if overriding_candidate
                else None
            ),
        }
        review["notes"] = (
            (review.get("notes") or "")
            + (
                " Confident open-set proposal replaced the first-pass candidate after registry validation."
                if overriding_candidate
                else " Confident open-set proposal accepted after registry validation."
            )
            + (
                " The proposal bypassed the hard geographic filter because the image contradicts the local avifauna."
                if geo_filter_applied
                else ""
            )
        )
        status.update({
            "approved": True,
            "reason": (
                "confident_proposal_overrode_candidate"
                if overriding_candidate
                else "confident_proposal_accepted"
            ),
            "overrodeCandidate": bool(overriding_candidate),
            "geoFilterBypassed": bool(geo_filter_applied),
            "acceptedCandidate": {
                "stableId": entry.get("stableId"),
                "scientificName": scientific_name,
                "commonName": common_name,
            },
        })
        return status
    status["reason"] = "all_proposals_rejected"
    return status


def create_boss_report(original_image, focus_image, local_result, quality_score, review_trigger=None, event_images=None, event_focus_crops=None, *, pass_context=None):
    if event_images is not None:
        event_images = tuple(event_images)
        if len(event_images) != 3 or any(not isinstance(image, Image.Image) for image in event_images):
            raise ValueError("event_images must contain exactly three PIL images.")
    api_key, base_url, model = load_config()
    # 显式 pass 上下文（优先），兼容旧私有键；新代码请只传 pass_context。
    is_second_pass = pass_context == "second_pass" or bool(local_result.get("_openSetEscapeSecondPass"))
    is_single_pass = pass_context == "single_pass" or bool(local_result.get("_openSetEscapeSinglePass"))
    if review_trigger is None:
        review_trigger = "risk" if local_result.get("risk", {}).get("needsReview") else "unspecified"
    mode = choose_prompt_mode(local_result, review_trigger)
    escape_settings = open_set_escape_settings(local_result)
    force_candidate = bool((local_result.get("pipelineConfig") or {}).get("reviewForceCandidate"))
    force_allows_unknown = bool((local_result.get("pipelineConfig") or {}).get("reviewForceAllowsUnknown"))
    location = valid_wgs84_location(local_result.get("location"))
    has_location = location is not None
    geographic_prior = local_result.get("geographicPrior") or {}
    if has_location:
        latitude, longitude = location
        rarity_rule = (
            f"A supplied WGS84 location is available at latitude {latitude:.5f}, longitude {longitude:.5f}. "
            "Coordinates and retrieval support weights are not calibrated abundance evidence. "
            "Set rarity to Unknown; do not infer or invent Common, Rare, or other rarity labels."
        )
    else:
        rarity_rule = (
            'No valid sourced WGS84 coordinates were supplied, so rarity must be "Unknown" and rarityReason '
            "must say that location was not supplied."
        )
    if geographic_prior:
        rarity_rule += " Supplied geographic prior status summary: " + json.dumps(geographic_prior, ensure_ascii=False, separators=(",", ":")) + "."

    single_pass_status = None
    if (
        escape_settings["enabled"]
        and escape_settings["confirmationMode"] == "single_pass_shortlist"
        and not is_single_pass
        and not is_second_pass
    ):
        local_result, single_pass_status = build_open_set_single_pass_local_result(
            local_result, escape_settings
        )
    candidates = candidate_context(local_result)
    candidate_count = candidates["candidateCount"]
    if candidate_count < 1:
        raise ValueError("No review candidates were supplied.")
    prompt_evidence_policy = review_evidence_policy(
        force_candidate, candidates["candidateSlots"]
    )
    prompt_pair_fields = prompt_evidence_policy["top1Top2StructuredFields"]
    challenge_rows, challenge_status = challenge_candidate_context(
        local_result, candidates["candidateSlots"], escape_settings
    )
    allow_confident_proposal = bool(
        escape_settings["enabled"]
        and escape_settings.get("allowConfidentProposal")
        and not is_second_pass
    )
    allow_open_set_proposals = allow_confident_proposal or bool(
        escape_settings["enabled"]
        and escape_settings["confirmationMode"] == "second_pass"
        and not is_second_pass
        and challenge_rows
    )
    detail_board = (
        build_head_detail_board(focus_image)
        if event_images is None and escape_settings.get("headDetailBoardEnabled", False)
        else None
    )
    detail_evidence_rule = (
        "The user message also includes a 2x2 board of overlapping top-left, "
        "top-right, bottom-left, and bottom-right crops from the same detected "
        "bird. These are duplicate detail views, not four birds. Use the board "
        "only to inspect small head, eye, crest, and bill features. If shadow, "
        "oblique angle, blur, or crop boundaries prevent a clear observation, "
        "mark the feature unobservable. Failure to see yellow bare facial skin "
        "is not evidence that it is absent. Ruffled forehead feathers are not a "
        "crest unless a distinct protruding tuft contour is clearly visible in "
        "both the original bird view and a detail panel.\n"
        if detail_board is not None
        else ""
    )
    if mode == "scene_only":
        scene_route = local_result.get("sceneRouter", {})
        comparison_rule = (
            "First make an independent bird-presence judgment from the images. "
            "The local BioCLIP candidate list may be wrong and must not influence this judgment. "
            "Report no bird only if no bird is visible in any supplied image, even if BioCLIP supplied bird candidates. "
            "The presence of a non-bird animal or person does not rule out a coexisting bird; still record any non-bird animal or person you can see. "
            "Only after independently confirming that a bird is visible may you inspect the supplied bird candidates. "
        )
    elif mode == "evidence_full":
        comparison_rule = (
            f"Compare all {candidate_count} supplied candidate slots independently, "
            "feature by feature. Reject every candidate whose visible morphology conflicts with the image. "
        )
    else:
        comparison_rule = "First form an independent broad-family judgment, then verify the leading candidates against directly visible morphology. "
    candidate_section = "Allowed local candidates (potentially wrong; never accept without visual support):\n" + json.dumps(candidates, ensure_ascii=False) + "\n"
    reference_section, card_status = reference_hint_section(
        local_result, candidates, mode
    )
    graph_section, graph_status = graph_hint_section(
        local_result,
        candidates,
        allow_open_set_proposals=allow_open_set_proposals,
    )
    trait_section, trait_status = candidate_trait_section(local_result, candidates)
    if allow_open_set_proposals:
        challenge_section = (
            "Program-supplied challenge candidates (scores hidden; comparison only; "
            "not current output choices):\n"
            + json.dumps(challenge_status, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        escape_rule = f"""
Challenge rule: a final answer is still restricted to the supplied candidate slots. Independently compare every program-supplied challenge candidate against the image, even when you currently prefer one of the supplied candidates. If a challenge candidate visibly fits better, place it in outsideCandidates while keeping the current-pass decision restricted to the original candidate slots. You may add up to {escape_settings['maximumProposals']} outsideCandidates, using only exact scientific names from the challenge pool. Each proposal must list at least {escape_settings['minimumProposalFeatures']} directly visible diagnostic features and identify at least one supplied candidate rank contradicted by the image. A challenge proposal is not a final answer and always requires a separate confirming review. Do not propose any challenge candidate when none is visibly stronger.
"""
        shortlist_rule = (
            "You must not answer with an unlisted species; unlisted species names are allowed only "
            "inside the non-final outsideCandidates proposal field."
        )
    else:
        escape_rule = ""
        challenge_section = ""
        shortlist_rule = "You must not add any unlisted species."
    quality_rule = (
        "The forced-choice experiment may identify below quality 50 only when its evidence safety gates pass."
        if force_candidate else "If it is below 50, canIdentify must be false."
    )
    evidence_rule = (
        "Use only directly visible diagnostic contrasts; apply the forced-choice safety gates below."
        if force_candidate else
        f"Requires at least {local_result['pipelineConfig']['reviewMinimumSupportingFeatures']} visibly supporting, species-specific diagnostic features."
    )
    system_prompt = f"""输出保持精炼：说明性文字不超过 40 个英文单词，notes 不超过 2 句，healthDescription 不超过 25 个英文单词；八项特征与诊断性判定仍须完整给出，不得省略证据。
你是专业的鸟类图像审核代理。所有说明性文字必须使用英文，以确保下游契约不变。
本地质量分为 {quality_score}，按质量管理要求执行。{quality_rule}
{comparison_rule}
决策策略：鸟清晰可见时，选择可见形态与图像最一致的已提供候选。不得仅因部分特征不可观察、属共有特征或其它槽位存在冲突就返回拒选；只有当没有任何已提供候选与清晰可见的鸟一致时（例如某项可见诊断特征主动否定了所有槽位）才拒选。即使少数特征模糊或被遮挡，最佳可见匹配仍是有效决策。
已提供的 BioCLIP 候选只是本次决策的封闭短名单，不是事实。其编号是稳定哈希生成的普通槽位号，不是置信度排名；检索分数与本地分类器投票已被刻意隐藏。绝不偏向较小槽位号。{shortlist_rule} {evidence_rule} 若判断只能到属或科，则返回所选候选列出的层级。所有候选都冲突或证据不足时应拒选；看不到鸟时应报告无鸟。
严格评估以下八项特征：喙、眼圈、冠羽、面部/头部图纹、翼斑、尾形、体色、眼周裸皮与虹膜颜色。体色评估整体色调及上体与下体的对比；眼周裸皮评估眼周或眼后裸皮与虹膜颜色，其与羽状眼圈不同。每项都要给出状态判断、诊断性判断，以及它真正能区分的其它候选槽位。面罩、眉纹、颊斑、顶纹或边界清晰的头部色块等可见面部/头部图纹应与冠羽分开。多数鸟共有的宽泛特征不算物种证据；不得按科级刻板印象推断喙形、冠羽、面部图纹或羽色。一般性的粗喙或深色喙、平滑头部或无冠羽、宽泛的棕/红褐翼色，除非已知确切的可见形状或图案能排除某个具名竞争候选，否则不算物种特异。体色偏深/偏浅，或虹膜与裸皮偏浅/偏深，仅在对比候选已知恰好在该维度不同时才属物种特异；不知道确切对比时按共有特征处理。下文相对差异说明给出了已提供候选对的确切对比维度：明确列出的维度视为已知，须目视评估，不得按共有特征处理。不可观察的特征不得声称具有物种诊断性，也不得声称能区分任何候选。绝不虚构隐藏特征或图像不支持的对比。
{detail_evidence_rule}
name 使用完整 eBird 英文常用名，categories 使用双名法拉丁学名。除非直接可见，不得推断性别、年龄、羽色、健康问题、行为或标签。证据缺失时如实留空，不要编造。
{rarity_rule} 无可靠别名或亚科时不要编造。
健康评估仅为目视筛查，按语义标注外观健康、外观病态、外观受伤或外观应激；出现健康问题时标明受影响部位：羽毛、眼、喙、腿/脚、呼吸或行为。
{escape_rule}
{TAG_GUIDE}
{candidate_section}{trait_section}{challenge_section}{reference_section}{graph_section}"""
    if force_candidate:
        system_prompt += (
            "\n强制选择模式：比较已提供的候选槽位，但不得覆盖程序安全门控。"
            "鸟可见且证据通过门控时，必须选择一个已提供候选；看不到鸟时，应报告无鸟。"
        )
        if force_allows_unknown:
            if prompt_pair_fields:
                # strict: both top-2 candidates have curated morphology fields
                system_prompt += (
                    " 前二候选均有结构化特征字段，按严格证据执行："
                    "若五项或以上必需特征不可观察、无任何物种特异诊断支持，或冲突数超过支持数，"
                    "则应拒选，不要猜测物种，并保持低置信度。"
                    "证据只有一个候选时，不得虚构平票或前二比较。"
                )
            elif quality_score >= 80:
                # relaxed + high quality: still force a best-effort candidate
                system_prompt += (
                    " 当前图像质量高，但前二候选未命中结构化特征字段："
                    "只要鸟可见且形态与某个候选整体吻合，即使仅有属共有特征支持，"
                    "也必须选该候选并给出具体物种，并保持低置信度；"
                    "仅当没有任何可见特征支持，或冲突数超过支持数时，才应拒选。"
                    "证据只有一个候选时，不得虚构平票或前二比较。"
                )
            else:
                # relaxed + low quality
                system_prompt += (
                    " 前二候选未同时命中结构化特征字段，放宽证据门槛："
                    "只要鸟清晰可见、至少一项可见特征支持某个候选，且冲突数不超过支持数，"
                    "就应选择整体最吻合的候选，不得仅因缺少物种特异证据而拒选；"
                    "此时保持低置信度，且不得虚构图像中不可见的诊断特征。"
                    "仅当没有任何可见特征支持，或冲突数超过支持数时，才应拒选。"
                    "证据只有一个候选时，不得虚构平票或前二比较。"
                )

    if allow_confident_proposal:
        system_prompt += (
            "\n开放提案：如果所有已提供候选都与可见形态明显不符，而图像证据明确指向候选池以外的物种，"
            f"且你能给出至少 {int(escape_settings.get('proposalMinimumVisibleFeatures', 2))} 项直接可见的形态证据、"
            "并指出被它否定的候选槽位号，可以在 outsideCandidates 中填写该物种："
            "scientificName 用双名法拉丁学名（eBird/AviList 规范名），commonName 用 eBird 英文常用名，"
            "observedFeatureEvidence 列出可见证据，contradictedCandidateRanks 列出被否定的候选槽位号。"
            "即使首轮已选出一个候选，只要该候选与可见形态仍明显不符、而池外物种更吻合，也可以这样提案，"
            "并把该候选的槽位号一并写进 contradictedCandidateRanks。"
            "没有把握时不要填写，不得编造物种。"
        )
        geo_filter_context = (local_result.get("geographicPrior") or {}).get("filter") or {}
        if geo_filter_context.get("applied"):
            location = local_result.get("location") or {}
            try:
                coordinate_text = (
                    f"latitude {float(location.get('latitude')):.5f}, "
                    f"longitude {float(location.get('longitude')):.5f}"
                )
            except (TypeError, ValueError):
                coordinate_text = "unknown"
            system_prompt += (
                f"\n地理约束说明：设备坐标为 {coordinate_text}，候选槽位已按所在 3° 网格"
                f"（{geo_filter_context.get('cellId')}）的本地鸟类池"
                f"（{geo_filter_context.get('poolSpecies')} 种）过滤。"
                "如果图像清晰显示本地不可能出现的物种，"
                "你可以在 outsideCandidates 中给出该物种，并在 notes 说明为何排除本地候选；"
                "如果只是在本地鸟种之间取舍，不要提案。"
            )

    if mode == "scene_only":
        user_instruction = (
            "先独立检查图像判断是否有鸟，再考虑已提供候选。"
            "本地场景路由（含 uncertain 或 non_bird）不能证明无鸟。"
            "仅当所有提供图像中都没有鸟时，才报告无鸟。"
            "若鸟可见，按现有证据规则比较已提供候选；同时记录画面中可见的其它动物或人。"
        )
    elif focus_image is None:
        user_instruction = (
            "检查提供的最佳可用鸟类视图，然后仅比较已提供候选。返回报告与审核证据。"
        )
    else:
        user_instruction = (
            "先检查原图，再将焦点图仅作为辅助证据。返回报告与审核证据。"
        )
    if event_images is not None:
        user_instruction = (
            "这是同一相机事件的多帧画面：最多 3 张原图，以及最多 2 张鸟体特写（由检测框裁剪），"
            "通常为同一个体的不同姿态。"
            "除非明显显示不同个体（例如不同的目标动物），否则按同一只鸟处理。"
            "先独立观察每张原图与每张特写，再合并该个体跨帧的直接可见证据：某帧隐藏或遮挡的特征可能在另一帧清晰可见，应以最清晰帧为准。"
            "某帧中不可见的特征在该帧为不可观察，不等于缺失，也不构成冲突；不得让模糊、过远或不完整的帧覆盖清晰的诊断视图。"
            "不要平均冲突的颜色或图案，优先采用该特征最清晰、遮挡最少的帧。"
            "若各组画面明显为不同个体，以诊断视图最清晰的帧作为该事件的唯一决策依据，并说明依据哪一帧。"
            "只有所有画面都无鸟时才用 no_bird。"
            "若鸟可见，仅按现有证据规则比较已提供候选。"
            "返回报告与审核证据，保留共存的 animal/person 布尔字段。"
        )
        content = [{"type": "input_text", "text": user_instruction}]
        for index, image in enumerate(event_images):
            label = (
                "事件帧 1（原图，整体场景）："
                if index == 0
                else f"事件帧 {index + 1}（原图，补充视角）："
            )
            content.extend([
                {"type": "input_text", "text": label},
                {"type": "input_image", "image_url": image_data_url(image, 1024, 88), "detail": "high"},
            ])
        crops = tuple(image for image in (event_focus_crops or ()) if isinstance(image, Image.Image))
        for index, crop in enumerate(crops[:2], 1):
            content.extend([
                {"type": "input_text", "text": f"鸟体特写 {index}（检测框裁剪，用于判读羽色、斑纹、喙形等细节）："},
                {"type": "input_image", "image_url": image_data_url(crop, 900, 92), "detail": "high"},
            ])
    else:
        content = [
            {"type": "input_text", "text": user_instruction},
            {"type": "input_image", "image_url": image_data_url(original_image, 1024, 88), "detail": "high" if mode == "scene_only" or focus_image is None else "low"},
        ]
    if event_images is None and focus_image is not None:
        content.extend([
            {"type": "input_text", "text": "这是自动检测的鸟类裁剪图，可能有误，请与原图交叉核对。"},
            {"type": "input_image", "image_url": image_data_url(focus_image, 900, 90), "detail": "high"},
        ])
    if detail_board is not None:
        content.extend([
            {
                "type": "input_text",
                "text": (
                    "此 2x2 拼图为同一检测鸟的四个重叠角部细节，"
                    "其中一格可能更清晰地显示头部。"
                ),
            },
            {
                "type": "input_image",
                "image_url": image_data_url(detail_board, 1024, 92),
                "detail": "high",
            },
        ])

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
        "reasoning": {"effort": os.environ.get("DOUBAO_REASONING_EFFORT", "minimal")},
        "max_output_tokens": int(os.environ.get("DOUBAO_MAX_OUTPUT_TOKENS", "1400")),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bird_monitor_candidate_review",
                "schema": review_schema(
                    quality_score,
                    candidate_count,
                    allow_open_set_proposals=allow_open_set_proposals,
                    outside_scientific_names=(
                        []
                        if allow_confident_proposal
                        else [row["scientificName"] for row in challenge_rows]
                    ),
                    force_candidate=force_candidate,
                    force_allows_unknown=force_allows_unknown,
                ),
                "strict": True,
            }
        },
    }
    temperature = os.environ.get("DOUBAO_TEMPERATURE", "0")
    if temperature not in (None, ""):
        payload["temperature"] = float(temperature)
    request = urllib.request.Request(
        responses_endpoint(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message") or body
        except json.JSONDecodeError:
            message = body
        raise RuntimeError(f"Doubao request failed ({error.code}): {message}") from error

    parsed = json.loads(response_text(response_payload))
    if not isinstance(parsed, dict) or "report" not in parsed or "review" not in parsed:
        raise ValueError("Doubao output is missing report/review sections.")
    report = sanitize_report(parsed["report"])
    report, review = enforce_candidate_decision(
        report,
        parsed["review"],
        candidates["candidateSlots"],
        quality_score,
        local_result["pipelineConfig"]["reviewMinimumSupportingFeatures"],
        local_result=local_result,
        force_candidate=force_candidate,
    )
    # Retrieval support is not an abundance/rarity estimate, even with GPS.
    report["rarity"] = "Unknown"
    report["rarityReason"] = "No calibrated local abundance data was supplied."
    validate_report(
        report,
        quality_score,
        allow_low_quality=force_candidate,
        allow_location=False,
    )
    meta = {
        "model": model,
        "promptMode": mode,
        "reviewTrigger": review_trigger,
        "inputImageCount": sum(item["type"] == "input_image" for item in content),
        "eventImageCount": len(event_images) if event_images is not None else 0,
        "latencyMs": round((time.perf_counter() - started) * 1000),
        "usage": response_payload.get("usage"),
        "id": response_payload.get("id"),
        "review": review,
        "identificationCards": card_status,
        "knowledgeGraph": graph_status,
        "candidateTraits": trait_status,
        "geoStatus": "valid_sourced_wgs84" if has_location else "unknown_no_valid_sourced_wgs84",
    }
    if allow_confident_proposal:
        geo_filter_applied = bool(
            ((local_result.get("geographicPrior") or {}).get("filter") or {}).get("applied")
        )
        proposal_status = accept_confident_proposal(
            report,
            review,
            candidates["candidateSlots"],
            quality_score,
            escape_settings,
            geo_filter_applied=geo_filter_applied,
        )
        if proposal_status.get("approved"):
            meta["openSetEscape"] = proposal_status
            validate_report(
                report,
                quality_score,
                allow_low_quality=force_candidate,
                allow_location=has_location,
            )
            return report, meta
        meta["confidentProposal"] = proposal_status
    if single_pass_status is not None:
        injected_by_id = {
            row["stableId"]: row
            for row in single_pass_status["injectedCandidates"]
        }
        selected_id = review.get("selectedStableId")
        evidence = review.get("evidenceValidation") or {}
        original_contrast_count = _original_candidate_contrast_count(
            review,
            candidates["candidateSlots"],
            single_pass_status.get("originalCandidateStableIds") or [],
        )
        minimum_original_contrasts = int(
            escape_settings.get("singlePassMinimumOriginalCandidateContrasts", 0)
        )
        approved = bool(
            review.get("decision") == "candidate"
            and selected_id in injected_by_id
            and int(evidence.get("diagnosticSupportCount") or 0)
            >= int(local_result["pipelineConfig"]["reviewMinimumSupportingFeatures"])
            and int(evidence.get("conflictCount") or 0) == 0
            and original_contrast_count >= minimum_original_contrasts
        )
        if selected_id in injected_by_id and not approved:
            selected = injected_by_id[selected_id]
            genus = str(selected.get("scientificName") or "").split()[0]
            report["canIdentify"] = False
            report["categories"] = genus or "Unknown"
            report["name"] = (
                f"Uncertain {genus} species" if genus else "Unknown Bird"
            )
            report["alias"] = ""
            report["confidence"] = min(
                0.4, float(report.get("confidence") or 0.0)
            )
            review["decision"] = "genus" if genus else "reject_all"
            review["selectedStableId"] = None
            review["evidenceCandidateStableId"] = selected_id
        single_pass_status["approved"] = approved
        single_pass_status["originalCandidateContrastCount"] = (
            original_contrast_count
        )
        single_pass_status["minimumOriginalCandidateContrasts"] = (
            minimum_original_contrasts
        )
        single_pass_status["acceptedCandidates"] = (
            [injected_by_id[selected_id]] if approved else []
        )
        single_pass_status["reason"] = (
            "single_pass_confirmed_injected_candidate"
            if approved
            else "single_pass_injected_candidate_failed_zero_conflict_gate"
            if selected_id in injected_by_id
            else "single_pass_kept_original_candidate"
            if review.get("decision") == "candidate"
            else "single_pass_inconclusive"
        )
        meta["openSetEscape"] = single_pass_status
        validate_report(
            report,
            quality_score,
            allow_low_quality=force_candidate,
            allow_location=has_location,
        )
        return report, meta
    if not allow_open_set_proposals:
        return report, meta

    accepted, escape_status = select_open_set_candidates(
        review,
        local_result,
        candidates["candidateSlots"],
        escape_settings,
    )
    meta["openSetEscape"] = escape_status
    if not accepted:
        return report, meta
    if event_images is not None:
        escape_status.update({
            "approved": False,
            "reason": "event_single_request_requires_separate_confirmation",
        })
        return report, meta

    second_local_result = build_open_set_second_pass_local_result(
        local_result,
        candidates["candidateSlots"],
        accepted,
        escape_settings,
    )
    try:
        second_report, second_meta = create_boss_report(
            original_image,
            focus_image,
            second_local_result,
            quality_score,
            review_trigger="open_set_escape",
            pass_context="second_pass",
        )
    except Exception as error:
        escape_status.update(
            {
                "approved": False,
                "reason": "second_pass_failed",
                "errorType": type(error).__name__,
                "error": str(error),
            }
        )
        return report, meta

    accepted_ids = {row["stableId"] for row in accepted}
    second_review = second_meta.get("review") or {}
    evidence = second_review.get("evidenceValidation") or {}
    approved = bool(
        second_review.get("decision") == "candidate"
        and second_review.get("selectedStableId") in accepted_ids
        and int(evidence.get("diagnosticSupportCount") or 0)
        >= int(escape_settings["secondPassMinimumSupportingFeatures"])
        and int(evidence.get("conflictCount") or 0) == 0
    )
    escape_status.update(
        {
            "approved": approved,
            "reason": (
                "second_pass_confirmed_outside_candidate"
                if approved
                else "second_pass_kept_original_candidate"
                if second_review.get("decision") == "candidate"
                else "second_pass_inconclusive"
            ),
            "secondPass": second_meta,
        }
    )
    meta["latencyMs"] += int(second_meta.get("latencyMs") or 0)
    meta["review"] = second_review
    meta["identificationCards"] = second_meta.get("identificationCards")
    meta["knowledgeGraph"] = second_meta.get("knowledgeGraph")
    return second_report, meta
