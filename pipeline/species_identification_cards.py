"""Load and budget optional species-identification reference excerpts."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


CARD_SCHEMA_VERSION = "species-identification-card-v1"
MANIFEST_SCHEMA_VERSION = "species-identification-cards-manifest-v1"
EXPECTED_SPECIES_COUNT = 1454
EXPECTED_SPECIES_SET_SHA256 = (
    "f3ad8cbd4d790355337b3dfa2dff220738b5cb9661dc115d6b06318cc259f35a"
)
FEATURE_ORDER = (
    "bill",
    "eye_ring",
    "crest",
    "face_head_pattern",
    "wing_marks",
    "tail_shape",
)


class CardPackError(ValueError):
    """The configured identification-card pack is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _species_set_fingerprint(cards) -> str:
    pairs = sorted(
        (str(card["stableId"]), str(card["scientificName"])) for card in cards
    )
    payload = "".join(
        f"{stable_id}\t{scientific_name}\n"
        for stable_id, scientific_name in pairs
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _positive_int(settings, name, default, maximum):
    try:
        value = int(settings.get(name, default))
    except (TypeError, ValueError) as error:
        raise CardPackError(f"identificationCards.{name} must be an integer") from error
    if not 1 <= value <= maximum:
        raise CardPackError(
            f"identificationCards.{name} must be between 1 and {maximum}"
        )
    return value


def resolve_card_settings(config):
    """Return a copy of card settings with file paths resolved to the config root."""
    settings = dict(config.get("identificationCards") or {"mode": "off"})
    settings["mode"] = str(settings.get("mode", "off"))
    config_root = Path(config["_path"]).parent if config.get("_path") else Path.cwd()
    for key in ("cardsFile", "manifestFile"):
        value = settings.get(key)
        if value:
            path = Path(value)
            settings[key] = str(path if path.is_absolute() else config_root / path)
    return settings


def _read_manifest(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CardPackError(f"cannot read card manifest: {path}") from error
    if not isinstance(payload, dict):
        raise CardPackError("card manifest must be a JSON object")
    return payload


@lru_cache(maxsize=4)
def load_card_pack(
    cards_file,
    manifest_file,
    expected_manifest_sha256,
    expected_cards_sha256,
    expected_species_count=EXPECTED_SPECIES_COUNT,
    expected_species_set_sha256=EXPECTED_SPECIES_SET_SHA256,
):
    """Load an immutable card pack once and verify its complete hash contract."""
    cards_path = Path(cards_file)
    manifest_path = Path(manifest_file)
    if not cards_path.is_file() or not manifest_path.is_file():
        raise CardPackError("card pack files are missing")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if (
        not expected_manifest_sha256
        or actual_manifest_sha256 != expected_manifest_sha256
    ):
        raise CardPackError("card manifest does not match the configured SHA-256")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise CardPackError("unexpected card manifest schema")
    if int(manifest.get("speciesCount", -1)) != int(expected_species_count):
        raise CardPackError("card species count does not match configuration")
    if manifest.get("speciesSetSha256") != expected_species_set_sha256:
        raise CardPackError("card species-set fingerprint does not match configuration")

    output_contract = (manifest.get("outputs") or {}).get("cards.jsonl") or {}
    actual_cards_sha256 = _sha256_file(cards_path)
    if not expected_cards_sha256 or actual_cards_sha256 != expected_cards_sha256:
        raise CardPackError("cards.jsonl does not match the configured SHA-256")
    if output_contract.get("sha256") != actual_cards_sha256:
        raise CardPackError("cards.jsonl does not match its manifest SHA-256")

    cards_by_stable_id = {}
    line_count = 0
    try:
        with cards_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                line_count += 1
                card = json.loads(line)
                if not isinstance(card, dict):
                    raise CardPackError(f"card row {line_number} is not an object")
                if card.get("schemaVersion") != CARD_SCHEMA_VERSION:
                    raise CardPackError(f"card row {line_number} has an unexpected schema")
                stable_id = str(card.get("stableId") or "").strip()
                scientific_name = str(card.get("scientificName") or "").strip()
                if not stable_id or not scientific_name or stable_id in cards_by_stable_id:
                    raise CardPackError(f"card row {line_number} has an empty or duplicate stableId")
                expected_card_sha256 = card.get("cardSha256")
                unhashed = dict(card)
                unhashed.pop("cardSha256", None)
                if expected_card_sha256 != _sha256_bytes(_canonical_json_bytes(unhashed)):
                    raise CardPackError(f"card row {line_number} has an invalid cardSha256")
                cards_by_stable_id[stable_id] = card
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CardPackError(f"cannot parse card pack: {cards_path}") from error

    manifest_lines = int(output_contract.get("lineCount", -1))
    if line_count != int(expected_species_count) or line_count != manifest_lines:
        raise CardPackError("card JSONL line count does not match its contract")
    actual_fingerprint = _species_set_fingerprint(cards_by_stable_id.values())
    if actual_fingerprint != expected_species_set_sha256:
        raise CardPackError("card rows do not match the configured species-set fingerprint")
    return {
        "cardsByStableId": cards_by_stable_id,
        "cardsSha256": actual_cards_sha256,
        "manifestSha256": actual_manifest_sha256,
        "speciesCount": line_count,
        "speciesSetSha256": expected_species_set_sha256,
    }


def _eligible_sentences(card, mode, max_statements, max_chars):
    if mode == "approved":
        if not card.get("humanApproved") or not card.get("runtimeEligible"):
            return []
    elif mode == "source_excerpt":
        if card.get("knowledgeStatus") != "auto_source_excerpt":
            return []
        if card.get("sourceMode") != "morphology_sections":
            return []
    else:
        return []

    selected = []
    seen = set()
    used_chars = 0
    feature_evidence = card.get("featureEvidence") or {}
    maximum_rows = max(
        (len(feature_evidence.get(feature) or []) for feature in FEATURE_ORDER),
        default=0,
    )
    for evidence_index in range(maximum_rows):
        for feature in FEATURE_ORDER:
            rows = feature_evidence.get(feature) or []
            if evidence_index >= len(rows):
                continue
            row = rows[evidence_index]
            sentence = str(row.get("sentence") or "").strip()
            if not sentence or sentence in seen:
                continue
            if used_chars + len(sentence) > max_chars:
                continue
            selected.append({
                "feature": feature,
                "text": sentence,
                "heading": str(row.get("heading") or ""),
                "sentenceSha256": str(row.get("sentenceSha256") or ""),
            })
            seen.add(sentence)
            used_chars += len(sentence)
            if len(selected) >= max_statements:
                return selected
    return selected


def _serialized_chars(payload) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _prompt_statement(option):
    return {"feature": option["feature"], "text": option["text"]}


def _citation_source(source):
    source = source or {}
    return {
        key: source[key]
        for key in (
            "provider",
            "fixedRevisionUrl",
            "revisionId",
            "revisionTimestamp",
            "sourceSha256",
            "license",
            "licenseUrl",
        )
        if source.get(key) not in {None, ""}
    }


def build_reference_hints(candidate_slots, settings, prompt_mode):
    """Build complete-coverage hints or return a baseline-preserving fallback status."""
    mode = str((settings or {}).get("mode", "off"))
    if mode == "off":
        return None, {"mode": mode, "applied": False, "reason": "disabled"}
    if prompt_mode == "scene_only":
        return None, {"mode": mode, "applied": False, "reason": "scene_only"}
    if mode not in {"approved", "source_excerpt"}:
        return None, {"mode": mode, "applied": False, "reason": "invalid_mode"}
    if mode == "source_excerpt" and not settings.get("allowUnreviewedSourceExcerpts"):
        return None, {
            "mode": mode,
            "applied": False,
            "reason": "unreviewed_source_excerpts_not_allowed",
        }

    try:
        max_candidates = _positive_int(settings, "maxCandidates", 5, 5)
        max_statements = _positive_int(settings, "maxStatementsPerCard", 3, 5)
        max_chars = _positive_int(settings, "maxCharsPerCard", 360, 2000)
        max_total_chars = _positive_int(settings, "maxTotalChars", 2400, 10000)
        if not candidate_slots or len(candidate_slots) > max_candidates:
            raise CardPackError("candidate count exceeds the card-hint contract")
        pack = load_card_pack(
            settings.get("cardsFile", ""),
            settings.get("manifestFile", ""),
            str(settings.get("expectedManifestSha256") or ""),
            str(settings.get("expectedCardsSha256") or ""),
            int(settings.get("expectedSpeciesCount", EXPECTED_SPECIES_COUNT)),
            str(
                settings.get("expectedSpeciesSetSha256")
                or EXPECTED_SPECIES_SET_SHA256
            ),
        )
    except (CardPackError, OSError, TypeError, ValueError) as error:
        return None, {
            "mode": mode,
            "applied": False,
            "reason": "card_pack_unavailable",
            "errorType": type(error).__name__,
        }

    prepared = []
    missing = []
    allow_incomplete = bool(settings.get("allowIncompleteCandidateCoverage", False))
    for slot in candidate_slots:
        stable_id = str(slot.get("stableId") or "")
        card = pack["cardsByStableId"].get(stable_id)
        if card is not None and card.get("scientificName") != slot.get("scientificName"):
            card = None
        options = (
            _eligible_sentences(card, mode, max_statements, max_chars)
            if card is not None
            else []
        )
        if not options:
            missing.append(stable_id)
            if allow_incomplete:
                prepared.append({
                    "candidateRank": slot["rank"],
                    "stableId": stable_id,
                    "scientificName": slot.get("scientificName", ""),
                    "commonName": slot.get("commonName", ""),
                    "sourceMode": card.get("sourceMode", "unavailable") if card else "unavailable",
                    "source": _citation_source(card.get("source")) if card else {},
                    "options": [],
                })
            continue
        prepared.append({
            "candidateRank": slot["rank"],
            "stableId": stable_id,
            "scientificName": card["scientificName"],
            "commonName": card.get("commonName", ""),
            "sourceMode": card["sourceMode"],
            "source": _citation_source(card.get("source")),
            "options": options,
        })
    if missing and not allow_incomplete:
        return None, {
            "mode": mode,
            "applied": False,
            "reason": "incomplete_candidate_coverage",
            "missingCandidateCount": len(missing),
        }

    hints = {
        "policy": (
            "human_approved_identification_cards"
            if mode == "approved"
            else "unreviewed_verbatim_source_excerpts"
        ),
        "candidateHints": [
            {
                "candidateRank": item["candidateRank"],
                "stableId": item["stableId"],
                "sourceMode": item["sourceMode"],
                "statements": (
                    [_prompt_statement(item["options"][0])]
                    if item["options"]
                    else []
                ),
            }
            for item in prepared
        ],
    }
    if _serialized_chars(hints) > max_total_chars:
        return None, {
            "mode": mode,
            "applied": False,
            "reason": "prompt_budget_too_small_for_complete_coverage",
        }

    for statement_index in range(1, max_statements):
        for item, hint in zip(prepared, hints["candidateHints"]):
            if statement_index >= len(item["options"]):
                continue
            candidate = item["options"][statement_index]
            hint["statements"].append(_prompt_statement(candidate))
            if _serialized_chars(hints) > max_total_chars:
                hint["statements"].pop()

    prompt_chars = _serialized_chars(hints)
    citations = []
    for item, hint in zip(prepared, hints["candidateHints"]):
        options_by_text = {option["text"]: option for option in item["options"]}
        citations.append({
            "candidateRank": item["candidateRank"],
            "stableId": item["stableId"],
            "scientificName": item["scientificName"],
            "commonName": item["commonName"],
            "source": item["source"],
            "statements": [
                {
                    "feature": statement["feature"],
                    "text": statement["text"],
                    "heading": options_by_text[statement["text"]]["heading"],
                    "sentenceSha256": options_by_text[statement["text"]]["sentenceSha256"],
                }
                for statement in hint["statements"]
            ],
        })
    return hints, {
        "mode": mode,
        "applied": True,
        "reason": (
            "partial_candidate_coverage"
            if missing
            else "complete_candidate_coverage"
        ),
        "missingCandidateCount": len(missing),
        "missingStableIds": missing,
        "candidateCount": len(prepared),
        "statementCount": sum(
            len(item["statements"]) for item in hints["candidateHints"]
        ),
        "promptChars": prompt_chars,
        "cardsSha256": pack["cardsSha256"],
        "citations": citations,
    }
