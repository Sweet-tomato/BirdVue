"""Single-image fine-grained helpers for the bird classifier.

The helpers in this module are intentionally model-agnostic.  They create
overlapping detail crops from the already detected bird view, combine detail
scores with the existing whole-bird score, and conservatively prevent a VLM
review from replacing strong agreement between independent visual branches.
"""

from __future__ import annotations

import copy

import numpy as np


def parse_settings(config):
    settings = dict(config.get("singleImageFineGrained") or {})
    enabled = bool(settings.get("enabled", False))
    mode = settings.get("mode", "shadow")
    if mode not in {"shadow", "rerank"}:
        raise ValueError("singleImageFineGrained.mode must be shadow or rerank")
    trigger_mode = settings.get("triggerMode", "always")
    if trigger_mode not in {"always", "view_conflict"}:
        raise ValueError(
            "singleImageFineGrained.triggerMode must be always or view_conflict"
        )
    crop_fraction = float(settings.get("cropFraction", 0.72))
    if not 0.5 <= crop_fraction < 1.0:
        raise ValueError("singleImageFineGrained.cropFraction must be in [0.5, 1.0)")
    detail_weight = float(settings.get("detailWeight", 0.18))
    if not 0.0 <= detail_weight <= 0.5:
        raise ValueError("singleImageFineGrained.detailWeight must be in [0.0, 0.5]")
    top_views = int(settings.get("topViews", 2))
    if not 1 <= top_views <= 4:
        raise ValueError("singleImageFineGrained.topViews must be between 1 and 4")
    minimum_side = int(settings.get("minimumSide", 128))
    if minimum_side < 32:
        raise ValueError("singleImageFineGrained.minimumSide must be at least 32")
    return {
        "enabled": enabled,
        "mode": mode,
        "triggerMode": trigger_mode,
        "cropFraction": crop_fraction,
        "detailWeight": detail_weight,
        "topViews": top_views,
        "minimumSide": minimum_side,
    }


def build_detail_views(image, *, crop_fraction=0.72, minimum_side=128):
    """Return four overlapping, orientation-neutral crops of one bird view."""

    width, height = image.size
    if min(width, height) < int(minimum_side):
        return []
    crop_width = max(1, round(width * float(crop_fraction)))
    crop_height = max(1, round(height * float(crop_fraction)))
    right = width - crop_width
    bottom = height - crop_height
    definitions = (
        ("upper_left", 0, 0),
        ("upper_right", right, 0),
        ("lower_left", 0, bottom),
        ("lower_right", right, bottom),
    )
    views = []
    for name, left, top in definitions:
        box = (left, top, left + crop_width, top + crop_height)
        views.append({"name": name, "box": list(box), "image": image.crop(box)})
    return views


def aggregate_scores(base_scores, detail_scores, *, detail_weight=0.18, top_views=2):
    """Blend a whole-bird score with the strongest consistent detail views."""

    base = np.asarray(base_scores, dtype=np.float32)
    details = np.asarray(detail_scores, dtype=np.float32)
    if base.ndim != 1:
        raise ValueError("base_scores must be one-dimensional")
    if details.ndim != 2 or details.shape[0] != base.shape[0]:
        raise ValueError("detail_scores must have shape (classes, views)")
    if details.shape[1] == 0:
        return base.copy()
    view_count = min(int(top_views), details.shape[1])
    strongest = np.partition(details, details.shape[1] - view_count, axis=1)[:, -view_count:]
    detail_score = strongest.mean(axis=1)
    weight = float(detail_weight)
    return ((1.0 - weight) * base + weight * detail_score).astype(np.float32, copy=False)


def apply_visual_consensus_guard(report, doubao_meta, local_result, config):
    """Downgrade a conflicting VLM species when visual branches strongly agree.

    This guard never promotes a species.  It only reduces a conflicting VLM
    species decision to genus (or Unknown) while preserving the original review
    record for audit.
    """

    settings = dict(config.get("visualConsensusGuard") or {})
    status = {
        "enabled": bool(settings.get("enabled", False)),
        "applied": False,
        "reason": "disabled",
    }
    if not status["enabled"]:
        return report, doubao_meta, status
    review = doubao_meta.get("review") if isinstance(doubao_meta, dict) else None
    if not isinstance(review, dict) or review.get("decision") != "candidate":
        status["reason"] = "review_not_species_candidate"
        return report, doubao_meta, status
    escape = doubao_meta.get("openSetEscape") or {}
    accepted_escape_ids = {
        row.get("stableId") for row in escape.get("acceptedCandidates") or []
    }
    if (
        bool((config.get("openSetEscape") or {}).get("enabled", False))
        and escape.get("approved") is True
        and review.get("selectedStableId") in accepted_escape_ids
    ):
        status["reason"] = "validated_open_set_escape"
        status["reviewSelectedStableId"] = review.get("selectedStableId")
        return report, doubao_meta, status

    global_bioclip = local_result.get("globalBioCLIP", {})
    global_rows = global_bioclip.get("regionAdjustedTop5") or []
    prototype_rows = local_result.get("imagePrototypeRetrieval", {}).get("topCandidates") or []
    if not global_rows:
        status["reason"] = "visual_branch_unavailable"
        return report, doubao_meta, status
    global_top = global_rows[0]
    consensus_id = global_top.get("stableId")
    if not consensus_id:
        status["reason"] = "visual_branch_unavailable"
        return report, doubao_meta, status

    focus = local_result.get("birdFocus") or {}
    branch_ids = {
        "enhanced": consensus_id,
        "original": (focus.get("originalTop1") or {}).get("stableId"),
        "focus": (focus.get("focusTop1") or {}).get("stableId"),
    }
    minimum_support = int(settings.get("minimumPrototypeSupport", 2))
    prototype_top = prototype_rows[0] if prototype_rows else None
    prototype_eligible = bool(
        prototype_top
        and int(prototype_top.get("prototypeSupportCount") or 0) >= minimum_support
    )
    branch_ids["prototype"] = (
        prototype_top.get("stableId") if prototype_eligible else None
    )

    if bool(settings.get("requirePrototypeAgreement", True)):
        if not prototype_top:
            status["reason"] = "visual_branch_unavailable"
            return report, doubao_meta, status
        if not prototype_eligible:
            status["reason"] = "prototype_support_too_small"
            return report, doubao_meta, status
        if branch_ids["prototype"] != consensus_id:
            status["reason"] = "text_prototype_disagree"
            return report, doubao_meta, status

    if bool(settings.get("requireOriginalFocusAgreement", True)):
        original_id = branch_ids["original"]
        focus_id = branch_ids["focus"]
        if not original_id or not focus_id or original_id != consensus_id or focus_id != consensus_id:
            status["reason"] = "original_focus_not_in_consensus"
            return report, doubao_meta, status
    else:
        minimum_votes = int(settings.get("minimumVisualVotes", 2))
        matching_branches = [
            name for name, stable_id in branch_ids.items()
            if stable_id == consensus_id
        ]
        status["matchingBranches"] = matching_branches
        if len(matching_branches) < minimum_votes:
            status["reason"] = "insufficient_visual_votes"
            return report, doubao_meta, status

    selected_id = review.get("selectedStableId")
    status.update(
        {
            "consensusStableId": consensus_id,
            "reviewSelectedStableId": selected_id,
        }
    )
    if not selected_id or selected_id == consensus_id:
        status["reason"] = "review_matches_visual_consensus"
        return report, doubao_meta, status

    guarded_report = copy.deepcopy(report)
    genus = str(global_top.get("genus") or "").strip()
    fallback_level = settings.get("fallbackLevel", "genus")
    guarded_report["canIdentify"] = False
    guarded_report["bird"] = True
    guarded_report["animal"] = True
    guarded_report["alias"] = ""
    guarded_report["confidence"] = min(0.4, float(guarded_report.get("confidence") or 0.0))
    if fallback_level == "genus" and genus:
        guarded_report["categories"] = genus
        guarded_report["name"] = f"Uncertain {genus} species"
    else:
        guarded_report["categories"] = "Unknown"
        guarded_report["name"] = "Unknown Bird"
    guarded_report["describe"] = (
        "The VLM species choice conflicted with agreement between independent "
        "visual views, so the result was conservatively reduced."
    )
    guarded_meta = copy.deepcopy(doubao_meta)
    status.update(
        {
            "applied": True,
            "reason": "vlm_species_conflicted_with_visual_consensus",
            "originalCategories": report.get("categories"),
            "finalCategories": guarded_report["categories"],
        }
    )
    return guarded_report, guarded_meta, status
