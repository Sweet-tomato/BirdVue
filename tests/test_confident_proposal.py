"""Confident open-set proposal contract tests (no model / no Doubao calls)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import doubao_report  # noqa: E402

SETTINGS = {
    "allowConfidentProposal": True,
    "proposalMinimumVisibleFeatures": 2,
    "proposalMinimumQuality": 50,
    "proposalConfidenceCap": 0.5,
    "proposalOverrideCandidate": True,
    "proposalOverrideMinEvidence": 3,
    "proposalOverrideConfidenceCap": 0.4,
    "proposalGeoBypassMinEvidence": 3,
    "maximumProposals": 1,
}

SLOTS = [
    {"rank": 1, "stableId": "avibase-B483CC1C", "scientificName": "Acridotheres grandis", "family": "Sturnidae"},
    {"rank": 2, "stableId": "avibase-1EFA9BD0", "scientificName": "Lophura swinhoii", "family": "Phasianidae"},
]

REPORT = {
    "canIdentify": True,
    "bird": True,
    "animal": True,
    "categories": "Acridotheres grandis",
    "name": "Great Myna",
    "alias": "",
    "confidence": 0.92,
}

JUNGLE = {
    "scientificName": "Acridotheres fuscus",
    "commonName": "Jungle Myna",
    "observedFeatureEvidence": ["white wing patch", "yellow bill", "dark underparts"],
    "contradictedCandidateRanks": [1],
}


def review(decision="candidate", rank=1, proposal=None, features=None):
    payload = {
        "decision": decision,
        "candidateRank": rank,
        "selectedStableId": SLOTS[rank - 1]["stableId"] if decision == "candidate" and rank else None,
        "notes": "",
    }
    if proposal is not None:
        payload["outsideCandidates"] = [proposal]
    return payload


class ConfidentProposalTests(unittest.TestCase):
    def test_candidate_override_by_same_family_proposal(self):
        rep = dict(REPORT)
        rev = review("candidate", 1, JUNGLE)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, SETTINGS)
        self.assertTrue(status["approved"])
        self.assertEqual(status["reason"], "confident_proposal_overrode_candidate")
        self.assertTrue(status["overrodeCandidate"])
        self.assertEqual(rep["categories"], "Acridotheres fuscus")
        self.assertEqual(rep["name"], "Jungle Myna")
        self.assertLessEqual(rep["confidence"], 0.4)
        self.assertEqual(rev["openSetProposal"]["overrodeCandidate"]["stableId"], "avibase-B483CC1C")

    def test_candidate_kept_when_family_differs(self):
        rep = dict(REPORT)
        proposal = dict(JUNGLE, scientificName="Pica serica", commonName="Oriental Magpie")
        rev = review("candidate", 1, proposal)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, SETTINGS)
        self.assertFalse(status["approved"])
        self.assertIn(
            "proposal_family_differs_from_selected_candidate",
            [item["reason"] for item in status["rejections"]],
        )
        self.assertEqual(rep["categories"], "Acridotheres grandis")

    def test_candidate_kept_when_selected_rank_not_contradicted(self):
        rep = dict(REPORT)
        proposal = dict(JUNGLE, contradictedCandidateRanks=[2])
        rev = review("candidate", 1, proposal)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, SETTINGS)
        self.assertFalse(status["approved"])
        self.assertIn(
            "selected_candidate_not_contradicted",
            [item["reason"] for item in status["rejections"]],
        )

    def test_candidate_kept_when_evidence_too_thin(self):
        rep = dict(REPORT)
        proposal = dict(JUNGLE, observedFeatureEvidence=["white wing patch", "yellow bill"])
        rev = review("candidate", 1, proposal)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, SETTINGS)
        self.assertFalse(status["approved"])
        self.assertIn(
            "insufficient_visible_feature_evidence",
            [item["reason"] for item in status["rejections"]],
        )

    def test_reject_all_proposal_accepted_outside_family(self):
        rep = dict(REPORT, canIdentify=False, categories="Unknown", name="Unknown Bird", confidence=0.3)
        proposal = {
            "scientificName": "Chloebia gouldiae",
            "commonName": "Gouldian Finch",
            "observedFeatureEvidence": ["red face", "blue nape"],
            "contradictedCandidateRanks": [1, 2],
        }
        rev = review("reject_all", None, proposal)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, SETTINGS)
        self.assertTrue(status["approved"])
        self.assertEqual(status["reason"], "confident_proposal_accepted")
        self.assertFalse(status["overrodeCandidate"])
        self.assertEqual(rep["categories"], "Chloebia gouldiae")

    def test_unresolvable_name_rejected(self):
        rep = dict(REPORT)
        proposal = {
            "scientificName": "Nonexistent birbname",
            "commonName": "Nonexistent bird",
            "observedFeatureEvidence": ["white wing patch", "yellow bill", "dark underparts"],
            "contradictedCandidateRanks": [1],
        }
        rev = review("candidate", 1, proposal)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, SETTINGS)
        self.assertFalse(status["approved"])
        self.assertIn(
            "species_not_in_registry",
            [item["reason"] for item in status["rejections"]],
        )

    def test_override_disabled_keeps_candidate(self):
        settings = dict(SETTINGS, proposalOverrideCandidate=False)
        rep = dict(REPORT)
        rev = review("candidate", 1, JUNGLE)
        status = doubao_report.accept_confident_proposal(rep, rev, SLOTS, 95, settings)
        self.assertFalse(status["approved"])
        self.assertEqual(status["reason"], "first_pass_candidate_kept")

    def test_geo_filter_bypass_marks_proposal(self):
        rep = dict(REPORT, canIdentify=False, categories="Unknown", name="Unknown Bird", confidence=0.3)
        proposal = {
            "scientificName": "Chloebia gouldiae",
            "commonName": "Gouldian Finch",
            "observedFeatureEvidence": ["red face", "blue nape", "green back"],
            "contradictedCandidateRanks": [1, 2],
        }
        rev = review("reject_all", None, proposal)
        status = doubao_report.accept_confident_proposal(
            rep, rev, SLOTS, 95, SETTINGS, geo_filter_applied=True
        )
        self.assertTrue(status["approved"])
        self.assertTrue(status["geoFilterBypassed"])
        self.assertTrue(rev["openSetProposal"]["geoFilterBypassed"])

    def test_geo_filter_bypass_needs_more_evidence(self):
        rep = dict(REPORT, canIdentify=False, categories="Unknown", name="Unknown Bird", confidence=0.3)
        proposal = {
            "scientificName": "Chloebia gouldiae",
            "commonName": "Gouldian Finch",
            "observedFeatureEvidence": ["red face", "blue nape"],
            "contradictedCandidateRanks": [1],
        }
        rev = review("reject_all", None, proposal)
        status = doubao_report.accept_confident_proposal(
            rep, rev, SLOTS, 95, SETTINGS, geo_filter_applied=True
        )
        self.assertFalse(status["approved"])
        self.assertIn(
            "insufficient_visible_feature_evidence",
            [item["reason"] for item in status["rejections"]],
        )


if __name__ == "__main__":
    unittest.main()
