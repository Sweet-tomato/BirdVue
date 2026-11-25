"""Golden decision-contract tests: no model inference, pure logic safety net.

Covers assess_risk branch matrix and the unified finalize_decision (A1).
Any refactor must keep these decisions byte-identical.
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import bird_cli


def base_local_result():
    return {
        "qualityScore": 85,
        "birdFocus": {
            "box": [10, 10, 100, 100],
            "score": 0.95,
            "clippedEdges": 0,
            "viewConsistency": "agree",
            "birdOccupancy": 0.6,
        },
        "globalBioCLIP": {
            "regionAdjustedTop5": [
                {"adjustedSimilarity": 0.60, "family": "Paridae"},
                {"adjustedSimilarity": 0.50, "family": "Paridae"},
                {"adjustedSimilarity": 0.40, "family": "Paridae"},
            ],
            "supplementalCoverage": {"conflict": False},
        },
        "sceneRouter": {"route": "bird"},
    }


class AssessRiskGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = bird_cli.load_pipeline_config(
            Path(__file__).resolve().parent.parent / "config" / "pipeline_config.json"
        )

    def risk(self, mutate=None, config=None):
        local = base_local_result()
        if mutate:
            mutate(local)
        return bird_cli.assess_risk(local, config or self.config)

    def test_direct_accept_when_margin_sufficient(self):
        risk = self.risk()
        self.assertFalse(risk["needsReview"], risk["blockingReasons"])
        self.assertEqual(risk["blockingReasons"], [])
        self.assertIn("bird_large_in_frame", risk["auxiliarySignals"])
        self.assertAlmostEqual(risk["metrics"]["candidateMargin"], 0.10, places=5)

    def test_margin_below_threshold_blocks(self):
        def mutate(local):
            local["globalBioCLIP"]["regionAdjustedTop5"][1]["adjustedSimilarity"] = 0.57
        risk = self.risk(mutate)
        self.assertTrue(risk["needsReview"])
        self.assertIn("candidate_margin_below_calibrated_threshold", risk["blockingReasons"])

    def test_uncalibrated_margin_threshold_blocks(self):
        config = copy.deepcopy(self.config)
        config["localDecision"]["marginThreshold"] = None
        risk = self.risk(config=config)
        self.assertIn("margin_threshold_not_calibrated", risk["blockingReasons"])

    def test_scene_uncertain_blocks(self):
        risk = self.risk(lambda l: l["sceneRouter"].update(route="uncertain"))
        self.assertIn("scene_uncertain", risk["blockingReasons"])

    def test_non_bird_route_blocks(self):
        risk = self.risk(lambda l: l["sceneRouter"].update(route="non_bird"))
        self.assertIn("scene_non_bird", risk["blockingReasons"])

    def test_detection_unreliable_blocks(self):
        risk = self.risk(lambda l: l["birdFocus"].update(score=0.05))
        self.assertIn("detection_unreliable", risk["blockingReasons"])

    def test_no_detection_box_blocks(self):
        risk = self.risk(lambda l: l["birdFocus"].update(box=None))
        self.assertIn("target_not_detected", risk["blockingReasons"])

    def test_clipped_edges_block(self):
        risk = self.risk(lambda l: l["birdFocus"].update(clippedEdges=3))
        self.assertIn("bird_incomplete_or_clipped", risk["blockingReasons"])

    def test_view_conflict_blocks(self):
        risk = self.risk(lambda l: l["birdFocus"].update(viewConsistency="conflict"))
        self.assertIn("original_focus_conflict", risk["blockingReasons"])

    def test_cross_family_supplemental_conflict_blocks(self):
        # 生产配置 blockOnCrossFamilyConflict=false：冲突不阻塞（冻结现状）。
        risk = self.risk(lambda l: l["globalBioCLIP"].update(supplementalCoverage={"conflict": True}))
        self.assertNotIn("taxonomy_coverage_cross_family_conflict", risk["blockingReasons"])
        # 开关为 true 时必须阻塞（锁定分支语义）。
        config = copy.deepcopy(self.config)
        config["supplementalCandidates"]["blockOnCrossFamilyConflict"] = True
        risk = self.risk(lambda l: l["globalBioCLIP"].update(supplementalCoverage={"conflict": True}), config=config)
        self.assertIn("taxonomy_coverage_cross_family_conflict", risk["blockingReasons"])

    def test_single_candidate_row_blocks_margin(self):
        def mutate(local):
            local["globalBioCLIP"]["regionAdjustedTop5"] = [
                {"adjustedSimilarity": 0.60, "family": "Paridae"}
            ]
        risk = self.risk(mutate)
        self.assertTrue(risk["needsReview"])
        self.assertIsNone(risk["metrics"]["candidateMargin"])

    def test_cross_family_top3_is_auxiliary_only(self):
        def mutate(local):
            local["globalBioCLIP"]["regionAdjustedTop5"][1]["family"] = "Corvidae"
        risk = self.risk(mutate)
        self.assertFalse(risk["needsReview"], risk["blockingReasons"])
        self.assertIn("top3_cross_family", risk["auxiliarySignals"])


class FinalizeDecisionGolden(unittest.TestCase):
    def call(self, report=None, doubao=None, route="bird", source="doubao_risk_review"):
        return bird_cli.finalize_decision(
            report if report is not None else {"bird": True},
            doubao if doubao is not None else {"status": "ok", "review": {"decision": "candidate"}},
            {"sceneRouter": {"route": route}},
            source,
        )

    def test_local_direct_confirms_bird(self):
        out = self.call(source="local_direct", doubao={"status": "skipped"})
        self.assertEqual(out["meta"]["presenceStatus"], "bird_confirmed")
        self.assertEqual(out["meta"]["decisionSource"], "local_direct")

    def test_doubao_no_bird(self):
        out = self.call(doubao={"status": "ok", "review": {"decision": "no_bird"}})
        self.assertEqual(out["meta"]["presenceStatus"], "non_bird_confirmed")

    def test_doubao_confirms_bird(self):
        out = self.call()
        self.assertEqual(out["meta"]["presenceStatus"], "bird_confirmed")

    def test_doubao_failure_bird_route_hint(self):
        out = self.call(doubao={"status": "error"})
        self.assertEqual(out["meta"]["presenceStatus"], "bird_hint_unconfirmed")

    def test_doubao_failure_non_bird_route_hint(self):
        out = self.call(doubao={"status": "error"}, route="non_bird")
        self.assertEqual(out["meta"]["presenceStatus"], "non_bird_hint_unconfirmed")

    def test_doubao_failure_uncertain_route(self):
        out = self.call(doubao={"status": "error"}, route="uncertain")
        self.assertEqual(out["meta"]["presenceStatus"], "uncertain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
