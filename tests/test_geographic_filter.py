"""Hard geographic filter contract tests (no model inference)."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))

import geographic_prior  # noqa: E402

CONFIG = json.loads((REPO / "config" / "pipeline_config.json").read_text(encoding="utf-8"))
SETTINGS = dict(CONFIG.get("geographicPrior") or {})
ROOT = REPO / "config"
TAXONOMY = (CONFIG.get("taxonomy") or {}).get("version")


def observation(lat, lon):
    return {
        "enabled": True,
        "crs": "WGS84",
        "source": "device",
        "latitude": lat,
        "longitude": lon,
        "observedAt": "2026-11-25T09:00:00+08:00",
    }


class GeographicFilterTests(unittest.TestCase):
    def test_filter_mode_off_returns_none(self):
        settings = dict(SETTINGS, filterMode="off")
        ids, status = geographic_prior.allowed_species(
            settings, observation(39.9042, 116.4074), root=ROOT, taxonomy_version=TAXONOMY
        )
        self.assertIsNone(ids)
        self.assertEqual(status["reason"], "disabled")

    def test_missing_location_is_neutral(self):
        ids, status = geographic_prior.allowed_species(
            SETTINGS, None, root=ROOT, taxonomy_version=TAXONOMY
        )
        self.assertIsNone(ids)
        self.assertEqual(status["reason"], "missing_or_invalid_wgs84_location")

    def test_outside_coverage_is_neutral(self):
        ids, status = geographic_prior.allowed_species(
            SETTINGS, observation(0.0, 0.0), root=ROOT, taxonomy_version=TAXONOMY
        )
        self.assertIsNone(ids)
        self.assertEqual(status["reason"], "outside_coverage")

    def test_beijing_cell_returns_local_pool(self):
        ids, status = geographic_prior.allowed_species(
            SETTINGS, observation(39.9042, 116.4074), root=ROOT, taxonomy_version=TAXONOMY
        )
        self.assertIsNotNone(ids, status)
        self.assertTrue(status["applied"])
        self.assertEqual(status["reason"], "hard_filter")
        self.assertEqual(status["cellId"], "W115_S39")
        self.assertGreaterEqual(len(ids), int(SETTINGS.get("minPoolSpecies", 20)))
        self.assertIn("avibase-", next(iter(ids)))

    def test_pool_too_small_is_neutral(self):
        settings = dict(SETTINGS, minPoolSpecies=100000)
        ids, status = geographic_prior.allowed_species(
            settings, observation(39.9042, 116.4074), root=ROOT, taxonomy_version=TAXONOMY
        )
        self.assertIsNone(ids)
        self.assertEqual(status["reason"], "pool_too_small")

    def test_bad_sha_is_neutral(self):
        settings = dict(SETTINGS, expectedSha256="0" * 64)
        ids, status = geographic_prior.allowed_species(
            settings, observation(39.9042, 116.4074), root=ROOT, taxonomy_version=TAXONOMY
        )
        self.assertIsNone(ids)
        self.assertEqual(status["reason"], "invalid_or_unavailable_asset")


if __name__ == "__main__":
    unittest.main()
