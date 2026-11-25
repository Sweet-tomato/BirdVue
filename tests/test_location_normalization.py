"""Field-device location payload normalization tests (no model / no network)."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from api._common import normalize_location  # noqa: E402
import geographic_prior  # noqa: E402


class NormalizeLocationTests(unittest.TestCase):
    def test_plain_numeric_location(self):
        found = normalize_location({"location": {"latitude": 22.5, "longitude": 114.0}})
        self.assertEqual(found, {"latitude": 22.5, "longitude": 114.0, "source": "location"})

    def test_numeric_string_location(self):
        found = normalize_location({"location": {"latitude": "22.5828", "longitude": "113.93482"}})
        self.assertAlmostEqual(found["latitude"], 22.5828)
        self.assertAlmostEqual(found["longitude"], 113.93482)

    def test_gps_fence_json_string(self):
        fence = json.dumps({"latitude": 22.58303269793829, "longitude": 113.9343140112498, "radius": 50})
        found = normalize_location({"gpsFence": fence})
        self.assertAlmostEqual(found["latitude"], 22.58303269793829)
        self.assertAlmostEqual(found["longitude"], 113.9343140112498)
        self.assertEqual(found["source"], "gpsFence")

    def test_attachment_json_string_with_string_coords(self):
        attachment = json.dumps({
            "moonphase": "FULL_MOON",
            "latitude": "22.5828",
            "longitude": "113.93482",
            "batteryPower": 71,
            "timestamp": 1795491998414,
        })
        found = normalize_location({"attachment": attachment})
        self.assertAlmostEqual(found["latitude"], 22.5828)
        self.assertAlmostEqual(found["longitude"], 113.93482)
        self.assertEqual(found["source"], "attachment")

    def test_attachment_double_encoded(self):
        attachment = json.dumps(json.dumps({"latitude": "22.5", "longitude": "114.25"}))
        found = normalize_location({"attachment": attachment})
        self.assertAlmostEqual(found["latitude"], 22.5)
        self.assertAlmostEqual(found["longitude"], 114.25)

    def test_device_info_container(self):
        device_info = {
            "gpsFence": json.dumps({"latitude": 22.58, "longitude": 113.93, "radius": 50}),
            "attachment": json.dumps({"latitude": "1.0", "longitude": "2.0"}),
        }
        found = normalize_location({"deviceInfo": device_info})
        self.assertAlmostEqual(found["latitude"], 22.58)
        self.assertAlmostEqual(found["longitude"], 113.93)

    def test_location_wins_over_gps_fence(self):
        payload = {
            "location": {"latitude": 1.0, "longitude": 2.0},
            "gpsFence": json.dumps({"latitude": 9.0, "longitude": 9.0}),
        }
        found = normalize_location(payload)
        self.assertEqual(found["latitude"], 1.0)
        self.assertEqual(found["longitude"], 2.0)

    def test_missing_returns_none(self):
        self.assertIsNone(normalize_location({}))
        self.assertIsNone(normalize_location(None))

    def test_out_of_range_returns_none(self):
        self.assertIsNone(normalize_location({"location": {"latitude": 91, "longitude": 0}}))
        self.assertIsNone(normalize_location({"location": {"latitude": 0, "longitude": 181}}))

    def test_invalid_json_returns_none(self):
        self.assertIsNone(normalize_location({"gpsFence": "{not json"}))
        self.assertIsNone(normalize_location({"attachment": "null"}))


class ValidLocationStringTests(unittest.TestCase):
    def test_accepts_numeric_strings(self):
        location = {
            "enabled": True,
            "crs": "WGS84",
            "source": "device:gpsFence",
            "latitude": "22.5828",
            "longitude": "113.93482",
        }
        self.assertEqual(geographic_prior.valid_location(location), (22.5828, 113.93482))

    def test_rejects_garbage(self):
        location = {
            "enabled": True,
            "crs": "WGS84",
            "source": "device",
            "latitude": "abc",
            "longitude": 113.0,
        }
        self.assertIsNone(geographic_prior.valid_location(location))


if __name__ == "__main__":
    unittest.main()
