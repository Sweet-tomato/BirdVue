# -*- coding: utf-8 -*-
"""Equivalence test for the vectorized YOLOX decode (_decode_yolox).

Reimplements the original per-cell decode loop and compares candidate lists
on fixed pseudo-random rows, including NaN/inf edge cases.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import bird_cli as bc  # noqa: E402


def reference_decode(rows, width, height, scale):
    """The pre-vectorization per-cell decode, kept verbatim for comparison."""
    candidates = []
    index = 0
    for stride in (8, 16, 32):
        size = bc.DETECTION_INPUT_SIZE // stride
        for grid_y in range(size):
            for grid_x in range(size):
                row = rows[index]
                index += 1
                score = float(row[4] * row[5 + bc.BIRD_CLASS_ID])
                if score < bc.DETECTION_THRESHOLD:
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
    return candidates


def rows_total():
    return sum((bc.DETECTION_INPUT_SIZE // s) ** 2 for s in (8, 16, 32))


def make_rows(seed=0, with_nan=False):
    rng = np.random.default_rng(seed)
    rows = (rng.standard_normal((rows_total(), 85)) * 3.0).astype(np.float32)
    rows[:, 4] = np.abs(rows[:, 4]) * 0.5 + 0.02  # objectness in a sane range
    if with_nan:
        rows[10, 0] = np.nan
        rows[11, 19] = np.nan
        rows[12, 2] = np.inf
    return rows


def assert_same_candidates(testcase, expected, actual):
    testcase.assertEqual(len(expected), len(actual))
    for exp, act in zip(expected, actual):
        if math.isnan(exp["score"]) or math.isnan(act["score"]):
            testcase.assertTrue(math.isnan(exp["score"]) and math.isnan(act["score"]))
        else:
            testcase.assertEqual(exp["score"], act["score"])
        testcase.assertEqual(len(exp["box"]), len(act["box"]))
        for e, a in zip(exp["box"], act["box"]):
            if math.isnan(e) or math.isnan(a):
                testcase.assertTrue(math.isnan(e) and math.isnan(a))
            else:
                testcase.assertEqual(e, a)


class YoloxDecodeEquivalenceTest(unittest.TestCase):
    def test_random_rows_bit_identical(self):
        rows = make_rows()
        expected = reference_decode(rows, 1920, 1080, 416 / 1920)
        actual = bc._decode_yolox(rows, 1920, 1080, 416 / 1920)
        assert_same_candidates(self, expected, actual)

    def test_nan_and_inf_rows_bit_identical(self):
        rows = make_rows(with_nan=True)
        expected = reference_decode(rows, 640, 480, 416 / 640)
        actual = bc._decode_yolox(rows, 640, 480, 416 / 640)
        assert_same_candidates(self, expected, actual)

    def test_empty_scores_produce_no_candidates(self):
        rows = np.zeros((rows_total(), 85), dtype=np.float32)
        self.assertEqual(bc._decode_yolox(rows, 100, 100, 1.0), [])


if __name__ == "__main__":
    unittest.main()
