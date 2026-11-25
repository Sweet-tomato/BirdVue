# -*- coding: utf-8 -*-
"""Prove identical-tile forward dedup is output-equivalent.

The real bird_cli imports heavy model dependencies, so the four pure functions
under test are extracted from the source with the AST and executed against a
deterministic stub of infer_birds. No model weights or third-party runtime are
needed beyond numpy.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "pipeline" / "bird_cli.py"
WANTED = {"positions", "overlap", "primary_detection", "detect_bird"}


class FakeTile:
    def __init__(self, box):
        self.box = tuple(box)
        self.size = (box[2] - box[0], box[3] - box[1])


class FakeImage:
    def __init__(self, size):
        self.size = size
        self.box = (0, 0, size[0], size[1])

    def crop(self, box):
        return FakeTile(box)


def load_namespace(dedup: bool, image_size=(640, 480)) -> dict:
    text = SRC.read_bytes().decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(text)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    assert len(nodes) == len(WANTED), "missing functions: %s" % (
        WANTED - {n.name for n in nodes})
    module = ast.Module(body=nodes, type_ignores=[])
    import numpy as np

    ns = {
        "np": np,
        "DETECTION_INPUT_SIZE": 416,
        "ZOOM_LEVELS": ((0.55, 0.20), (0.34, 0.20)),
        "FULL_FRAME_ACCEPT": 0.35,
        "DETECTION_THRESHOLD": 0.05,
        "NMS_THRESHOLD": 0.45,
        "BIRD_CLASS_ID": 14,
        "DETECT_DEDUP": dedup,
        "calls": [],
        "final_candidates": [],
    }

    def infer_birds(image, focus_net):
        ns["calls"].append(tuple(getattr(image, "box", (0, 0, 640, 480))))
        return [{"score": 0.06, "box": [10.0, 10.0, 30.0, 30.0]}]

    ns["infer_birds"] = infer_birds
    exec(compile(module, "<bird_cli_extract>", "exec"), ns)
    original = ns["primary_detection"]

    def spy(candidates, width, height):
        ns["final_candidates"] = list(candidates)
        return original(candidates, width, height)

    ns["primary_detection"] = spy
    ns["_image"] = FakeImage(image_size)
    return ns


class DetectDedupTest(unittest.TestCase):
    def test_identical_result_with_fewer_forwards_640x480(self):
        off = load_namespace(False)
        on = load_namespace(True)
        result_off = off["detect_bird"](off["_image"], object())
        result_on = on["detect_bird"](on["_image"], object())
        self.assertEqual(result_off, result_on, "best/passes/source changed")
        self.assertEqual(off["final_candidates"], on["final_candidates"],
                         "candidate list changed")
        self.assertEqual(len(set(off["calls"])), len(set(on["calls"])),
                         "unique tile geometry changed")
        self.assertEqual(len(off["calls"]), 9)
        self.assertEqual(len(on["calls"]), 5)
        self.assertLess(len(on["calls"]), len(off["calls"]))

    def test_different_tile_geometry_still_equivalent(self):
        size = (1920, 1080)
        off = load_namespace(False, size)
        on = load_namespace(True, size)
        self.assertEqual(
            off["detect_bird"](off["_image"], object()),
            on["detect_bird"](on["_image"], object()))
        self.assertEqual(off["final_candidates"], on["final_candidates"])
        self.assertLessEqual(len(on["calls"]), len(off["calls"]))

    def test_full_frame_accept_returns_before_tiles(self):
        ns = load_namespace(True)

        def strong(image, focus_net):
            ns["calls"].append(tuple(getattr(image, "box", (0, 0, 640, 480))))
            return [{"score": 0.9, "box": [5.0, 5.0, 25.0, 25.0]}]

        ns["infer_birds"] = strong
        best, passes, source = ns["detect_bird"](ns["_image"], object())
        self.assertEqual(source, "full")
        self.assertEqual(passes, 1)
        self.assertEqual(len(ns["calls"]), 1)
        self.assertIsNotNone(best)


if __name__ == "__main__":
    unittest.main()
