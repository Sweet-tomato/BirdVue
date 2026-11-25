# -*- coding: utf-8 -*-
"""Tests for the shared read-only bundle (no model weights required).

The real module imports open_clip / bird_cli / image_prototype_retrieval, none
of which exist in this lean mirror. We therefore extract the two small cache
functions from the source AST and execute them in an isolated namespace, so
the tests still exercise the exact production code.
"""
from __future__ import annotations

import ast
import builtins
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "pipeline" / "bioclip25_runtime.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def exec_functions(*names):
    wanted = [n for n in TREE.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(wanted) == len(names), (names, [n.name for n in wanted])
    ns = {
        "threading": threading,
        "_BUNDLE_CACHE": {},
        "_BUNDLE_LOCK": threading.Lock(),
    }
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, "<shared-bundle>", "exec"), ns)
    return ns


class CacheTests(unittest.TestCase):
    def test_factory_runs_once_for_the_same_key(self):
        ns = exec_functions("cached_bundle", "bundle_stats")
        calls = []

        def factory():
            calls.append(1)
            return {"matrix": np.ones((2, 2), dtype=np.float32)}

        first = ns["cached_bundle"](("cfg.json", False), factory)
        second = ns["cached_bundle"](("cfg.json", False), factory)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ns["bundle_stats"]()["entries"], 1)

    def test_different_keys_build_separate_bundles(self):
        ns = exec_functions("cached_bundle")
        a = ns["cached_bundle"](("a.json", False), lambda: {"v": 1})
        b = ns["cached_bundle"](("a.json", True), lambda: {"v": 2})
        self.assertEqual(a["v"], 1)
        self.assertEqual(b["v"], 2)

    def test_concurrent_callers_share_one_build(self):
        ns = exec_functions("cached_bundle")
        calls = []
        barrier = threading.Barrier(6)

        def factory():
            calls.append(1)
            return {"v": object()}

        out = []

        def run(_i):
            barrier.wait()
            out.append(ns["cached_bundle"](("cfg", False), factory))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(x is out[0] for x in out))


class FrozenArrayTests(unittest.TestCase):
    def test_source_freezes_both_text_matrices(self):
        freeze = [n for n in ast.walk(TREE)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "setflags"]
        self.assertEqual(len(freeze), 1)
        loop = [n for n in ast.walk(TREE)
                if isinstance(n, ast.For)
                and any(isinstance(x, ast.Call) and
                        isinstance(x.func, ast.Attribute) and
                        x.func.attr == "setflags" for x in ast.walk(n))]
        self.assertEqual(len(loop), 1)
        names = {e.value for e in loop[0].iter.elts
                 if isinstance(e, ast.Constant)}
        self.assertEqual(names, {"global_text", "supplemental_text"})

    def test_frozen_array_rejects_downstream_writes(self):
        matrix = np.ones((3, 4), dtype=np.float32)
        matrix.setflags(write=False)
        with self.assertRaises(ValueError):
            matrix[0, 0] = 9.0
        # Read-only consumers (matrix multiply) still work.
        self.assertEqual((matrix @ np.ones((4, 1), dtype=np.float32)).shape, (3, 1))


class RuntimeStructureTests(unittest.TestCase):
    def _function(self, name):
        return next(n for n in TREE.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_load_runtime_only_rebuilds_per_instance_objects(self):
        fn = self._function("load_runtime")
        self.assertIn("load_shared_bundle", ast.unparse(fn))
        self.assertIn("create_model_and_transforms", ast.unparse(fn))
        self.assertIn("readNetFromONNX", ast.unparse(fn))
        # Asset verification belongs to the shared bundle, not load_runtime.
        self.assertNotIn("verify_manifest_file", ast.unparse(fn))
        self.assertNotIn("apply_text_geometry", ast.unparse(fn))

    def test_every_name_used_by_load_runtime_is_defined(self):
        """Catches a forgotten bundle key after the refactor."""
        fn = self._function("load_runtime")
        stored = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Store)}
        stored |= {arg.arg for arg in fn.args.args}
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    stored.add((alias.asname or alias.name).split(".")[0])
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Load)}
        module_level = {"__file__"}
        for node in TREE.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                module_level.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_level.add(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    module_level.add((alias.asname or alias.name).split(".")[0])
        missing = loaded - stored - module_level - set(dir(builtins))
        self.assertEqual(missing, set(), "undefined names in load_runtime: %s" % missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
