"""Shared pipeline config loading (pure, dependency-free)."""

import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.json"


def load_pipeline_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_path"] = str(path.resolve())
    return config
