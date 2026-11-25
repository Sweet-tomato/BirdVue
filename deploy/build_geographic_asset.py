#!/usr/bin/env python3
"""构建地理先验数据资产（geographicPrior.dataFile）。

用法：
  python deploy/build_geographic_asset.py <source.json> <output.json> --taxonomy-version AviList-v2025b

source.json 结构（与目标 schema 一致，taxonomyVersion 可省略由参数注入）：
{
  "source": {"name": "eBird basic dataset 2024", "license": "CC-BY-4.0 摘要统计",
             "commercialUseApproved": true},
  "cells": [
    {"id": "N22.5_E114.0",
     "bounds": [113.75, 22.25, 114.25, 22.75],
     "species": {
       "<stableId>": {"yearRound": 0.95, "months": {"5": 0.9}}
     }}
  ]
}

脚本职责：注入 taxonomyVersion → 校验（validate_asset）→ cells 按 id 排序规范化 →
写出后计算 SHA-256 并打印 geographicPrior 配置片段。权重语义见 pipeline/geographic_prior.py：
curated support weights，NOT occurrence probabilities；缺失=未知≠不存在。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from geographic_prior import validate_asset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="输入源 JSON 文件")
    parser.add_argument("output", help="输出资产 JSON 文件")
    parser.add_argument("--taxonomy-version", required=True,
                        help="必须与 pipeline_config.json taxonomy.version 一致")
    args = parser.parse_args()

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    asset = {
        "schemaVersion": 1,
        "crs": "EPSG:4326",
        "taxonomyVersion": args.taxonomy_version,
        "source": source["source"],
        "cells": sorted(source["cells"], key=lambda cell: str(cell.get("id", ""))),
    }
    validate_asset(asset, args.taxonomy_version)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asset, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    output_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    print(f"written: {output_path} ({len(asset['cells'])} cells)")
    print(f"sha256: {digest}")
    print("pipeline_config.json 配置片段：")
    print(json.dumps({
        "geographicPrior": {
            "enabled": True,
            "maxBonus": 0.02,
            "dataFile": f"../models/geographic/{output_path.name}",
            "expectedSha256": digest,
        }
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
