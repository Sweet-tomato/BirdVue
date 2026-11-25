#!/usr/bin/env python3
"""从 GBIF 拉取鸟类分面统计，生成 geographicPrior 源 JSON（供 build_geographic_asset.py 使用）。

用法：
  python deploy/fetch_gbif_distribution.py --step 3 --out /tmp/species_support_source.json

做法：按经纬网格逐格查询 GBIF occurrence/search（class Aves, taxonKey=212），
limit=0 + facet=scientificName 拿到该格内每个物种的观测记录数，再按“同物种跨格最大
记录数”归一化并开方，得到 0..1 的 curated support weight（不是出现概率）。
物种名与本地 AviList 注册表（stableId）匹配，匹配不上直接跳过并统计。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.gbif.org/v1/occurrence/search"
AVES_TAXON_KEY = 212
BINOMIAL = re.compile(r"^([A-Z][a-z]+ [a-z][a-z-]*)")


def load_stable_ids() -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    count = 0
    for name in ("avilist_v2025b_species.json", "avilist_v2025b_supplemental_species.json"):
        path = ROOT / "models" / "bioclip25" / "birds" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["species"]:
            count += 1
            stable_id = item.get("stableId")
            for key in ("scientificName", "modelScientificName"):
                value = str(item.get(key) or "").strip()
                if value and stable_id and value not in mapping:
                    mapping[value] = stable_id
    return mapping, count


def binomial(name: str) -> str | None:
    match = BINOMIAL.match(name.strip())
    return match.group(1) if match else None


def resolve_via_gbif(name: str) -> str | None:
    """把 GBIF 上的 AviList 新名（如 Schoeniclus aureolus）解析回骨架物种名（Emberiza aureola）。"""
    query = urllib.parse.urlencode({"name": name, "kingdom": "Animalia"})
    try:
        with urllib.request.urlopen(f"https://api.gbif.org/v1/species/match?{query}", timeout=30) as response:
            match = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - 解析失败按未匹配处理
        return None
    if match.get("class") != "Aves" or (match.get("confidence") or 0) < 90:
        return None
    return match.get("species") or match.get("canonicalName")


def fetch_cell(west: float, south: float, east: float, north: float) -> list[dict]:
    polygon = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    query = urllib.parse.urlencode({
        "geometry": polygon,
        "taxonKey": AVES_TAXON_KEY,
        "limit": 0,
        "facet": "scientificName",
        "facetLimit": 5000,
        "facetMincount": 1,
    })
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            facets = payload.get("facets") or []
            return facets[0]["counts"] if facets else []
        except Exception as error:  # noqa: BLE001 - retry any transport/parse error
            last_error = error
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"GBIF cell query failed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--west", type=float, default=70.0)
    parser.add_argument("--east", type=float, default=140.0)
    parser.add_argument("--south", type=float, default=15.0)
    parser.add_argument("--north", type=float, default=55.0)
    parser.add_argument("--step", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pause", type=float, default=0.3, help="两次请求间隔秒数")
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="格子右上界内缩，避免落在 3° 整倍数上的坐标同时命中相邻两格")
    args = parser.parse_args()

    mapping, registry_size = load_stable_ids()
    lon_steps = int(round((args.east - args.west) / args.step))
    lat_steps = int(round((args.north - args.south) / args.step))

    raw: dict[str, dict[str, int]] = {}
    unmatched = Counter()
    resolved_cache: dict[str, str | None] = {}
    species_max: dict[str, int] = defaultdict(int)

    def lookup(binomial_name: str) -> str | None:
        stable_id = mapping.get(binomial_name)
        if stable_id:
            return stable_id
        if binomial_name not in resolved_cache:
            canonical = resolve_via_gbif(binomial_name)
            resolved_cache[binomial_name] = mapping.get(canonical) if canonical else None
        return resolved_cache[binomial_name]
    started = time.time()
    total = lon_steps * lat_steps

    for row in range(lat_steps):
        for col in range(lon_steps):
            west = args.west + col * args.step
            south = args.south + row * args.step
            counts = fetch_cell(west, south, west + args.step, south + args.step)
            cell_species: dict[str, int] = {}
            for entry in counts:
                name = binomial(entry.get("name") or "")
                if not name:
                    continue
                stable_id = lookup(name)
                if not stable_id:
                    unmatched[name] += entry.get("count") or 0
                    continue
                cell_species[stable_id] = cell_species.get(stable_id, 0) + int(entry.get("count") or 0)
            if cell_species:
                cell_id = f"W{west:g}_S{south:g}"
                raw[cell_id] = {
                    "id": cell_id,
                    "bounds": [west, south,
                               round(west + args.step - args.epsilon, 6),
                               round(south + args.step - args.epsilon, 6)],
                    "species_counts": cell_species,
                }
                for stable_id, value in cell_species.items():
                    species_max[stable_id] = max(species_max[stable_id], value)
            done = row * lon_steps + col + 1
            if done % 20 == 0 or done == total:
                print(f"[{done}/{total}] cells_with_data={len(raw)} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)
            time.sleep(args.pause)

    cells = []
    species_total = 0
    for cell in raw.values():
        species = {}
        for stable_id, count in cell.pop("species_counts").items():
            support = math.sqrt(count / species_max[stable_id]) if species_max[stable_id] else 0.0
            support = round(min(1.0, max(0.0, support)), 4)
            if support > 0:
                species[stable_id] = {"yearRound": support}
        if species:
            species_total += len(species)
            cells.append({**cell, "species": species})

    source = {
        "source": {
            "name": f"GBIF occurrence facet summary (Aves, {args.step:g}° grid, "
                    f"{args.west:g}-{args.east:g}E {args.south:g}-{args.north:g}N)",
            "license": "CC BY 4.0（GBIF 聚合记录数；逐条记录许可见 GBIF）",
            "commercialUseApproved": True,
        },
        "cells": cells,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(source, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"written: {output}")
    print(f"cells={len(cells)} cell_species_entries={species_total} "
          f"distinct_species={len(species_max)} registry={registry_size} "
          f"unmatched_names={len(unmatched)}")
    print("top unmatched:", unmatched.most_common(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
