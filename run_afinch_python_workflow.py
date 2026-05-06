#!/usr/bin/env python
"""Run AFINCH-style routing and optional map generation in one Python command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _cfg_get(cfg: dict, key: str, default):
    if not cfg:
        return default
    if key in cfg:
        return cfg[key]
    alt = key.replace("_", "-")
    if alt in cfg:
        return cfg[alt]
    return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="Path to JSON config file")
    pre_args, _ = pre.parse_known_args(argv)

    cfg: dict = {}
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if not cfg_path.is_absolute():
            cfg_path = (BASE_DIR / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    p = argparse.ArgumentParser(description="Python AFINCH workflow: route + map")
    p.add_argument("--config", default=None, help="Path to JSON config file")
    p.add_argument(
        "--start-date",
        default=_cfg_get(cfg, "start_date", "2018-01-01"),
        help="Inclusive month start, YYYY-MM-01",
    )
    p.add_argument(
        "--end-date",
        default=_cfg_get(cfg, "end_date", "2018-01-01"),
        help="Inclusive month end, YYYY-MM-01",
    )
    p.add_argument(
        "--network-source",
        choices=["nhd_hr", "nhd_medium"],
        default=_cfg_get(cfg, "network_source", "nhd_medium"),
        help="Flowline network to route on",
    )
    p.add_argument(
        "--output-dir",
        default=_cfg_get(cfg, "output_dir", "output/nhd_medium_usgs_wam"),
        help="Directory for routed csv/gpkg and map output",
    )
    p.add_argument("--usgs-trust", type=float, default=float(_cfg_get(cfg, "usgs_trust", 1.0)))
    p.add_argument("--wam-trust", type=float, default=float(_cfg_get(cfg, "wam_trust", 0.75)))
    p.add_argument("--afinch-iters", type=int, default=int(_cfg_get(cfg, "afinch_iters", 8)))
    p.add_argument("--afinch-damping", type=float, default=float(_cfg_get(cfg, "afinch_damping", 0.9)))
    p.add_argument("--max-output-cms", type=float, default=float(_cfg_get(cfg, "max_output_cms", 10_000.0)))
    p.add_argument("--map-year", type=int, default=int(_cfg_get(cfg, "map_year", 2018)))
    p.add_argument("--map-month", type=int, default=int(_cfg_get(cfg, "map_month", 1)))
    p.add_argument(
        "--skip-map",
        action="store_true",
        default=bool(_cfg_get(cfg, "skip_map", False)),
        help="Run routing only and skip map generation",
    )
    return p.parse_args(argv)


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)


def main() -> None:
    args = parse_args()

    route_cmd = [
        sys.executable,
        str(BASE_DIR / "route_points_to_comid_afinch_usgs_wam.py"),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--network-source",
        args.network_source,
        "--output-dir",
        args.output_dir,
        "--usgs-trust",
        str(args.usgs_trust),
        "--wam-trust",
        str(args.wam_trust),
        "--afinch-iters",
        str(args.afinch_iters),
        "--afinch-damping",
        str(args.afinch_damping),
        "--max-output-cms",
        str(args.max_output_cms),
    ]
    run_cmd(route_cmd)

    if args.skip_map:
        print("Skipped map generation (--skip-map).")
        return

    map_cmd = [
        sys.executable,
        str(BASE_DIR / "create_routed_flow_map.py"),
        "--output-dir",
        args.output_dir,
        "--year",
        str(args.map_year),
        "--month",
        str(args.map_month),
    ]
    run_cmd(map_cmd)

    out_dir = (BASE_DIR / args.output_dir).resolve()
    map_file = out_dir / f"nhd_{args.map_year}{args.map_month:02d}_flow_map.html"
    print(f"Workflow complete. Map: {map_file}")


if __name__ == "__main__":
    main()
