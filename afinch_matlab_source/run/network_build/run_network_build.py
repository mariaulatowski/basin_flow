from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return cfg


def _resolve(base_dir: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\nCommand:")
    print("  " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible network build runner")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("build_config.json")),
        help="Path to build config JSON",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate/preview without writing")
    parser.add_argument("--skip-upstream", action="store_true", help="Skip upstream gaged-catchment build")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    cfg = _load_config(config_path)

    config_dir = config_path.parent
    base_dir = _resolve(config_dir, cfg["base_dir"])
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    builder_script = (base_dir / "afinch_python_modules" / "build_brazos_basin_network.py").resolve()
    upstream_script = (base_dir / "afinch_python_modules" / "build_usgs_upstream_spatial.py").resolve()
    if not builder_script.exists():
        raise FileNotFoundError(f"Missing builder script: {builder_script}")

    required_inputs = [
        _resolve(base_dir, cfg["basin_shp"]),
        _resolve(base_dir, cfg["gdb_root"]),
        _resolve(base_dir, cfg["nlcd_raster"]),
        _resolve(base_dir, cfg["prism_ppt_dir"]),
        _resolve(base_dir, cfg["prism_tmean_dir"]),
        _resolve(base_dir, cfg["gages_csv"]),
    ]
    for p in required_inputs:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    build_cmd = [
        sys.executable,
        str(builder_script),
        "--base-dir",
        str(base_dir),
        "--ths",
        str(cfg["ths"]),
        "--hsr",
        str(cfg["hsr"]),
        "--basin-shp",
        cfg["basin_shp"],
        "--basin-field",
        cfg["basin_field"],
        "--basin-value",
        cfg["basin_value"],
        "--gdb-root",
        cfg["gdb_root"],
        "--wy",
        str(cfg["build_wy"]),
        "--nlcd-raster",
        cfg["nlcd_raster"],
        "--prism-ppt-dir",
        cfg["prism_ppt_dir"],
        "--prism-tmean-dir",
        cfg["prism_tmean_dir"],
    ]

    if cfg.get("hu4s"):
        build_cmd += ["--hu4s", cfg["hu4s"]]
    if cfg.get("gages_csv"):
        build_cmd += ["--gages-csv", cfg["gages_csv"]]
    if cfg.get("wam_csv"):
        build_cmd += ["--wam-csv", cfg["wam_csv"]]
    if not args.dry_run:
        build_cmd.append("--apply")

    _run_cmd(build_cmd, cwd=base_dir)

    if args.skip_upstream:
        print("\nSkipped upstream gaged-catchment build by request.")
        return

    if not upstream_script.exists():
        raise FileNotFoundError(f"Missing upstream script: {upstream_script}")

    upstream_cmd = [
        sys.executable,
        str(upstream_script),
        "--base-dir",
        str(base_dir),
        "--hsr",
        str(cfg["hsr"]),
        "--wy",
        str(cfg["build_wy"]),
        "--catchment-gpkg",
        cfg["catchment_gpkg"],
    ]
    if not args.dry_run:
        upstream_cmd.append("--apply")

    _run_cmd(upstream_cmd, cwd=base_dir)


if __name__ == "__main__":
    main()
