#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from brazos_streamflow_model import _read_flowline_source


ROOT = Path(__file__).resolve().parent
THS_FLOWLINES = ROOT / "HSR1200" / "Flowlines" / "nhdflowline.txt"
THS_CATCHMENTS = ROOT / "inputData" / "NHDPlusCatchment_1201.gpkg"
HR_GDB_DIR = ROOT / "inputData" / "texas_nhdplusgrb" / "_extracted_gdb"
OUT_DIR = ROOT / "output"
OUT_PNG = OUT_DIR / "ths_1201_lines_used_in_model_run.png"


def main() -> None:
    if not THS_FLOWLINES.exists():
        raise FileNotFoundError(THS_FLOWLINES)
    if not THS_CATCHMENTS.exists():
        raise FileNotFoundError(THS_CATCHMENTS)
    if not HR_GDB_DIR.exists():
        raise FileNotFoundError(HR_GDB_DIR)

    ths = pd.read_csv(THS_FLOWLINES)
    ths_ids = pd.to_numeric(ths["ComID"], errors="coerce").dropna().astype("int64")
    ths_id_set = set(ths_ids.tolist())
    print(f"THS ComIDs listed: {len(ths_id_set):,}")

    # Restrict read extent to THS 1201 catchment bounds for speed.
    catch = gpd.read_file(THS_CATCHMENTS)
    minx, miny, maxx, maxy = catch.total_bounds
    print(f"THS bounds: {(minx, miny, maxx, maxy)}")

    flow = _read_flowline_source(
        flowline_source=str(HR_GDB_DIR),
        flowline_layer="NHDFlowline",
        bbox=(float(minx), float(miny), float(maxx), float(maxy)),
    )

    if "nhdplusid" not in flow.columns:
        raise ValueError("Expected nhdplusid column in HR flowline source.")

    flow["nhdplusid"] = pd.to_numeric(flow["nhdplusid"], errors="coerce")
    flow = flow.dropna(subset=["nhdplusid"]).copy()
    flow["nhdplusid"] = flow["nhdplusid"].astype("int64")

    ths_lines = flow[flow["nhdplusid"].isin(ths_id_set)].copy()
    if ths_lines.empty:
        raise RuntimeError("No THS lines matched HR flowline geometry.")

    print(f"HR flowlines in bbox: {len(flow):,}")
    print(f"THS lines matched: {len(ths_lines):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 11), dpi=200)
    ths_lines = ths_lines.to_crs(4326)
    ths_lines.plot(ax=ax, color="#0a5ea8", linewidth=0.35, alpha=0.85)

    ax.set_title("THS 1201 Stream Lines Used in AFINCH Model Run", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.25, alpha=0.3)

    fig.text(
        0.01,
        0.01,
        f"Lines plotted: {len(ths_lines):,} | Source: THS ComIDs from HSR1200/Flowlines/nhdflowline.txt",
        fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
