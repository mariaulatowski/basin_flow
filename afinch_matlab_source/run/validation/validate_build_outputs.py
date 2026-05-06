from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AFINCH build output consistency")
    parser.add_argument("--base-dir", default=".", help="Workspace base directory")
    parser.add_argument("--hsr", default="HSR1200", help="HSR folder name")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve() / args.hsr

    vaa = pd.read_csv(base / "GIS" / "NHDFlowlineVAA.txt")
    nonzero_to = int((pd.to_numeric(vaa["ToNode"], errors="coerce").fillna(0) != 0).sum())

    gaged = base / "GagedCatchments"
    stations = [s.strip() for s in (gaged / "StationList.txt").read_text(encoding="utf-8").splitlines() if s.strip()]

    counts = []
    active_comids: set[int] = set()
    for s in stations:
        f = gaged / f"{s}.dat"
        if not f.exists():
            counts.append(0)
            continue
        d = pd.read_csv(f, skiprows=1, header=None, names=["GridCode", "ComID", "AreaSqKm", "ReachCode"])
        counts.append(len(d))
        active_comids.update(pd.to_numeric(d["ComID"], errors="coerce").dropna().astype("int64").tolist())

    nlcd = pd.read_csv(base / "NLCD" / "catchmentattributesnlcd.txt")
    nlcd_comids = set(pd.to_numeric(nlcd.iloc[:, 0], errors="coerce").dropna().astype("int64").tolist())
    overlap = len(active_comids & nlcd_comids)

    arr = np.asarray(counts, dtype=int) if counts else np.asarray([], dtype=int)

    print("VALIDATION SUMMARY")
    print("------------------")
    print(f"VAA rows: {len(vaa)}")
    print(f"VAA ToNode non-zero: {nonzero_to}")
    print("VAA connected?:", "YES" if nonzero_to > 0 else "NO")
    print(f"Stations in StationList: {len(stations)}")
    if arr.size:
        print(
            "Contributors per station:"
            f" min={int(arr.min())} median={int(np.median(arr))} max={int(arr.max())} gt1={int((arr > 1).sum())}"
        )
    else:
        print("Contributors per station: none")
    print(f"Unique active station ComIDs: {len(active_comids)}")
    print(f"NLCD overlap with active station ComIDs: {overlap}")


if __name__ == "__main__":
    main()
