from __future__ import annotations

from pathlib import Path

ROOT = Path(r"c:\Users\mu3575\Documents\WAM")

# Minimal required paths for converted AFINCH module workflow.
REQUIRED = [
    "HSR0100/NLCD/catchmentattributesnlcd.txt",
    "HSR0100/Flowlines/nhdflowline.txt",
    "HSR0100/Flowlines/GridCodeComID.txt",
    "HSR0100/Flowlines/StationComID.csv",
    "HSR0100/PRISM/Precipitation/PrismPrecipWY2000.dat",
    "HSR0100/PRISM/Temperature/PrismTempAveWY2000.dat",
    "HSR0100/WaterUse/ComID_WU_All.dat",
    "HSR0100/GIS/NHDFlowlineVAA.txt",
    "HSR0100/Streamflow/StationDASqMi.csv",
    "HSR0101/GagedCatchments/StationList.txt",
]


def run() -> int:
    missing = []
    present = []
    for rel in REQUIRED:
        p = ROOT / rel
        if p.exists():
            present.append(rel)
        else:
            missing.append(rel)

    print("CONVERTED_AFINCH_READINESS")
    print(f"present={len(present)} missing={len(missing)}")
    if present:
        print("PRESENT:")
        for rel in present:
            print(f"  {rel}")
    if missing:
        print("MISSING:")
        for rel in missing:
            print(f"  {rel}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(run())
