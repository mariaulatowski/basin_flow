from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"c:\Users\mu3575\Documents\WAM")
INPUT_DIR = ROOT / "inputData"

THS = "1201"
HSR_REGION = ROOT / "HSR1200"
HSR_THS = ROOT / "HSR1201"
TARGET_WY = 2018

NHD_GDB = (
    ROOT
    / "inputData"
    / "texas_nhdplusgrb"
    / "_extracted_gdb"
    / "NHDPLUS_H_1201_HU4_GDB"
    / "NHDPLUS_H_1201_HU4_GDB.gdb"
)

SQKM_TO_SQMI = 0.386102159


def _ensure_dirs() -> None:
    for p in [
        HSR_REGION / "NLCD",
        HSR_REGION / "Flowlines",
        HSR_REGION / "PRISM" / "Precipitation",
        HSR_REGION / "PRISM" / "Temperature",
        HSR_REGION / "WaterUse",
        HSR_REGION / "GIS",
        HSR_REGION / "Streamflow",
        HSR_THS / "GagedCatchments",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _build_base_from_nhdplus_gdb() -> pd.DataFrame:
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "geopandas is required to build from NHDPlus GDB. Install geopandas and dependencies."
        ) from exc

    if not NHD_GDB.exists():
        raise FileNotFoundError(f"NHD geodatabase not found: {NHD_GDB}")

    catch = gpd.read_file(NHD_GDB, layer="NHDPlusCatchment")
    flow = gpd.read_file(NHD_GDB, layer="NHDFlowline")

    if "NHDPlusID" not in catch.columns or "GridCode" not in catch.columns:
        raise RuntimeError("NHDPlusCatchment missing required fields NHDPlusID/GridCode")

    if "NHDPlusID" not in flow.columns:
        raise RuntimeError("NHDFlowline missing required field NHDPlusID")

    reach_col = "ReachCode" if "ReachCode" in flow.columns else None
    if reach_col is None:
        # Fallback keeps parser compatibility in converted module.
        flow["ReachCode"] = [f"{THS}{i:010d}" for i in range(1, len(flow) + 1)]
        reach_col = "ReachCode"

    length_col = "LengthKM" if "LengthKM" in flow.columns else None
    if length_col is None:
        flow["LengthKM"] = 1.0
        length_col = "LengthKM"

    catch_df = catch[["NHDPlusID", "GridCode"]].copy()
    flow_df = flow[["NHDPlusID", reach_col, length_col]].copy()

    catch_df["ComID"] = pd.to_numeric(catch_df["NHDPlusID"], errors="coerce").astype("Int64")
    catch_df["GridCode"] = pd.to_numeric(catch_df["GridCode"], errors="coerce").astype("Int64")

    flow_df["ComID"] = pd.to_numeric(flow_df["NHDPlusID"], errors="coerce").astype("Int64")
    flow_df["ReachCode"] = flow_df[reach_col].astype(str)
    flow_df["LengthKm"] = pd.to_numeric(flow_df[length_col], errors="coerce").fillna(1.0)

    catch_df = catch_df.dropna(subset=["ComID", "GridCode"]).astype({"ComID": "int64", "GridCode": "int64"})
    flow_df = flow_df.dropna(subset=["ComID"]).astype({"ComID": "int64"})

    # Keep THS subset by ReachCode prefix where available.
    flow_df = flow_df[flow_df["ReachCode"].str.startswith(THS)].copy()

    base = flow_df.merge(catch_df[["ComID", "GridCode"]], on="ComID", how="inner")
    base = base.drop_duplicates(subset=["ComID"], keep="first").sort_values("ComID").reset_index(drop=True)

    if base.empty:
        raise RuntimeError("No overlap between NHDFlowline (THS) and NHDPlusCatchment for base linkage")

    return base[["ComID", "GridCode", "ReachCode", "LengthKm"]]


def _write_flowline_and_crosswalk(base: pd.DataFrame) -> pd.DataFrame:
    flow = base[["ComID", "LengthKm", "ReachCode"]].copy()
    flow.to_csv(HSR_REGION / "Flowlines" / "nhdflowline.txt", index=False)

    xwalk = base[["GridCode", "ComID"]].copy().sort_values("ComID")
    with (HSR_REGION / "Flowlines" / "GridCodeComID.txt").open("w", encoding="utf-8") as f:
        f.write("GridCode,ComID\n")
        xwalk.to_csv(f, index=False, header=False)

    comids = xwalk["ComID"].to_numpy(dtype=np.int64)
    vaa = pd.DataFrame(
        {
            "ComID": comids,
            "HydroSeq": np.arange(1, len(comids) + 1, dtype=np.int64),
            "DnHydroSeq": np.append(np.arange(2, len(comids) + 1, dtype=np.int64), 0),
        }
    )
    vaa.to_csv(HSR_REGION / "GIS" / "NHDFlowlineVAA.txt", index=False)

    return xwalk[["ComID", "GridCode"]]


def _write_nlcd(base: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    nlcd_cols = [
        "NLCD11", "NLCD12", "NLCD21", "NLCD22", "NLCD23", "NLCD31", "NLCD32", "NLCD33",
        "NLCD41", "NLCD42", "NLCD43", "NLCD51", "NLCD61", "NLCD71", "NLCD81", "NLCD82",
        "NLCD83", "NLCD84", "NLCD85", "NLCD91", "NLCD92",
    ]

    raw = rng.uniform(0.1, 1.0, size=(len(base), len(nlcd_cols)))
    pct = (raw / raw.sum(axis=1, keepdims=True)) * 100.0

    nlcd = base.copy()
    for idx, c in enumerate(nlcd_cols):
        nlcd[c] = pct[:, idx]
    nlcd["PCTCN"] = 0.0
    nlcd["PCTMX"] = 0.0
    nlcd["SUMPCT"] = 100.0
    nlcd.to_csv(HSR_REGION / "NLCD" / "catchmentattributesnlcd.txt", index=False)


def _write_prism(base: pd.DataFrame, wy: int) -> None:
    n = len(base)
    area_sq_mi = np.full(n, 1.0, dtype=float)

    p = np.tile(np.array([1.2, 1.1, 1.4, 2.0, 2.5, 3.0, 2.8, 2.6, 2.3, 2.0, 1.7, 1.4, 2.0]), (n, 1))
    prism_p = pd.DataFrame(np.column_stack([base["GridCode"].to_numpy(), area_sq_mi, p]))

    p_path = HSR_REGION / "PRISM" / "Precipitation" / f"PrismPrecipWY{wy}.dat"
    with p_path.open("w", encoding="utf-8") as f:
        f.write("PRISM precipitation\n")
        f.write("Synthetic adapter dataset\n")
        f.write("GridCode GCAreaSqMi PIn_01..PIn_13\n")
        f.write("Units: inches\n")
        prism_p.to_csv(f, index=False, header=False, sep=" ")

    t = np.tile(np.array([8, 10, 13, 17, 21, 25, 27, 27, 23, 18, 12, 9], dtype=float), (n, 1))
    prism_t = pd.DataFrame(np.column_stack([base["GridCode"].to_numpy(), t]))

    t_path = HSR_REGION / "PRISM" / "Temperature" / f"PrismTempAveWY{wy}.dat"
    with t_path.open("w", encoding="utf-8") as f:
        f.write("PRISM temperature\n")
        f.write("Synthetic adapter dataset\n")
        f.write("GridCode TdC_01..TdC_12\n")
        f.write("Units: degC\n")
        prism_t.to_csv(f, index=False, header=False, sep=" ")


def _write_water_use(base: pd.DataFrame) -> None:
    wu = pd.DataFrame({"ComID": base["ComID"].astype("int64")})
    for m in range(1, 13):
        wu[f"WU{m:02d}"] = 0.0
    wu.to_csv(HSR_REGION / "WaterUse" / "ComID_WU_All.dat", index=False, header=False, sep=" ")


def _write_station_files(base: pd.DataFrame, wy: int) -> None:
    mw = pd.read_csv(INPUT_DIR / "inputs" / "monthly_wide_acft.csv")
    wy_df = mw[mw["Year"] == wy].copy()
    if wy_df.empty:
        wy_df = mw[mw["Year"] == mw["Year"].max()].copy()

    wy_df = wy_df.drop_duplicates(subset=["Gage_ID_norm"]).head(25)
    stations = wy_df["Gage_ID_norm"].astype(str).tolist()

    with (HSR_THS / "GagedCatchments" / "StationList.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(stations) + "\n")

    comids = base["ComID"].to_numpy()
    gridcodes = base["GridCode"].to_numpy()
    reachcodes = base["ReachCode"].astype(str).to_numpy()
    n = len(comids)

    rows_for_station_comid = []
    for idx, sta in enumerate(stations):
        count = 6
        start = (idx * count) % n
        inds = np.arange(start, start + count) % n

        dat = pd.DataFrame(
            {
                "GridCode": gridcodes[inds],
                "ComID": comids[inds],
                "AreaSqKm": np.full(count, 2.5, dtype=float),
                "ReachCode": reachcodes[inds],
            }
        )
        out = HSR_THS / "GagedCatchments" / f"{sta}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write("GridCode,ComID,AreaSqKm,ReachCode\n")
            dat.to_csv(f, index=False, header=False)

        rows_for_station_comid.append({"Station": sta, "ComID": int(comids[inds[0]])})

    pd.DataFrame(rows_for_station_comid).to_csv(
        HSR_REGION / "Flowlines" / "StationComID.csv", index=False
    )

    da = pd.DataFrame(
        {
            "Station": stations,
            "DASqMi": np.full(len(stations), 5.79, dtype=float),
        }
    )
    da.to_csv(HSR_REGION / "Streamflow" / "StationDASqMi.csv", index=False)

    month_cols = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP"]
    have_months = [c for c in month_cols if c in wy_df.columns]
    use = wy_df[["Gage_ID_norm", *have_months]].copy()

    for m in month_cols:
        if m not in use.columns:
            use[m] = 0.0

    cfs_month = use[month_cols].astype(float).to_numpy() / 60.0
    annual_like = cfs_month.mean(axis=1, keepdims=True)
    q13 = np.hstack([cfs_month, annual_like])

    station_comid = pd.read_csv(HSR_REGION / "Flowlines" / "StationComID.csv")
    station_to_comid = dict(zip(station_comid["Station"].astype(str), station_comid["ComID"].astype(int)))

    rows = []
    for i, sta in enumerate(use["Gage_ID_norm"].astype(str)):
        rows.append([
            station_to_comid.get(sta, int(base["ComID"].iloc[i % len(base)])),
            sta,
            5.79,
            *q13[i, :].tolist(),
        ])

    q_cols = [f"Q{i:02d}" for i in range(1, 14)]
    out_df = pd.DataFrame(rows, columns=["ComIDSta", "StaWY", "NWISArea", *q_cols])
    out_df.to_csv(
        HSR_REGION / "Streamflow" / f"ComIDStationDAMoAnQ{wy}.dat",
        index=False,
        header=False,
        sep=" ",
    )


def build_adapter(wy: int = TARGET_WY) -> None:
    _ensure_dirs()
    base = _build_base_from_nhdplus_gdb()
    base_small = _write_flowline_and_crosswalk(base)
    _write_nlcd(base_small)
    _write_prism(base_small, wy)
    _write_water_use(base_small)
    _write_station_files(base, wy)

    print("ADAPTER_BUILD_COMPLETE")
    print(f"HSR region dir: {HSR_REGION}")
    print(f"HSR THS dir: {HSR_THS}")
    print(f"comids={len(base_small)} stations={sum(1 for _ in (HSR_THS / 'GagedCatchments').glob('*.dat'))}")


if __name__ == "__main__":
    build_adapter(TARGET_WY)
