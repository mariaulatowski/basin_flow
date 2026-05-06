from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

SQKM_TO_SQMI = 0.386102159


def _normalize_station_id(value: object) -> str:
    text = str(value).strip()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a runnable Brazos AFINCH network package from HU4-1205 medium-resolution NHD data."
    )
    p.add_argument("--base-dir", default=".", help="Workspace base directory")
    p.add_argument("--ths", default="1205", help="Target THS code")
    p.add_argument("--hsr", default="HSR1200", help="HSR folder name used by converted runtime")
    p.add_argument(
        "--gdb",
        default="inputData/nhd_medium_res_gdb/NHD_H_1205_HU4_GDB/NHD_H_1205_HU4_GDB.gdb",
        help="Source geodatabase path",
    )
    p.add_argument("--wy", type=int, default=2018, help="Water year to generate PRISM files for")
    p.add_argument("--apply", action="store_true", help="Apply updates into HSR files (writes backups)")
    return p.parse_args()


def _backup(path: Path, suffix: str = ".pre1205.bak") -> None:
    if path.exists():
        backup = path.with_name(path.name + suffix)
        if not backup.exists():
            shutil.copy2(path, backup)


def _load_flowline_and_vaa(gdb: Path, ths: str) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    fl = gpd.read_file(gdb, layer="NHDFlowline")
    vaa = gpd.read_file(gdb, layer="NHDFlowlineVAA")

    fl["ComID"] = pd.to_numeric(fl["permanent_identifier"], errors="coerce")
    fl["LengthKm"] = pd.to_numeric(fl.get("lengthkm", 1.0), errors="coerce")
    fl["ReachCode"] = fl.get("reachcode", "").astype(str).str.strip()

    fl = fl.dropna(subset=["ComID", "geometry"]).copy()
    fl = fl[~fl.geometry.is_empty].copy()
    fl["ComID"] = fl["ComID"].astype("int64")
    fl["LengthKm"] = fl["LengthKm"].fillna(1.0).clip(lower=0.01)

    # Keep THS-specific reach codes where available; fallback to generated reachcode if missing.
    has_rc = fl["ReachCode"].str.len() > 0
    fl.loc[~has_rc, "ReachCode"] = [f"{ths}{i:010d}" for i in range(1, (~has_rc).sum() + 1)]
    starts_ths = fl["ReachCode"].str.startswith(ths)
    if int(starts_ths.sum()) > 0:
        fl = fl[starts_ths].copy()

    fl = fl[["ComID", "LengthKm", "ReachCode", "geometry"]].drop_duplicates(subset=["ComID"], keep="first")
    fl = fl.sort_values("ComID").reset_index(drop=True)

    vaa["ComID"] = pd.to_numeric(vaa["permanent_identifier"], errors="coerce")
    vaa["HydroSeq"] = pd.to_numeric(vaa.get("hydroseq", np.nan), errors="coerce")
    vaa["FromNode"] = pd.to_numeric(vaa.get("fromnode", np.nan), errors="coerce")
    vaa["ToNode"] = pd.to_numeric(vaa.get("tonode", np.nan), errors="coerce")
    vaa["Divergence"] = pd.to_numeric(vaa.get("divergenceflag", 0), errors="coerce").fillna(0)
    vaa["StartFlag"] = pd.to_numeric(vaa.get("startflag", 0), errors="coerce").fillna(0)

    vaa = vaa.dropna(subset=["ComID"]).copy()
    vaa["ComID"] = vaa["ComID"].astype("int64")
    vaa = vaa[vaa["ComID"].isin(fl["ComID"])].copy()

    if vaa.empty:
        # Synthetic topologic ordering fallback.
        synthetic = fl[["ComID"]].copy()
        synthetic["HydroSeq"] = np.arange(len(synthetic), 0, -1, dtype=np.int64)
        synthetic["FromNode"] = synthetic["HydroSeq"]
        synthetic["ToNode"] = np.append(synthetic["HydroSeq"].to_numpy(dtype=np.int64)[1:], 0)
        synthetic["Divergence"] = 0
        synthetic["StartFlag"] = 0
        synthetic.loc[synthetic.index[0], "StartFlag"] = 1
        vaa_out = synthetic
    else:
        vaa = vaa[["ComID", "FromNode", "ToNode", "HydroSeq", "Divergence", "StartFlag"]].copy()
        vaa["HydroSeq"] = vaa["HydroSeq"].fillna(0)
        missing_hs = vaa["HydroSeq"] <= 0
        if missing_hs.any():
            fill_vals = np.arange(missing_hs.sum(), 0, -1, dtype=np.int64)
            vaa.loc[missing_hs, "HydroSeq"] = fill_vals
        vaa["HydroSeq"] = vaa["HydroSeq"].astype("int64")
        vaa["FromNode"] = vaa["FromNode"].fillna(vaa["HydroSeq"]).astype("int64")
        vaa["ToNode"] = vaa["ToNode"].fillna(0).astype("int64")
        vaa["Divergence"] = vaa["Divergence"].fillna(0).astype("int64")
        vaa["StartFlag"] = vaa["StartFlag"].fillna(0).astype("int64")

        # Ensure every flowline ComID has a row.
        missing = np.setdiff1d(fl["ComID"].to_numpy(dtype=np.int64), vaa["ComID"].to_numpy(dtype=np.int64))
        if len(missing) > 0:
            extra = pd.DataFrame({"ComID": missing})
            extra["HydroSeq"] = np.arange(len(vaa) + len(extra), len(vaa), -1, dtype=np.int64)
            extra["FromNode"] = extra["HydroSeq"]
            extra["ToNode"] = 0
            extra["Divergence"] = 0
            extra["StartFlag"] = 1
            vaa = pd.concat([vaa, extra], ignore_index=True)

        vaa_out = vaa.sort_values("ComID").drop_duplicates(subset=["ComID"], keep="first").reset_index(drop=True)

    return fl[["ComID", "LengthKm", "ReachCode"]].copy(), vaa_out, fl[["ComID", "geometry"]].copy()


def _load_station_points(base_dir: Path) -> pd.DataFrame:
    usgs_path = base_dir / "inputData" / "inputs" / "monthly_wide_acft.csv"
    wam_path = base_dir / "HSR1200" / "Streamflow" / "Brazos_new_wam_locations_nhdplus.csv"

    usgs = pd.read_csv(usgs_path, dtype={"Gage_ID_norm": str})
    usgs_pts = pd.DataFrame(
        {
            "Station": usgs["Gage_ID_norm"].astype(str).map(_normalize_station_id),
            "LAT": pd.to_numeric(usgs["LAT"], errors="coerce"),
            "LONG": pd.to_numeric(usgs["LONG"], errors="coerce"),
            "Source": "USGS",
        }
    ).dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")

    wam = pd.read_csv(wam_path, dtype={"CPID": str})
    if "lat" in wam.columns and "LAT" not in wam.columns:
        wam = wam.rename(columns={"lat": "LAT"})
    if "lon" in wam.columns and "LONG" not in wam.columns:
        wam = wam.rename(columns={"lon": "LONG"})
    wam_pts = pd.DataFrame(
        {
            "Station": wam["CPID"].astype(str).map(_normalize_station_id),
            "LAT": pd.to_numeric(wam["LAT"], errors="coerce"),
            "LONG": pd.to_numeric(wam["LONG"], errors="coerce"),
            "Source": "WAM",
        }
    ).dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")

    pts = pd.concat([usgs_pts, wam_pts], ignore_index=True)
    pts = pts.sort_values("Source").drop_duplicates(subset=["Station"], keep="first").reset_index(drop=True)
    return pts


def _map_stations_to_comid(
    base_dir: Path,
    ths: str,
    flow_geom: gpd.GeoDataFrame,
    station_points: pd.DataFrame,
) -> pd.DataFrame:
    station_list_path = base_dir / f"HSR{ths}" / "GagedCatchments" / "StationList.txt"
    if not station_list_path.exists():
        fallback = base_dir / "HSR1201" / "GagedCatchments" / "StationList.txt"
        if fallback.exists():
            station_list_path = fallback
        else:
            raise FileNotFoundError(f"Missing station list: {station_list_path}")
    station_list = {_normalize_station_id(s.strip()) for s in station_list_path.read_text(encoding="utf-8").splitlines() if s.strip()}

    pts = station_points[station_points["Station"].isin(station_list)].copy()
    pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["LONG"], pts["LAT"]), crs="EPSG:4326")

    flow_gdf = flow_geom.copy()
    if flow_gdf.crs is None:
        flow_gdf = flow_gdf.set_crs("EPSG:4326")

    proj = flow_gdf.estimate_utm_crs()
    flow_proj = flow_gdf.to_crs(proj)
    pts_proj = pts.to_crs(proj)

    nearest = gpd.sjoin_nearest(
        pts_proj[["Station", "Source", "LAT", "LONG", "geometry"]],
        flow_proj[["ComID", "geometry"]],
        how="left",
        distance_col="snap_dist_m",
    )
    nearest = nearest.drop(columns=[c for c in ["index_right"] if c in nearest.columns])

    nearest = nearest.dropna(subset=["ComID"]).copy()
    nearest["ComID"] = pd.to_numeric(nearest["ComID"], errors="coerce").astype("int64")
    nearest = nearest.drop_duplicates(subset=["Station"], keep="first")

    return pd.DataFrame(nearest[["Station", "ComID", "Source", "snap_dist_m"]]).sort_values("Station")


def _read_streamflow_dat(dat_path: Path) -> pd.DataFrame:
    qcols = [f"Q{i:02d}" for i in range(1, 14)]
    cols = ["ComIDSta", "StaWY", "NWISArea", *qcols]
    df = pd.read_csv(dat_path, sep=r"\s+", header=None, names=cols, engine="python")
    df["StaWY"] = df["StaWY"].astype(str).map(_normalize_station_id)
    return df


def _write_streamflow_dat(df: pd.DataFrame, out_path: Path) -> None:
    qcols = [f"Q{i:02d}" for i in range(1, 14)]
    out = df[["ComIDSta", "StaWY", "NWISArea", *qcols]].copy()
    out.to_csv(out_path, index=False, header=False, sep=" ")


def _build_synthetic_nlcd(comids: np.ndarray) -> pd.DataFrame:
    nlcd_cols = [
        "NLCD11", "NLCD12", "NLCD21", "NLCD22", "NLCD23", "NLCD31", "NLCD32", "NLCD33",
        "NLCD41", "NLCD42", "NLCD43", "NLCD51", "NLCD61", "NLCD71", "NLCD81", "NLCD82",
        "NLCD83", "NLCD84", "NLCD85", "NLCD91", "NLCD92",
    ]
    rng = np.random.default_rng(1205)
    raw = rng.uniform(0.05, 1.0, size=(len(comids), len(nlcd_cols)))
    pct = raw / raw.sum(axis=1, keepdims=True) * 100.0

    df = pd.DataFrame({"ComID": comids.astype("int64"), "GridCode": comids.astype("int64")})
    for i, c in enumerate(nlcd_cols):
        df[c] = pct[:, i]
    df["PCTCN"] = 0.0
    df["PCTMX"] = 0.0
    df["SUMPCT"] = 100.0
    return df


def _build_synthetic_prism(comids: np.ndarray, wy: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(comids), dtype=float)
    base = (comids % 97).astype(float) / 97.0

    p = np.zeros((len(comids), 13), dtype=float)
    t = np.zeros((len(comids), 12), dtype=float)

    p_month_pattern = np.array([1.4, 1.2, 1.1, 1.3, 1.4, 1.9, 2.1, 2.4, 2.3, 2.1, 1.8, 1.5], dtype=float)
    t_month_pattern = np.array([11.0, 13.0, 15.0, 18.0, 21.0, 25.0, 28.0, 28.0, 25.0, 20.0, 15.0, 12.0], dtype=float)

    for m in range(12):
        p[:, m] = np.maximum(0.05, p_month_pattern[m] + 0.8 * base + 0.05 * np.sin((idx + m) * 0.09))
        t[:, m] = t_month_pattern[m] + 4.0 * (base - 0.5) + 0.3 * np.cos((idx + m) * 0.05)
    p[:, 12] = p[:, :12].mean(axis=1)

    precip = pd.DataFrame({"GridCode": comids.astype("int64"), "GCAreaSqMi": np.full(len(comids), 1.0, dtype=float)})
    for m in range(13):
        precip[f"PIn_{m + 1:02d}"] = p[:, m]

    temp = pd.DataFrame({"GridCode": comids.astype("int64")})
    for m in range(12):
        temp[f"TdC_{m + 1:02d}"] = t[:, m]

    return precip, temp


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).resolve()
    gdb = (base_dir / args.gdb).resolve()
    hsr_dir = base_dir / args.hsr

    if not gdb.exists():
        raise FileNotFoundError(gdb)

    flow_dir = hsr_dir / "Flowlines"
    gis_dir = hsr_dir / "GIS"
    nlcd_dir = hsr_dir / "NLCD"
    p_dir = hsr_dir / "PRISM" / "Precipitation"
    t_dir = hsr_dir / "PRISM" / "Temperature"
    wu_dir = hsr_dir / "WaterUse"
    sf_dir = hsr_dir / "Streamflow"
    gaged_dir = base_dir / f"HSR{args.ths}" / "GagedCatchments"

    for d in [flow_dir, gis_dir, nlcd_dir, p_dir, t_dir, wu_dir, sf_dir, gaged_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Reading HU4-1205 flowline and VAA layers...")
    flow_tbl, vaa_tbl, flow_geom = _load_flowline_and_vaa(gdb=gdb, ths=args.ths)
    comids = flow_tbl["ComID"].to_numpy(dtype=np.int64)
    print(f"Flowlines prepared: {len(flow_tbl):,} reaches")

    station_points = _load_station_points(base_dir)
    station_map = _map_stations_to_comid(base_dir, args.ths, flow_geom, station_points)
    print(f"Station mappings prepared: {len(station_map):,} stations")

    station_comid_path = flow_dir / "StationComID.csv"
    nhdflowline_path = flow_dir / "nhdflowline.txt"
    xwalk_path = flow_dir / "GridCodeComID.txt"
    vaa_path = gis_dir / "NHDFlowlineVAA.txt"
    nlcd_path = nlcd_dir / "catchmentattributesnlcd.txt"
    precip_path = p_dir / f"PrismPrecipWY{args.wy}.dat"
    temp_path = t_dir / f"PrismTempAveWY{args.wy}.dat"
    wu_path = wu_dir / "ComID_WU_All.dat"
    dat_path = sf_dir / f"ComIDStationDAMoAnQ{args.wy}.dat"
    da_path = sf_dir / "StationDASqMi.csv"
    catchment_gpkg = base_dir / "inputData" / "NHDPlusCatchment_1205.gpkg"

    if args.apply:
        for p in [station_comid_path, nhdflowline_path, xwalk_path, vaa_path, nlcd_path, precip_path, temp_path, wu_path, dat_path, da_path]:
            _backup(p)

    flow_tbl.to_csv(nhdflowline_path, index=False)

    with xwalk_path.open("w", encoding="utf-8") as f:
        f.write("GridCode,ComID\n")
        pd.DataFrame({"GridCode": comids, "ComID": comids}).to_csv(f, index=False, header=False)

    vaa_tbl[["ComID", "FromNode", "ToNode", "HydroSeq", "Divergence", "StartFlag"]].to_csv(vaa_path, index=False)

    # Synthetic NLCD and PRISM for runnable converted workflow.
    nlcd_df = _build_synthetic_nlcd(comids)
    nlcd_df.to_csv(nlcd_path, index=False)

    precip_df, temp_df = _build_synthetic_prism(comids, args.wy)
    with precip_path.open("w", encoding="utf-8") as f:
        f.write("PRISM precipitation\n")
        f.write("Synthetic Brazos HU4-1205 dataset\n")
        f.write("GridCode GCAreaSqMi PIn_01..PIn_13\n")
        f.write("Units: inches\n")
        precip_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

    with temp_path.open("w", encoding="utf-8") as f:
        f.write("PRISM temperature\n")
        f.write("Synthetic Brazos HU4-1205 dataset\n")
        f.write("GridCode TdC_01..TdC_12\n")
        f.write("Units: degC\n")
        temp_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

    wu_df = pd.DataFrame({"ComID": comids})
    for m in range(1, 13):
        wu_df[f"WU{m:02d}"] = 0.0
    wu_df.to_csv(wu_path, index=False, header=False, sep=" ")

    station_map.to_csv(station_comid_path, index=False)

    if dat_path.exists():
        dat = _read_streamflow_dat(dat_path)
        sm = station_map[["Station", "ComID"]].rename(columns={"Station": "StaWY", "ComID": "ComID_new"})
        dat = dat.merge(sm, on="StaWY", how="left")
        dat["ComIDSta"] = np.where(dat["ComID_new"].notna(), dat["ComID_new"], dat["ComIDSta"])
        dat = dat.drop(columns=["ComID_new"])
        dat["ComIDSta"] = pd.to_numeric(dat["ComIDSta"], errors="coerce").fillna(0).astype(np.int64)
        _write_streamflow_dat(dat, dat_path)

        da = dat[["StaWY", "ComIDSta", "NWISArea"]].drop_duplicates(subset=["StaWY"], keep="first").copy()
        da = da.rename(columns={"StaWY": "Station", "ComIDSta": "ComID", "NWISArea": "DASqMi"})
        da.to_csv(da_path, index=False)
    else:
        print(f"WARNING: streamflow DAT not found at {dat_path}; station DA/catchment files may be incomplete")
        da = pd.DataFrame(columns=["Station", "ComID", "DASqMi"])

    reach_lookup = flow_tbl.set_index("ComID")["ReachCode"].to_dict()
    da_map = da.set_index("Station")["DASqMi"].to_dict() if not da.empty else {}

    gaged_written = 0
    for _, row in station_map.iterrows():
        station = row["Station"]
        comid = int(row["ComID"])
        dasqmi = float(da_map.get(station, 1.0))
        area_sqkm = max(0.01, dasqmi / SQKM_TO_SQMI)
        reach = str(reach_lookup.get(comid, f"{args.ths}0000000000"))

        out = gaged_dir / f"{station}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write("GridCode,ComID,AreaSqKm,ReachCode\n")
            f.write(f"{comid},{comid},{area_sqkm:.6f},{reach}\n")
        gaged_written += 1

    # Provide geometry for mapping/export joins: line geometry with expected NHDPlusID field.
    catch = flow_geom.copy()
    catch = catch.rename(columns={"ComID": "NHDPlusID"})
    catch["GridCode"] = catch["NHDPlusID"]
    if catch.crs is None:
        catch = catch.set_crs("EPSG:4269")
    catch.to_file(catchment_gpkg, driver="GPKG")

    print("Done building Brazos HU4-1205 package")
    print(f"  Flowlines: {nhdflowline_path}")
    print(f"  VAA:       {vaa_path}")
    print(f"  Station map unique COMIDs: {station_map['ComID'].nunique():,} / stations {len(station_map):,}")
    print(f"  Gaged catchment files written: {gaged_written:,}")
    print(f"  Catchment GPKG: {catchment_gpkg}")


if __name__ == "__main__":
    main()
