from __future__ import annotations

from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(r"c:\Users\mu3575\Documents\WAM")
HSR_REGION = ROOT / "HSR1200"
HSR_THS = ROOT / "HSR1201" / "GagedCatchments"

USGS_DAT = HSR_REGION / "Streamflow" / "ComIDStationDAMoAnQ2018.dat"
WAM_DAT = HSR_REGION / "Streamflow" / "ComIDStationDAMoAnQ2018_WAM.dat"
COMBINED_DAT = HSR_REGION / "Streamflow" / "ComIDStationDAMoAnQ2018_USGS_WAM_Combined.dat"

STATION_COMID = HSR_REGION / "Flowlines" / "StationComID.csv"
STATION_DA = HSR_REGION / "Streamflow" / "StationDASqMi.csv"
STATION_LIST = HSR_THS / "StationList.txt"

WAM_DA_CSV = HSR_REGION / "Streamflow" / "Brazos_StationDASqMi.csv"
WAM_POINTS_CSV = ROOT / "inputData" / "inputs" / "monthly_wide_cfs_from_hecdss.csv"
WAM_POINTS_CORRECTED_CSV = HSR_REGION / "Streamflow" / "Brazos_new_wam_locations_nhdplus.csv"
CATCHMENT_GPKG = ROOT / "inputData" / "NHDPlusCatchment_1201.gpkg"
FLOWLINE_TXT = HSR_REGION / "Flowlines" / "nhdflowline.txt"

BACKUP_SUFFIX = ".usgs_only.bak"


def _backup_once(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def _read_station_dat(dat_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        dat_path,
        sep=r"\s+",
        header=None,
        names=["ComIDSta", "StaWY", "NWISArea", *[f"Q{i:02d}" for i in range(1, 14)]],
        engine="python",
    )


def _normalize_station_id(value: object) -> str:
    text = str(value).strip()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def _load_wam_points(stations: list[str]) -> gpd.GeoDataFrame:
    # Prefer curated CPID lat/lon corrections if available.
    src = WAM_POINTS_CORRECTED_CSV if WAM_POINTS_CORRECTED_CSV.exists() else WAM_POINTS_CSV
    pts = pd.read_csv(src, dtype={"CPID": str})
    pts.columns = [str(c).strip() for c in pts.columns]

    # Normalize expected column names across sources.
    ren = {}
    if "lat" in pts.columns and "LAT" not in pts.columns:
        ren["lat"] = "LAT"
    if "lon" in pts.columns and "LONG" not in pts.columns:
        ren["lon"] = "LONG"
    if ren:
        pts = pts.rename(columns=ren)

    missing = {"CPID", "LAT", "LONG"} - set(pts.columns)
    if missing:
        raise KeyError(f"Missing required columns in {src}: {sorted(missing)}")

    pts["CPID"] = pts["CPID"].astype(str).str.strip()
    pts = pts[pts["CPID"].isin(stations)].copy()
    pts = pts.drop_duplicates(subset=["CPID"], keep="first")
    pts["LAT"] = pd.to_numeric(pts["LAT"], errors="coerce")
    pts["LONG"] = pd.to_numeric(pts["LONG"], errors="coerce")
    pts = pts.dropna(subset=["LAT", "LONG"]).copy()
    return gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["LONG"], pts["LAT"]),
        crs="EPSG:4326",
    )


def _build_wam_station_mapping(stations: list[str]) -> pd.DataFrame:
    catch = gpd.read_file(CATCHMENT_GPKG)
    catch["ComID"] = pd.to_numeric(catch["NHDPlusID"], errors="coerce")
    catch["GridCode"] = pd.to_numeric(catch["GridCode"], errors="coerce")
    catch = catch.dropna(subset=["ComID", "GridCode"]).copy()
    catch["ComID"] = catch["ComID"].astype(np.int64)
    catch["GridCode"] = catch["GridCode"].astype(np.int64)

    flow = pd.read_csv(FLOWLINE_TXT)
    flow["ComID"] = pd.to_numeric(flow["ComID"], errors="coerce").astype(np.int64)
    catch = catch.merge(flow[["ComID", "ReachCode"]], on="ComID", how="inner")

    catch_proj = catch.to_crs(catch.estimate_utm_crs())
    catch_proj["AreaSqKm"] = catch_proj.geometry.area / 1_000_000.0

    points = _load_wam_points(stations)
    points_proj = points.to_crs(catch_proj.crs)

    joined = gpd.sjoin(
        points_proj[["CPID", "geometry"]],
        catch_proj[["ComID", "GridCode", "ReachCode", "AreaSqKm", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])

    missing = joined["ComID"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            points_proj.loc[missing, ["CPID", "geometry"]],
            catch_proj[["ComID", "GridCode", "ReachCode", "AreaSqKm", "geometry"]],
            how="left",
            distance_col="snap_dist_m",
        )
        nearest = nearest.drop(columns=[c for c in ["index_right"] if c in nearest.columns])
        joined.loc[missing, ["ComID", "GridCode", "ReachCode", "AreaSqKm"]] = nearest[
            ["ComID", "GridCode", "ReachCode", "AreaSqKm"]
        ].to_numpy()

    joined = joined.dropna(subset=["ComID", "GridCode", "ReachCode", "AreaSqKm"]).copy()
    joined["ComID"] = joined["ComID"].astype(np.int64)
    joined["GridCode"] = joined["GridCode"].astype(np.int64)
    return pd.DataFrame(joined[["CPID", "ComID", "GridCode", "ReachCode", "AreaSqKm"]]).drop_duplicates(
        subset=["CPID"], keep="first"
    )


def _write_wam_gaged_catchments(mapping: pd.DataFrame) -> None:
    for _, row in mapping.iterrows():
        out = HSR_THS / f"{row['CPID']}.dat"
        df = pd.DataFrame(
            {
                "GridCode": [int(row["GridCode"])],
                "ComID": [int(row["ComID"])],
                "AreaSqKm": [float(row["AreaSqKm"])],
                "ReachCode": [str(row["ReachCode"])],
            }
        )
        with out.open("w", encoding="utf-8") as f:
            f.write("GridCode,ComID,AreaSqKm,ReachCode\n")
            df.to_csv(f, index=False, header=False)


def build() -> None:
    for path in [USGS_DAT, STATION_COMID, STATION_DA, STATION_LIST]:
        _backup_once(path)

    usgs = _read_station_dat(USGS_DAT)
    wam = _read_station_dat(WAM_DAT)
    wam_da = pd.read_csv(WAM_DA_CSV, dtype={"station": str})
    wam_da["station"] = wam_da["station"].astype(str).str.strip()
    wam_da["comid"] = pd.to_numeric(wam_da["comid"], errors="coerce")
    wam_da["DASqMi"] = pd.to_numeric(wam_da["DASqMi"], errors="coerce")
    wam_da = wam_da.dropna(subset=["station", "DASqMi"]).drop_duplicates(subset=["station"], keep="first")

    wam["StaWY"] = wam["StaWY"].astype(str).str.strip()
    wam = wam[wam["StaWY"].isin(wam_da["station"])].copy()

    mapping = _build_wam_station_mapping(wam["StaWY"].astype(str).tolist())
    if mapping.empty:
        raise RuntimeError("No WAM stations could be mapped onto the THS catchment network.")

    wam = wam.merge(mapping, left_on="StaWY", right_on="CPID", how="inner")
    wam = wam.merge(wam_da[["station", "DASqMi"]], left_on="StaWY", right_on="station", how="left")
    wam["ComIDSta"] = wam["ComID"].astype(np.int64)
    wam["NWISArea"] = pd.to_numeric(wam["DASqMi"], errors="coerce")
    wam = wam[["ComIDSta", "StaWY", "NWISArea", *[f"Q{i:02d}" for i in range(1, 14)]]].copy()

    usgs_out = usgs.copy()
    usgs_out["StaWY"] = usgs_out["StaWY"].map(_normalize_station_id)
    combined = pd.concat([usgs_out, wam], ignore_index=True)
    combined = combined.drop_duplicates(subset=["StaWY"], keep="first")
    combined.to_csv(COMBINED_DAT, index=False, header=False, sep=" ")
    combined.to_csv(USGS_DAT, index=False, header=False, sep=" ")

    station_comid_existing = pd.read_csv(STATION_COMID)
    station_comid_existing["Station"] = station_comid_existing["Station"].map(_normalize_station_id)
    station_comid_wam = mapping.rename(columns={"CPID": "Station", "ComID": "ComID"})[["Station", "ComID"]].copy()
    station_comid_all = pd.concat([station_comid_existing, station_comid_wam], ignore_index=True)
    station_comid_all = station_comid_all.drop_duplicates(subset=["Station"], keep="first")
    station_comid_all.to_csv(STATION_COMID, index=False)

    usgs_da = pd.read_csv(STATION_DA)
    if "Station" in usgs_da.columns:
        usgs_da["Station"] = usgs_da["Station"].map(_normalize_station_id)
    wam_da_out = mapping.rename(columns={"CPID": "Station", "ComID": "ComID"})[["Station", "ComID"]].merge(
        wam_da[["station", "DASqMi"]], left_on="Station", right_on="station", how="left"
    )
    wam_da_out = wam_da_out[["Station", "ComID", "DASqMi"]]
    da_all = pd.concat([usgs_da, wam_da_out], ignore_index=True)
    da_all = da_all.drop_duplicates(subset=[da_all.columns[0]], keep="first")
    da_all.to_csv(STATION_DA, index=False)

    existing_station_list = [s.strip() for s in STATION_LIST.read_text(encoding="utf-8").splitlines() if s.strip()]
    station_list_all = existing_station_list + [s for s in wam["StaWY"].astype(str).tolist() if s not in existing_station_list]
    STATION_LIST.write_text("\n".join(station_list_all) + "\n", encoding="utf-8")

    _write_wam_gaged_catchments(mapping)

    print(f"Combined DAT rows: {len(combined)}")
    print(f"Mapped WAM stations: {len(mapping)}")
    print(f"Station list size: {len(station_list_all)}")
    print(f"Combined DAT: {COMBINED_DAT}")


if __name__ == "__main__":
    build()