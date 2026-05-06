from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _norm_station(value: object) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.lstrip("0") or "0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export USGS+WAM station points with monthly cfs for ArcGIS")
    p.add_argument("--base-dir", default=".", help="Workspace base directory")
    p.add_argument("--hsr", default="HSR1200", help="HSR directory name")
    p.add_argument("--wy", type=int, default=2018, help="Water year used by DAT file")
    p.add_argument("--month", type=int, default=1, help="Calendar month 1-12")
    p.add_argument("--output", default="output/gis/usgs_wam_points_2018_01.shp", help="Output path (.shp or .gpkg)")
    p.add_argument("--layer", default="usgs_wam_points", help="Layer name for gpkg output")
    p.add_argument(
        "--max-cfs",
        type=float,
        default=1_000_000.0,
        help="Upper cap for modeled cfs to prevent GIS field overflow on divergent reaches",
    )
    return p.parse_args()


def _wy_q_col_for_calendar_month(month: int) -> str:
    if month < 1 or month > 12:
        raise ValueError("--month must be in [1,12]")
    # WY columns are Q01..Q12 = Oct..Sep.
    wy_index = ((month + 2) % 12) + 1
    return f"Q{wy_index:02d}"


def _flowacc_col_for_calendar_month(month: int) -> str:
    if month < 1 or month > 12:
        raise ValueError("--month must be in [1,12]")
    return f"QAccCon{calendar.month_abbr[month]}"


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir).resolve()

    dat_path = base / args.hsr / "Streamflow" / f"ComIDStationDAMoAnQ{args.wy}_USGS_WAM_Combined.dat"
    if not dat_path.exists():
        raise FileNotFoundError(dat_path)

    usgs_csv = base / "inputData" / "inputs" / "monthly_wide_acft.csv"
    wam_csv = base / args.hsr / "Streamflow" / "Brazos_new_wam_locations_nhdplus.csv"
    station_comid_csv = base / args.hsr / "Flowlines" / "StationComID.csv"
    flowacc_csv = base / args.hsr / "Output" / "FlowAccum" / f"ComIDQ12WY{args.wy}.csv"
    if not usgs_csv.exists():
        raise FileNotFoundError(usgs_csv)
    if not wam_csv.exists():
        raise FileNotFoundError(wam_csv)
    if not station_comid_csv.exists():
        raise FileNotFoundError(station_comid_csv)
    if not flowacc_csv.exists():
        raise FileNotFoundError(flowacc_csv)

    qcols = [f"Q{i:02d}" for i in range(1, 14)]
    dat_cols = ["ComID", "Station", "AreaSqMi", *qcols]
    dat = pd.read_csv(dat_path, sep=r"\s+", header=None, names=dat_cols, engine="python")
    dat["Station"] = dat["Station"].map(_norm_station)
    dat["DAT_COMID"] = dat["ComID"].astype(str)

    qcol = _wy_q_col_for_calendar_month(args.month)
    pts = dat[["DAT_COMID", "Station", qcol]].copy().rename(columns={qcol: "OBS_CFS"})
    pts["OBS_CFS"] = pd.to_numeric(pts["OBS_CFS"], errors="coerce").fillna(0.0)

    usgs = pd.read_csv(usgs_csv)
    usgs = usgs[usgs["Year"] == 2018].copy()
    usgs["Station"] = usgs["CPID"].map(_norm_station)
    usgs_pts = usgs[["Station", "LAT", "LONG"]].dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")
    usgs_pts["SOURCE"] = "USGS"

    wam = pd.read_csv(wam_csv)
    wam["Station"] = wam["CPID"].map(_norm_station)
    wam_pts = wam.rename(columns={"lat": "LAT", "lon": "LONG"})[["Station", "LAT", "LONG"]].dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")
    wam_pts["SOURCE"] = "WAM"

    station_xy = pd.concat([usgs_pts, wam_pts], ignore_index=True)
    station_xy = station_xy.sort_values("SOURCE").drop_duplicates(subset=["Station"], keep="first")

    station_map = pd.read_csv(station_comid_csv)
    station_map["Station"] = station_map["Station"].map(_norm_station)
    station_map["MAP_COMID"] = pd.to_numeric(station_map["ComID"], errors="coerce").astype("Int64")
    station_map["SNAP_M"] = pd.to_numeric(station_map.get("snap_dist_m", 0.0), errors="coerce").fillna(0.0)
    station_map = station_map[["Station", "MAP_COMID", "SNAP_M"]].dropna(subset=["MAP_COMID"]).drop_duplicates(subset=["Station"], keep="first")
    station_map["MAP_COMID"] = station_map["MAP_COMID"].astype("int64")

    flowacc = pd.read_csv(flowacc_csv)
    fcol = _flowacc_col_for_calendar_month(args.month)
    if "ComIDVAA" not in flowacc.columns:
        raise ValueError(f"Missing ComIDVAA in {flowacc_csv}")
    if fcol not in flowacc.columns:
        raise ValueError(f"Missing {fcol} in {flowacc_csv}")
    flowacc["MAP_COMID"] = pd.to_numeric(flowacc["ComIDVAA"], errors="coerce").astype("Int64")
    flowacc["MOD_CFS"] = pd.to_numeric(flowacc[fcol], errors="coerce").fillna(0.0)
    flowacc = flowacc[["MAP_COMID", "MOD_CFS"]].dropna(subset=["MAP_COMID"]).drop_duplicates(subset=["MAP_COMID"], keep="first")
    flowacc["MAP_COMID"] = flowacc["MAP_COMID"].astype("int64")

    out = pts.merge(station_xy, on="Station", how="left")
    out = out.merge(station_map, on="Station", how="left")
    out = out.merge(flowacc, on="MAP_COMID", how="left")
    out = out.dropna(subset=["LAT", "LONG"]).copy()
    out["MOD_CFS"] = pd.to_numeric(out["MOD_CFS"], errors="coerce").fillna(0.0)

    clipped_n = 0
    if args.max_cfs is not None and args.max_cfs > 0:
        over = out["MOD_CFS"] > float(args.max_cfs)
        clipped_n = int(over.sum())
        if clipped_n > 0:
            out.loc[over, "MOD_CFS"] = float(args.max_cfs)

    out["DEL_CFS"] = out["MOD_CFS"] - out["OBS_CFS"]
    out["YEAR"] = 2018
    out["MONTH"] = int(args.month)
    out["WY"] = int(args.wy)

    out = out.rename(columns={"Station": "STA_ID"})

    keep_cols = [
        "STA_ID",
        "SOURCE",
        "MAP_COMID",
        "DAT_COMID",
        "OBS_CFS",
        "MOD_CFS",
        "DEL_CFS",
        "SNAP_M",
        "YEAR",
        "MONTH",
        "WY",
        "LAT",
        "LONG",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()

    gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["LONG"], out["LAT"]), crs="EPSG:4326")

    output = Path(args.output)
    if not output.is_absolute():
        output = (base / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".gpkg":
        gdf.to_file(output, layer=args.layer, driver="GPKG")
    else:
        gdf.to_file(output, driver="ESRI Shapefile")

    csv_out = output.with_suffix(".csv")
    out.to_csv(csv_out, index=False)

    print("AFINCH_POINT_EXPORT_COMPLETE")
    print(f"Output GIS: {output}")
    print(f"Output CSV: {csv_out}")
    print(f"Rows: {len(gdf):,}")
    print(f"USGS rows: {(gdf['SOURCE'] == 'USGS').sum():,}")
    print(f"WAM rows: {(gdf['SOURCE'] == 'WAM').sum():,}")
    print(f"Q column used: {qcol}")
    print(f"FlowAccum column used: {fcol}")
    print(f"Clipped MOD_CFS rows: {clipped_n:,}")


if __name__ == "__main__":
    main()
