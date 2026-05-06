from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _calendar_to_wy(year: int, month: int) -> int:
    return year + 1 if month >= 10 else year


def _month_col(month: int) -> str:
    return f"QAccCon{calendar.month_abbr[month]}"


def _load_monthly_flow(base_dir: Path, hsr: str, year: int, month: int) -> pd.DataFrame:
    wy = _calendar_to_wy(year, month)
    col = _month_col(month)
    flowacc_path = base_dir / hsr / "Output" / "FlowAccum" / f"ComIDQ12WY{wy}.csv"
    if not flowacc_path.exists():
        raise FileNotFoundError(flowacc_path)

    df = pd.read_csv(flowacc_path)
    if "ComIDVAA" not in df.columns:
        raise ValueError(f"Missing ComIDVAA in {flowacc_path}")
    if col not in df.columns:
        raise ValueError(f"Missing {col} in {flowacc_path}")

    days = calendar.monthrange(year, month)[1]
    cfs = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    acft = cfs * days * 86400.0 / 43560.0

    out = pd.DataFrame(
        {
            "COMID": pd.to_numeric(df["ComIDVAA"], errors="coerce"),
            "FLOW_CFS": cfs,
            "FLOW_ACFT": acft,
        }
    ).dropna(subset=["COMID"])
    out["COMID"] = out["COMID"].astype("int64")
    return out


def _load_geometry(base_dir: Path, ths: str) -> gpd.GeoDataFrame:
    gpkg = base_dir / "inputData" / f"NHDPlusCatchment_{ths}.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(gpkg)

    gdf = gpd.read_file(gpkg)
    if "NHDPlusID" not in gdf.columns:
        raise ValueError(f"Missing NHDPlusID in {gpkg}")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf["COMID"] = pd.to_numeric(gdf["NHDPlusID"], errors="coerce")
    gdf = gdf.dropna(subset=["COMID"]).copy()
    gdf["COMID"] = gdf["COMID"].astype("int64")
    return gdf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export AFINCH monthly flow output to ArcGIS-friendly GIS file")
    p.add_argument("--base-dir", default=".", help="Workspace base directory")
    p.add_argument("--hsr", default="HSR1200", help="HSR folder holding AFINCH FlowAccum output")
    p.add_argument("--ths", default="1200", help="THS code used by catchment GeoPackage")
    p.add_argument("--year", type=int, default=2018, help="Calendar year")
    p.add_argument("--month", type=int, default=1, help="Calendar month 1-12")
    p.add_argument("--output", default="output/gis/afinch_flow_201801.gpkg", help="Output GIS path")
    p.add_argument("--layer", default="afinch_flow", help="Layer name for GeoPackage output")
    p.add_argument("--format", choices=["gpkg", "shp"], default="gpkg", help="Output format")
    p.add_argument("--nonzero-only", action="store_true", help="Keep only reaches with FLOW_CFS > 0")
    p.add_argument(
        "--max-cfs",
        type=float,
        default=1_000_000.0,
        help="Upper cap for FLOW_CFS to prevent GIS field overflow on divergent reaches",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.month < 1 or args.month > 12:
        raise ValueError("--month must be in [1,12]")

    base_dir = Path(args.base_dir).resolve()
    out_path = (base_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flow_df = _load_monthly_flow(base_dir, args.hsr, args.year, args.month)
    geom_gdf = _load_geometry(base_dir, args.ths)

    merged = geom_gdf.merge(flow_df, on="COMID", how="left")
    merged["FLOW_CFS"] = pd.to_numeric(merged["FLOW_CFS"], errors="coerce")
    merged["FLOW_CFS"] = merged["FLOW_CFS"].where(pd.Series(pd.notna(merged["FLOW_CFS"]) & pd.Series(pd.to_numeric(merged["FLOW_CFS"], errors="coerce")).notna()), 0.0)
    merged["FLOW_CFS"] = merged["FLOW_CFS"].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    clipped_n = 0
    if args.max_cfs is not None and args.max_cfs > 0:
        over = merged["FLOW_CFS"] > float(args.max_cfs)
        clipped_n = int(over.sum())
        if clipped_n > 0:
            merged.loc[over, "FLOW_CFS"] = float(args.max_cfs)

    days = calendar.monthrange(args.year, args.month)[1]
    merged["FLOW_ACFT"] = merged["FLOW_CFS"] * days * 86400.0 / 43560.0
    merged["YEAR"] = int(args.year)
    merged["MONTH"] = int(args.month)
    merged["WY"] = int(_calendar_to_wy(args.year, args.month))

    if args.nonzero_only:
        merged = merged[merged["FLOW_CFS"] > 0].copy()

    keep_cols = [c for c in ["COMID", "GridCode", "ReachCode", "FLOW_CFS", "FLOW_ACFT", "YEAR", "MONTH", "WY", "geometry"] if c in merged.columns]
    merged = merged[keep_cols].copy()

    if args.format == "gpkg":
        merged.to_file(out_path, layer=args.layer, driver="GPKG")
    else:
        merged.to_file(out_path, driver="ESRI Shapefile")

    print("AFINCH_ARCGIS_EXPORT_COMPLETE")
    print(f"Output: {out_path}")
    if args.format == "gpkg":
        print(f"Layer: {args.layer}")
    print(f"Rows: {len(merged):,}")
    print(f"Nonzero FLOW_CFS rows: {(merged['FLOW_CFS'] > 0).sum():,}")
    print(f"Clipped FLOW_CFS rows: {clipped_n:,}")


if __name__ == "__main__":
    main()
