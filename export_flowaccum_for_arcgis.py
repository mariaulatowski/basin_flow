from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export AFINCH accumulated flow for ArcGIS Pro")
    p.add_argument("--base-dir", default=".", help="Project base directory")
    p.add_argument("--hsr", default="HSR1200", help="HSR key (e.g., HSR1200)")
    p.add_argument("--wy", type=int, required=True, help="Water year to export, e.g. 2018")
    p.add_argument(
        "--catchment-gpkg",
        default="inputData/NHDPlusCatchment_1205.gpkg",
        help="Catchment geometry with NHDPlusID matching ComIDVAA",
    )
    p.add_argument(
        "--output-dir",
        default="output/arcgis_exports",
        help="Directory for ArcGIS-ready exports",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = (base_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    flowacc_csv = base_dir / args.hsr / "Output" / "FlowAccum" / f"ComIDQ12WY{args.wy}.csv"
    if not flowacc_csv.exists():
        raise FileNotFoundError(flowacc_csv)

    catchment_path = (base_dir / args.catchment_gpkg).resolve()
    if not catchment_path.exists():
        raise FileNotFoundError(catchment_path)

    print(f"Reading accumulated flow table: {flowacc_csv}")
    flow = pd.read_csv(flowacc_csv)
    flow["ComIDVAA"] = pd.to_numeric(flow["ComIDVAA"], errors="coerce")
    flow = flow.dropna(subset=["ComIDVAA"]).copy()
    flow["ComIDVAA"] = flow["ComIDVAA"].astype("int64")

    # ArcGIS-friendly table export
    table_out = out_dir / f"ComIDQ12WY{args.wy}_table.csv"
    flow.to_csv(table_out, index=False)
    print(f"Wrote table: {table_out}")

    print(f"Reading catchment geometry: {catchment_path}")
    catch = gpd.read_file(catchment_path)
    if "NHDPlusID" not in catch.columns:
        raise ValueError(f"Expected NHDPlusID field in {catchment_path}")
    catch["NHDPlusID"] = pd.to_numeric(catch["NHDPlusID"], errors="coerce")
    catch = catch.dropna(subset=["NHDPlusID"]).copy()
    catch["NHDPlusID"] = catch["NHDPlusID"].astype("int64")

    merged = catch.merge(flow, left_on="NHDPlusID", right_on="ComIDVAA", how="left")

    overlap = int(merged["ComIDVAA"].notna().sum())
    print(f"Joined features with flow values: {overlap:,} / {len(merged):,}")

    gpkg_out = out_dir / f"ComIDQ12WY{args.wy}_catchments.gpkg"
    merged.to_file(gpkg_out, driver="GPKG")
    print(f"Wrote GeoPackage: {gpkg_out}")

    shp_out = out_dir / f"ComIDQ12WY{args.wy}_catchments.shp"
    merged.to_file(shp_out, driver="ESRI Shapefile")
    print(f"Wrote Shapefile: {shp_out}")

    print("Done. Load the GeoPackage or Shapefile in ArcGIS Pro.")


if __name__ == "__main__":
    main()
