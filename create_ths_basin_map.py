#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CATCHMENT_GPKG = BASE_DIR / "inputData" / "NHDPlusCatchment_1201.gpkg"
DEFAULT_BRAZOS_FLOWLINES = BASE_DIR / "inputData" / "flowlines" / "Brazos_Flowline.shp"
DEFAULT_THS_FLOWLINES = BASE_DIR / "HSR1200" / "Flowlines" / "nhdflowline.txt"
DEFAULT_OUT = BASE_DIR / "output" / "ths_brazos_map.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive map of THS reaches in the Brazos basin")
    parser.add_argument("--ths-flowlines", default=str(DEFAULT_THS_FLOWLINES), help="THS nhdflowline.txt file")
    parser.add_argument("--catchment-gpkg", default=str(DEFAULT_CATCHMENT_GPKG), help="THS catchment geopackage")
    parser.add_argument("--brazos-flowlines", default=str(DEFAULT_BRAZOS_FLOWLINES), help="Brazos reference flowline shapefile")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output HTML map path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ths_path = Path(args.ths_flowlines)
    catchment_path = Path(args.catchment_gpkg)
    brazos_flowline_path = Path(args.brazos_flowlines)
    out_path = Path(args.out)

    if not ths_path.exists():
        raise FileNotFoundError(ths_path)
    if not catchment_path.exists():
        raise FileNotFoundError(catchment_path)

    print(f"Reading THS reach list: {ths_path}")
    ths_df = pd.read_csv(ths_path)
    ths_df["ComID"] = pd.to_numeric(ths_df["ComID"], errors="coerce")
    ths_df = ths_df.dropna(subset=["ComID"]).copy()
    ths_df["ComID"] = ths_df["ComID"].astype("int64")
    ths_comids = set(ths_df["ComID"].tolist())
    print(f"  THS reaches listed: {len(ths_df):,}")

    print(f"Reading THS catchment geometry: {catchment_path}")
    catchment_gdf = gpd.read_file(catchment_path).to_crs(4326)
    catchment_gdf["NHDPlusID"] = pd.to_numeric(catchment_gdf["NHDPlusID"], errors="coerce")
    catchment_gdf = catchment_gdf.dropna(subset=["NHDPlusID"]).copy()
    catchment_gdf["NHDPlusID"] = catchment_gdf["NHDPlusID"].astype("int64")

    ths_gdf = catchment_gdf[catchment_gdf["NHDPlusID"].isin(ths_comids)].copy()
    print(f"  THS catchments available: {len(catchment_gdf):,}")
    print(f"  THS catchments matched from reach table: {len(ths_gdf):,}")

    if ths_gdf.empty:
        raise ValueError("No THS ComIDs matched the THS catchment geometry.")

    brazos_flow_gdf = None
    if brazos_flowline_path.exists():
        print(f"Reading Brazos reference flowlines: {brazos_flowline_path}")
        brazos_flow_gdf = gpd.read_file(brazos_flowline_path).to_crs(4326)
        brazos_flow_gdf = brazos_flow_gdf[~brazos_flow_gdf.geometry.is_empty & brazos_flow_gdf.geometry.notna()].copy()
        print(f"  Brazos reference flowlines: {len(brazos_flow_gdf):,}")

    center_lat = float(ths_gdf.geometry.centroid.y.median())
    center_lon = float(ths_gdf.geometry.centroid.x.median())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if brazos_flow_gdf is not None:
        brazos_layer = folium.FeatureGroup(name=f"Brazos reference flowlines ({len(brazos_flow_gdf):,})", show=True)
        folium.GeoJson(
            data=brazos_flow_gdf[["geometry"]].to_json(),
            style_function=lambda _: {"color": "#909090", "weight": 1.2, "opacity": 0.45},
        ).add_to(brazos_layer)
        brazos_layer.add_to(fmap)

    ths_layer = folium.FeatureGroup(name=f"THS catchments/reaches ({len(ths_gdf):,})", show=True)
    folium.GeoJson(
        data=ths_gdf[["NHDPlusID", "GridCode", "geometry"]].to_json(),
        style_function=lambda _: {
            "color": "#005f99",
            "weight": 1.0,
            "fillColor": "#4ea5d9",
            "fillOpacity": 0.35,
            "opacity": 0.9,
        },
        highlight_function=lambda _: {"color": "#ff7f11", "weight": 2.0, "fillOpacity": 0.55, "opacity": 1.0},
        tooltip=folium.GeoJsonTooltip(fields=["NHDPlusID", "GridCode"], aliases=["THS ComID:", "GridCode:"]),
    ).add_to(ths_layer)
    ths_layer.add_to(fmap)

    legend_html = f"""
    <div style="position:fixed;bottom:40px;right:40px;width:260px;background:white;
         border:2px solid #888;z-index:9999;font-size:12px;padding:10px;
         border-radius:6px;box-shadow:0 0 6px rgba(0,0,0,0.3);">
        <b>Brazos THS Network</b><br>
        <small>Blue polygons are THS catchments keyed to the THS reach ComIDs used by the converted AFINCH run.</small><br><br>
        <span style="display:inline-block;width:18px;height:10px;background:#4ea5d9;border:1px solid #005f99;"></span>
        THS catchments: {len(ths_gdf):,}<br>
        <span style="display:inline-block;width:18px;height:3px;background:#909090;"></span>
        Brazos reference flowlines<br><br>
        <small>This shows the THS model domain for Brazos. The model routes on the THS network tables; the grey line layer is just geographic reference.</small>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print(f"Saved THS Brazos map: {out_path}")


if __name__ == "__main__":
    main()