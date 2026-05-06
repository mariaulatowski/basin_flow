#!/usr/bin/env python
"""
Create interactive map of routed COMID flows as a flowline choropleth.

Renders:
- NHD flowlines colored by modeled discharge (choropleth, Jan 2018)
- USGS + WAM constraint observation points as distinct markers
- Enforcement diagnostics popup on constraint reaches

Uses the AFINCH USGS+WAM output in output/nhd_medium_usgs_wam/.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_SUBDIR = "output/nhd_medium_usgs_wam"

# Fallback Brazos simplified flowline (used as thin grey reference)
FLOWLINES_SHP = BASE_DIR / "inputData" / "flowlines" / "Brazos_Flowline.shp"


CMAP_NAME = "YlOrRd"  # matplotlib colormap for flow coloring
FLOW_ACFT_CLIP = 50_000  # ac-ft/month — cap divergent AFINCH reaches for colour scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create interactive map from routed AFINCH outputs")
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Output directory containing modeled_monthly_comid_flows_from_points.csv and nhd_brazos_flowlines.gpkg",
    )
    p.add_argument("--year", type=int, default=2018, help="Year to map")
    p.add_argument("--month", type=int, default=1, help="Month to map (1-12)")
    p.add_argument(
        "--flow-source",
        choices=["afinch", "routed"],
        default="routed",
        help=(
            "Flow source: 'afinch' reads HSRxxxx/Output/FlowAccum/ComIDQ12WY*.csv (mass-balanced cfs); "
            "'routed' reads modeled_monthly_comid_flows_from_points.csv"
        ),
    )
    p.add_argument(
        "--afinch-hsr",
        default="HSR1200",
        help="HSR directory (used when --flow-source=afinch)",
    )
    p.add_argument(
        "--map-file",
        default=None,
        help="Optional output map filename (defaults to nhd_YYYYMM_flow_map.html inside --output-dir)",
    )
    return p.parse_args()


def _flow_to_hex(norm_val: float, cmap) -> str:
    """Convert a 0-1 normalised value to a hex colour string using `cmap`."""
    r, g, b, _ = cmap(float(np.clip(norm_val, 0.0, 1.0)))
    return mcolors.to_hex((r, g, b))


def _build_color_column(series: pd.Series, cmap) -> list[str]:
    """Map a Series of flow values (any range) to hex colours via log scale."""
    vals = series.to_numpy(dtype=float)
    log_vals = np.log1p(np.where(vals > 0, vals, 0))
    lo, hi = log_vals.min(), log_vals.max()
    if hi == lo:
        return ["#fee5d9"] * len(vals)
    norm = (log_vals - lo) / (hi - lo)
    return [_flow_to_hex(n, cmap) for n in norm]


def _calendar_to_wy(year: int, month: int) -> int:
    # Water year starts in October.
    return year + 1 if month >= 10 else year


def _load_flows_from_routed(routed_csv: Path, year: int, month: int) -> pd.DataFrame:
    routed_df = pd.read_csv(routed_csv)
    routed_df["date"] = pd.to_datetime(routed_df["date"], errors="coerce")
    routed_df = routed_df[
        (routed_df["date"].dt.year == year) & (routed_df["date"].dt.month == month)
    ].copy()
    routed_df["nhd_comid"] = pd.to_numeric(routed_df["nhd_comid"], errors="coerce")
    routed_df["flow_acft"] = pd.to_numeric(routed_df["flow_acft"], errors="coerce")
    routed_df["flow_cms"] = pd.to_numeric(routed_df["flow_cms"], errors="coerce")
    routed_df = routed_df.dropna(subset=["nhd_comid"]).copy()
    routed_df["nhd_comid"] = routed_df["nhd_comid"].astype("int64")
    routed_df["flow_cfs"] = routed_df["flow_cms"] / 0.028316846592
    flow_by_comid = routed_df.groupby("nhd_comid", as_index=False).agg(
        flow_cfs=("flow_cfs", "first"),
        flow_acft=("flow_acft", "first"),
    )
    return flow_by_comid


def _load_flows_from_afinch(base_dir: Path, hsr: str, year: int, month: int) -> pd.DataFrame:
    wy = _calendar_to_wy(year, month)
    month_abbrev = calendar.month_abbr[month]
    col = f"QAccCon{month_abbrev}"

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
            "nhd_comid": pd.to_numeric(df["ComIDVAA"], errors="coerce"),
            "flow_cfs": cfs,
            "flow_acft": acft,
        }
    ).dropna(subset=["nhd_comid"])
    out["nhd_comid"] = out["nhd_comid"].astype("int64")
    return out


def _catchment_gpkg_for_hsr(base_dir: Path, hsr: str) -> Path:
    code = str(hsr).strip().upper().replace("HSR", "")
    if not code.isdigit() or len(code) != 4:
        raise ValueError(f"Expected HSR like HSR1200, got: {hsr}")
    return base_dir / "inputData" / f"NHDPlusCatchment_{code}.gpkg"


def main() -> None:
    args = parse_args()
    if args.month < 1 or args.month > 12:
        raise ValueError("--month must be in [1, 12]")

    out_dir = (BASE_DIR / args.output_dir).resolve()
    nhd_gpkg = out_dir / "nhd_brazos_flowlines.gpkg"
    routed_csv = out_dir / "modeled_monthly_comid_flows_from_points.csv"
    snapped_csv = out_dir / "snapped_point_diagnostics.csv"
    enforcement_csv = out_dir / "enforcement_diagnostics.csv"
    if args.map_file:
        output_map = Path(args.map_file).resolve()
    else:
        output_map = out_dir / f"nhd_{args.year}{args.month:02d}_flow_map.html"

    for p in [nhd_gpkg]:
        if not p.exists():
            raise FileNotFoundError(p)
    if args.flow_source == "routed" and not routed_csv.exists():
        raise FileNotFoundError(routed_csv)

    cmap = plt.get_cmap(CMAP_NAME)

    # ------------------------------------------------------------------ #
    # Load routed flows for Jan 2018                                       #
    # ------------------------------------------------------------------ #
    if args.flow_source == "afinch":
        print(f"Loading AFINCH accumulated flows ({args.year}-{args.month:02d}) from {args.afinch_hsr} ...")
        flow_by_comid = _load_flows_from_afinch(BASE_DIR, args.afinch_hsr, args.year, args.month)
    else:
        print(f"Loading routed monthly flows ({args.year}-{args.month:02d}) ...")
        flow_by_comid = _load_flows_from_routed(routed_csv, args.year, args.month)
        # Helpful warning for known clipping artifact in routed products.
        if len(flow_by_comid) > 0:
            n_cap = int((np.isclose(flow_by_comid["flow_cfs"] * 0.028316846592, 10000.0)).sum())
            if n_cap > 0:
                print(f"WARNING: detected {n_cap:,} reaches at 10,000 cms cap in routed source.")
    print(f"  {len(flow_by_comid):,} COMIDs with flow data")

    # ------------------------------------------------------------------ #
    # Load geometry and merge flow                                        #
    # ------------------------------------------------------------------ #
    if args.flow_source == "afinch":
        catchment_gpkg = _catchment_gpkg_for_hsr(BASE_DIR, args.afinch_hsr)
        if not catchment_gpkg.exists():
            raise FileNotFoundError(catchment_gpkg)
        print("Loading THS catchment geometry (NHDPlusID) for AFINCH output join ...")
        fl_gdf = gpd.read_file(str(catchment_gpkg)).to_crs(4326)
        fl_gdf = fl_gdf[~fl_gdf.geometry.is_empty & fl_gdf.geometry.notna()].copy()
        fl_gdf["COMID"] = pd.to_numeric(fl_gdf["NHDPlusID"], errors="coerce")
        fl_gdf = fl_gdf.dropna(subset=["COMID"]).copy()
        fl_gdf["COMID"] = fl_gdf["COMID"].astype("int64")
        print(f"  {len(fl_gdf):,} catchment features loaded")
        print("Simplifying geometries ...")
        fl_gdf["geometry"] = fl_gdf.geometry.simplify(0.001, preserve_topology=True)
    else:
        print("Loading NHD flowlines from GeoPackage ...")
        fl_gdf = gpd.read_file(str(nhd_gpkg), layer="flowlines").to_crs(4326)
        fl_gdf = fl_gdf[~fl_gdf.geometry.is_empty & fl_gdf.geometry.notna()].copy()
        fl_gdf["COMID"] = pd.to_numeric(fl_gdf["COMID"], errors="coerce")
        fl_gdf = fl_gdf.dropna(subset=["COMID"]).copy()
        fl_gdf["COMID"] = fl_gdf["COMID"].astype("int64")
        print(f"  {len(fl_gdf):,} flowline features loaded")

        # Drop reaches shorter than 100 m — they add noise without visible contribution
        fl_gdf = fl_gdf[fl_gdf["lengthkm"] >= 0.1].copy()
        print(f"  {len(fl_gdf):,} reaches after removing <0.1 km segments")

        # Simplify geometry for browser performance (≈500 m tolerance)
        print("Simplifying geometries ...")
        fl_gdf["geometry"] = fl_gdf.geometry.simplify(0.005, preserve_topology=True)

    # Merge flow data
    fl_merged = fl_gdf.merge(
        flow_by_comid[["nhd_comid", "flow_cfs", "flow_acft"]],
        left_on="COMID",
        right_on="nhd_comid",
        how="left",
    )
    fl_merged["flow_acft"] = fl_merged["flow_acft"].fillna(0.0)
    fl_merged["flow_cfs"] = fl_merged["flow_cfs"].fillna(0.0)

    # Assign pre-computed hex colours (log scale — rivers vary many orders of magnitude)
    # Clip divergent AFINCH outliers before colour mapping
    display_flow = fl_merged["flow_acft"].clip(upper=FLOW_ACFT_CLIP)
    fl_merged["_color"] = _build_color_column(display_flow, cmap)
    # Line weight: thin for dry/low-flow, thicker for high-flow
    log_vals = np.log1p(display_flow.clip(lower=0).to_numpy(dtype=float))
    lo, hi = log_vals.min(), log_vals.max()
    norm_w = (log_vals - lo) / (hi - lo) if hi > lo else np.zeros(len(log_vals))
    fl_merged["_weight"] = (0.5 + norm_w * 3.0).round(2)

    # Keep only the columns we need for the GeoJSON (keeps file size down)
    fl_all = fl_merged[["COMID", "geometry", "flow_cfs", "flow_acft", "_color", "_weight"]].copy()
    # Split into major/minor buckets for diagnostics.
    _major_thresh = 50.0
    fl_export = fl_all[fl_all["flow_acft"] >= _major_thresh].copy()
    fl_minor = fl_all[fl_all["flow_acft"] < _major_thresh].copy()
    print(f"  Major reaches (>={_major_thresh:.0f} ac-ft): {len(fl_export):,}")
    print(f"  Minor reaches (<{_major_thresh:.0f} ac-ft): {len(fl_minor):,}")

    # ------------------------------------------------------------------ #
    # Build folium map                                                     #
    # ------------------------------------------------------------------ #
    center_source = fl_export if len(fl_export) > 0 else fl_all
    center_lat = center_source.geometry.centroid.y.median()
    center_lon = center_source.geometry.centroid.x.median()
    if not np.isfinite(center_lat) or not np.isfinite(center_lon):
        center_lat = fl_gdf.geometry.centroid.y.median()
        center_lon = fl_gdf.geometry.centroid.x.median()
    print(f"Creating map centered at ({center_lat:.4f}, {center_lon:.4f}) ...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ---- Flowline choropleth layer ------------------------------------ #
    print("Building flowline choropleth layer ...")
    if args.flow_source == "afinch":
        layer_title = f"AFINCH Accumulated Flow — {args.year}-{args.month:02d} (all reaches)"
    else:
        layer_title = f"NHD Major Flow — {args.year}-{args.month:02d} (≥50 ac-ft, {len(fl_export):,} reaches)"

    fl_fg = folium.FeatureGroup(name=layer_title, show=True)

    # In AFINCH mode, render all reaches/catchments so the full accumulation
    # output is visible. In routed mode, keep major-flow focus for performance.
    if args.flow_source == "afinch":
        flow_for_render = fl_all
    else:
        flow_for_render = fl_export if len(fl_export) > 0 else fl_all
    rendered_count = len(flow_for_render)
    if args.flow_source != "afinch" and len(fl_export) == 0:
        print("WARNING: no reaches met major-flow threshold; rendering all reaches instead.")

    has_polygon_geom = any(gt in {"Polygon", "MultiPolygon"} for gt in flow_for_render.geometry.geom_type.unique())

    def _style(feature):
        props = feature["properties"]
        if has_polygon_geom:
            return {
                "color": "#777777",
                "weight": 0.3,
                "opacity": 0.5,
                "fillColor": props.get("_color", "#cccccc"),
                "fillOpacity": 0.75,
            }
        return {
            "color": props.get("_color", "#cccccc"),
            "weight": props.get("_weight", 0.8),
            "opacity": 0.85,
        }

    def _highlight(feature):
        return {"weight": 4, "color": "#333333", "opacity": 1.0}

    folium.GeoJson(
        data=flow_for_render.to_json(),
        style_function=_style,
        highlight_function=_highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=["COMID", "flow_cfs", "flow_acft"],
            aliases=["NHD COMID:", "Flow (cfs):", "Flow (ac-ft/mo):"],
            localize=True,
            sticky=False,
        ),
    ).add_to(fl_fg)
    fl_fg.add_to(m)

    # ---- Constraint observation points from combined DAT file -------- #
    print("Loading station data from combined DAT file ...")
    dat_path = BASE_DIR / args.afinch_hsr / "Streamflow" / f"ComIDStationDAMoAnQ2018_USGS_WAM_Combined.dat"
    
    if dat_path.exists():
        # Read DAT file: ComID, StationID, AreaSqKm, Q01-Q13
        dat_data = []
        with open(dat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        comid = parts[0]
                        station_id = parts[1]
                        flow_cfs = float(parts[3])  # Q01 = January, column index 3, already in cfs
                        dat_data.append({
                            'comid': comid,
                            'station_id': station_id,
                            'flow_cfs': flow_cfs,
                        })
                    except (ValueError, IndexError):
                        continue
        
        dat_df = pd.DataFrame(dat_data)
        print(f"  Loaded {len(dat_df):,} stations from DAT file")
        
        # Load USGS coordinates from monthly_wide_acft.csv
        usgs_csv = BASE_DIR / "inputData" / "inputs" / "monthly_wide_acft.csv"
        usgs_coords = {}
        if usgs_csv.exists():
            usgs_df = pd.read_csv(usgs_csv)
            # Keep only 2018 data and unique coords
            usgs_2018 = usgs_df[usgs_df['Year'] == 2018].drop_duplicates(subset=['CPID'])[['CPID', 'LAT', 'LONG']].copy()
            usgs_coords = {str(row['CPID']): (row['LAT'], row['LONG']) for _, row in usgs_2018.iterrows()}
            print(f"  Loaded {len(usgs_coords):,} USGS station coordinates")
        
        # Load WAM coordinates from corrected locations file
        wam_csv = BASE_DIR / args.afinch_hsr / "Streamflow" / "Brazos_new_wam_locations_nhdplus.csv"
        wam_coords = {}
        if wam_csv.exists():
            wam_df = pd.read_csv(wam_csv)
            wam_coords = {str(row['CPID']): (row['lat'], row['lon']) for _, row in wam_df.iterrows()}
            print(f"  Loaded {len(wam_coords):,} WAM station coordinates")
        
        # Create feature groups
        usgs_fg = folium.FeatureGroup(name="USGS Gauge Points", show=True)
        wam_fg = folium.FeatureGroup(name="WAM Control Points", show=True)
        
        usgs_count = 0
        wam_count = 0
        
        for _, row in dat_df.iterrows():
            station_id = str(row['station_id'])
            flow_cfs = float(row['flow_cfs'])
            
            # Determine source and get coordinates
            if station_id in usgs_coords:
                source = "USGS"
                lat, lon = usgs_coords[station_id]
                fg = usgs_fg
                color = "#1a78c2"
                usgs_count += 1
            elif station_id in wam_coords:
                source = "WAM"
                lat, lon = wam_coords[station_id]
                fg = wam_fg
                color = "#d7191c"
                wam_count += 1
            else:
                # Skip stations without coordinates
                continue
            
            # Convert cfs to ac-ft/month for display (January = 31 days)
            days_jan = 31
            flow_acft = flow_cfs * days_jan * 86400.0 / 43560.0
            
            popup_html = f"""
            <div style="font-family: monospace; font-size: 11px; width: 260px;">
                <b>Station ID:</b> {station_id}<br>
                <b>Source:</b> {source}<br>
                <b>ComID:</b> {row['comid']}<br>
                <hr style="margin: 4px 0;">
                <b>January 2018 Flow:</b><br>
                &nbsp;&nbsp;{flow_cfs:,.2f} cfs<br>
                &nbsp;&nbsp;{flow_acft:,.1f} ac-ft<br>
                <hr style="margin: 4px 0;">
                <b>Lat:</b> {lat:.5f} &nbsp; <b>Lon:</b> {lon:.5f}
            </div>
            """
            
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="white",
                weight=1.5,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{source} {station_id}: {flow_cfs:,.2f} cfs",
            )
            marker.add_to(fg)
        
        usgs_fg.add_to(m)
        wam_fg.add_to(m)
        print(f"  Added {usgs_count:,} USGS points and {wam_count:,} WAM points")

    # ---- Legend ------------------------------------------------------- #
    # Build colour ramp swatches from the matplotlib cmap (in cfs)
    # Convert ac-ft bounds to cfs for display: cfs = acft / (days * 86400 / 43560)
    swatch_bins_acft = [0, 20, 100, 500, 2000, 10000]
    swatch_bins_cfs = [v / (31 * 86400.0 / 43560.0) for v in swatch_bins_acft]
    log_lo = np.log1p(0)
    log_hi = np.log1p(FLOW_ACFT_CLIP)
    swatches_html = ""
    for acft_v, cfs_v in zip(swatch_bins_acft, swatch_bins_cfs):
        norm_val = (np.log1p(acft_v) - log_lo) / (log_hi - log_lo) if log_hi > log_lo else 0.5
        hex_c = _flow_to_hex(norm_val, cmap)
        swatches_html += (
            f'<span style="display:inline-block;width:18px;height:12px;'
            f'background:{hex_c};border:1px solid #aaa;margin-right:4px;"></span>'
            f'{cfs_v:,.0f} cfs<br>\n'
        )

    legend_html = f"""
    <div style="position:fixed;bottom:50px;right:50px;width:240px;background:white;
         border:2px solid #888;z-index:9999;font-size:12px;padding:10px;
         border-radius:6px;box-shadow:0 0 6px rgba(0,0,0,0.3);">
        <b>{args.year}-{args.month:02d} — Brazos AFINCH Flow</b><br>
        <i style="font-size:10px;">All units in <b>cfs</b> (cubic feet per second)</i><br>
        <i style="font-size:10px;">NHD+ reaches ({rendered_count:,} total)</i><br><br>
        <b>Flowline color (cfs):</b><br>
        {swatches_html}
        <hr style="margin:6px 0;">
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
             background:#1a78c2;border:1.5px solid white;"></span> USGS gauge<br>
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
             background:#d7191c;border:1.5px solid white;"></span> WAM control point<br>
        <hr style="margin:6px 0;">
        <small>Hover flowlines for discharge.<br>Click station points for details.</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save map
    output_map.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_map))

    print(f"\nSaved interactive map: {output_map}")
    print(f"NHD flowlines rendered: {rendered_count:,}")
    sane_flows = fl_merged["flow_acft"][fl_merged["flow_acft"].between(0, FLOW_ACFT_CLIP)]
    print(f"Reaches with sane flow (<={FLOW_ACFT_CLIP:,} ac-ft): {len(sane_flows):,}")
    print(f"Overflow/diverged reaches clipped: {(fl_merged['flow_acft'] > FLOW_ACFT_CLIP).sum():,}")
    print(f"Sane flow range: {sane_flows.min():.4f} – {sane_flows.max():.0f} ac-ft/mo")


if __name__ == "__main__":
    main()
