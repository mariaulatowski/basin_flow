from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FLOWLINE_PATH = BASE_DIR / "inputData" / "flowlines" / "Brazos_Flowline.shp"
FISHNET_PATH = BASE_DIR / "inputData" / "inputs" / "hc_fishnet_shapefile" / "Fishnet_HillCountry.shp"
ROUTED_FLOW_PATH = BASE_DIR / "output" / "brazos" / "modeled_monthly_comid_flows_from_points.csv"
USGS_POINTS_PATH = BASE_DIR / "inputData" / "inputs" / "monthly_wide_acft.csv"
WAM_POINTS_PATH = BASE_DIR / "inputData" / "inputs" / "monthly_wide_acft_from_hecdss.csv"
XW_DIAG_PATH = BASE_DIR / "output" / "brazos" / "brazos_to_nhd_crosswalk_diagnostics.csv"
MAP_OUTPUT_PATH = BASE_DIR / "output" / "usgs_wam_flowlines_texas_map.html"


def fmt_discharge(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.6f}"


def prepare_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    for col in ["CPID", "Type", "ID", "Gage_ID", "LAT", "LONG", "Year", "JAN"]:
        if col not in df.columns:
            df[col] = ""
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LONG"] = pd.to_numeric(df["LONG"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["JAN"] = pd.to_numeric(df["JAN"], errors="coerce")
    df_2018 = df[df["Year"] == 2018].copy()
    out = df_2018.drop_duplicates(subset=["CPID"]).copy() if not df_2018.empty else df.drop_duplicates(subset=["CPID"]).copy()
    return out.dropna(subset=["LAT", "LONG"]).copy()


def main() -> None:
    for p in [FLOWLINE_PATH, FISHNET_PATH, ROUTED_FLOW_PATH, USGS_POINTS_PATH, WAM_POINTS_PATH, XW_DIAG_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    flow_gdf = gpd.read_file(FLOWLINE_PATH).to_crs(4326)
    flow_gdf = flow_gdf[~flow_gdf.geometry.is_empty & flow_gdf.geometry.notna()].copy()
    flow_gdf["geometry"] = flow_gdf.geometry.simplify(0.00015, preserve_topology=True)
    flow_gdf["COMID"] = pd.to_numeric(flow_gdf["COMID"], errors="coerce")
    flow_gdf = flow_gdf.dropna(subset=["COMID"]).copy()
    flow_gdf["COMID"] = flow_gdf["COMID"].astype("int64")

    fishnet_gdf = gpd.read_file(FISHNET_PATH).to_crs(4326)
    fishnet_gdf = fishnet_gdf[~fishnet_gdf.geometry.is_empty & fishnet_gdf.geometry.notna()].copy()
    fishnet_gdf["geometry"] = fishnet_gdf.geometry.simplify(0.00005, preserve_topology=True)

    routed_df = pd.read_csv(ROUTED_FLOW_PATH)
    routed_df["date"] = pd.to_datetime(routed_df.get("date"), errors="coerce")
    routed_df = routed_df[(routed_df["date"].dt.year == 2018) & (routed_df["date"].dt.month == 1)].copy()
    routed_df["brazos_comid"] = pd.to_numeric(routed_df.get("brazos_comid"), errors="coerce")
    routed_df["flow_cms"] = pd.to_numeric(routed_df.get("flow_cms"), errors="coerce")
    routed_df["flow_acft"] = pd.to_numeric(routed_df.get("flow_acft"), errors="coerce")
    routed_df = routed_df.dropna(subset=["brazos_comid"]).copy()
    routed_df["brazos_comid"] = routed_df["brazos_comid"].astype("int64")

    routed_jan = routed_df.groupby("brazos_comid", as_index=False).agg(
        flow_cms=("flow_cms", "sum"),
        flow_acft=("flow_acft", "sum"),
    )

    xw = pd.read_csv(XW_DIAG_PATH)
    xw["brazos_comid"] = pd.to_numeric(xw["brazos_comid"], errors="coerce")
    xw["proximity_ratio"] = pd.to_numeric(xw.get("proximity_ratio"), errors="coerce")
    xw["matched_flag"] = xw.get("matched_flag", False).fillna(False).astype(bool)
    xw = xw.dropna(subset=["brazos_comid"]).copy()
    xw["brazos_comid"] = xw["brazos_comid"].astype("int64")

    xw_hc = xw[(xw["matched_flag"]) & (xw["proximity_ratio"] >= 0.10)].copy()
    hc_comids = set(xw_hc["brazos_comid"].tolist())

    flow_layer_data = flow_gdf.merge(routed_jan, how="left", left_on="COMID", right_on="brazos_comid")
    flow_layer_data["flow_cms_txt"] = flow_layer_data["flow_cms"].apply(fmt_discharge)
    flow_layer_data["flow_acft_txt"] = flow_layer_data["flow_acft"].apply(fmt_discharge)

    flow_hc = flow_layer_data[flow_layer_data["COMID"].isin(hc_comids)].copy()
    flow_hc = flow_hc.merge(
        xw_hc[["brazos_comid", "proximity_ratio", "match_dist_m"]],
        how="left",
        left_on="COMID",
        right_on="brazos_comid",
    )
    flow_hc["proximity_txt"] = flow_hc["proximity_ratio"].apply(fmt_discharge)
    flow_hc["match_dist_txt"] = flow_hc["match_dist_m"].apply(fmt_discharge)

    flow_fields = [c for c in ["COMID", "GNIS_NAME", "LENGTHKM", "REACHCODE", "flow_cms_txt", "flow_acft_txt"] if c in flow_layer_data.columns]
    fishnet_fields = [c for c in fishnet_gdf.columns if c != "geometry"][:5]

    usgs_df = prepare_points(USGS_POINTS_PATH)
    wam_df = prepare_points(WAM_POINTS_PATH)

    m = folium.Map(location=[31.0, -99.0], zoom_start=6, tiles="CartoDB positron", control_scale=True)
    m.fit_bounds([[25.8, -106.7], [36.6, -93.4]])

    # Enforce visual stacking order.
    folium.map.CustomPane("pane_fishnet", z_index=350).add_to(m)
    folium.map.CustomPane("pane_flowlines", z_index=450).add_to(m)
    folium.map.CustomPane("pane_points", z_index=650).add_to(m)

    fishnet_fg = folium.FeatureGroup(name=f"HC fishnet ({len(fishnet_gdf):,})", show=True)
    folium.GeoJson(
        fishnet_gdf[fishnet_fields + ["geometry"]].to_json() if fishnet_fields else fishnet_gdf[["geometry"]].to_json(),
        pane="pane_fishnet",
        style_function=lambda _: {"color": "#6a51a3", "weight": 0.6, "opacity": 0.55, "fillColor": "#9e9ac8", "fillOpacity": 0.03},
    ).add_to(fishnet_fg)
    fishnet_fg.add_to(m)

    flow_fg = folium.FeatureGroup(name=f"Flowlines ({len(flow_layer_data):,})", show=True)
    folium.GeoJson(
        flow_layer_data[flow_fields + ["geometry"]].to_json() if flow_fields else flow_layer_data[["geometry"]].to_json(),
        pane="pane_flowlines",
        style_function=lambda _: {"color": "#2b8cbe", "weight": 1.2, "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=flow_fields, aliases=flow_fields, sticky=False, localize=True) if flow_fields else None,
        popup=folium.GeoJsonPopup(fields=flow_fields, aliases=flow_fields, localize=True, labels=True) if flow_fields else None,
    ).add_to(flow_fg)
    flow_fg.add_to(m)

    flow_hc_fields = [
        c
        for c in ["COMID", "GNIS_NAME", "LENGTHKM", "REACHCODE", "flow_cms_txt", "flow_acft_txt", "proximity_txt", "match_dist_txt"]
        if c in flow_hc.columns
    ]
    flow_hc_aliases = [
        "Jan 2018 Flow (cms)" if c == "flow_cms_txt" else
        "Jan 2018 Flow (ac-ft)" if c == "flow_acft_txt" else
        "Proximity Ratio" if c == "proximity_txt" else
        "Match Dist (m)" if c == "match_dist_txt" else c
        for c in flow_hc_fields
    ]
    flow_hc_fg = folium.FeatureGroup(name=f"High-confidence flowlines ({len(flow_hc):,})", show=True)
    folium.GeoJson(
        flow_hc[flow_hc_fields + ["geometry"]].to_json() if flow_hc_fields else flow_hc[["geometry"]].to_json(),
        pane="pane_flowlines",
        style_function=lambda _: {"color": "#084594", "weight": 2.0, "opacity": 0.95},
        tooltip=folium.GeoJsonTooltip(fields=flow_hc_fields, aliases=flow_hc_aliases, sticky=False, localize=True) if flow_hc_fields else None,
        popup=folium.GeoJsonPopup(fields=flow_hc_fields, aliases=flow_hc_aliases, localize=True, labels=True) if flow_hc_fields else None,
    ).add_to(flow_hc_fg)
    flow_hc_fg.add_to(m)

    usgs_fg = folium.FeatureGroup(name=f"USGS gages ({len(usgs_df):,})", show=True)
    for _, r in usgs_df.iterrows():
        folium.CircleMarker(
            location=[float(r["LAT"]), float(r["LONG"])],
            radius=4,
            color="#084081",
            fill=True,
            fill_color="#0868ac",
            fill_opacity=0.9,
            weight=1,
            pane="pane_points",
            popup=folium.Popup(
                f"USGS<br>CPID: {r.get('CPID','')}<br>Gage_ID: {r.get('Gage_ID','')}<br>Jan(ac-ft): {fmt_discharge(r.get('JAN'))}",
                max_width=340,
            ),
        ).add_to(usgs_fg)
    usgs_fg.add_to(m)

    wam_fg = folium.FeatureGroup(name=f"WAM points ({len(wam_df):,})", show=True)
    for _, r in wam_df.iterrows():
        folium.CircleMarker(
            location=[float(r["LAT"]), float(r["LONG"])],
            radius=3,
            color="#cb181d",
            fill=True,
            fill_color="#ef3b2c",
            fill_opacity=0.78,
            weight=1,
            pane="pane_points",
            popup=folium.Popup(
                f"WAM<br>CPID: {r.get('CPID','')}<br>Gage_ID: {r.get('Gage_ID','')}<br>Jan(ac-ft): {fmt_discharge(r.get('JAN'))}",
                max_width=340,
            ),
        ).add_to(wam_fg)
    wam_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(MAP_OUTPUT_PATH)

    print(f"Saved interactive map HTML: {MAP_OUTPUT_PATH}")
    print(f"Flowline features: {len(flow_layer_data):,}")
    print(f"Flowlines with routed Jan 2018 discharge: {int(flow_layer_data['flow_cms'].notna().sum()):,}")
    print(f"High-confidence flowlines: {len(flow_hc):,}")
    print(f"Crosswalk rows flagged high-confidence: {len(xw_hc):,}")
    print(f"Fishnet cells plotted: {len(fishnet_gdf):,}")
    print(f"USGS gages plotted: {len(usgs_df):,}")
    print(f"WAM points plotted: {len(wam_df):,}")


if __name__ == "__main__":
    main()
