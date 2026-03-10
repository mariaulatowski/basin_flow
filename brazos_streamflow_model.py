#!/usr/bin/env python
"""
Regional monthly streamflow routing model for the Brazos basin.

Features:
- Loads NHD flowlines (single file or tiled directory) and VAA routing table
- Clips network to a basin polygon (e.g., TWDB Brazos basin)
- Snaps WRAP control points (CPs) to nearest routed flowline
- Parses WRAP .flo monthly files (acre-ft/month) to time series
- Optional gage ID -> COMID crosswalk for gage enforcement and calibration
- Vectorized routing through topological order
- Channel loss scaling between CP pairs
- Basic topology/consistency validation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


MONTH_COLS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


@dataclass
class ModelConfig:
    flowline_source: str
    flowline_layer: str | None
    vaa_file: str | None
    basin_shp: str
    basin_name_field: str
    basin_name_value: str
    cp_meta_file: str
    flo_file: str
    output_file: str
    output_shapefile: str | None = None
    shapefile_date: str | None = None
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-01"
    runoff_coeff_m_per_month: float = 0.0008
    gage_crosswalk_file: str | None = None
    calibrate: bool = False
    calibration_months: int = 24


def _read_flowline_source(
    flowline_source: str,
    flowline_layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    def _read_with_layer_fallback(path: str, preferred_layer: str | None, use_bbox):
        candidates = []
        if preferred_layer:
            candidates.append(preferred_layer)
        candidates.extend(["NetworkNHDFlowline", "NHDFlowline"])
        seen = set()
        last_exc = None
        for layer in candidates:
            if layer in seen:
                continue
            seen.add(layer)
            try:
                return gpd.read_file(path, layer=layer, bbox=use_bbox)
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise ValueError(f"Could not read flowline layer from {path}")

    src = Path(flowline_source)
    if src.suffix.lower() == ".gdb":
        flow = _read_with_layer_fallback(str(src), flowline_layer, bbox)
    elif src.is_dir():
        gdb_paths = sorted(src.rglob("*.gdb"))
        if gdb_paths:
            frames = [_read_with_layer_fallback(str(p), flowline_layer, bbox) for p in gdb_paths]
            flow = pd.concat(frames, ignore_index=True)
            if flow.empty:
                raise ValueError(f"No flowlines read from GDB directories under {src}")
        else:
            shp_paths = sorted(src.glob("NHDFlowline_*.shp"))
            if not shp_paths:
                raise FileNotFoundError(f"No NHDFlowline_*.shp files in {src}")
            frames = [gpd.read_file(str(p)) for p in shp_paths]
            flow = pd.concat(frames, ignore_index=True)
    else:
        flow = gpd.read_file(str(src))

    # Normalize column naming across source variants.
    flow = flow.rename(columns={c: c.lower() for c in flow.columns})

    # Normalize key fields across source variants (legacy SHP vs national GDB).
    if "permanent_" not in flow.columns:
        if "permanent_identifier" in flow.columns:
            flow["permanent_"] = flow["permanent_identifier"].astype(str)
        else:
            flow["permanent_"] = pd.Series(np.arange(len(flow), dtype=np.int64) + 1).astype(str)
    else:
        flow["permanent_"] = flow["permanent_"].astype(str)

    if "nhdplusid" in flow.columns:
        flow["nhdplusid"] = pd.to_numeric(flow["nhdplusid"], errors="coerce")

    return flow


def _read_internal_vaa_from_gdbs(flowline_source: str) -> pd.DataFrame:
    src = Path(flowline_source)
    gdb_paths: list[Path]
    if src.suffix.lower() == ".gdb":
        gdb_paths = [src]
    elif src.is_dir():
        gdb_paths = sorted(src.rglob("*.gdb"))
    else:
        gdb_paths = []

    if not gdb_paths:
        raise ValueError("No GDBs available for internal VAA load.")

    frames = []
    for gdb in gdb_paths:
        try:
            vaa = gpd.read_file(str(gdb), layer="NHDPlusFlowlineVAA", ignore_geometry=True)
        except Exception:
            continue
        vaa = vaa.rename(columns={c: c.lower() for c in vaa.columns})
        frames.append(vaa)

    if not frames:
        raise ValueError("Could not load NHDPlusFlowlineVAA from provided GDB source.")
    out = pd.concat(frames, ignore_index=True)
    if "nhdplusid" in out.columns:
        out["nhdplusid"] = pd.to_numeric(out["nhdplusid"], errors="coerce")
    return out


def load_basin_polygon(basin_shp: str, name_field: str, name_value: str) -> gpd.GeoSeries:
    basin = gpd.read_file(basin_shp)
    if name_field not in basin.columns:
        raise KeyError(f"Basin field '{name_field}' not found in {basin_shp}.")
    basin_sel = basin[basin[name_field].astype(str).str.lower() == name_value.lower()].copy()
    if basin_sel.empty:
        raise ValueError(f"No basin rows found where {name_field} == '{name_value}'.")
    return basin_sel.geometry


def build_network(
    flowline_source: str,
    flowline_layer: str | None,
    vaa_file: str | None,
    basin_shp: str,
    basin_name_field: str,
    basin_name_value: str,
) -> tuple[gpd.GeoDataFrame, dict[str, int], np.ndarray, list[list[int]], nx.DiGraph]:
    print("Clipping flowlines to basin...")
    basin_geom = load_basin_polygon(basin_shp, basin_name_field, basin_name_value)
    basin_bounds = tuple(basin_geom.total_bounds.tolist())

    print("Loading flowlines...")
    flow = _read_flowline_source(flowline_source, flowline_layer=flowline_layer, bbox=basin_bounds)
    if flow.empty:
        src = Path(flowline_source)
        if src.suffix.lower() == ".gdb":
            print("Bbox read returned no rows for GDB source; skipping full-layer retry.")
        else:
            print("Bbox read returned no rows; retrying full flowline read...")
            try:
                flow = _read_flowline_source(flowline_source, flowline_layer=flowline_layer, bbox=None)
            except Exception as exc:
                print(f"Full flowline read failed: {exc}")
                flow = gpd.GeoDataFrame()

    if flow.empty:
        src = Path(flowline_source)
        if src.suffix.lower() == ".gdb":
            msg = None
            try:
                sample = gpd.read_file(
                    str(src),
                    layer=(flowline_layer or "NetworkNHDFlowline"),
                    ignore_geometry=True,
                    columns=["vpuid"],
                    rows=200000,
                )
                vpus = sorted(sample["vpuid"].dropna().astype(str).unique().tolist())[:25]
                msg = (
                    "No flowlines returned from the selected GDB/layer for this basin. "
                    f"Sample VPUIDs starts with: {vpus}. "
                    "Texas typically needs VPUIDs in 11xx/12xx/13xx."
                )
            except Exception:
                msg = None
            if msg:
                raise ValueError(msg)
        raise ValueError("No flowlines loaded from source.")

    basin_geom = basin_geom.to_crs(flow.crs)
    mask = basin_geom.union_all()
    flow = flow[flow.intersects(mask)].copy()
    if flow.empty:
        src = Path(flowline_source)
        if src.suffix.lower() == ".gdb":
            msg = None
            try:
                sample = gpd.read_file(
                    str(src),
                    layer=(flowline_layer or "NetworkNHDFlowline"),
                    ignore_geometry=True,
                    columns=["vpuid"],
                    rows=200000,
                )
                vpus = sorted(sample["vpuid"].dropna().astype(str).unique().tolist())[:25]
                msg = (
                    "No flowlines intersect the selected basin. "
                    f"This GDB sample VPUIDs starts with: {vpus}. "
                    "Texas typically needs VPUIDs in 11xx/12xx/13xx."
                )
            except Exception:
                msg = None
            if msg:
                raise ValueError(msg)
        raise ValueError("No flowlines intersect the selected basin.")

    has_nodes = {"fromnode", "tonode"}.issubset(flow.columns)
    if has_nodes:
        merged = flow.copy()
    else:
        if vaa_file:
            print("Loading VAA routing table...")
            vaa = gpd.read_file(vaa_file)
            vaa = vaa.rename(columns={c: c.lower() for c in vaa.columns})
        else:
            print("Loading internal VAA table from GDB source...")
            vaa = _read_internal_vaa_from_gdbs(flowline_source)

        if {"nhdplusid", "fromnode", "tonode"}.issubset(vaa.columns) and "nhdplusid" in flow.columns:
            vaa["nhdplusid"] = pd.to_numeric(vaa["nhdplusid"], errors="coerce")
            merged = flow.merge(
                vaa[["nhdplusid", "fromnode", "tonode", "hydroseq"]],
                on="nhdplusid",
                how="left",
            )
        elif {"permanent_", "fromnode", "tonode"}.issubset(vaa.columns):
            vaa["permanent_"] = vaa["permanent_"].astype(str)
            merged = flow.merge(
                vaa[["permanent_", "fromnode", "tonode", "hydroseq"]],
                on="permanent_",
                how="left",
            )
        else:
            raise KeyError("VAA table must contain either nhdplusid or permanent_ plus fromnode/tonode.")

    merged = merged.dropna(subset=["fromnode", "tonode"]).copy()
    if merged.empty:
        raise ValueError("No routed flowlines after VAA merge.")

    if "nhdplusid" in merged.columns and merged["nhdplusid"].notna().any():
        merged["COMID"] = pd.to_numeric(merged["nhdplusid"], errors="coerce").round().astype("Int64")
        merged = merged.dropna(subset=["COMID"]).copy()
        merged["COMID"] = merged["COMID"].astype(np.int64)
    else:
        merged["COMID"] = np.arange(1, len(merged) + 1, dtype=np.int64)

    p_to_comid = dict(zip(merged["permanent_"], merged["COMID"]))

    nodes = merged[["COMID", "fromnode", "tonode", "hydroseq"]].copy()

    edges_df = nodes.merge(
        nodes,
        left_on="tonode",
        right_on="fromnode",
        suffixes=("_up", "_ds"),
    )[["COMID_up", "COMID_ds", "hydroseq_up", "hydroseq_ds"]]
    # Keep only physically downstream edges where hydroseq decreases.
    has_hs = edges_df["hydroseq_up"].notna() & edges_df["hydroseq_ds"].notna()
    edges_df = edges_df[~has_hs | (edges_df["hydroseq_up"] > edges_df["hydroseq_ds"])]
    edges_df = edges_df[["COMID_up", "COMID_ds"]]
    edges_df = edges_df.drop_duplicates()

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["COMID"].tolist())
    graph.add_edges_from(edges_df.itertuples(index=False, name=None))

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Routing graph is not a DAG; cycle(s) detected.")
    topo_order = list(nx.topological_sort(graph))

    if len(topo_order) != len(graph.nodes):
        raise ValueError("Cycle or disconnected routing issue: topological order length mismatch.")

    idx = {c: i for i, c in enumerate(topo_order)}
    downstream = [[] for _ in topo_order]
    for up, ds in graph.edges:
        if up in idx and ds in idx:
            downstream[idx[up]].append(idx[ds])

    merged = merged.set_index("COMID").loc[topo_order].reset_index()
    return merged, p_to_comid, np.array(topo_order, dtype=np.int64), downstream, graph


def _local_area_proxy_km2(flow: gpd.GeoDataFrame) -> np.ndarray:
    # Prefer real local catchment area when available in NHDPlus HR.
    if "areasqkm" in flow.columns and pd.to_numeric(flow["areasqkm"], errors="coerce").notna().any():
        return pd.to_numeric(flow["areasqkm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Fall back to total drainage area only if local area is not present.
    if "totdasqkm" in flow.columns and pd.to_numeric(flow["totdasqkm"], errors="coerce").notna().any():
        return pd.to_numeric(flow["totdasqkm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "lengthkm" in flow.columns:
        return flow["lengthkm"].astype(float).to_numpy() * 2.0
    geom_crs = flow.estimate_utm_crs()
    proj = flow.to_crs(geom_crs)
    return (proj.geometry.length.to_numpy() / 1000.0) * 2.0


def validate_network(flow: gpd.GeoDataFrame, topo_comids: np.ndarray, downstream: list[list[int]]) -> None:
    print("Validating network...")
    n = len(topo_comids)
    edge_count = sum(len(v) for v in downstream)
    if n == 0:
        raise ValueError("Network has zero routed COMIDs.")
    if edge_count == 0:
        print("Warning: network has no routed edges.")

    indeg = np.zeros(n, dtype=np.int64)
    outdeg = np.zeros(n, dtype=np.int64)
    for i, ds in enumerate(downstream):
        outdeg[i] = len(ds)
        for j in ds:
            indeg[j] += 1

    sources = int((indeg == 0).sum())
    sinks = int((outdeg == 0).sum())
    print(f"Network summary: nodes={n}, edges={edge_count}, sources={sources}, sinks={sinks}")


def snap_control_points_to_comid(flow: gpd.GeoDataFrame, cp_meta: pd.DataFrame) -> pd.DataFrame:
    req = {"UP_CP", "Next_DS_CP", "ChLosFac", "LAT", "LONG"}
    missing = req - set(cp_meta.columns)
    if missing:
        raise KeyError(f"CP metadata missing required columns: {sorted(missing)}")

    work = cp_meta.copy()
    work["UP_CP"] = work["UP_CP"].astype(str)
    work["Next_DS_CP"] = work["Next_DS_CP"].astype(str)

    cp_unique = work[["UP_CP", "LAT", "LONG"]].drop_duplicates("UP_CP").copy()
    cp_points = gpd.GeoDataFrame(
        cp_unique,
        geometry=gpd.points_from_xy(cp_unique["LONG"], cp_unique["LAT"]),
        crs="EPSG:4326",
    )

    # Use projected coordinates for nearest-neighbor distance queries.
    if flow.crs and flow.crs.is_geographic:
        proj_crs = flow.estimate_utm_crs()
        flow_proj = flow.to_crs(proj_crs)
        cp_points = cp_points.to_crs(proj_crs)
    else:
        flow_proj = flow
        cp_points = cp_points.to_crs(flow.crs)

    cent = flow_proj.geometry.centroid
    tree = cKDTree(np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()]))
    q = np.column_stack([cp_points.geometry.x.to_numpy(), cp_points.geometry.y.to_numpy()])
    _, nn = tree.query(q, k=1)
    cp_points["COMID"] = flow.iloc[nn]["COMID"].to_numpy()

    cp_to_comid = dict(zip(cp_points["UP_CP"], cp_points["COMID"]))

    work["UP_COMID"] = work["UP_CP"].map(cp_to_comid)
    work["DS_COMID"] = work["Next_DS_CP"].map(cp_to_comid)
    return work


def parse_flo_monthly(flo_file: str) -> pd.DataFrame:
    rows: list[dict] = []
    with open(flo_file, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            parts = raw.split()
            if len(parts) < 14:
                continue
            station = parts[0]
            year = int(parts[1])
            vals = [float(x) for x in parts[2:14]]
            for m, acft in enumerate(vals, start=1):
                date = pd.Timestamp(year=year, month=m, day=1)
                rows.append({"date": date, "gage_id": station, "acft": acft})
    if not rows:
        raise ValueError(f"No parseable rows found in {flo_file}")
    return pd.DataFrame(rows).sort_values(["date", "gage_id"]).reset_index(drop=True)


def acft_month_to_cms(acft: np.ndarray, dates: pd.Series) -> np.ndarray:
    days = dates.dt.days_in_month.to_numpy(dtype=float)
    seconds = days * 24.0 * 3600.0
    m3 = acft * 1233.48184
    return m3 / seconds


def load_gage_crosswalk(crosswalk_file: str) -> pd.DataFrame:
    xw = pd.read_csv(crosswalk_file)
    req = {"gage_id", "COMID"}
    missing = req - set(xw.columns)
    if missing:
        raise KeyError(f"Gage crosswalk missing required columns: {sorted(missing)}")
    xw["gage_id"] = xw["gage_id"].astype(str)
    xw["COMID"] = xw["COMID"].astype(np.int64)
    return xw[["gage_id", "COMID"]].drop_duplicates()


def compile_gage_observations(
    flo_file: str,
    start_date: str,
    end_date: str,
    gage_crosswalk_file: str | None,
) -> pd.DataFrame:
    flo = parse_flo_monthly(flo_file)
    flo = flo[(flo["date"] >= start_date) & (flo["date"] <= end_date)].copy()
    flo["flow_cms"] = acft_month_to_cms(flo["acft"].to_numpy(), flo["date"])

    if gage_crosswalk_file is None:
        return pd.DataFrame(columns=["date", "COMID", "flow_cms"])

    xw = load_gage_crosswalk(gage_crosswalk_file)
    obs = flo.merge(xw, on="gage_id", how="inner")
    return obs[["date", "COMID", "flow_cms"]]


def _build_edge_index(downstream: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    up_idx = []
    ds_idx = []
    for up, ds_list in enumerate(downstream):
        for ds in ds_list:
            up_idx.append(up)
            ds_idx.append(ds)
    return np.array(up_idx, dtype=np.int64), np.array(ds_idx, dtype=np.int64)


def route_monthly(local_q: np.ndarray, downstream: list[list[int]]) -> np.ndarray:
    q = local_q.copy()
    for i in range(len(q)):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in downstream[i]:
            q[j] += qi
    return q


def _descendants_by_index(downstream: list[list[int]], start: int) -> np.ndarray:
    seen = set([start])
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in downstream[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return np.array(sorted(seen), dtype=np.int64)


def _find_path_indices(graph: nx.DiGraph, comid_to_idx: dict[int, int], up_comid: int, ds_comid: int) -> list[int] | None:
    try:
        path = nx.shortest_path(graph, up_comid, ds_comid)
    except nx.NetworkXNoPath:
        return None
    return [comid_to_idx[c] for c in path if c in comid_to_idx]


def calibrate_runoff_coeff(
    coeff_grid: Iterable[float],
    local_area_km2: np.ndarray,
    downstream: list[list[int]],
    gage_obs: pd.DataFrame,
    comid_to_idx: dict[int, int],
    dates: pd.DatetimeIndex,
    calibration_months: int,
) -> float:
    if gage_obs.empty:
        raise ValueError("Cannot calibrate runoff coefficient without mapped gage observations.")

    cal_dates = dates[:calibration_months]
    obs = gage_obs[gage_obs["date"].isin(cal_dates)].copy()
    if obs.empty:
        raise ValueError("No gage observations in calibration period.")

    obs["idx"] = obs["COMID"].map(comid_to_idx)
    obs = obs.dropna(subset=["idx"]).copy()
    obs["idx"] = obs["idx"].astype(np.int64)
    if obs.empty:
        raise ValueError("No calibration gages matched network COMIDs.")

    best_coeff = None
    best_rmse = np.inf

    seconds_30day = 30.0 * 24.0 * 3600.0
    for coeff in coeff_grid:
        sq_err = []
        local_q = (coeff * local_area_km2 * 1e6) / seconds_30day
        q_routed = route_monthly(local_q, downstream)
        for _, r in obs.iterrows():
            modeled = q_routed[int(r["idx"])]
            sq_err.append((modeled - float(r["flow_cms"])) ** 2)
        if not sq_err:
            continue
        rmse = float(np.sqrt(np.mean(sq_err)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_coeff = float(coeff)

    if best_coeff is None:
        raise ValueError("Calibration failed to evaluate any coefficients.")
    print(f"Calibrated runoff coeff = {best_coeff:.6f} m/month (RMSE={best_rmse:.3f} cms)")
    return best_coeff


def run_model(cfg: ModelConfig, progress_cb=None) -> None:
    def update_progress(percent: float, stage: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0.0, min(100.0, float(percent))), stage)

    update_progress(1, "Starting")
    flow, _, topo_comids, downstream, graph = build_network(
        cfg.flowline_source,
        cfg.flowline_layer,
        cfg.vaa_file,
        cfg.basin_shp,
        cfg.basin_name_field,
        cfg.basin_name_value,
    )
    update_progress(20, "Network loaded")

    validate_network(flow, topo_comids, downstream)
    update_progress(25, "Network validated")

    print("Loading CP metadata and snapping control points...")
    cp_meta = pd.read_csv(cfg.cp_meta_file)
    cp_map = snap_control_points_to_comid(flow, cp_meta)
    update_progress(35, "Control points snapped")

    comid_to_idx = {int(c): i for i, c in enumerate(topo_comids.tolist())}
    cp_rows = cp_map.dropna(subset=["UP_COMID", "DS_COMID"]).copy()
    cp_rows["UP_COMID"] = cp_rows["UP_COMID"].astype(np.int64)
    cp_rows["DS_COMID"] = cp_rows["DS_COMID"].astype(np.int64)

    cp_paths: list[tuple[list[int], float]] = []
    for _, r in cp_rows.iterrows():
        up = int(r["UP_COMID"])
        ds = int(r["DS_COMID"])
        if up not in comid_to_idx or ds not in comid_to_idx:
            continue
        p = _find_path_indices(graph, comid_to_idx, up, ds)
        if p and len(p) > 1:
            cp_paths.append((p, float(r["ChLosFac"])))

    print(f"CP loss paths prepared: {len(cp_paths)}")

    gage_obs = compile_gage_observations(
        cfg.flo_file,
        cfg.start_date,
        cfg.end_date,
        cfg.gage_crosswalk_file,
    )
    update_progress(42, "Gage observations loaded")
    if cfg.gage_crosswalk_file is None:
        print("No gage crosswalk provided: gage enforcement and calibration are skipped.")
    else:
        print(f"Mapped gage observations: {len(gage_obs)}")

    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="MS")

    # Prepare point geometry + lat/lon for GIS-friendly outputs.
    flow_proj = flow.to_crs(flow.estimate_utm_crs())
    cent_proj = flow_proj.geometry.centroid
    cent_wgs84 = gpd.GeoSeries(cent_proj, crs=flow_proj.crs).to_crs(4326)
    lat_arr = cent_wgs84.y.to_numpy()
    lon_arr = cent_wgs84.x.to_numpy()

    local_area_km2 = _local_area_proxy_km2(flow)
    runoff_coeff = cfg.runoff_coeff_m_per_month
    if cfg.calibrate:
        coeff_grid = np.linspace(0.0001, 0.0050, 30)
        runoff_coeff = calibrate_runoff_coeff(
            coeff_grid,
            local_area_km2,
            downstream,
            gage_obs,
            comid_to_idx,
            dates,
            cfg.calibration_months,
        )

    gage_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if not gage_obs.empty:
        for d, sub in gage_obs.groupby("date"):
            gage_by_date[pd.Timestamp(d)] = sub.copy()

    print("Routing monthly flows...")
    out_rows: list[dict] = []
    total_months = len(dates)
    for month_idx, d in enumerate(dates):
        seconds = float(d.days_in_month * 24 * 3600)
        local_q = (runoff_coeff * local_area_km2 * 1e6) / seconds
        q = route_monthly(local_q, downstream)

        # WRAP-style channel loss scaling on downstream path between CP pairs.
        for path_idx, fcl in cp_paths:
            up_idx = path_idx[0]
            ds_idx = path_idx[-1]
            q_up = q[up_idx]
            q_target_ds = q_up * (1.0 - fcl)
            q_current_ds = q[ds_idx]
            if q_current_ds > 0.0:
                scale = q_target_ds / q_current_ds
                q[np.array(path_idx[1:], dtype=np.int64)] *= scale

        # Enforce gages by scaling each gage node and all descendants.
        if d in gage_by_date:
            for _, r in gage_by_date[d].iterrows():
                c = int(r["COMID"])
                if c not in comid_to_idx:
                    continue
                i = comid_to_idx[c]
                modeled = q[i]
                if modeled <= 0.0:
                    continue
                scale = float(r["flow_cms"]) / modeled
                idxs = _descendants_by_index(downstream, i)
                q[idxs] *= scale

        for i, c in enumerate(topo_comids):
            out_rows.append(
                {
                    "date": d,
                    "COMID": int(c),
                    "flow_cms": float(q[i]),
                    "lat": float(lat_arr[i]),
                    "lon": float(lon_arr[i]),
                }
            )
        # 42 -> 90 over monthly routing loop.
        update_progress(42 + (48.0 * (month_idx + 1) / max(total_months, 1)), f"Routing {d.strftime('%Y-%m')}")

    out = pd.DataFrame(out_rows)
    update_progress(92, "Writing CSV output")
    out.to_csv(cfg.output_file, index=False)
    print(f"Saved modeled flows: {cfg.output_file}")

    if cfg.output_shapefile:
        update_progress(96, "Writing shapefile output")
        shp_date = pd.Timestamp(cfg.shapefile_date) if cfg.shapefile_date else pd.Timestamp(dates[-1])
        shp_rows = out[out["date"] == shp_date].copy()
        if shp_rows.empty:
            raise ValueError(f"No modeled rows found for shapefile date {shp_date.date()}.")

        # Use point geometry at each COMID centroid for quick GIS display.
        shp_points = gpd.GeoDataFrame(
            shp_rows.copy(),
            geometry=gpd.points_from_xy(shp_rows["lon"], shp_rows["lat"]),
            crs="EPSG:4326",
        )
        # Shapefile field names are limited; keep names short.
        shp_points = shp_points.rename(
            columns={
                "date": "DATE",
                "COMID": "COMID",
                "flow_cms": "FLOW_CMS",
                "lat": "LAT",
                "lon": "LON",
            }
        )
        shp_points["DATE"] = pd.to_datetime(shp_points["DATE"]).dt.strftime("%Y-%m-%d")
        # Shapefile numeric field width is limited; keep a clipped numeric + full text value.
        shp_points["FLOW_TXT"] = shp_points["FLOW_CMS"].map(lambda x: f"{float(x):.6e}")
        shp_points["FLOW_CMS"] = pd.to_numeric(shp_points["FLOW_CMS"], errors="coerce").clip(-1e20, 1e20)

        shp_path = Path(cfg.output_shapefile)
        shp_path.parent.mkdir(parents=True, exist_ok=True)
        shp_points[["DATE", "COMID", "FLOW_CMS", "FLOW_TXT", "LAT", "LON", "geometry"]].to_file(
            str(shp_path), driver="ESRI Shapefile"
        )
        print(f"Saved shapefile: {cfg.output_shapefile}")
    update_progress(100, "Complete")


def parse_args() -> ModelConfig:
    p = argparse.ArgumentParser(description="Brazos monthly streamflow routing model")
    p.add_argument("--flowline-source", default="Shape")
    p.add_argument("--flowline-layer", default=None)
    p.add_argument("--vaa-file", default="Shape/NHDFlowlineVAA.dbf")
    p.add_argument("--basin-shp", default="river_basin/TWDB_MRBs_2014.shp")
    p.add_argument("--basin-name-field", default="basin_name")
    p.add_argument("--basin-name-value", default="Brazos")
    p.add_argument("--cp-meta-file", default="Brazos_Metadata.csv")
    p.add_argument("--flo-file", default="brazos_final_acft.flo")
    p.add_argument("--gage-crosswalk-file", default=None)
    p.add_argument("--output-file", default="modeled_monthly_comid_flows.csv")
    p.add_argument("--output-shapefile", default=None)
    p.add_argument("--shapefile-date", default=None, help="YYYY-MM-01 month to export as shapefile")
    p.add_argument("--start-date", default="2010-01-01")
    p.add_argument("--end-date", default="2024-12-01")
    p.add_argument("--runoff-coeff-m-per-month", type=float, default=0.0008)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibration-months", type=int, default=24)
    a = p.parse_args()
    return ModelConfig(
        flowline_source=a.flowline_source,
        flowline_layer=a.flowline_layer,
        vaa_file=a.vaa_file,
        basin_shp=a.basin_shp,
        basin_name_field=a.basin_name_field,
        basin_name_value=a.basin_name_value,
        cp_meta_file=a.cp_meta_file,
        flo_file=a.flo_file,
        output_file=a.output_file,
        output_shapefile=a.output_shapefile,
        shapefile_date=a.shapefile_date,
        start_date=a.start_date,
        end_date=a.end_date,
        runoff_coeff_m_per_month=a.runoff_coeff_m_per_month,
        gage_crosswalk_file=a.gage_crosswalk_file,
        calibrate=a.calibrate,
        calibration_months=a.calibration_months,
    )


if __name__ == "__main__":
    run_model(parse_args())
