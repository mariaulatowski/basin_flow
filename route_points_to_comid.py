#!/usr/bin/env python
"""
Route point discharge constraints (USGS + WAM monthly_wide CSVs) to routed COMID flows.

Method summary:
- Build routed Brazos network from NHD flowlines + VAA topology.
- Parse monthly point discharges from monthly_wide style files.
- Snap each point to nearest routed COMID centroid.
- Build a prior local inflow proxy weighted by area, stream order, length, and width.
- Route upstream to downstream (mass conservation on DAG).
- Enforce point discharges by scaling each constrained COMID and all descendants.

Output:
- output/brazos/modeled_monthly_comid_flows_from_points.csv (Brazos flowlines only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from brazos_streamflow_model import (
    _descendants_by_index,
    _local_area_proxy_km2,
    build_network,
    route_monthly,
)

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

DEFAULT_SOURCE_TRUST = {
    "USGS": 1.0,
    "WAM": 0.75,
}


def monthly_wide_to_observations(csv_path: Path, source_name: str, input_units: str = "acft") -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)

    units = str(input_units).strip().lower()
    if units not in {"acft", "cfs"}:
        raise ValueError(f"Unsupported input_units={input_units!r}. Use 'acft' or 'cfs'.")

    for col in ["CPID", "Year", "LAT", "LONG"] + list(MONTH_MAP.keys()):
        if col not in df.columns:
            df[col] = ""

    df["CPID"] = df["CPID"].fillna("").astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LONG"] = pd.to_numeric(df["LONG"], errors="coerce")

    rows: list[dict] = []
    for _, r in df.iterrows():
        cpid = r["CPID"]
        year = r["Year"]
        lat = r["LAT"]
        lon = r["LONG"]
        if not cpid or pd.isna(year) or pd.isna(lat) or pd.isna(lon):
            continue

        for mon_name, mon_num in MONTH_MAP.items():
            value = pd.to_numeric(r.get(mon_name), errors="coerce")
            if pd.isna(value):
                continue
            d = pd.Timestamp(year=int(year), month=mon_num, day=1)
            if units == "acft":
                seconds = float(d.days_in_month * 24 * 3600)
                flow_cms = float(value) * 1233.48184 / seconds
            else:
                flow_cms = float(value) * 0.028316846592
            rows.append(
                {
                    "date": d,
                    "CPID": cpid,
                    "LAT": float(lat),
                    "LONG": float(lon),
                    "flow_cms": flow_cms,
                    "source": source_name,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "CPID", "LAT", "LONG", "flow_cms", "source"])
    return pd.DataFrame(rows)


def pick_stream_order_col(flow_df: pd.DataFrame) -> str | None:
    for c in ["streamorde", "streamorde_", "ord_stra", "strahler", "streamorder"]:
        if c in flow_df.columns:
            return c
    return None


def pick_width_col(flow_df: pd.DataFrame) -> str | None:
    for c in ["widthkm", "bankfullwid", "streamwidt", "width", "bfwidth"]:
        if c in flow_df.columns:
            return c
    return None


def pick_total_area_col(flow_df: pd.DataFrame) -> str | None:
    for c in ["totdasqkm", "divdasqkm", "areasqkm"]:
        if c in flow_df.columns:
            return c
    return None


def _weighted_geomean(values: np.ndarray, weights: np.ndarray, min_value: float = 1e-12) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0.0) & (weights > 0.0)
    if not mask.any():
        return np.nan
    v = np.clip(values[mask], min_value, None)
    w = weights[mask]
    return float(np.exp(np.sum(w * np.log(v)) / np.sum(w)))


def build_monthly_yield_prior(
    flow: gpd.GeoDataFrame,
    constraints_obs: pd.DataFrame,
    downstream: list[list[int]],
    base_coeff_m_per_month: float,
) -> tuple[np.ndarray, dict[pd.Timestamp, dict[str, float]]]:
    """Build an AFINCH-style monthly prior from specific yield plus a fallback climatology.

    The prior is estimated as monthly catchment yield and then multiplied by local area,
    rather than relying only on fixed geometric proxy runoff.
    """
    local_area_km2 = np.maximum(_local_area_proxy_km2(flow), 1e-6)

    total_area_col = pick_total_area_col(flow)
    if total_area_col is not None:
        routed_area_km2 = pd.to_numeric(flow[total_area_col], errors="coerce").to_numpy(dtype=float)
        invalid_area = ~np.isfinite(routed_area_km2) | (routed_area_km2 <= 0.0)
        if invalid_area.any():
            routed_area_km2[invalid_area] = route_monthly(local_area_km2, downstream)[invalid_area]
    else:
        routed_area_km2 = route_monthly(local_area_km2, downstream)
    routed_area_km2 = np.maximum(routed_area_km2, local_area_km2)

    seconds_30day = 30.0 * 24.0 * 3600.0
    base_yield_cms_per_km2 = ((base_coeff_m_per_month * 1e6) / seconds_30day) / 1e6
    base_local_q = np.maximum(local_area_km2, 1e-6) * base_yield_cms_per_km2

    obs = constraints_obs.copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs["trust"] = pd.to_numeric(obs.get("trust"), errors="coerce").fillna(1.0)
    obs["idx"] = pd.to_numeric(obs.get("idx"), errors="coerce").astype("Int64")
    obs = obs.dropna(subset=["date", "idx", "flow_cms", "trust"]).copy()
    obs["idx"] = obs["idx"].astype(np.int64)
    obs["routed_area_km2"] = routed_area_km2[obs["idx"].to_numpy()]
    obs = obs[obs["routed_area_km2"] > 0.0].copy()
    obs["specific_yield_cms_per_km2"] = obs["flow_cms"] / obs["routed_area_km2"]
    obs = obs[np.isfinite(obs["specific_yield_cms_per_km2"]) & (obs["specific_yield_cms_per_km2"] > 0.0)].copy()
    obs["month_num"] = obs["date"].dt.month

    global_yield = _weighted_geomean(
        obs["specific_yield_cms_per_km2"].to_numpy(),
        obs["trust"].to_numpy(),
        min_value=max(base_yield_cms_per_km2 * 0.01, 1e-12),
    )
    if not np.isfinite(global_yield):
        global_yield = base_yield_cms_per_km2

    month_climatology: dict[int, float] = {}
    for month_num, sub in obs.groupby("month_num"):
        month_climatology[int(month_num)] = _weighted_geomean(
            sub["specific_yield_cms_per_km2"].to_numpy(),
            sub["trust"].to_numpy(),
            min_value=max(base_yield_cms_per_km2 * 0.01, 1e-12),
        )

    monthly_local_q: list[np.ndarray] = []
    diagnostics: dict[pd.Timestamp, dict[str, float]] = {}
    for d in sorted(obs["date"].drop_duplicates()) if not obs.empty else []:
        d = pd.Timestamp(d)
        sub_date = obs[obs["date"] == d]
        date_yield = _weighted_geomean(
            sub_date["specific_yield_cms_per_km2"].to_numpy(),
            sub_date["trust"].to_numpy(),
            min_value=max(base_yield_cms_per_km2 * 0.01, 1e-12),
        )
        clim_yield = month_climatology.get(int(d.month), global_yield)
        if not np.isfinite(date_yield):
            date_yield = clim_yield

        prior_yield = (0.65 * date_yield) + (0.25 * clim_yield) + (0.10 * global_yield)
        prior_yield = max(float(prior_yield), base_yield_cms_per_km2 * 0.10)
        monthly_local_q.append(local_area_km2 * prior_yield)
        diagnostics[d] = {
            "prior_yield_cms_per_km2": float(prior_yield),
            "date_yield_cms_per_km2": float(date_yield),
            "month_climatology_cms_per_km2": float(clim_yield),
            "global_yield_cms_per_km2": float(global_yield),
            "constraint_count": int(len(sub_date)),
        }

    if not monthly_local_q:
        return base_local_q[None, :], {}
    return np.vstack(monthly_local_q), diagnostics


def build_source_trust_map(usgs_trust: float, wam_trust: float) -> dict[str, float]:
    trust = DEFAULT_SOURCE_TRUST.copy()
    trust["USGS"] = float(usgs_trust)
    trust["WAM"] = float(wam_trust)
    return trust


def build_prior_score(flow: gpd.GeoDataFrame) -> np.ndarray:
    local_area_km2 = _local_area_proxy_km2(flow)

    length_km = pd.to_numeric(flow.get("lengthkm", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    length_norm = np.maximum(length_km, 0.01) / np.nanmedian(np.maximum(length_km, 0.01))

    order_col = pick_stream_order_col(flow)
    if order_col is not None:
        so = pd.to_numeric(flow[order_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        so_min = float(np.nanmin(so))
        so_max = float(np.nanmax(so))
        if so_max > so_min:
            order_factor = 1.0 + 0.2 * ((so - so_min) / (so_max - so_min))
        else:
            order_factor = np.ones_like(so)
    else:
        order_factor = np.ones(len(flow), dtype=float)

    width_col = pick_width_col(flow)
    if width_col is not None:
        width_vals = pd.to_numeric(flow[width_col], errors="coerce").fillna(np.nan)
        width_vals = width_vals.fillna(width_vals.median()).fillna(1.0).to_numpy(dtype=float)
        width_norm = np.maximum(width_vals, 0.01) / np.nanmedian(np.maximum(width_vals, 0.01))
    else:
        width_norm = np.ones(len(flow), dtype=float)

    prior_score = np.maximum(local_area_km2, 0.001) * np.sqrt(length_norm) * np.sqrt(width_norm) * order_factor
    prior_score = prior_score / np.nanmean(prior_score)
    return prior_score


def build_brazos_to_nhd_crosswalk(
    routed_flow: gpd.GeoDataFrame,
    brazos_flowline_path: Path,
) -> pd.DataFrame:
    """Map each Brazos flowline to routed NHD using overlap- and distance-based scoring."""
    if not brazos_flowline_path.exists():
        raise FileNotFoundError(f"Missing Brazos flowline shapefile: {brazos_flowline_path}")

    brazos = gpd.read_file(str(brazos_flowline_path))
    if "COMID" not in brazos.columns:
        raise KeyError("Brazos flowline shapefile must contain COMID column.")
    brazos = brazos[~brazos.geometry.is_empty & brazos.geometry.notna()].copy()
    if brazos.empty:
        raise ValueError("Brazos flowline shapefile contains no valid geometries.")

    proj_crs = routed_flow.estimate_utm_crs()
    routed_proj = routed_flow.to_crs(proj_crs)
    brazos_proj = brazos.to_crs(proj_crs)

    routed_cent = routed_proj.geometry.centroid
    tree = cKDTree(np.column_stack([routed_cent.x.to_numpy(), routed_cent.y.to_numpy()]))
    routed_sindex = routed_proj.sindex

    def _endpoints(line):
        if line is None or line.is_empty:
            return None, None
        geom = line
        if geom.geom_type == "MultiLineString":
            geom = max(list(geom.geoms), key=lambda g: g.length)
        coords = list(geom.coords)
        if not coords:
            return None, None
        return coords[0], coords[-1]

    search_radius_m = 2000.0
    rows: list[dict] = []

    for i, b_row in brazos_proj.iterrows():
        b_geom = b_row.geometry
        b_comid = pd.to_numeric(brazos.loc[i, "COMID"], errors="coerce")
        if pd.isna(b_comid) or b_geom is None or b_geom.is_empty:
            continue

        b_len = float(max(b_geom.length, 1e-6))
        b_start, b_end = _endpoints(b_geom)
        # Generalized layers often do not overlap exactly. Use a corridor around Brazos line.
        b_corridor = b_geom.buffer(250.0)

        query_geom = b_geom.buffer(search_radius_m)
        cand_idx = list(routed_sindex.intersection(query_geom.bounds))
        candidates = routed_proj.iloc[cand_idx] if cand_idx else routed_proj.iloc[[]]

        best = None
        best_score = -np.inf

        for j, r_row in candidates.iterrows():
            r_geom = r_row.geometry
            if r_geom is None or r_geom.is_empty:
                continue

            try:
                inter_len = float(b_geom.intersection(r_geom).length)
            except Exception:
                inter_len = 0.0
            overlap_ratio = inter_len / b_len

            try:
                prox_len = float(r_geom.intersection(b_corridor).length)
            except Exception:
                prox_len = 0.0
            r_len = float(max(r_geom.length, 1e-6))
            proximity_ratio = prox_len / r_len

            centroid_dist = float(b_geom.centroid.distance(r_geom.centroid))
            line_dist = float(b_geom.distance(r_geom))

            r_start, r_end = _endpoints(r_geom)
            endpoint_dist = np.nan
            if b_start is not None and b_end is not None and r_start is not None and r_end is not None:
                d1 = np.hypot(b_start[0] - r_start[0], b_start[1] - r_start[1]) + np.hypot(b_end[0] - r_end[0], b_end[1] - r_end[1])
                d2 = np.hypot(b_start[0] - r_end[0], b_start[1] - r_end[1]) + np.hypot(b_end[0] - r_start[0], b_end[1] - r_start[1])
                endpoint_dist = float(min(d1, d2) / 2.0)

            # Favor proximity corridor + exact overlap first, then geometric distances.
            score = (12.0 * proximity_ratio) + (8.0 * overlap_ratio) - (0.0015 * line_dist) - (0.0005 * centroid_dist)
            if np.isfinite(endpoint_dist):
                score -= 0.0008 * endpoint_dist

            if score > best_score:
                best_score = score
                best = {
                    "brazos_comid": int(b_comid),
                    "nhd_comid": int(routed_flow.iloc[j]["COMID"]),
                    "match_dist_m": line_dist,
                    "centroid_dist_m": centroid_dist,
                    "endpoint_dist_m": endpoint_dist,
                    "overlap_ratio": overlap_ratio,
                    "proximity_ratio": proximity_ratio,
                    "match_score": score,
                    "match_method": "line_proximity_score",
                }

        # Fallback to nearest centroid if no candidate scored.
        if best is None:
            b_cent = b_geom.centroid
            dist_m, nn = tree.query(np.array([[b_cent.x, b_cent.y]]), k=1)
            nn_idx = int(nn[0])
            best = {
                "brazos_comid": int(b_comid),
                "nhd_comid": int(routed_flow.iloc[nn_idx]["COMID"]),
                "match_dist_m": float(dist_m[0]),
                "centroid_dist_m": float(dist_m[0]),
                "endpoint_dist_m": np.nan,
                "overlap_ratio": 0.0,
                "proximity_ratio": 0.0,
                "match_score": -9999.0,
                "match_method": "nearest_centroid_fallback",
            }

        rows.append(best)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Crosswalk matching produced no rows.")

    # One best NHD match per Brazos COMID.
    out = out.sort_values(["brazos_comid", "match_score", "proximity_ratio", "overlap_ratio"], ascending=[True, False, False, False])
    out = out.drop_duplicates(subset=["brazos_comid"], keep="first").copy()
    out["matched_flag"] = (out["proximity_ratio"] >= 0.10) | (out["match_dist_m"] <= 150.0)
    return out


def save_transferred_brazos_flowlines(
    routed_flow: gpd.GeoDataFrame,
    brazos_flowline_path: Path,
    crosswalk: pd.DataFrame,
    output_gpkg: Path,
) -> gpd.GeoDataFrame:
    """Create a Brazos geometry file with transferred NHD attributes and match diagnostics."""
    brazos = gpd.read_file(str(brazos_flowline_path))
    if "COMID" not in brazos.columns:
        raise KeyError("Brazos flowline shapefile must contain COMID column.")
    brazos = brazos[~brazos.geometry.is_empty & brazos.geometry.notna()].copy()
    if brazos.empty:
        raise ValueError("Brazos flowline shapefile contains no valid geometries.")

    brazos["brazos_comid"] = pd.to_numeric(brazos["COMID"], errors="coerce")
    brazos = brazos.dropna(subset=["brazos_comid"]).copy()
    brazos["brazos_comid"] = brazos["brazos_comid"].astype(np.int64)
    brazos["source_feature_count"] = 1

    dissolve_aggs = {"source_feature_count": "sum"}
    for col in brazos.columns:
        if col in {"geometry", "COMID", "brazos_comid", "source_feature_count"}:
            continue
        dissolve_aggs[col] = "first"
    brazos = brazos.dissolve(by="brazos_comid", as_index=False, aggfunc=dissolve_aggs)
    brazos["COMID"] = brazos["brazos_comid"]

    xw = crosswalk.copy()
    xw["brazos_comid"] = pd.to_numeric(xw["brazos_comid"], errors="coerce")
    xw["nhd_comid"] = pd.to_numeric(xw["nhd_comid"], errors="coerce")
    xw = xw.dropna(subset=["brazos_comid", "nhd_comid"]).copy()
    xw["brazos_comid"] = xw["brazos_comid"].astype(np.int64)
    xw["nhd_comid"] = xw["nhd_comid"].astype(np.int64)

    routed_attrs = routed_flow.copy()
    routed_attrs["COMID"] = pd.to_numeric(routed_attrs["COMID"], errors="coerce")
    routed_attrs = routed_attrs.dropna(subset=["COMID"]).copy()
    routed_attrs["COMID"] = routed_attrs["COMID"].astype(np.int64)

    candidate_cols = [
        "COMID",
        "permanent_",
        "fromnode",
        "tonode",
        "hydroseq",
        "lengthkm",
        "ftype",
        "fcode",
        "streamleve",
        "streamorde",
        "totdasqkm",
        "divdasqkm",
        "areasqkm",
        "qama",
        "vama",
        "widthkm",
        "bankfullwid",
        "streamwidt",
        "width",
        "bfwidth",
    ]
    keep = [c for c in candidate_cols if c in routed_attrs.columns]
    routed_attrs = routed_attrs[keep].copy()
    routed_attrs = routed_attrs.rename(columns={"COMID": "nhd_comid"})

    routed_lower = {c.lower() for c in routed_attrs.columns}
    overlapping = [
        c for c in brazos.columns
        if c not in {"geometry", "brazos_comid", "COMID"} and c.lower() in routed_lower
    ]
    if overlapping:
        brazos = brazos.drop(columns=overlapping, errors="ignore")

    out = brazos.merge(xw, on="brazos_comid", how="left")
    out = out.merge(routed_attrs, on="nhd_comid", how="left")

    # GPKG field names are case-insensitive; normalize and drop any duplicates.
    rename_map = {c: c.lower() for c in out.columns if c != "geometry"}
    out = out.rename(columns=rename_map)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()
    out.to_file(output_gpkg, layer="flowlines", driver="GPKG")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route point discharges to COMID monthly flows")
    p.add_argument("--start-date", default=None, help="Inclusive month start, format YYYY-MM-01")
    p.add_argument("--end-date", default=None, help="Inclusive month end, format YYYY-MM-01")
    p.add_argument(
        "--network-domain",
        default="nhd",
        choices=["nhd", "transferred-brazos"],
        help="nhd: route directly on Brazos-clipped NHD flowlines; transferred-brazos: route on Brazos geometry with transferred NHD topology",
    )
    p.add_argument(
        "--routing-mode",
        default="afinch",
        choices=["afinch", "inverse", "forward"],
        help="afinch: ratio-adjust upstream local yields; inverse: allocate from downstream constraints upstream; forward: prior route + downstream scaling",
    )
    p.add_argument("--usgs-trust", type=float, default=DEFAULT_SOURCE_TRUST["USGS"], help="Observation trust for USGS constraints, from 0 to 1")
    p.add_argument("--wam-trust", type=float, default=DEFAULT_SOURCE_TRUST["WAM"], help="Observation trust for WAM constraints, from 0 to 1")
    p.add_argument("--afinch-iters", type=int, default=5, help="Iterations for AFINCH upstream yield ratio adjustment")
    p.add_argument("--afinch-damping", type=float, default=0.8, help="Damping factor (0-1) for each AFINCH iteration")
    return p.parse_args()


def _build_upstream_from_downstream(downstream: list[list[int]], n: int) -> list[list[int]]:
    upstream = [[] for _ in range(n)]
    for i, ds_list in enumerate(downstream):
        for j in ds_list:
            upstream[j].append(i)
    return upstream


def inverse_route_monthly(
    local_prior_q: np.ndarray,
    prior_routed_q: np.ndarray,
    downstream: list[list[int]],
    constraint_targets: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate constrained downstream flows back upstream using prior fork weights.

    Returns:
        q_final: routed flow after inverse-derived local inflow and consistency forward pass
        local_inv: inverse-derived local inflow
    """
    n = len(prior_routed_q)
    upstream = _build_upstream_from_downstream(downstream, n)

    q_req = np.full(n, np.nan, dtype=float)
    fixed = np.zeros(n, dtype=bool)
    for idx, val in constraint_targets.items():
        q_req[idx] = float(val)
        fixed[idx] = True

    # Reverse topological pass: push downstream constraints to upstream forks.
    for j in range(n - 1, -1, -1):
        if not np.isfinite(q_req[j]):
            continue
        parents = upstream[j]
        if not parents:
            continue

        upstream_need = max(float(q_req[j]) - float(max(local_prior_q[j], 0.0)), 0.0)
        parent_prior = np.array([max(prior_routed_q[p], 0.0) for p in parents], dtype=float)
        if parent_prior.sum() <= 0.0:
            w = np.full(len(parents), 1.0 / len(parents), dtype=float)
        else:
            w = parent_prior / parent_prior.sum()

        for p, wp in zip(parents, w):
            proposed = upstream_need * float(wp)
            if fixed[p]:
                continue
            if not np.isfinite(q_req[p]):
                q_req[p] = proposed
            else:
                q_req[p] = max(float(q_req[p]), proposed)

    # Fill unconstrained nodes with prior routed flow.
    q_seed = np.where(np.isfinite(q_req), q_req, prior_routed_q)

    # Convert seeded routed flow target to local inflow, then re-route for consistency.
    local_inv = np.zeros(n, dtype=float)
    for i in range(n):
        up_sum = float(sum(q_seed[p] for p in upstream[i]))
        local_inv[i] = max(float(q_seed[i]) - up_sum, 0.0)

    q_final = route_monthly(local_inv, downstream)

    # Keep constrained reaches exact after consistency pass.
    for idx, val in constraint_targets.items():
        q_final[idx] = float(val)

    return q_final, local_inv


def build_upstream_index_cache(
    downstream: list[list[int]],
    constraint_indices: list[int],
) -> dict[int, np.ndarray]:
    upstream = _build_upstream_from_downstream(downstream, len(downstream))
    cache: dict[int, np.ndarray] = {}
    for idx in sorted(set(int(i) for i in constraint_indices)):
        cache[idx] = _descendants_by_index(upstream, idx)
    return cache


def afinch_route_monthly(
    local_prior_q: np.ndarray,
    constraint_targets: dict[int, float],
    constraint_trust: dict[int, float],
    downstream: list[list[int]],
    upstream_cache: dict[int, np.ndarray],
    max_iters: int,
    damping: float,
    protected_mask: np.ndarray | None = None,
    max_multiplier: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """AFINCH-like adjustment: apply gage/point ratios to upstream local yields.

    For each constrained reach, compute ratio = target / prior_routed_flow and apply
    a weighted log-ratio blend to all upstream local inflows that contribute to that reach.

    ``protected_mask`` is a boolean array of length n; True entries (e.g. artificial
    connectors, FType 343) are excluded from the log-ratio correction so their yields
    are never amplified by the iterative adjustment.

    ``max_multiplier`` caps the per-iteration multiplicative factor applied to local
    yields (default 10×), preventing runaway amplification on very short reaches whose
    prior local inflow is near zero.
    """
    n = len(local_prior_q)
    local_adj = local_prior_q.copy()
    eps = 1e-12
    iters = max(int(max_iters), 1)
    damp = float(np.clip(damping, 0.0, 1.0))
    log_cap = float(np.log(max(float(max_multiplier), 1.0 + eps)))

    # Pre-compute protected indices to exclude from adjustment
    if protected_mask is not None and protected_mask.any():
        protected_indices = np.where(protected_mask)[0]
    else:
        protected_indices = np.empty(0, dtype=int)

    for _ in range(iters):
        q_curr = route_monthly(local_adj, downstream)
        log_ratio_sum = np.zeros(n, dtype=float)
        weight_sum = np.zeros(n, dtype=float)

        for idx, target in constraint_targets.items():
            modeled = max(float(q_curr[idx]), eps)
            ratio = max(float(target) / modeled, eps)
            log_ratio = damp * float(np.log(ratio))
            w = max(float(constraint_trust.get(idx, 1.0)), eps)

            upstream_nodes = upstream_cache.get(idx)
            if upstream_nodes is None or len(upstream_nodes) == 0:
                continue

            log_ratio_sum[upstream_nodes] += w * log_ratio
            weight_sum[upstream_nodes] += w

        has_adj = weight_sum > 0.0
        # Do not adjust artificial connectors or other protected reaches
        if len(protected_indices) > 0:
            has_adj[protected_indices] = False
        if not has_adj.any():
            break
        raw_log_adj = log_ratio_sum[has_adj] / weight_sum[has_adj]
        # Cap per-iteration multiplier to avoid runaway amplification on tiny priors
        capped_log_adj = np.clip(raw_log_adj, -log_cap, log_cap)
        local_adj[has_adj] *= np.exp(capped_log_adj)
        local_adj = np.maximum(local_adj, 0.0)

    q_final = route_monthly(local_adj, downstream)
    return q_final, local_adj


def main() -> None:
    args = parse_args()
    if not (0.0 <= float(args.usgs_trust) <= 1.0 and 0.0 <= float(args.wam_trust) <= 1.0):
        raise ValueError("Source trust values must be between 0 and 1.")
    if int(args.afinch_iters) < 1:
        raise ValueError("--afinch-iters must be at least 1.")
    if not (0.0 <= float(args.afinch_damping) <= 1.0):
        raise ValueError("--afinch-damping must be between 0 and 1.")
    base_dir = Path(__file__).resolve().parent
    inputs_dir = base_dir / "inputData" / "inputs"

    usgs_csv = inputs_dir / "monthly_wide_acft.csv"
    wam_csv = inputs_dir / "monthly_wide_acft_from_hecdss.csv"
    flowline_source = base_dir / "inputData" / "texas_nhdplusgrb" / "_extracted_gdb"
    brazos_flowlines = base_dir / "inputData" / "flowlines" / "Brazos_Flowline.shp"
    basin_shp = base_dir / "inputData" / "river_basin" / "TWDB_MRBs_2014.shp"
    output_dir = base_dir / "output" / ("nhd_afinch" if args.network_domain == "nhd" else "brazos")
    output_csv = output_dir / "modeled_monthly_comid_flows_from_points.csv"
    crosswalk_csv = output_dir / "brazos_to_nhd_crosswalk_diagnostics.csv"
    transferred_flowlines_gpkg = output_dir / "brazos_flowlines_with_transferred_nhd.gpkg"
    nhd_domain_gpkg = output_dir / "nhd_brazos_flowlines.gpkg"
    snapped_points_csv = output_dir / "snapped_point_diagnostics.csv"
    enforcement_csv = output_dir / "enforcement_diagnostics.csv"
    constraint_conflicts_csv = output_dir / "constraint_conflicts.csv"

    if not usgs_csv.exists():
        raise FileNotFoundError(f"Missing USGS monthly_wide CSV: {usgs_csv}")
    if not wam_csv.exists():
        raise FileNotFoundError(f"Missing WAM monthly_wide CSV: {wam_csv}")

    nhd_flow, _, nhd_topo_comids, nhd_downstream, _ = build_network(
        flowline_source=str(flowline_source),
        flowline_layer="NHDFlowline",
        vaa_file=None,
        basin_shp=str(basin_shp),
        basin_name_field="basin_name",
        basin_name_value="Brazos",
    )
    nhd_to_brazos: pd.DataFrame | None = None
    if args.network_domain == "transferred-brazos":
        # Build transferred Brazos geometry and route on that domain.
        brazos_xw = build_brazos_to_nhd_crosswalk(nhd_flow, brazos_flowlines)
        nhd_to_brazos = brazos_xw.copy()
        crosswalk_csv.parent.mkdir(parents=True, exist_ok=True)
        nhd_to_brazos.to_csv(crosswalk_csv, index=False)
        save_transferred_brazos_flowlines(
            routed_flow=nhd_flow,
            brazos_flowline_path=brazos_flowlines,
            crosswalk=nhd_to_brazos,
            output_gpkg=transferred_flowlines_gpkg,
        )

        flow, _, topo_comids, downstream, _ = build_network(
            flowline_source=str(transferred_flowlines_gpkg),
            flowline_layer=None,
            vaa_file=None,
            basin_shp=str(basin_shp),
            basin_name_field="basin_name",
            basin_name_value="Brazos",
        )
    else:
        # Route directly on NHD flowlines (AFINCH-like domain).
        flow = nhd_flow
        topo_comids = nhd_topo_comids
        downstream = nhd_downstream
        nhd_domain_gpkg.parent.mkdir(parents=True, exist_ok=True)
        flow.to_file(nhd_domain_gpkg, layer="flowlines", driver="GPKG")

    usgs_obs = monthly_wide_to_observations(usgs_csv, "USGS")
    wam_obs = monthly_wide_to_observations(wam_csv, "WAM")
    obs = pd.concat([usgs_obs, wam_obs], ignore_index=True)
    if obs.empty:
        raise ValueError("No point observations parsed from monthly_wide CSVs.")
    source_trust = build_source_trust_map(args.usgs_trust, args.wam_trust)
    obs["trust"] = obs["source"].map(source_trust).fillna(1.0).astype(float)

    # Keep WAM values when both sources provide the same CPID+month.
    obs["priority"] = np.where(obs["source"].eq("WAM"), 2, 1)
    obs = obs.sort_values(["date", "CPID", "priority"]).drop_duplicates(["date", "CPID"], keep="last")

    flow_proj = flow.to_crs(flow.estimate_utm_crs())

    # Snap each point to nearest flowline geometry (not centroid) to improve tributary assignment.
    obs_pts = gpd.GeoDataFrame(
        obs.copy().reset_index(drop=True),
        geometry=gpd.points_from_xy(obs["LONG"], obs["LAT"]),
        crs="EPSG:4326",
    ).to_crs(flow_proj.crs)
    obs_pts["obs_row_id"] = np.arange(len(obs_pts), dtype=np.int64)

    snap_join = gpd.sjoin_nearest(
        obs_pts[["obs_row_id", "geometry"]],
        flow_proj[["COMID", "geometry"]],
        how="left",
        distance_col="snap_dist_m",
    )

    obs = obs_pts.drop(columns=["geometry"]).merge(
        snap_join[["obs_row_id", "COMID", "snap_dist_m"]],
        on="obs_row_id",
        how="left",
    )
    obs["COMID"] = pd.to_numeric(obs["COMID"], errors="coerce")
    obs = obs.dropna(subset=["COMID"]).copy()
    obs["COMID"] = obs["COMID"].astype(np.int64)

    comid_to_idx = {int(c): i for i, c in enumerate(topo_comids.tolist())}
    valid_obs = obs[obs["COMID"].map(comid_to_idx).notna()].copy()
    valid_obs["idx"] = valid_obs["COMID"].map(comid_to_idx).astype(np.int64)
    if valid_obs.empty:
        raise ValueError("No snapped observations mapped to routed COMIDs.")

    if args.start_date:
        start_date = pd.Timestamp(args.start_date)
        valid_obs = valid_obs[valid_obs["date"] >= start_date].copy()
    if args.end_date:
        end_date = pd.Timestamp(args.end_date)
        valid_obs = valid_obs[valid_obs["date"] <= end_date].copy()
    if valid_obs.empty:
        raise ValueError("No observations remain after applying date filter.")

    # If multiple points snap to the same reach in the same month, prefer higher-trust then nearest snap.
    valid_obs = valid_obs.sort_values(["date", "idx", "trust", "snap_dist_m", "CPID"], ascending=[True, True, False, True, True]).copy()
    conflict_mask = valid_obs.duplicated(subset=["date", "idx"], keep=False)
    conflict_rows = valid_obs.loc[conflict_mask].copy()
    constraints_obs = valid_obs.drop_duplicates(subset=["date", "idx"], keep="first").copy()

    valid_obs["obs_acft"] = valid_obs.apply(
        lambda r: float(r["flow_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
        axis=1,
    )
    constraints_obs["obs_acft"] = constraints_obs.apply(
        lambda r: float(r["flow_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
        axis=1,
    )

    base_coeff_m_per_month = 0.0008
    dates = sorted(constraints_obs["date"].drop_duplicates())
    monthly_local_q, prior_yield_diag = build_monthly_yield_prior(
        flow=flow,
        constraints_obs=constraints_obs,
        downstream=downstream,
        base_coeff_m_per_month=base_coeff_m_per_month,
    )
    date_to_local_q = {
        pd.Timestamp(d): monthly_local_q[i]
        for i, d in enumerate(sorted(prior_yield_diag))
    }

    # Fallback if date diagnostics were empty.
    if not date_to_local_q and dates:
        fallback_local_q = monthly_local_q[0]
        date_to_local_q = {pd.Timestamp(d): fallback_local_q for d in dates}

    upstream_cache = build_upstream_index_cache(
        downstream=downstream,
        constraint_indices=constraints_obs["idx"].astype(int).tolist(),
    )

    # Build a boolean mask of artificial connector reaches (NHD FType 343).
    # These reaches have no real catchment and cause AFINCH ratio amplification
    # to diverge; exclude them from the iterative log-ratio correction.
    if "ftype" in flow.columns:
        ftype_vals = pd.to_numeric(flow["ftype"], errors="coerce").fillna(-1).astype(int)
        artificial_mask = (ftype_vals == 343).to_numpy(dtype=bool)
        n_artificial = int(artificial_mask.sum())
        if n_artificial:
            print(f"  Protecting {n_artificial:,} artificial connector reaches (FType 343) from AFINCH adjustment")
    else:
        artificial_mask = None

    out_rows: list[dict] = []
    enforcement_rows: list[dict] = []
    for d in dates:
        local_q = date_to_local_q.get(pd.Timestamp(d))
        if local_q is None:
            local_q = monthly_local_q[0]
        q_prior = route_monthly(local_q, downstream)

        sub = constraints_obs[constraints_obs["date"] == d].sort_values("idx")
        if args.routing_mode == "afinch":
            targets: dict[int, float] = {}
            trusts: dict[int, float] = {}
            target_meta = {}
            for _, r in sub.iterrows():
                idx = int(r["idx"])
                trust = float(r.get("trust", 1.0))
                obs_q = float(r["flow_cms"])
                effective_target = float(q_prior[idx]) + trust * (obs_q - float(q_prior[idx]))
                targets[idx] = effective_target
                trusts[idx] = trust
                target_meta[idx] = {
                    "obs_cms": obs_q,
                    "effective_target_cms": effective_target,
                    "trust": trust,
                }

            q, _ = afinch_route_monthly(
                local_prior_q=local_q,
                constraint_targets=targets,
                constraint_trust=trusts,
                downstream=downstream,
                upstream_cache=upstream_cache,
                max_iters=args.afinch_iters,
                damping=args.afinch_damping,
                protected_mask=artificial_mask,
            )

            for _, r in sub.iterrows():
                i = int(r["idx"])
                obs_q = float(r["flow_cms"])
                before = float(q_prior[i])
                meta = target_meta[i]
                effective_target = float(meta["effective_target_cms"])
                trust = float(meta["trust"])
                scale = (effective_target / before) if before > 0.0 else np.nan
                enforcement_rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "CPID": r["CPID"],
                        "source": r["source"],
                        "brazos_comid": int(topo_comids[i]),
                        "modeled_before_cms": before,
                        "obs_cms": obs_q,
                        "effective_target_cms": effective_target,
                        "trust": trust,
                        "scale": scale,
                        "modeled_after_cms": float(q[i]),
                        "routing_mode": "afinch",
                        "prior_yield_cms_per_km2": prior_yield_diag.get(pd.Timestamp(d), {}).get("prior_yield_cms_per_km2", np.nan),
                    }
                )
        elif args.routing_mode == "forward":
            q = q_prior.copy()
            for _, r in sub.iterrows():
                i = int(r["idx"])
                modeled = float(q[i])
                obs_q = float(r["flow_cms"])
                trust = float(r.get("trust", 1.0))
                effective_target = modeled + trust * (obs_q - modeled)
                if modeled <= 0.0:
                    enforcement_rows.append(
                        {
                            "date": pd.Timestamp(d),
                            "CPID": r["CPID"],
                            "source": r["source"],
                            "brazos_comid": int(topo_comids[i]),
                            "modeled_before_cms": modeled,
                            "obs_cms": obs_q,
                            "effective_target_cms": effective_target,
                            "trust": trust,
                            "scale": np.nan,
                            "modeled_after_cms": modeled,
                            "routing_mode": "forward",
                            "prior_yield_cms_per_km2": prior_yield_diag.get(pd.Timestamp(d), {}).get("prior_yield_cms_per_km2", np.nan),
                        }
                    )
                    continue
                scale = effective_target / modeled
                desc = _descendants_by_index(downstream, i)
                q[desc] *= scale
                enforcement_rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "CPID": r["CPID"],
                        "source": r["source"],
                        "brazos_comid": int(topo_comids[i]),
                        "modeled_before_cms": modeled,
                        "obs_cms": obs_q,
                        "effective_target_cms": effective_target,
                        "trust": trust,
                        "scale": scale,
                        "modeled_after_cms": float(q[i]),
                        "routing_mode": "forward",
                        "prior_yield_cms_per_km2": prior_yield_diag.get(pd.Timestamp(d), {}).get("prior_yield_cms_per_km2", np.nan),
                    }
                )
        else:
            targets = {}
            target_meta = {}
            for _, r in sub.iterrows():
                idx = int(r["idx"])
                trust = float(r.get("trust", 1.0))
                obs_q = float(r["flow_cms"])
                effective_target = float(q_prior[idx]) + trust * (obs_q - float(q_prior[idx]))
                targets[idx] = effective_target
                target_meta[idx] = {
                    "obs_cms": obs_q,
                    "effective_target_cms": effective_target,
                    "trust": trust,
                    "CPID": r["CPID"],
                    "source": r["source"],
                }
            q, _ = inverse_route_monthly(
                local_prior_q=local_q,
                prior_routed_q=q_prior,
                downstream=downstream,
                constraint_targets=targets,
            )
            for _, r in sub.iterrows():
                i = int(r["idx"])
                obs_q = float(r["flow_cms"])
                before = float(q_prior[i])
                meta = target_meta[i]
                effective_target = float(meta["effective_target_cms"])
                trust = float(meta["trust"])
                scale = (effective_target / before) if before > 0.0 else np.nan
                enforcement_rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "CPID": r["CPID"],
                        "source": r["source"],
                        "brazos_comid": int(topo_comids[i]),
                        "modeled_before_cms": before,
                        "obs_cms": obs_q,
                        "effective_target_cms": effective_target,
                        "trust": trust,
                        "scale": scale,
                        "modeled_after_cms": float(q[i]),
                        "routing_mode": "inverse",
                        "prior_yield_cms_per_km2": prior_yield_diag.get(pd.Timestamp(d), {}).get("prior_yield_cms_per_km2", np.nan),
                    }
                )

        # Clip divergence-overflow values. NHD HR networks contain braided/divergent
        # channels; route_monthly double-counts flow at confluences downstream of splits.
        # Cap at a physically plausible maximum (10× max observed flow) before output.
        _obs_max_cms = max(float(sub["flow_cms"].max()) if not sub.empty else 1.0, 1.0)
        _q_cap = max(_obs_max_cms * 10.0, 1e4)  # at least 10,000 cms cap
        q_clipped = np.clip(q, 0.0, _q_cap)
        n_clipped = int((q > _q_cap).sum())
        if n_clipped > 0:
            print(f"  [{d.date()}] Clipped {n_clipped:,} divergence-overflow reaches at {_q_cap:.0f} cms")

        for i, c in enumerate(topo_comids):
            out_rows.append(
                {
                    "date": pd.Timestamp(d),
                    "COMID": int(c),
                    "flow_cms": float(q_clipped[i]),
                }
            )

    modeled = pd.DataFrame(out_rows)

    modeled["flow_acft"] = modeled.apply(
        lambda r: float(r["flow_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
        axis=1,
    )

    # Attach coordinates and match diagnostics for the selected routing domain.
    modeled = modeled.rename(columns={"COMID": "brazos_comid"})

    tf = flow[["COMID", "geometry"]].copy()
    tf = gpd.GeoDataFrame(tf, geometry="geometry", crs=flow.crs)
    tf_proj = tf.to_crs(tf.estimate_utm_crs())
    tf_cent = gpd.GeoSeries(tf_proj.geometry.centroid, crs=tf_proj.crs).to_crs(4326)
    tf_coords = pd.DataFrame(
        {
            "brazos_comid": tf["COMID"].astype(np.int64).to_numpy(),
            "lat": tf_cent.y.to_numpy(),
            "lon": tf_cent.x.to_numpy(),
        }
    )

    if args.network_domain == "transferred-brazos":
        reach_meta_cols = [c for c in ["COMID", "brazos_comid", "nhd_comid", "match_dist_m", "proximity_ratio", "matched_flag"] if c in flow.columns]
        reach_meta = flow[reach_meta_cols].copy()
        if "COMID" in reach_meta.columns:
            reach_meta = reach_meta.rename(columns={"COMID": "brazos_comid"})
    else:
        reach_meta = flow[["COMID"]].copy().rename(columns={"COMID": "brazos_comid"})
        reach_meta["nhd_comid"] = reach_meta["brazos_comid"].astype(np.int64)
        reach_meta["match_dist_m"] = np.nan
        reach_meta["proximity_ratio"] = np.nan
        reach_meta["matched_flag"] = True
    reach_meta = reach_meta.loc[:, ~reach_meta.columns.duplicated()].copy()

    snapped_diag = valid_obs.rename(columns={"COMID": "brazos_comid", "LAT": "obs_lat", "LONG": "obs_lon"}).copy()
    snapped_diag["used_for_enforcement"] = snapped_diag.set_index(["date", "idx"]).index.isin(
        constraints_obs.set_index(["date", "idx"]).index
    )
    snapped_diag = snapped_diag.merge(reach_meta, on="brazos_comid", how="left")
    snapped_diag = snapped_diag.merge(tf_coords, on="brazos_comid", how="left")
    snapped_diag = snapped_diag[
        [
            "date",
            "CPID",
            "source",
                    "trust",
            "obs_acft",
            "flow_cms",
            "obs_lat",
            "obs_lon",
            "brazos_comid",
            "nhd_comid",
            "lat",
            "lon",
            "snap_dist_m",
            "match_dist_m",
            "proximity_ratio",
            "matched_flag",
        ]
    ]

    enforcement_diag = pd.DataFrame(enforcement_rows)
    if not enforcement_diag.empty:
        enforcement_diag["modeled_before_acft"] = enforcement_diag.apply(
            lambda r: float(r["modeled_before_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
            axis=1,
        )
        enforcement_diag["obs_acft"] = enforcement_diag.apply(
            lambda r: float(r["obs_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
            axis=1,
        )
        enforcement_diag["modeled_after_acft"] = enforcement_diag.apply(
            lambda r: float(r["modeled_after_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
            axis=1,
        )
        enforcement_diag["effective_target_acft"] = enforcement_diag.apply(
            lambda r: float(r["effective_target_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
            axis=1,
        )

    modeled = modeled.merge(tf_coords, on="brazos_comid", how="left")
    modeled = modeled.merge(reach_meta, on="brazos_comid", how="left")
    modeled = modeled[
        [
            "date",
            "brazos_comid",
            "nhd_comid",
            "flow_cms",
            "flow_acft",
            "lat",
            "lon",
            "match_dist_m",
            "proximity_ratio",
            "matched_flag",
        ]
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    modeled.to_csv(output_csv, index=False)
    snapped_diag.to_csv(snapped_points_csv, index=False)
    enforcement_diag.to_csv(enforcement_csv, index=False)
    if conflict_rows.empty:
        pd.DataFrame(columns=["date", "idx", "CPID", "source", "flow_cms", "obs_acft", "snap_dist_m"]).to_csv(constraint_conflicts_csv, index=False)
    else:
        conflict_out = conflict_rows[["date", "idx", "CPID", "source", "trust", "flow_cms", "snap_dist_m"]].copy()
        conflict_out["obs_acft"] = conflict_out.apply(
            lambda r: float(r["flow_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
            axis=1,
        )
        conflict_out = conflict_out[["date", "idx", "CPID", "source", "trust", "flow_cms", "obs_acft", "snap_dist_m"]]
        conflict_out.to_csv(constraint_conflicts_csv, index=False)

    print(f"Wrote routed COMID flows: {output_csv}")
    if args.network_domain == "transferred-brazos":
        print(f"Wrote crosswalk diagnostics: {crosswalk_csv}")
        print(f"Wrote transferred flowlines file: {transferred_flowlines_gpkg}")
    else:
        print(f"Wrote NHD domain flowlines file: {nhd_domain_gpkg}")
    print(f"Wrote snapped point diagnostics: {snapped_points_csv}")
    print(f"Wrote enforcement diagnostics: {enforcement_csv}")
    print(f"Wrote constraint conflicts: {constraint_conflicts_csv}")
    print(f"Network domain: {args.network_domain}")
    print(f"Routing mode: {args.routing_mode}")
    if args.routing_mode == "afinch":
        print(f"AFINCH iterations: {args.afinch_iters}, damping: {args.afinch_damping:.2f}")
    print(f"Source trust: USGS={args.usgs_trust:.2f}, WAM={args.wam_trust:.2f}")
    print(f"Constraints after conflict resolution: {len(constraints_obs):,} (from {len(valid_obs):,} snapped points)")
    print(f"Rows: {len(modeled):,}")
    print(f"Routed COMIDs: {modeled['brazos_comid'].nunique():,}")
    print(f"Matched NHD COMIDs: {modeled['nhd_comid'].nunique():,}")
    print(f"Months: {modeled['date'].nunique():,}")
    print(f"Obs points used: {len(valid_obs):,}")
    print(f"Median snap distance (m): {valid_obs['snap_dist_m'].median():.2f}")
    if args.network_domain == "transferred-brazos" and nhd_to_brazos is not None:
        print(f"Median Brazos<->NHD match distance (m): {modeled['match_dist_m'].median():.2f}")
        print(f"Median overlap ratio: {nhd_to_brazos['overlap_ratio'].median():.3f}")
        print(f"Median proximity ratio: {nhd_to_brazos['proximity_ratio'].median():.3f}")
        print(f"Low-proximity matches (<0.10): {(nhd_to_brazos['proximity_ratio'] < 0.10).sum():,}")


if __name__ == "__main__":
    main()
