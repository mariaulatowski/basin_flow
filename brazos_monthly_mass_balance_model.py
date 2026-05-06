#!/usr/bin/env python
"""Monthly mass-balance routing model for Brazos using USGS + WAM constraints.

This script solves for local monthly inflows on each reach (q_local) and then
routes them downstream with conservation of mass:

    Q_downstream = q_local + sum(Q_upstream)

Unknown q_local is estimated each month by a constrained inverse solve:
- Fit observed monthly flow constraints at snapped USGS/WAM points
- Keep q_local near a physically plausible prior
- Enforce nonnegative local inflow

The prior and regularization can use stream length, stream order, and drainage
area (when available).
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, eye, vstack

from brazos_streamflow_model import build_network, route_monthly
from route_points_to_comid import build_monthly_yield_prior, monthly_wide_to_observations

SECONDS_PER_DAY = 86400.0
ACFT_PER_CMS_DAY = SECONDS_PER_DAY / 1233.48184


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly mass-balance routing on Brazos flowlines")
    p.add_argument("--start-date", required=True, help="Inclusive month start (YYYY-MM-01)")
    p.add_argument("--end-date", required=True, help="Inclusive month end (YYYY-MM-01)")
    p.add_argument("--network-source", choices=["nhd_medium", "nhd_hr"], default="nhd_medium")
    p.add_argument("--usgs-trust", type=float, default=1.0)
    p.add_argument("--wam-trust", type=float, default=0.75)
    p.add_argument("--lambda-base", type=float, default=1.0, help="Base regularization strength")
    p.add_argument("--lambda-order", type=float, default=0.6, help="Order weight in regularization")
    p.add_argument("--lambda-length", type=float, default=0.2, help="Length weight in regularization")
    p.add_argument("--lambda-area", type=float, default=0.2, help="Area weight in regularization")
    p.add_argument("--max-output-cms", type=float, default=10000.0)
    p.add_argument("--output-dir", default="output/brazos_mass_balance")
    return p.parse_args()


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _build_reverse_adjacency(downstream: list[list[int]]) -> list[list[int]]:
    n = len(downstream)
    upstream = [[] for _ in range(n)]
    for up_idx, ds_list in enumerate(downstream):
        for ds_idx in ds_list:
            upstream[ds_idx].append(up_idx)
    return upstream


def _compute_strahler_order(upstream: list[list[int]], topo_n: int) -> np.ndarray:
    # topological indices flow upstream->downstream, so evaluate in reverse
    order = np.ones(topo_n, dtype=np.int32)
    for i in range(topo_n - 1, -1, -1):
        ups = upstream[i]
        if not ups:
            order[i] = 1
            continue
        up_orders = order[np.asarray(ups, dtype=int)]
        m = int(up_orders.max())
        order[i] = m + 1 if int((up_orders == m).sum()) >= 2 else m
    return order


def _candidate_area_column(flow: pd.DataFrame) -> np.ndarray | None:
    candidates = [
        "areasqkm",
        "totdasqkm",
        "tot_dasqkm",
        "totdrainagekm2",
        "drainagearea_km2",
        "drain_area_km2",
        "catarea",
    ]
    for c in candidates:
        if c in flow.columns:
            vals = pd.to_numeric(flow[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).any() and np.nanmax(vals) > 0:
                return vals
    return None


def _upstream_contributors(target_idx: int, upstream: list[list[int]]) -> np.ndarray:
    visited = set([target_idx])
    q = deque([target_idx])
    while q:
        j = q.popleft()
        for up in upstream[j]:
            if up not in visited:
                visited.add(up)
                q.append(up)
    return np.fromiter(sorted(visited), dtype=np.int64)


def _build_observation_matrix(obs_idx: np.ndarray, upstream: list[list[int]], n_reaches: int) -> csr_matrix:
    row_inds: list[int] = []
    col_inds: list[int] = []
    data: list[float] = []

    cache: dict[int, np.ndarray] = {}
    for r, idx in enumerate(obs_idx.astype(int).tolist()):
        if idx not in cache:
            cache[idx] = _upstream_contributors(idx, upstream)
        cols = cache[idx]
        row_inds.extend([r] * len(cols))
        col_inds.extend(cols.tolist())
        data.extend([1.0] * len(cols))

    return csr_matrix((data, (row_inds, col_inds)), shape=(len(obs_idx), n_reaches), dtype=float)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    last = pd.Timestamp(year=end.year, month=end.month, day=1)
    while cur <= last:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def main() -> None:
    args = parse_args()

    if not (0 <= args.usgs_trust <= 1 and 0 <= args.wam_trust <= 1):
        raise ValueError("--usgs-trust and --wam-trust must be in [0,1]")

    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    usgs_csv = base_dir / "inputData" / "inputs" / "monthly_wide_acft.csv"
    wam_csv = base_dir / "inputData" / "inputs" / "monthly_wide_acft_from_hecdss.csv"
    basin_shp = base_dir / "inputData" / "river_basin" / "TWDB_MRBs_2014.shp"
    flowline_source = (
        base_dir / "inputData" / "nhd_medium_res_gdb"
        if args.network_source == "nhd_medium"
        else base_dir / "inputData" / "texas_nhdplusgrb" / "_extracted_gdb"
    )

    flow, _, topo_comids, downstream, _ = build_network(
        flowline_source=str(flowline_source),
        flowline_layer="NHDFlowline",
        vaa_file=None,
        basin_shp=str(basin_shp),
        basin_name_field="basin_name",
        basin_name_value="Brazos",
    )

    n = len(topo_comids)
    comid_to_idx = {int(c): i for i, c in enumerate(topo_comids.tolist())}
    upstream = _build_reverse_adjacency(downstream)

    # Attributes for regularization terms
    length_km = pd.to_numeric(flow.get("lengthkm", np.nan), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(length_km).any():
        length_km = np.ones(n, dtype=float)

    area_vals = _candidate_area_column(flow)
    if area_vals is None:
        area_vals = np.ones(n, dtype=float)

    strahler = _compute_strahler_order(upstream, n)

    length_norm = _minmax_norm(np.nan_to_num(length_km, nan=np.nanmedian(length_km)))
    area_norm = _minmax_norm(np.nan_to_num(area_vals, nan=np.nanmedian(area_vals)))
    order_norm = _minmax_norm(strahler.astype(float))

    # Higher lambda_i means stronger pull toward prior for that reach.
    lambda_i = args.lambda_base * (
        1.0 + args.lambda_order * order_norm + args.lambda_length * length_norm + args.lambda_area * area_norm
    )
    lambda_i = np.clip(lambda_i, 1e-6, None)

    # Observations
    usgs_obs = monthly_wide_to_observations(usgs_csv, "USGS")
    wam_obs = monthly_wide_to_observations(wam_csv, "WAM")
    obs = pd.concat([usgs_obs, wam_obs], ignore_index=True)
    if obs.empty:
        raise ValueError("No parsed observations from USGS/WAM monthly input files")

    trust_map = {"USGS": float(args.usgs_trust), "WAM": float(args.wam_trust)}
    obs["trust"] = obs["source"].map(trust_map).fillna(1.0)

    # Snap points to nearest flowline
    flow_proj = flow.to_crs(flow.estimate_utm_crs())
    obs_pts = gpd.GeoDataFrame(
        obs.copy().reset_index(drop=True),
        geometry=gpd.points_from_xy(obs["LONG"], obs["LAT"]),
        crs="EPSG:4326",
    ).to_crs(flow_proj.crs)
    obs_pts["obs_row_id"] = np.arange(len(obs_pts), dtype=np.int64)

    snap = gpd.sjoin_nearest(
        obs_pts[["obs_row_id", "geometry"]],
        flow_proj[["COMID", "geometry"]],
        how="left",
        distance_col="snap_dist_m",
    )
    obs2 = obs_pts.drop(columns=["geometry"]).merge(
        snap[["obs_row_id", "COMID", "snap_dist_m"]],
        on="obs_row_id",
        how="left",
    )
    obs2["COMID"] = pd.to_numeric(obs2["COMID"], errors="coerce")
    obs2 = obs2.dropna(subset=["COMID"]).copy()
    obs2["COMID"] = obs2["COMID"].astype(np.int64)
    obs2 = obs2[obs2["COMID"].map(comid_to_idx).notna()].copy()
    obs2["idx"] = obs2["COMID"].map(comid_to_idx).astype(np.int64)

    if obs2.empty:
        raise ValueError("No snapped observations mapped to network reaches")

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    months = _months_between(start, end)

    # Build a prior local-inflow field by month from existing area/yield helper.
    prior_input = obs2.copy().rename(columns={"COMID": "COMID"})
    monthly_prior, _ = build_monthly_yield_prior(
        flow=flow,
        constraints_obs=prior_input,
        downstream=downstream,
        base_coeff_m_per_month=0.0008,
    )
    # build_monthly_yield_prior returns month sequence in sorted(unique dates) of constraints,
    # so map by date available there; fallback to first prior if needed.
    prior_dates = sorted(pd.to_datetime(prior_input["date"].drop_duplicates()).tolist())
    prior_map = {pd.Timestamp(d): monthly_prior[i] for i, d in enumerate(prior_dates)}

    out_rows: list[dict] = []
    enforce_rows: list[dict] = []
    snap_rows: list[dict] = []

    for d in months:
        sub = obs2[obs2["date"] == d].copy()
        if sub.empty:
            continue

        # If multiple points hit same reach-month, keep strongest trust then nearest snap.
        sub = sub.sort_values(["idx", "trust", "snap_dist_m"], ascending=[True, False, True])
        sub = sub.drop_duplicates(subset=["idx"], keep="first").copy()

        y = pd.to_numeric(sub["flow_cms"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.clip(sub["trust"].to_numpy(dtype=float), 0.0, 1.0)
        obs_idx = sub["idx"].to_numpy(dtype=np.int64)

        q_prior = prior_map.get(d, monthly_prior[0] if len(monthly_prior) else np.full(n, 1e-6, dtype=float))
        q_prior = np.clip(np.asarray(q_prior, dtype=float), 0.0, None)

        A = _build_observation_matrix(obs_idx=obs_idx, upstream=upstream, n_reaches=n)
        W = csr_matrix((w, (np.arange(len(w)), np.arange(len(w)))), shape=(len(w), len(w)))
        WA = W @ A
        wy = w * y

        sqrt_lambda = np.sqrt(lambda_i)
        R = csr_matrix((sqrt_lambda, (np.arange(n), np.arange(n))), shape=(n, n))
        rhs_reg = sqrt_lambda * q_prior

        A_aug = vstack([WA, R], format="csr")
        b_aug = np.concatenate([wy, rhs_reg])

        res = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto", verbose=0)
        q_local = np.clip(res.x, 0.0, None)

        q_modeled = route_monthly(q_local, downstream)
        q_out = np.clip(q_modeled, 0.0, float(args.max_output_cms))

        clipped_n = int((q_modeled > float(args.max_output_cms)).sum())
        if clipped_n > 0:
            print(f"[{d.date()}] clipped {clipped_n:,} reaches at {args.max_output_cms:.0f} cms")

        days = float(pd.Timestamp(d).days_in_month)
        for i, c in enumerate(topo_comids):
            out_rows.append(
                {
                    "date": d,
                    "nhd_comid": int(c),
                    "flow_cms": float(q_out[i]),
                    "flow_acft": float(q_out[i] * days * ACFT_PER_CMS_DAY),
                }
            )

        modeled_at_obs = A @ q_local
        for j, (_, r) in enumerate(sub.iterrows()):
            modeled_before = float(modeled_at_obs[j])
            modeled_after = float(q_out[int(r["idx"])])
            enforce_rows.append(
                {
                    "date": d,
                    "CPID": r.get("CPID", ""),
                    "source": r.get("source", ""),
                    "nhd_comid": int(r["COMID"]),
                    "obs_cms": float(y[j]),
                    "trust": float(w[j]),
                    "modeled_before_cms": modeled_before,
                    "modeled_after_cms": modeled_after,
                    "snap_dist_m": float(r.get("snap_dist_m", np.nan)),
                    "solver_status": int(res.status),
                    "solver_cost": float(res.cost),
                }
            )

        snap_rows.extend(sub.to_dict(orient="records"))

    if not out_rows:
        raise ValueError("No months produced output; check date range and observations")

    modeled_df = pd.DataFrame(out_rows)

    # attach centroid coordinates for map joins
    tf = flow[["COMID", "geometry"]].copy()
    tf = gpd.GeoDataFrame(tf, geometry="geometry", crs=flow.crs)
    tf_proj = tf.to_crs(tf.estimate_utm_crs())
    cent = gpd.GeoSeries(tf_proj.geometry.centroid, crs=tf_proj.crs).to_crs(4326)
    coords = pd.DataFrame(
        {
            "nhd_comid": tf["COMID"].astype(np.int64).to_numpy(),
            "lat": cent.y.to_numpy(),
            "lon": cent.x.to_numpy(),
        }
    )

    modeled_df = modeled_df.merge(coords, on="nhd_comid", how="left")
    snap_df = pd.DataFrame(snap_rows)
    if not snap_df.empty:
        snap_df = snap_df.rename(columns={"COMID": "nhd_comid", "LAT": "obs_lat", "LONG": "obs_lon"})
        snap_df = snap_df.merge(coords, on="nhd_comid", how="left")
        snap_df["used_for_enforcement"] = True

    enf_df = pd.DataFrame(enforce_rows)
    if not enf_df.empty:
        for c in ["modeled_before_cms", "modeled_after_cms", "obs_cms"]:
            enf_df[c.replace("_cms", "_acft")] = (
                enf_df[c].astype(float)
                * enf_df["date"].apply(lambda x: pd.Timestamp(x).days_in_month).astype(float)
                * ACFT_PER_CMS_DAY
            )

    flow_out = output_dir / "modeled_monthly_comid_flows_from_points.csv"
    gpkg_out = output_dir / "nhd_brazos_flowlines.gpkg"
    snap_out = output_dir / "snapped_point_diagnostics.csv"
    enf_out = output_dir / "enforcement_diagnostics.csv"

    modeled_df.to_csv(flow_out, index=False)
    flow.to_file(gpkg_out, layer="flowlines", driver="GPKG")
    snap_df.to_csv(snap_out, index=False)
    enf_df.to_csv(enf_out, index=False)

    print(f"Wrote routed flows: {flow_out}")
    print(f"Wrote flowlines: {gpkg_out}")
    print(f"Wrote snapped diagnostics: {snap_out}")
    print(f"Wrote enforcement diagnostics: {enf_out}")
    print(f"Rows: {len(modeled_df):,}")
    print(f"Routed COMIDs: {modeled_df['nhd_comid'].nunique():,}")
    print(f"Months: {modeled_df['date'].nunique():,}")


if __name__ == "__main__":
    main()
