#!/usr/bin/env python
"""
AFINCH-style monthly routing on Brazos NHD flowlines using BOTH USGS and WAM points.

This is a clean standalone script focused on one mode:
- network domain: NHD flowlines
- routing mode: AFINCH-style iterative upstream yield adjustment
- constraints: combined USGS + WAM monthly_wide observations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from brazos_streamflow_model import build_network, route_monthly
from route_points_to_comid import (
    afinch_route_monthly,
    build_monthly_yield_prior,
    build_source_trust_map,
    build_upstream_index_cache,
    monthly_wide_to_observations,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AFINCH-style routing with combined USGS and WAM constraints"
    )
    p.add_argument("--start-date", default=None, help="Inclusive month start, format YYYY-MM-01")
    p.add_argument("--end-date", default=None, help="Inclusive month end, format YYYY-MM-01")
    p.add_argument("--usgs-trust", type=float, default=1.0, help="USGS trust weight in [0, 1]")
    p.add_argument("--wam-trust", type=float, default=0.75, help="WAM trust weight in [0, 1]")
    p.add_argument("--afinch-iters", type=int, default=8, help="AFINCH iteration count")
    p.add_argument("--afinch-damping", type=float, default=0.9, help="AFINCH damping in [0, 1]")
    p.add_argument(
        "--usgs-input-csv",
        default="inputData/inputs/monthly_wide_acft.csv",
        help="USGS monthly-wide input CSV path, relative to workspace root or absolute",
    )
    p.add_argument(
        "--wam-input-csv",
        default="inputData/inputs/monthly_wide_acft_from_hecdss.csv",
        help="WAM monthly-wide input CSV path, relative to workspace root or absolute",
    )
    p.add_argument(
        "--input-units",
        choices=["acft", "cfs"],
        default="acft",
        help="Units used in monthly columns of both input CSV files",
    )
    p.add_argument(
        "--max-output-cms",
        type=float,
        default=10_000.0,
        help="Cap routed output flow to avoid split/rejoin divergence artifacts",
    )
    p.add_argument(
        "--output-dir",
        default="output/nhd_afinch_usgs_wam",
        help="Directory for csv/gpkg outputs",
    )
    p.add_argument(
        "--network-source",
        choices=["nhd_hr", "nhd_medium"],
        default="nhd_hr",
        help=(
            "Flowline network to route on. "
            "'nhd_hr' = NHD High-Resolution (inputData/texas_nhdplusgrb/_extracted_gdb, ~281k reaches). "
            "'nhd_medium' = NHD Medium-Resolution (inputData/nhd_medium_res_gdb, ~24k Brazos reaches)."
        ),
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()

    if not (0.0 <= float(args.usgs_trust) <= 1.0 and 0.0 <= float(args.wam_trust) <= 1.0):
        raise ValueError("Source trust values must be between 0 and 1.")
    if int(args.afinch_iters) < 1:
        raise ValueError("--afinch-iters must be at least 1.")
    if not (0.0 <= float(args.afinch_damping) <= 1.0):
        raise ValueError("--afinch-damping must be between 0 and 1.")
    if float(args.max_output_cms) <= 0:
        raise ValueError("--max-output-cms must be > 0.")

    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir / args.output_dir).resolve()

    usgs_csv = Path(args.usgs_input_csv)
    if not usgs_csv.is_absolute():
        usgs_csv = (base_dir / usgs_csv).resolve()
    wam_csv = Path(args.wam_input_csv)
    if not wam_csv.is_absolute():
        wam_csv = (base_dir / wam_csv).resolve()
    if args.network_source == "nhd_medium":
        flowline_source = base_dir / "inputData" / "nhd_medium_res_gdb"
    else:
        flowline_source = base_dir / "inputData" / "texas_nhdplusgrb" / "_extracted_gdb"
    basin_shp = base_dir / "inputData" / "river_basin" / "TWDB_MRBs_2014.shp"

    output_csv = output_dir / "modeled_monthly_comid_flows_from_points.csv"
    snapped_points_csv = output_dir / "snapped_point_diagnostics.csv"
    enforcement_csv = output_dir / "enforcement_diagnostics.csv"
    conflict_csv = output_dir / "constraint_conflicts.csv"
    flowlines_gpkg = output_dir / "nhd_brazos_flowlines.gpkg"

    if not usgs_csv.exists():
        raise FileNotFoundError(f"Missing USGS CSV: {usgs_csv}")
    if not wam_csv.exists():
        raise FileNotFoundError(f"Missing WAM CSV: {wam_csv}")

    flow, _, topo_comids, downstream, _ = build_network(
        flowline_source=str(flowline_source),
        flowline_layer="NHDFlowline",
        vaa_file=None,
        basin_shp=str(basin_shp),
        basin_name_field="basin_name",
        basin_name_value="Brazos",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    flow.to_file(flowlines_gpkg, layer="flowlines", driver="GPKG")

    usgs_obs = monthly_wide_to_observations(usgs_csv, "USGS", input_units=args.input_units)
    wam_obs = monthly_wide_to_observations(wam_csv, "WAM", input_units=args.input_units)
    obs = pd.concat([usgs_obs, wam_obs], ignore_index=True)
    if obs.empty:
        raise ValueError("No observations parsed from input CSV files.")

    source_trust = build_source_trust_map(args.usgs_trust, args.wam_trust)
    obs["trust"] = obs["source"].map(source_trust).fillna(1.0).astype(float)

    # If both sources contain same CPID in same month, keep WAM by default (legacy behavior).
    obs["priority"] = np.where(obs["source"].eq("WAM"), 2, 1)
    obs = obs.sort_values(["date", "CPID", "priority"]).drop_duplicates(["date", "CPID"], keep="last")

    flow_proj = flow.to_crs(flow.estimate_utm_crs())
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
        valid_obs = valid_obs[valid_obs["date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        valid_obs = valid_obs[valid_obs["date"] <= pd.Timestamp(args.end_date)].copy()
    if valid_obs.empty:
        raise ValueError("No observations remain after applying date filter.")

    # If multiple points land on same COMID and month, keep highest trust then nearest.
    valid_obs = valid_obs.sort_values(
        ["date", "idx", "trust", "snap_dist_m", "CPID"],
        ascending=[True, True, False, True, True],
    ).copy()
    conflict_mask = valid_obs.duplicated(subset=["date", "idx"], keep=False)
    conflict_rows = valid_obs.loc[conflict_mask].copy()
    constraints_obs = valid_obs.drop_duplicates(subset=["date", "idx"], keep="first").copy()

    constraints_obs["obs_acft"] = constraints_obs.apply(
        lambda r: float(r["flow_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
        axis=1,
    )

    monthly_local_q, prior_yield_diag = build_monthly_yield_prior(
        flow=flow,
        constraints_obs=constraints_obs,
        downstream=downstream,
        base_coeff_m_per_month=0.0008,
    )
    dates = sorted(constraints_obs["date"].drop_duplicates())
    date_to_local_q = {pd.Timestamp(d): monthly_local_q[i] for i, d in enumerate(sorted(prior_yield_diag))}
    if not date_to_local_q and dates:
        date_to_local_q = {pd.Timestamp(d): monthly_local_q[0] for d in dates}

    upstream_cache = build_upstream_index_cache(
        downstream=downstream,
        constraint_indices=constraints_obs["idx"].astype(int).tolist(),
    )

    # Protect NHD ArtificialPath segments from iterative ratio corrections.
    if "ftype" in flow.columns:
        ftype_vals = pd.to_numeric(flow["ftype"], errors="coerce").fillna(-1).astype(int)
        protected_mask = (ftype_vals == 558).to_numpy(dtype=bool)
    else:
        protected_mask = None

    out_rows: list[dict] = []
    enforcement_rows: list[dict] = []

    for d in dates:
        local_q = date_to_local_q.get(pd.Timestamp(d), monthly_local_q[0])
        q_prior = route_monthly(local_q, downstream)

        sub = constraints_obs[constraints_obs["date"] == d].sort_values("idx")
        targets: dict[int, float] = {}
        trusts: dict[int, float] = {}
        target_meta: dict[int, dict[str, float]] = {}

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
            protected_mask=protected_mask,
            max_multiplier=10.0,
        )

        q_out = np.clip(q, 0.0, float(args.max_output_cms))
        n_clipped = int((q > float(args.max_output_cms)).sum())
        if n_clipped > 0:
            print(f"[{pd.Timestamp(d).date()}] clipped {n_clipped:,} reaches at {args.max_output_cms:.0f} cms")

        for _, r in sub.iterrows():
            i = int(r["idx"])
            meta = target_meta[i]
            before = float(q_prior[i])
            enforcement_rows.append(
                {
                    "date": pd.Timestamp(d),
                    "CPID": r["CPID"],
                    "source": r["source"],
                    "nhd_comid": int(topo_comids[i]),
                    "modeled_before_cms": before,
                    "obs_cms": float(meta["obs_cms"]),
                    "effective_target_cms": float(meta["effective_target_cms"]),
                    "trust": float(meta["trust"]),
                    "modeled_after_cms": float(q_out[i]),
                    "routing_mode": "afinch",
                    "prior_yield_cms_per_km2": prior_yield_diag.get(pd.Timestamp(d), {}).get(
                        "prior_yield_cms_per_km2", np.nan
                    ),
                }
            )

        for i, c in enumerate(topo_comids):
            out_rows.append({"date": pd.Timestamp(d), "nhd_comid": int(c), "flow_cms": float(q_out[i])})

    modeled = pd.DataFrame(out_rows)
    modeled["flow_acft"] = modeled.apply(
        lambda r: float(r["flow_cms"]) * (pd.Timestamp(r["date"]).days_in_month * 24 * 3600) / 1233.48184,
        axis=1,
    )

    # Join centroid coordinates for mapped outputs.
    tf = flow[["COMID", "geometry"]].copy()
    tf = gpd.GeoDataFrame(tf, geometry="geometry", crs=flow.crs)
    tf_proj = tf.to_crs(tf.estimate_utm_crs())
    tf_cent = gpd.GeoSeries(tf_proj.geometry.centroid, crs=tf_proj.crs).to_crs(4326)
    tf_coords = pd.DataFrame(
        {"nhd_comid": tf["COMID"].astype(np.int64).to_numpy(), "lat": tf_cent.y.to_numpy(), "lon": tf_cent.x.to_numpy()}
    )

    modeled = modeled.merge(tf_coords, on="nhd_comid", how="left")

    snapped_diag = valid_obs.rename(columns={"COMID": "nhd_comid", "LAT": "obs_lat", "LONG": "obs_lon"}).copy()
    snapped_diag = snapped_diag.merge(tf_coords, on="nhd_comid", how="left")
    snapped_diag["used_for_enforcement"] = snapped_diag.set_index(["date", "idx"]).index.isin(
        constraints_obs.set_index(["date", "idx"]).index
    )

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

    modeled.to_csv(output_csv, index=False)
    snapped_diag.to_csv(snapped_points_csv, index=False)
    enforcement_diag.to_csv(enforcement_csv, index=False)

    if conflict_rows.empty:
        pd.DataFrame(columns=["date", "idx", "CPID", "source", "trust", "flow_cms", "snap_dist_m"]).to_csv(
            conflict_csv, index=False
        )
    else:
        out = conflict_rows[["date", "idx", "CPID", "source", "trust", "flow_cms", "snap_dist_m"]].copy()
        out.to_csv(conflict_csv, index=False)

    print(f"Wrote routed COMID flows: {output_csv}")
    print(f"Wrote flowlines gpkg: {flowlines_gpkg}")
    print(f"Wrote snapped point diagnostics: {snapped_points_csv}")
    print(f"Wrote enforcement diagnostics: {enforcement_csv}")
    print(f"Wrote constraint conflicts: {conflict_csv}")
    print(f"AFINCH iterations: {args.afinch_iters}, damping: {args.afinch_damping:.2f}")
    print(f"Source trust: USGS={args.usgs_trust:.2f}, WAM={args.wam_trust:.2f}")
    print(f"Constraints after conflict resolution: {len(constraints_obs):,} (from {len(valid_obs):,} snapped points)")
    print(f"Rows: {len(modeled):,}")
    print(f"Routed COMIDs: {modeled['nhd_comid'].nunique():,}")
    print(f"Months: {modeled['date'].nunique():,}")


if __name__ == "__main__":
    run()
