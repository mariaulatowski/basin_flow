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
        description=(
            "Section-based holdout evaluation: find high-density USGS sections, "
            "withhold a subset of gages, reroute, and score predictive skill."
        )
    )
    p.add_argument("--year", type=int, default=2018)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--network-source", choices=["nhd_medium", "nhd_hr"], default="nhd_medium")
    p.add_argument("--section-cell-deg", type=float, default=0.5, help="Lat/lon cell size (degrees) for section search.")
    p.add_argument("--min-usgs-per-section", type=int, default=8)
    p.add_argument("--top-sections", type=int, default=4)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--min-holdout", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=1, help="Number of random holdout realizations to run.")
    p.add_argument("--usgs-trust", type=float, default=1.0)
    p.add_argument("--wam-trust", type=float, default=0.75)
    p.add_argument("--afinch-iters", type=int, default=8)
    p.add_argument("--afinch-damping", type=float, default=0.9)
    p.add_argument("--output-dir", default="output/section_holdout_eval")
    return p.parse_args()


def _metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    e = sim - obs
    mae = float(np.mean(np.abs(e))) if len(e) else np.nan
    rmse = float(np.sqrt(np.mean(e**2))) if len(e) else np.nan
    bias = float(np.mean(e)) if len(e) else np.nan
    pbias = float(100.0 * np.sum(e) / np.sum(obs)) if len(e) and np.sum(obs) != 0 else np.nan
    r = float(np.corrcoef(obs, sim)[0, 1]) if len(e) > 1 and np.std(obs) > 0 and np.std(sim) > 0 else np.nan
    denom = float(np.sum((obs - np.mean(obs)) ** 2)) if len(e) else np.nan
    nse = float(1.0 - np.sum((sim - obs) ** 2) / denom) if len(e) and denom and denom > 0 else np.nan
    return {"n": int(len(e)), "MAE_cms": mae, "RMSE_cms": rmse, "Bias_cms": bias, "PBIAS_pct": pbias, "R": r, "NSE": nse}


def _snap_obs_to_flow(flow: gpd.GeoDataFrame, obs: pd.DataFrame) -> pd.DataFrame:
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
    out = obs_pts.drop(columns=["geometry"]).merge(
        snap[["obs_row_id", "COMID", "snap_dist_m"]],
        on="obs_row_id",
        how="left",
    )
    out["COMID"] = pd.to_numeric(out["COMID"], errors="coerce")
    out = out.dropna(subset=["COMID"]).copy()
    out["COMID"] = out["COMID"].astype(np.int64)
    return out


def main() -> None:
    args = parse_args()
    if not (1 <= args.month <= 12):
        raise ValueError("--month must be 1..12")
    if args.top_sections < 1:
        raise ValueError("--top-sections must be >=1")
    if not (0 < args.holdout_frac < 1):
        raise ValueError("--holdout-frac must be in (0,1)")
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >=1")

    base_dir = Path(__file__).resolve().parent
    out_dir = (base_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = base_dir / "inputData" / "inputs"
    usgs_csv = inputs_dir / "monthly_wide_acft.csv"
    wam_csv = inputs_dir / "monthly_wide_acft_from_hecdss.csv"
    basin_shp = base_dir / "inputData" / "river_basin" / "TWDB_MRBs_2014.shp"
    flowline_source = (
        base_dir / "inputData" / "nhd_medium_res_gdb"
        if args.network_source == "nhd_medium"
        else base_dir / "inputData" / "texas_nhdplusgrb" / "_extracted_gdb"
    )

    print("Building network...")
    flow, _, topo_comids, downstream, _ = build_network(
        flowline_source=str(flowline_source),
        flowline_layer="NHDFlowline",
        vaa_file=None,
        basin_shp=str(basin_shp),
        basin_name_field="basin_name",
        basin_name_value="Brazos",
    )
    comid_to_idx = {int(c): i for i, c in enumerate(topo_comids.tolist())}

    print("Loading monthly observations...")
    usgs_obs = monthly_wide_to_observations(usgs_csv, "USGS")
    wam_obs = monthly_wide_to_observations(wam_csv, "WAM")
    obs = pd.concat([usgs_obs, wam_obs], ignore_index=True)
    obs = obs[(obs["date"].dt.year == args.year) & (obs["date"].dt.month == args.month)].copy()
    if obs.empty:
        raise ValueError("No observations for selected year/month.")

    source_trust = build_source_trust_map(args.usgs_trust, args.wam_trust)
    obs["trust"] = obs["source"].map(source_trust).fillna(1.0).astype(float)

    print("Snapping observations to routed COMIDs...")
    obs = _snap_obs_to_flow(flow, obs)
    obs = obs[obs["COMID"].map(comid_to_idx).notna()].copy()
    obs["idx"] = obs["COMID"].map(comid_to_idx).astype(np.int64)
    if obs.empty:
        raise ValueError("No snapped observations mapped to network.")

    # Resolve duplicates on same routed reach/date: keep highest trust then nearest
    obs = obs.sort_values(["date", "idx", "trust", "snap_dist_m", "CPID"], ascending=[True, True, False, True, True])
    obs = obs.drop_duplicates(["date", "idx"], keep="first").copy()

    usgs = obs[obs["source"] == "USGS"].copy()
    if usgs.empty:
        raise ValueError("No USGS points after snapping/filtering.")

    # Build simple spatial sections by lat/lon bins and rank by USGS density.
    cell = float(args.section_cell_deg)
    usgs["cell_x"] = np.floor(usgs["LONG"] / cell).astype(int)
    usgs["cell_y"] = np.floor(usgs["LAT"] / cell).astype(int)
    counts = (
        usgs.groupby(["cell_x", "cell_y"], as_index=False)
        .agg(n_usgs=("CPID", "count"), lat_mean=("LAT", "mean"), lon_mean=("LONG", "mean"))
        .sort_values("n_usgs", ascending=False)
    )
    candidates = counts[counts["n_usgs"] >= int(args.min_usgs_per_section)].head(int(args.top_sections)).copy()
    if candidates.empty:
        raise ValueError("No sections meet min-usgs-per-section threshold.")

    candidates["section_id"] = [f"S{i+1}" for i in range(len(candidates))]
    candidates.to_csv(out_dir / "section_candidates.csv", index=False)

    # Build protection mask for artificial paths (if present).
    if "ftype" in flow.columns:
        ftype_vals = pd.to_numeric(flow["ftype"], errors="coerce").fillna(-1).astype(int)
        protected_mask = (ftype_vals == 558).to_numpy(dtype=bool)
    else:
        protected_mask = None

    detail_rows: list[dict] = []
    section_metric_rows: list[dict] = []
    seed_values = [int(args.seed) + i for i in range(int(args.n_seeds))]

    for seed in seed_values:
        rng = np.random.default_rng(seed)
        seed_detail_rows: list[dict] = []
        print(f"\nRunning holdout seed={seed}...")

        # Evaluate each section independently (hold out only USGS points in section).
        for _, sec in candidates.iterrows():
            section_id = str(sec["section_id"])
            sx = int(sec["cell_x"])
            sy = int(sec["cell_y"])
            sec_usgs = usgs[(usgs["cell_x"] == sx) & (usgs["cell_y"] == sy)].copy()
            n_sec = len(sec_usgs)
            n_hold = max(int(np.ceil(n_sec * args.holdout_frac)), int(args.min_holdout))
            n_hold = min(n_hold, max(n_sec - 1, 1))

            hold_idx = rng.choice(sec_usgs.index.to_numpy(), size=n_hold, replace=False)
            hold = sec_usgs.loc[hold_idx].copy()
            train = obs.drop(index=hold.index, errors="ignore").copy()

            # Build monthly prior from training constraints (single month slice).
            monthly_local_q, _ = build_monthly_yield_prior(
                flow=flow,
                constraints_obs=train,
                downstream=downstream,
                base_coeff_m_per_month=0.0008,
            )
            local_q = monthly_local_q[0]
            q_prior = route_monthly(local_q, downstream)

            sub = train.sort_values("idx")
            targets: dict[int, float] = {}
            trusts: dict[int, float] = {}
            for _, r in sub.iterrows():
                i = int(r["idx"])
                trust = float(r.get("trust", 1.0))
                obs_q = float(r["flow_cms"])
                targets[i] = float(q_prior[i]) + trust * (obs_q - float(q_prior[i]))
                trusts[i] = trust

            upstream_cache = build_upstream_index_cache(downstream=downstream, constraint_indices=list(targets.keys()))
            q_sim, _ = afinch_route_monthly(
                local_prior_q=local_q,
                constraint_targets=targets,
                constraint_trust=trusts,
                downstream=downstream,
                upstream_cache=upstream_cache,
                max_iters=int(args.afinch_iters),
                damping=float(args.afinch_damping),
                protected_mask=protected_mask,
                max_multiplier=10.0,
            )

            # Score holdouts.
            obs_vec = []
            sim_vec = []
            for _, r in hold.iterrows():
                i = int(r["idx"])
                obs_q = float(r["flow_cms"])
                sim_q = float(q_sim[i])
                obs_vec.append(obs_q)
                sim_vec.append(sim_q)
                row = {
                    "seed": int(seed),
                    "section_id": section_id,
                    "cell_x": sx,
                    "cell_y": sy,
                    "CPID": r.get("CPID", ""),
                    "source": r.get("source", ""),
                    "LAT": float(r["LAT"]),
                    "LONG": float(r["LONG"]),
                    "nhd_comid": int(r["COMID"]),
                    "obs_cms": obs_q,
                    "sim_cms": sim_q,
                    "err_cms": sim_q - obs_q,
                    "abs_err_cms": abs(sim_q - obs_q),
                }
                seed_detail_rows.append(row)
                detail_rows.append(row)

            met = _metrics(np.asarray(obs_vec, dtype=float), np.asarray(sim_vec, dtype=float))
            met.update(
                {
                    "seed": int(seed),
                    "section_id": section_id,
                    "cell_x": sx,
                    "cell_y": sy,
                    "n_usgs_section": int(n_sec),
                    "n_holdout": int(n_hold),
                    "lat_mean": float(sec["lat_mean"]),
                    "lon_mean": float(sec["lon_mean"]),
                    "year": int(args.year),
                    "month": int(args.month),
                }
            )
            section_metric_rows.append(met)
            print(
                f"{section_id}: n={n_sec}, holdout={n_hold}, "
                f"RMSE={met['RMSE_cms']:.3f} cms, NSE={met['NSE']:.3f}, R={met['R']:.3f}"
            )

        seed_df = pd.DataFrame(seed_detail_rows)
        overall = _metrics(seed_df["obs_cms"].to_numpy(float), seed_df["sim_cms"].to_numpy(float))
        overall_row = {
            "seed": int(seed),
            "section_id": "ALL",
            "cell_x": np.nan,
            "cell_y": np.nan,
            "n_usgs_section": int(candidates["n_usgs"].sum()),
            "n_holdout": int(len(seed_df)),
            "lat_mean": np.nan,
            "lon_mean": np.nan,
            "year": int(args.year),
            "month": int(args.month),
            **overall,
        }
        section_metric_rows.append(overall_row)
        print(
            f"ALL(seed={seed}): holdout={overall_row['n_holdout']}, "
            f"RMSE={overall_row['RMSE_cms']:.3f} cms, NSE={overall_row['NSE']:.3f}, R={overall_row['R']:.3f}"
        )

    detail_df = pd.DataFrame(detail_rows)
    sec_df = pd.DataFrame(section_metric_rows).sort_values(["section_id", "seed"], kind="stable")

    detail_path = out_dir / f"section_holdout_predictions_{args.year}{args.month:02d}.csv"
    metric_path = out_dir / f"section_holdout_metrics_{args.year}{args.month:02d}.csv"
    summary_path = out_dir / f"section_holdout_summary_{args.year}{args.month:02d}.csv"
    sec_df.to_csv(metric_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    if int(args.n_seeds) > 1:
        summary_rows: list[dict] = []
        for section_id, g in sec_df.groupby("section_id", sort=False):
            g = g.copy()
            row = {
                "section_id": str(section_id),
                "n_seeds": int(len(g)),
                "RMSE_mean": float(g["RMSE_cms"].mean()),
                "RMSE_median": float(g["RMSE_cms"].median()),
                "RMSE_p10": float(g["RMSE_cms"].quantile(0.10)),
                "RMSE_p90": float(g["RMSE_cms"].quantile(0.90)),
                "NSE_mean": float(g["NSE"].mean()),
                "NSE_median": float(g["NSE"].median()),
                "NSE_p10": float(g["NSE"].quantile(0.10)),
                "NSE_p90": float(g["NSE"].quantile(0.90)),
                "PBIAS_mean": float(g["PBIAS_pct"].mean()),
                "PBIAS_median": float(g["PBIAS_pct"].median()),
                "PBIAS_p10": float(g["PBIAS_pct"].quantile(0.10)),
                "PBIAS_p90": float(g["PBIAS_pct"].quantile(0.90)),
                "R_mean": float(g["R"].mean()),
                "R_median": float(g["R"].median()),
                "R_p10": float(g["R"].quantile(0.10)),
                "R_p90": float(g["R"].quantile(0.90)),
            }
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nSaved section candidates: {out_dir / 'section_candidates.csv'}")
    print(f"Saved section metrics:    {metric_path}")
    print(f"Saved holdout details:    {detail_path}")
    if int(args.n_seeds) > 1:
        print(f"Saved seed summary:       {summary_path}")
    print("\nSection metrics:")
    print(sec_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
