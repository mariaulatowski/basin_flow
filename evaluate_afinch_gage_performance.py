from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MONTH_MAP = [
    ("Oct", "Q01", "QAccConOct"),
    ("Nov", "Q02", "QAccConNov"),
    ("Dec", "Q03", "QAccConDec"),
    ("Jan", "Q04", "QAccConJan"),
    ("Feb", "Q05", "QAccConFeb"),
    ("Mar", "Q06", "QAccConMar"),
    ("Apr", "Q07", "QAccConApr"),
    ("May", "Q08", "QAccConMay"),
    ("Jun", "Q09", "QAccConJun"),
    ("Jul", "Q10", "QAccConJul"),
    ("Aug", "Q11", "QAccConAug"),
    ("Sep", "Q12", "QAccConSep"),
]


def _safe_corr(obs: np.ndarray, sim: np.ndarray) -> float:
    if len(obs) < 2:
        return np.nan
    so = np.std(obs)
    ss = np.std(sim)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1])


def _metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    err = sim - obs
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    denom = float(np.sum(obs))
    pbias = float(100.0 * np.sum(err) / denom) if denom != 0 else np.nan
    obs_var_denom = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = float(1.0 - np.sum((sim - obs) ** 2) / obs_var_denom) if obs_var_denom != 0 else np.nan
    nonzero = np.abs(obs) > 0
    mape = float(np.mean(np.abs(err[nonzero] / obs[nonzero])) * 100.0) if np.any(nonzero) else np.nan
    r = _safe_corr(obs, sim)
    return {
        "MAE_cfs": mae,
        "RMSE_cfs": rmse,
        "Bias_cfs": bias,
        "PBIAS_pct": pbias,
        "R": r,
        "NSE": nse,
        "MAPE_pct": mape,
    }


def _load_station_comid_groups(hsr_dir: Path) -> dict[str, list[int]]:
    # HSR1200 -> THS 1201 -> HSR1201/GagedCatchments
    hsr_name = hsr_dir.name
    ths = hsr_name.replace("HSR", "")
    if len(ths) != 4:
        raise ValueError(f"Expected HSR directory like HSR1200, got: {hsr_name}")
    gaged_dir = hsr_dir.parent / f"HSR{ths[:2]}01" / "GagedCatchments"
    if not gaged_dir.exists():
        raise FileNotFoundError(f"Gaged catchments directory not found: {gaged_dir}")

    station_to_comids: dict[str, list[int]] = {}
    for dat_path in sorted(gaged_dir.glob("*.dat")):
        if dat_path.name.lower() == "stationlist.txt":
            continue
        try:
            d = pd.read_csv(
                dat_path,
                skiprows=1,
                header=None,
                names=["GridCode", "ComID", "AreaSqKm", "ReachCode"],
            )
            comids = (
                pd.to_numeric(d["ComID"], errors="coerce")
                .dropna()
                .astype(np.int64)
                .tolist()
            )
            if comids:
                station_to_comids[dat_path.stem.strip()] = comids
        except Exception:
            continue
    if not station_to_comids:
        raise RuntimeError(f"No station COMID groups found in {gaged_dir}")
    return station_to_comids


def evaluate(hsr_dir: Path, wy: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flowacc_path = hsr_dir / "Output" / "FlowAccum" / f"ComIDQ12WY{wy}.csv"
    measured_path = hsr_dir / "Streamflow" / f"ComIDStationDAMoAnQ{wy}.dat"

    modeled = pd.read_csv(flowacc_path)
    modeled["ComIDVAA"] = pd.to_numeric(modeled["ComIDVAA"], errors="coerce")
    modeled = modeled.dropna(subset=["ComIDVAA"]).copy()
    modeled["ComIDVAA"] = modeled["ComIDVAA"].astype(np.int64)
    modeled = modeled.set_index("ComIDVAA")

    station_to_comids = _load_station_comid_groups(hsr_dir)

    q_cols = [f"Q{i:02d}" for i in range(1, 14)]
    cols = ["ComIDSta", "StaWY", "NWISArea", *q_cols]
    measured = pd.read_csv(measured_path, sep=r"\s+", header=None, names=cols, engine="python")
    measured["StaWY"] = measured["StaWY"].astype(str).str.strip()
    measured = measured[measured["StaWY"].isin(station_to_comids.keys())].copy()

    long_rows: list[pd.DataFrame] = []
    for month_name, obs_col, sim_col in MONTH_MAP:
        sim_vals = []
        sim_comid_counts = []
        for _, row in measured.iterrows():
            station = str(row["StaWY"]).strip()
            comids = station_to_comids.get(station, [])
            if not comids:
                sim_vals.append(np.nan)
                sim_comid_counts.append(0)
                continue
            valid = [c for c in comids if c in modeled.index]
            sim_vals.append(float(modeled.loc[valid, sim_col].sum()) if valid else np.nan)
            sim_comid_counts.append(len(valid))

        tmp = measured[["StaWY", obs_col]].copy()
        tmp = tmp.rename(columns={obs_col: "obs_cfs"})
        tmp["sim_cfs"] = sim_vals
        tmp["n_sim_comids"] = sim_comid_counts
        tmp["month"] = month_name
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["obs_cfs"] = pd.to_numeric(long_df["obs_cfs"], errors="coerce")
    long_df["sim_cfs"] = pd.to_numeric(long_df["sim_cfs"], errors="coerce")
    long_df = long_df.dropna(subset=["obs_cfs", "sim_cfs"]).copy()
    long_df["err_cfs"] = long_df["sim_cfs"] - long_df["obs_cfs"]
    long_df["abs_err_cfs"] = long_df["err_cfs"].abs()

    month_metrics = []
    for mo, grp in long_df.groupby("month", sort=False):
        obs = grp["obs_cfs"].to_numpy(dtype=float)
        sim = grp["sim_cfs"].to_numpy(dtype=float)
        row = {"month": mo, "n": len(grp)}
        row.update(_metrics(obs, sim))
        month_metrics.append(row)

    obs_all = long_df["obs_cfs"].to_numpy(dtype=float)
    sim_all = long_df["sim_cfs"].to_numpy(dtype=float)
    all_row = {"month": "ALL", "n": len(long_df)}
    all_row.update(_metrics(obs_all, sim_all))
    month_metrics.append(all_row)
    month_df = pd.DataFrame(month_metrics)

    station_metrics = []
    for sta, grp in long_df.groupby("StaWY", sort=True):
        obs = grp["obs_cfs"].to_numpy(dtype=float)
        sim = grp["sim_cfs"].to_numpy(dtype=float)
        row = {"Station": sta, "n": len(grp)}
        row.update(_metrics(obs, sim))
        station_metrics.append(row)
    station_df = pd.DataFrame(station_metrics).sort_values("RMSE_cfs", ascending=True)

    return long_df, month_df, station_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AFINCH accumulated flows against gage monthly flows.")
    parser.add_argument("--hsr-dir", default=r"c:\Users\mu3575\Documents\WAM\HSR1200", help="HSR directory path")
    parser.add_argument("--wy", type=int, default=2018, help="Water year")
    args = parser.parse_args()

    hsr_dir = Path(args.hsr_dir)
    long_df, month_df, station_df = evaluate(hsr_dir, args.wy)

    out_dir = hsr_dir / "Output" / "Diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / f"AFINCH_GageComparison_WY{args.wy}_Long.csv"
    month_path = out_dir / f"AFINCH_GageComparison_WY{args.wy}_MonthMetrics.csv"
    station_path = out_dir / f"AFINCH_GageComparison_WY{args.wy}_StationMetrics.csv"

    long_df.to_csv(long_path, index=False)
    month_df.to_csv(month_path, index=False)
    station_df.to_csv(station_path, index=False)

    print(f"Saved: {long_path}")
    print(f"Saved: {month_path}")
    print(f"Saved: {station_path}")
    print("\nMonthly and overall performance:")
    print(month_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print("\nBest 5 stations by RMSE:")
    print(station_df.head(5).to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print("\nWorst 5 stations by RMSE:")
    print(station_df.tail(5).to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


if __name__ == "__main__":
    main()
