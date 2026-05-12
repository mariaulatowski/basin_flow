from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(r"c:\Users\mu3575\Documents\WAM")
WY = 2018

SUBPLOT_DIR = BASE / "HSR1200" / "Output" / "Comparisons" / "all_53_station_subplots"
LONG_OUT = SUBPLOT_DIR / f"obs_vs_modeled_wy{WY}_all53_long.csv"
SUMMARY_OUT = SUBPLOT_DIR / f"obs_vs_modeled_wy{WY}_all53_station_summary.csv"
FLOWACCUM = BASE / "HSR1200" / "Output" / "FlowAccum" / f"ComIDQ12WY{WY}.csv"
STATION_FLOW = BASE / "HSR1200" / "Streamflow" / f"ComIDStationDAMoAnQ{WY}.dat"
NAME_SOURCE = SUBPLOT_DIR / f"obs_vs_modeled_wy{WY}_all53_long.csv"

MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
FLOW_COLS = {
    "Oct": "QAccConOct",
    "Nov": "QAccConNov",
    "Dec": "QAccConDec",
    "Jan": "QAccConJan",
    "Feb": "QAccConFeb",
    "Mar": "QAccConMar",
    "Apr": "QAccConApr",
    "May": "QAccConMay",
    "Jun": "QAccConJun",
    "Jul": "QAccConJul",
    "Aug": "QAccConAug",
    "Sep": "QAccConSep",
}


def _clean_station_name(name: str) -> str:
    s = str(name or "").strip()
    return s if s else "(no name)"


def main() -> None:
    if not FLOWACCUM.exists():
        raise FileNotFoundError(f"Missing flow accumulation CSV: {FLOWACCUM}")
    if not STATION_FLOW.exists():
        raise FileNotFoundError(f"Missing station flow file: {STATION_FLOW}")

    SUBPLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional station-name lookup from existing comparison long CSV.
    name_lookup: dict[str, str] = {}
    if NAME_SOURCE.exists():
        prior = pd.read_csv(NAME_SOURCE)
        if {"Station", "Station_Name"}.issubset(prior.columns):
            prior["Station"] = prior["Station"].astype(str).str.strip()
            for row in prior[["Station", "Station_Name"]].drop_duplicates().itertuples(index=False):
                name_lookup[str(row.Station)] = _clean_station_name(str(row.Station_Name))

    # Current-run station flow file format:
    # ComID StaWY NWISArea Q01..Q12 Q13Annual
    station_raw = pd.read_csv(STATION_FLOW, sep=r"\s+", header=None)
    if station_raw.shape[1] < 16:
        raise ValueError(
            f"Unexpected station flow schema in {STATION_FLOW}; expected >=16 columns, got {station_raw.shape[1]}"
        )
    station_cols = ["ComID", "Station", "NWISArea"] + [f"Q{i:02d}" for i in range(1, 14)]
    station_df = station_raw.iloc[:, :16].copy()
    station_df.columns = station_cols
    station_df["ComID"] = pd.to_numeric(station_df["ComID"], errors="coerce").astype("Int64")
    station_df["Station"] = station_df["Station"].astype(str).str.strip()

    rows = []
    for rec in station_df.itertuples(index=False):
        for i, mo in enumerate(MONTHS, start=1):
            rows.append(
                {
                    "Station": rec.Station,
                    "Station_Name": name_lookup.get(rec.Station, "(no name)"),
                    "ComID": int(rec.ComID) if pd.notna(rec.ComID) else np.nan,
                    "Month": mo,
                    "Observed_CFS": float(getattr(rec, f"Q{i:02d}")),
                }
            )
    long_df = pd.DataFrame(rows)

    flow_df = pd.read_csv(FLOWACCUM, usecols=["ComIDVAA", *FLOW_COLS.values()])
    flow_df["ComIDVAA"] = pd.to_numeric(flow_df["ComIDVAA"], errors="coerce").astype("Int64")
    flow_df = flow_df.dropna(subset=["ComIDVAA"]).copy()
    flow_df["ComIDVAA"] = flow_df["ComIDVAA"].astype("int64")
    flow_df = flow_df.set_index("ComIDVAA")

    long_df["ComID"] = pd.to_numeric(long_df["ComID"], errors="coerce").astype("Int64")
    long_df["Observed_CFS"] = pd.to_numeric(long_df["Observed_CFS"], errors="coerce")

    modeled_vals = []
    for row in long_df.itertuples(index=False):
        month = str(row.Month)
        comid = row.ComID
        if pd.isna(comid) or month not in FLOW_COLS:
            modeled_vals.append(np.nan)
            continue
        c = int(comid)
        if c not in flow_df.index:
            modeled_vals.append(np.nan)
            continue
        modeled_vals.append(float(flow_df.at[c, FLOW_COLS[month]]))

    long_df["Modeled_CFS"] = modeled_vals
    long_df["Delta_CFS"] = long_df["Modeled_CFS"] - long_df["Observed_CFS"]

    # Keep station order as it appears in the current-run station file.
    station_ids = station_df["Station"].astype(str).tolist()
    long_df["Station"] = pd.Categorical(long_df["Station"], categories=station_ids, ordered=True)
    long_df["Station"] = long_df["Station"].astype(str)
    long_df.to_csv(LONG_OUT, index=False)

    summary_rows = []
    station_ids = station_df["Station"].astype(str).tolist()
    for sta in station_ids:
        s = long_df[long_df["Station"].astype(str) == sta].copy()
        s = s[np.isfinite(s["Observed_CFS"]) & np.isfinite(s["Modeled_CFS"])].copy()
        if s.empty:
            rmse = np.nan
            mae = np.nan
            comid = np.nan
            name = "(no data)"
        else:
            err = s["Modeled_CFS"].to_numpy() - s["Observed_CFS"].to_numpy()
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            comid = int(s["ComID"].iloc[0]) if pd.notna(s["ComID"].iloc[0]) else np.nan
            name = _clean_station_name(s["Station_Name"].iloc[0])
        summary_rows.append(
            {
                "Station": sta,
                "Station_Name": name,
                "ComID": comid,
                "RMSE_CFS": rmse,
                "MAE_CFS": mae,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    # Clear old page files before regenerating.
    for old in SUBPLOT_DIR.glob(f"obs_vs_modeled_wy{WY}_page_*.png"):
        old.unlink()

    per_page = 9
    n_stations = len(station_ids)
    n_pages = int(math.ceil(n_stations / per_page))

    month_to_x = {m: i for i, m in enumerate(MONTHS)}

    for page in range(n_pages):
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
        axes = axes.flatten()
        start = page * per_page
        end = min(start + per_page, n_stations)

        for ax_idx, sta in enumerate(station_ids[start:end]):
            ax = axes[ax_idx]
            s = long_df[long_df["Station"].astype(str) == sta].copy()
            s["x"] = s["Month"].map(month_to_x)
            s = s.sort_values("x")

            y_obs = s["Observed_CFS"].to_numpy(dtype=float)
            y_mod = s["Modeled_CFS"].to_numpy(dtype=float)

            ax.plot(MONTHS, y_obs, marker="o", linewidth=1.5, label="Observed")
            ax.plot(MONTHS, y_mod, marker="s", linewidth=1.2, label="Modeled")

            name = _clean_station_name(s["Station_Name"].iloc[0]) if not s.empty else "(no name)"
            name_short = (name[:42] + "...") if len(name) > 45 else name

            ss = summary_df[summary_df["Station"].astype(str) == str(sta)]
            rmse = float(ss["RMSE_CFS"].iloc[0]) if not ss.empty else np.nan
            mae = float(ss["MAE_CFS"].iloc[0]) if not ss.empty else np.nan

            ax.set_title(f"{sta} | RMSE={rmse:.3g} MAE={mae:.3g}\n{name_short}", fontsize=9)
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.25)

        for ax in axes[end - start :]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Observed vs Modeled Monthly CFS at 53 Stations - WY{WY} (Current Run)", fontsize=14)

        out_png = SUBPLOT_DIR / f"obs_vs_modeled_wy{WY}_page_{page + 1:02d}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

    print(f"Wrote: {LONG_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")
    print(f"Wrote {n_pages} page plot(s) to: {SUBPLOT_DIR}")


if __name__ == "__main__":
    main()
