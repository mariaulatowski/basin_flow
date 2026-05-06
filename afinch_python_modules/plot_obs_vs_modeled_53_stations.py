from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(r"c:\Users\mu3575\Documents\WAM")
HSR_DIR = BASE_DIR / "HSR1200"
OUT_DIR = HSR_DIR / "Output" / "Comparisons" / "all_53_station_subplots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAT_PATH = HSR_DIR / "Streamflow" / "ComIDStationDAMoAnQ2018.dat"
STATION_MAP_PATH = HSR_DIR / "Streamflow" / "StationDASqMi.csv"
STATION_COMID_PATH = HSR_DIR / "Flowlines" / "StationComID.csv"
STATION_LIST_PATH = HSR_DIR / "GagedCatchments" / "StationList.txt"
FLOWACC_PATH = HSR_DIR / "Output" / "FlowAccum" / "ComIDQ12WY2018.csv"
MONTHLY_WIDE_PATH = BASE_DIR / "inputData" / "inputs" / "monthly_wide_cfs.csv"

MONTHS_WY = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
Q_COLS = [f"Q{i:02d}" for i in range(1, 13)]
FLOW_COLS = [f"QAccCon{m}" for m in MONTHS_WY]


def _norm_station(x: object) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.lstrip("0")
    return s if s else "0"


def _load_station_names() -> dict[str, str]:
    if not MONTHLY_WIDE_PATH.exists():
        return {}
    df = pd.read_csv(MONTHLY_WIDE_PATH)
    if "Gage_ID_norm" not in df.columns or "Station_Name" not in df.columns:
        return {}
    df = df[df["Year"] == 2018].copy()
    df["Station"] = df["Gage_ID_norm"].map(_norm_station)
    df = df.drop_duplicates(subset=["Station"], keep="first")
    return dict(zip(df["Station"], df["Station_Name"]))


def main() -> None:
    # Active USGS station list (53 expected for current setup)
    station_list = [
        _norm_station(x)
        for x in STATION_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if str(x).strip()
    ]

    # Station -> COMID mapping from rebuilt flowline mapping table.
    station_map = pd.read_csv(STATION_COMID_PATH)
    station_map["Station"] = station_map["Station"].map(_norm_station)
    station_map["ComID"] = pd.to_numeric(station_map["ComID"], errors="coerce")
    station_map = station_map.dropna(subset=["ComID"]).copy()
    station_map["ComID"] = station_map["ComID"].astype("int64")
    station_map = station_map[station_map["Station"].isin(station_list)].copy()
    station_map = station_map.drop_duplicates(subset=["Station"], keep="first")

    # Observed monthly station flow (Q01..Q12 == Oct..Sep)
    dat_cols = ["ComIDSta", "Station", "AreaSqMi"] + [f"Q{i:02d}" for i in range(1, 14)]
    dat = pd.read_csv(DAT_PATH, sep=r"\s+", header=None, names=dat_cols, engine="python")
    dat["Station"] = dat["Station"].map(_norm_station)
    dat = dat.drop_duplicates(subset=["Station"], keep="first")

    # Modeled monthly flow at each COMID
    flow = pd.read_csv(FLOWACC_PATH)
    flow["ComIDVAA"] = pd.to_numeric(flow["ComIDVAA"], errors="coerce")
    flow = flow.dropna(subset=["ComIDVAA"]).copy()
    flow["ComIDVAA"] = flow["ComIDVAA"].astype("int64")
    flow = flow.drop_duplicates(subset=["ComIDVAA"], keep="first")

    station_names = _load_station_names()

    # Build station-level monthly comparison table
    rows: list[dict[str, object]] = []
    for _, rec in station_map.sort_values("Station").iterrows():
        sta = rec["Station"]
        comid = int(rec["ComID"])

        dat_match = dat[dat["Station"] == sta]
        flow_match = flow[flow["ComIDVAA"] == comid]

        if dat_match.empty or flow_match.empty:
            continue

        d = dat_match.iloc[0]
        f = flow_match.iloc[0]

        obs = np.array([pd.to_numeric(d[q], errors="coerce") for q in Q_COLS], dtype=float)
        mod = np.array([pd.to_numeric(f[c], errors="coerce") for c in FLOW_COLS], dtype=float)

        for i, month in enumerate(MONTHS_WY):
            rows.append(
                {
                    "Station": sta,
                    "Station_Name": station_names.get(sta, ""),
                    "ComID": comid,
                    "Month": month,
                    "Observed_CFS": float(obs[i]),
                    "Modeled_CFS": float(mod[i]),
                    "Delta_CFS": float(mod[i] - obs[i]),
                }
            )

    comp = pd.DataFrame(rows)
    comp_csv = OUT_DIR / "obs_vs_modeled_wy2018_all53_long.csv"
    comp.to_csv(comp_csv, index=False)

    # Create one summary row per station
    station_summary = []
    for sta, g in comp.groupby("Station", sort=True):
        obs = g["Observed_CFS"].to_numpy(dtype=float)
        mod = g["Modeled_CFS"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
        mae = float(np.mean(np.abs(mod - obs)))
        station_summary.append(
            {
                "Station": sta,
                "Station_Name": g["Station_Name"].iloc[0],
                "ComID": int(g["ComID"].iloc[0]),
                "RMSE_CFS": rmse,
                "MAE_CFS": mae,
            }
        )
    summary_df = pd.DataFrame(station_summary).sort_values("Station")
    summary_csv = OUT_DIR / "obs_vs_modeled_wy2018_all53_station_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Subplot pages
    stations = summary_df["Station"].tolist()
    n_per_page = 9
    n_rows, n_cols = 3, 3
    n_pages = int(ceil(len(stations) / n_per_page))

    for page in range(n_pages):
        start = page * n_per_page
        end = min((page + 1) * n_per_page, len(stations))
        page_stations = stations[start:end]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)
        axes = axes.flatten()

        for i, sta in enumerate(page_stations):
            ax = axes[i]
            g = comp[comp["Station"] == sta].copy()
            g["Month"] = pd.Categorical(g["Month"], categories=MONTHS_WY, ordered=True)
            g = g.sort_values("Month")

            x = np.arange(12)
            obs = g["Observed_CFS"].to_numpy(dtype=float)
            mod = g["Modeled_CFS"].to_numpy(dtype=float)

            ax.plot(x, obs, marker="o", linewidth=1.8, label="Observed", color="#1f77b4")
            ax.plot(x, mod, marker="s", linewidth=1.4, linestyle="--", label="Modeled", color="#ff7f0e")

            name = str(g["Station_Name"].iloc[0]).strip()
            if name:
                ax.set_title(f"{sta} | {name[:36]}", fontsize=9)
            else:
                ax.set_title(f"{sta}", fontsize=9)

            ax.grid(alpha=0.25)
            ax.set_xticks(np.arange(12))
            ax.set_xticklabels(MONTHS_WY, rotation=45, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)

        # Hide empty axes on last page
        for j in range(len(page_stations), n_rows * n_cols):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(
            f"Observed vs Modeled Streamflow by Station (WY2018, Oct-Sep) | Page {page + 1} of {n_pages}",
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_png = OUT_DIR / f"obs_vs_modeled_wy2018_page_{page + 1:02d}.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

    print("PLOTS_COMPLETE")
    print(f"Stations in StationList: {len(station_list)}")
    print(f"Stations with map+obs+model match: {len(stations)}")
    print(f"Stations plotted: {len(stations)}")
    print(f"Pages written: {n_pages}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Long table: {comp_csv}")
    print(f"Station summary: {summary_csv}")


if __name__ == "__main__":
    main()
