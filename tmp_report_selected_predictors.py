from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\mu3575\Documents\WAM")
SRC = ROOT / "afinch_matlab_source"

THS = "1201"
HSR_KEY = "HSR1200"
WY1 = 2018
NY = 1


def _load(module_name: str, file_name: str):
    path = SRC / file_name
    loader = SourceFileLoader(module_name, str(path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError(f"No module spec for {path}")
    mod = module_from_spec(spec)
    sys.modules[module_name] = mod
    loader.exec_module(mod)
    return mod


def _station_y_adj_inc(q_adj_inc_wy: np.ndarray, nhd_area_iwy: np.ndarray, days_in_mo: np.ndarray) -> np.ndarray:
    conv = np.asarray(days_in_mo[:12], dtype=float) * (24.0 * 3600.0 * 12.0 / (5280.0 ** 2))
    area = np.asarray(nhd_area_iwy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = q_adj_inc_wy[:, :12] / area[:, np.newaxis] * conv[np.newaxis, :]
    y[~np.isfinite(y)] = 0.0
    return y


def main() -> None:
    m_setup = _load("m_AFsetupData", "AFsetupData")
    m_nlcd = _load("m_AFReadNLCD", "AFReadNLCD")
    m_prec = _load("m_AFReadPrismPrec", "AFReadPrismPrec")
    m_gen = _load("m_AFGenStrucData", "AFGenStrucData")
    m_in = _load("m_AFReadInFlowWY", "AFReadInFlowWY")
    m_sta = _load("m_AFStaBasinGridComIDWY", "AFStaBasinGridComIDWY")
    m_plot = _load("m_AFPlotAreasFlows", "AFPlotAreasFlows")
    m_temp = _load("m_AFReadPrismTemp", "AFReadPrismTemp")
    m_regpoa = _load("m_AFRegressPOA", "AFRegressPOA")

    ctx = m_setup.setup_data(wy1=WY1, iy=0, ths=THS)
    wy = ctx.wy
    days_in_mo = np.asarray(ctx.days_in_mo, dtype=float)
    mo_name = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

    nlcd = m_nlcd.read_nlcd(ROOT, THS)
    prism = m_prec.read_prism_prec(
        base_dir=ROOT,
        ths=THS,
        wy=wy,
        comid_ths_flowline=nlcd.comid_ths,
        gridcode_ths_nlcd=nlcd.gridcode_ths,
    )

    p_cols = [f"PIn_{i:02d}" for i in range(1, 13)]
    p_in = prism.prism_ths[p_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p_in0 = p_in.copy()

    afstruct, poa, stations = m_gen.gen_struc_data(
        base_dir=ROOT,
        ths=THS,
        hsr=HSR_KEY,
        wy=wy,
        iy=0,
        ny=NY,
        afstruct=None,
        poa=None,
    )

    inflow = m_in.read_in_flow_wy(
        base_dir=ROOT,
        ths=THS,
        hsr=HSR_KEY,
        wy=wy,
        iy=1,
        sta_ths=stations,
        poa=poa,
        comid_ths=nlcd.comid_ths,
        n_reaches=len(nlcd.comid_ths),
    )

    q_cols = [f"Q{i:02d}" for i in range(1, 14)]
    q = inflow.station_flow_df[q_cols].to_numpy(dtype=float)
    sta_wy = inflow.station_flow_df["StaWY"].astype(str).tolist()
    nwis_area = inflow.station_flow_df["NWISArea"].to_numpy(dtype=float)
    comid_wu = inflow.comid_wu_df["ComID_WU"].to_numpy(dtype=np.int64)

    diag_dir = ROOT / HSR_KEY / "Output" / "Diagnostics"
    sta_res = m_sta.sta_basin_grid_comid_wy(
        afstruct=afstruct,
        hsr=HSR_KEY,
        ths=THS,
        wy=wy,
        iy=0,
        sta_wy=sta_wy,
        q=q,
        nwis_area=nwis_area,
        comid_wu=comid_wu,
        comid_ths=nlcd.comid_ths,
        reach_wu=inflow.reach_wu,
        output_dir=diag_dir,
        sta_hist=None,
        plot_matrix=False,
    )

    month_name = [
        "October", "November", "December", "January", "February", "March",
        "April", "May", "June", "July", "August", "September",
    ]

    plot_res = m_plot.plot_areas_flows(
        afstruct=sta_res.afstruct,
        sta_hist=sta_res.sta_hist,
        hsr=HSR_KEY,
        iy=0,
        wy=wy,
        net_design=sta_res.net_design,
        q_tot_wy=sta_res.sta_hist[0].q_tot_wy,
        nhd_area_iwy=sta_res.sta_hist[0].nhd_area_iwy,
        nwis_area_iwy=sta_res.sta_hist[0].nwis_area_iwy,
        month_names=month_name,
        make_plots=False,
    )

    temp_res = m_temp.read_prism_temp(
        base_dir=ROOT,
        ths=THS,
        hsr=HSR_KEY,
        wy=wy,
        wy_n=wy,
        iy=0,
        ny=NY,
        n_ths=len(nlcd.comid_ths),
        sta_ndx=sta_res.sta_ndx,
        grid_code_p_ths=prism.prism_ths["GridCode"].to_numpy(dtype=np.int64),
        comid_ths=nlcd.comid_ths,
        nlcd_ths=nlcd.nlcd_ths,
        p_in=p_in,
        afstruct=plot_res.afstruct,
        output_dir=diag_dir,
    )

    for sidx in sta_res.sta_ndx:
        rec = temp_res.afstruct[HSR_KEY][0][int(sidx)]
        rec["Precip"] = np.nan_to_num(np.asarray(rec["Precip"], dtype=float), nan=0.0)
        rec["Temp"] = np.nan_to_num(np.asarray(rec["Temp"], dtype=float), nan=0.0)

    sta_hist0 = {
        "StaList": sta_res.sta_list,
        "StaNdx": sta_res.sta_ndx,
        "NStaAct": sta_res.sta_hist[0].n_sta_act,
        "QTotWY": sta_res.sta_hist[0].q_tot_wy,
        "YAdjIncWY": _station_y_adj_inc(plot_res.q_adj_inc_wy, sta_res.sta_hist[0].nhd_area_iwy, days_in_mo),
    }
    sta_hist_list = [sta_hist0]

    nr = 6
    cb_matrix = np.zeros((nr, 24), dtype=int)
    cb_matrix[0, 21] = 1
    cb_matrix[1, 22] = 1
    cb_matrix[2, 23] = 1
    cb_matrix[3, 2] = 1
    cb_matrix[3, 3] = 1
    cb_matrix[3, 4] = 1
    cb_matrix[4, 8] = 1
    cb_matrix[4, 9] = 1
    cb_matrix[4, 10] = 1
    cb_matrix[5, 13] = 1
    cb_matrix[5, 14] = 1
    cb_matrix[5, 15] = 1
    reg_var_name = ["PPT", "TEMP", "PPT_LAG1", "NLCD_DEV", "NLCD_FOR", "NLCD_AG"]

    reg_poa = m_regpoa.regress_poa(
        afstruct=temp_res.afstruct,
        hsr=HSR_KEY,
        sta_hist=sta_hist_list,
        wy1=WY1,
        ny=NY,
        nr=nr,
        cb_matrix=cb_matrix,
        p_in0=p_in0,
        prsm_prec_ths=np.nan_to_num(prism.prism_ths[p_cols].to_numpy(dtype=float), nan=0.0).reshape(NY, -1, 12),
        reg_var_name=reg_var_name,
        mo_name=mo_name,
        prompt_pvalues=None,
        make_plot=False,
    )

    print("\nSelected predictors by month:")
    for i, m in enumerate(reg_poa.reg_month):
        sel = [reg_var_name[j] for j, on in enumerate(np.asarray(m.inmodel, dtype=bool)) if on]
        if len(sel) == 0:
            sel = ["<none>"]
        print(f"{mo_name[i]}: {', '.join(sel)}")


if __name__ == "__main__":
    main()
