from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys

import numpy as np

ROOT = Path(r"c:\Users\mu3575\Documents\WAM")
SRC = ROOT / "afinch_matlab_source"
BASE_DIR = ROOT

THS = "1201"
HSR_KEY = "HSR1200"
WY = 2018
IY = 1
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


def run() -> None:
    m_nlcd = _load("m_AFReadNLCD", "AFReadNLCD")
    m_prec = _load("m_AFReadPrismPrec", "AFReadPrismPrec")
    m_gen = _load("m_AFGenStrucData", "AFGenStrucData")
    m_in = _load("m_AFReadInFlowWY", "AFReadInFlowWY")
    m_sta = _load("m_AFStaBasinGridComIDWY", "AFStaBasinGridComIDWY")

    nlcd = m_nlcd.read_nlcd(BASE_DIR, THS)
    print(f"NLCD: comids={len(nlcd.comid_ths)}")

    prism = m_prec.read_prism_prec(
        base_dir=BASE_DIR,
        ths=THS,
        wy=WY,
        comid_ths_flowline=nlcd.comid_ths,
        gridcode_ths_nlcd=nlcd.gridcode_ths,
    )
    print(f"PRISM: rows={len(prism.prism_ths)} unmatched_gridcodes={len(prism.unmatched_gridcodes)}")

    afstruct, poa, stations = m_gen.gen_struc_data(
        base_dir=BASE_DIR,
        ths=THS,
        hsr=HSR_KEY,
        wy=WY,
        iy=IY,
        ny=NY,
        afstruct=None,
        poa=None,
    )
    print(f"AFSTRUCT: stations={len(stations)}")

    inflow = m_in.read_in_flow_wy(
        base_dir=BASE_DIR,
        ths=THS,
        hsr=HSR_KEY,
        wy=WY,
        iy=IY,
        sta_ths=stations,
        poa=poa,
        comid_ths=nlcd.comid_ths,
        n_reaches=len(nlcd.comid_ths),
    )
    print(f"INFLOW: active_stations={len(inflow.station_flow_ths_df)}")

    q_cols = [f"Q{i:02d}" for i in range(1, 14)]
    q = inflow.station_flow_df[q_cols].to_numpy(dtype=float)
    sta_wy = inflow.station_flow_df["StaWY"].astype(str).tolist()
    nwis_area = inflow.station_flow_df["NWISArea"].to_numpy(dtype=float)
    comid_wu = inflow.comid_wu_df["ComID_WU"].to_numpy(dtype=np.int64)

    out_dir = BASE_DIR / HSR_KEY / "Output" / "Diagnostics"
    sta_res = m_sta.sta_basin_grid_comid_wy(
        afstruct=afstruct,
        hsr=HSR_KEY,
        ths=THS,
        wy=WY,
        iy=IY,
        sta_wy=sta_wy,
        q=q,
        nwis_area=nwis_area,
        comid_wu=comid_wu,
        comid_ths=nlcd.comid_ths,
        reach_wu=inflow.reach_wu,
        output_dir=out_dir,
        sta_hist=None,
        plot_matrix=False,
    )

    print("CONVERTED_AFINCH_CORE_RUN_COMPLETE")
    print(f"net_design_shape={sta_res.net_design.shape}")
    print(f"diag_file={(out_dir / f'StaBasinGridComIDWY{WY}.dat')}")


if __name__ == "__main__":
    run()
