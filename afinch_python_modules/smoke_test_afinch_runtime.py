from __future__ import annotations

import tempfile
from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(r"c:\Users\mu3575\Documents\WAM")
SRC = ROOT / "afinch_matlab_source"


def load_module(module_name: str, file_name: str):
    path = SRC / file_name
    loader = SourceFileLoader(module_name, str(path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError(f"No module spec for {path}")
    mod = module_from_spec(spec)
    sys.modules[module_name] = mod
    loader.exec_module(mod)
    return mod


def run():
    failures = []

    # Import smoke for all translated modules in afinch_matlab_source.
    files = [
        "AFBoxplotExplanVar",
        "AFCallRegCheckBox",
        "AFConFlowAccum",
        "AFGenLag1Precp",
        "AFGenStrucData",
        "AFid",
        "AFImagePOAYield",
        "AFPlotAreasFlows",
        "AFPlotQmMeaEst",
        "AFPlotRegressCoeff",
        "AFQConAdjInc",
        "AFQEstAdjInc",
        "AFReadInFlowWY",
        "AFReadNLCD",
        "AFReadPrismPrec",
        "AFReadPrismTemp",
        "AFRegCheckBoxGUI",
        "AFRegressByWY",
        "AFRegressPOA",
        "AFsetupData",
        "AFStaBasinGridComIDWY",
        "AFTrendDurations",
        "AFWrtQYEstCon",
        "AFYieldAtGagesGUI",
        "AFYieldImage",
        "FKenSen",
        "gui_afinch",
        "initialize_var_AFlniAFStrct",
        "starting_afinch",
    ]

    mods = {}
    for f in files:
        try:
            mods[f] = load_module(f"m_{f}", f)
        except Exception as e:
            failures.append((f"import:{f}", str(e)))

    # Functional smoke tests for key non-GUI computational paths.
    try:
        m = mods["AFGenLag1Precp"]
        arr = np.arange(2 * 3 * 12, dtype=float).reshape(2, 3, 12)
        p0 = np.ones((3, 12), dtype=float)
        out = m.gen_lag1_prec(arr, p0)
        assert out.shape == (2, 3, 12)
    except Exception as e:
        failures.append(("call:AFGenLag1Precp.gen_lag1_prec", str(e)))

    try:
        m = mods["AFQEstAdjInc"]
        ny, nths, nr = 2, 3, 2
        nlcd = np.ones((nths, 21), dtype=float)
        pp = np.ones((ny, nths, 12), dtype=float)
        pt = np.ones((ny, nths, 12), dtype=float) * 2
        pm = np.ones((ny, nths, 12), dtype=float) * 3
        cb = np.zeros((nr, 24), dtype=int)
        cb[:, 0] = 1
        reg_month = [SimpleNamespace(inmodel=np.array([True, True])) for _ in range(12)]
        reg_hist = [[SimpleNamespace(RobustB=np.array([0.5, 0.1, 0.2])) for _ in range(12)] for _ in range(ny)]
        area = np.ones(nths, dtype=float)
        dim = np.ones(12, dtype=float) * 30
        yest, qest = m.q_est_adj_inc(nlcd, pp, pt, pm, cb, reg_month, reg_hist, area, dim)
        assert yest.shape == (ny, nths, 12)
        assert qest.shape == (ny, nths, 12)
    except Exception as e:
        failures.append(("call:AFQEstAdjInc.q_est_adj_inc", str(e)))

    try:
        m = mods["AFQConAdjInc"]
        afstruct = {"HSR0100": [[{"SBGridCode": np.array([11, 12]), "QMeaAdjInc": np.ones(12) * 5.0}]]}
        sta_hist = [{"StaNdx": np.array([0])}]
        grid = np.array([11, 12, 13])
        q_est = np.ones((1, 3, 12), dtype=float)
        area = np.array([1.0, 2.0, 1.5])
        dim = np.ones(12) * 30
        _, con, qcon, ycon = m.q_con_adj_inc(afstruct, "HSR0100", sta_hist, grid, q_est, area, dim)
        assert con.shape[0] == 1
        assert qcon.shape == (1, 3, 12)
        assert ycon.shape == (1, 3, 12)
    except Exception as e:
        failures.append(("call:AFQConAdjInc.q_con_adj_inc", str(e)))

    try:
        m = mods["AFid"]
        datatip = [{"Position": [1, 1, 0.5]}]
        q = np.ones((1, 1, 12), dtype=float) * 4
        y = np.ones((1, 1, 12), dtype=float) * 2
        recs = m.af_id(
            datatip=datatip,
            station_ths=["01234567"],
            wy1=2000,
            y=y,
            q=q,
            target_month=1,
            mo_name=["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
            da_sq_mi_ths=np.array([2.0]),
            sta_hist=[{"StaList": ["01234567"]}],
        )
        assert len(recs) == 1
    except Exception as e:
        failures.append(("call:AFid.af_id", str(e)))

    try:
        m = mods["AFsetupData"]
        ctx = m.setup_data(wy1=2000, iy=0, ths="1201")
        assert ctx.wy == 2000
        assert len(ctx.days_in_mo) == 13
    except Exception as e:
        failures.append(("call:AFsetupData.setup_data", str(e)))

    try:
        m = mods["AFWrtQYEstCon"]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hsr = "HSR1200"
            (td / hsr / "Output" / "FlowYield").mkdir(parents=True, exist_ok=True)
            ny, nths = 1, 2
            arr = np.ones((ny, nths, 12), dtype=float)
            out = m.write_qy_est_con(
                base_dir=td,
                hsr=hsr,
                ths="1201",
                iy=0,
                wy1=2000,
                ny=1,
                grid_code_ths=np.array([1, 2]),
                comid_ths=np.array([100, 200]),
                gc_area_sq_mi=np.array([1.0, 2.0]),
                q_est_adj_inc=arr,
                y_est_adj_inc=arr,
                q_con_adj_inc=arr,
                y_con_adj_inc=arr,
                sta_ths=["001", "002"],
                poa=np.ones((2, 1), dtype=int),
            )
            assert out.exists()
    except Exception as e:
        failures.append(("call:AFWrtQYEstCon.write_qy_est_con", str(e)))

    print("SMOKE_TEST_FAILURES", len(failures))
    for k, v in failures:
        print(f"{k}: {v}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
