"""Integration workflow for AFYieldAtGagesGUI -> AFImagePOAYield -> AFid.

This module wires the translated appendix components together in a runnable
Python workflow while the source files remain extensionless in
`afinch_matlab_source/`.
"""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

import pandas as pd


def _load_module_from_file(module_name: str, file_path: Path):
    loader = SourceFileLoader(module_name, str(file_path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError(f"Could not load module spec for {file_path}")
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _read_trend_duration_matrix(base_dir: Path, hsr: str, wy1: int, ny: int) -> list[pd.DataFrame]:
    out = []
    for iy in range(ny):
        wy = wy1 + iy
        path = base_dir / hsr / "Output" / "FlowAccum" / f"ComIDQ12WY{wy}.csv"
        out.append(pd.read_csv(path))
    return out


def run_yield_at_gages_workflow(
    base_dir: Path | str,
    ths: str,
    hsr: str,
    month_name: Sequence[str],
    mo_name: Sequence[str],
    wy1: int,
    wyn: int,
    ny: int,
    sta_hist: Sequence[Dict[str, Any]],
    datatip_provider: Callable[[str, Dict[str, Any] | None], Iterable[Any]] | None = None,
) -> None:
    """Launch integrated GUI workflow for POA yield image and datatip identification.

    Parameters
    ----------
    base_dir, ths, hsr, month_name, mo_name, wy1, wyn, ny, sta_hist:
        Core inputs already used by the translated appendix sections.
    datatip_provider:
        Callback used at Submit time to resolve the user-entered datatip name into
        iterable datatip records. Signature: `(datatip_name, month_result) -> iterable`.
    """
    base = Path(base_dir)
    src_dir = base / "afinch_matlab_source"

    gui_mod = _load_module_from_file("AFYieldAtGagesGUI_mod", src_dir / "AFYieldAtGagesGUI")
    image_mod = _load_module_from_file("AFImagePOAYield_mod", src_dir / "AFImagePOAYield")
    afid_mod = _load_module_from_file("AFid_mod", src_dir / "AFid")

    tdm = _read_trend_duration_matrix(base, hsr, wy1, ny)

    state: Dict[str, Any] = {"last": None}

    def on_month_selected(target_month: int):
        result = image_mod.image_poa_yield(
            base_dir=base,
            ths=ths,
            hsr=hsr,
            trend_duration_matrix=tdm,
            target_month=target_month,
            month_name=month_name,
            wy1=wy1,
            wyn=wyn,
            ny=ny,
        )
        state["last"] = result
        return result

    def on_submit_dtip(datatip_name: str, month_result: Dict[str, Any] | None = None):
        if month_result is None:
            month_result = state.get("last")
        if month_result is None:
            print("No month image has been generated yet. Select a month first.")
            return
        if datatip_provider is None:
            print("No datatip provider supplied; submit action skipped.")
            return

        datatips = datatip_provider(datatip_name, month_result)
        afid_mod.af_id(
            datatip=datatips,
            station_ths=month_result["station_ths_plot"],
            wy1=wy1,
            y=month_result["y"],
            q=month_result["q"],
            target_month=int(month_result["target_month"]),
            mo_name=mo_name,
            da_sq_mi_ths=month_result["da_sq_mi_plot"],
            sta_hist=sta_hist,
        )

    gui_mod.launch_yield_at_gages_gui(
        on_month_selected=on_month_selected,
        on_submit_datatip=on_submit_dtip,
        initial_datatip="cursor_info",
    )
