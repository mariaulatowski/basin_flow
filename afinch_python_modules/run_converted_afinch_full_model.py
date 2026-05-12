from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import geopandas as gpd


os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(r"c:\Users\mu3575\Documents\WAM")
SRC = ROOT / "afinch_matlab_source"

THS = "1206"
HSR_KEY = "HSR1206"
WY1 = 2018
NY = 1
# Multi-year regression parameters: use 2010-2024 for predictor building
WY1_REG = 2010
NY_REG = 15
P_ENTER = 0.05
P_REMOVE = 0.05


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


def _ensure_vaa_for_accumulation(base_dir: Path, hsr: str) -> None:
    vaa_path = base_dir / hsr / "GIS" / "NHDFlowlineVAA.txt"
    if not vaa_path.exists():
        raise FileNotFoundError(f"Missing VAA file: {vaa_path}")

    vaa = pd.read_csv(vaa_path)
    if vaa.shape[1] >= 6:
        return

    expected = {"ComID", "HydroSeq", "DnHydroSeq"}
    if not expected.issubset(set(vaa.columns)):
        raise ValueError(
            f"VAA file must contain at least columns {sorted(expected)} when only 3 columns are present. "
            f"Found columns: {list(vaa.columns)}"
        )

    work = vaa[["ComID", "HydroSeq", "DnHydroSeq"]].copy()
    work["ComID"] = pd.to_numeric(work["ComID"], errors="coerce").astype("Int64")
    work["HydroSeq"] = pd.to_numeric(work["HydroSeq"], errors="coerce").astype("Int64")
    work["DnHydroSeq"] = pd.to_numeric(work["DnHydroSeq"], errors="coerce").fillna(0).astype("Int64")
    work = work.dropna(subset=["ComID", "HydroSeq"]).astype(
        {"ComID": "int64", "HydroSeq": "int64", "DnHydroSeq": "int64"}
    )

    work["FromNode"] = work["HydroSeq"]
    work["ToNode"] = work["DnHydroSeq"]
    work["Divergence"] = 0
    upstream_targets = set(work["ToNode"].tolist())
    work["StartFlag"] = (~work["FromNode"].isin(upstream_targets)).astype(int)

    out = work[["ComID", "FromNode", "ToNode", "HydroSeq", "Divergence", "StartFlag"]]
    out.to_csv(vaa_path, index=False)


def _station_y_adj_inc(q_adj_inc_wy: np.ndarray, nhd_area_iwy: np.ndarray, days_in_mo: np.ndarray) -> np.ndarray:
    conv = np.asarray(days_in_mo[:12], dtype=float) * (24.0 * 3600.0 * 12.0 / (5280.0 ** 2))
    area = np.asarray(nhd_area_iwy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = q_adj_inc_wy[:, :12] / area[:, np.newaxis] * conv[np.newaxis, :]
    y[~np.isfinite(y)] = 0.0
    return y


def _normalize_sta_ndx_for_year(sta_ndx: np.ndarray, year_block_len: int) -> np.ndarray:
    """Normalize station indices to zero-based for one year block.

    Some source paths provide 1-based station indices. Detect that pattern
    and convert to 0-based so downstream afstruct indexing is valid.
    """
    idx = np.asarray(sta_ndx, dtype=int)
    if idx.size == 0:
        return idx

    if np.any(idx >= year_block_len):
        one_based_candidate = bool(np.all(idx >= 1) and np.max(idx) <= year_block_len)
        if one_based_candidate:
            return idx - 1
    return idx


def _align_sta_ndx_to_year_block(sta_ndx: np.ndarray, year_block: Any) -> np.ndarray:
    """Align station indices to the current year's afstruct container.

    Handles both list-like (0-based) and dict-like year blocks where keys may be
    0-based or 1-based integers.
    """
    idx = np.asarray(sta_ndx, dtype=int)
    if idx.size == 0:
        return idx

    if isinstance(year_block, dict):
        key_set: set[int] = set()
        for k in year_block.keys():
            if isinstance(k, (int, np.integer)):
                key_set.add(int(k))
            elif isinstance(k, str) and k.strip().isdigit():
                key_set.add(int(k.strip()))

        if key_set:
            if np.all([int(v) in key_set for v in idx]):
                return idx
            if np.all([int(v + 1) in key_set for v in idx]):
                return idx + 1
            if np.all([int(v - 1) in key_set for v in idx]):
                return idx - 1
        return idx

    return _normalize_sta_ndx_for_year(idx, len(year_block))


def _year_block_has_index(year_block: Any, sidx: int) -> bool:
    if isinstance(year_block, dict):
        if sidx in year_block:
            return True
        return str(sidx) in year_block
    return 0 <= sidx < len(year_block)


def _pick_prism_source_dir(base_dir: Path, element: str) -> str:
    """Pick best available PRISM raster folder for ppt/tmean.

    Preference order:
    1) inputData/prism_monthly/<element>/clipped
    2) inputData/prism_monthly/<element>/extracted
    """
    candidates = [
        base_dir / "inputData" / "prism_monthly" / element / "clipped",
        base_dir / "inputData" / "prism_monthly" / element / "extracted",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.tif")):
            return str(c)
    raise FileNotFoundError(
        f"No PRISM source rasters found for '{element}'. Checked: {candidates}"
    )


@dataclass
class ConvertedAFinchContext:
    """Mutable state carried across the converted AFINCH step workflow."""

    modules: dict[str, Any] = field(default_factory=dict)
    wy: int | None = None
    days_in_mo: np.ndarray | None = None
    month_name: list[str] = field(default_factory=list)
    mo_name: list[str] = field(default_factory=list)
    nlcd: Any = None
    prism: Any = None
    p_in0: np.ndarray | None = None
    afstruct: Any = None
    poa: Any = None
    stations: Any = None
    inflow: Any = None
    sta_res: Any = None
    plot_res: Any = None
    temp_res: Any = None
    sta_hist_list: list[dict[str, Any]] = field(default_factory=list)
    cb_matrix: np.ndarray | None = None
    reg_var_name: list[str] = field(default_factory=list)
    reg_poa: Any = None
    reg_hist: Any = None
    prsm_prec_ths: np.ndarray | None = None
    prsm_temp_ths: np.ndarray | None = None
    prsm_prem_ths: np.ndarray | None = None
    gc_area_sq_mi: np.ndarray | None = None
    y_est_adj_inc: np.ndarray | None = None
    q_est_adj_inc: np.ndarray | None = None
    afstruct_con: Any = None
    con_adjust: np.ndarray | None = None
    q_con_adj_inc: np.ndarray | None = None
    y_con_adj_inc: np.ndarray | None = None
    qy_path: Path | None = None
    flow_comid: np.ndarray | None = None
    flow_accum: np.ndarray | None = None
    flow_accum_path: Path | None = None


class ConvertedAFinchPipeline:
    """Stepwise runner mirroring the original MATLAB phase flow."""

    def __init__(
        self,
        base_dir: Path,
        src_dir: Path,
        ths: str,
        hsr_key: str,
        wy1: int,
        ny: int,
        logger: Callable[[str], None] | None = None,
        wy1_reg: int | None = None,
        ny_reg: int | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.src_dir = Path(src_dir)
        self.ths = str(ths)
        self.hsr_key = str(hsr_key)
        self.wy1 = int(wy1)
        self.ny = int(ny)
        # Multi-year regression: use separate years if specified
        self.wy1_reg = int(wy1_reg) if wy1_reg is not None else int(wy1)
        self.ny_reg = int(ny_reg) if ny_reg is not None else int(ny)
        self.log = logger or (lambda msg: None)
        self.ctx = ConvertedAFinchContext()
        # Store for multi-year regression loading
        self._prism_reg: Any = None
        self._prsm_prec_reg: np.ndarray | None = None
        self._p_in0_reg: np.ndarray | None = None
        self._sta_hist_list_reg: list[dict[str, Any]] = []
        self._temp_res_reg: Any = None

    def _ensure_regression_prism_files(self) -> None:
        """Ensure HSR PRISM yearly files exist for all regression years.

        Creates missing files PrismPrecipWY####.dat and PrismTempAveWY####.dat
        using raw PRISM monthly rasters from inputData/prism_monthly.
        """
        p_dir = self.base_dir / self.hsr_key / "PRISM" / "Precipitation"
        t_dir = self.base_dir / self.hsr_key / "PRISM" / "Temperature"
        p_dir.mkdir(parents=True, exist_ok=True)
        t_dir.mkdir(parents=True, exist_ok=True)

        # Geometry with ComID is produced by the network build step. New builds write
        # polygon NHDPlusCatchment geometry; older builds wrote flowline geometry.
        catchment_gpkg = self.base_dir / "inputData" / f"NHDPlusCatchment_{self.ths}.gpkg"
        if not catchment_gpkg.exists():
            raise FileNotFoundError(
                f"Missing catchment geometry required for PRISM build: {catchment_gpkg}. "
                "Run Build Network first."
            )

        gdf = gpd.read_file(catchment_gpkg)
        comid_col = None
        for c in ["NHDPlusID", "ComID", "COMID", "GridCode"]:
            if c in gdf.columns:
                comid_col = c
                break
        if comid_col is None:
            raise KeyError(
                f"No ComID-like field in {catchment_gpkg}. Columns: {list(gdf.columns)}"
            )
        gdf["ComID"] = pd.to_numeric(gdf[comid_col], errors="coerce")
        gdf = gdf.dropna(subset=["ComID", "geometry"]).copy()
        gdf["ComID"] = gdf["ComID"].astype("int64")
        prism_geom_cols = ["ComID", "geometry"]
        if "AreaSqKm" in gdf.columns:
            prism_geom_cols.insert(1, "AreaSqKm")
        prism_geom = gdf[prism_geom_cols].copy()
        geom_types = set(prism_geom.geometry.geom_type.dropna().unique())
        use_catchment_zonal = bool(geom_types & {"Polygon", "MultiPolygon"})

        def _has_catchment_zonal_header(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                return "catchment-zonal" in path.read_text(encoding="utf-8", errors="ignore")[:512].lower()
            except OSError:
                return False

        years_to_build: list[int] = []
        stale_years: list[int] = []
        for wy in range(self.wy1_reg, self.wy1_reg + self.ny_reg):
            p_path = p_dir / f"PrismPrecipWY{wy}.dat"
            t_path = t_dir / f"PrismTempAveWY{wy}.dat"
            missing = not p_path.exists() or not t_path.exists()
            stale = (
                use_catchment_zonal
                and not missing
                and (not _has_catchment_zonal_header(p_path) or not _has_catchment_zonal_header(t_path))
            )
            if missing or stale:
                years_to_build.append(wy)
            if stale:
                stale_years.append(wy)

        if not years_to_build:
            return

        kind = "catchment-zonal" if use_catchment_zonal else "raster-sampled"
        self.log(
            f"[Regression] Building {kind} HSR PRISM files for WY {years_to_build[0]}-{years_to_build[-1]} "
            "from inputData/prism_monthly rasters...\n"
        )
        if stale_years:
            self.log(
                f"[Regression] Rebuilding stale point-sampled PRISM files as catchment-zonal for "
                f"WY {stale_years[0]}-{stale_years[-1]}.\n"
            )

        # Reuse PRISM sampling logic from basin builder.
        from afinch_python_modules.build_brazos_basin_network import _build_catchment_prism, _build_real_prism

        prism_ppt_src = _pick_prism_source_dir(self.base_dir, "ppt")
        prism_tmean_src = _pick_prism_source_dir(self.base_dir, "tmean")

        for wy in years_to_build:
            p_path = p_dir / f"PrismPrecipWY{wy}.dat"
            t_path = t_dir / f"PrismTempAveWY{wy}.dat"

            self.log(f"[Regression] Building HSR PRISM yearly files for WY{wy}...\n")
            if use_catchment_zonal:
                precip_df, temp_df = _build_catchment_prism(
                    self.base_dir,
                    prism_geom,
                    wy,
                    prism_ppt_src,
                    prism_tmean_src,
                )
            else:
                precip_df, temp_df = _build_real_prism(
                    self.base_dir,
                    prism_geom[["ComID", "geometry"]].copy(),
                    wy,
                    prism_ppt_src,
                    prism_tmean_src,
                )

            with p_path.open("w", encoding="utf-8") as f:
                f.write("PRISM precipitation\n")
                if use_catchment_zonal:
                    f.write("Real basin catchment-zonal dataset\n")
                else:
                    f.write("Real basin raster-sampled dataset\n")
                f.write("GridCode GCAreaSqMi PIn_01..PIn_13\n")
                f.write("Units: source PRISM raster units\n")
                precip_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

            with t_path.open("w", encoding="utf-8") as f:
                f.write("PRISM temperature\n")
                if use_catchment_zonal:
                    f.write("Real basin catchment-zonal dataset\n")
                else:
                    f.write("Real basin raster-sampled dataset\n")
                f.write("GridCode TdC_01..TdC_12\n")
                f.write("Units: source PRISM raster units\n")
                temp_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

        self.log("[Regression] HSR PRISM yearly files generated.\n")

    def _load(self, module_name: str, file_name: str):
        path = self.src_dir / file_name
        loader = SourceFileLoader(module_name, str(path))
        spec = spec_from_loader(module_name, loader)
        if spec is None:
            raise RuntimeError(f"No module spec for {path}")
        mod = module_from_spec(spec)
        sys.modules[module_name] = mod
        loader.exec_module(mod)
        return mod

    def _load_modules(self) -> None:
        if self.ctx.modules:
            return
        self.ctx.modules = {
            "setup": self._load("m_AFsetupData", "AFsetupData"),
            "nlcd": self._load("m_AFReadNLCD", "AFReadNLCD"),
            "prec": self._load("m_AFReadPrismPrec", "AFReadPrismPrec"),
            "gen": self._load("m_AFGenStrucData", "AFGenStrucData"),
            "in": self._load("m_AFReadInFlowWY", "AFReadInFlowWY"),
            "sta": self._load("m_AFStaBasinGridComIDWY", "AFStaBasinGridComIDWY"),
            "plot": self._load("m_AFPlotAreasFlows", "AFPlotAreasFlows"),
            "temp": self._load("m_AFReadPrismTemp", "AFReadPrismTemp"),
            "regpoa": self._load("m_AFRegressPOA", "AFRegressPOA"),
            "regwy": self._load("m_AFRegressByWY", "AFRegressByWY"),
            "plotcoeff": self._load("m_AFPlotRegressCoeff", "AFPlotRegressCoeff"),
            "lag": self._load("m_AFGenLag1Precp", "AFGenLag1Precp"),
            "qest": self._load("m_AFQEstAdjInc", "AFQEstAdjInc"),
            "qcon": self._load("m_AFQConAdjInc", "AFQConAdjInc"),
            "wrt": self._load("m_AFWrtQYEstCon", "AFWrtQYEstCon"),
            "acc": self._load("m_AFConFlowAccum", "AFConFlowAccum"),
        }

    def step_setup_inputs(self) -> ConvertedAFinchContext:
        self._load_modules()
        m = self.ctx.modules

        self.log("[STEP 1/6] Setup, NLCD/PRISM, stations, and diagnostics...\n")
        setup_ctx = m["setup"].setup_data(wy1=self.wy1, iy=0, ths=self.ths)
        self.ctx.wy = setup_ctx.wy
        self.ctx.days_in_mo = np.asarray(setup_ctx.days_in_mo, dtype=float)
        self.ctx.month_name = [
            "October", "November", "December", "January", "February", "March",
            "April", "May", "June", "July", "August", "September",
        ]
        self.ctx.mo_name = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

        self.ctx.nlcd = m["nlcd"].read_nlcd(self.base_dir, self.ths, hsr=self.hsr_key)
        self.log(f"NLCD loaded: comids={len(self.ctx.nlcd.comid_ths)}\n")

        self.ctx.prism = m["prec"].read_prism_prec(
            base_dir=self.base_dir,
            ths=self.ths,
            hsr=self.hsr_key,
            wy=self.ctx.wy,
            comid_ths_flowline=self.ctx.nlcd.comid_ths,
            gridcode_ths_nlcd=self.ctx.nlcd.gridcode_ths,
        )
        self.log(
            f"PRISM loaded: rows={len(self.ctx.prism.prism_ths)} "
            f"unmatched_gridcodes={len(self.ctx.prism.unmatched_gridcodes)}\n"
        )

        p_cols = [f"PIn_{i:02d}" for i in range(1, 13)]
        p_in = self.ctx.prism.prism_ths[p_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.ctx.p_in0 = p_in.copy()

        self.ctx.afstruct, self.ctx.poa, self.ctx.stations = m["gen"].gen_struc_data(
            base_dir=self.base_dir,
            ths=self.ths,
            hsr=self.hsr_key,
            wy=self.ctx.wy,
            iy=0,
            ny=self.ny,
            afstruct=None,
            poa=None,
        )
        self.log(f"Network structure built: stations={len(self.ctx.stations)}\n")
        contributor_counts = [
            len(np.asarray(rec.get("ComID", [])))
            for rec in self.ctx.afstruct.get(self.hsr_key, {}).get(0, {}).values()
        ]
        if contributor_counts:
            counts_s = pd.Series(contributor_counts)
            one_reach = int((counts_s <= 1).sum())
            self.log(
                "Loaded gaged contributors: "
                f"min={int(counts_s.min()):,}, median={int(counts_s.median()):,}, "
                f"max={int(counts_s.max()):,}, one-reach={one_reach:,}\n"
            )
            if one_reach == len(contributor_counts):
                raise RuntimeError(
                    "Step 1 loaded one-reach gaged catchments for every station. "
                    "Run Build Network again and confirm the upstream gaged catchment verification passes."
                )

        self.ctx.inflow = m["in"].read_in_flow_wy(
            base_dir=self.base_dir,
            ths=self.ths,
            hsr=self.hsr_key,
            wy=self.ctx.wy,
            iy=1,
            sta_ths=self.ctx.stations,
            poa=self.ctx.poa,
            comid_ths=self.ctx.nlcd.comid_ths,
            n_reaches=len(self.ctx.nlcd.comid_ths),
        )

        q_cols = [f"Q{i:02d}" for i in range(1, 14)]
        q = self.ctx.inflow.station_flow_df[q_cols].to_numpy(dtype=float)
        sta_wy = self.ctx.inflow.station_flow_df["StaWY"].astype(str).tolist()
        nwis_area = self.ctx.inflow.station_flow_df["NWISArea"].to_numpy(dtype=float)
        comid_wu = self.ctx.inflow.comid_wu_df["ComID_WU"].to_numpy(dtype=np.int64)
        diag_dir = self.base_dir / self.hsr_key / "Output" / "Diagnostics"

        self.ctx.sta_res = m["sta"].sta_basin_grid_comid_wy(
            afstruct=self.ctx.afstruct,
            hsr=self.hsr_key,
            ths=self.ths,
            wy=self.ctx.wy,
            iy=0,
            sta_wy=sta_wy,
            q=q,
            nwis_area=nwis_area,
            comid_wu=comid_wu,
            comid_ths=self.ctx.nlcd.comid_ths,
            reach_wu=self.ctx.inflow.reach_wu,
            output_dir=diag_dir,
            sta_hist=None,
            plot_matrix=False,
        )

        self.ctx.plot_res = m["plot"].plot_areas_flows(
            afstruct=self.ctx.sta_res.afstruct,
            sta_hist=self.ctx.sta_res.sta_hist,
            hsr=self.hsr_key,
            iy=0,
            wy=self.ctx.wy,
            net_design=self.ctx.sta_res.net_design,
            q_tot_wy=self.ctx.sta_res.sta_hist[0].q_tot_wy,
            nhd_area_iwy=self.ctx.sta_res.sta_hist[0].nhd_area_iwy,
            nwis_area_iwy=self.ctx.sta_res.sta_hist[0].nwis_area_iwy,
            month_names=self.ctx.month_name,
            make_plots=False,
        )

        self.ctx.temp_res = m["temp"].read_prism_temp(
            base_dir=self.base_dir,
            ths=self.ths,
            hsr=self.hsr_key,
            wy=self.ctx.wy,
            wy_n=self.ctx.wy,
            iy=0,
            ny=self.ny,
            n_ths=len(self.ctx.nlcd.comid_ths),
            sta_ndx=self.ctx.sta_res.sta_ndx,
            grid_code_p_ths=self.ctx.prism.prism_ths["GridCode"].to_numpy(dtype=np.int64),
            comid_ths=self.ctx.nlcd.comid_ths,
            nlcd_ths=self.ctx.nlcd.nlcd_ths,
            p_in=p_in,
            afstruct=self.ctx.plot_res.afstruct,
            output_dir=diag_dir,
        )

        grid_to_nlcd: dict[int, np.ndarray] = {
            int(g): np.nan_to_num(np.asarray(v, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            for g, v in zip(self.ctx.nlcd.gridcode_ths, self.ctx.nlcd.nlcd_ths)
        }

        for sidx in self.ctx.sta_res.sta_ndx:
            rec = self.ctx.temp_res.afstruct[self.hsr_key][0][int(sidx)]
            rec["Precip"] = np.nan_to_num(np.asarray(rec.get("Precip", np.zeros(12)), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            rec["Temp"] = np.nan_to_num(np.asarray(rec.get("Temp", np.zeros(12)), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

            nlcd_vec = np.asarray(rec.get("NLCD", np.zeros(21)), dtype=float)
            if not np.isfinite(nlcd_vec).all():
                sb_codes = np.asarray(rec.get("SBGridCode", []), dtype=float)
                sb_codes = sb_codes[np.isfinite(sb_codes)].astype(int)
                candidates = [grid_to_nlcd[int(code)] for code in sb_codes if int(code) in grid_to_nlcd]
                if candidates:
                    nlcd_vec = np.mean(np.vstack(candidates), axis=0)
                else:
                    nlcd_vec = np.zeros(21, dtype=float)
            rec["NLCD"] = np.nan_to_num(nlcd_vec, nan=0.0, posinf=0.0, neginf=0.0)

        self.ctx.sta_hist_list = [{
            "StaList": self.ctx.sta_res.sta_list,
            "StaNdx": self.ctx.sta_res.sta_ndx,
            "NStaAct": self.ctx.sta_res.sta_hist[0].n_sta_act,
            "QTotWY": self.ctx.sta_res.sta_hist[0].q_tot_wy,
            "NHD_Area_IWY": np.asarray(self.ctx.sta_res.sta_hist[0].nhd_area_iwy, dtype=float),
            "YAdjIncWY": _station_y_adj_inc(
                self.ctx.plot_res.q_adj_inc_wy,
                self.ctx.sta_res.sta_hist[0].nhd_area_iwy,
                self.ctx.days_in_mo,
            ),
        }]

        self.log("Step 1 complete.\n")
        return self.ctx

    def _load_multiyear_regression_data(self) -> None:
        """Load PRISM, temperature, and station history for multi-year regression."""
        if self.wy1_reg == self.wy1 and self.ny_reg == self.ny:
            # Use same data as already loaded in Step 1
            self._prism_reg = self.ctx.prism
            self._p_in0_reg = self.ctx.p_in0
            self._sta_hist_list_reg = self.ctx.sta_hist_list
            self._temp_res_reg = self.ctx.temp_res
            return

        # Auto-detect available streamflow files and adjust regression year range
        requested_start = self.wy1_reg
        requested_end = self.wy1_reg + self.ny_reg - 1
        hsr_streamflow_dir = self.base_dir / self.hsr_key / "Streamflow"
        
        available_years = []
        if hsr_streamflow_dir.exists():
            for f in hsr_streamflow_dir.glob("ComIDStationDAMoAnQ*.dat"):
                try:
                    wy = int(f.stem.replace("ComIDStationDAMoAnQ", ""))
                    available_years.append(wy)
                except ValueError:
                    pass
            available_years.sort()
        
        # Filter to requested range
        available_in_range = [wy for wy in available_years if requested_start <= wy <= requested_end]
        
        if not available_in_range:
            raise FileNotFoundError(
                f"No streamflow files found in {hsr_streamflow_dir} for requested range WY{requested_start}-{requested_end}. "
                f"Available years: {available_years if available_years else 'none'}. "
                f"Run Build Network to generate missing years."
            )
        
        # Use available years instead of all requested years
        if available_in_range != list(range(requested_start, requested_end + 1)):
            self.log(f"[Regression] Requested WY{requested_start}-{requested_end}, but only found: {available_in_range}\n")
            self.log(f"[Regression] Using these years for regression.\n")
            self.wy1_reg = available_in_range[0]
            self.ny_reg = len(available_in_range)
        
        # Ensure yearly HSR PRISM dat files exist for all regression years.
        self._ensure_regression_prism_files()

        years_for_regression = list(range(self.wy1_reg, self.wy1_reg + self.ny_reg))
        self.log(f"[Regression] Loading multi-year data for years: {years_for_regression}\n")
        m = self.ctx.modules
        p_cols = [f"PIn_{i:02d}" for i in range(1, 13)]

        # Load PRISM for all regression years
        prism_list = []
        p_in_list = []
        for wy_reg in years_for_regression:
            prism = m["prec"].read_prism_prec(
                base_dir=self.base_dir,
                ths=self.ths,
                hsr=self.hsr_key,
                wy=wy_reg,
                comid_ths_flowline=self.ctx.nlcd.comid_ths,
                gridcode_ths_nlcd=self.ctx.nlcd.gridcode_ths,
            )
            prism_list.append(prism)
            p_in = prism.prism_ths[p_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            p_in_list.append(p_in)
        self._prism_reg = prism_list[0] if prism_list else None
        self._prsm_prec_reg = np.stack(p_in_list, axis=0) if p_in_list else None
        self._p_in0_reg = p_in_list[0] if p_in_list else None

        # Load station history for all regression years
        self._sta_hist_list_reg = []
        temp_res_list = []
        afstruct_years: list[Any] = []
        for iy, wy_reg in enumerate(years_for_regression):
            prism = prism_list[iy]
            setup_ctx = m["setup"].setup_data(wy1=wy_reg, iy=0, ths=self.ths)
            days_in_mo_reg = np.asarray(setup_ctx.days_in_mo, dtype=float)

            afstruct, poa, stations = m["gen"].gen_struc_data(
                base_dir=self.base_dir,
                ths=self.ths,
                hsr=self.hsr_key,
                wy=wy_reg,
                iy=0,
                ny=1,
                afstruct=None,
                poa=None,
            )

            inflow = m["in"].read_in_flow_wy(
                base_dir=self.base_dir,
                ths=self.ths,
                hsr=self.hsr_key,
                wy=wy_reg,
                iy=1,
                sta_ths=stations,
                poa=poa,
                comid_ths=self.ctx.nlcd.comid_ths,
                n_reaches=len(self.ctx.nlcd.comid_ths),
            )

            q_cols = [f"Q{i:02d}" for i in range(1, 14)]
            q = inflow.station_flow_df[q_cols].to_numpy(dtype=float)
            sta_wy = inflow.station_flow_df["StaWY"].astype(str).tolist()
            nwis_area = inflow.station_flow_df["NWISArea"].to_numpy(dtype=float)
            comid_wu = inflow.comid_wu_df["ComID_WU"].to_numpy(dtype=np.int64)

            diag_dir = self.base_dir / self.hsr_key / "Output" / "Diagnostics"
            sta_res = m["sta"].sta_basin_grid_comid_wy(
                afstruct=afstruct,
                hsr=self.hsr_key,
                ths=self.ths,
                wy=wy_reg,
                iy=0,
                sta_wy=sta_wy,
                q=q,
                nwis_area=nwis_area,
                comid_wu=comid_wu,
                comid_ths=self.ctx.nlcd.comid_ths,
                reach_wu=inflow.reach_wu,
                output_dir=diag_dir,
                sta_hist=None,
                plot_matrix=False,
            )

            plot_res = m["plot"].plot_areas_flows(
                afstruct=sta_res.afstruct,
                sta_hist=sta_res.sta_hist,
                hsr=self.hsr_key,
                iy=0,
                wy=wy_reg,
                net_design=sta_res.net_design,
                q_tot_wy=sta_res.sta_hist[0].q_tot_wy,
                nhd_area_iwy=sta_res.sta_hist[0].nhd_area_iwy,
                nwis_area_iwy=sta_res.sta_hist[0].nwis_area_iwy,
                month_names=self.ctx.month_name,
                make_plots=False,
            )

            temp_res = m["temp"].read_prism_temp(
                base_dir=self.base_dir,
                ths=self.ths,
                hsr=self.hsr_key,
                wy=wy_reg,
                wy_n=wy_reg,
                iy=0,
                ny=1,
                n_ths=len(self.ctx.nlcd.comid_ths),
                sta_ndx=sta_res.sta_ndx,
                grid_code_p_ths=prism.prism_ths["GridCode"].to_numpy(dtype=np.int64),
                comid_ths=self.ctx.nlcd.comid_ths,
                nlcd_ths=self.ctx.nlcd.nlcd_ths,
                p_in=p_in_list[iy],
                afstruct=plot_res.afstruct,
                output_dir=diag_dir,
            )
            if temp_res is None:
                raise RuntimeError(f"Temperature preprocessing returned no result for WY{wy_reg}")
            temp_res_list.append(temp_res)
            afstruct_years.append(temp_res.afstruct[self.hsr_key][0])

            q_adj_inc_wy = plot_res.q_adj_inc_wy
            if q_adj_inc_wy is None:
                nsta = len(sta_res.sta_ndx)
                q_adj_inc_wy = np.zeros((nsta, 12), dtype=float)

            year_block = temp_res.afstruct[self.hsr_key][0]
            sta_ndx_reg = _align_sta_ndx_to_year_block(sta_res.sta_ndx, year_block)

            # Mirror Step-1 cleanup so regression predictors remain finite.
            grid_to_nlcd: dict[int, np.ndarray] = {
                int(g): np.nan_to_num(np.asarray(v, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                for g, v in zip(self.ctx.nlcd.gridcode_ths, self.ctx.nlcd.nlcd_ths)
            }
            for sidx in sta_ndx_reg:
                if isinstance(year_block, dict):
                    rec_key = sidx if sidx in year_block else str(sidx)
                    rec = year_block[rec_key]
                else:
                    rec_key = sidx
                    rec = year_block[sidx]

                rec["Precip"] = np.nan_to_num(
                    np.asarray(rec.get("Precip", np.zeros(12)), dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                rec["Temp"] = np.nan_to_num(
                    np.asarray(rec.get("Temp", np.zeros(12)), dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                nlcd_vec = np.asarray(rec.get("NLCD", np.zeros(21)), dtype=float)
                if not np.isfinite(nlcd_vec).all():
                    sb_codes = np.asarray(rec.get("SBGridCode", []), dtype=float)
                    sb_codes = sb_codes[np.isfinite(sb_codes)].astype(int)
                    candidates = [grid_to_nlcd[int(code)] for code in sb_codes if int(code) in grid_to_nlcd]
                    if candidates:
                        nlcd_vec = np.mean(np.vstack(candidates), axis=0)
                    else:
                        nlcd_vec = np.zeros(21, dtype=float)
                rec["NLCD"] = np.nan_to_num(nlcd_vec, nan=0.0, posinf=0.0, neginf=0.0)

                if isinstance(year_block, dict):
                    year_block[rec_key] = rec
                else:
                    year_block[sidx] = rec

            sta_hist = {
                "StaList": sta_res.sta_list,
                "StaNdx": sta_ndx_reg,
                "NStaAct": sta_res.sta_hist[0].n_sta_act,
                "QTotWY": sta_res.sta_hist[0].q_tot_wy,
                "NHD_Area_IWY": np.asarray(sta_res.sta_hist[0].nhd_area_iwy, dtype=float),
                "YAdjIncWY": _station_y_adj_inc(q_adj_inc_wy, sta_res.sta_hist[0].nhd_area_iwy, days_in_mo_reg),
            }
            self._sta_hist_list_reg.append(sta_hist)

        if temp_res_list:
            combined_afstruct = {self.hsr_key: afstruct_years}
            last_temp_res = temp_res_list[-1]
            self._temp_res_reg = SimpleNamespace(
                afstruct=combined_afstruct,
                array_nlcd=getattr(last_temp_res, "array_nlcd", None),
                prsm_prec=getattr(last_temp_res, "prsm_prec", None),
                prsm_temp=getattr(last_temp_res, "prsm_temp", None),
                prsm_temp_ths=getattr(last_temp_res, "prsm_temp_ths", None),
            )
        else:
            self._temp_res_reg = self.ctx.temp_res
        self.log(f"[Regression] Loaded {self.ny_reg} years of data\n")

    def step_run_regression(self) -> ConvertedAFinchContext:
        if self.ctx.temp_res is None or self.ctx.sta_hist_list == [] or self.ctx.prism is None or self.ctx.p_in0 is None:
            raise RuntimeError("Step 1 must be completed before regression.")
        m = self.ctx.modules

        self.log("[STEP 2/6] Running monthly and annual regressions...\n")
        
        # Load multi-year data if regression years differ from main model years
        if self.wy1_reg != self.wy1 or self.ny_reg != self.ny:
            self._load_multiyear_regression_data()
            wy1_for_reg = self.wy1_reg
            ny_for_reg = self.ny_reg
            temp_res_for_reg = self._temp_res_reg
            sta_hist_for_reg = self._sta_hist_list_reg
            prism_for_reg = self._prism_reg
            p_in0_for_reg = self._p_in0_reg
            self.log(f"[Regression] Using multi-year data: WY{wy1_for_reg}-{wy1_for_reg + ny_for_reg - 1}\n")
        else:
            wy1_for_reg = self.wy1
            ny_for_reg = self.ny
            temp_res_for_reg = self.ctx.temp_res
            sta_hist_for_reg = self.ctx.sta_hist_list
            prism_for_reg = self.ctx.prism
            p_in0_for_reg = self.ctx.p_in0
            self.log(f"[Regression] Using single year data: WY{wy1_for_reg}\n")

        # Validate structures before entering AFRegressPOA, which otherwise raises
        # opaque NoneType subscripting errors when a station/year entry is missing.
        if temp_res_for_reg is None or getattr(temp_res_for_reg, "afstruct", None) is None:
            raise RuntimeError("Regression temperature/afstruct inputs are missing.")
        afstruct_reg = temp_res_for_reg.afstruct
        if self.hsr_key not in afstruct_reg:
            raise RuntimeError(f"Regression afstruct missing key '{self.hsr_key}'.")
        afstruct_years = afstruct_reg[self.hsr_key]
        if len(afstruct_years) < ny_for_reg:
            raise RuntimeError(
                f"Regression afstruct has {len(afstruct_years)} year blocks, expected {ny_for_reg}."
            )
        if len(sta_hist_for_reg) < ny_for_reg:
            raise RuntimeError(
                f"Regression station history has {len(sta_hist_for_reg)} years, expected {ny_for_reg}."
            )

        for iy in range(ny_for_reg):
            year_block = afstruct_years[iy]
            if year_block is None:
                raise RuntimeError(f"Regression afstruct year block is None for WY{wy1_for_reg + iy}.")

            hist = sta_hist_for_reg[iy]
            sta_ndx = np.asarray(hist.get("StaNdx", []), dtype=int)
            n_sta_act = int(hist.get("NStaAct", len(sta_ndx)))
            n_check = min(n_sta_act, len(sta_ndx))
            for is_idx in range(n_check):
                sidx = int(sta_ndx[is_idx])
                if not _year_block_has_index(year_block, sidx):
                    raise RuntimeError(
                        f"Station index out of range in regression data: WY{wy1_for_reg + iy}, "
                        f"StaNdx={sidx}, year_block_type={type(year_block).__name__}, "
                        f"year_block_len={len(year_block)}"
                    )
                rec = year_block[sidx] if (not isinstance(year_block, dict) or sidx in year_block) else year_block[str(sidx)]
                if rec is None:
                    raise RuntimeError(
                        f"Missing station struct in regression data: WY{wy1_for_reg + iy}, StaNdx={sidx}"
                    )

        nr = 10
        self.ctx.cb_matrix = np.zeros((nr, 24), dtype=int)
        self.ctx.cb_matrix[0, 21] = 1
        self.ctx.cb_matrix[1, 22] = 1
        self.ctx.cb_matrix[2, 23] = 1
        self.ctx.cb_matrix[3, 2] = 1
        self.ctx.cb_matrix[3, 3] = 1
        self.ctx.cb_matrix[3, 4] = 1
        self.ctx.cb_matrix[4, 8] = 1
        self.ctx.cb_matrix[4, 9] = 1
        self.ctx.cb_matrix[4, 10] = 1
        self.ctx.cb_matrix[5, 13] = 1
        self.ctx.cb_matrix[5, 14] = 1
        self.ctx.cb_matrix[5, 15] = 1
        self.ctx.cb_matrix[5, 16] = 1
        self.ctx.cb_matrix[5, 17] = 1
        self.ctx.cb_matrix[5, 18] = 1
        self.ctx.cb_matrix[6, 0] = 1
        self.ctx.cb_matrix[6, 1] = 1
        self.ctx.cb_matrix[7, 5] = 1
        self.ctx.cb_matrix[7, 6] = 1
        self.ctx.cb_matrix[7, 7] = 1
        self.ctx.cb_matrix[8, 11] = 1
        self.ctx.cb_matrix[8, 12] = 1
        self.ctx.cb_matrix[8, 13] = 1
        self.ctx.cb_matrix[9, 19] = 1
        self.ctx.cb_matrix[9, 20] = 1
        self.ctx.reg_var_name = [
            "PPT",
            "TEMP",
            "PPT_LAG1",
            "NLCD_DEV",
            "NLCD_FOR",
            "NLCD_AG",
            "NLCD_WATER",
            "NLCD_BARREN",
            "NLCD_SHRUB_GRASS",
            "NLCD_WETLAND",
        ]

        p_cols = [f"PIn_{i:02d}" for i in range(1, 13)]
        reg_prsm_prec = self._prsm_prec_reg
        if reg_prsm_prec is None:
            reg_prsm_prec = np.nan_to_num(
                prism_for_reg.prism_ths[p_cols].to_numpy(dtype=float),
                nan=0.0,
            ).reshape(ny_for_reg, -1, 12)
        elif reg_prsm_prec.shape[0] != ny_for_reg:
            raise RuntimeError(
                f"Regression precipitation years mismatch: got {reg_prsm_prec.shape[0]}, expected {ny_for_reg}."
            )

        self.ctx.reg_poa = m["regpoa"].regress_poa(
            afstruct=temp_res_for_reg.afstruct,
            hsr=self.hsr_key,
            sta_hist=sta_hist_for_reg,
            wy1=wy1_for_reg,
            ny=ny_for_reg,
            nr=nr,
            cb_matrix=self.ctx.cb_matrix,
            p_in0=p_in0_for_reg,
            prsm_prec_ths=reg_prsm_prec,
            reg_var_name=self.ctx.reg_var_name,
            mo_name=self.ctx.mo_name,
            prompt_pvalues=lambda: (P_ENTER, P_REMOVE),
            make_plot=True,
        )
        self._write_regression_diagnostics(
            reg_mat=self.ctx.reg_poa.reg_mat,
            reg_month=self.ctx.reg_poa.reg_month,
            reg_var_name=self.ctx.reg_var_name,
            wy1=wy1_for_reg,
            ny=ny_for_reg,
        )

        self.ctx.reg_hist = m["regwy"].regress_by_wy(
            reg_mat=self.ctx.reg_poa.reg_mat,
            reg_month=self.ctx.reg_poa.reg_month,
            wy1=wy1_for_reg,
            ny=ny_for_reg,
            month_name=self.ctx.month_name,
            make_plots=True,
        )

        self._write_regression_observed_modeled(
            reg_mat=self.ctx.reg_poa.reg_mat,
            reg_month=self.ctx.reg_poa.reg_month,
            reg_hist=self.ctx.reg_hist,
            wy1=wy1_for_reg,
            ny=ny_for_reg,
        )
        self._save_regression_plots(
            reg_month=self.ctx.reg_poa.reg_month,
            reg_hist=self.ctx.reg_hist,
            wy1=wy1_for_reg,
            ny=ny_for_reg,
        )

        self.log("Step 2 complete.\n")
        return self.ctx

    def _write_regression_diagnostics(
        self,
        reg_mat: np.ndarray,
        reg_month: list[Any],
        reg_var_name: list[str],
        wy1: int,
        ny: int,
    ) -> None:
        diag_dir = self.base_dir / self.hsr_key / "Output" / "Diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        cols = ["WY", "Station", "Month", "YAdjIncWY", *reg_var_name]
        design = pd.DataFrame(reg_mat, columns=cols)
        design["WY"] = design["WY"].astype(int)
        design["Month"] = design["Month"].astype(int)
        design_path = diag_dir / f"RegressionDesignMatrix_WY{wy1}_{wy1 + ny - 1}.csv"
        design.to_csv(design_path, index=False)

        rows = []
        for im, month in enumerate(reg_month):
            r2 = 1.0 - month.stats.SSresid / month.stats.SStotal if month.stats.SStotal != 0 else np.nan
            selected = [name for name, keep in zip(reg_var_name, month.inmodel) if bool(keep)]
            month_design = design[design["Month"] == im + 1]
            y = pd.to_numeric(month_design["YAdjIncWY"], errors="coerce")
            rows.append({
                "Month": self.ctx.mo_name[im] if im < len(self.ctx.mo_name) else im + 1,
                "N": int(y.notna().sum()),
                "YMean": float(y.mean()),
                "YStd": float(y.std()),
                "YMin": float(y.min()),
                "YMax": float(y.max()),
                "SelectedVariables": ";".join(selected),
                "NumVariables": int(month.stats.df0),
                "DFE": int(month.stats.dfe),
                "RMSE": float(month.stats.rmse),
                "FStat": float(month.stats.fstat),
                "PValue": float(month.stats.pval),
                "R2": float(r2),
            })

        summary_path = diag_dir / f"RegressionMonthlySummary_WY{wy1}_{wy1 + ny - 1}.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        self.log(f"Regression diagnostics written: {design_path.name}, {summary_path.name}\n")

    def _write_regression_observed_modeled(
        self,
        reg_mat: np.ndarray,
        reg_month: list[Any],
        reg_hist: list[list[Any]],
        wy1: int,
        ny: int,
    ) -> None:
        diag_dir = self.base_dir / self.hsr_key / "Output" / "Diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for iy in range(ny):
            wy = wy1 + iy
            ndx_wy = np.flatnonzero(reg_mat[:, 0] == wy)
            for im in range(12):
                ndx_mo = np.flatnonzero(reg_mat[:, 2] == (im + 1))
                ndx_obs = np.intersect1d(ndx_wy, ndx_mo)
                if ndx_obs.size == 0:
                    continue

                observed = reg_mat[ndx_obs, 3].astype(float)
                station_ids = reg_mat[ndx_obs, 1]
                entry = reg_hist[iy][im]

                ols_pred = np.asarray(entry.OLSPredRUAdj, dtype=float)
                robust_pred = np.asarray(entry.RobustPredRUAdj, dtype=float)
                if ols_pred.shape[0] != observed.shape[0]:
                    ols_pred = np.full(observed.shape[0], np.nan, dtype=float)
                if robust_pred.shape[0] != observed.shape[0]:
                    robust_pred = np.full(observed.shape[0], np.nan, dtype=float)

                ols_resid = observed - ols_pred
                robust_resid = observed - robust_pred

                poa_stats = reg_month[im].stats
                ols_r2 = float(entry.stats[0]) if np.asarray(entry.stats).size >= 1 else np.nan
                ols_mse = float(entry.stats[3]) if np.asarray(entry.stats).size >= 4 else np.nan
                ols_rmse = float(np.sqrt(ols_mse)) if np.isfinite(ols_mse) else np.nan
                robust_rmse = float(np.sqrt(np.nanmean(robust_resid ** 2))) if robust_resid.size else np.nan

                for i in range(observed.shape[0]):
                    rows.append({
                        "WY": int(wy),
                        "MonthNum": int(im + 1),
                        "Month": self.mo_name[im] if im < len(self.mo_name) else str(im + 1),
                        "Station": str(station_ids[i]),
                        "ObservedYAdjInc": float(observed[i]),
                        "OLSPredYAdjInc": float(ols_pred[i]) if np.isfinite(ols_pred[i]) else np.nan,
                        "RobustPredYAdjInc": float(robust_pred[i]) if np.isfinite(robust_pred[i]) else np.nan,
                        "OLSResidual": float(ols_resid[i]) if np.isfinite(ols_resid[i]) else np.nan,
                        "RobustResidual": float(robust_resid[i]) if np.isfinite(robust_resid[i]) else np.nan,
                        "POA_R2": float(1.0 - poa_stats.SSresid / poa_stats.SStotal) if poa_stats.SStotal != 0 else np.nan,
                        "POA_RMSE": float(poa_stats.rmse),
                        "WY_OLS_R2": ols_r2,
                        "WY_OLS_RMSE": ols_rmse,
                        "WY_Robust_RMSE": robust_rmse,
                    })

        out_path = diag_dir / f"RegressionObservedVsModeled_WY{wy1}_{wy1 + ny - 1}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.log(f"Regression observed-vs-modeled CSV written: {out_path.name}\n")

    def _save_regression_plots(
        self,
        reg_month: list[Any],
        reg_hist: list[list[Any]],
        wy1: int,
        ny: int,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.log(f"[Regression] Plot save skipped (matplotlib unavailable): {exc}\n")
            return

        m = self.ctx.modules
        plot_dir = self.base_dir / self.hsr_key / "Output" / "Diagnostics" / "RegressionPlots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # POA t-value heatmap (figure 58 from AFRegressPOA)
        if plt.fignum_exists(58):
            fig58 = plt.figure(58)
            fig58.savefig(plot_dir / f"POA_TValues_WY{wy1}_{wy1 + ny - 1}.png", dpi=200, bbox_inches="tight")
            plt.close(fig58)

        # WY observed-vs-estimated subplots (figures 60+iy from AFRegressByWY)
        for iy in range(ny):
            fig_no = 60 + iy
            if not plt.fignum_exists(fig_no):
                continue
            wy = wy1 + iy
            fig = plt.figure(fig_no)
            fig.savefig(plot_dir / f"ObservedVsEstimated_WY{wy}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Coefficient plots for each month via AFPlotRegressCoeff (figure 82)
        if "plotcoeff" in m:
            for im in range(12):
                m["plotcoeff"].plot_regress_coeff(
                    reg_hist=reg_hist,
                    reg_month=reg_month,
                    reg_var_name=self.ctx.reg_var_name,
                    month_name=self.month_name,
                    im=im,
                    hsr=self.hsr_key,
                    wy1=wy1,
                    wyn=wy1 + ny - 1,
                    ny=ny,
                )
                if plt.fignum_exists(82):
                    fig82 = plt.figure(82)
                    mo = self.mo_name[im] if im < len(self.mo_name) else f"M{im + 1:02d}"
                    fig82.savefig(plot_dir / f"RegressCoeff_{mo}_WY{wy1}_{wy1 + ny - 1}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig82)

        self.log(f"Regression plots saved to: {plot_dir}\n")

    def step_estimate_incremental(self) -> ConvertedAFinchContext:
        if self.ctx.reg_poa is None or self.ctx.reg_hist is None:
            raise RuntimeError("Step 2 must be completed before incremental estimation.")
        m = self.ctx.modules

        self.log("[STEP 3/6] Estimating unconstrained incremental flow/yield...\n")
        p_cols = [f"PIn_{i:02d}" for i in range(1, 13)]
        self.ctx.prsm_prec_ths = np.nan_to_num(self.ctx.prism.prism_ths[p_cols].to_numpy(dtype=float), nan=0.0).reshape(self.ny, -1, 12)
        self.ctx.prsm_temp_ths = np.nan_to_num(self.ctx.temp_res.prsm_temp_ths, nan=0.0)
        self.ctx.prsm_prem_ths = m["lag"].gen_lag1_prec(prsm_prec_ths=self.ctx.prsm_prec_ths, p_in0=self.ctx.p_in0)
        self.ctx.gc_area_sq_mi = np.nan_to_num(
            pd.to_numeric(self.ctx.prism.prism_ths["GCAreaSqMi"], errors="coerce").to_numpy(dtype=float),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        self.ctx.y_est_adj_inc, self.ctx.q_est_adj_inc = m["qest"].q_est_adj_inc(
            nlcd_ths=self.ctx.nlcd.nlcd_ths,
            prsm_prec_ths=self.ctx.prsm_prec_ths,
            prsm_temp_ths=self.ctx.prsm_temp_ths,
            prsm_prem_ths=self.ctx.prsm_prem_ths,
            cb_matrix=self.ctx.cb_matrix,
            reg_month=self.ctx.reg_poa.reg_month,
            reg_hist=self.ctx.reg_hist,
            gc_area_sq_mi=self.ctx.gc_area_sq_mi,
            days_in_mo=np.asarray(self.ctx.days_in_mo[:12], dtype=float),
        )
        self.log("Step 3 complete.\n")
        return self.ctx

    def step_constrain_incremental(self) -> ConvertedAFinchContext:
        if self.ctx.q_est_adj_inc is None or self.ctx.gc_area_sq_mi is None:
            raise RuntimeError("Step 3 must be completed before applying constraints.")
        m = self.ctx.modules

        self.log("[STEP 4/6] Applying station and network constraints...\n")
        (
            self.ctx.afstruct_con,
            self.ctx.con_adjust,
            self.ctx.q_con_adj_inc,
            self.ctx.y_con_adj_inc,
        ) = m["qcon"].q_con_adj_inc(
            afstruct=self.ctx.temp_res.afstruct,
            hsr=self.hsr_key,
            sta_hist=self.ctx.sta_hist_list,
            grid_code_ths=self.ctx.nlcd.gridcode_ths,
            q_est_adj_inc=self.ctx.q_est_adj_inc,
            gc_area_sq_mi=self.ctx.gc_area_sq_mi,
            days_in_mo=np.asarray(self.ctx.days_in_mo[:12], dtype=float),
        )
        self.log("Step 4 complete.\n")
        return self.ctx

    def step_write_incremental_output(self) -> ConvertedAFinchContext:
        if self.ctx.q_con_adj_inc is None or self.ctx.y_con_adj_inc is None:
            raise RuntimeError("Step 4 must be completed before writing outputs.")
        m = self.ctx.modules

        self.log("[STEP 5/6] Writing THS incremental flow/yield outputs...\n")
        self.ctx.qy_path = m["wrt"].write_qy_est_con(
            base_dir=self.base_dir,
            hsr=self.hsr_key,
            ths=self.ths,
            iy=0,
            wy1=self.wy1,
            ny=self.ny,
            grid_code_ths=self.ctx.nlcd.gridcode_ths,
            comid_ths=self.ctx.nlcd.comid_ths,
            gc_area_sq_mi=self.ctx.gc_area_sq_mi,
            q_est_adj_inc=self.ctx.q_est_adj_inc,
            y_est_adj_inc=self.ctx.y_est_adj_inc,
            q_con_adj_inc=self.ctx.q_con_adj_inc,
            y_con_adj_inc=self.ctx.y_con_adj_inc,
            sta_ths=self.ctx.stations,
            poa=self.ctx.poa,
        )
        self.log(f"Step 5 complete. Output: {self.ctx.qy_path}\n")
        return self.ctx

    def step_accumulate_flow(self) -> ConvertedAFinchContext:
        if self.ctx.sta_hist_list == []:
            raise RuntimeError("Step 1 must be completed before flow accumulation.")
        m = self.ctx.modules

        self.log("[STEP 6/6] Accumulating constrained flow through HydroSeq...\n")
        _ensure_vaa_for_accumulation(self.base_dir, self.hsr_key)
        self.ctx.flow_comid, self.ctx.flow_accum = m["acc"].con_flow_accum(
            base_dir=self.base_dir,
            hsr=self.hsr_key,
            iy=0,
            wy1=self.wy1,
            sta_hist=self.ctx.sta_hist_list,
            month_name=self.ctx.month_name,
            plot_debug=False,
        )
        self.ctx.flow_accum_path = self.base_dir / self.hsr_key / "Output" / "FlowAccum" / f"ComIDQ12WY{self.ctx.wy}.csv"
        self.log(f"Step 6 complete. Output: {self.ctx.flow_accum_path}\n")
        return self.ctx

    def run_all(self) -> ConvertedAFinchContext:
        self.step_setup_inputs()
        self.step_run_regression()
        self.step_estimate_incremental()
        self.step_constrain_incremental()
        self.step_write_incremental_output()
        self.step_accumulate_flow()
        return self.ctx


def run() -> None:
    pipeline = ConvertedAFinchPipeline(
        base_dir=ROOT,
        src_dir=SRC,
        ths=THS,
        hsr_key=HSR_KEY,
        wy1=WY1,
        ny=NY,
        wy1_reg=WY1_REG,
        ny_reg=NY_REG,
        logger=print,
    )
    ctx = pipeline.run_all()
    print("CONVERTED_AFINCH_FULL_RUN_COMPLETE")
    print(f"QY output: {ctx.qy_path}")
    print(f"Flow accumulation output: {ctx.flow_accum_path}")
    print(f"flow_accum_shape={ctx.flow_accum.shape}")
    print(f"con_adjust_shape={ctx.con_adjust.shape}")
    print(f"n_flow_comid={len(ctx.flow_comid)}")


if __name__ == "__main__":
    run()
