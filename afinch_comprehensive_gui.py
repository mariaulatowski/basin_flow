"""
AFINCH Comprehensive GUI
Handles the full workflow from basin setup to model run and shapefile export.
Designed to be usable by colleagues for any Texas basin (Brazos, Colorado, etc.)

Tabs:
  1. Basin Setup   – basin shapefile, NHD flowlines, gage CSV
  2. Climate Data  – NLCD raster, PRISM directories (auto-detected)
  3. Build Network – prepares HSR directory from inputs above
  4. Run Model     – steps 1-6, single or multi-year
  5. Export        – step 7: shapefile of accumulated flows on NHD lines for ArcGIS
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Any, Callable
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

import geopandas as gpd

# ──────────────────────────────────────────────────────────────────────────────
# Palette and fonts
# ──────────────────────────────────────────────────────────────────────────────
BG = "#1e3a5f"          # dark navy
PANEL = "#f4f8fb"       # very light blue-white
ACCENT = "#2e7dba"      # USGS blue
ACCENT2 = "#43b374"     # green: success / complete
WARN = "#e3a020"        # amber: in-progress / partial
ERROR = "#c0392b"       # red: error

TITLE_FONT = ("Segoe UI", 15, "bold")
TAB_FONT = ("Segoe UI", 10, "bold")
LABEL_FONT = ("Segoe UI", 9, "bold")
ENTRY_FONT = ("Segoe UI", 9)
LOG_FONT = ("Consolas", 8)
SMALL_FONT = ("Segoe UI", 8)

STEP_LABELS = {
    1: "1 · Setup Inputs",
    2: "2 · Run Regression",
    3: "3 · Estimate Incremental",
    4: "4 · Constrain Incremental",
    5: "5 · Write Output",
    6: "6 · Accumulate Flow",
    7: "7 · Export Shapefile",
}

STEP_DEPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


# ──────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ──────────────────────────────────────────────────────────────────────────────

class BrowseRow:
    """Label + Entry + Browse button row."""

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.StringVar,
        row: int,
        mode: str = "file",          # "file" | "dir" | "savefile"
        filetypes: list | None = None,
        tip: str = "",
        entry_width: int = 52,
    ):
        tk.Label(parent, text=label, font=LABEL_FONT, anchor="w", bg=PANEL).grid(
            row=row, column=0, sticky="w", padx=(8, 4), pady=3
        )
        ent = tk.Entry(parent, textvariable=var, width=entry_width, font=ENTRY_FONT)
        ent.grid(row=row, column=1, sticky="ew", padx=4, pady=3)

        ft = filetypes or [("All files", "*.*")]

        def browse():
            if mode == "dir":
                path = filedialog.askdirectory(title=f"Select: {label}")
            elif mode == "savefile":
                path = filedialog.asksaveasfilename(title=f"Save: {label}", filetypes=ft)
            else:
                path = filedialog.askopenfilename(title=f"Select: {label}", filetypes=ft)
            if path:
                var.set(path)

        tk.Button(
            parent, text="Browse…", command=browse, font=SMALL_FONT,
            bg=ACCENT, fg="white", activebackground="#1a6aa8", relief="flat", padx=6
        ).grid(row=row, column=2, padx=(4, 8), pady=3)
        parent.grid_columnconfigure(1, weight=1)


class Section(ttk.LabelFrame):
    """Styled LabelFrame."""
    def __init__(self, parent, text: str, **kw):
        super().__init__(parent, text=text, padding=10, **kw)


class StepIndicator(tk.Label):
    """Coloured circle indicator for step status."""
    def __init__(self, parent, **kw):
        super().__init__(parent, text="●", font=("Segoe UI", 14), **kw)
        self.set_idle()

    def set_idle(self):     self.configure(fg="#b0bec5")
    def set_running(self):  self.configure(fg=WARN)
    def set_done(self):     self.configure(fg=ACCENT2)
    def set_error(self):    self.configure(fg=ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────

class AFinchComprehensiveGUI:
    VERSION = "2.0"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AFINCH · Analysis of Flow in Networks of Channels")
        self.root.geometry("1020x780+200+80")
        self.root.configure(bg=BG)
        self.root.minsize(800, 640)

        self._log_q: queue.Queue[tuple[str, str]] = queue.Queue()
        self._is_running = False
        self._run_thread: threading.Thread | None = None
        self._pipeline: Any = None
        self._pipeline_sig: tuple | None = None
        self._completed_steps: set[int] = set()
        self._step_indicators: dict[int, Stepindicator] = {}
        self._step_btns: dict[int, tk.Button] = {}

        # ── State variables ──────────────────────────────────────────────────
        # Basin Setup
        self.v_base_dir        = tk.StringVar(value=str(Path.cwd()))
        self.v_basin_shp       = tk.StringVar()
        self.v_basin_field     = tk.StringVar(value="basin_name")
        self.v_basin_value     = tk.StringVar()
        self.v_nhd_dir         = tk.StringVar()
        self.v_gages_csv       = tk.StringVar()
        self.v_wam_csv         = tk.StringVar()
        self.v_usgs_only       = tk.BooleanVar(value=False)
        self.v_hu4_codes       = tk.StringVar(value="")
        self.v_hsr_name        = tk.StringVar(value="HSR1200")
        self.v_ths_code        = tk.StringVar(value="1200")

        # Climate Data
        self.v_nlcd_raster     = tk.StringVar()
        self.v_prism_ppt_dir   = tk.StringVar()
        self.v_prism_tmean_dir = tk.StringVar()
        self.v_build_wy        = tk.StringVar(value="2018")

        # Run Model
        self.v_run_wy_start    = tk.StringVar(value="2018")
        self.v_run_wy_end      = tk.StringVar(value="2018")

        # Export
        self.v_export_wy       = tk.StringVar(value="2018")
        self.v_export_month    = tk.StringVar(value="1")
        self.v_export_shp      = tk.StringVar()

        self._build_ui()
        self.root.after(150, self._drain_log)

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(
            hdr, text="AFINCH  ·  Analysis of Flow in Networks of Channels",
            font=TITLE_FONT, fg="white", bg=BG
        ).pack(side="left")
        tk.Label(
            hdr, text=f"v{self.VERSION}", font=SMALL_FONT, fg="#90caf9", bg=BG
        ).pack(side="left", padx=8, anchor="s")

        # Notebook
        style = ttk.Style()
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=TAB_FONT, padding=[14, 6])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=8)

        self._tab_basin   = self._make_tab(nb, "🗺  Basin Setup")
        self._tab_climate = self._make_tab(nb, "🌧  Climate Data")
        self._tab_build   = self._make_tab(nb, "⚙  Build Network")
        self._tab_run     = self._make_tab(nb, "▶  Run Model")
        self._tab_export  = self._make_tab(nb, "📂  Export")

        self._build_tab_basin()
        self._build_tab_climate()
        self._build_tab_build()
        self._build_tab_run()
        self._build_tab_export()

        # Shared log at the bottom
        log_frame = tk.LabelFrame(self.root, text="Log", font=LABEL_FONT, bg=BG, fg="white", padx=4, pady=4)
        log_frame.pack(fill="both", expand=False, padx=10, pady=(0, 6))
        self.log_box = scrolledtext.ScrolledText(log_frame, height=9, font=LOG_FONT, wrap=tk.WORD,
                                                  bg="#0d1b2a", fg="#c8e6fa", insertbackground="white")
        self.log_box.pack(fill="both", expand=True)
        self.log_box.tag_configure("ok",   foreground=ACCENT2)
        self.log_box.tag_configure("warn", foreground=WARN)
        self.log_box.tag_configure("err",  foreground=ERROR)

        self._append_log("AFINCH GUI initialized. Start with the '🗺 Basin Setup' tab.\n")

    def _make_tab(self, nb: ttk.Notebook, label: str) -> ttk.Frame:
        frame = ttk.Frame(nb)
        nb.add(frame, text=label)
        return frame

    # ──────────────────── Tab 1: Basin Setup ─────────────────────────────────

    def _build_tab_basin(self):
        outer = tk.Frame(self._tab_basin, bg=PANEL)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        # Base directory
        s0 = Section(outer, "Workspace")
        s0.pack(fill="x", padx=6, pady=(6, 0))
        BrowseRow(s0, "Base Directory", self.v_base_dir, 0, mode="dir",
                  tip="Root workspace folder (contains HSR folders, inputData, etc.)")
        tk.Button(s0, text="Auto-detect paths from Base Directory",
                  command=self._autodetect_paths,
                  font=SMALL_FONT, bg=ACCENT2, fg="white", relief="flat", padx=8
                  ).grid(row=1, column=1, sticky="w", pady=(4, 2))

        # Basin
        s1 = Section(outer, "Basin Polygon (Shapefile)")
        s1.pack(fill="x", padx=6, pady=(6, 0))
        BrowseRow(s1, "Basin Shapefile", self.v_basin_shp, 0, mode="file",
                  filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")],
                  tip="Polygon shapefile defining the basin boundary (e.g. TWDB_MRBs_2014.shp)")
        tk.Button(s1, text="Load Basin Names", command=self._load_basin_names,
                  font=SMALL_FONT, bg=ACCENT, fg="white", relief="flat", padx=6
                  ).grid(row=1, column=1, sticky="w", pady=(2, 0))

        row_field = tk.Frame(s1, bg=PANEL)
        row_field.grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        tk.Label(row_field, text="Name Field:", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 6))
        tk.Entry(row_field, textvariable=self.v_basin_field, width=20, font=ENTRY_FONT).pack(side="left", padx=4)
        tk.Label(row_field, text="Basin Value:", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(12, 6))
        self._basin_menu = tk.OptionMenu(row_field, self.v_basin_value, "")
        self._basin_menu.pack(side="left", padx=4)
        tk.Label(row_field, text="   (or type manually)", font=SMALL_FONT, fg="#777", bg=PANEL).pack(side="left")
        tk.Entry(row_field, textvariable=self.v_basin_value, width=20, font=ENTRY_FONT).pack(side="left", padx=4)

        # NHD source
        s2 = Section(outer, "NHD Flowline Source")
        s2.pack(fill="x", padx=6, pady=(6, 0))
        BrowseRow(s2, "NHD GDB/Directory", self.v_nhd_dir, 0, mode="dir",
                  tip="Folder with NHD_H_*_HU4_GDB subfolders (medium-resolution NHDPlus)")
        tk.Label(s2, text="The model will auto-discover which HU4 units intersect your basin polygon.",
                 font=SMALL_FONT, fg="#666", bg=PANEL).grid(row=1, column=0, columnspan=3, sticky="w", padx=8)

        # Gages
        s3 = Section(outer, "Gage Stations CSV")
        s3.pack(fill="x", padx=6, pady=(6, 0))
        BrowseRow(s3, "USGS Gages CSV", self.v_gages_csv, 0, mode="file",
                  filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
                  tip="CSV with columns: Station (or Gage_ID_norm), LAT, LONG")
        BrowseRow(s3, "WAM Control Points CSV", self.v_wam_csv, 1, mode="file",
                  filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
                  tip="Optional: CSV with WAM control point stations (columns: Station or CPID, LAT, LONG)")
        tk.Checkbutton(
            s3, text="Use USGS gages only (skip WAM control points entirely)",
            variable=self.v_usgs_only, font=SMALL_FONT, bg=PANEL, anchor="w"
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 0))
        tk.Label(s3, text="Required columns: Station, LAT, LONG  |  WAM CSV is optional; ignored when 'USGS only' is checked.",
                 font=SMALL_FONT, fg="#666", bg=PANEL).grid(row=3, column=0, columnspan=3, sticky="w", padx=8)

        # HSR / THS identifiers
        s4 = Section(outer, "Model Identifiers")
        s4.pack(fill="x", padx=6, pady=(6, 8))
        row_hu4 = tk.Frame(s4, bg=PANEL)
        row_hu4.pack(fill="x", pady=(0, 4))
        tk.Label(row_hu4, text="Hydrologic Subregion (HU4):", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 6))
        tk.Entry(row_hu4, textvariable=self.v_hu4_codes, width=16, font=ENTRY_FONT).pack(side="left", padx=(0, 6))
        tk.Button(
            row_hu4,
            text="Apply to THS/HSR",
            command=self._apply_hu4_to_ids,
            font=SMALL_FONT,
            bg=ACCENT,
            fg="white",
            relief="flat",
            padx=6,
        ).pack(side="left")
        tk.Label(
            row_hu4,
            text="Optional: 4-digit code (e.g., 1206) or comma-list for multi-HU4 build",
            font=SMALL_FONT,
            fg="#666",
            bg=PANEL,
        ).pack(side="left", padx=8)

        row_ids = tk.Frame(s4, bg=PANEL)
        row_ids.pack(fill="x")
        tk.Label(row_ids, text="HSR Folder Name:", font=LABEL_FONT, bg=PANEL).grid(row=0, column=0, sticky="w", padx=(0, 4))
        tk.Entry(row_ids, textvariable=self.v_hsr_name, width=14, font=ENTRY_FONT).grid(row=0, column=1, sticky="w", padx=4)
        tk.Label(row_ids, text="THS Code:", font=LABEL_FONT, bg=PANEL).grid(row=0, column=2, sticky="w", padx=(16, 4))
        tk.Entry(row_ids, textvariable=self.v_ths_code, width=8, font=ENTRY_FONT).grid(row=0, column=3, sticky="w", padx=4)
        tk.Label(row_ids, text="  e.g. HSR1200 / 1200 for Brazos,  HSR1300 / 1300 for Colorado",
                 font=SMALL_FONT, fg="#666", bg=PANEL).grid(row=1, column=0, columnspan=6, sticky="w", pady=(2, 0))

    # ──────────────────── Tab 2: Climate Data ────────────────────────────────

    def _build_tab_climate(self):
        outer = tk.Frame(self._tab_climate, bg=PANEL)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        s1 = Section(outer, "NLCD Land Cover Raster")
        s1.pack(fill="x", padx=6, pady=(6, 0))
        BrowseRow(s1, "NLCD Raster (.tif)", self.v_nlcd_raster, 0, mode="file",
                  filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
                  tip="Categorical NLCD raster (e.g. Annual_NLCD_LndCov_2018_CU_C1V1.tif)")
        tk.Label(s1, text="Download from: https://www.mrlc.gov/data  |  Any year 2016-2021 works for most analyses.",
                 font=SMALL_FONT, fg="#666", bg=PANEL).grid(row=1, column=0, columnspan=3, sticky="w", padx=8)

        s2 = Section(outer, "PRISM Climate Rasters (monthly grids)")
        s2.pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(s2, text="Point to the folder containing monthly PRISM .tif files for each variable.",
                 font=SMALL_FONT, fg="#666", bg=PANEL).grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        BrowseRow(s2, "Precipitation dir (ppt)", self.v_prism_ppt_dir, 1, mode="dir",
                  tip="Folder with prism_ppt_us_25m_YYYYMM.tif files (clipped or extracted)")
        BrowseRow(s2, "Temperature dir (tmean)", self.v_prism_tmean_dir, 2, mode="dir",
                  tip="Folder with prism_tmean_us_25m_YYYYMM.tif files (clipped or extracted)")
        tk.Button(s2, text="Auto-detect PRISM paths", command=self._autodetect_prism,
                  font=SMALL_FONT, bg=ACCENT, fg="white", relief="flat", padx=6
                  ).grid(row=3, column=1, sticky="w", pady=(4, 0))

        s3 = Section(outer, "Water Year to Sample PRISM for Network Build")
        s3.pack(fill="x", padx=6, pady=(6, 8))
        row_wy = tk.Frame(s3, bg=PANEL)
        row_wy.pack(fill="x")
        tk.Label(row_wy, text="Water Year:", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 6))
        tk.Entry(row_wy, textvariable=self.v_build_wy, width=8, font=ENTRY_FONT).pack(side="left")
        tk.Label(row_wy, text="   Used only for the Build Network step. Each modeled year uses that year's own PRISM data.",
                 font=SMALL_FONT, fg="#666", bg=PANEL).pack(side="left", padx=8)
        tk.Label(
            s3,
            text="Build Network will also generate upstream contributing catchments for gages.",
            font=SMALL_FONT,
            fg="#666",
            bg=PANEL,
        ).pack(anchor="w", padx=4, pady=(6, 0))

    # ──────────────────── Tab 3: Build Network ───────────────────────────────

    def _build_tab_build(self):
        outer = tk.Frame(self._tab_build, bg=PANEL)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        note = Section(outer, "What this step does")
        note.pack(fill="x", padx=6, pady=(6, 0))
        info = (
            "Build Network prepares all the HSR data files the model needs:\n\n"
            "  • Clips NHD flowlines to your basin polygon\n"
            "  • Discovers which HU4 datasets intersect the basin\n"
            "  • If HU4 is specified, restricts the build to that subregion(s)\n"
            "  • Snaps your gage stations to the nearest NHD reach (ComID)\n"
            "  • Filters submitted gages to only those inside selected HU4 footprint\n"
            "  • Extracts NLCD land-cover percentages per reach\n"
            "  • Extracts PRISM precipitation/temperature per reach for the chosen year\n"
            "  • Writes all required HSR files  (Flowlines/, NLCD/, PRISM/, GagedCatchments/, etc.)\n\n"
            "You only need to run this once per basin, or when your input data changes."
        )
        tk.Label(note, text=info, font=SMALL_FONT, justify="left", bg=PANEL, fg="#333").pack(anchor="w", padx=4)

        s1 = Section(outer, "Run Build Network")
        s1.pack(fill="x", padx=6, pady=(8, 0))

        self._build_status = tk.StringVar(value="Not built yet")
        self._build_status_lbl = tk.Label(s1, textvariable=self._build_status, font=LABEL_FONT, bg=PANEL, fg=WARN)
        self._build_status_lbl.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        btn_row = tk.Frame(s1, bg=PANEL)
        btn_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)

        self._dry_run_btn = tk.Button(
            btn_row, text="Dry Run (Validate Inputs)",
            command=lambda: self._start_build_network(dry_run=True),
            font=LABEL_FONT, bg=ACCENT, fg="white", relief="flat", padx=12, pady=6
        )
        self._dry_run_btn.pack(side="left", padx=(0, 8))

        self._build_btn = tk.Button(
            btn_row, text="✓  Build Network  (writes HSR files)",
            command=lambda: self._start_build_network(dry_run=False),
            font=LABEL_FONT, bg="#27ae60", fg="white", relief="flat", padx=12, pady=6
        )
        self._build_btn.pack(side="left")

        tk.Label(s1, text="Run 'Dry Run' first to verify inputs without writing any files.",
                 font=SMALL_FONT, fg="#666", bg=PANEL
                 ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 0))

    # ──────────────────── Tab 4: Run Model ───────────────────────────────────

    def _build_tab_run(self):
        outer = tk.Frame(self._tab_run, bg=PANEL)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        s1 = Section(outer, "Water Year(s)")
        s1.pack(fill="x", padx=6, pady=(6, 0))
        wy_row = tk.Frame(s1, bg=PANEL)
        wy_row.pack(fill="x")
        tk.Label(wy_row, text="Start WY:", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 4))
        tk.Entry(wy_row, textvariable=self.v_run_wy_start, width=8, font=ENTRY_FONT).pack(side="left", padx=(0, 16))
        tk.Label(wy_row, text="End WY:", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 4))
        tk.Entry(wy_row, textvariable=self.v_run_wy_end, width=8, font=ENTRY_FONT).pack(side="left")
        tk.Label(wy_row, text="   (single year: set both the same)",
                 font=SMALL_FONT, fg="#666", bg=PANEL).pack(side="left", padx=8)

        # Full run button
        full_row = tk.Frame(outer, bg=PANEL)
        full_row.pack(fill="x", padx=12, pady=(10, 4))
        self._run_all_btn = tk.Button(
            full_row, text="▶ Run Full Model (Steps 1–6)",
            command=self._start_full_run,
            font=("Segoe UI", 10, "bold"), bg=ACCENT, fg="white", relief="flat", padx=18, pady=8
        )
        self._run_all_btn.pack(side="left")
        self._reset_btn = tk.Button(
            full_row, text="Reset Steps", command=self._reset_steps,
            font=SMALL_FONT, bg="#7f8c8d", fg="white", relief="flat", padx=8, pady=4
        )
        self._reset_btn.pack(side="left", padx=12)

        # Step-by-step grid
        s2 = Section(outer, "Step-by-Step (run individually, in order 1 → 6)")
        s2.pack(fill="x", padx=6, pady=(8, 6))

        for step_no in range(1, 7):
            ind = StepIndicator(s2, bg=PANEL)
            ind.grid(row=step_no - 1, column=0, padx=(6, 0), pady=3)
            self._step_indicators[step_no] = ind

            btn = tk.Button(
                s2, text=STEP_LABELS[step_no],
                command=lambda s=step_no: self._start_single_step(s),
                font=LABEL_FONT, bg="#2c3e50", fg="white", relief="flat", padx=10, pady=4, width=28
            )
            btn.grid(row=step_no - 1, column=1, sticky="w", padx=6, pady=3)
            self._step_btns[step_no] = btn

        # Progress
        self._progress = ttk.Progressbar(outer, mode="indeterminate", length=300)
        self._progress.pack(padx=12, pady=(4, 0), anchor="w")
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(outer, textvariable=self._status_var, font=SMALL_FONT, bg=PANEL).pack(anchor="w", padx=14)

    # ──────────────────── Tab 5: Export Shapefile ────────────────────────────

    def _build_tab_export(self):
        outer = tk.Frame(self._tab_export, bg=PANEL)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        note = Section(outer, "Step 7 · Export Accumulated Flow Shapefile for ArcGIS / QGIS")
        note.pack(fill="x", padx=6, pady=(6, 0))
        info = (
            "Creates a line shapefile of the NHD flowlines for your basin with monthly\n"
            "accumulated flow values attached as attributes — one column per month.\n\n"
            "Open the output .shp in ArcGIS Pro, ArcMap, or QGIS and visualize\n"
            "flow by symbolizing on any month column (in CFS)."
        )
        tk.Label(note, text=info, font=SMALL_FONT, justify="left", bg=PANEL, fg="#333").pack(anchor="w", padx=4, pady=4)

        s1 = Section(outer, "Export Options")
        s1.pack(fill="x", padx=6, pady=(6, 0))

        row0 = tk.Frame(s1, bg=PANEL)
        row0.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        tk.Label(row0, text="Water Year to Export:", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 6))
        tk.Entry(row0, textvariable=self.v_export_wy, width=8, font=ENTRY_FONT).pack(side="left", padx=(0, 20))
        tk.Label(row0, text="Month (1-12, or ALL):", font=LABEL_FONT, bg=PANEL).pack(side="left", padx=(0, 6))
        tk.Entry(row0, textvariable=self.v_export_month, width=8, font=ENTRY_FONT).pack(side="left")
        tk.Label(row0, text=" (leave 'ALL' to export all 12 months as attributes)",
                 font=SMALL_FONT, fg="#666", bg=PANEL).pack(side="left", padx=6)

        BrowseRow(
            s1, "Output Shapefile (.shp)", self.v_export_shp, 1, mode="savefile",
            filetypes=[("Shapefile", "*.shp")],
            tip="Path to write the output shapefile for ArcGIS"
        )

        self._export_btn = tk.Button(
            outer,
            text="📂  Export Shapefile",
            command=self._start_export,
            font=("Segoe UI", 10, "bold"), bg=ACCENT, fg="white", relief="flat", padx=18, pady=8
        )
        self._export_btn.pack(anchor="w", padx=14, pady=12)

        self._export_status = tk.Label(outer, text="", font=SMALL_FONT, bg=PANEL, fg=ACCENT2)
        self._export_status.pack(anchor="w", padx=14)

    # ── Auto-detect helpers ──────────────────────────────────────────────────

    def _autodetect_paths(self):
        base = Path(self.v_base_dir.get().strip())
        if not base.exists():
            messagebox.showerror("Error", f"Base directory not found:\n{base}")
            return
        changed = []

        # Basin shapefile
        for cand in [
            "afinch_matlab_source/input_data/basin/TWDB_MRBs_2014.shp",
            "afinch_matlab_source/input_data/basin/*.shp",
            "inputData/river_basin/TWDB_MRBs_2014.shp",
            "inputData/river_basin/*.shp",
        ]:
            matches = list(base.glob(cand)) if "*" in cand else ([base / cand] if (base / cand).exists() else [])
            if matches:
                self.v_basin_shp.set(str(matches[0]))
                changed.append("Basin shapefile")
                break

        # NHD
        for cand in [
            "afinch_matlab_source/input_data/nhd/_extracted_gdb",
            "afinch_matlab_source/input_data/nhd",
            "inputData/texas_nhdplusgrb/_extracted_gdb",
            "inputData/texas_nhdplusgrb",
            "inputData/nhd_medium_res_gdb",
            "inputData/nhd",
        ]:
            p = base / cand
            if p.exists():
                self.v_nhd_dir.set(str(p))
                changed.append("NHD directory")
                break

        # NLCD
        nlcd_candidates = list(base.glob("afinch_matlab_source/input_data/**/*.tif")) + list(base.glob("inputData/**/*.tif"))
        for cand in sorted(nlcd_candidates):
            if "nlcd" in cand.name.lower() or "landcov" in cand.name.lower():
                self.v_nlcd_raster.set(str(cand))
                changed.append("NLCD raster")
                break

        # PRISM
        self._autodetect_prism()

        # Gages
        for cand in [
            "afinch_matlab_source/input_data/gages/monthly_wide_cfs.csv",
            "afinch_matlab_source/input_data/gages/monthly_wide_acft.csv",
            "afinch_matlab_source/input_data/gages/Brazos_Colocated.csv",
            "inputData/inputs/monthly_wide_cfs.csv",
            "inputData/inputs/monthly_wide_acft.csv",
            "inputData/inputs/Brazos_Colocated.csv",
        ]:
            p = base / cand
            if p.exists():
                self.v_gages_csv.set(str(p))
                changed.append("Gages CSV")
                break

        if changed:
            self._log(f"Auto-detected: {', '.join(changed)}\n", "ok")
        else:
            self._log("Auto-detect: no matching files found. Set paths manually.\n", "warn")

    def _autodetect_prism(self):
        base = Path(self.v_base_dir.get().strip())

        prism_roots = [
            base / "afinch_matlab_source" / "input_data" / "prism",
            base / "inputData" / "prism_monthly",
        ]

        def pick_dir(element: str) -> str | None:
            # Prefer clipped (Texas-only, smaller), then extracted (full US)
            for root in prism_roots:
                for sub in ["clipped", "extracted"]:
                    d = root / element / sub
                    if d.exists() and list(d.glob("*.tif")):
                        return str(d)
            return None

        ppt = pick_dir("ppt")
        tmean = pick_dir("tmean")

        if ppt:
            self.v_prism_ppt_dir.set(ppt)
            self._log(f"Auto-detected PRISM ppt: {ppt}\n", "ok")
        if tmean:
            self.v_prism_tmean_dir.set(tmean)
            self._log(f"Auto-detected PRISM tmean: {tmean}\n", "ok")
        if not ppt and not tmean:
            self._log("Could not auto-detect PRISM paths. Set manually on the Climate Data tab.\n", "warn")

    def _load_basin_names(self):
        shp = self.v_basin_shp.get().strip()
        if not shp or not Path(shp).exists():
            messagebox.showerror("Missing", "Set and verify the basin shapefile path first.")
            return
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shp)
            field = self.v_basin_field.get().strip()
            if field not in gdf.columns:
                # Try to find a string column named 'basin_name', 'name', etc.
                str_cols = list(gdf.select_dtypes(include=["object", "string"]).columns)
                for cand in ["basin_name", "basin", "name", "riverbasin"]:
                    for col in str_cols:
                        if col.lower() == cand:
                            field = col
                            self.v_basin_field.set(field)
                            break
                    if field in gdf.columns:
                        break
                if field not in gdf.columns and str_cols:
                    field = str_cols[0]
                    self.v_basin_field.set(field)

            values = sorted(str(v) for v in gdf[field].dropna().unique())
            menu = self._basin_menu["menu"]
            menu.delete(0, "end")
            for v in values:
                menu.add_command(label=v, command=lambda x=v: self.v_basin_value.set(x))
            if values:
                self.v_basin_value.set(values[0])
            self._log(f"Loaded {len(values)} basin name(s) from '{field}' column.\n", "ok")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ── Build Network ────────────────────────────────────────────────────────

    def _start_build_network(self, dry_run: bool):
        if self._is_running:
            messagebox.showwarning("Busy", "Another operation is in progress.")
            return
        try:
            cfg = self._validate_build_cfg()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        tag = "Dry Run" if dry_run else "Build Network"
        self._log(f"\n=== {tag.upper()} START ===\n", "ok")
        self._set_running(True)
        self._build_status.set("Building…" if not dry_run else "Validating…")
        self._build_status_lbl.configure(fg=WARN)

        def worker():
            try:
                self._do_build_network(cfg, dry_run=dry_run)
                if not dry_run:
                    self._do_build_upstream_gaged(cfg)
                status_msg = "✓ Build complete" if not dry_run else "✓ Dry run passed"
                self._log_q.put(("status", ("build_ok", status_msg)))
            except Exception as exc:
                self._log(f"ERROR: {exc}\n", "err")
                self._log_q.put(("status", ("build_err", f"✗ Error: {exc}")))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _do_build_network(self, cfg: dict, dry_run: bool):
        base = cfg["base_dir"]
        builder_path = base / "afinch_python_modules" / "build_brazos_basin_network.py"
        if not builder_path.exists():
            raise FileNotFoundError(f"Builder script not found:\n{builder_path}")

        cmd = [
            sys.executable, str(builder_path),
            "--base-dir",   str(base),
            "--ths",        cfg["ths"],
            "--hsr",        cfg["hsr"],
            "--basin-shp",  cfg["basin_shp_rel"],
            "--basin-field", cfg["basin_field"],
            "--basin-value", cfg["basin_value"],
            "--gdb-root",   cfg["nhd_rel"],
            "--wy",         str(cfg["build_wy"]),
            "--nlcd-raster", cfg["nlcd_rel"],
            "--prism-ppt-dir",   cfg["prism_ppt_rel"],
            "--prism-tmean-dir", cfg["prism_tmean_rel"],
        ]
        if cfg.get("hu4s"):
            cmd += ["--hu4s", cfg["hu4s"]]
        if cfg.get("gages_csv"):
            cmd += ["--gages-csv", cfg["gages_csv"]]
        if cfg.get("wam_csv"):
            cmd += ["--wam-csv", cfg["wam_csv"]]
        if not dry_run:
            cmd.append("--apply")

        self._log(f"Command:\n  {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=str(base),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Builder exited with code {rc}")

    def _do_build_upstream_gaged(self, cfg: dict):
        """Optional legacy step: generate upstream contributing catchments for USGS gages."""
        base = cfg["base_dir"]
        script_path = base / "afinch_python_modules" / "build_usgs_upstream_spatial.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Upstream mapping script not found:\n{script_path}")

        catchment_rel = f"inputData/NHDPlusCatchment_{cfg['ths']}.gpkg"
        catchment_path = base / catchment_rel
        if not catchment_path.exists():
            self._log(
                f"Skipping legacy upstream gaged-catchment build: catchment file not found ({catchment_path}).\n",
                "warn",
            )
            return

        catchments = gpd.read_file(catchment_path)
        geom_types = set(catchments.geometry.geom_type.dropna().astype(str))
        polygon_types = {"Polygon", "MultiPolygon"}
        if geom_types and not geom_types.issubset(polygon_types):
            self._log(
                "Skipping legacy upstream gaged-catchment build: "
                f"{catchment_path.name} contains {sorted(geom_types)} geometry, but the legacy script requires polygon catchments.\n",
                "warn",
            )
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--base-dir",
            str(base),
            "--hsr",
            cfg["hsr"],
            "--wy",
            str(cfg["build_wy"]),
            "--catchment-gpkg",
            catchment_rel,
            "--apply",
        ]

        self._log("\nRunning legacy upstream gaged-catchment build...\n", "ok")
        self._log(f"Command:\n  {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(base),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Upstream gaged-catchment builder exited with code {rc}")

    def _validate_build_cfg(self) -> dict:
        base = Path(self.v_base_dir.get().strip())
        if not base.exists():
            raise FileNotFoundError(f"Base directory not found: {base}")

        basin_shp = Path(self.v_basin_shp.get().strip())
        if not basin_shp.exists():
            raise FileNotFoundError(f"Basin shapefile not found: {basin_shp}")

        nhd = Path(self.v_nhd_dir.get().strip())
        if not nhd.exists():
            raise FileNotFoundError(f"NHD directory not found: {nhd}")

        nlcd = Path(self.v_nlcd_raster.get().strip())
        if not nlcd.exists():
            raise FileNotFoundError(f"NLCD raster not found: {nlcd}")

        ppt  = Path(self.v_prism_ppt_dir.get().strip())
        if not ppt.exists():
            raise FileNotFoundError(f"PRISM ppt directory not found: {ppt}")

        tmean = Path(self.v_prism_tmean_dir.get().strip())
        if not tmean.exists():
            raise FileNotFoundError(f"PRISM tmean directory not found: {tmean}")

        def rel(p: Path) -> str:
            try:
                return str(p.relative_to(base))
            except ValueError:
                return str(p)

        hu4s = self.v_hu4_codes.get().strip().replace(" ", "")
        if hu4s:
            hu4_parts = [p for p in hu4s.split(",") if p]
            if not hu4_parts:
                raise ValueError("HU4 selection is empty after parsing.")
            bad = [p for p in hu4_parts if not (len(p) == 4 and p.isdigit())]
            if bad:
                raise ValueError(f"Invalid HU4 code(s): {', '.join(bad)}. Use 4-digit values like 1206.")
            hu4s = ",".join(hu4_parts)

        return {
            "base_dir":       base,
            "ths":            self.v_ths_code.get().strip(),
            "hsr":            self.v_hsr_name.get().strip(),
            "hu4s":           hu4s,
            "basin_shp_rel":  rel(basin_shp),
            "basin_field":    self.v_basin_field.get().strip(),
            "basin_value":    self.v_basin_value.get().strip(),
            "nhd_rel":        rel(nhd),
            "nlcd_rel":       rel(nlcd),
            "prism_ppt_rel":  rel(ppt),
            "prism_tmean_rel": rel(tmean),
            "build_wy":       int(self.v_build_wy.get().strip()),
            "gages_csv":      self.v_gages_csv.get().strip(),
            "wam_csv":        "NONE" if self.v_usgs_only.get() else self.v_wam_csv.get().strip(),
        }

    def _apply_hu4_to_ids(self):
        hu4s = self.v_hu4_codes.get().strip().replace(" ", "")
        if not hu4s:
            messagebox.showinfo("HU4", "Enter a 4-digit HU4 code first (e.g., 1206).")
            return
        parts = [p for p in hu4s.split(",") if p]
        if len(parts) != 1:
            messagebox.showinfo(
                "HU4",
                "Auto-apply to THS/HSR supports one HU4 code at a time.\n"
                "For multi-HU4 builds, keep THS/HSR as desired manually.",
            )
            return
        code = parts[0]
        if not (len(code) == 4 and code.isdigit()):
            messagebox.showerror("HU4", f"Invalid HU4 code: {code}. Use 4 digits (e.g., 1206).")
            return
        self.v_ths_code.set(code)
        self.v_hsr_name.set(f"HSR{code}")
        self._log(f"Applied HU4 {code} -> THS={code}, HSR=HSR{code}\n", "ok")

    # ── Run Model ────────────────────────────────────────────────────────────

    def _validate_run_cfg(self) -> dict:
        base = Path(self.v_base_dir.get().strip())
        if not base.exists():
            raise FileNotFoundError(f"Base directory not found: {base}")
        wy_start = int(self.v_run_wy_start.get().strip())
        wy_end   = int(self.v_run_wy_end.get().strip())
        if wy_end < wy_start:
            raise ValueError("End WY must be ≥ Start WY")
        return {
            "base_dir": base,
            "ths":      self.v_ths_code.get().strip(),
            "hsr_key":  self.v_hsr_name.get().strip(),
            "wy_start": wy_start,
            "ny":       wy_end - wy_start + 1,
        }

    def _load_pipeline(self, cfg: dict, force_new: bool = False):
        sig = (str(cfg["base_dir"]), cfg["ths"], cfg["hsr_key"], cfg["wy_start"], cfg["ny"])
        if not force_new and self._pipeline is not None and self._pipeline_sig == sig:
            return

        runner = cfg["base_dir"] / "afinch_python_modules" / "run_converted_afinch_full_model.py"
        if not runner.exists():
            raise FileNotFoundError(f"Runner not found: {runner}")

        self._log(f"Loading pipeline from: {runner}\n")
        mod = self._load_module("afinch_gui_runner", runner)
        # Use pipeline defaults for regression calibration years (typically multi-year),
        # then fall back to the modeled run window if defaults are unavailable.
        reg_wy_start = int(getattr(mod, "WY1_REG", cfg["wy_start"]))
        reg_ny = int(getattr(mod, "NY_REG", cfg["ny"]))
        self._pipeline = mod.ConvertedAFinchPipeline(
            base_dir=cfg["base_dir"],
            src_dir=cfg["base_dir"] / "afinch_matlab_source",
            ths=cfg["ths"],
            hsr_key=cfg["hsr_key"],
            wy1=cfg["wy_start"],
            ny=cfg["ny"],
            logger=self._log,
            wy1_reg=reg_wy_start,
            ny_reg=reg_ny,
        )
        self._log(
            f"Regression window: WY{reg_wy_start}-{reg_wy_start + reg_ny - 1} "
            f"(pipeline default calibration window)\n"
        )
        self._pipeline_sig = sig
        self._completed_steps.clear()
        for ind in self._step_indicators.values():
            ind.set_idle()
        self._log("Pipeline initialized.\n", "ok")

    def _run_step(self, cfg: dict, step_no: int):
        dep = STEP_DEPS.get(step_no)
        if dep is not None and dep not in self._completed_steps:
            raise RuntimeError(f"Step {dep} must be completed before Step {step_no}")

        ind = self._step_indicators.get(step_no)
        if ind:
            self.root.after(0, ind.set_running)

        try:
            if step_no == 1:
                self._load_pipeline(cfg, force_new=True)
                self._pipeline.step_setup_inputs()
            elif step_no == 2:
                self._pipeline.step_run_regression()
            elif step_no == 3:
                self._pipeline.step_estimate_incremental()
            elif step_no == 4:
                self._pipeline.step_constrain_incremental()
            elif step_no == 5:
                self._pipeline.step_write_incremental_output()
            elif step_no == 6:
                self._pipeline.step_accumulate_flow()
            else:
                raise ValueError(f"Unknown step {step_no}")
        except Exception:
            if ind:
                self.root.after(0, ind.set_error)
            raise

        if step_no == 1:
            self._completed_steps = {1}
        else:
            self._completed_steps.add(step_no)

        if ind:
            self.root.after(0, ind.set_done)

    def _start_single_step(self, step_no: int):
        if self._is_running:
            messagebox.showwarning("Busy", "Another operation is in progress.")
            return
        try:
            cfg = self._validate_run_cfg()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._log(f"\n=== STEP {step_no}: {STEP_LABELS[step_no].upper()} ===\n")
        self._set_running(True)
        self._status_var.set(f"Running step {step_no}…")
        self._progress.start(10)

        def worker():
            try:
                self._run_step(cfg, step_no)
                self._log(f"Step {step_no} complete.\n", "ok")
                done = ", ".join(str(s) for s in sorted(self._completed_steps))
                self._log(f"Completed steps: {done}\n")
                self.root.after(0, lambda: self._status_var.set(f"Step {step_no} complete"))
            except Exception as exc:
                self._log(f"ERROR in step {step_no}: {exc}\n", "err")
                self._log(traceback.format_exc() + "\n", "err")
                self.root.after(0, lambda: self._status_var.set(f"Error in step {step_no}"))
            finally:
                self.root.after(0, self._progress.stop)
                self.root.after(0, lambda: self._set_running(False))

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _start_full_run(self):
        if self._is_running:
            messagebox.showwarning("Busy", "Another operation is in progress.")
            return
        try:
            cfg = self._validate_run_cfg()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._log("\n=== FULL MODEL RUN (Steps 1–6) ===\n", "ok")
        self._set_running(True)
        self._status_var.set("Running full model…")
        self._progress.start(10)

        def worker():
            try:
                for step_no in range(1, 7):
                    self._log(f"\n--- Step {step_no}: {STEP_LABELS[step_no]} ---\n")
                    self.root.after(0, lambda s=step_no: self._status_var.set(f"Running step {s}…"))
                    self._run_step(cfg, step_no)
                self._log("=== FULL MODEL RUN COMPLETE ===\n", "ok")
                self.root.after(0, lambda: self._status_var.set("Full run complete  ✓"))
            except Exception as exc:
                self._log(f"FAILED: {exc}\n", "err")
                self._log(traceback.format_exc() + "\n", "err")
                self.root.after(0, lambda: self._status_var.set("Run failed — see log"))
            finally:
                self.root.after(0, self._progress.stop)
                self.root.after(0, lambda: self._set_running(False))

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _reset_steps(self):
        if self._is_running:
            messagebox.showwarning("Busy", "Cannot reset while running.")
            return
        self._pipeline = None
        self._pipeline_sig = None
        self._completed_steps.clear()
        for ind in self._step_indicators.values():
            ind.set_idle()
        self._status_var.set("Steps reset. Run Step 1 to begin.")
        self._log("Step state reset.\n")

    # ── Export Shapefile (Step 7) ─────────────────────────────────────────────

    def _start_export(self):
        if self._is_running:
            messagebox.showwarning("Busy", "Another operation is in progress.")
            return
        if 6 not in self._completed_steps:
            if not messagebox.askyesno("Step 6 not run", "Step 6 (Accumulate Flow) hasn't been run this session.\nExport will use the most recent HSR output files. Continue?"):
                return

        base = Path(self.v_base_dir.get().strip())
        hsr_key = self.v_hsr_name.get().strip()
        try:
            wy = int(self.v_export_wy.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Water Year must be an integer.")
            return

        month_str = self.v_export_month.get().strip().upper()
        months_to_export = list(range(1, 13)) if month_str == "ALL" else [int(month_str)]

        out_shp = self.v_export_shp.get().strip()
        if not out_shp:
            # Auto-generate path
            hsr_out = base / hsr_key / "Output"
            hsr_out.mkdir(parents=True, exist_ok=True)
            out_shp = str(hsr_out / f"accumulated_flow_WY{wy}.shp")
            self.v_export_shp.set(out_shp)

        self._log(f"\n=== STEP 7: EXPORT SHAPEFILE ===\n", "ok")
        self._set_running(True)

        def worker():
            try:
                self._do_export_shapefile(base, hsr_key, wy, months_to_export, Path(out_shp))
                msg = f"Shapefile written:\n{out_shp}"
                self._log(f"✓ {msg}\n", "ok")
                self.root.after(0, lambda: self._export_status.configure(text=f"✓ Written: {out_shp}"))
            except Exception as exc:
                self._log(f"ERROR: {exc}\n", "err")
                self.root.after(0, lambda: self._export_status.configure(text=f"✗ Error: {exc}"))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _do_export_shapefile(self, base: Path, hsr_key: str, wy: int, months: list[int], out_shp: Path):
        try:
            import geopandas as gpd
            import pandas as pd
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(f"Missing package: {exc}.")

        ths = self.v_ths_code.get().strip()
        hsr_dir = base / hsr_key

        # ── 1. Load accumulated flow data from HSR output ──────────────────────
        # Try the QY output file from step 6 (AFConFlowAccum writes QY*.dat files)
        flow_file = hsr_dir / "Output" / f"QYConWY{wy}.dat"
        if not flow_file.exists():
            # Fallback: look for any QY dat in output
            candidates = sorted((hsr_dir / "Output").glob(f"QYCon*{wy}*.dat")) if (hsr_dir / "Output").exists() else []
            if not candidates:
                raise FileNotFoundError(
                    f"Accumulated flow file not found: {flow_file}\n"
                    f"Run Steps 1-6 first, or check that WY{wy} data exists in {hsr_dir / 'Output'}"
                )
            flow_file = candidates[0]
            self._log(f"Using accumulated flow file: {flow_file}\n")

        # Read flow data — expect columns: ComID, M01..M12 (CFS)
        self._log(f"Reading flow data from {flow_file}\n")
        try:
            flow_df = pd.read_csv(flow_file, sep=r"\s+", comment="#")
        except Exception:
            flow_df = pd.read_csv(flow_file)

        # Identify ComID column (first column) and month columns
        cols = list(flow_df.columns)
        comid_col = cols[0]
        flow_df[comid_col] = pd.to_numeric(flow_df[comid_col], errors="coerce").astype("Int64")
        flow_df = flow_df.dropna(subset=[comid_col]).copy()

        # Build month column mapping: 12 data columns after ComID
        month_data_cols = [c for c in cols[1:] if c != comid_col][:12]
        cfs_month_map: dict[int, str] = {}
        for i, col in enumerate(month_data_cols):
            mo = i + 1  # WY month 1=Oct … 12=Sep
            cfs_month_map[mo] = col

        # ── 2. Load NHD flowline geometry from GagedCatchments or GIS ──────────
        catchment_gpkg = base / "inputData" / f"NHDPlusCatchment_{ths}.gpkg"
        flowline_shp   = hsr_dir / "Flowlines" / "nhdflowline.txt"

        if catchment_gpkg.exists():
            self._log(f"Loading reach geometry from {catchment_gpkg}\n")
            geom_gdf = gpd.read_file(catchment_gpkg)
            comid_field = "NHDPlusID" if "NHDPlusID" in geom_gdf.columns else geom_gdf.columns[0]
            geom_gdf = geom_gdf.rename(columns={comid_field: "ComID"})
        else:
            # Try to load geometries from NHD source
            self._log("Catchment GeoPackage not found. Attempting to load from NHD source...\n")
            nhd_dir = Path(self.v_nhd_dir.get().strip())
            if not nhd_dir.exists():
                raise FileNotFoundError(
                    f"Neither {catchment_gpkg} nor NHD directory found. "
                    "Run 'Build Network' first."
                )
            # Load reach table to get ComIDs, build point-geometry fallback
            reach_tbl = pd.read_csv(flowline_shp)
            nhdflowline_col = "ComID" if "ComID" in reach_tbl.columns else reach_tbl.columns[0]
            comids = reach_tbl[nhdflowline_col].dropna().astype("int64").tolist()
            self._log(f"⚠ No geometry available for {len(comids)} reaches. Shapefile will be geometry-only if NHD load fails.\n", "warn")
            # Load NHDFlowline geometries from GDB directories
            gdbs = sorted(nhd_dir.rglob("*.gdb"))
            if not gdbs:
                raise FileNotFoundError(f"No .gdb files found under {nhd_dir}")
            parts = []
            for gdb in gdbs:
                try:
                    fl = gpd.read_file(gdb, layer="NHDFlowline")
                    fl = fl.rename(columns={c: c.lower() for c in fl.columns})
                    if "permanent_identifier" in fl.columns:
                        fl["ComID"] = pd.to_numeric(fl["permanent_identifier"], errors="coerce")
                    fl = fl[fl["ComID"].isin(comids)].copy()
                    if not fl.empty:
                        parts.append(fl[["ComID", "geometry"]])
                except Exception:
                    continue
            if not parts:
                raise RuntimeError("Could not load any NHD flowline geometries matching your basin.")
            geom_gdf = pd.concat(parts, ignore_index=True)
            geom_gdf["ComID"] = geom_gdf["ComID"].astype("int64")
            geom_gdf = gpd.GeoDataFrame(geom_gdf, geometry="geometry", crs="EPSG:4269")

        geom_gdf["ComID"] = pd.to_numeric(geom_gdf["ComID"], errors="coerce").astype("Int64")
        geom_gdf = geom_gdf.dropna(subset=["ComID"])

        # ── 3. Join flow data to geometry ──────────────────────────────────────
        self._log(f"Joining flow data to {len(geom_gdf)} geometries…\n")

        # WY month → calendar month label mapping (Oct=1 → Oct, …, Sep=12 → Sep)
        WY_MONTH_NAMES = {
            1: "Oct", 2: "Nov", 3: "Dec",
            4: "Jan", 5: "Feb", 6: "Mar",
            7: "Apr", 8: "May", 9: "Jun",
            10: "Jul", 11: "Aug", 12: "Sep",
        }

        out_df = geom_gdf[["ComID", "geometry"]].copy()
        out_df["ComID"] = out_df["ComID"].astype("int64")

        for mo in months:
            if mo not in cfs_month_map:
                self._log(f"  ⚠ WY Month {mo} not found in flow data, skipping.\n", "warn")
                continue
            src_col = cfs_month_map[mo]
            mo_label = WY_MONTH_NAMES.get(mo, f"M{mo:02d}")
            col_name = f"{wy}_{mo_label}_CFS"[:10]  # shapefile field limit 10 chars

            mo_data = flow_df[[comid_col, src_col]].rename(columns={comid_col: "ComID", src_col: col_name})
            mo_data["ComID"] = mo_data["ComID"].astype("int64")
            out_df = out_df.merge(mo_data, on="ComID", how="left")

        out_gdf = gpd.GeoDataFrame(out_df, geometry="geometry", crs=geom_gdf.crs)

        # Reproject to standard geographic CRS for ArcGIS compatibility
        if out_gdf.crs is not None and not out_gdf.crs.is_geographic:
            out_gdf = out_gdf.to_crs("EPSG:4326")

        # ── 4. Write shapefile ─────────────────────────────────────────────────
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"Writing shapefile: {out_shp}\n")
        out_gdf.to_file(str(out_shp), driver="ESRI Shapefile")
        n_rows = len(out_gdf)
        n_mo = len([c for c in out_gdf.columns if "CFS" in c])
        self._log(
            f"Done. {n_rows:,} reaches, {n_mo} month column(s). "
            f"File: {out_shp}\n"
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = ""):
        self._log_q.put(("log", (msg, tag)))

    def _append_log(self, msg: str, tag: str = ""):
        if self.log_box is None:
            return
        if tag:
            self.log_box.insert(tk.END, msg, tag)
        else:
            self.log_box.insert(tk.END, msg)
        self.log_box.see(tk.END)

    def _drain_log(self):
        try:
            while True:
                typ, payload = self._log_q.get_nowait()
                if typ == "log":
                    msg, tag = payload
                    self._append_log(msg, tag)
                elif typ == "status":
                    key, val = payload
                    if key == "build_ok":
                        self._build_status.set(val)
                        self._build_status_lbl.configure(fg=ACCENT2)
                    elif key == "build_err":
                        self._build_status.set(val)
                        self._build_status_lbl.configure(fg=ERROR)
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._drain_log)

    # ── Runtime helpers ───────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self._is_running = running
        state = tk.DISABLED if running else tk.NORMAL
        for btn in [
            self._run_all_btn, self._reset_btn,
            self._dry_run_btn, self._build_btn, self._export_btn,
        ]:
            if btn:
                btn.configure(state=state)
        for btn in self._step_btns.values():
            btn.configure(state=state)

    def _load_module(self, name: str, path: Path):
        loader = SourceFileLoader(name, str(path))
        spec = spec_from_loader(name, loader)
        if spec is None:
            raise RuntimeError(f"No module spec for {path}")
        mod = module_from_spec(spec)
        sys.modules[name] = mod
        loader.exec_module(mod)
        return mod

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AFinchComprehensiveGUI()
    app.run()
