import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import font as tkfont
from pathlib import Path
from typing import Optional
import zipfile

import geopandas as gpd
from PIL import Image, ImageTk, ImageDraw, ImageFont

from brazos_streamflow_model import ModelConfig, run_model


class ToolTip:
    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            bg="#143f5f",
            fg="white",
            padx=10,
            pady=6,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        lbl.pack()

    def _hide(self, _event=None) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


def prepare_flowline_source(flowline_source: str) -> str:
    """
    If a directory contains Texas NHDPlusHR zip files, extract them and
    return a directory containing .gdb folders. Otherwise return input.
    """
    src = Path(flowline_source)
    if not src.is_dir():
        return str(src)

    zips = sorted(src.glob("NHDPLUS_H_*_HU4_GDB.zip"))
    if not zips:
        return str(src)

    extract_root = src / "_extracted_gdb"
    extract_root.mkdir(parents=True, exist_ok=True)

    bad_zips = []
    for z in zips:
        stem = z.stem  # e.g., NHDPLUS_H_1101_HU4_GDB
        target_dir = extract_root / stem
        # Skip if already extracted and contains a gdb.
        if target_dir.exists() and list(target_dir.glob("*.gdb")):
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(target_dir)
        except zipfile.BadZipFile:
            bad_zips.append(z.name)
            continue

    gdbs = list(extract_root.rglob("*.gdb"))
    if not gdbs:
        if bad_zips:
            raise ValueError(
                "No valid GDB extracted. Corrupt zip file(s): "
                + ", ".join(bad_zips)
            )
        raise ValueError("No .gdb folders found after extraction.")

    if bad_zips:
        print(f"Warning: skipped corrupt zip(s): {', '.join(bad_zips)}")
    return str(extract_root)


class WAMApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Regional Streamflow Model")
        self.root.geometry("1100x760")
        self.base_width = 1100
        self.base_height = 760
        self.canvas_width = self.base_width
        self.canvas_height = self.base_height

        self.flowline_var = tk.StringVar(value="texas_nhdplusgrb")
        self.flowline_layer_var = tk.StringVar(value="NHDFlowline")
        self.vaa_var = tk.StringVar(value="")
        self.cp_meta_var = tk.StringVar(value="Brazos_Metadata.csv")
        self.flo_var = tk.StringVar(value="brazos_final_acft.flo")
        self.gage_crosswalk_var = tk.StringVar(value="")
        self.basin_var = tk.StringVar(value="river_basin/TWDB_MRBs_2014.shp")
        self.basin_name_var = tk.StringVar()
        self.basin_gdf = None
        self.basin_name_field: Optional[str] = None

        self.start_year_var = tk.StringVar(value="2010")
        self.end_year_var = tk.StringVar(value="2024")
        self.output_var = tk.StringVar(value="modeled_monthly_comid_flows.csv")
        self.output_shp_var = tk.StringVar(value="modeled_monthly_comid_flows.shp")
        self.output_shp_date_var = tk.StringVar(value="2024-12-01")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready")

        self._load_assets()
        self._set_cursor()

        self._build_ui()
        if Path(self.basin_var.get()).exists():
            self._load_basin_options(self.basin_var.get())

    def _load_assets(self) -> None:
        bg_candidates = [
            Path("backgrounds") / "river_topview.jpg",
            Path("backgrounds") / "river_topview.avif",
            Path("backgrounds") / "river.jpg",
            Path("backgrounds") / "riverfront.jpg",
            Path("backgrounds") / "river2.jpg",
        ]
        processing_candidates = [
            Path("backgrounds") / "river2.jpg",
            Path("backgrounds") / "river.jpg",
        ]

        self.bg_photo = None
        self.bg_image = None
        self.bg_source_image = None
        self.processing_photo = None

        for bg_path in bg_candidates:
            if not bg_path.exists():
                continue
            try:
                src_img = Image.open(bg_path).convert("RGB")
                bg_img = src_img.resize((self.base_width, self.base_height))
                self.bg_source_image = src_img
                self.bg_image = bg_img
                self.bg_photo = ImageTk.PhotoImage(bg_img)
                break
            except Exception:
                continue

        for processing_path in processing_candidates:
            if not processing_path.exists():
                continue
            try:
                proc_img = Image.open(processing_path).resize((150, 150))
                self.processing_photo = ImageTk.PhotoImage(proc_img)
                break
            except Exception:
                continue

    def _set_cursor(self) -> None:
        # Try a water-themed cursor first, then fallback safely.
        for cur in ("boat", "hand2", "arrow"):
            try:
                self.root.configure(cursor=cur)
                return
            except Exception:
                continue

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(
            self.root,
            width=self.base_width,
            height=self.base_height,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.bg_canvas_item = None
        if self.bg_photo is not None:
            self.bg_canvas_item = self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        try:
            title_f = ImageFont.truetype("comic.ttf", 38)
        except Exception:
            title_f = ImageFont.load_default()
        try:
            label_f = ImageFont.truetype("comic.ttf", 22)
        except Exception:
            label_f = ImageFont.load_default()

        # Draw title + left labels directly, so no widget background rectangles.
        _ = ImageDraw.Draw(self.bg_image.copy()) if self.bg_image is not None else None
        self.canvas.create_text(552, 36, text="Regional Streamflow Model", fill="#0b2d42", font=("Comic Sans MS", 24, "bold"))
        self.canvas.create_text(550, 34, text="Regional Streamflow Model", fill="#eaf8ff", font=("Comic Sans MS", 24, "bold"))

        labels = [
            "Flowline source (dir/file/gdb):",
            "Flowline layer (for GDB):",
            "VAA file (optional):",
            "CP metadata CSV:",
            "FLO file:",
            "Gage crosswalk CSV (optional):",
            "Basin shapefile:",
            "Basin name:",
            "Start year:",
            "End year:",
            "Output CSV:",
            "Output shapefile (optional):",
            "Shapefile month (YYYY-MM-01):",
        ]

        x_label = 110
        x_entry = 430
        x_button = 960
        center_x = 550
        y_start = 100
        row_h = 44

        for i, txt in enumerate(labels):
            y = y_start + i * row_h
            self.canvas.create_text(
                x_label + 2,
                y + 2,
                text=txt,
                fill="#0b2d42",
                anchor="w",
                font=("Comic Sans MS", 14, "bold"),
            )
            self.canvas.create_text(
                x_label,
                y,
                text=txt,
                fill="#f1fbff",
                anchor="w",
                font=("Comic Sans MS", 14, "bold"),
            )

        entry_font = tkfont.Font(family="Segoe UI", size=10)
        button_font = tkfont.Font(family="Comic Sans MS", size=10, weight="bold")

        def add_entry_row(index: int, var: tk.StringVar, width: int, browse_cmd=None, tip: str = ""):
            y = y_start + index * row_h
            entry = tk.Entry(self.canvas, textvariable=var, width=width, font=entry_font)
            self.canvas.create_window(x_entry, y, anchor="w", window=entry)
            if browse_cmd is not None:
                btn = tk.Button(
                    self.canvas,
                    text="Browse",
                    command=browse_cmd,
                    bg="#86c5ff",
                    fg="white",
                    font=button_font,
                    activebackground="#68b3f7",
                    activeforeground="white",
                )
                self.canvas.create_window(x_button, y, anchor="w", window=btn)
            if tip:
                ToolTip(entry, tip)
            return entry

        flowline_entry = add_entry_row(
            0,
            self.flowline_var,
            54,
            browse_cmd=self._browse_flowline,
            tip="Folder with NHDPlus HU4 zips, or a .gdb/.shp flowline source.",
        )
        ToolTip(flowline_entry, "Folder with NHDPlus HU4 zips, or a .gdb/.shp flowline source.")
        _ = add_entry_row(1, self.flowline_layer_var, 30, tip="Layer name inside GDB. Usually NHDFlowline.")
        _ = add_entry_row(2, self.vaa_var, 54, browse_cmd=self._browse_vaa, tip="Optional external VAA table (.dbf). Leave blank for HU4 GDB internal VAA.")
        _ = add_entry_row(3, self.cp_meta_var, 54, browse_cmd=self._browse_cp, tip="CSV with UP_CP, Next_DS_CP, ChLosFac, LAT, LONG.")
        _ = add_entry_row(4, self.flo_var, 54, browse_cmd=self._browse_flo, tip="WRAP flow file (.flo): station year monthly acre-ft values.")
        _ = add_entry_row(5, self.gage_crosswalk_var, 54, browse_cmd=self._browse_xwalk, tip="Optional CSV mapping gage_id to COMID for gage enforcement/calibration.")
        _ = add_entry_row(6, self.basin_var, 54, browse_cmd=self._browse_basin, tip="Polygon shapefile used to clip the model domain (e.g., TWDB basins).")

        y = y_start + 7 * row_h
        self.basin_menu = tk.OptionMenu(self.canvas, self.basin_name_var, "")
        self.basin_menu.configure(state="disabled")
        self.canvas.create_window(x_entry, y, anchor="w", window=self.basin_menu)

        _ = add_entry_row(8, self.start_year_var, 12, tip="First model year (e.g., 2010).")
        _ = add_entry_row(9, self.end_year_var, 12, tip="Last model year (e.g., 2024).")
        _ = add_entry_row(10, self.output_var, 54, browse_cmd=self._browse_output, tip="Path for modeled monthly COMID flows CSV output.")
        _ = add_entry_row(11, self.output_shp_var, 54, browse_cmd=self._browse_output_shp, tip="Optional GIS shapefile output for one selected month.")
        _ = add_entry_row(12, self.output_shp_date_var, 20, tip="Month to export to shapefile, e.g. 2024-12-01.")

        run_btn = tk.Button(
            self.canvas,
            text="Run Simulation",
            command=self._run,
            bg="#58b7ff",
            fg="white",
            font=button_font,
            activebackground="#3fa7f3",
            activeforeground="white",
            padx=14,
            pady=5,
        )
        self.canvas.create_window(center_x, y_start + 13 * row_h + 6, anchor="center", window=run_btn)

        progress = ttk.Progressbar(self.canvas, variable=self.progress_var, orient="horizontal", mode="determinate", length=500)
        self.canvas.create_window(center_x, y_start + 14 * row_h + 6, anchor="center", window=progress)

        status_label = tk.Label(self.canvas, textvariable=self.status_var, fg="white", bg="#143f5f", font=("Comic Sans MS", 10, "bold"))
        self.canvas.create_window(center_x, y_start + 15 * row_h - 8, anchor="center", window=status_label)

        self.processing_label = tk.Label(self.canvas, bg="#143f5f")
        if self.processing_photo is not None:
            self.processing_label.configure(image=self.processing_photo)
        self.processing_window = self.canvas.create_window(center_x, y_start + 11 * row_h + 12, anchor="center", window=self.processing_label)
        self.canvas.itemconfigure(self.processing_window, state="hidden")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, event) -> None:
        if event.width <= 1 or event.height <= 1:
            return
        if event.width == self.canvas_width and event.height == self.canvas_height:
            return

        sx = event.width / self.canvas_width
        sy = event.height / self.canvas_height
        self.canvas.scale("all", 0, 0, sx, sy)
        self.canvas_width = event.width
        self.canvas_height = event.height

        if self.bg_source_image is not None and self.bg_canvas_item is not None:
            resized = self.bg_source_image.resize((event.width, event.height))
            self.bg_photo = ImageTk.PhotoImage(resized)
            self.canvas.itemconfigure(self.bg_canvas_item, image=self.bg_photo)

    def _browse_flowline(self) -> None:
        pick_dir = messagebox.askyesno(
            "Flowline source",
            "Click YES to choose a folder (e.g., texas_nhdplusgrb).\nClick NO to choose a file/gdb."
        )
        if pick_dir:
            path = filedialog.askdirectory(title="Select flowline source folder")
        else:
            path = filedialog.askopenfilename(
                title="Select flowline file or gdb",
                filetypes=[("All files", "*.*")],
            )
        if path:
            self.flowline_var.set(path)

    def _browse_vaa(self) -> None:
        path = filedialog.askopenfilename(title="Select optional VAA file", filetypes=[("DBF", "*.dbf"), ("All files", "*.*")])
        if path:
            self.vaa_var.set(path)

    def _browse_cp(self) -> None:
        path = filedialog.askopenfilename(title="Select CP metadata CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.cp_meta_var.set(path)

    def _browse_flo(self) -> None:
        path = filedialog.askopenfilename(title="Select FLO file", filetypes=[("FLO/CSV", "*.flo *.csv"), ("All files", "*.*")])
        if path:
            self.flo_var.set(path)

    def _browse_xwalk(self) -> None:
        path = filedialog.askopenfilename(title="Select gage crosswalk CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.gage_crosswalk_var.set(path)

    def _browse_basin(self) -> None:
        path = filedialog.askopenfilename(title="Select basin shapefile", filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")])
        if not path:
            return
        self.basin_var.set(path)
        self._load_basin_options(path)

    def _load_basin_options(self, path: str) -> None:
        basin_gdf = gpd.read_file(path)
        string_cols = list(basin_gdf.select_dtypes(include=["object", "string"]).columns)
        if not string_cols:
            raise ValueError("No text columns found in basin shapefile for basin names.")

        name_field = None
        for cand in ["basin_name", "basin", "name", "riverbasin"]:
            for col in string_cols:
                if col.lower() == cand:
                    name_field = col
                    break
            if name_field:
                break
        if name_field is None:
            name_field = string_cols[0]

        vals = sorted({str(v) for v in basin_gdf[name_field].dropna().unique()})
        if not vals:
            raise ValueError(f"No non-empty values found in basin field '{name_field}'.")

        self.basin_gdf = basin_gdf
        self.basin_name_field = name_field

        menu = self.basin_menu["menu"]
        menu.delete(0, "end")
        for v in vals:
            menu.add_command(label=v, command=lambda x=v: self.basin_name_var.set(x))
        self.basin_name_var.set("Brazos" if "Brazos" in vals else vals[0])
        self.basin_menu.configure(state="normal")

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output CSV as",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _browse_output_shp(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output shapefile as",
            defaultextension=".shp",
            filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")],
        )
        if path:
            self.output_shp_var.set(path)

    def _run(self) -> None:
        try:
            self.progress_var.set(0.0)
            self.status_var.set("Preparing inputs...")
            if hasattr(self, "processing_window"):
                self.canvas.itemconfigure(self.processing_window, state="normal")
            self.root.update_idletasks()

            flowline_raw = self.flowline_var.get().strip()
            cp_meta = self.cp_meta_var.get().strip()
            flo = self.flo_var.get().strip()
            basin = self.basin_var.get().strip()
            basin_name = self.basin_name_var.get().strip()
            output = self.output_var.get().strip()
            output_shp = self.output_shp_var.get().strip() or None

            if not flowline_raw or not cp_meta or not flo or not basin or not basin_name or not output:
                raise ValueError("Missing required input.")

            start_year = int(self.start_year_var.get().strip())
            end_year = int(self.end_year_var.get().strip())
            if end_year < start_year:
                raise ValueError("End year must be >= start year.")

            flowline_source = prepare_flowline_source(flowline_raw)
            vaa_file = self.vaa_var.get().strip() or None
            gage_crosswalk = self.gage_crosswalk_var.get().strip() or None

            cfg = ModelConfig(
                flowline_source=flowline_source,
                flowline_layer=self.flowline_layer_var.get().strip() or None,
                vaa_file=vaa_file,
                basin_shp=basin,
                basin_name_field=self.basin_name_field or "basin_name",
                basin_name_value=basin_name,
                cp_meta_file=cp_meta,
                flo_file=flo,
                output_file=output,
                output_shapefile=output_shp,
                shapefile_date=(self.output_shp_date_var.get().strip() or None),
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-01",
                gage_crosswalk_file=gage_crosswalk,
            )

            def on_progress(percent: float, stage: str) -> None:
                self.progress_var.set(percent)
                self.status_var.set(f"{stage} ({percent:.1f}%)")
                self.root.update_idletasks()

            run_model(cfg, progress_cb=on_progress)
            self.progress_var.set(100.0)
            self.status_var.set("Complete (100.0%)")
            if hasattr(self, "processing_window"):
                self.canvas.itemconfigure(self.processing_window, state="hidden")
            messagebox.showinfo("Done", f"Simulation complete.\nResults saved to:\n{output}")
        except Exception as exc:
            if hasattr(self, "processing_window"):
                self.canvas.itemconfigure(self.processing_window, state="hidden")
            self.status_var.set("Failed")
            messagebox.showerror("Error", f"Simulation failed:\n{exc}")


def main() -> None:
    root = tk.Tk()
    app = WAMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
