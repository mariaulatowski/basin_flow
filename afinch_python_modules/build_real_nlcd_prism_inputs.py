from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NLCD_CLASSES = [
    11, 12, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 61, 71, 81, 82, 83, 84, 85, 91, 92
]


@dataclass
class WyMonth:
    wy_month: int
    cal_year: int
    cal_month: int


def _water_year_months(wy: int) -> list[WyMonth]:
    months: list[WyMonth] = []
    # WY month 1..12 corresponds to Oct..Sep.
    for wy_month, (year, month) in enumerate(
        [
            (wy - 1, 10),
            (wy - 1, 11),
            (wy - 1, 12),
            (wy, 1),
            (wy, 2),
            (wy, 3),
            (wy, 4),
            (wy, 5),
            (wy, 6),
            (wy, 7),
            (wy, 8),
            (wy, 9),
        ],
        start=1,
    ):
        months.append(WyMonth(wy_month=wy_month, cal_year=year, cal_month=month))
    return months


def _require_geospatial_deps() -> tuple[object, object, object]:
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing required geospatial dependencies. Install: geopandas rasterio shapely pyproj fiona"
        ) from exc
    return gpd, rasterio, mask


def _read_gridcode_comid(xwalk_path: Path) -> pd.DataFrame:
    if not xwalk_path.exists():
        raise FileNotFoundError(f"Missing required xwalk file: {xwalk_path}")

    # Support both legacy whitespace-delimited and current comma-delimited formats.
    xwalk = pd.read_csv(
        xwalk_path,
        sep=r"[\s,]+",
        engine="python",
        comment="#",
        header=0,
        names=["GridCode", "ComID"],
        skip_blank_lines=True,
    )
    xwalk["GridCode"] = pd.to_numeric(xwalk["GridCode"], errors="coerce").astype("Int64")
    xwalk["ComID"] = pd.to_numeric(xwalk["ComID"], errors="coerce").astype("Int64")
    xwalk = xwalk.dropna().astype({"GridCode": "int64", "ComID": "int64"})
    xwalk = xwalk.drop_duplicates(subset=["ComID"]).sort_values("ComID").reset_index(drop=True)
    return xwalk


def _resolve_raster_path(pattern: str, year: int, month: int) -> Path:
    p = Path(pattern.format(yyyy=year, mm=month))
    if not p.exists():
        raise FileNotFoundError(f"Raster not found for year={year}, month={month}: {p}")
    return p


def _zonal_histogram(
    gdf: object,
    raster_path: Path,
    class_values: Iterable[int],
    rasterio_mod: object,
    mask_fn: object,
) -> np.ndarray:
    with rasterio_mod.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        out = np.zeros((len(gdf), len(class_values)), dtype=float)
        class_values = list(class_values)

        for i, geom in enumerate(gdf.geometry):
            arr, _ = mask_fn(src, [geom], crop=True, all_touched=False, filled=True)
            band = arr[0]

            nodata = src.nodata
            if nodata is None:
                valid = np.isfinite(band)
            else:
                valid = np.isfinite(band) & (band != nodata)

            vals = band[valid].astype(np.int32)
            if vals.size == 0:
                continue

            total = float(vals.size)
            for j, cls in enumerate(class_values):
                out[i, j] = 100.0 * float(np.count_nonzero(vals == cls)) / total

        return out


def _zonal_mean(
    gdf: object,
    raster_path: Path,
    rasterio_mod: object,
    mask_fn: object,
) -> np.ndarray:
    with rasterio_mod.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        out = np.full(len(gdf), np.nan, dtype=float)

        for i, geom in enumerate(gdf.geometry):
            arr, _ = mask_fn(src, [geom], crop=True, all_touched=False, filled=True)
            band = arr[0].astype(float)

            nodata = src.nodata
            if nodata is None:
                valid = np.isfinite(band)
            else:
                valid = np.isfinite(band) & (band != nodata)

            vals = band[valid]
            if vals.size > 0:
                out[i] = float(vals.mean())

        return out


def _write_nlcd_output(out_path: Path, table: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)


def _write_prism_prec_output(out_path: Path, table: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("PRISM precipitation\n")
        f.write("Built from user-provided rasters\n")
        f.write("GridCode GCAreaSqMi PIn_01..PIn_13\n")
        f.write("Units preserved from source raster\n")
        table.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")


def _write_prism_temp_output(out_path: Path, table: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("PRISM temperature\n")
        f.write("Built from user-provided rasters\n")
        f.write("GridCode TdC_01..TdC_12\n")
        f.write("Units preserved from source raster\n")
        table.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")


def build_inputs(
    root: Path,
    hsr: str,
    ths: str,
    wy: int,
    catchments_path: Path,
    catchments_comid_field: str,
    nlcd_raster: Path,
    prism_ppt_pattern: str,
    prism_tmean_pattern: str,
) -> None:
    gpd, rasterio_mod, mask_fn = _require_geospatial_deps()

    hsr_dir = root / f"HSR{hsr}"
    xwalk_path = hsr_dir / "Flowlines" / "GridCodeComID.txt"

    xwalk = _read_gridcode_comid(xwalk_path)

    catch = gpd.read_file(catchments_path)
    if catchments_comid_field not in catch.columns:
        raise ValueError(
            f"Catchments file does not contain COMID field '{catchments_comid_field}'. "
            f"Available columns: {list(catch.columns)}"
        )

    catch = catch[[catchments_comid_field, "geometry"]].rename(columns={catchments_comid_field: "ComID"})
    catch["ComID"] = pd.to_numeric(catch["ComID"], errors="coerce")
    catch = catch.dropna(subset=["ComID"]).copy()
    catch["ComID"] = catch["ComID"].astype("int64")
    catch = gpd.GeoDataFrame(catch, geometry="geometry", crs=catch.crs)

    # Keep only THS catchments available in GridCode/ComID xwalk.
    merged = xwalk.merge(catch, on="ComID", how="inner")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=catch.crs)
    if merged.empty:
        raise RuntimeError("No overlap between catchments and GridCodeComID xwalk.")

    merged = merged.sort_values(["ComID"]).reset_index(drop=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=catch.crs)

    # Build NLCD class percentages by catchment.
    nlcd_pct = _zonal_histogram(merged, nlcd_raster, NLCD_CLASSES, rasterio_mod, mask_fn)

    nlcd_cols = [f"NLCD{c}" for c in NLCD_CLASSES]
    nlcd_out = pd.DataFrame({
        "ComID": merged["ComID"].to_numpy(dtype=np.int64),
        "GridCode": merged["GridCode"].to_numpy(dtype=np.int64),
    })

    for idx, c in enumerate(nlcd_cols):
        nlcd_out[c] = nlcd_pct[:, idx]

    # Keep legacy compatibility columns.
    nlcd_out["PCTCN"] = 0.0
    nlcd_out["PCTMX"] = 0.0
    nlcd_out["SUMPCT"] = nlcd_out[nlcd_cols].sum(axis=1)

    nlcd_path = hsr_dir / "NLCD" / "catchmentattributesnlcd.txt"
    _write_nlcd_output(nlcd_path, nlcd_out)

    # Prepare monthly PRISM tables (WY month order Oct..Sep).
    wy_months = _water_year_months(wy)
    ppt_month_arrays: list[np.ndarray] = []
    tmean_month_arrays: list[np.ndarray] = []

    for wm in wy_months:
        ppt_path = _resolve_raster_path(prism_ppt_pattern, wm.cal_year, wm.cal_month)
        tmean_path = _resolve_raster_path(prism_tmean_pattern, wm.cal_year, wm.cal_month)

        ppt_arr = _zonal_mean(merged, ppt_path, rasterio_mod, mask_fn)
        tmean_arr = _zonal_mean(merged, tmean_path, rasterio_mod, mask_fn)
        ppt_month_arrays.append(ppt_arr)
        tmean_month_arrays.append(tmean_arr)

    ppt_stack = np.column_stack(ppt_month_arrays)  # shape ncatch x 12
    tmean_stack = np.column_stack(tmean_month_arrays)

    # The converted reader expects 13 precip columns; use mean annual as month 13.
    p13 = np.nanmean(ppt_stack, axis=1)
    ppt_table = pd.DataFrame({
        "GridCode": merged["GridCode"].to_numpy(dtype=np.int64),
        "GCAreaSqMi": np.full(len(merged), 1.0, dtype=float),
    })
    for i in range(12):
        ppt_table[f"PIn_{i + 1:02d}"] = ppt_stack[:, i]
    ppt_table["PIn_13"] = p13

    tmean_table = pd.DataFrame({
        "GridCode": merged["GridCode"].to_numpy(dtype=np.int64),
    })
    for i in range(12):
        tmean_table[f"TdC_{i + 1:02d}"] = tmean_stack[:, i]

    precip_out = hsr_dir / "PRISM" / "Precipitation" / f"PrismPrecipWY{wy}.dat"
    temp_out = hsr_dir / "PRISM" / "Temperature" / f"PrismTempAveWY{wy}.dat"

    _write_prism_prec_output(precip_out, ppt_table)
    _write_prism_temp_output(temp_out, tmean_table)

    print("REAL_NLCD_PRISM_BUILD_COMPLETE")
    print(f"THS={ths} WY={wy} catchments={len(merged)}")
    print(f"NLCD: {nlcd_path}")
    print(f"PRISM precipitation: {precip_out}")
    print(f"PRISM temperature: {temp_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build real HSR NLCD/PRISM inputs for converted AFINCH from catchment polygons and raster datasets."
        )
    )

    p.add_argument("--root", type=Path, default=Path(r"c:\Users\mu3575\Documents\WAM"))
    p.add_argument("--hsr", type=str, default="1200", help="HSR region code (e.g., 1200)")
    p.add_argument("--ths", type=str, default="1201", help="Target hydrologic subregion code (e.g., 1201)")
    p.add_argument("--wy", type=int, required=True, help="Water year (e.g., 2018)")

    p.add_argument(
        "--catchments",
        type=Path,
        required=True,
        help="Path to catchment polygons (GeoPackage/Shapefile/GeoJSON) with COMID field",
    )
    p.add_argument(
        "--catchments-comid-field",
        type=str,
        default="FEATUREID",
        help="Catchment COMID field name in the catchment layer",
    )

    p.add_argument("--nlcd-raster", type=Path, required=True, help="Path to NLCD categorical raster")

    p.add_argument(
        "--prism-ppt-pattern",
        type=str,
        required=True,
        help=(
            "Python format string for monthly PPT rasters with {yyyy} and {mm}. "
            "Example: D:/prism/ppt/PRISM_ppt_stable_4kmM3_{yyyy}{mm:02d}_bil.tif"
        ),
    )
    p.add_argument(
        "--prism-tmean-pattern",
        type=str,
        required=True,
        help=(
            "Python format string for monthly TMEAN rasters with {yyyy} and {mm}. "
            "Example: D:/prism/tmean/PRISM_tmean_stable_4kmM3_{yyyy}{mm:02d}_bil.tif"
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_inputs(
        root=args.root,
        hsr=args.hsr,
        ths=args.ths,
        wy=args.wy,
        catchments_path=args.catchments,
        catchments_comid_field=args.catchments_comid_field,
        nlcd_raster=args.nlcd_raster,
        prism_ppt_pattern=args.prism_ppt_pattern,
        prism_tmean_pattern=args.prism_tmean_pattern,
    )


if __name__ == "__main__":
    main()
