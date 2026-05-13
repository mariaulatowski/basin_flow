from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests

try:
    from build_brazos_hu4_1205_network import (
        SQKM_TO_SQMI,
        _backup,
        _load_station_points,
        _read_streamflow_dat,
        _write_streamflow_dat,
    )
except ModuleNotFoundError:
    from afinch_python_modules.build_brazos_hu4_1205_network import (
        SQKM_TO_SQMI,
        _backup,
        _load_station_points,
        _read_streamflow_dat,
        _write_streamflow_dat,
    )


NLCD_CLASSES = [11, 12, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 61, 71, 81, 82, 83, 84, 85, 91, 92]


def _norm_station_id(v: object) -> str:
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "nat"}:
        return ""
    if s.endswith('.0'):
        s = s[:-2]
    return s.lstrip('0') or '0'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a basin-wide runnable Brazos AFINCH package from all intersecting medium-resolution HU4 NHD datasets."
    )
    parser.add_argument("--base-dir", default=".", help="Workspace base directory")
    parser.add_argument("--ths", default="1200", help="Synthetic basin-wide THS code used by the converted runtime")
    parser.add_argument("--hsr", default="HSR1200", help="HSR folder name used by converted runtime")
    parser.add_argument(
        "--basin-shp",
        default="inputData/river_basin/TWDB_MRBs_2014.shp",
        help="Basin polygon shapefile used to clip the network",
    )
    parser.add_argument("--basin-field", default="basin_name", help="Basin name field in the basin shapefile")
    parser.add_argument("--basin-value", default="Brazos", help="Basin name value to extract")
    parser.add_argument(
        "--basin-buffer-m",
        type=float,
        default=0.0,
        help=(
            "Optional buffer distance in meters applied to the basin polygon for network selection "
            "(HU4 discovery, flowline selection, and catchment selection)."
        ),
    )
    parser.add_argument(
        "--gdb-root",
        default="inputData/nhd_medium_res_gdb",
        help="Directory containing NHD_H_XXXX_HU4_GDB folders",
    )
    parser.add_argument(
        "--hu4s",
        default="",
        help="Optional comma-separated HU4 codes to force instead of auto-discovery",
    )
    parser.add_argument("--wy", type=int, default=2018, help="Water year to generate PRISM files for")
    parser.add_argument(
        "--nlcd-raster",
        default="inputData/Annual_NLCD_LndCov_2018_CU_C1V1.tif",
        help="NLCD raster used to derive catchment NLCD attributes",
    )
    parser.add_argument(
        "--prism-ppt-dir",
        default="inputData/prism_monthly/ppt/extracted",
        help="Directory with monthly PRISM precipitation rasters",
    )
    parser.add_argument(
        "--prism-tmean-dir",
        default="inputData/prism_monthly/tmean/extracted",
        help="Directory with monthly PRISM tmean rasters",
    )
    parser.add_argument(
        "--gages-csv",
        default="",
        help=(
            "CSV of USGS gage stations to snap to reaches. "
            "Required columns: Station (or Gage_ID_norm), LAT, LONG. "
            "If omitted, falls back to inputData/inputs/monthly_wide_acft.csv (Brazos default)."
        ),
    )
    parser.add_argument(
        "--gages-flow-units",
        choices=["auto", "cfs", "acft"],
        default="auto",
        help=(
            "Units of monthly flow values in --gages-csv when building ComIDStationDAMoAnQYYYY.dat. "
            "Use auto (default), cfs, or acft."
        ),
    )
    parser.add_argument(
        "--wam-csv",
        default="",
        help=(
            "Optional CSV of WAM control-point stations. "
            "Required columns: Station (or CPID), LAT, LONG. "
            "If omitted, falls back to HSR1200/Streamflow/Brazos_new_wam_locations_nhdplus.csv "
            "(Brazos default), or skipped silently if that file doesn't exist."
        ),
    )
    parser.add_argument(
        "--annual-only",
        action="store_true",
        help=(
            "Rebuild only year-varying artifacts (PRISM yearly files and Streamflow dat) "
            "using previously built static network files."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates into HSR files (writes backups)")
    return parser.parse_args()


def _wy_months(wy: int) -> list[tuple[int, int]]:
    return [
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
    ]


def _resolve_prism_raster(base_dir: Path, prism_dir: str, kind: str, year: int, month: int) -> Path:
    root = (base_dir / prism_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    pattern = f"prism_{kind}_us_25m_{year}{month:02d}.tif"
    exact = root / pattern
    if exact.exists():
        return exact

    matches = sorted(root.glob(f"*{year}{month:02d}*.tif"))
    if not matches:
        raise FileNotFoundError(f"No PRISM raster matched {kind} {year}-{month:02d} in {root}")
    return matches[0]


def _sample_raster(points_4269: gpd.GeoDataFrame, raster_path: Path) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        pts = points_4269.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        vals = np.array([row[0] for row in src.sample(coords)], dtype=float)
        nodata = src.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        return vals


def _days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return 29 if leap else 28


def _monthly_acft_to_cfs(acft: float, year: int, month: int) -> float:
    secs = float(_days_in_month(year, month) * 24 * 3600)
    return float(acft) * 43560.0 / secs


def _fetch_usgs_drainage_areas(stations: list[str]) -> dict[str, float]:
    station_ids = sorted(
        {
            station_id
            for s in stations
            for station_id in [_norm_station_id(s)]
            if station_id and station_id.isdigit()
        }
    )
    if not station_ids:
        return {}

    url = "https://waterservices.usgs.gov/nwis/site/"
    drain_map: dict[str, float] = {}

    for i in range(0, len(station_ids), 25):
        batch = station_ids[i:i + 25]
        params = {
            "format": "rdb",
            "sites": ",".join(s.zfill(8) for s in batch),
            "siteOutput": "expanded",
            "siteStatus": "all",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        lines = response.text.splitlines()
        header_idx = next((idx for idx, line in enumerate(lines) if line.startswith("agency_cd\t")), None)
        if header_idx is None:
            continue

        data_lines = [line for line in lines[header_idx:] if line and not line.startswith("#")]
        reader = csv.DictReader(StringIO("\n".join(data_lines)), delimiter="\t")
        next(reader, None)
        for row in reader:
            site_no = _norm_station_id(str(row.get("site_no", "")).strip())
            drain_area = str(row.get("drain_area_va", "")).strip()
            if not site_no or not drain_area:
                continue
            try:
                drain_map[site_no] = float(drain_area)
            except ValueError:
                continue

    return drain_map


def _build_streamflow_dat_from_gages_csv(
    base_dir: Path,
    wy: int,
    station_map: pd.DataFrame,
    gages_csv: str,
    units: str,
    out_dat_path: Path,
    out_da_path: Path,
) -> pd.DataFrame:
    if gages_csv:
        gages_path = Path(gages_csv)
        if not gages_path.is_absolute():
            gages_path = base_dir / gages_csv
    else:
        gages_path = base_dir / "inputData" / "inputs" / "monthly_wide_acft.csv"

    if not gages_path.exists() or gages_path.is_dir():
        return pd.DataFrame()

    src = pd.read_csv(gages_path)

    id_col = next((c for c in src.columns if c in ("Gage_ID_norm", "Station", "station", "STATION", "CPID", "cpid")), None)
    if id_col is None:
        return pd.DataFrame()

    year_col = next((c for c in src.columns if str(c).lower() in ("year", "wy", "water_year")), None)
    if year_col is None:
        return pd.DataFrame()

    month_cols = {m: next((c for c in src.columns if str(c).upper() == m), None) for m in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]}
    if any(v is None for v in month_cols.values()):
        return pd.DataFrame()

    local_units = units
    if local_units == "auto":
        name = gages_path.name.lower()
        if "cfs" in name:
            local_units = "cfs"
        elif "acft" in name or "acre" in name:
            local_units = "acft"
        else:
            local_units = "cfs"

    src = src.copy()
    src["StationN"] = src[id_col].map(_norm_station_id)
    src = src[src["StationN"] != ""].copy()
    src[year_col] = pd.to_numeric(src[year_col], errors="coerce")
    src = src[src[year_col] == float(wy)].copy()
    if src.empty:
        return pd.DataFrame()
    src = src.drop_duplicates(subset=["StationN"], keep="first")

    area_col = next(
        (
            c for c in src.columns
            if str(c).strip().lower() in {
                "dasqmi",
                "drain_area_va",
                "drainage_area",
                "drainage_area_sqmi",
                "nwisarea",
                "area_sqmi",
            }
        ),
        None,
    )

    area_map: dict[str, float] = {}
    if area_col is not None:
        area_series = pd.to_numeric(src[area_col], errors="coerce")
        area_map = {
            station: float(area)
            for station, area in zip(src["StationN"], area_series)
            if pd.notna(area) and float(area) > 0
        }

    missing_for_api = sorted({s for s in src["StationN"].tolist() if s not in area_map})
    if missing_for_api:
        try:
            fetched_map = _fetch_usgs_drainage_areas(missing_for_api)
            area_map.update(fetched_map)
            print(
                f"Fetched USGS drainage area for {len(fetched_map):,} of {len(missing_for_api):,} stations "
                f"from NWIS site metadata."
            )
        except Exception as exc:
            print(f"WARNING: USGS drainage-area lookup failed; falling back where needed. {exc}")

    sm = station_map.copy()
    sm["StationN"] = sm["Station"].map(_norm_station_id)
    sm = sm[sm["StationN"] != ""].copy()
    sm = sm.drop_duplicates(subset=["StationN"], keep="first")

    work = sm.merge(src[["StationN", *month_cols.values()]], on="StationN", how="inner")
    if work.empty:
        return pd.DataFrame()

    q_wy_order = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP"]
    rows = []
    for _, r in work.iterrows():
        vals = []
        for m in q_wy_order:
            v = pd.to_numeric(r[month_cols[m]], errors="coerce")
            v = 0.0 if pd.isna(v) else float(v)
            if local_units == "acft":
                cal_month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"].index(m) + 1
                v = _monthly_acft_to_cfs(v, wy, cal_month)
            vals.append(v)

        q13 = float(np.mean(vals))
        rows.append([
            int(pd.to_numeric(r["ComID"], errors="coerce")),
            str(r["StationN"]),
            float(area_map.get(str(r["StationN"]), 5.79)),
            *vals,
            q13,
        ])

    if not rows:
        return pd.DataFrame()

    cols = ["ComIDSta", "StaWY", "NWISArea", *[f"Q{i:02d}" for i in range(1, 14)]]
    dat = pd.DataFrame(rows, columns=cols)
    dat["ComIDSta"] = pd.to_numeric(dat["ComIDSta"], errors="coerce").fillna(0).astype(np.int64)
    dat["StaWY"] = dat["StaWY"].map(_norm_station_id)
    _write_streamflow_dat(dat, out_dat_path)

    da = dat[["StaWY", "ComIDSta", "NWISArea"]].drop_duplicates(subset=["StaWY"], keep="first").copy()
    da = da.rename(columns={"StaWY": "Station", "ComIDSta": "ComID", "NWISArea": "DASqMi"})
    da.to_csv(out_da_path, index=False)
    return dat


def _build_real_nlcd(flow_geom: gpd.GeoDataFrame, nlcd_raster: Path) -> pd.DataFrame:
    points = flow_geom[["ComID", "geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=flow_geom.crs)

    cls = pd.Series(_sample_raster(points, nlcd_raster)).round().astype("Int64")
    out = pd.DataFrame({"ComID": points["ComID"].to_numpy(dtype=np.int64), "GridCode": points["ComID"].to_numpy(dtype=np.int64)})
    for code in NLCD_CLASSES:
        col = f"NLCD{code}"
        out[col] = np.where(cls.to_numpy(dtype="float64", na_value=np.nan) == float(code), 100.0, 0.0)

    out["PCTCN"] = 0.0
    out["PCTMX"] = 0.0
    nlcd_cols = [f"NLCD{c}" for c in NLCD_CLASSES]
    out["SUMPCT"] = out[nlcd_cols].sum(axis=1)
    return out


def _iter_raster_block_zone_values(
    polygons: gpd.GeoDataFrame,
    raster_path: Path,
    *,
    categorical_codes: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray] | np.ndarray, np.ndarray]:
    from rasterio.features import bounds as geom_bounds, rasterize
    from shapely.geometry import box

    if polygons.empty:
        empty = np.zeros(0, dtype=float)
        return empty, {code: empty.copy() for code in categorical_codes or []} if categorical_codes else empty.copy(), empty

    with rasterio.open(raster_path) as src:
        zones = polygons[["ComID", "geometry"]].copy()
        if zones.crs is None:
            zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4269")
        zones = zones.to_crs(src.crs)
        zones = zones.dropna(subset=["geometry"]).copy()
        zones = zones[~zones.geometry.is_empty].copy()
        zones = zones.reset_index(drop=True)
        zone_count = len(zones)
        zone_ids = np.arange(1, zone_count + 1, dtype=np.int32)
        sindex = zones.sindex

        total = np.zeros(zone_count + 1, dtype=float)
        if categorical_codes is not None:
            cat_counts = {code: np.zeros(zone_count + 1, dtype=float) for code in categorical_codes}
        else:
            sums = np.zeros(zone_count + 1, dtype=float)

        raster_bounds = box(*src.bounds)
        for _, window in src.block_windows(1):
            win_bounds = rasterio.windows.bounds(window, src.transform)
            win_box = box(*win_bounds)
            if not win_box.intersects(raster_bounds):
                continue
            idx = list(sindex.query(win_box, predicate="intersects"))
            if not idx:
                continue

            arr = src.read(1, window=window, masked=True)
            if arr.size == 0:
                continue
            arr_data = np.asarray(arr.data, dtype=float)
            valid_data = np.isfinite(arr_data)
            if np.ma.isMaskedArray(arr):
                valid_data &= ~np.asarray(arr.mask)
            if not valid_data.any():
                continue

            transform = src.window_transform(window)
            shapes = []
            for i in idx:
                geom = zones.geometry.iloc[i]
                try:
                    if box(*geom_bounds(geom)).intersects(win_box):
                        shapes.append((geom, int(zone_ids[i])))
                except Exception:
                    continue
            if not shapes:
                continue

            zone_arr = rasterize(
                shapes,
                out_shape=arr_data.shape,
                transform=transform,
                fill=0,
                dtype="int32",
                all_touched=False,
            )
            valid = (zone_arr > 0) & valid_data
            if not valid.any():
                continue

            z = zone_arr[valid].ravel()
            total += np.bincount(z, minlength=zone_count + 1).astype(float)
            vals = arr_data[valid].ravel()
            if categorical_codes is not None:
                rounded = np.rint(vals).astype(int)
                for code in categorical_codes:
                    code_z = z[rounded == code]
                    if code_z.size:
                        cat_counts[code] += np.bincount(code_z, minlength=zone_count + 1).astype(float)
            else:
                sums += np.bincount(z, weights=vals, minlength=zone_count + 1).astype(float)

        if categorical_codes is not None:
            return zones["ComID"].to_numpy(dtype=np.int64), {code: vals[1:] for code, vals in cat_counts.items()}, total[1:]
        return zones["ComID"].to_numpy(dtype=np.int64), sums[1:], total[1:]


def _build_catchment_nlcd(catch_geom: gpd.GeoDataFrame, nlcd_raster: Path) -> pd.DataFrame:
    comids, counts_by_code, totals = _iter_raster_block_zone_values(
        catch_geom,
        nlcd_raster,
        categorical_codes=NLCD_CLASSES,
    )
    out = pd.DataFrame({"ComID": comids, "GridCode": comids})
    denom = np.where(totals > 0, totals, np.nan)
    for code in NLCD_CLASSES:
        out[f"NLCD{code}"] = np.nan_to_num((counts_by_code[code] / denom) * 100.0, nan=0.0)
    out["PCTCN"] = 0.0
    out["PCTMX"] = 0.0
    nlcd_cols = [f"NLCD{c}" for c in NLCD_CLASSES]
    out["SUMPCT"] = out[nlcd_cols].sum(axis=1)
    return out


def _build_real_prism(base_dir: Path, flow_geom: gpd.GeoDataFrame, wy: int, prism_ppt_dir: str, prism_tmean_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = flow_geom[["ComID", "geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=flow_geom.crs)

    ppt_vals: list[np.ndarray] = []
    tmean_vals: list[np.ndarray] = []
    for year, month in _wy_months(wy):
        ppt_path = _resolve_prism_raster(base_dir, prism_ppt_dir, "ppt", year, month)
        tmean_path = _resolve_prism_raster(base_dir, prism_tmean_dir, "tmean", year, month)
        ppt_vals.append(_sample_raster(points, ppt_path))
        tmean_vals.append(_sample_raster(points, tmean_path))

    ppt_stack = np.column_stack(ppt_vals)
    tmean_stack = np.column_stack(tmean_vals)

    # Fill sparse nodata with month medians to keep converted runtime stable.
    for col in range(ppt_stack.shape[1]):
        month = ppt_stack[:, col]
        med = np.nanmedian(month)
        if not np.isfinite(med):
            med = 0.0
        month[np.isnan(month)] = med
        ppt_stack[:, col] = month

    for col in range(tmean_stack.shape[1]):
        month = tmean_stack[:, col]
        med = np.nanmedian(month)
        if not np.isfinite(med):
            med = 0.0
        month[np.isnan(month)] = med
        tmean_stack[:, col] = month

    precip = pd.DataFrame({"GridCode": points["ComID"].to_numpy(dtype=np.int64), "GCAreaSqMi": np.full(len(points), 1.0, dtype=float)})
    for month in range(12):
        precip[f"PIn_{month + 1:02d}"] = ppt_stack[:, month]
    precip["PIn_13"] = ppt_stack.mean(axis=1)

    temp = pd.DataFrame({"GridCode": points["ComID"].to_numpy(dtype=np.int64)})
    for month in range(12):
        temp[f"TdC_{month + 1:02d}"] = tmean_stack[:, month]

    return precip, temp


def _build_catchment_prism(base_dir: Path, catch_geom: gpd.GeoDataFrame, wy: int, prism_ppt_dir: str, prism_tmean_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    area_map = catch_geom.set_index("ComID")["AreaSqKm"].to_dict() if "AreaSqKm" in catch_geom.columns else {}
    ppt_vals: list[np.ndarray] = []
    tmean_vals: list[np.ndarray] = []
    comids_ref: np.ndarray | None = None

    for year, month in _wy_months(wy):
        ppt_path = _resolve_prism_raster(base_dir, prism_ppt_dir, "ppt", year, month)
        tmean_path = _resolve_prism_raster(base_dir, prism_tmean_dir, "tmean", year, month)

        ppt_comids, ppt_sums, ppt_counts = _iter_raster_block_zone_values(catch_geom, ppt_path)
        tmp_comids, tmp_sums, tmp_counts = _iter_raster_block_zone_values(catch_geom, tmean_path)
        if comids_ref is None:
            comids_ref = ppt_comids
        if not np.array_equal(comids_ref, ppt_comids) or not np.array_equal(comids_ref, tmp_comids):
            raise RuntimeError("Catchment zonal raster processing returned inconsistent ComID order")

        ppt_vals.append(np.divide(ppt_sums, ppt_counts, out=np.full_like(ppt_sums, np.nan, dtype=float), where=ppt_counts > 0))
        tmean_vals.append(np.divide(tmp_sums, tmp_counts, out=np.full_like(tmp_sums, np.nan, dtype=float), where=tmp_counts > 0))

    if comids_ref is None:
        return pd.DataFrame(), pd.DataFrame()

    ppt_stack = np.column_stack(ppt_vals)
    tmean_stack = np.column_stack(tmean_vals)
    for stack in [ppt_stack, tmean_stack]:
        for col in range(stack.shape[1]):
            vals = stack[:, col]
            med = np.nanmedian(vals)
            if not np.isfinite(med):
                med = 0.0
            vals[np.isnan(vals)] = med
            stack[:, col] = vals

    area_sqmi = np.array([float(area_map.get(int(c), 1.0)) * SQKM_TO_SQMI for c in comids_ref], dtype=float)
    precip = pd.DataFrame({"GridCode": comids_ref, "GCAreaSqMi": area_sqmi})
    for month in range(12):
        precip[f"PIn_{month + 1:02d}"] = ppt_stack[:, month]
    precip["PIn_13"] = ppt_stack.mean(axis=1)

    temp = pd.DataFrame({"GridCode": comids_ref})
    for month in range(12):
        temp[f"TdC_{month + 1:02d}"] = tmean_stack[:, month]
    return precip, temp


def _load_basin_polygon(base_dir: Path, basin_shp: str, basin_field: str, basin_value: str) -> gpd.GeoDataFrame:
    basin_path = (base_dir / basin_shp).resolve()
    if not basin_path.exists():
        raise FileNotFoundError(basin_path)

    basin = gpd.read_file(basin_path)
    if basin_field not in basin.columns:
        raise KeyError(f"Missing basin field '{basin_field}' in {basin_path}")

    basin = basin[basin[basin_field].astype(str).str.contains(basin_value, case=False, na=False)].copy()
    if basin.empty:
        raise ValueError(f"No basin polygons matched {basin_field}={basin_value!r}")
    if basin.crs is None:
        basin = basin.set_crs("EPSG:4269")
    return basin


def _load_forced_hu4_polygon(base_dir: Path, gdb_root: str, forced_hu4s: str) -> gpd.GeoDataFrame:
    root = (base_dir / gdb_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    forced = [item.strip() for item in forced_hu4s.split(",") if item.strip()]
    if not forced:
        raise ValueError("HU4-only build requested, but no HU4 codes were provided.")

    mask_parts: list[gpd.GeoDataFrame] = []
    gdb_candidates = sorted(root.glob("NHD_H_*_HU4_GDB/*.gdb")) + sorted(root.glob("NHDPLUS_H_*_HU4_GDB/*.gdb"))
    for gdb in gdb_candidates:
        parts = gdb.parent.name.split("_")
        if len(parts) < 4:
            continue
        hu4 = parts[2]
        if hu4 not in forced:
            continue

        wbd = gpd.read_file(gdb, layer="WBDHU4")
        if wbd.empty:
            continue
        huc4_col = next((c for c in wbd.columns if c.lower() == "huc4"), None)
        if huc4_col is not None:
            wbd = wbd[wbd[huc4_col].astype(str) == hu4]
        if wbd.empty:
            continue
        if wbd.crs is None:
            wbd = wbd.set_crs("EPSG:4269")
        mask_parts.append(wbd[["geometry"]].copy())

    if not mask_parts:
        raise ValueError(f"Could not load WBDHU4 polygons for HU4(s): {', '.join(forced)}")

    return gpd.GeoDataFrame(pd.concat(mask_parts, ignore_index=True), geometry="geometry", crs=mask_parts[0].crs)


def _buffer_polygon_meters(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    if buffer_m <= 0:
        return gdf
    proj = gdf.to_crs(5070)
    buffered = proj.copy()
    buffered["geometry"] = proj.geometry.buffer(buffer_m)
    return buffered.to_crs(gdf.crs)


def _discover_hu4_gdbs(base_dir: Path, gdb_root: str, basin_gdf: gpd.GeoDataFrame, forced_hu4s: str) -> list[tuple[str, Path]]:
    root = (base_dir / gdb_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    forced = {item.strip() for item in forced_hu4s.split(",") if item.strip()}
    basin_proj = basin_gdf.to_crs(5070)
    basin_union = basin_proj.geometry.union_all()

    hits: list[tuple[str, Path]] = []

    # Support both naming conventions found in this workspace:
    # 1) NHD_H_1205_HU4_GDB/<name>.gdb
    # 2) NHDPLUS_H_1205_HU4_GDB/<name>.gdb
    gdb_candidates = sorted(root.glob("NHD_H_*_HU4_GDB/*.gdb")) + sorted(root.glob("NHDPLUS_H_*_HU4_GDB/*.gdb"))

    for gdb in gdb_candidates:
        parts = gdb.parent.name.split("_")
        if len(parts) < 4:
            continue
        hu4 = parts[2]
        if forced and hu4 not in forced:
            continue

        wbd = gpd.read_file(gdb, layer="WBDHU4")
        if wbd.empty:
            continue
        if wbd.crs is None:
            wbd = wbd.set_crs(basin_gdf.crs or "EPSG:4269")
        inter_area = wbd.to_crs(5070).geometry.union_all().intersection(basin_union).area
        if inter_area > 0:
            hits.append((hu4, gdb))

    if not hits:
        raise ValueError("No HU4 geodatabases intersect the selected basin polygon")
    return hits


def _load_selected_hu4_mask(hu4_gdbs: list[tuple[str, Path]], basin_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask_parts: list[gpd.GeoDataFrame] = []
    for hu4, gdb in hu4_gdbs:
        wbd = gpd.read_file(gdb, layer="WBDHU4")
        if wbd.empty:
            continue
        huc4_col = next((c for c in wbd.columns if c.lower() == "huc4"), None)
        if huc4_col is not None:
            wbd = wbd[wbd[huc4_col].astype(str) == str(hu4)]
        if wbd.empty:
            continue
        if wbd.crs is None:
            wbd = wbd.set_crs(basin_gdf.crs or "EPSG:4269")
        mask_parts.append(wbd[["geometry"]].copy())

    if not mask_parts:
        raise ValueError("Selected HU4 polygons could not be loaded from WBDHU4 layers")

    mask = gpd.GeoDataFrame(pd.concat(mask_parts, ignore_index=True), geometry="geometry", crs=mask_parts[0].crs)
    mask = mask.to_crs(basin_gdf.crs or mask.crs)
    basin_union = basin_gdf.geometry.union_all()
    hu4_union = mask.geometry.union_all()
    clipped = hu4_union.intersection(basin_union)
    return gpd.GeoDataFrame({"geometry": [clipped]}, geometry="geometry", crs=basin_gdf.crs or mask.crs)


def _load_basin_flowline_and_vaa(
    gdbs: list[tuple[str, Path]],
    basin_gdf: gpd.GeoDataFrame,
    ths: str,
    basin_buffer_m: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    basin_sel = _buffer_polygon_meters(basin_gdf, basin_buffer_m)
    basin_latlon = basin_sel.to_crs(4269)
    bbox = tuple(float(v) for v in basin_latlon.total_bounds)

    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower_map = {str(c).lower(): c for c in df.columns}
        for name in candidates:
            found = lower_map.get(name.lower())
            if found is not None:
                return found
        return None

    def _load_one_gdb(hu4_gdb: tuple[str, Path]):
        """Load flowline + VAA from a single GDB; return (flow_part, vaa_part) or (None, None)."""
        hu4, gdb = hu4_gdb
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fl = gpd.read_file(
                gdb,
                layer="NHDFlowline",
                bbox=bbox,
                columns=["permanent_identifier", "COMID", "NHDPlusID", "lengthkm", "LengthKM", "reachcode", "ReachCode", "geometry"],
            )
        if fl.empty:
            return None, None
        if fl.crs is None:
            fl = fl.set_crs(basin_latlon.crs)
        fl = gpd.clip(fl.to_crs(basin_latlon.crs), basin_latlon)
        if fl.empty:
            return None, None

        comid_col = _pick_col(fl, ["comid", "nhdplusid", "permanent_identifier", "permanent_id", "permanentidentifier"])
        if comid_col is None:
            raise KeyError(f"No ComID-like field found in NHDFlowline for {gdb}. Columns: {list(fl.columns)}")
        fl["ComID"] = pd.to_numeric(fl[comid_col], errors="coerce")

        len_col = _pick_col(fl, ["lengthkm", "length_km", "shape_length"])
        fl["LengthKm"] = pd.to_numeric(fl[len_col], errors="coerce") if len_col else 1.0

        reach_col = _pick_col(fl, ["reachcode", "reach_code", "reachcode_1"])
        fl = fl.dropna(subset=["ComID", "geometry"]).copy()
        fl = fl[~fl.geometry.is_empty].copy()
        fl["ComID"] = fl["ComID"].astype("int64")
        fl["LengthKm"] = fl["LengthKm"].fillna(1.0).clip(lower=0.01)
        fl["OrigReachCode"] = fl[reach_col].astype(str).str.strip() if reach_col else ""
        fl["ReachCode"] = ""  # will be reassigned after concat
        flow_part = fl[["ComID", "LengthKm", "ReachCode", "OrigReachCode", "geometry"]].copy()

        vaa = None
        vaa_layer_names = ["NHDFlowlineVAA", "NHDPlusFlowlineVAA"]
        vaa_last_exc: Exception | None = None
        for vaa_layer in vaa_layer_names:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Read full schema: VAA field names are often CamelCase
                    # (ToNode, HydroSeq, DnHydroSeq, StartFlag, Divergence).
                    # Column filtering here can silently drop topology fields
                    # when case does not match exactly.
                    vaa = gpd.read_file(gdb, layer=vaa_layer)
                break
            except Exception as exc:
                vaa_last_exc = exc
                continue
        if vaa is None:
            raise RuntimeError(f"Could not open VAA layer in {gdb}. Tried: {vaa_layer_names}. Last error: {vaa_last_exc}")

        vaa_comid_col = _pick_col(vaa, ["comid", "nhdplusid", "permanent_identifier", "permanent_id", "permanentidentifier"])
        if vaa_comid_col is None:
            raise KeyError(f"No ComID-like field found in NHDFlowlineVAA for {gdb}. Columns: {list(vaa.columns)}")
        hydro_col = _pick_col(vaa, ["hydroseq"])
        dn_hydro_col = _pick_col(vaa, ["dnhydroseq", "dn_hydroseq"])
        from_col = _pick_col(vaa, ["fromnode"])
        to_col = _pick_col(vaa, ["tonode"])
        div_col = _pick_col(vaa, ["divergenceflag", "divergence"])
        start_col = _pick_col(vaa, ["startflag", "start_flag"])

        vaa["ComID"] = pd.to_numeric(vaa[vaa_comid_col], errors="coerce")
        vaa["HydroSeq"] = pd.to_numeric(vaa[hydro_col], errors="coerce") if hydro_col else np.nan
        vaa["DnHydroSeq"] = pd.to_numeric(vaa[dn_hydro_col], errors="coerce") if dn_hydro_col else np.nan
        vaa["FromNode"] = pd.to_numeric(vaa[from_col], errors="coerce") if from_col else np.nan
        vaa["ToNode"] = pd.to_numeric(vaa[to_col], errors="coerce") if to_col else np.nan
        vaa["Divergence"] = pd.to_numeric(vaa[div_col], errors="coerce").fillna(0) if div_col else 0
        vaa["StartFlag"] = pd.to_numeric(vaa[start_col], errors="coerce").fillna(0) if start_col else 0
        vaa_part = vaa[["ComID", "FromNode", "ToNode", "HydroSeq", "DnHydroSeq", "Divergence", "StartFlag"]].copy()
        return flow_part, vaa_part

    # --- parallel load ---
    flow_parts: list[gpd.GeoDataFrame] = []
    vaa_parts: list[pd.DataFrame] = []
    n_workers = min(len(gdbs), 6)  # cap at 6 to avoid memory pressure
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one_gdb, item): item[0] for item in gdbs}
        for fut in as_completed(futures):
            hu4 = futures[fut]
            fp, vp = fut.result()  # raises immediately on error
            print(f"  Loaded GDB {hu4}")
            if fp is not None:
                flow_parts.append(fp)
            if vp is not None:
                vaa_parts.append(vp)

    # Legacy sequential loop removed — replaced by parallel block above.
    if not flow_parts:
        raise ValueError("No clipped flowlines were found for the selected basin")

    flow = pd.concat(flow_parts, ignore_index=True)
    flow = flow.sort_values(["ComID", "LengthKm"], ascending=[True, False]).drop_duplicates(subset=["ComID"], keep="first")
    flow = flow.reset_index(drop=True)
    flow["ReachCode"] = [f"{ths}{i:010d}" for i in range(1, len(flow) + 1)]

    flow_geom = gpd.GeoDataFrame(flow[["ComID", "ReachCode", "OrigReachCode", "geometry"]].copy(), geometry="geometry", crs=basin_latlon.crs)
    flow_tbl = flow[["ComID", "LengthKm", "ReachCode"]].copy()

    if vaa_parts:
        vaa = pd.concat(vaa_parts, ignore_index=True)
        vaa = vaa.dropna(subset=["ComID"]).copy()
        vaa["ComID"] = vaa["ComID"].astype("int64")
        vaa = vaa[vaa["ComID"].isin(flow_tbl["ComID"])].copy()
    else:
        vaa = pd.DataFrame(columns=["ComID", "FromNode", "ToNode", "HydroSeq", "Divergence", "StartFlag"])

    if vaa.empty:
        synthetic = flow_tbl[["ComID"]].copy()
        synthetic["HydroSeq"] = np.arange(len(synthetic), 0, -1, dtype=np.int64)
        synthetic["FromNode"] = synthetic["HydroSeq"]
        synthetic["ToNode"] = np.append(synthetic["HydroSeq"].to_numpy(dtype=np.int64)[1:], 0)
        synthetic["Divergence"] = 0
        synthetic["StartFlag"] = 0
        synthetic.loc[synthetic.index[0], "StartFlag"] = 1
        vaa_out = synthetic[["ComID", "FromNode", "ToNode", "HydroSeq", "Divergence", "StartFlag"]]
    else:
        vaa = vaa[["ComID", "FromNode", "ToNode", "HydroSeq", "DnHydroSeq", "Divergence", "StartFlag"]].copy()
        vaa["HydroSeq"] = pd.to_numeric(vaa["HydroSeq"], errors="coerce").fillna(0)
        missing_hs = vaa["HydroSeq"] <= 0
        if missing_hs.any():
            fill_vals = np.arange(missing_hs.sum(), 0, -1, dtype=np.int64)
            vaa.loc[missing_hs, "HydroSeq"] = fill_vals
        vaa["HydroSeq"] = vaa["HydroSeq"].astype("int64")
        vaa["FromNode"] = pd.to_numeric(vaa["FromNode"], errors="coerce")
        vaa["ToNode"] = pd.to_numeric(vaa["ToNode"], errors="coerce")
        vaa["DnHydroSeq"] = pd.to_numeric(vaa["DnHydroSeq"], errors="coerce").fillna(0)

        # Some source VAA tables provide HydroSeq/DnHydroSeq but leave node ids empty.
        # In that case, derive node connectivity directly from hydro-sequence linkage.
        to_nonzero = int((vaa["ToNode"].fillna(0) != 0).sum())
        dn_nonzero = int((vaa["DnHydroSeq"] != 0).sum())
        if to_nonzero == 0 and dn_nonzero > 0:
            print("WARNING: VAA ToNode values are empty; deriving connectivity from HydroSeq/DnHydroSeq.")
            # Use DnHydroSeq directly as ToNode — interior reaches will chain correctly.
            # Outlet reaches naturally have DnHydroSeq=0 (which becomes ToNode=0).
            # Do NOT filter by valid_hs: after basin clipping, downstream boundary reaches
            # won't be in the set, which would zero out all interior ToNode values.
            vaa["FromNode"] = vaa["HydroSeq"].astype("int64")
            vaa["ToNode"] = vaa["DnHydroSeq"].astype("int64")

        vaa["FromNode"] = vaa["FromNode"].fillna(vaa["HydroSeq"]).astype("int64")
        vaa["ToNode"] = vaa["ToNode"].fillna(0).astype("int64")
        vaa["Divergence"] = pd.to_numeric(vaa["Divergence"], errors="coerce").fillna(0).astype("int64")
        vaa["StartFlag"] = pd.to_numeric(vaa["StartFlag"], errors="coerce").fillna(0).astype("int64")

        # Recompute StartFlag for consistency with final node topology.
        upstream_targets = set(vaa["ToNode"].tolist())
        vaa["StartFlag"] = (~vaa["FromNode"].isin(upstream_targets)).astype("int64")

        missing = np.setdiff1d(flow_tbl["ComID"].to_numpy(dtype=np.int64), vaa["ComID"].to_numpy(dtype=np.int64))
        if len(missing) > 0:
            extra = pd.DataFrame({"ComID": missing})
            extra["HydroSeq"] = np.arange(len(vaa) + len(extra), len(vaa), -1, dtype=np.int64)
            extra["FromNode"] = extra["HydroSeq"]
            extra["ToNode"] = 0
            extra["Divergence"] = 0
            extra["StartFlag"] = 1
            vaa = pd.concat([vaa, extra], ignore_index=True)
        vaa_out = vaa.sort_values("ComID").drop_duplicates(subset=["ComID"], keep="first").reset_index(drop=True)

    nonzero_to = int((vaa_out["ToNode"] != 0).sum())
    if nonzero_to == 0 and len(vaa_out) > 1:
        # Last-resort fallback for source datasets with unusable node fields.
        # Build a simple connected downstream chain from HydroSeq order so
        # upstream-tracing doesn't collapse to one reach per station.
        print("WARNING: VAA ToNode still all zero; synthesizing connectivity from HydroSeq ordering.")
        synth = vaa_out.copy().sort_values("HydroSeq", ascending=False).reset_index(drop=True)
        n = len(synth)
        synth["FromNode"] = np.arange(n, 0, -1, dtype=np.int64)
        synth["ToNode"] = np.append(synth["FromNode"].to_numpy(dtype=np.int64)[1:], 0)
        synth["StartFlag"] = 0
        synth.loc[synth.index[0], "StartFlag"] = 1
        vaa_out = synth.sort_values("ComID").reset_index(drop=True)
        nonzero_to = int((vaa_out["ToNode"] != 0).sum())

    if nonzero_to == 0:
        raise RuntimeError(
            "Built VAA has no downstream links (ToNode all zero). "
            "Upstream gaged-catchment tracing will collapse to 1 reach per station. "
            "Check VAA source fields (FromNode/ToNode or HydroSeq/DnHydroSeq) before applying build outputs."
        )

    return flow_tbl, vaa_out, flow_geom[["ComID", "ReachCode", "OrigReachCode", "geometry"]].copy()


def _load_basin_catchments(
    gdbs: list[tuple[str, Path]],
    basin_gdf: gpd.GeoDataFrame,
    flow_tbl: pd.DataFrame,
    basin_buffer_m: float = 0.0,
) -> gpd.GeoDataFrame:
    import fiona
    import warnings

    basin_sel = _buffer_polygon_meters(basin_gdf, basin_buffer_m)
    basin_latlon = basin_sel.to_crs(4269)
    bbox = tuple(float(v) for v in basin_latlon.total_bounds)
    valid_comids = set(flow_tbl["ComID"].astype("int64").tolist())
    parts: list[gpd.GeoDataFrame] = []

    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower_map = {str(c).lower(): c for c in df.columns}
        for name in candidates:
            found = lower_map.get(name.lower())
            if found is not None:
                return found
        return None

    for hu4, gdb in gdbs:
        try:
            layers = set(fiona.listlayers(gdb))
        except Exception:
            continue
        catch_layer = next((name for name in layers if name.lower() in {"nhdpluscatchment", "catchment"}), None)
        if catch_layer is None:
            catch_layer = next((name for name in layers if "catchment" in name.lower()), None)
        if catch_layer is None:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            catch = gpd.read_file(
                gdb,
                layer=catch_layer,
                bbox=bbox,
                columns=["NHDPlusID", "ComID", "COMID", "GridCode", "AreaSqKm", "geometry"],
            )
        if catch.empty:
            continue
        if catch.crs is None:
            catch = catch.set_crs(basin_latlon.crs)
        catch = catch.to_crs(basin_latlon.crs)
        # Keep full catchment geometry when it intersects the selected polygon
        # (including optional buffered extent) instead of clipping geometry.
        sel_union = basin_latlon.geometry.union_all()
        catch = catch[catch.geometry.intersects(sel_union)].copy()
        if catch.empty:
            continue

        id_col = _pick_col(catch, ["nhdplusid", "comid", "gridcode"])
        if id_col is None:
            print(f"WARNING: No ComID-like field in catchment layer for {hu4}; skipping")
            continue
        catch["ComID"] = pd.to_numeric(catch[id_col], errors="coerce")
        catch = catch.dropna(subset=["ComID", "geometry"]).copy()
        catch["ComID"] = catch["ComID"].astype("int64")
        catch = catch[catch["ComID"].isin(valid_comids)].copy()
        if catch.empty:
            continue

        area_col = _pick_col(catch, ["areasqkm", "area_sqkm"])
        if area_col is not None:
            catch["AreaSqKm"] = pd.to_numeric(catch[area_col], errors="coerce")
        else:
            catch["AreaSqKm"] = np.nan
        missing_area = catch["AreaSqKm"].isna() | (catch["AreaSqKm"] <= 0)
        if missing_area.any():
            area_proj = catch.loc[missing_area, ["geometry"]].to_crs(5070)
            catch.loc[missing_area, "AreaSqKm"] = area_proj.geometry.area.to_numpy(dtype=float) / 1_000_000.0

        parts.append(catch[["ComID", "AreaSqKm", "geometry"]].copy())
        print(f"  Loaded catchments {hu4}: {len(catch):,}")

    if not parts:
        return gpd.GeoDataFrame(columns=["ComID", "AreaSqKm", "geometry"], geometry="geometry", crs=basin_latlon.crs)

    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)
    out = out.sort_values(["ComID", "AreaSqKm"], ascending=[True, False]).drop_duplicates(subset=["ComID"], keep="first")
    out = out.reset_index(drop=True)
    return out


def _map_stations_to_comid(
    flow_geom: gpd.GeoDataFrame,
    station_points: pd.DataFrame,
    station_mask: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(station_points.copy(), geometry=gpd.points_from_xy(station_points["LONG"], station_points["LAT"]), crs="EPSG:4326")

    if station_mask is not None:
        mask = station_mask.copy()
        if mask.crs is None:
            mask = mask.set_crs("EPSG:4269")
        mask = mask.to_crs(pts.crs)
        keep = pts.geometry.intersects(mask.geometry.union_all())
        kept = int(keep.sum())
        dropped = int((~keep).sum())
        print(f"Filtered stations to selected HU4 footprint: kept {kept:,}, dropped {dropped:,}")
        pts = pts.loc[keep].copy()

    flow_gdf = flow_geom.copy()
    if flow_gdf.crs is None:
        flow_gdf = flow_gdf.set_crs("EPSG:4269")

    proj = flow_gdf.estimate_utm_crs()
    flow_proj = flow_gdf.to_crs(proj)
    pts_proj = pts.to_crs(proj)
    nearest = gpd.sjoin_nearest(
        pts_proj[["Station", "Source", "LAT", "LONG", "geometry"]],
        flow_proj[["ComID", "geometry"]],
        how="left",
        distance_col="snap_dist_m",
    )
    nearest = nearest.drop(columns=[c for c in ["index_right"] if c in nearest.columns])
    nearest = nearest.dropna(subset=["ComID"]).copy()
    nearest["ComID"] = pd.to_numeric(nearest["ComID"], errors="coerce").astype("int64")
    nearest = nearest.sort_values(["Station", "snap_dist_m"]).drop_duplicates(subset=["Station"], keep="first")
    return pd.DataFrame(nearest[["Station", "ComID", "Source", "snap_dist_m"]]).sort_values("Station")


def _build_vaa_upstream_map(vaa_df: pd.DataFrame) -> dict[int, list[int]]:
    """Build downstream COMID -> immediate upstream COMIDs from VAA node topology."""
    cols = {str(c).lower(): c for c in vaa_df.columns}
    required = ["comid", "fromnode", "tonode", "hydroseq"]
    missing = [name for name in required if name not in cols]
    if missing:
        raise KeyError(f"VAA missing required columns for upstream tracing: {missing}")

    work = vaa_df[[cols["comid"], cols["fromnode"], cols["tonode"], cols["hydroseq"]]].copy()
    work.columns = ["ComID", "FromNode", "ToNode", "HydroSeq"]
    for col in ["ComID", "FromNode", "ToNode", "HydroSeq"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["ComID", "FromNode", "ToNode", "HydroSeq"]).copy()
    work = work[(work["FromNode"] > 0) & (work["ToNode"] > 0)].copy()
    work["ComID"] = work["ComID"].astype("int64")
    work["FromNode"] = work["FromNode"].astype("int64")
    work["ToNode"] = work["ToNode"].astype("int64")

    upstream = work[["ComID", "ToNode", "HydroSeq"]].rename(
        columns={"ComID": "UpComID", "ToNode": "JoinNode", "HydroSeq": "UpHydroSeq"}
    )
    downstream = work[["ComID", "FromNode", "HydroSeq"]].rename(
        columns={"ComID": "DsComID", "FromNode": "JoinNode", "HydroSeq": "DsHydroSeq"}
    )
    edges = upstream.merge(downstream, on="JoinNode", how="inner")
    edges = edges[edges["UpComID"] != edges["DsComID"]]
    edges = edges[edges["UpHydroSeq"] > edges["DsHydroSeq"]]
    edges = edges[["DsComID", "UpComID"]].drop_duplicates()

    upstream_map: dict[int, list[int]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        upstream_map[int(row.DsComID)].append(int(row.UpComID))
    return {k: sorted(v) for k, v in upstream_map.items()}


def _find_upstream_comids(start_comid: int, upstream_map: dict[int, list[int]], valid_comids: set[int]) -> list[int]:
    visited = {int(start_comid)}
    queue = deque([int(start_comid)])

    while queue:
        current = queue.popleft()
        for upstream_comid in upstream_map.get(current, []):
            upstream_comid = int(upstream_comid)
            if upstream_comid in visited or upstream_comid not in valid_comids:
                continue
            visited.add(upstream_comid)
            queue.append(upstream_comid)

    return sorted(visited)


def _allocate_station_member_areas(
    members: list[int],
    dasqmi: float,
    length_km_map: dict[int, float],
) -> dict[int, float]:
    if not members:
        return {}

    total_sqkm = max(0.01, float(dasqmi) / SQKM_TO_SQMI)
    weights = np.array([max(0.0, float(length_km_map.get(comid, 0.0))) for comid in members], dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(len(members), dtype=float)

    areas = total_sqkm * weights / float(weights.sum())
    areas = np.maximum(areas, 0.000001)
    return {int(comid): float(area) for comid, area in zip(members, areas)}


def _normalize_station_id_local(s: str) -> str:
    """Normalize station IDs (strip leading zeros, upper-case)."""
    s = str(s).strip().upper()
    # Strip leading zeros from purely numeric ids
    if s.isdigit():
        s = str(int(s))
    return s


def _load_station_points_flexible(
    base_dir: Path,
    gages_csv: str = "",
    wam_csv: str = "",
) -> pd.DataFrame:
    """
    Load gage station lat/long for reach-snapping.

    If ``gages_csv`` is provided it is used as the USGS/primary gage source.
    Accepted column names: Station or Gage_ID_norm (id), LAT, LONG.

    If ``wam_csv`` is provided it is used as the WAM control-point source.
    Accepted column names: Station or CPID (id), LAT, LONG.

    If either path is omitted the function falls back to the Brazos-specific
    default files so the existing Brazos workflow is unchanged.
    """
    parts: list[pd.DataFrame] = []

    # ── Primary (USGS) gage source ─────────────────────────────────────────
    if gages_csv:
        primary_path = Path(gages_csv)
        if not primary_path.is_absolute():
            primary_path = base_dir / gages_csv
    else:
        primary_path = base_dir / "inputData" / "inputs" / "monthly_wide_acft.csv"

    if primary_path.exists():
        df = pd.read_csv(primary_path, dtype=str)
        # Flexible column detection for station id
        id_col = next(
            (c for c in df.columns if c in ("Station", "Gage_ID_norm", "STATION", "station")),
            df.columns[0],
        )
        lat_col  = next((c for c in df.columns if c.upper() == "LAT"),  None)
        long_col = next((c for c in df.columns if c.upper() in ("LONG", "LON", "LONGITUDE")), None)
        if lat_col and long_col:
            src = pd.DataFrame({
                "Station": df[id_col].astype(str).apply(_normalize_station_id_local),
                "LAT":  pd.to_numeric(df[lat_col],  errors="coerce"),
                "LONG": pd.to_numeric(df[long_col], errors="coerce"),
                "Source": "USGS",
            }).dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")
            parts.append(src)
            print(f"Primary gages loaded: {len(src):,} stations from {primary_path.name}")
        else:
            print(f"WARNING: Could not find LAT/LONG columns in {primary_path}. Available: {list(df.columns)}")
    else:
        print(f"WARNING: Primary gages CSV not found: {primary_path}. No USGS stations will be loaded.")

    # ── Secondary (WAM) control-point source ───────────────────────────────
    # wam_csv=="NONE" means explicitly skip WAM (USGS-only mode)
    if wam_csv != "NONE":
        if wam_csv:
            secondary_path = Path(wam_csv)
            if not secondary_path.is_absolute():
                secondary_path = base_dir / wam_csv
        else:
            secondary_path = base_dir / "HSR1200" / "Streamflow" / "Brazos_new_wam_locations_nhdplus.csv"

        if secondary_path.exists():
            df2 = pd.read_csv(secondary_path, dtype=str)
            id_col2 = next(
                (c for c in df2.columns if c in ("Station", "CPID", "STATION", "station", "cpid")),
                df2.columns[0],
            )
            lat_col2  = next((c for c in df2.columns if c.upper() == "LAT"),  None)
            long_col2 = next((c for c in df2.columns if c.upper() in ("LONG", "LON", "LONGITUDE")), None)
            if lat_col2 and long_col2:
                src2 = pd.DataFrame({
                    "Station": df2[id_col2].astype(str).apply(_normalize_station_id_local),
                    "LAT":  pd.to_numeric(df2[lat_col2],  errors="coerce"),
                    "LONG": pd.to_numeric(df2[long_col2], errors="coerce"),
                    "Source": "WAM",
                }).dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")
                parts.append(src2)
                print(f"WAM control points loaded: {len(src2):,} stations from {secondary_path.name}")
            else:
                print(f"WARNING: Could not find LAT/LONG columns in {secondary_path}. Available: {list(df2.columns)}")
        # silently skip if no --wam-csv and the Brazos fallback doesn't exist
    else:
        print("WAM control points skipped (USGS-only mode).")

    if not parts:
        raise RuntimeError(
            "No gage stations were loaded. "
            "Provide --gages-csv with a CSV containing Station, LAT, LONG columns."
        )

    pts = pd.concat(parts, ignore_index=True)
    pts = pts.sort_values("Source").drop_duplicates(subset=["Station"], keep="first").reset_index(drop=True)
    return pts


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).resolve()
    hsr_dir = base_dir / args.hsr

    # Common output paths used by both full and annual-only modes.
    flow_dir = hsr_dir / "Flowlines"
    gis_dir = hsr_dir / "GIS"
    nlcd_dir = hsr_dir / "NLCD"
    p_dir = hsr_dir / "PRISM" / "Precipitation"
    t_dir = hsr_dir / "PRISM" / "Temperature"
    wu_dir = hsr_dir / "WaterUse"
    sf_dir = hsr_dir / "Streamflow"
    gaged_dir = hsr_dir / "GagedCatchments"
    for d in [flow_dir, gis_dir, nlcd_dir, p_dir, t_dir, wu_dir, sf_dir, gaged_dir]:
        d.mkdir(parents=True, exist_ok=True)

    station_comid_path = flow_dir / "StationComID.csv"
    nhdflowline_path = flow_dir / "nhdflowline.txt"
    nhdflowline_geom_path = flow_dir / "nhdflowline_geometry.gpkg"
    xwalk_path = flow_dir / "GridCodeComID.txt"
    vaa_path = gis_dir / "NHDFlowlineVAA.txt"
    nlcd_path = nlcd_dir / "catchmentattributesnlcd.txt"
    precip_path = p_dir / f"PrismPrecipWY{args.wy}.dat"
    temp_path = t_dir / f"PrismTempAveWY{args.wy}.dat"
    wu_path = wu_dir / "ComID_WU_All.dat"
    dat_path = sf_dir / f"ComIDStationDAMoAnQ{args.wy}.dat"
    da_path = sf_dir / "StationDASqMi.csv"
    catchment_gpkg = base_dir / "inputData" / f"NHDPlusCatchment_{args.ths}.gpkg"
    flowline_gpkg = base_dir / "inputData" / f"NHDFlowline_{args.ths}.gpkg"
    static_output_paths = [
        station_comid_path,
        nhdflowline_path,
        vaa_path,
        nlcd_path,
        wu_path,
        catchment_gpkg,
        flowline_gpkg,
    ]
    yearly_output_paths = [precip_path, temp_path, dat_path, da_path]

    if args.apply and args.annual_only and all(path.exists() for path in yearly_output_paths):
        print(f"Skipping WY{args.wy}; yearly outputs already exist.")
        return

    if args.apply and (not args.annual_only) and all(path.exists() for path in [*static_output_paths, *yearly_output_paths]):
        print(f"Skipping full build for WY{args.wy}; static and yearly outputs already exist.")
        return

    if args.annual_only:
        print(f"Running annual-only update for WY{args.wy} (static network files reused).")

        if not catchment_gpkg.exists():
            raise FileNotFoundError(
                f"Missing catchment geometry: {catchment_gpkg}. Run full Build Network once before annual-only mode."
            )
        if not station_comid_path.exists():
            raise FileNotFoundError(
                f"Missing station map: {station_comid_path}. Run full Build Network once before annual-only mode."
            )

        # Prepare catchment geometry used for zonal PRISM sampling.
        catch_geom = gpd.read_file(catchment_gpkg)
        comid_col = next((c for c in ["NHDPlusID", "ComID", "COMID", "GridCode"] if c in catch_geom.columns), None)
        if comid_col is None:
            raise KeyError(
                f"No ComID-like field in {catchment_gpkg}. Columns: {list(catch_geom.columns)}"
            )
        catch_geom = catch_geom.rename(columns={comid_col: "ComID"}).copy()
        catch_geom["ComID"] = pd.to_numeric(catch_geom["ComID"], errors="coerce").astype("Int64")
        catch_geom = catch_geom.dropna(subset=["ComID", "geometry"]).copy()
        catch_geom["ComID"] = catch_geom["ComID"].astype("int64")

        # Keep only columns required by builders.
        geom_cols = ["ComID", "geometry"]
        if "AreaSqKm" in catch_geom.columns:
            geom_cols.insert(1, "AreaSqKm")
        catch_geom = catch_geom[geom_cols].copy()

        # Build yearly PRISM files.
        if not args.apply:
            print("Dry run (annual-only): validated static prerequisites and yearly output paths.")
            return

        _backup(precip_path, suffix=f".pre{args.ths}.bak")
        _backup(temp_path, suffix=f".pre{args.ths}.bak")
        _backup(dat_path, suffix=f".pre{args.ths}.bak")
        _backup(da_path, suffix=f".pre{args.ths}.bak")

        precip_df, temp_df = _build_catchment_prism(base_dir, catch_geom, args.wy, args.prism_ppt_dir, args.prism_tmean_dir)
        with precip_path.open("w", encoding="utf-8") as f:
            f.write("PRISM precipitation\n")
            f.write("Real basin catchment-zonal dataset\n")
            f.write("GridCode GCAreaSqMi PIn_01..PIn_13\n")
            f.write("Units: source PRISM raster units\n")
            precip_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

        with temp_path.open("w", encoding="utf-8") as f:
            f.write("PRISM temperature\n")
            f.write("Real basin catchment-zonal dataset\n")
            f.write("GridCode TdC_01..TdC_12\n")
            f.write("Units: source PRISM raster units\n")
            temp_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

        station_map = pd.read_csv(station_comid_path)
        if "ComID" in station_map.columns:
            station_map["ComID"] = pd.to_numeric(station_map["ComID"], errors="coerce").astype("Int64")
            station_map = station_map.dropna(subset=["ComID"]).copy()
            station_map["ComID"] = station_map["ComID"].astype("int64")
        dat = _build_streamflow_dat_from_gages_csv(
            base_dir=base_dir,
            wy=args.wy,
            station_map=station_map,
            gages_csv=args.gages_csv,
            units=args.gages_flow_units,
            out_dat_path=dat_path,
            out_da_path=da_path,
        )

        if dat.empty:
            print(f"WARNING: No streamflow records generated for WY{args.wy}; check gages CSV year coverage.")

        print(f"Annual-only update complete for WY{args.wy}")
        print(f"PRISM outputs: {precip_path}, {temp_path}")
        print(f"Streamflow output: {dat_path}")
        return

    basin_shp_arg = str(args.basin_shp).strip()
    basin_shp_exists = bool(basin_shp_arg) and (base_dir / basin_shp_arg).resolve().exists()

    if args.basin_buffer_m < 0:
        raise ValueError("--basin-buffer-m must be >= 0")

    if str(args.hu4s).strip() and not basin_shp_exists:
        basin_gdf = _load_forced_hu4_polygon(base_dir, args.gdb_root, args.hu4s)
        print(f"Using HU4-only build extent from WBDHU4 polygons: {args.hu4s}")
    else:
        basin_gdf = _load_basin_polygon(base_dir, args.basin_shp, args.basin_field, args.basin_value)
    basin_select_gdf = _buffer_polygon_meters(basin_gdf, args.basin_buffer_m)
    if args.basin_buffer_m > 0:
        print(f"Applying basin selection buffer: {args.basin_buffer_m:.1f} meters")

    hu4_gdbs = _discover_hu4_gdbs(base_dir, args.gdb_root, basin_select_gdf, args.hu4s)
    hu4_codes = [code for code, _ in hu4_gdbs]
    station_mask = _load_selected_hu4_mask(hu4_gdbs, basin_select_gdf)
    print(f"Selected HU4 geodatabases: {', '.join(hu4_codes)}")

    flow_tbl, vaa_tbl, flow_geom = _load_basin_flowline_and_vaa(
        hu4_gdbs,
        basin_select_gdf,
        args.ths,
        basin_buffer_m=args.basin_buffer_m,
    )
    comids = flow_tbl["ComID"].to_numpy(dtype=np.int64)
    print(f"Flowlines prepared: {len(flow_tbl):,} reaches")
    catch_geom = _load_basin_catchments(
        hu4_gdbs,
        basin_select_gdf,
        flow_tbl,
        basin_buffer_m=args.basin_buffer_m,
    )
    if not catch_geom.empty:
        print(f"Catchment polygons prepared: {len(catch_geom):,} polygons")
    else:
        print("WARNING: No matching NHDPlus catchment polygons found; falling back to point-sampled raster attributes.")

    station_points = _load_station_points_flexible(base_dir, args.gages_csv, args.wam_csv)
    station_map = _map_stations_to_comid(flow_geom, station_points, station_mask=station_mask)
    print(f"Station mappings prepared: {len(station_map):,} stations")
    print(f"Station map unique COMIDs: {station_map['ComID'].nunique():,} / stations {len(station_map):,}")

    if not args.apply:
        print("Dry run complete. Re-run with --apply to write HSR files and catchment/map geometry.")
        return

    if args.apply:
        for path in [station_comid_path, nhdflowline_path, xwalk_path, vaa_path, nlcd_path, precip_path, temp_path, wu_path, dat_path, da_path]:
            _backup(path, suffix=f".pre{args.ths}.bak")

    flow_tbl.to_csv(nhdflowline_path, index=False)
    flow_export = flow_geom.copy()
    keep_cols = [c for c in ["ComID", "ReachCode", "OrigReachCode", "LengthKm", "geometry"] if c in flow_export.columns]
    flow_export = flow_export[keep_cols]
    flow_export.to_file(nhdflowline_geom_path, driver="GPKG")
    flow_export.to_file(flowline_gpkg, driver="GPKG")
    with xwalk_path.open("w", encoding="utf-8") as f:
        f.write("GridCode,ComID\n")
        pd.DataFrame({"GridCode": comids, "ComID": comids}).to_csv(f, index=False, header=False)
    vaa_tbl[["ComID", "FromNode", "ToNode", "HydroSeq", "Divergence", "StartFlag"]].to_csv(vaa_path, index=False)

    nlcd_raster_path = (base_dir / args.nlcd_raster).resolve()
    if not nlcd_raster_path.exists():
        raise FileNotFoundError(nlcd_raster_path)

    if not catch_geom.empty:
        nlcd_df = _build_catchment_nlcd(catch_geom, nlcd_raster_path)
    else:
        nlcd_df = _build_real_nlcd(flow_geom, nlcd_raster_path)
    nlcd_df.to_csv(nlcd_path, index=False)

    if not catch_geom.empty:
        precip_df, temp_df = _build_catchment_prism(base_dir, catch_geom, args.wy, args.prism_ppt_dir, args.prism_tmean_dir)
    else:
        precip_df, temp_df = _build_real_prism(base_dir, flow_geom, args.wy, args.prism_ppt_dir, args.prism_tmean_dir)
    with precip_path.open("w", encoding="utf-8") as f:
        f.write("PRISM precipitation\n")
        if not catch_geom.empty:
            f.write("Real Brazos full-basin catchment-zonal dataset\n")
        else:
            f.write("Real Brazos full-basin raster-sampled dataset\n")
        f.write("GridCode GCAreaSqMi PIn_01..PIn_13\n")
        f.write("Units: source PRISM raster units\n")
        precip_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

    with temp_path.open("w", encoding="utf-8") as f:
        f.write("PRISM temperature\n")
        if not catch_geom.empty:
            f.write("Real Brazos full-basin catchment-zonal dataset\n")
        else:
            f.write("Real Brazos full-basin raster-sampled dataset\n")
        f.write("GridCode TdC_01..TdC_12\n")
        f.write("Units: source PRISM raster units\n")
        temp_df.to_csv(f, index=False, header=False, sep=" ", float_format="%.6f")

    wu_df = pd.DataFrame({"ComID": comids})
    for month in range(1, 13):
        wu_df[f"WU{month:02d}"] = 0.0
    wu_df.to_csv(wu_path, index=False, header=False, sep=" ")

    station_map.to_csv(station_comid_path, index=False)
    dat = _build_streamflow_dat_from_gages_csv(
        base_dir=base_dir,
        wy=args.wy,
        station_map=station_map,
        gages_csv=args.gages_csv,
        units=args.gages_flow_units,
        out_dat_path=dat_path,
        out_da_path=da_path,
    )

    if dat.empty and dat_path.exists():
        # Fallback for legacy projects that already provide ComIDStationDAMoAnQYYYY.dat.
        dat = _read_streamflow_dat(dat_path)
        sm = station_map[["Station", "ComID"]].rename(columns={"Station": "StaWY", "ComID": "ComID_new"})
        sm["StaWY"] = sm["StaWY"].map(_norm_station_id)
        dat["StaWY"] = dat["StaWY"].map(_norm_station_id)
        dat = dat.merge(sm, on="StaWY", how="left")
        dat["ComIDSta"] = np.where(dat["ComID_new"].notna(), dat["ComID_new"], dat["ComIDSta"])
        dat = dat.drop(columns=["ComID_new"])
        dat["ComIDSta"] = pd.to_numeric(dat["ComIDSta"], errors="coerce").fillna(0).astype(np.int64)
        _write_streamflow_dat(dat, dat_path)

        da = dat[["StaWY", "ComIDSta", "NWISArea"]].drop_duplicates(subset=["StaWY"], keep="first").copy()
        da = da.rename(columns={"StaWY": "Station", "ComIDSta": "ComID", "NWISArea": "DASqMi"})
        da.to_csv(da_path, index=False)
    elif dat.empty:
        da = pd.DataFrame(columns=["Station", "ComID", "DASqMi"])
    else:
        da = pd.read_csv(da_path) if da_path.exists() else pd.DataFrame(columns=["Station", "ComID", "DASqMi"])

    reach_lookup = flow_geom.set_index("ComID")["ReachCode"].to_dict()
    length_km_map = flow_tbl.set_index("ComID")["LengthKm"].to_dict()
    catch_area_map = catch_geom.set_index("ComID")["AreaSqKm"].to_dict() if not catch_geom.empty else {}
    if not da.empty:
        da = da.copy()
        da["Station"] = da["Station"].map(_norm_station_id)
        da_map = da.set_index("Station")["DASqMi"].to_dict()
    else:
        da_map = {}

    valid_comids = set(flow_tbl["ComID"].astype("int64").tolist())
    upstream_map = _build_vaa_upstream_map(vaa_tbl)
    print(f"VAA upstream topology prepared: {sum(len(v) for v in upstream_map.values()):,} immediate upstream links")

    station_list_path = gaged_dir / "StationList.txt"
    station_ids = station_map["Station"].astype(str).sort_values().tolist()
    station_list_path.write_text("\n".join(station_ids) + "\n", encoding="utf-8")

    upstream_counts: list[int] = []
    for _, row in station_map.iterrows():
        station = row["Station"]
        comid = int(row["ComID"])
        dasqmi = float(da_map.get(station, 1.0))
        members = _find_upstream_comids(comid, upstream_map, valid_comids)
        areas = {
            int(member): float(catch_area_map[member])
            for member in members
            if member in catch_area_map and np.isfinite(float(catch_area_map[member])) and float(catch_area_map[member]) > 0
        }
        if len(areas) < len(members):
            allocated = _allocate_station_member_areas(members, dasqmi, length_km_map)
            for member in members:
                areas.setdefault(int(member), allocated.get(int(member), 0.01))
        upstream_counts.append(len(members))

        out = gaged_dir / f"{station}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write("GridCode,ComID,AreaSqKm,ReachCode\n")
            for member_comid in members:
                area_sqkm = areas.get(member_comid, 0.000001)
                reach = str(reach_lookup.get(member_comid, f"{args.ths}0000000000"))
                f.write(f"{member_comid},{member_comid},{area_sqkm:.6f},{reach}\n")

    if upstream_counts:
        counts = pd.Series(upstream_counts)
        one_reach = int((counts <= 1).sum())
        print(
            "Gaged upstream catchments written: "
            f"stations={len(upstream_counts):,}, min={int(counts.min()):,}, "
            f"median={int(counts.median()):,}, max={int(counts.max()):,}, "
            f"one-reach={one_reach:,}"
        )

    if not catch_geom.empty:
        catch = catch_geom.rename(columns={"ComID": "NHDPlusID"}).copy()
        catch["GridCode"] = catch["NHDPlusID"]
        catch = catch.merge(flow_geom[["ComID", "ReachCode", "OrigReachCode"]], left_on="NHDPlusID", right_on="ComID", how="left")
        catch = catch.drop(columns=[c for c in ["ComID"] if c in catch.columns])
        catch = catch[["NHDPlusID", "GridCode", "AreaSqKm", "ReachCode", "OrigReachCode", "geometry"]]
    else:
        catch = flow_geom.rename(columns={"ComID": "NHDPlusID"}).copy()
        catch["GridCode"] = catch["NHDPlusID"]
        catch = catch[["NHDPlusID", "GridCode", "ReachCode", "OrigReachCode", "geometry"]]
    catch.to_file(catchment_gpkg, driver="GPKG")

    print(f"Wrote basin package for THS {args.ths} / {args.hsr}")
    print(f"Catchment/map geometry: {catchment_gpkg}")
    print(f"NLCD source: {nlcd_raster_path}")
    print(f"PRISM sources: {(base_dir / args.prism_ppt_dir).resolve()} and {(base_dir / args.prism_tmean_dir).resolve()}")


if __name__ == "__main__":
    main()
