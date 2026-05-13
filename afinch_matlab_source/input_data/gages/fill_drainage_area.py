import pandas as pd
import requests
import time
from pathlib import Path
from io import StringIO
import geopandas as gpd
from shapely.geometry import Point

# Input and output paths
infile = Path(r"c:/Users/mu3575/Documents/WAM/afinch_matlab_source/input_data/gages/all_gage_data_cfs_by_year.csv")
outfile = infile.parent / (infile.stem + "_DA.csv")
fallback_file = Path(r"C:/Users/mu3575/Box/Data_SurfaceWater/afinch_model/snapped_points_final_reviewed_dasqmi_fix.csv")
nhd_gdb_root = Path(r"C:/Users/mu3575/Documents/WAM/inputData/texas_nhdplusgrb/_extracted_gdb")


def _norm_gage_id(value):
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    text = text.split(".")[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return digits
    return None


def _build_fallback_da_map(path: Path):
    """Build fallback DA map keyed by normalized 8-digit gage id."""
    if not path.exists():
        print(f"Fallback file not found: {path}")
        return {}

    try:
        cp = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception as exc:
        print(f"Failed to read fallback file: {path} ({exc})")
        return {}

    if "DASqMi" not in cp.columns:
        print(f"Fallback file missing DASqMi column: {path}")
        return {}

    id_candidates = ["ID", "Gage_ID", "USGS_ID", "StationID", "CPID"]
    use_cols = [c for c in id_candidates if c in cp.columns]
    if not use_cols:
        print(f"Fallback file has no supported ID columns. Found: {list(cp.columns)}")
        return {}

    cp = cp.copy()
    cp["DASqMi_num"] = pd.to_numeric(cp["DASqMi"], errors="coerce")
    cp = cp[cp["DASqMi_num"].notna()]
    if cp.empty:
        print("Fallback file has no numeric DASqMi values.")
        return {}

    # Build a long lookup table from all candidate ID columns and a last-8-digit fallback.
    rows = []
    for col in use_cols:
        base = cp[[col, "DASqMi_num"]].copy()
        base["raw"] = base[col].astype(str)
        base = base.drop(columns=[col])

        exact = base.copy()
        exact["gage"] = exact["raw"].map(_norm_gage_id)
        exact = exact[exact["gage"].notna()]
        rows.append(exact[["gage", "DASqMi_num"]])

        tail = base.copy()
        tail["raw_digits"] = tail["raw"].str.replace(r"\\.0$", "", regex=True).str.replace(r"\\D", "", regex=True)
        tail["gage"] = tail["raw_digits"].str.extract(r"(\\d{8})$", expand=False)
        tail = tail[tail["gage"].notna()]
        rows.append(tail[["gage", "DASqMi_num"]])

    if not rows:
        return {}

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.dropna(subset=["gage", "DASqMi_num"])
    if long_df.empty:
        return {}

    # Use median DA where duplicate keys exist.
    grouped = long_df.groupby("gage", as_index=False)["DASqMi_num"].median()
    return dict(zip(grouped["gage"], grouped["DASqMi_num"]))


def _build_vaa_lookup_by_gdb(gdb_paths):
    """Read NHDPlusFlowlineVAA tables once and cache NHDPlusID->TotDASqKm."""
    lookup = {}
    for gdb in gdb_paths:
        try:
            vaa = gpd.read_file(gdb, layer="NHDPlusFlowlineVAA")
            if "NHDPlusID" not in vaa.columns or "TotDASqKm" not in vaa.columns:
                continue
            slim = vaa[["NHDPlusID", "TotDASqKm"]].copy()
            slim["NHDPlusID"] = pd.to_numeric(slim["NHDPlusID"], errors="coerce")
            slim["TotDASqKm"] = pd.to_numeric(slim["TotDASqKm"], errors="coerce")
            slim = slim.dropna(subset=["NHDPlusID", "TotDASqKm"])
            lookup[gdb] = dict(zip(slim["NHDPlusID"].astype(float), slim["TotDASqKm"].astype(float)))
        except Exception:
            continue
    return lookup


def _nearest_nhd_total_da_sqmi(lat, lon, gdb_paths, vaa_lookup_by_gdb):
    """Find nearest NHD flowline and return its total upstream DA (sq mi)."""
    if not pd.notna(lat) or not pd.notna(lon):
        return None

    pt = Point(float(lon), float(lat))
    pt_gdf = gpd.GeoDataFrame(geometry=[pt], crs="EPSG:4326").to_crs("EPSG:5070")
    pt_proj = pt_gdf.geometry.iloc[0]

    best = None
    search_buffers_deg = [0.15, 0.35, 0.75]
    for radius_deg in search_buffers_deg:
        minx, miny, maxx, maxy = (
            float(lon) - radius_deg,
            float(lat) - radius_deg,
            float(lon) + radius_deg,
            float(lat) + radius_deg,
        )

        for gdb in gdb_paths:
            try:
                fl = gpd.read_file(gdb, layer="NHDFlowline", bbox=(minx, miny, maxx, maxy))
            except Exception:
                continue

            if fl.empty or "NHDPlusID" not in fl.columns:
                continue

            fl = fl[["NHDPlusID", "geometry"]].copy()
            fl = fl.dropna(subset=["NHDPlusID", "geometry"])
            if fl.empty:
                continue

            try:
                fl_proj = fl.to_crs("EPSG:5070")
            except Exception:
                continue

            dists = fl_proj.geometry.distance(pt_proj)
            if dists.empty:
                continue
            idx = dists.idxmin()
            dist_m = float(dists.loc[idx])
            nhd_id = pd.to_numeric(fl.loc[idx, "NHDPlusID"], errors="coerce")
            if not pd.notna(nhd_id):
                continue

            if (best is None) or (dist_m < best["dist_m"]):
                best = {
                    "dist_m": dist_m,
                    "gdb": gdb,
                    "nhd_id": float(nhd_id),
                }

        # Stop expanding search once we found at least one candidate.
        if best is not None:
            break

    if best is None:
        return None

    # Conservative quality guard for bad snaps.
    if best["dist_m"] > 20000.0:
        return None

    vaa_map = vaa_lookup_by_gdb.get(best["gdb"], {})
    tot_da_sqkm = vaa_map.get(best["nhd_id"])
    if tot_da_sqkm is None or not pd.notna(tot_da_sqkm):
        return None

    sqmi_per_sqkm = 0.386102159
    return float(tot_da_sqkm) * sqmi_per_sqkm

def fetch_drainage_area_usgs(site_no):
    """Query USGS NWIS Site Service for drainage area in square miles."""
    # siteOutput=expanded ensures drainage-area fields are included when available.
    url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded&sites={site_no}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        rdb = pd.read_csv(
            StringIO(resp.text),
            sep="\t",
            comment="#",
            dtype=str,
            engine="python",
        )

        if rdb.empty:
            return None

        # NWIS includes a format-definition row (e.g., 5s, 15s, ...).
        if "agency_cd" in rdb.columns:
            rdb = rdb[rdb["agency_cd"].astype(str).str.upper() != "5S"]
        if rdb.empty:
            return None

        row = rdb.iloc[0]
        drain = row.get("drain_area_va")
        contrib = row.get("contrib_drain_area_va")

        for candidate in (drain, contrib):
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text == "" or text.lower() == "nan":
                continue
            try:
                return float(text)
            except Exception:
                continue
        return None
    except Exception:
        return None

def main():
    df = pd.read_csv(infile, dtype={"Gage_ID": str})
    df["Gage_ID"] = df["Gage_ID"].astype(str).str.zfill(8).str.split(".").str[0]
    unique_gages = df["Gage_ID"].unique()
    fallback_map = _build_fallback_da_map(fallback_file)
    print(f"Loaded fallback DA keys: {len(fallback_map)} from {fallback_file}")

    gdb_paths = sorted(nhd_gdb_root.glob("*/**/*.gdb"))
    if not gdb_paths:
        gdb_paths = sorted(nhd_gdb_root.glob("*/NHDPLUS_H_*_GDB.gdb"))
    print(f"Found NHD geodatabases: {len(gdb_paths)}")
    vaa_lookup_by_gdb = _build_vaa_lookup_by_gdb(gdb_paths)
    print(f"Loaded VAA lookup tables: {len(vaa_lookup_by_gdb)}")

    gage_latlon = (
        df.groupby("Gage_ID", as_index=False)
        .agg({"LAT": "first", "LONG": "first"})
        .set_index("Gage_ID")
    )

    da_map = {}
    missing = []
    filled_from_usgs = 0
    filled_from_fallback = 0
    filled_from_nhd = 0
    for gage in unique_gages:
        da = fetch_drainage_area_usgs(gage)
        if da is not None:
            da_map[gage] = da
            filled_from_usgs += 1
        else:
            fallback_da = fallback_map.get(gage)
            if fallback_da is not None and pd.notna(fallback_da):
                da_map[gage] = float(fallback_da)
                filled_from_fallback += 1
            else:
                lat = pd.to_numeric(gage_latlon.at[gage, "LAT"], errors="coerce") if gage in gage_latlon.index else None
                lon = pd.to_numeric(gage_latlon.at[gage, "LONG"], errors="coerce") if gage in gage_latlon.index else None
                nhd_da = _nearest_nhd_total_da_sqmi(lat, lon, gdb_paths, vaa_lookup_by_gdb)
                if nhd_da is not None:
                    da_map[gage] = float(nhd_da)
                    filled_from_nhd += 1
                else:
                    da_map[gage] = None
                    missing.append(gage)
        time.sleep(0.5)  # polite delay

    df["DA_sqmi"] = df["Gage_ID"].map(da_map)
    df.to_csv(outfile, index=False)
    print(f"Saved with DA column: {outfile}")
    print(f"Filled from USGS: {filled_from_usgs}")
    print(f"Filled from fallback file: {filled_from_fallback}")
    print(f"Filled from NHD nearest flowline: {filled_from_nhd}")
    print(f"Missing unique gages after fallback: {len(missing)}")
    if missing:
        print("Gages with missing DA:", missing)
        print("Rows with missing DA:")
        print(df[df["DA_sqmi"].isna()][["Gage_ID", "Station_Name"]].drop_duplicates())
    else:
        print("All gages had DA values.")

if __name__ == "__main__":
    main()
