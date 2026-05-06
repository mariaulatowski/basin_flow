from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd


def _normalize_station_id(value: object) -> str:
    text = str(value).strip()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild AFINCH Station->COMID mapping from station lat/lon and NHDPlus catchments. "
            "Writes candidate outputs by default and can apply changes with --apply."
        )
    )
    p.add_argument("--base-dir", default=".", help="Workspace base directory")
    p.add_argument("--hsr", default="HSR1200", help="HSR directory name")
    p.add_argument("--ths", default="1201", help="THS code (for summary only)")
    p.add_argument(
        "--catchment",
        default="inputData/NHDPlusCatchment_1201.gpkg",
        help="Catchment geometry with NHDPlusID/GridCode",
    )
    p.add_argument(
        "--station-list",
        default="HSR1201/GagedCatchments/StationList.txt",
        help="Station list used by converted AFINCH",
    )
    p.add_argument(
        "--usgs-monthly",
        default="inputData/inputs/monthly_wide_acft.csv",
        help="USGS monthly wide file with Gage_ID_norm/LAT/LONG",
    )
    p.add_argument(
        "--wam-points",
        default="HSR1200/Streamflow/Brazos_new_wam_locations_nhdplus.csv",
        help="WAM station point file with CPID + lat/lon",
    )
    p.add_argument(
        "--flowline",
        default="HSR1200/Flowlines/nhdflowline.txt",
        help="Flowline table with ComID/ReachCode",
    )
    p.add_argument(
        "--output-dir",
        default="output/brazos_rebuild",
        help="Directory for candidate outputs and diagnostics",
    )
    p.add_argument(
        "--existing-station-comid",
        default="HSR1200/Flowlines/StationComID.csv",
        help="Existing StationComID file for baseline diagnostics",
    )
    p.add_argument(
        "--max-snap-m",
        type=float,
        default=20000.0,
        help="Flag stations snapped farther than this distance (meters)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply candidate StationComID mapping to HSR Flowlines/StationComID.csv",
    )
    p.add_argument(
        "--write-gaged-catchments",
        action="store_true",
        help="When used with --apply, also rewrite HSR1201/GagedCatchments/<Station>.dat",
    )
    return p.parse_args()


def _load_station_list(path: Path) -> set[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return {_normalize_station_id(s) for s in lines if s}


def _load_usgs_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Gage_ID_norm": str})
    req = {"Gage_ID_norm", "LAT", "LONG"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"USGS monthly file missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "Station": df["Gage_ID_norm"].astype(str).map(_normalize_station_id),
            "LAT": pd.to_numeric(df["LAT"], errors="coerce"),
            "LONG": pd.to_numeric(df["LONG"], errors="coerce"),
            "source": "USGS",
        }
    )
    out = out.dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")
    return out


def _load_wam_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"CPID": str})
    rename = {}
    if "lat" in df.columns and "LAT" not in df.columns:
        rename["lat"] = "LAT"
    if "lon" in df.columns and "LONG" not in df.columns:
        rename["lon"] = "LONG"
    if rename:
        df = df.rename(columns=rename)

    req = {"CPID", "LAT", "LONG"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"WAM points file missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "Station": df["CPID"].astype(str).map(_normalize_station_id),
            "LAT": pd.to_numeric(df["LAT"], errors="coerce"),
            "LONG": pd.to_numeric(df["LONG"], errors="coerce"),
            "source": "WAM",
        }
    )
    out = out.dropna(subset=["LAT", "LONG"]).drop_duplicates(subset=["Station"], keep="first")
    return out


def _summarize_station_comid(label: str, df: pd.DataFrame) -> None:
    unique_sta = df["Station"].nunique() if not df.empty else 0
    unique_comid = df["ComID"].nunique() if not df.empty else 0
    ratio = (unique_comid / unique_sta) if unique_sta else 0.0
    top = (
        df.groupby("ComID")["Station"].nunique().sort_values(ascending=False).head(5)
        if not df.empty
        else pd.Series(dtype="int64")
    )

    print(f"[{label}] rows={len(df):,} unique_stations={unique_sta:,} unique_comids={unique_comid:,} ratio={ratio:.4f}")
    if not top.empty:
        print("  top station counts per COMID:")
        for comid, ct in top.items():
            print(f"    COMID {int(comid)}: {int(ct)} stations")


def _load_catchment(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    required = {"NHDPlusID", "GridCode"}
    missing = required - set(gdf.columns)
    if missing:
        raise KeyError(f"Catchment file missing columns: {sorted(missing)}")

    gdf["ComID"] = pd.to_numeric(gdf["NHDPlusID"], errors="coerce")
    gdf["GridCode"] = pd.to_numeric(gdf["GridCode"], errors="coerce")
    gdf = gdf.dropna(subset=["ComID", "GridCode", "geometry"]).copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["ComID"] = gdf["ComID"].astype(np.int64)
    gdf["GridCode"] = gdf["GridCode"].astype(np.int64)
    return gdf


def _build_mapping(
    station_points: pd.DataFrame,
    station_set: set[str],
    catch: gpd.GeoDataFrame,
    max_snap_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = station_points.copy()
    points = points[points["Station"].isin(station_set)].copy()
    points = points.drop_duplicates(subset=["Station"], keep="first")

    pts = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["LONG"], points["LAT"]),
        crs="EPSG:4326",
    )

    if catch.crs is None:
        catch = catch.set_crs("EPSG:4326")

    pts_proj = pts.to_crs(catch.estimate_utm_crs())
    catch_proj = catch.to_crs(pts_proj.crs)

    catch_proj = catch_proj[["ComID", "GridCode", "geometry"]].copy()
    catch_proj["AreaSqKm"] = catch_proj.geometry.area / 1_000_000.0

    within = gpd.sjoin(
        pts_proj[["Station", "source", "LAT", "LONG", "geometry"]],
        catch_proj,
        how="left",
        predicate="within",
    )
    within = within.drop(columns=[c for c in ["index_right"] if c in within.columns])

    miss_mask = within["ComID"].isna()
    nearest = gpd.GeoDataFrame(columns=[])
    if miss_mask.any():
        nearest = gpd.sjoin_nearest(
            pts_proj.loc[miss_mask, ["Station", "source", "LAT", "LONG", "geometry"]],
            catch_proj,
            how="left",
            distance_col="snap_dist_m",
        )
        nearest = nearest.drop(columns=[c for c in ["index_right"] if c in nearest.columns])

        within.loc[miss_mask, ["ComID", "GridCode", "AreaSqKm", "snap_dist_m"]] = nearest[
            ["ComID", "GridCode", "AreaSqKm", "snap_dist_m"]
        ].to_numpy()

    within["method"] = np.where(within["snap_dist_m"].notna(), "nearest", "within")
    within["snap_dist_m"] = pd.to_numeric(within["snap_dist_m"], errors="coerce").fillna(0.0)
    within = within.dropna(subset=["ComID", "GridCode"]).copy()
    within["ComID"] = pd.to_numeric(within["ComID"], errors="coerce").astype(np.int64)
    within["GridCode"] = pd.to_numeric(within["GridCode"], errors="coerce").astype(np.int64)

    station_map = within[["Station", "ComID"]].drop_duplicates(subset=["Station"], keep="first").copy()
    station_map = station_map.sort_values("Station").reset_index(drop=True)

    qc = within[["Station", "source", "LAT", "LONG", "ComID", "GridCode", "AreaSqKm", "method", "snap_dist_m"]].copy()
    qc["flag_far_snap"] = qc["snap_dist_m"] > float(max_snap_m)
    qc = qc.sort_values(["flag_far_snap", "snap_dist_m"], ascending=[False, False]).reset_index(drop=True)

    return station_map, qc


def _write_gaged_catchments(
    gaged_dir: Path,
    mapping_qc: pd.DataFrame,
    flowline_path: Path,
) -> int:
    flow = pd.read_csv(flowline_path)
    req = {"ComID", "ReachCode"}
    missing = req - set(flow.columns)
    if missing:
        raise KeyError(f"Flowline file missing columns: {sorted(missing)}")

    flow["ComID"] = pd.to_numeric(flow["ComID"], errors="coerce")
    flow = flow.dropna(subset=["ComID"]).copy()
    flow["ComID"] = flow["ComID"].astype(np.int64)
    flow = flow.drop_duplicates(subset=["ComID"], keep="first")

    joined = mapping_qc.merge(flow[["ComID", "ReachCode"]], on="ComID", how="left")
    written = 0
    for _, row in joined.drop_duplicates(subset=["Station"], keep="first").iterrows():
        out = gaged_dir / f"{row['Station']}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write("GridCode,ComID,AreaSqKm,ReachCode\n")
            f.write(
                f"{int(row['GridCode'])},{int(row['ComID'])},{float(row['AreaSqKm']):.6f},{str(row.get('ReachCode', ''))}\n"
            )
        written += 1
    return written


def _write_readiness_report(
    report_path: Path,
    current: pd.DataFrame | None,
    candidate: pd.DataFrame,
    qc: pd.DataFrame,
    missing_stations: list[str],
    base_dir: Path,
) -> None:
    current_unique = int(current["ComID"].nunique()) if current is not None and not current.empty else 0
    cand_unique = int(candidate["ComID"].nunique()) if not candidate.empty else 0
    total = int(len(candidate))
    far = int(qc["flag_far_snap"].sum()) if not qc.empty else 0
    far_frac = (far / total) if total else 1.0

    hu4_1205_gdb = base_dir / "inputData" / "nhd_medium_res_gdb" / "NHD_H_1205_HU4_GDB" / "NHD_H_1205_HU4_GDB.gdb"
    hu4_1205_exists = hu4_1205_gdb.exists()

    verdict = "PASS"
    if far_frac > 0.2 or cand_unique < 100:
        verdict = "FAIL"
    elif far_frac > 0.05 or cand_unique < 500:
        verdict = "WARN"

    lines = []
    lines.append("Brazos AFINCH Input Readiness Report")
    lines.append("===================================")
    lines.append("")
    lines.append(f"Verdict: {verdict}")
    lines.append(f"Candidate stations mapped: {total}")
    lines.append(f"Candidate unique COMIDs: {cand_unique}")
    lines.append(f"Current unique COMIDs: {current_unique}")
    lines.append(f"Far snaps (> max threshold): {far} ({far_frac:.1%})")
    lines.append(f"Missing station point locations: {len(missing_stations)}")
    lines.append("")

    if verdict != "PASS":
        lines.append("Interpretation")
        lines.append("--------------")
        lines.append(
            "Current catchment/network domain is not consistent with the station locations expected for Brazos; "
            "station mapping collapses onto very few COMIDs and nearest-snap distances are excessively large."
        )
        lines.append("")

    lines.append("Required Next Steps")
    lines.append("-------------------")
    lines.append("1. Build a Brazos-consistent geometry/network package from the same source domain (single authoritative HU4/HUC scope).")
    if hu4_1205_exists:
        lines.append(f"   - Available source detected: {hu4_1205_gdb}")
    else:
        lines.append("   - HU4 1205 geodatabase not found at expected path; locate/import it before proceeding.")
    lines.append("2. Recreate HSR flowline support files from that same source: nhdflowline.txt, GridCodeComID.txt, NHDFlowlineVAA.txt.")
    lines.append("3. Re-snap stations to the rebuilt Brazos catchments and regenerate StationComID.csv and HSR1201/GagedCatchments/*.dat.")
    lines.append("4. Rebuild ComIDStationDAMoAnQ*.dat so each station record aligns to the rebuilt StationComID mapping.")
    lines.append("5. Rerun converted AFINCH Steps 1-6 and confirm map/export joins without fallback nearest snaps.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).resolve()

    catchment = (base_dir / args.catchment).resolve()
    station_list = (base_dir / args.station_list).resolve()
    usgs_monthly = (base_dir / args.usgs_monthly).resolve()
    wam_points = (base_dir / args.wam_points).resolve()
    out_dir = (base_dir / args.output_dir).resolve()
    existing_station_comid = (base_dir / args.existing_station_comid).resolve()

    hsr_dir = base_dir / args.hsr
    target_station_comid = hsr_dir / "Flowlines" / "StationComID.csv"
    gaged_dir = base_dir / "HSR1201" / "GagedCatchments"
    flowline_path = (base_dir / args.flowline).resolve()

    for p in [catchment, station_list, usgs_monthly, wam_points]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building candidate mapping for THS={args.ths}, HSR={args.hsr}")
    print(f"Base dir: {base_dir}")
    print(f"Catchment: {catchment}")

    station_set = _load_station_list(station_list)
    print(f"Station list entries: {len(station_set):,}")

    usgs = _load_usgs_points(usgs_monthly)
    wam = _load_wam_points(wam_points)

    points = pd.concat([usgs, wam], ignore_index=True)
    points = points.sort_values("source", ascending=True).drop_duplicates(subset=["Station"], keep="first")
    print(f"Point records (deduped): {len(points):,}")

    catch = _load_catchment(catchment)
    print(f"Catchments loaded: {len(catch):,}")

    station_map, qc = _build_mapping(points, station_set, catch, max_snap_m=args.max_snap_m)

    missing_stations = sorted(station_set - set(station_map["Station"]))

    cand_path = out_dir / "StationComID_candidate.csv"
    qc_path = out_dir / "StationComID_candidate_qc.csv"
    missing_path = out_dir / "station_list_missing_from_points.txt"

    station_map.to_csv(cand_path, index=False)
    qc.to_csv(qc_path, index=False)
    missing_path.write_text("\n".join(missing_stations) + ("\n" if missing_stations else ""), encoding="utf-8")

    print(f"Candidate mapping written: {cand_path}")
    print(f"QC table written:        {qc_path}")
    print(f"Missing stations list:   {missing_path}")

    current_df: pd.DataFrame | None = None
    if existing_station_comid.exists():
        current = pd.read_csv(existing_station_comid)
        if {"Station", "ComID"}.issubset(set(current.columns)):
            current = current[["Station", "ComID"]].copy()
            current["Station"] = current["Station"].map(_normalize_station_id)
            current["ComID"] = pd.to_numeric(current["ComID"], errors="coerce")
            current = current.dropna(subset=["ComID"]).copy()
            current["ComID"] = current["ComID"].astype(np.int64)
            current_df = current
            _summarize_station_comid("CURRENT", current)

    _summarize_station_comid("CANDIDATE", station_map)
    far_count = int(qc["flag_far_snap"].sum())
    print(f"QC far-snap count (> {args.max_snap_m:.0f} m): {far_count:,}")
    if missing_stations:
        print(f"Stations with no available point location: {len(missing_stations):,}")

    report_path = out_dir / "brazos_input_readiness_report.txt"
    _write_readiness_report(
        report_path=report_path,
        current=current_df,
        candidate=station_map,
        qc=qc,
        missing_stations=missing_stations,
        base_dir=base_dir,
    )
    print(f"Readiness report:       {report_path}")

    if args.apply:
        backup = target_station_comid.with_suffix(".csv.pre_rebuild.bak")
        if target_station_comid.exists() and not backup.exists():
            shutil.copy2(target_station_comid, backup)
            print(f"Backed up existing StationComID to: {backup}")
        elif backup.exists():
            print(f"Backup already exists: {backup}")

        station_map.to_csv(target_station_comid, index=False)
        print(f"Applied StationComID mapping to: {target_station_comid}")

        if args.write_gaged_catchments:
            gaged_dir.mkdir(parents=True, exist_ok=True)
            written = _write_gaged_catchments(gaged_dir, qc, flowline_path)
            print(f"Rewrote gaged catchment files: {written:,}")


if __name__ == "__main__":
    main()
