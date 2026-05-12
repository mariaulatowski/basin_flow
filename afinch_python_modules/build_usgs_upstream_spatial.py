"""
Build USGS-only upstream gaged catchments from NHD VAA topology.

The basin builder exports line geometry for mapping, not true NHDPlus catchment
polygons. This script therefore treats the GeoPackage as optional metadata and
uses NHDFlowlineVAA as the authority for upstream membership.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


def _build_valid_area_override_map(qy: pd.DataFrame, qy_path: Path) -> dict[int, float]:
    qy = qy.copy()
    qy['ComID'] = pd.to_numeric(qy['ComID'], errors='coerce').astype('Int64')
    qy = qy.dropna(subset=['ComID']).copy()
    qy['ComID'] = qy['ComID'].astype('int64')

    area_sqmi = pd.to_numeric(qy['AreaSqMi'], errors='coerce')
    area_sqmi = area_sqmi[np.isfinite(area_sqmi)]
    if area_sqmi.empty:
        raise ValueError(f"FlowYield override file has no finite AreaSqMi values: {qy_path}")

    unique_area = np.unique(area_sqmi.to_numpy(dtype=float))
    if unique_area.size == 1 and np.isclose(unique_area[0], 1.0):
        raise ValueError(
            "FlowYield area override file contains a constant AreaSqMi=1.0 for all reaches. "
            f"This would turn drainage area into upstream reach count: {qy_path}"
        )

    return dict(
        zip(
            qy['ComID'].tolist(),
            (pd.to_numeric(qy['AreaSqMi'], errors='coerce').fillna(0.0) / 0.386102159).tolist(),
        )
    )


def _norm_station(value: object) -> str:
    s = str(value).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s.lstrip('0') or '0'


def _backup(path: Path, suffix: str = '.pre_spatial_upstream.bak') -> None:
    if path.exists():
        bak = path.with_name(path.name + suffix)
        if not bak.exists():
            shutil.copy2(path, bak)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Build USGS-only upstream gaged catchment files using NHD VAA topology'
    )
    p.add_argument('--base-dir', default='.', help='Workspace base directory')
    p.add_argument('--hsr', default='HSR1200', help='HSR folder name')
    p.add_argument('--wy', type=int, default=2018, help='Water year for area lookup')
    p.add_argument('--apply', action='store_true', help='Write outputs to HSR/GagedCatchments')
    p.add_argument('--catchment-gpkg', default='inputData/NHDPlusCatchment_1200.gpkg',
                   help='Optional NHDPlusCatchment GPKG with COMID field')
    return p.parse_args()


def _build_vaa_upstream_map(vaa_df: pd.DataFrame) -> dict[int, list[int]]:
    """Build map: downstream ComID -> list of immediate upstream ComIDs.

    Topology rule:
    - upstream reach ToNode == downstream reach FromNode
    - upstream HydroSeq > downstream HydroSeq (prevents reverse/invalid links)
    """
    cols = {c.lower(): c for c in vaa_df.columns}
    req = ["comid", "fromnode", "tonode", "hydroseq"]
    missing = [r for r in req if r not in cols]
    if missing:
        raise KeyError(f"VAA missing required columns: {missing}")

    work = vaa_df[[cols["comid"], cols["fromnode"], cols["tonode"], cols["hydroseq"]]].copy()
    work.columns = ["ComID", "FromNode", "ToNode", "HydroSeq"]
    for col in ["ComID", "FromNode", "ToNode", "HydroSeq"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["ComID", "FromNode", "ToNode", "HydroSeq"]).copy()
    work = work[(work["FromNode"] > 0) & (work["ToNode"] > 0)].copy()
    work["ComID"] = work["ComID"].astype("int64")
    work["FromNode"] = work["FromNode"].astype("int64")
    work["ToNode"] = work["ToNode"].astype("int64")

    up = work[["ComID", "ToNode", "HydroSeq"]].rename(
        columns={"ComID": "UpComID", "ToNode": "JoinNode", "HydroSeq": "UpHydroSeq"}
    )
    ds = work[["ComID", "FromNode", "HydroSeq"]].rename(
        columns={"ComID": "DsComID", "FromNode": "JoinNode", "HydroSeq": "DsHydroSeq"}
    )
    edges = up.merge(ds, on="JoinNode", how="inner")
    edges = edges[edges["UpComID"] != edges["DsComID"]]
    edges = edges[edges["UpHydroSeq"] > edges["DsHydroSeq"]]
    edges = edges[["UpComID", "DsComID"]].drop_duplicates()

    upstream_map: dict[int, list[int]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        upstream_map[int(row.DsComID)].append(int(row.UpComID))
    return {k: sorted(v) for k, v in upstream_map.items()}


def _find_all_upstream(start_comid: int, upstream_map: dict[int, list[int]], 
                       all_comids: set[int]) -> set[int]:
    """Recursively find all upstream ComIDs starting from a target COMID."""
    visited = {start_comid}
    queue = deque([start_comid])
    
    while queue:
        current = queue.popleft()
        
        for upstream_comid in upstream_map.get(current, []):
            if upstream_comid not in visited and upstream_comid in all_comids:
                visited.add(upstream_comid)
                queue.append(upstream_comid)
    
    return visited


def _allocate_station_member_areas(members: list[int], dasqmi: float, length_km_map: dict[int, float]) -> dict[int, float]:
    if not members:
        return {}
    total_sqkm = max(0.01, float(dasqmi) / 0.386102159)
    weights = np.array([max(0.0, float(length_km_map.get(comid, 0.0))) for comid in members], dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(len(members), dtype=float)
    areas = total_sqkm * weights / float(weights.sum())
    areas = np.maximum(areas, 0.000001)
    return {int(comid): float(area) for comid, area in zip(members, areas)}


def main() -> None:
    args = _parse_args()
    base = Path(args.base_dir).resolve()
    hsr_dir = base / args.hsr
    
    # Load input files
    station_map_path = hsr_dir / 'Flowlines' / 'StationComID.csv'
    vaa_path = hsr_dir / 'GIS' / 'NHDFlowlineVAA.txt'
    catchment_gpkg_path = base / args.catchment_gpkg
    flow_path = hsr_dir / 'Flowlines' / 'nhdflowline.txt'
    da_path = hsr_dir / 'Streamflow' / 'StationDASqMi.csv'
    gaged_dir = hsr_dir / 'GagedCatchments'
    
    for p in [station_map_path, vaa_path, flow_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
    
    print("Loading input data...")
    
    # Load and filter to USGS stations
    station_map = pd.read_csv(station_map_path)
    station_map['StationN'] = station_map['Station'].map(_norm_station)
    usgs = station_map[station_map['Source'].astype(str).str.upper() == 'USGS'].copy()
    usgs['ComID'] = pd.to_numeric(usgs['ComID'], errors='coerce').astype('Int64')
    usgs = usgs.dropna(subset=['ComID']).copy()
    usgs['ComID'] = usgs['ComID'].astype('int64')
    usgs = usgs.drop_duplicates(subset=['StationN'], keep='first').sort_values('StationN').reset_index(drop=True)
    
    print(f"USGS stations: {len(usgs)}")
    
    # Load VAA
    print("Loading VAA...")
    vaa = pd.read_csv(vaa_path, sep=',')
    all_comids = set(vaa['ComID'].astype('int64').unique())
    upstream_map = _build_vaa_upstream_map(vaa)
    print(f"Immediate upstream links: {sum(len(v) for v in upstream_map.values()):,}")
    
    # Load flowline metadata
    flow = pd.read_csv(flow_path)
    flow['ComID'] = pd.to_numeric(flow['ComID'], errors='coerce').astype('Int64')
    flow = flow.dropna(subset=['ComID']).copy()
    flow['ComID'] = flow['ComID'].astype('int64')
    reach_map = dict(zip(flow['ComID'].tolist(), flow['ReachCode'].astype(str).tolist()))
    length_km_map = dict(zip(flow['ComID'].tolist(), pd.to_numeric(flow['LengthKm'], errors='coerce').fillna(1.0).tolist()))
    flow_comids = set(flow['ComID'].unique())

    if da_path.exists():
        da = pd.read_csv(da_path)
        da['StationN'] = da['Station'].map(_norm_station)
        da['DASqMi'] = pd.to_numeric(da['DASqMi'], errors='coerce')
        da_map = dict(zip(da['StationN'], da['DASqMi']))
    else:
        da_map = {}
        print(f"WARNING: {da_path} not found; station member areas will use 1.0 sq mi totals")

    # Do not use model output as build input here. Preserve each station's
    # drainage-area total and distribute it across upstream members by length.
    area_sqkm_map: dict[int, float] = {}
    print("Using length-weighted station drainage areas for gaged catchment rows")
    
    # Dry run: compute upstream sets
    print("\nComputing upstream basin membership for each USGS gage...")
    sizes = []
    ups_dict = {}
    
    for idx, row in usgs.iterrows():
        gage_comid = int(row['ComID'])
        all_upstream = _find_all_upstream(gage_comid, upstream_map, all_comids)
        upstream_with_catchments = [c for c in all_upstream if c in flow_comids]
        ups_dict[gage_comid] = sorted(upstream_with_catchments)
        sizes.append(len(upstream_with_catchments))
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(usgs)} gages")
    
    print(f"\nDry run complete.")
    print(f"Upstream member counts: min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}")
    print(f"Total unique upstream reaches across all gages: {len(set().union(*ups_dict.values()))}")
    
    if not args.apply:
        return
    
    # Apply: write to .dat files
    print("\nApplying upstream gaged catchments...")
    gaged_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup existing files
    station_list_path = gaged_dir / 'StationList.txt'
    _backup(station_list_path)
    for _, row in usgs.iterrows():
        _backup(gaged_dir / f"{row['StationN']}.dat")
    
    # Write new StationList
    station_ids = usgs['StationN'].astype(str).tolist()
    station_list_path.write_text('\n'.join(station_ids) + '\n', encoding='utf-8')
    
    # Write .dat files
    for _, row in usgs.iterrows():
        sta = str(row['StationN'])
        gage_comid = int(row['ComID'])
        members = ups_dict[gage_comid]
        dasqmi = float(da_map.get(sta, 1.0))
        allocated_areas = _allocate_station_member_areas(members, dasqmi, length_km_map)
        
        lines = ['GridCode,ComID,AreaSqKm,ReachCode']
        for comid in members:
            area_sqkm = float(area_sqkm_map.get(comid, allocated_areas.get(comid, 0.01)))
            area_sqkm = max(0.01, area_sqkm)
            rc = reach_map.get(comid, '12000000000000')
            lines.append(f'{comid},{comid},{area_sqkm:.6f},{rc}')
        
        out_path = gaged_dir / f'{sta}.dat'
        out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    
    print(f"USGS upstream gaged catchments written.")
    print(f"StationList: {station_list_path}")
    print(f"Upstream member counts: min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}")


if __name__ == '__main__':
    main()
