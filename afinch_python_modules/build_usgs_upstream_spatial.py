"""
Build USGS-only upstream gaged catchments using spatial relationships from NHDPlusCatchment geometry.

Instead of VAA node traversal, this uses the catchment polygon layer to determine upstream membership
by analyzing the spatial/topological structure. A catchment is "upstream" of a gage if its polygon 
is part of the accumulated drainage area to that gage.

Approach:
1. Load gage points and their snapped COMIDs from StationComID.csv
2. Load NHDPlusCatchment GPKG (polygons with COMID field)
3. Load NHDFlowlineVAA to determine downstream relationships
4. For each gage COMID, recursively find all upstream COMIDs using VAA
5. Filter to only those upstream COMIDs that have catchment polygons
6. Write to .dat files
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import pandas as pd


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
        description='Build USGS-only upstream gaged catchment files using spatial NHDPlusCatchment geometry'
    )
    p.add_argument('--base-dir', default='.', help='Workspace base directory')
    p.add_argument('--hsr', default='HSR1200', help='HSR folder name')
    p.add_argument('--wy', type=int, default=2018, help='Water year for area lookup')
    p.add_argument('--apply', action='store_true', help='Write outputs to HSR/GagedCatchments')
    p.add_argument('--catchment-gpkg', default='inputData/NHDPlusCatchment_1200.gpkg',
                   help='Path to NHDPlusCatchment GPKG with COMID field')
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
    work["ComID"] = pd.to_numeric(work["ComID"], errors="coerce").astype("Int64")
    work["FromNode"] = pd.to_numeric(work["FromNode"], errors="coerce").astype("Int64")
    work["ToNode"] = pd.to_numeric(work["ToNode"], errors="coerce").astype("Int64")
    work["HydroSeq"] = pd.to_numeric(work["HydroSeq"], errors="coerce")
    work = work.dropna(subset=["ComID", "FromNode", "ToNode", "HydroSeq"]).copy()
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
    for _, row in edges.iterrows():
        upstream_map[int(row["DsComID"])].append(int(row["UpComID"]))
    return dict(upstream_map)


def _find_all_upstream(start_comid: int, upstream_map: dict[int, list[int]], 
                      all_comids: set[int], max_iterations: int = 10000) -> set[int]:
    """Recursively find all upstream ComIDs starting from a target COMID."""
    visited = {start_comid}
    queue = deque([start_comid])
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        current = queue.popleft()
        
        for upstream_comid in upstream_map.get(current, []):
            if upstream_comid not in visited and upstream_comid in all_comids:
                visited.add(upstream_comid)
                queue.append(upstream_comid)
    
    return visited


def main() -> None:
    args = _parse_args()
    base = Path(args.base_dir).resolve()
    hsr_dir = base / args.hsr
    
    # Load input files
    station_map_path = hsr_dir / 'Flowlines' / 'StationComID.csv'
    vaa_path = hsr_dir / 'GIS' / 'NHDFlowlineVAA.txt'
    catchment_gpkg_path = base / args.catchment_gpkg
    flow_path = hsr_dir / 'Flowlines' / 'nhdflowline.txt'
    qy_path = hsr_dir / 'Output' / 'FlowYield' / f'QY{args.hsr}WY{args.wy}.csv'
    gaged_dir = hsr_dir / 'GagedCatchments'
    
    for p in [station_map_path, vaa_path, catchment_gpkg_path, flow_path]:
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
    
    # Load catchments
    print("Loading NHDPlusCatchment...")
    catchments = gpd.read_file(catchment_gpkg_path)
    if 'COMID' not in catchments.columns and 'ComID' not in catchments.columns:
        # Try other common field names
        id_cols = [
            c
            for c in catchments.columns
            if 'comid' in c.lower() or 'featureid' in c.lower() or 'nhdplusid' in c.lower()
        ]
        if not id_cols:
            raise KeyError(f"No ComID-like column in catchments. Columns: {catchments.columns.tolist()}")
        id_col = id_cols[0]
        catchments = catchments.rename(columns={id_col: 'COMID'})
    
    catchments['COMID'] = pd.to_numeric(catchments['COMID'], errors='coerce')
    catchments = catchments.dropna(subset=['COMID', 'geometry']).copy()
    catchments['COMID'] = catchments['COMID'].astype('int64')
    catchment_comids = set(catchments['COMID'].unique())
    print(f"Catchments available: {len(catchment_comids)}")

    # Default area source: catchment polygon area (sq km)
    catch_area = catchments.copy()
    if catch_area.crs is None:
        catch_area = catch_area.set_crs('EPSG:4269')
    catch_area_proj = catch_area.to_crs(5070)
    catch_area['AreaSqKmGeom'] = catch_area_proj.geometry.area / 1_000_000.0
    area_sqkm_map = dict(zip(catch_area['COMID'].tolist(), catch_area['AreaSqKmGeom'].astype(float).tolist()))
    
    # Load flowline metadata
    flow = pd.read_csv(flow_path)
    flow['ComID'] = pd.to_numeric(flow['ComID'], errors='coerce').astype('Int64')
    flow = flow.dropna(subset=['ComID']).copy()
    flow['ComID'] = flow['ComID'].astype('int64')
    reach_map = dict(zip(flow['ComID'].tolist(), flow['ReachCode'].astype(str).tolist()))
    
    # Optional area source override: FlowYield QY file (AreaSqMi -> AreaSqKm)
    if qy_path.exists():
        qy = pd.read_csv(qy_path)
        qy['ComID'] = pd.to_numeric(qy['ComID'], errors='coerce').astype('Int64')
        qy = qy.dropna(subset=['ComID']).copy()
        qy['ComID'] = qy['ComID'].astype('int64')
        qy_area = dict(
            zip(
                qy['ComID'].tolist(),
                (pd.to_numeric(qy['AreaSqMi'], errors='coerce').fillna(0.0) / 0.386102159).tolist(),
            )
        )
        area_sqkm_map.update(qy_area)
        print(f"Loaded area overrides from {qy_path.name}: {len(qy_area):,} reaches")
    else:
        print(f"WARNING: {qy_path} not found; using catchment polygon areas for AreaSqKm")
    
    # Dry run: compute upstream sets
    print("\nComputing upstream basin membership for each USGS gage...")
    sizes = []
    ups_dict = {}
    
    for idx, row in usgs.iterrows():
        gage_comid = int(row['ComID'])
        # Find all upstream, but only include those with catchment polygons
        all_upstream = _find_all_upstream(gage_comid, upstream_map, all_comids)
        # Filter to those that have catchment polygons
        upstream_with_catchments = [c for c in all_upstream if c in catchment_comids]
        ups_dict[gage_comid] = sorted(upstream_with_catchments)
        sizes.append(len(upstream_with_catchments))
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(usgs)} gages")
    
    print(f"\nDry run complete.")
    print(f"Upstream member counts (with catchments): min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}")
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
        
        lines = ['GridCode,ComID,AreaSqKm,ReachCode']
        for comid in members:
            area_sqkm = float(area_sqkm_map.get(comid, 0.01))
            area_sqkm = max(0.01, area_sqkm)
            rc = reach_map.get(comid, '12000000000000')
            lines.append(f'{comid},{comid},{area_sqkm:.6f},{rc}')
        
        out_path = gaged_dir / f'{sta}.dat'
        out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    
    print(f"USGS upstream gaged catchments written.")
    print(f"StationList: {station_list_path}")
    print(f"Upstream member counts (with catchments): min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}")


if __name__ == '__main__':
    main()
