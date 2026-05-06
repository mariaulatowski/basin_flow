from __future__ import annotations

import argparse
import shutil
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


def _norm_station(value: object) -> str:
    s = str(value).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s.lstrip('0') or '0'


def _backup(path: Path, suffix: str = '.pre_usgs_only.bak') -> None:
    if path.exists():
        bak = path.with_name(path.name + suffix)
        if not bak.exists():
            shutil.copy2(path, bak)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build USGS-only upstream gaged catchment files for converted AFINCH')
    p.add_argument('--base-dir', default='.', help='Workspace base directory')
    p.add_argument('--hsr', default='HSR1200', help='HSR folder name')
    p.add_argument('--wy', type=int, default=2018, help='Water year used for area lookup file')
    p.add_argument('--apply', action='store_true', help='Write outputs into HSR/GagedCatchments')
    return p.parse_args()


def _upstream_sets_for_targets(vaa: pd.DataFrame, target_comids: list[int]) -> dict[int, list[int]]:
    from_node = pd.to_numeric(vaa['FromNode'], errors='coerce').fillna(-1).astype('int64').to_numpy()
    to_node = pd.to_numeric(vaa['ToNode'], errors='coerce').fillna(-1).astype('int64').to_numpy()
    comids = pd.to_numeric(vaa['ComID'], errors='coerce').fillna(-1).astype('int64').to_numpy()

    # Map ToNode -> list of indices (reaches that END at this node)
    # These are the reaches flowing INTO a given node (upstream direction)
    by_to_node: dict[int, list[int]] = defaultdict(list)
    for idx, tn in enumerate(to_node):
        by_to_node[int(tn)].append(idx)

    comid_to_idx = {int(c): i for i, c in enumerate(comids) if int(c) > 0}
    out: dict[int, list[int]] = {}
    for start_comid in target_comids:
        start_idx = comid_to_idx.get(int(start_comid))
        if start_idx is None:
            out[int(start_comid)] = [int(start_comid)]
            continue

        seen = set()
        q = deque([start_idx])
        while q:
            idx = q.popleft()
            if idx in seen:
                continue
            seen.add(idx)
            # Get FromNode of this reach (start of reach)
            fn = int(from_node[idx])
            # Find all reaches that flow INTO this FromNode (ends where this one starts)
            for up_idx in by_to_node.get(fn, []):
                if up_idx != idx:
                    q.append(up_idx)

        members = sorted(int(comids[i]) for i in seen if int(comids[i]) > 0)
        out[int(start_comid)] = members if members else [int(start_comid)]
    return out


def main() -> None:
    args = _parse_args()
    base = Path(args.base_dir).resolve()
    hsr_dir = base / args.hsr

    station_map_path = hsr_dir / 'Flowlines' / 'StationComID.csv'
    vaa_path = hsr_dir / 'GIS' / 'NHDFlowlineVAA.txt'
    flow_path = hsr_dir / 'Flowlines' / 'nhdflowline.txt'
    qy_path = hsr_dir / 'Output' / 'FlowYield' / f'QY{args.hsr}WY{args.wy}.csv'
    gaged_dir = hsr_dir / 'GagedCatchments'

    for p in [station_map_path, vaa_path, flow_path, qy_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    station_map = pd.read_csv(station_map_path)
    station_map['StationN'] = station_map['Station'].map(_norm_station)
    usgs = station_map[station_map['Source'].astype(str).str.upper() == 'USGS'].copy()
    usgs['ComID'] = pd.to_numeric(usgs['ComID'], errors='coerce').astype('Int64')
    usgs = usgs.dropna(subset=['ComID']).copy()
    usgs['ComID'] = usgs['ComID'].astype('int64')
    usgs = usgs.drop_duplicates(subset=['StationN'], keep='first').sort_values('StationN').reset_index(drop=True)

    vaa = pd.read_csv(vaa_path)
    flow = pd.read_csv(flow_path)
    flow['ComID'] = pd.to_numeric(flow['ComID'], errors='coerce').astype('Int64')
    flow = flow.dropna(subset=['ComID']).copy()
    flow['ComID'] = flow['ComID'].astype('int64')
    reach_map = dict(zip(flow['ComID'].tolist(), flow['ReachCode'].astype(str).tolist()))

    qy = pd.read_csv(qy_path)
    qy['ComID'] = pd.to_numeric(qy['ComID'], errors='coerce').astype('Int64')
    qy = qy.dropna(subset=['ComID']).copy()
    qy['ComID'] = qy['ComID'].astype('int64')
    area_sqkm_map = dict(zip(qy['ComID'].tolist(), (pd.to_numeric(qy['AreaSqMi'], errors='coerce').fillna(0.0) / 0.386102159).tolist()))

    target_comids = usgs['ComID'].astype(int).tolist()
    ups = _upstream_sets_for_targets(vaa, target_comids)

    print(f'USGS stations selected: {len(usgs):,}')
    if len(usgs) == 0:
        raise ValueError('No USGS stations found in StationComID.csv')

    if not args.apply:
        sizes = []
        for _, row in usgs.iterrows():
            sizes.append(len(ups.get(int(row['ComID']), [int(row['ComID'])])))
        print('Dry run complete.')
        print(f'Upstream member counts: min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}')
        return

    gaged_dir.mkdir(parents=True, exist_ok=True)
    station_list_path = gaged_dir / 'StationList.txt'
    _backup(station_list_path)

    # Backup existing USGS station files before overwrite.
    for _, row in usgs.iterrows():
        _backup(gaged_dir / f"{row['StationN']}.dat")

    station_ids = usgs['StationN'].astype(str).tolist()
    station_list_path.write_text('\n'.join(station_ids) + '\n', encoding='utf-8')

    sizes = []
    for _, row in usgs.iterrows():
        sta = str(row['StationN'])
        comid0 = int(row['ComID'])
        members = ups.get(comid0, [comid0])
        sizes.append(len(members))

        lines = ['GridCode,ComID,AreaSqKm,ReachCode']
        for comid in members:
            area_sqkm = float(area_sqkm_map.get(comid, 0.01))
            area_sqkm = max(0.01, area_sqkm)
            rc = reach_map.get(comid, '12000000000000')
            lines.append(f'{comid},{comid},{area_sqkm:.6f},{rc}')

        out_path = gaged_dir / f'{sta}.dat'
        out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('USGS upstream gaged catchments written.')
    print(f'StationList: {station_list_path}')
    print(f'Upstream member counts: min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}')


if __name__ == '__main__':
    main()
