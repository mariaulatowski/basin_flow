from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_station(v: object) -> str:
    s = str(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s.lstrip('0') or '0'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build ComIDStationDAMoAnQYYYY.dat files from monthly_wide_cfs.csv')
    p.add_argument('--base-dir', default='.', help='Workspace base directory')
    p.add_argument('--hsr', default='HSR1200', help='HSR folder')
    p.add_argument(
        '--monthly-cfs-csv',
        default=r'C:/Users/mu3575/Documents/GSA/Brazos/brazos_pipeline_outputs/04_monthly_and_flo/monthly_wide_cfs.csv',
        help='Path to monthly_wide_cfs.csv',
    )
    p.add_argument('--start-year', type=int, default=2010, help='Start calendar year')
    p.add_argument('--end-year', type=int, default=2024, help='End calendar year')
    p.add_argument('--usgs-only', action='store_true', help='Use only USGS stations from StationComID.csv')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base = Path(args.base_dir).resolve()
    hsr_dir = base / args.hsr

    station_map_path = hsr_dir / 'Flowlines' / 'StationComID.csv'
    station_da_path = hsr_dir / 'Streamflow' / 'StationDASqMi.csv'
    out_dir = hsr_dir / 'Streamflow'
    src_csv = Path(args.monthly_cfs_csv)

    if not station_map_path.exists():
        raise FileNotFoundError(station_map_path)
    if not src_csv.exists():
        raise FileNotFoundError(src_csv)

    station_map = pd.read_csv(station_map_path)
    station_map['StationN'] = station_map['Station'].map(_norm_station)
    if args.usgs_only and 'Source' in station_map.columns:
        station_map = station_map[station_map['Source'].astype(str).str.upper() == 'USGS'].copy()

    station_map = station_map.drop_duplicates(subset=['StationN'], keep='first').copy()
    station_to_comid = {
        str(r['StationN']): int(pd.to_numeric(r['ComID'], errors='coerce'))
        for _, r in station_map.iterrows()
        if pd.notna(pd.to_numeric(r['ComID'], errors='coerce'))
    }

    # Optional drainage-area map.
    station_to_da: dict[str, float] = {}
    if station_da_path.exists():
        da = pd.read_csv(station_da_path)
        if {'Station', 'DASqMi'}.issubset(da.columns):
            da['StationN'] = da['Station'].map(_norm_station)
            for _, r in da.iterrows():
                try:
                    station_to_da[str(r['StationN'])] = float(r['DASqMi'])
                except Exception:
                    continue

    src = pd.read_csv(src_csv)
    req = {'Gage_ID_norm', 'Year', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'}
    missing = req - set(src.columns)
    if missing:
        raise KeyError(f'monthly_wide_cfs.csv missing columns: {sorted(missing)}')

    src['StationN'] = src['Gage_ID_norm'].map(_norm_station)
    month_order = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']

    years = list(range(args.start_year, args.end_year + 1))
    print(f'Building streamflow DAT files for years {years[0]}..{years[-1]}')
    print(f'Stations in mapping: {len(station_to_comid)}')

    for yr in years:
        sub = src[src['Year'] == yr].copy()
        sub = sub.drop_duplicates(subset=['StationN'], keep='first')

        rows = []
        matched = 0
        for sta, comid in station_to_comid.items():
            r = sub[sub['StationN'] == sta]
            if r.empty:
                continue
            rr = r.iloc[0]
            q12 = [float(pd.to_numeric(rr[m], errors='coerce')) if pd.notna(pd.to_numeric(rr[m], errors='coerce')) else 0.0 for m in month_order]
            q13 = float(np.mean(q12))
            da = float(station_to_da.get(sta, 5.79))
            rows.append([int(comid), sta, da, *q12, q13])
            matched += 1

        cols = ['ComIDSta', 'StaWY', 'NWISArea'] + [f'Q{i:02d}' for i in range(1, 14)]
        out_df = pd.DataFrame(rows, columns=cols)
        out_path = out_dir / f'ComIDStationDAMoAnQ{yr}.dat'
        out_df.to_csv(out_path, index=False, header=False, sep=' ')
        print(f'  WY{yr}: wrote {len(out_df)} rows -> {out_path.name}')

    print('Done.')


if __name__ == '__main__':
    main()
