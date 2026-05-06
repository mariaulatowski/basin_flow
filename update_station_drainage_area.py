import pandas as pd
import dataretrieval.nwis as nwis

# Paths
station_file = r"HSR1200/Flowlines/StationComID.csv"
da_file = r"HSR1200/Streamflow/StationDASqMi.csv"

# Read station list
stations = pd.read_csv(station_file)
data = []

for idx, row in stations.iterrows():
    site = str(row['Station']).zfill(8)
    try:
        info, _ = nwis.get_info(sites=site)
        area = info['drain_area_va'].iloc[0] if 'drain_area_va' in info and not pd.isnull(info['drain_area_va'].iloc[0]) else ''
    except Exception:
        area = ''
    data.append({'Station': site, 'ComID': row['ComID'], 'AreaSqMi': area})

pd.DataFrame(data).to_csv(da_file, index=False)
print(f"Updated {da_file} with USGS drainage areas using dataretrieval.nwis.")
