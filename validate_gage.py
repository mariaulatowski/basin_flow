from pathlib import Path
import pandas as pd

# Read a specific USGS gage .dat file that was just created
gaged_file = Path('HSR1200/GagedCatchments/8111500.dat')
data = pd.read_csv(gaged_file)
print(f"Gage 8111500.dat: {len(data)} upstream reaches\n")
print("First 10 rows:")
print(data.head(10))

# Check what COMID this gage actually maps to
station_map = pd.read_csv('HSR1200/Flowlines/StationComID.csv')
gage_info = station_map[station_map['Station'] == '8111500']
print(f"\nGage 8111500 station info:")
print(gage_info[['Station', 'ComID', 'Source', 'snap_dist_m']])

# Now let's manually verify a few of those upstream reaches exist in VAA
vaa = pd.read_csv('HSR1200/GIS/NHDFlowlineVAA.txt', sep=',', low_memory=False)
sample_comids = data['ComID'].head(5).tolist()
print(f"\nVerifying first 5 upstream ComIDs exist in VAA:")
for comid in sample_comids:
    exists = comid in vaa['ComID'].values
    print(f"  ComID {comid}: {'✓' if exists else '✗ NOT FOUND'}")
