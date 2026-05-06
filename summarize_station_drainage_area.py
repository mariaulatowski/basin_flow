import pandas as pd

# Load updated drainage area file
da_file = r"HSR1200/Streamflow/StationDASqMi.csv"
df = pd.read_csv(da_file)

missing = df[df['AreaSqMi'].isnull() | (df['AreaSqMi'] == '')]
filled = df[~(df['AreaSqMi'].isnull() | (df['AreaSqMi'] == ''))]

print(f"Total stations: {len(df)}")
print(f"Stations with USGS drainage area: {len(filled)}")
print(f"Stations missing drainage area: {len(missing)}")

if not missing.empty:
    print("\nStations missing drainage area:")
    print(missing[['Station','ComID']].to_string(index=False))
else:
    print("All stations have USGS drainage area values.")
