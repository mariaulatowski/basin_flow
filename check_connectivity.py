import pandas as pd

df = pd.read_csv('HSR1200/GagedCatchments/8111500.dat')
gage_comid = 95924547
print(f'Gage ComID {gage_comid} in upstream list: {gage_comid in df["ComID"].values}')
print(f'Total rows: {len(df)}')
print(f'First 5 ComIDs: {df["ComID"].head().tolist()}')
print(f'Last 5 ComIDs: {df["ComID"].tail().tolist()}')

# Check if gage_comid flows into the first ComID
vaa = pd.read_csv('HSR1200/GIS/NHDFlowlineVAA.txt', sep=',')
gage_row = vaa[vaa['ComID'] == gage_comid]
if len(gage_row) > 0:
    print(f"\nGage ComID {gage_comid} info:")
    print(f"  FromNode: {gage_row.iloc[0]['FromNode']}")
    print(f"  ToNode: {gage_row.iloc[0]['ToNode']}")
    
first_upstream_comid = df['ComID'].iloc[0]
first_row = vaa[vaa['ComID'] == first_upstream_comid]
if len(first_row) > 0:
    print(f"\nFirst upstream ComID {first_upstream_comid} info:")
    print(f"  FromNode: {first_row.iloc[0]['FromNode']}")
    print(f"  ToNode: {first_row.iloc[0]['ToNode']}")
