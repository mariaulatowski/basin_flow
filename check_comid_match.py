import pandas as pd
import geopandas as gpd

# Load flow data
flow_df = pd.read_csv(r'c:\Users\mu3575\Documents\WAM\HSR1206\Output\FlowAccum\ComIDQ12WY2018.csv')
print('Flow file:')
print(f'  Columns: {list(flow_df.columns)}')
print(f'  Shape: {flow_df.shape}')
print(f'  Sample ComIDs: {flow_df.iloc[:, 0].head().tolist()}')
print(f'  Data type: {flow_df.iloc[:, 0].dtype}')

# Load geometry
gdf = gpd.read_file(r'c:\Users\mu3575\Documents\WAM\inputData\NHDPlusCatchment_1206.gpkg')
print(f'\nGeometry file:')
print(f'  Columns: {list(gdf.columns)}')
print(f'  Shape: {gdf.shape}')
print(f'  Sample NHDPlusID: {gdf["NHDPlusID"].head().tolist()}')
print(f'  Data type: {gdf["NHDPlusID"].dtype}')

# Check for matches
flow_comids = set(flow_df.iloc[:, 0].unique())
geom_comids = set(gdf['NHDPlusID'].unique())
matches = flow_comids & geom_comids
print(f'\nMatching ComIDs: {len(matches)} out of {len(flow_comids)} flow + {len(geom_comids)} geometry')
if len(matches) > 0:
    print(f'  Sample matching: {list(matches)[:3]}')
else:
    print('  NO MATCHES!')
    print(f'  Flow range: {min(flow_comids)} to {max(flow_comids)}')
    print(f'  Geom range: {min(geom_comids)} to {max(geom_comids)}')
