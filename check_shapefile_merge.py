import geopandas as gpd
import pandas as pd

shp = gpd.read_file(r'c:\Users\mu3575\Documents\WAM\output\arcgis_exports\ComIDQ12WY2018_flowlines.shp')
flow = pd.read_csv(r'c:\Users\mu3575\Documents\WAM\HSR1206\Output\FlowAccum\ComIDQ12WY2018.csv')

print('Shapefile:')
print(f'  COMID column: {shp["COMID"].dtype}, sample: {shp["COMID"].head().tolist()}')
print(f'  Total rows: {len(shp)}')

print(f'\nFlow file:')
print(f'  ComIDVAA column: {flow["ComIDVAA"].dtype}, sample: {flow["ComIDVAA"].head().tolist()}')
print(f'  Total rows: {len(flow)}')

# Check for matches
shp_comids = set(shp['COMID'].dropna().unique())
flow_comids = set(flow['ComIDVAA'].unique())
matches = shp_comids & flow_comids
print(f'\nMatching ComIDs: {len(matches)} out of {len(shp_comids)} shapefile + {len(flow_comids)} flow')
if len(matches) == 0:
    print(f'  Shapefile ComID range: {shp["COMID"].min()} to {shp["COMID"].max()}')
    print(f'  Flow ComID range: {flow["ComIDVAA"].min()} to {flow["ComIDVAA"].max()}')
else:
    print(f'  Sample matching: {list(matches)[:3]}')
