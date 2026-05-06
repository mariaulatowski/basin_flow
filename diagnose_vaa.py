import pandas as pd

# Sample USGS gage COMIDs
sample_comids = [128148221, 128151540, 138584078, 129104998, 97358530]

# Read VAA
print("Reading VAA...")
vaa = pd.read_csv('HSR1200/GIS/NHDFlowlineVAA.txt', sep=',', low_memory=False)
print(f"VAA shape: {vaa.shape}")
print(f"VAA columns (first 20): {vaa.columns.tolist()[:20]}")

# Check if our sample COMIDs exist in VAA
print(f"\nLooking up sample USGS gage COMIDs in VAA:")
for comid in sample_comids[:3]:
    if comid in vaa['ComID'].values:
        row = vaa[vaa['ComID'] == comid].iloc[0]
        print(f"\nCOMID {comid}:")
        print(f"  FromNode: {row.get('FromNode', 'N/A')}")
        print(f"  ToNode: {row.get('ToNode', 'N/A')}")
        print(f"  DnComID: {row.get('DnComID', 'N/A')}")
        print(f"  StreamOrde: {row.get('StreamOrde', 'N/A')}")
    else:
        print(f"\nCOMID {comid}: NOT FOUND in VAA")

# Now check for reaches flowing INTO these gages (upstream connectivity)
print("\n\n=== CHECKING UPSTREAM CONNECTIVITY ===")
for comid in sample_comids[:3]:
    if comid in vaa['ComID'].values:
        row = vaa[vaa['ComID'] == comid].iloc[0]
        from_node = row.get('FromNode')
        
        if pd.notna(from_node):
            # Find all reaches that flow TO this reach (ToNode = FromNode)
            upstream = vaa[vaa['ToNode'] == from_node]
            print(f"\nCOMID {comid} (FromNode {from_node}):")
            print(f"  Number of reaches flowing INTO it: {len(upstream)}")
            if len(upstream) > 0:
                print(f"  Upstream ComIDs: {upstream['ComID'].head(5).tolist()}")
        else:
            print(f"\nCOMID {comid}: FromNode is NaN")
