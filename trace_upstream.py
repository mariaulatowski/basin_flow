import pandas as pd

# Sample USGS gage COMIDs
sample_comids = [128148221, 128151540, 138584078]

# Read VAA
print("Reading VAA...")
vaa = pd.read_csv('HSR1200/GIS/NHDFlowlineVAA.txt', sep=',', low_memory=False)
print(f"VAA shape: {vaa.shape}\n")

# Build index of ToNode -> ComID for faster lookup
to_node_idx = {}
for idx, row in vaa.iterrows():
    to_node = row['ToNode']
    comid = row['ComID']
    if to_node not in to_node_idx:
        to_node_idx[to_node] = []
    to_node_idx[to_node].append(comid)

def trace_upstream(comid, max_levels=10):
    """Recursively trace all upstream reaches"""
    visited = set()
    stack = [comid]
    all_upstream = []
    level = 0
    
    while stack and level < max_levels:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        
        # Find FromNode for this ComID
        row_list = vaa[vaa['ComID'] == current]
        if len(row_list) == 0:
            continue
            
        from_node = row_list.iloc[0]['FromNode']
        
        # Find all reaches that flow into this FromNode (i.e., ToNode == FromNode)
        upstream_for_this = to_node_idx.get(from_node, [])
        all_upstream.extend(upstream_for_this)
        stack.extend(upstream_for_this)
        level += 1
    
    return visited, len(all_upstream)

# Trace upstream for sample gages
print("=== FULL UPSTREAM BASIN TRACING ===\n")
for comid in sample_comids:
    visited, total_upstream = trace_upstream(comid, max_levels=50)
    print(f"COMID {comid}:")
    print(f"  Total unique visited reaches (including self): {len(visited)}")
    print(f"  Estimated upstream members: {len(visited) - 1}")  # Subtract self
