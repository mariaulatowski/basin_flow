import pandas as pd
from collections import defaultdict, deque

# Read VAA
vaa = pd.read_csv('HSR1200/GIS/NHDFlowlineVAA.txt', sep=',', low_memory=False)

# Build downstream index (FromNode -> reaches that START at this node)
downstream_map = defaultdict(list)
for _, row in vaa.iterrows():
    start_node = int(row['FromNode'])
    downstream_map[start_node].append(int(row['ComID']))

# Sample a few "upstream" reaches from gage 8111500's .dat file
gage_file = pd.read_csv('HSR1200/GagedCatchments/8111500.dat')
gage_comid = 95924547

# Pick 5 random upstream reaches (not the gage itself)
upstream_reaches = gage_file[gage_file['ComID'] != gage_comid]['ComID'].sample(min(5, len(gage_file)-1)).tolist()

print(f"Testing if upstream reaches actually flow to gage {gage_comid}...\n")

for upstream_comid in upstream_reaches[:3]:
    print(f"Reach {upstream_comid}:")
    
    # Try to trace from this reach downstream toward the gage
    current_comid = upstream_comid
    path = [current_comid]
    found_gage = False
    steps = 0
    
    while current_comid != gage_comid and steps < 50:
        current_row = vaa[vaa['ComID'] == current_comid]
        if len(current_row) == 0:
            print(f"  ERROR: ComID {current_comid} not found in VAA")
            break
        
        to_node = int(current_row.iloc[0]['ToNode'])
        # Find what reaches START at this ToNode (i.e., what this reaches flows into)
        next_reaches = vaa[vaa['FromNode'] == to_node]['ComID'].tolist()
        
        if len(next_reaches) == 0:
            print(f"  → Trace stopped: ToNode {to_node} has no downstream reaches")
            break
        
        # In a tree, should be just 1 downstream, but could be multiple if data is odd
        current_comid = next_reaches[0]
        path.append(current_comid)
        steps += 1
        
        if current_comid == gage_comid:
            found_gage = True
            break
    
    if found_gage:
        print(f"  ✓ Path found in {steps+1} steps: {' → '.join(str(p) for p in path[:5])} ... → {gage_comid}")
    else:
        print(f"  ✗ Did NOT reach gage after {steps} steps. Stopped at {current_comid}")
        print(f"    Path: {' → '.join(str(p) for p in path[:5])} ... {current_comid}")
