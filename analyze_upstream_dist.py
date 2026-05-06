from pathlib import Path
import numpy as np

gached_dir = Path('HSR1200/GagedCatchments')
usgs_files = list(gached_dir.glob('*.dat'))
recent_files = [f for f in usgs_files if f.stat().st_mtime > (Path.cwd() / 'diagnose_vaa.py').stat().st_mtime][:53]

line_counts = []
for f in sorted(recent_files):
    lines = f.read_text(encoding='utf-8').strip().split('\n')
    # Subtract 1 for header
    upstream_count = len(lines) - 1
    line_counts.append(upstream_count)
    if len(line_counts) <= 5 or len(line_counts) > len(recent_files) - 5:
        print(f"{f.name}: {upstream_count} upstream reaches")

print(f"\nDistribution of {len(line_counts)} USGS gages:")
print(f"  Min: {min(line_counts)}")
print(f"  25th %ile: {int(np.quantile(line_counts, 0.25))}")
print(f"  Median: {int(np.quantile(line_counts, 0.5))}")
print(f"  75th %ile: {int(np.quantile(line_counts, 0.75))}")
print(f"  90th %ile: {int(np.quantile(line_counts, 0.9))}")
print(f"  Max: {max(line_counts)}")
print(f"  Mean: {np.mean(line_counts):.0f}")
print(f"\nTotal unique reaches in USGS gages: {sum(line_counts):,}")
print(f"Total reaches in network: 294,243")
