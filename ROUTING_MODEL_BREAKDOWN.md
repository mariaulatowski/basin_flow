# Routing Model Breakdown (Current Implementation)

This document explains what the code is doing in `route_points_to_comid.py` and `brazos_streamflow_model.py`.

## 1) High-level intent

The current pipeline now supports two domain options:

1. **NHD domain (default)**: route directly on Brazos-clipped NHD flowlines (AFINCH-style domain).
2. **Transferred Brazos domain**: transfer NHD topology to Brazos geometry and route there.

With default settings, the run is now:

- `--network-domain nhd`
- `--routing-mode afinch`

Legacy transferred-Brazos workflow is still available.

Transferred-Brazos pipeline does two stages:

1. Build a Brazos-domain flowline dataset by transferring NHD-derived routing/topology properties onto Brazos flowlines.
2. Route monthly flows on that transferred Brazos-domain network using WAM/USGS point constraints.

## 2) Inputs used by the model

Primary inputs:

- Brazos flowlines:
  - `inputData/flowlines/Brazos_Flowline.shp`
- NHD source network and topology source:
  - `inputData/texas_nhdplusgrb/_extracted_gdb`
- Basin polygon for clipping:
  - `inputData/river_basin/TWDB_MRBs_2014.shp` (Brazos)
- Point-flow monthly inputs (ac-ft/month):
  - `inputData/inputs/monthly_wide_acft.csv`
  - `inputData/inputs/monthly_wide_acft_from_hecdss.csv`

## 3) Stage A: Build transferred Brazos flowline file

### A1. Build NHD network in-basin

`build_network(...)` is called on NHD source to get routed NHD reaches and topology.

### A2. Crosswalk Brazos reaches to NHD reaches

`build_brazos_to_nhd_crosswalk(...)` computes best NHD match for each Brazos COMID using:

- corridor proximity ratio,
- exact overlap ratio,
- line/centroid/endpoint distances,
- score-based ranking.

### A3. Save transferred flowlines

`save_transferred_brazos_flowlines(...)` then:

- dissolves duplicate Brazos features to one geometry per Brazos COMID,
- joins crosswalk diagnostics,
- joins selected NHD routing/property fields,
- writes:
  - `output/brazos/brazos_flowlines_with_transferred_nhd.gpkg`

This is the file used as the model domain in Stage B.

## 4) Routing method details

### 4A. NHD domain (default)

- Build routed network directly from NHD flowlines clipped to Brazos basin.
- Snap USGS/WAM monthly points to nearest NHD line geometry.
- Build monthly yield prior from specific yield (`cms/km2`) and route downstream.
- Apply AFINCH-style ratio adjustment to **upstream local yields** for constrained reaches.

AFINCH ratio adjustment now runs iteratively (`--afinch-iters`, `--afinch-damping`) so constrained reaches are pulled toward trust-weighted targets over multiple passes.

### 4B. Transferred Brazos domain

### B1. Build routing graph from transferred file

`build_network(...)` is called again, this time using:

- `flowline_source = output/brazos/brazos_flowlines_with_transferred_nhd.gpkg`

The graph uses `fromnode`/`tonode` from transferred attributes.

### B2. Parse point monthly flows

`monthly_wide_to_observations(...)` converts each monthly value from ac-ft/month to cms.

### B3. Point snapping

Points are snapped to nearest flowline geometry (not centroid) using `sjoin_nearest` in projected CRS.

### B4. Constraint conflict resolution

If multiple points snap to the same reach in the same month:

- sort by source trust first, then nearest snap distance,
- keep the highest-trust / nearest retained constraint,
- record all duplicates in `output/brazos/constraint_conflicts.csv`.

Current default trust values are:

- USGS = 1.00
- WAM = 0.75

### B5. Build prior and route

The prior is now more AFINCH-like than the earlier geometric-only prior.

For each month:

1. Compute local catchment area for each routed reach.
2. Estimate routed drainage area at constrained reaches.
3. Convert constrained flows into specific yield (`cms / km2`).
4. Build a monthly basin yield prior from:
   - date-specific constrained yields,
   - same-calendar-month climatology across years,
   - global fallback yield.
5. Multiply monthly yield by local area to produce local inflow.
6. Route that local inflow downstream with `route_monthly(local_q, downstream)`.

This produces a monthly prior that changes with observed seasonal wet/dry conditions instead of using only a fixed runoff coefficient scaled by geometry.

The code supports three routing modes:

- `afinch` (default): ratio-adjust upstream local yields from constrained reaches
- `inverse`: downstream-to-upstream allocation from constrained reaches
- `forward`: legacy forward route + downstream scaling

### B6. Constraint handling

Before solving, each retained observation gets a trust value by source.

Observed flow is then converted to an effective target at the snapped reach:

- `effective_target = prior_at_reach + trust * (observed - prior_at_reach)`

Implications:

- USGS constraints with trust `1.0` remain hard constraints.
- WAM constraints with trust below `1.0` partially pull the solution toward the WAM value instead of forcing it exactly.

Then per mode:

#### AFINCH mode (`--routing-mode afinch`, default)

1. Build monthly routed prior.
2. For each constrained reach, compute a target ratio (`target / modeled`).
3. Apply that ratio to upstream local yields (trust-weighted and damped).
4. Iterate this process multiple times.
5. Route adjusted local yields to produce final reach flows.

This follows the AFINCH concept of adjusting upstream water yields using measured-to-modeled flow ratios while conserving flow through the NHD network.

#### Inverse mode (`--routing-mode inverse`)

1. Build monthly prior routed flow.
2. Convert each retained observation into its trust-weighted effective target.
3. Start from constrained reaches (downstream constraints).
4. Walk reverse-topological order (downstream back upstream).
5. At each constrained reach, allocate required upstream flow among parent forks by prior-flow weights.
6. Convert resulting routed targets into local inflow and run one forward consistency pass.
7. Keep constrained reaches at their effective targets.

Diagnostics are recorded in `output/brazos/enforcement_diagnostics.csv`.

#### Forward mode (`--routing-mode forward`)

Legacy behavior with trust weighting:

- forward route prior,
- compute trust-weighted target,
- scale constrained reach and all downstream descendants to that target.

### B7. Write outputs

Final routed output:

- `output/brazos/modeled_monthly_comid_flows_from_points.csv`

Point snap diagnostics:

- `output/brazos/snapped_point_diagnostics.csv`

Crosswalk diagnostics:

- `output/brazos/brazos_to_nhd_crosswalk_diagnostics.csv`

## 5) Is this doing downstream-to-upstream mass balance now?

Short answer: **Yes, in inverse mode**.

When `--routing-mode inverse` is used, the solver does the downstream-to-upstream allocation workflow:

- Given a downstream constrained flow (for example 500),
- propagate needed upstream flow to parent forks,
- split among forks by prior-flow weights,
- then run a consistency forward pass.

So this now matches your requested concept much more closely than the old forward scaling method.

## 6) Remaining limitations in inverse mode

1. Fork split rule currently uses prior-flow weights; this is not yet calibrated by gage history.
2. Monthly prior is basin-wide yield driven, not yet a full regression on climate, land cover, and geology like a richer AFINCH implementation.
3. Overlapping constraints can still induce tradeoffs where a full weighted least-squares network reconciliation would be better.
4. Local inflow reconstruction uses nonnegative clipping, which is stable but simplified.

## 7) Practical implication

NHD + AFINCH mode is now the closest implementation to your requested USGS AFINCH-style approach while still using WAM points.

Key QA files:

- `output/nhd_afinch/snapped_point_diagnostics.csv`
- `output/nhd_afinch/constraint_conflicts.csv`
- `output/nhd_afinch/enforcement_diagnostics.csv`
