# Streamflow Model Documentation

This document explains how streamflow is calculated in the current model implemented in:

- `brazos_streamflow_model.py` (core engine)
- `wam_gui.py` (user interface that builds config and runs the engine)

## 1. What the model does

The model simulates **monthly flow for every routed flowline COMID** in a selected basin (for example, Brazos), over a user-selected date range.

At a high level, each month it does:

1. Build local runoff on each flowline catchment proxy area.
2. Route upstream contributions downstream through the network.
3. Apply channel-loss scaling along control-point paths.
4. Optionally enforce observed gage flows.
5. Save results to CSV (and optionally shapefile).

## 2. Main input files

- **Flowline source**: NHDPlusHR flowlines (`NHDFlowline` / `NetworkNHDFlowline`) from GDBs or shapefiles.
- **Routing attributes**:
  - from flowline table directly (`fromnode`, `tonode`, `hydroseq`), or
  - from VAA table (`NHDPlusFlowlineVAA` or external VAA DBF).
- **Basin shapefile**: used to clip network to one basin name.
- **Control-point metadata CSV**: `UP_CP`, `Next_DS_CP`, `ChLosFac`, `LAT`, `LONG`.
- **FLO file**: monthly gage flows in acre-feet/month.
- **Optional gage crosswalk CSV**: maps `gage_id` -> `COMID`.

## 3. Network construction

The engine:

1. Loads all flowlines from chosen source.
2. Clips flowlines to selected basin polygon.
3. Ensures each routed segment has routing node fields.
4. Builds directed edges by node matching:
   - upstream segment `tonode` equals downstream segment `fromnode`.
5. Removes invalid edge directions using hydro sequence checks.
6. Validates directed acyclic graph (DAG) and creates topological order.

Topological order is critical because it guarantees upstream flow is calculated before downstream flow.

## 4. Area used for runoff generation

Local area per flowline is chosen in this order:

1. `areasqkm` (preferred, local catchment area)
2. `totdasqkm` (fallback)
3. Length-based proxy (`lengthkm * 2.0`) if areas are unavailable

## 5. Monthly local runoff calculation

For each month and each flowline:

- `runoff_coeff_m_per_month` is treated as runoff depth (m/month)
- Local volume:
  - `V = coeff * area_km2 * 1e6`  (m3/month)
- Local flow (cms):
  - `Q_local = V / seconds_in_month`

This yields a local inflow term for each COMID.

## 6. Routing calculation

Routing uses adjacency lists over topological order:

- Initialize `Q = Q_local`.
- For each upstream node `i` in topo order:
  - Add `Q[i]` into each downstream node `j`.

This accumulates all upstream contributions progressively downstream.

## 7. WRAP channel-loss handling

Control points are snapped to nearest routed COMID using CP lat/lon.

For each CP pair path:

1. Find path from `UP_COMID` to `DS_COMID`.
2. Compute expected downstream target:
   - `Q_target_ds = Q_up * (1 - ChLosFac)`
3. Compare to current downstream modeled flow and compute scale factor.
4. Scale the path nodes downstream of upstream CP.

This enforces WRAP-style attenuation between CPs.

## 8. Gage enforcement (optional)

If gage crosswalk is provided:

1. FLO monthly acre-ft values are converted to cms.
2. Gage IDs are mapped to COMIDs.
3. For each gage COMID in a month:
   - `scale = observed / modeled_at_gage`
   - Apply scale to that COMID and all descendants.

This forces modeled flow to match observed gage flow at enforced locations.

## 9. Calibration (optional)

If calibration is enabled:

1. Try a grid of runoff coefficients.
2. Route flows and compare modeled vs observed gage flows.
3. Compute RMSE.
4. Select coefficient with minimum RMSE.

## 10. Outputs

### CSV output

One row per `(date, COMID)` including:

- `date`
- `COMID`
- `flow_cms`
- `lat`
- `lon`

Lat/lon are centroid-based coordinates for GIS referencing.

### Optional shapefile output

Exports one selected month as point features with:

- `DATE`
- `COMID`
- `FLOW_CMS`
- `FLOW_TXT` (scientific-notation text backup)
- `LAT`
- `LON`
- geometry (point)

## 11. Important assumptions and limitations

1. This is a **monthly lumped runoff + routing model**, not a full hydrodynamic model.
2. Local runoff coefficient is simplified unless calibrated.
3. CP snapping is nearest-flowline based (spatial approximation).
4. Gage enforcement rescales downstream flows; multiple gages can interact.
5. Results depend heavily on network correctness and basin clipping inputs.

## 12. File-level responsibility

- `wam_gui.py`
  - collects user inputs
  - prepares extracted GDB folder from HU4 zip files
  - calls model with progress callback
- `brazos_streamflow_model.py`
  - all core calculations and I/O export logic

---
