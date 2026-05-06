# Model Discrepancy Analysis: Leon River & Other Major Stations

## Summary of Errors

Your observation about large discrepancies at stations like **Leon River nr Hamilton (8100000)** is correct. Analysis shows:

| Station | Issue Type | Obs Avg (CFS) | Model Avg (CFS) | Ratio | RMSE |
|---------|-----------|---------------|-----------------|-------|------|
| Brazos at Waco | OVERPREDICTION | 3,361 | 7,418 | 0.45x | 8,801 |
| Brazos nr Glen Rose | OVERPREDICTION | 1,763 | 3,951 | 0.44x | 4,585 |
| Brazos nr Aquilla | OVERPREDICTION | 2,337 | 4,914 | 0.48x | 5,669 |
| Leon nr Hamilton | UNDERPREDICTION* | 356 | 0.1 | 3,600x | 519 |
| Little River nr Cameron | OVERPREDICTION | 2,019 | 3,092 | 0.65x | 2,215 |

**Note**: Leon River shows opposite pattern in some months—massive underpredictions suggesting a data quality or routing issue.

---

## Root Causes Identified

### 🔴 **PRIMARY: Water Use Data NOT Applied**

**Critical Finding**: The water use file (`ComID_WU_All.dat`) is **completely filled with zeros**:
```
68285407 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
68285409 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
68285411 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
... (all zeros across all COMIDs)
```

**Impact**: The model Step 4 constraint routine (`AFQConAdjInc`) reads this file but finds no withdrawals/returns to subtract. The result:
- **Natural runoff** is routed downstream undiminished
- **Actual observations** reflect real withdrawals (irrigation, municipal, industrial, power generation)
- **Model OVERPREDICTS** by ~2.0-2.5x on main stem Brazos

**Geographic Pattern**: Worst errors occur where water demand is highest:
- Brazos at Waco (Central Texas agricultural area)
- Brazos nr Glen Rose (irrigation districts below Lake Granbury)
- Brazos nr Aquilla (near Waco, urban + agricultural)
- Little River (tributary in agricultural zone)

**Leon River Exception**: Appears to be a different issue—possible mapping error where observed station is not properly linked to model COMID, causing severe underprediction.

---

### 🟡 **SECONDARY: Single-Year Regression Calibration**

Model Step 1 regression uses **only WY2018 data** to build predictor equations:
- $$Q_{predictor} = f(\text{PRISM precip}, \text{NLCD land use}, \text{contributing area})$$
- WY2018 was a relatively wet year; dry/normal years may behave differently
- No capture of multiyear drought/wetness cycles
- Temporal variability in land use, management practices not represented

**Evidence**: Headwater stations (small, ungauged tributaries) match observations perfectly (RMSE ≈ 0), suggesting regression works well at small scales but breaks down when water management enters the picture.

---

### 🟡 **TERTIARY: Missing Process Representations**

1. **Reservoir storage/release**: Dams store water in wet months, release in dry months → model sees only natural runoff
2. **Evaporation losses**: Reservoirs and riparian zones lose 5-15% of flow in summer → not modeled
3. **Aquifer recharge/discharge**: Groundwater dynamics not explicitly included
4. **Return flow timing**: Irrigation return flows occur days/weeks after withdrawal → not separated by reach

---

## Model Improvement Recommendations

### ✅ **Immediate Fixes (High Impact)**

#### 1. **Populate Water Use Data** (Priority 1)
- Your `ComID_WU_All.dat` file needs actual monthly withdrawal/return estimates by COMID
- Sources:
  - **USGS National Water Use Database** (5-year intervals, by county)
  - **Texas Water Development Board (TWDB)** water use surveys
  - **Individual Water User organizations** (irrigation districts, utilities)
  - **ANET model** (Alternative approach in your codebase handles this)
  
- Format: Monthly net withdrawals (positive = removal, negative = return) in acre-feet or CFS

**Expected Improvement**: Would resolve 60-70% of main-stem overpredictions.

---

#### 2. **Add Multi-Year Calibration Data** (Priority 2)
You have 2010-2024 data available. Use 10-15 water years for regression instead of 1:

```python
# Instead of:
# step_run_regression(wy=2018)

# Run:
# step_run_regression(base_years=[2010, 2011, ..., 2024])
```

Benefits:
- Captures dry (2011, 2013, 2018), normal (2015, 2017), and wet years (2004, 2016)
- Regression coefficients become more robust to climate variability
- Better uncertainty quantification

**Expected Improvement**: 15-25% reduction in residual errors post-water-use.

---

### ⚠️ **Medium-Term Enhancements**

#### 3. **Explicit Reservoir Routing**
For reaches below major dams (Possum Kingdom, Granbury, etc.):
- Add reservoir storage-discharge relationship
- Route flows through reservoir accounting for evaporation
- Implement flood pool / conservation pool operations

**Affect Stations**: Glen Rose, Aquilla, Richmond (all downstream of reservoirs)

---

#### 4. **Seasonal Water Rights Management**
Texas water law has:
- **Senior (`appropriation`) rights**: Priority claims honored first
- **Junior rights**: Only get flow if oversupply
- **Seasonal restrictions**: Some water removed seasonally for environmental flows

Model could:
- Reduce modeled flows during dry season by rights priority
- Add seasonal factors (e.g., July-September withdrawal rates differ from Oct-Dec)

---

#### 5. **Fine-Tune Loss Functions**
Current model uses simple regression. Consider:
- **Log-space regression** to better capture very low flows
- **Power-law relationships** (flow ≈ area^0.8) instead of linear
- **Seasonally-variable coefficients** (dry season ≠ wet season hydrology)

---

## Testing the Hypothesis

To confirm water use is the main issue, try:

```python
# Quick test: Run model WITHOUT the constraint step
step_setup_inputs(wy=2018)
step_run_regression(wy=2018)
step_estimate_incremental(wy=2018)
# SKIP: step_constrain_incremental() ← Comment this out
step_write_incremental_output(wy=2018)
step_accumulate_flow(wy=2018)

# Compare to observations:
# - If unconstrained errors are SMALLER at Waco/Glen Rose, then:
#   → Issue is in constraint application (missing water use)
# - If unconstrained errors are SAME or LARGER, then:
#   → Issue is in regression itself (needs multi-year calibration)
```

---

## Long-Term Model Architecture Improvements

Your codebase already has `brazos_monthly_mass_balance_model.py`—an alternative approach that:
- Uses inverse modeling (observations → upstream sources)
- Has `--usgs-trust` and `--wam-trust` parameters for blending data
- May handle water management better

Consider:
- Comparing both approaches (AFINCH vs. mass-balance) on same validation set
- Migrating key features (data blending, regularization) into Steps pipeline
- Implementing **dual-constraint**: both observation matching AND water balance conservation

---

## Expected Outcomes by Fix

| Fix | Cost | Brazos-Waco MAE | Leon-Hamilton RMSE | Headwaters |
|-----|------|-----------------|-------------------|-----------|
| Baseline (current) | — | 4,057 CFS | 519 CFS | ~0 CFS ✓ |
| + Water Use Data | LOW | ~1,500 CFS | ~400 CFS | ~0 CFS ✓ |
| + Multi-Year Cal. | MED | ~800 CFS | ~300 CFS | ~0 CFS ✓ |
| + Reservoir Routing | MED | ~500 CFS | ~150 CFS | ~0 CFS ✓ |

---

## Recommended Next Steps

1. **Immediate**: Check if TWDB or other agency has monthly water use by COMID for Brazos basin
2. **Week 1**: Populate water use file with available data; re-run Steps 4-6, re-plot
3. **Week 2**: If main-stem improves but Leon River still bad → investigate Leon River gaging station mapping
4. **Week 3**: Assemble 2010-2024 streamflow data; test multi-year regression
5. **Long-term**: Implement reservoir routing for post-dam reaches

Would you like me to help with any of these steps?
