# AFINCH Pipeline Output Structure Reference

This document maps the expected output from the AFINCH pipeline and helps identify where data issues occur.

## Pipeline Execution Flow

The `ConvertedAFinchPipeline.run_all()` method calls six sequential steps:

```
Step 1: setup_inputs()
    ↓
Step 2: run_regression()
    ↓
Step 3: estimate_incremental()
    ↓
Step 4: constrain_incremental()
    ↓
Step 5: write_incremental_output()
    ↓
Step 6: accumulate_flow()
```

## Output Context Fields

The `ConvertedAFinchContext` object contains all intermediate and final outputs:

### Configuration Fields
- `wy` (int): Water year
- `days_in_mo` (np.ndarray): Days in each month [12,]
- `month_name` (list): Full month names
- `mo_name` (list): 3-letter month abbreviations

### Input Data (Step 1)
- `nlcd`: NLCD land cover data with `comid_ths` list
- `prism`: PRISM precipitation data with `prism_ths` DataFrame
- `p_in0` (ndarray): Precipitation input [n_gridcells, 12]
- `stations`: Station objects and metadata
- `inflow`: Inflow data with `station_flow_df` and `comid_wu_df`
- `afstruct`: Network structure (connectivity matrix)
- `poa`: Points of analysis

### Regression Outputs (Step 2)
- `sta_res`: Station results from regression
- `temp_res`: Temperature regression results
- `sta_hist_list` (list): Historical station data per month
- `reg_var_name` (list): Regression variable names
- `reg_poa`: Regression POA results
- `reg_hist`: Regression history

### Precipitation/Temperature Arrays (Step 2)
- `prsm_prec_ths` (ndarray): Precipitation for threshold gridcells [n_thresh, 12]
- `prsm_temp_ths` (ndarray): Temperature for threshold gridcells [n_thresh, 12]
- `prsm_prem_ths` (ndarray): Precipitation multiplier [n_thresh,]
- `gc_area_sq_mi` (ndarray): Gridcell areas [n_gridcells,]

### Estimation Outputs (Step 3)
- `y_est_adj_inc` (ndarray): Estimated surface runoff (incremental) [n_reaches, 12]
- `q_est_adj_inc` (ndarray): Estimated base flow (incremental) [n_reaches, 12]

**Shape:** (n_reaches × 12) - one value per reach per month

### Constraint Outputs (Step 4)
- `con_adjust` (ndarray): Constraint adjustment factors [n_reaches, 12]
- `q_con_adj_inc` (ndarray): Constrained base flow [n_reaches, 12]
- `y_con_adj_inc` (ndarray): Constrained surface runoff [n_reaches, 12]
- `afstruct_con`: Network structure with constraints

**Shape:** Same as estimation outputs (n_reaches × 12)

### File Outputs (Step 5)
- `qy_path` (Path): Path to written QY output file

### Flow Accumulation Outputs (Step 6)
- `flow_accum` (ndarray): Accumulated flow along network [n_reaches, 12]
- `flow_comid` (ndarray): COMID identifiers for reaches [n_reaches,]
- `flow_accum_path` (Path): Path to written accumulation output

## Expected Value Ranges

| Field | Type | Expected Range | Units | Notes |
|-------|------|-----------------|-------|-------|
| `wy` | int | 2000-2020 | year | Water year |
| `p_in0` | array | 0.1-80 | mm/mo | Extremely high variation by region |
| `prsm_temp_ths` | array | -10 to 40 | °C | Regional/seasonal variation |
| `y_est_adj_inc` | array | 0-500+ | mm/mo | Can exceed precipitation in humid areas |
| `q_est_adj_inc` | array | 0-200 | mm/mo | Base flow typically lower |
| `con_adjust` | array | 0.5-1.5 | multiplier | Adjusts flow for constraints |
| `flow_accum` | array | 0-10000+ | m³/s | Increases downstream |
| `gc_area_sq_mi` | array | 0.1-10 | mi² | Gridcell area in square miles |

## Common Issues and Debugging

### Issue: NaN values in arrays
**Check:**
- Input precipitation data loaded correctly?
- Temperature data contains valid values?
- Check `prsm_prec_ths` for missing values
- Verify PRISM file paths in configuration

### Issue: Negative flow values
**Check:**
- `con_adjust` factors reasonable (0.5-1.5)?
- `q_est_adj_inc` should be non-negative before constraints
- `flow_accum` should always be non-negative

### Issue: Outputs all zeros
**Check:**
- Did regression step complete? (`reg_var_name` populated?)
- Input data loaded? (`p_in0` non-zero?)
- Station data available? (`stations` list populated?)

### Issue: Mismatched dimensions
**Common causes:**
- Number of gridcells changes between steps
- Number of reaches or months incorrect
- Check shape consistency: should always be `(n_reaches, 12)`

## Using the Debug Script

```bash
python debug_afinch_output.py
```

Options:
1. **Full Context Inspection**: Shows all non-None fields and statistics
2. **Step-by-Step**: Run and inspect each step individually
3. **Validation**: Runs full pipeline and checks for common issues
4. **Quick Diagnostic**: Fast summary of pipeline completion and main outputs

## Example: Accessing Outputs in Code

```python
from run_converted_afinch_full_model import ConvertedAFinchPipeline
from pathlib import Path

pipeline = ConvertedAFinchPipeline(
    base_dir=Path("."),
    src_dir=Path("./src"),
    ths="HillCountry",
    hsr_key="Brazos",
    wy1=2010,
    ny=1,
)

ctx = pipeline.run_all()

# Access outputs
print(f"Flow accumulation shape: {ctx.flow_accum.shape}")
print(f"Output written to: {ctx.qy_path}")
print(f"Number of COMIDs: {len(ctx.flow_comid)}")
```
