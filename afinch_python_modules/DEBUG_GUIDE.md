# AFINCH Output Debugging Guide

## Quick Start

I've created three tools to help you understand and debug the AFINCH pipeline outputs:

1. **`debug_afinch_output.py`** - Interactive debugging script with multiple inspection options
2. **`inspect_utils.py`** - Reusable Python functions for quick checks and validation
3. **`AFINCH_OUTPUT_STRUCTURE.md`** - Reference documentation for all output fields

## Understanding the Output Structure

The AFINCH pipeline returns a `ConvertedAFinchContext` object containing:

- **Configuration**: water year, month names, days per month
- **Input Data**: precipitation, temperature, station data, network structure
- **Intermediate Results**: regression outputs, estimated flows, constraints
- **Final Outputs**: accumulated flows, COMID identifiers, written file paths

### Key Arrays to Monitor

| Array | Purpose | Shape | Expected Range |
|-------|---------|-------|-----------------|
| `flow_accum` | Final accumulated streamflow | (n_reaches, 12) | 0 to 10000+ m³/s |
| `y_est_adj_inc` | Estimated surface runoff | (n_reaches, 12) | 0-500+ mm/mo |
| `q_est_adj_inc` | Estimated base flow | (n_reaches, 12) | 0-200 mm/mo |
| `con_adjust` | Constraint multipliers | (n_reaches, 12) | 0.5-1.5 (typically) |
| `p_in0` | Input precipitation | (n_gridcells, 12) | 0-80 mm/mo |
| `flow_comid` | Reach identifiers | (n_reaches,) | USGS COMID values |

## Using the Debug Script

### Interactive Mode

```bash
python debug_afinch_output.py
```

Choose from 4 options:

**Option 1: Full Context Inspection**
- Shows all non-None fields with statistics
- Displays array shapes, min/max/mean values
- Lists NaN percentages
- Useful for comprehensive understanding of output

**Option 2: Step-by-Step Inspection**
- Run pipeline up to a specific step (1-6)
- See outputs and intermediate results at each stage
- Helps identify which step causes problems
- Steps are:
  1. Setup inputs (load data)
  2. Run regression
  3. Estimate incremental flows
  4. Apply constraints
  5. Write output files
  6. Accumulate flows

**Option 3: Validation**
- Runs full pipeline
- Checks for common issues:
  - NaN values in critical arrays
  - Negative values in flow data
  - Missing required outputs
  - Constraint factors outside reasonable range
- Generates validation report

**Option 4: Quick Diagnostic**
- Fast summary of pipeline completion
- Shows key statistics for flow data
- Quick issue count

### Programmatic Usage with `inspect_utils.py`

```python
from run_converted_afinch_full_model import ConvertedAFinchPipeline
from inspect_utils import quick_check, print_stats, print_shapes
from pathlib import Path

# Run pipeline
pipeline = ConvertedAFinchPipeline(
    base_dir=Path("./input"),
    src_dir=Path("./src"),
    ths="HillCountry",
    hsr_key="Brazos",
    wy1=2010,
    ny=1,
)
ctx = pipeline.run_all()

# Quick health check
print(quick_check(ctx))

# Print all array shapes
print_shapes(ctx)

# Detailed statistics for specific array
print_stats(ctx, 'flow_accum')

# Validate consistency
validate_dimension_consistency(ctx)
```

## Common Issues and Solutions

### Issue: "NaN values in p_in0"
**Cause**: Precipitation data not loaded correctly or missing values in input files  
**Check**:
```python
print_stats(ctx, 'p_in0')
if ctx.prism:
    print(ctx.prism.prism_ths.isnull().sum())
```

### Issue: "All zeros in output arrays"
**Cause**: Input data not loaded, regression failed, or station data missing  
**Check**:
```python
print(quick_check(ctx))
# Check if stations loaded:
print(f"Stations: {len(ctx.stations) if ctx.stations else 0}")
# Check if regression completed:
print(f"Regression vars: {ctx.reg_var_name}")
```

### Issue: "Negative values in flow_accum"
**Cause**: Constraint step overly aggressively reduced flows  
**Check**:
```python
print_stats(ctx, 'con_adjust')  # Should be 0.5-1.5
print_stats(ctx, 'q_est_adj_inc')  # Before constraints
print_stats(ctx, 'q_con_adj_inc')  # After constraints
```

### Issue: "Shape mismatch errors"
**Cause**: Inconsistent preprocessing or NLCD/PRISM data mismatch  
**Check**:
```python
validate_dimension_consistency(ctx)
```

## Step-by-Step Validation Workflow

When debugging pipeline issues:

1. **Run Option 4 (Quick Diagnostic)** to get overall status
2. **If issues found, run Option 2** to narrow down which step fails
3. **Use `print_stats()` or `print_shapes()`** on arrays around the problematic step
4. **Verify input data** in the early steps
5. **Check constraint factors** if flow values seem wrong

## Output Field Reference

### After Step 1 (Setup)
```python
ctx.wy              # Water year ✓
ctx.p_in0           # Precipitation loaded ✓
ctx.stations        # Station data ✓
ctx.afstruct        # Network structure ✓
ctx.nlcd            # Land cover ✓
ctx.prism           # PRISM data ✓
```

### After Step 2 (Regression)
```python
ctx.prsm_prec_ths   # Precipitation by gridcell ✓
ctx.prsm_temp_ths   # Temperature ✓
ctx.sta_res         # Station regression results ✓
ctx.reg_var_name    # Regression variable names ✓
```

### After Step 3 (Estimation)
```python
ctx.y_est_adj_inc   # Estimated surface runoff ✓
ctx.q_est_adj_inc   # Estimated base flow ✓
```

### After Step 4 (Constraint)
```python
ctx.con_adjust      # Constraint multipliers ✓
ctx.q_con_adj_inc   # Constrained base flow ✓
ctx.y_con_adj_inc   # Constrained surface runoff ✓
```

### After Step 5 (Write)
```python
ctx.qy_path         # Output file written ✓
```

### After Step 6 (Accumulation)
```python
ctx.flow_accum      # Final accumulated flows ✓
ctx.flow_comid      # COMID identifiers ✓
ctx.flow_accum_path # Accumulation file written ✓
```

## Integration with Your Code

### In notebooks
```python
# At cell where you run pipeline:
from pathlib import Path
from run_converted_afinch_full_model import ConvertedAFinchPipeline
from inspect_utils import quick_check, print_stats

pipeline = ConvertedAFinchPipeline(...)
ctx = pipeline.run_all()

# Add this for debugging:
print(quick_check(ctx))
print_stats(ctx, 'flow_accum')
```

### In continuous testing
```python
from inspect_utils import quick_check

def test_afinch_output():
    ctx = pipeline.run_all()
    status = quick_check(ctx)
    assert "✅" in status, f"Pipeline failed: {status}"
```

## Expected Behavior

**Successful run output:**
```
✅ All checks passed!

Array Shapes:
  flow_accum:      (1234, 12)
  flow_comid:      (1234,)
  con_adjust:      (1234, 12)
  ...
```

**Problematic run output:**
```
Issues found:
  ❌ flow_accum not computed
  ⚠️  p_in0 is all zeros
  ⚠️  qy_path file not found
```

## Advanced: Comparing Before/After Constraint

To see how constraints change flow values:

```python
# After step 3 (before constraints)
pipeline.step_estimate_incremental()
q_before = ctx.q_est_adj_inc.copy()

# After step 4 (after constraints)
pipeline.step_constrain_incremental()
compare_before_after(ctx, ctx, 'q_est_adj_inc')
```

This shows percentage of values changed and magnitude of changes.

---

**Questions or issues?** Check the `AFINCH_OUTPUT_STRUCTURE.md` for detailed field reference or review the debug script output comments for guidance.
