## Debug Tools Created ✓

I've created a comprehensive debugging toolkit for your AFINCH pipeline. Here's what you now have:

### 📋 Files Created

1. **`debug_afinch_output.py`**
   - Interactive debugging script with 4 menu options
   - Run immediately: `python debug_afinch_output.py`
   - Options: Full inspection, step-by-step, validation, quick diagnostic
   - Color-coded terminal output for easy reading

2. **`inspect_utils.py`**
   - Lightweight reusable functions for code integration
   - Functions: `quick_check()`, `print_stats()`, `print_shapes()`, `validate_dimension_consistency()`, `compare_before_after()`
   - Import into your notebooks and scripts as needed
   - Example: `from inspect_utils import quick_check`

3. **`AFINCH_OUTPUT_STRUCTURE.md`**
   - Complete reference documenting all output fields
   - Expected value ranges and units
   - Common issues and their solutions
   - Pipeline execution flow diagram
   - Quick reference table

4. **`DEBUG_GUIDE.md`**
   - Practical guide for using the debug tools
   - Step-by-step troubleshooting workflow
   - Code examples (interactive and programmatic)
   - Integration with notebooks
   - Expected behavior patterns

### 🎯 Quick Start

**For immediate debugging:**
```bash
cd afinch_python_modules
python debug_afinch_output.py
# Then select option 4 (Quick Diagnostic) or 3 (Validation)
```

**For code integration:**
```python
from inspect_utils import quick_check, print_stats

ctx = pipeline.run_all()
print(quick_check(ctx))        # ✅ or ❌ status
print_stats(ctx, 'flow_accum') # Detailed statistics
```

**For step-by-step debugging:**
```bash
python debug_afinch_output.py
# Select option 2, then choose step (1-6)
```

### 🔍 What Each Tool Does

| Tool | Purpose | When to Use |
|------|---------|------------|
| `debug_afinch_output.py` | Interactive inspection | First investigation, unfamiliar with data |
| `inspect_utils.py` | Programmatic checks | Automated testing, notebook integration |
| `AFINCH_OUTPUT_STRUCTURE.md` | Reference documentation | Understanding field definitions and ranges |
| `DEBUG_GUIDE.md` | How-to guide | Workflow and troubleshooting advice |

### 🛠️ Key Features

- **Dimension checking**: Verify arrays have consistent shapes (n_reaches × 12)
- **NaN detection**: Identify missing values and percentage of NaN
- **Range validation**: Check if values fall within expected ranges
- **Issue detection**: Automatic identification of common problems
- **Before/after comparison**: See how constraints change flow values
- **Step-by-step execution**: Run pipeline incrementally to pinpoint issues
- **Statistics summary**: Min, max, mean, median, std for any array

### 🔴 Common Issues Detected

✓ Output files not written  
✓ Negative flow values  
✓ NaN values exceeding threshold  
✓ Constraint factors out of range  
✓ Missing required computations  
✓ Dimension mismatches  
✓ Zero-value data  

### 📊 Expected Output Structure

The pipeline returns a `ConvertedAFinchContext` with:

```
Configuration: wy, days_in_mo, month_name, mo_name
Inputs:       nlcd, prism, p_in0, stations, inflow, afstruct
Regression:   prsm_prec_ths, prsm_temp_ths, temp_res, sta_res
Estimation:   y_est_adj_inc, q_est_adj_inc
Constraint:   con_adjust, q_con_adj_inc, y_con_adj_inc
Outputs:      flow_accum, flow_comid, qy_path, flow_accum_path
```

### 📖 Documentation Files

- **AFINCH_OUTPUT_STRUCTURE.md** (3 min read) - Field reference with ranges
- **DEBUG_GUIDE.md** (5 min read) - Workflow and examples  
- **This file** - Quick reference of what was created

### ✨ Next Steps

1. **First run**: Execute `python debug_afinch_output.py` → select option 4 (Quick Diagnostic)
2. **If issues found**: Go back and select option 3 (Validation) for detailed report
3. **For investigation**: Use option 2 (Step-by-step) to narrow down failure point
4. **For integration**: Import functions from `inspect_utils.py` into your code

### 💡 Pro Tips

- Use `inspect_utils.py` for CI/CD pipelines and automated tests
- Save debug script output to file: `python debug_afinch_output.py > debug_output.txt`
- Check `AFINCH_OUTPUT_STRUCTURE.md` for expected value ranges
- Use `print_stats()` to quickly spot data quality issues
- Constraint factors (`con_adjust`) typically range 0.5-1.5; outside means potential issues

---

**All files are in**: `afinch_python_modules/`  
**Ready to use**, no additional setup needed!
