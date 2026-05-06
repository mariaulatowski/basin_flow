"""
Quick inspection utilities for AFINCH pipeline outputs.
Use these functions to quickly validate or inspect context state.
"""

import numpy as np
from pathlib import Path
from run_converted_afinch_full_model import ConvertedAFinchContext

def quick_check(ctx: ConvertedAFinchContext) -> str:
    """
    Quick health check of context.
    Returns: status string with issues found (if any)
    
    Example:
        ctx = pipeline.run_all()
        print(quick_check(ctx))
    """
    issues = []
    
    # Critical checks
    if ctx.flow_accum is None:
        issues.append("❌ flow_accum not computed")
    elif np.any(ctx.flow_accum < 0):
        neg_count = np.sum(ctx.flow_accum < 0)
        issues.append(f"⚠️  flow_accum has {neg_count} negative values")
    
    if ctx.qy_path is None:
        issues.append("❌ qy_path not set")
    elif not Path(ctx.qy_path).exists():
        issues.append(f"⚠️  qy_path file not found: {ctx.qy_path}")
    
    if ctx.con_adjust is None:
        issues.append("❌ con_adjust not computed")
    elif np.any(ctx.con_adjust < 0.1) or np.any(ctx.con_adjust > 10):
        issues.append("⚠️  con_adjust outside expected range (0.5-1.5)")
    
    # Data checks
    if ctx.p_in0 is None:
        issues.append("❌ p_in0 (precipitation) not loaded")
    elif np.all(ctx.p_in0 == 0):
        issues.append("⚠️  p_in0 is all zeros")
    
    if ctx.flow_comid is None:
        issues.append("❌ flow_comid not computed")
    
    # Success case
    if not issues:
        return "✅ All checks passed!"
    
    return "Issues found:\n  " + "\n  ".join(issues)

def print_shapes(ctx: ConvertedAFinchContext):
    """Print shapes of all major arrays in context."""
    print("\nArray Shapes:")
    print(f"  p_in0:           {ctx.p_in0.shape if ctx.p_in0 else 'None'}")
    print(f"  flow_accum:      {ctx.flow_accum.shape if ctx.flow_accum else 'None'}")
    print(f"  flow_comid:      {ctx.flow_comid.shape if ctx.flow_comid else 'None'}")
    print(f"  con_adjust:      {ctx.con_adjust.shape if ctx.con_adjust else 'None'}")
    print(f"  y_est_adj_inc:   {ctx.y_est_adj_inc.shape if ctx.y_est_adj_inc else 'None'}")
    print(f"  q_est_adj_inc:   {ctx.q_est_adj_inc.shape if ctx.q_est_adj_inc else 'None'}")
    print(f"  prsm_prec_ths:   {ctx.prsm_prec_ths.shape if ctx.prsm_prec_ths else 'None'}")

def print_stats(ctx: ConvertedAFinchContext, array_name: str):
    """
    Print statistics for a single array.
    
    Example:
        print_stats(ctx, 'flow_accum')
        print_stats(ctx, 'p_in0')
    """
    arr = getattr(ctx, array_name, None)
    if arr is None:
        print(f"{array_name}: None (not computed)")
        return
    
    print(f"\n{array_name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    
    # Filter out NaNs for stats
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print("  ⚠️  All values are NaN!")
        return
    
    print(f"  Min:    {np.min(valid):12.4f}")
    print(f"  Max:    {np.max(valid):12.4f}")
    print(f"  Mean:   {np.mean(valid):12.4f}")
    print(f"  Median: {np.median(valid):12.4f}")
    print(f"  Std:    {np.std(valid):12.4f}")
    
    nan_count = np.sum(np.isnan(arr))
    if nan_count > 0:
        print(f"  NaN:    {nan_count} ({100*nan_count/arr.size:.1f}%)")
    
    neg_count = np.sum(arr < 0)
    if neg_count > 0:
        print(f"  Negative: {neg_count} ({100*neg_count/arr.size:.1f}%)")

def validate_dimension_consistency(ctx: ConvertedAFinchContext):
    """Check that all arrays have consistent dimensions."""
    print("\nDimension Consistency Check:")
    
    n_reaches = None
    arrays_to_check = {
        'flow_accum': ctx.flow_accum,
        'y_est_adj_inc': ctx.y_est_adj_inc,
        'q_est_adj_inc': ctx.q_est_adj_inc,
        'con_adjust': ctx.con_adjust,
        'q_con_adj_inc': ctx.q_con_adj_inc,
        'y_con_adj_inc': ctx.y_con_adj_inc,
    }
    
    for name, arr in arrays_to_check.items():
        if arr is None:
            print(f"  {name:20} - not computed")
        else:
            expected_shape = (n_reaches, 12) if n_reaches else (arr.shape[0], 12)
            matches = "✅" if arr.shape == expected_shape else "❌"
            print(f"  {matches} {name:20} {arr.shape}")
            if n_reaches is None:
                n_reaches = arr.shape[0]

def compare_before_after(ctx_before: ConvertedAFinchContext, 
                          ctx_after: ConvertedAFinchContext,
                          array_name: str):
    """
    Compare an array between two context states.
    Useful for checking if constraint step changed values.
    
    Example:
        ctx_before_constraint = ... # context after step 3
        ctx_after_constraint = ... # context after step 4
        compare_before_after(ctx_before_constraint, ctx_after_constraint, 'q_est_adj_inc')
    """
    arr_before = getattr(ctx_before, array_name, None)
    arr_after = getattr(ctx_after, array_name, None)
    
    if arr_before is None or arr_after is None:
        print(f"Cannot compare {array_name}: one or both not available")
        return
    
    if arr_before.shape != arr_after.shape:
        print(f"Shape mismatch: {arr_before.shape} vs {arr_after.shape}")
        return
    
    diff = np.abs(arr_after - arr_before)
    mask = ~(np.isnan(arr_before) | np.isnan(arr_after))
    
    if not np.any(mask):
        print(f"{array_name}: No valid values to compare")
        return
    
    valid_diff = diff[mask]
    changed = np.sum(valid_diff > 1e-10)
    
    print(f"\n{array_name} Comparison:")
    print(f"  Total elements:  {arr_before.size}")
    print(f"  Changed:         {changed} ({100*changed/arr_before.size:.1f}%)")
    print(f"  Max difference:  {np.max(valid_diff):.6f}")
    print(f"  Mean difference: {np.mean(valid_diff):.6f}")
    
    if changed == 0:
        print(f"  ℹ️  No changes detected")

# Example usage in a script:
if __name__ == "__main__":
    from run_converted_afinch_full_model import ConvertedAFinchPipeline
    
    pipeline = ConvertedAFinchPipeline(
        base_dir=Path("."),
        src_dir=Path("./src"),
        ths="HillCountry",
        hsr_key="Brazos",
        wy1=2010,
        ny=1,
    )
    
    ctx = pipeline.run_all()
    
    # Use the inspection functions
    print(quick_check(ctx))
    print_shapes(ctx)
    print_stats(ctx, 'flow_accum')
    print_stats(ctx, 'con_adjust')
    validate_dimension_consistency(ctx)
