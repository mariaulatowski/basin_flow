"""
Debug script to inspect AFINCH pipeline output and context structure.
Helps identify data issues and unexpected outputs at each stage.
"""

import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from run_converted_afinch_full_model import ConvertedAFinchContext, ConvertedAFinchPipeline

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"{Colors.OKBLUE}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*50}{Colors.ENDC}")

def inspect_array(name: str, arr: np.ndarray | None, max_items: int = 5):
    """Inspect and pretty-print numpy array information."""
    if arr is None:
        print(f"  {name}: {Colors.WARNING}None{Colors.ENDC}")
        return
    
    print(f"  {name}:")
    print(f"    dtype: {arr.dtype}, shape: {arr.shape}")
    print(f"    min: {np.nanmin(arr):.6f}, max: {np.nanmax(arr):.6f}, mean: {np.nanmean(arr):.6f}")
    if np.any(np.isnan(arr)):
        nan_count = np.sum(np.isnan(arr))
        nan_pct = (nan_count / arr.size) * 100
        print(f"    {Colors.WARNING}NaNs: {nan_count} ({nan_pct:.1f}%){Colors.ENDC}")
    if arr.size <= max_items:
        print(f"    values: {arr}")
    else:
        print(f"    first {max_items} values: {arr.flat[:max_items]}")

def inspect_dataframe(name: str, df: pd.DataFrame | None, max_rows: int = 3):
    """Inspect and pretty-print DataFrame information."""
    if df is None:
        print(f"  {name}: {Colors.WARNING}None{Colors.ENDC}")
        return
    
    print(f"  {name}:")
    print(f"    shape: {df.shape}, columns: {list(df.columns)}")
    if df.shape[0] > 0:
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"    null values: {dict(null_counts[null_counts > 0])}")
        print(f"    first {min(max_rows, df.shape[0])} rows:")
        print(df.head(max_rows).to_string(line_width=120).split('\n')[0:max_rows+2])

def inspect_context(ctx: ConvertedAFinchContext, verbose: bool = False):
    """Inspect the ConvertedAFinchContext object comprehensively."""
    print_section("CONTEXT INSPECTION")
    
    # Basic info
    print_subsection("Basic Configuration")
    print(f"  Water Year: {ctx.wy}")
    print(f"  Days in Month: {ctx.days_in_mo}")
    print(f"  Month Names: {ctx.mo_name}")
    
    # Data structures
    print_subsection("Data Structures")
    
    if ctx.nlcd:
        print(f"  NLCD Data Loaded:")
        print(f"    comid_ths: {len(ctx.nlcd.comid_ths) if hasattr(ctx.nlcd, 'comid_ths') else 'N/A'} items")
    
    if ctx.prism:
        print(f"  PRISM Data Loaded:")
        print(f"    prism_ths shape: {ctx.prism.prism_ths.shape if hasattr(ctx.prism, 'prism_ths') else 'N/A'}")
    
    if ctx.stations:
        print(f"  Stations: {len(ctx.stations)} stations loaded")
    
    if ctx.inflow:
        print(f"  Inflow Data:")
        print(f"    station_flow_df shape: {ctx.inflow.station_flow_df.shape if hasattr(ctx.inflow, 'station_flow_df') else 'N/A'}")
    
    # Arrays
    print_subsection("Numerical Arrays")
    inspect_array("p_in0 (precipitation)", ctx.p_in0)
    inspect_array("prsm_prec_ths", ctx.prsm_prec_ths)
    inspect_array("prsm_temp_ths", ctx.prsm_temp_ths)
    inspect_array("gc_area_sq_mi", ctx.gc_area_sq_mi)
    inspect_array("cb_matrix", ctx.cb_matrix)
    inspect_array("y_est_adj_inc", ctx.y_est_adj_inc)
    inspect_array("q_est_adj_inc", ctx.q_est_adj_inc)
    inspect_array("con_adjust", ctx.con_adjust)
    inspect_array("q_con_adj_inc", ctx.q_con_adj_inc)
    inspect_array("y_con_adj_inc", ctx.y_con_adj_inc)
    inspect_array("flow_comid", ctx.flow_comid)
    inspect_array("flow_accum", ctx.flow_accum)
    
    # Output paths
    print_subsection("Output Paths")
    print(f"  QY Output Path: {ctx.qy_path}")
    print(f"  Flow Accumulation Path: {ctx.flow_accum_path}")
    
    # Regression info
    print_subsection("Regression Results")
    if ctx.reg_var_name:
        print(f"  Regression variable names: {ctx.reg_var_name}")
    if ctx.reg_poa:
        print(f"  Regression POA loaded: Yes")
    if ctx.reg_hist:
        print(f"  Regression history loaded: Yes")
    
    # Modules loaded
    if ctx.modules:
        print_subsection("Loaded Modules")
        for mod_name in ctx.modules.keys():
            print(f"  - {mod_name}")

def inspect_step_outputs(pipeline: ConvertedAFinchPipeline, step_num: int):
    """Run pipeline up to step N and inspect outputs."""
    print_section(f"STEP {step_num} OUTPUT")
    
    ctx = pipeline.ctx
    
    if step_num >= 1:
        print_subsection("Step 1: Setup Inputs")
        pipeline.step_setup_inputs()
        print(f"  ✓ Data loaded and structured")
        print(f"    - NLCD comids: {len(ctx.nlcd.comid_ths)}")
        print(f"    - Stations: {len(ctx.stations)}")
        print(f"    - Precipitation shape: {ctx.p_in0.shape}")
    
    if step_num >= 2:
        print_subsection("Step 2: Run Regression")
        pipeline.step_run_regression()
        print(f"  ✓ Regression completed")
        print(f"    - Regression variables: {ctx.reg_var_name}")
        if ctx.reg_poa:
            print(f"    - POA regression outputs available")
    
    if step_num >= 3:
        print_subsection("Step 3: Estimate Incremental")
        pipeline.step_estimate_incremental()
        print(f"  ✓ Estimation completed")
        inspect_array("y_est_adj_inc", ctx.y_est_adj_inc, max_items=10)
        inspect_array("q_est_adj_inc", ctx.q_est_adj_inc, max_items=10)
    
    if step_num >= 4:
        print_subsection("Step 4: Constrain Incremental")
        pipeline.step_constrain_incremental()
        print(f"  ✓ Constraint applied")
        inspect_array("con_adjust", ctx.con_adjust, max_items=10)
    
    if step_num >= 5:
        print_subsection("Step 5: Write Output")
        pipeline.step_write_incremental_output()
        print(f"  ✓ Output written")
        print(f"    - QY path: {ctx.qy_path}")
    
    if step_num >= 6:
        print_subsection("Step 6: Accumulate Flow")
        pipeline.step_accumulate_flow()
        print(f"  ✓ Flow accumulation completed")
        inspect_array("flow_accum", ctx.flow_accum, max_items=10)
        inspect_array("flow_comid", ctx.flow_comid, max_items=10)

def compare_arrays(name: str, arr1: np.ndarray, arr2: np.ndarray):
    """Compare two numpy arrays for differences."""
    if arr1 is None or arr2 is None:
        print(f"  {name}: One or both arrays are None")
        return
    
    if arr1.shape != arr2.shape:
        print(f"  {Colors.FAIL}{name}: Shape mismatch - {arr1.shape} vs {arr2.shape}{Colors.ENDC}")
        return
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff[~np.isnan(diff)])
    
    if max_diff < 1e-10:
        print(f"  {Colors.OKGREEN}{name}: Arrays identical (max diff: {max_diff:.2e}){Colors.ENDC}")
    else:
        pct_diff = np.sum(diff > 1e-6) / arr1.size * 100
        print(f"  {Colors.WARNING}{name}: Differences detected (max: {max_diff:.6f}, mean: {mean_diff:.6f}, pct: {pct_diff:.1f}%){Colors.ENDC}")

def validate_context(ctx: ConvertedAFinchContext) -> dict[str, Any]:
    """Validate context state and collect issues."""
    issues = {
        'warnings': [],
        'errors': [],
        'missing': []
    }
    
    # Check for required fields
    if ctx.wy is None:
        issues['missing'].append("Water year not set")
    
    if ctx.p_in0 is None:
        issues['errors'].append("Precipitation data (p_in0) not initialized")
    else:
        if np.any(np.isnan(ctx.p_in0)):
            nan_pct = np.sum(np.isnan(ctx.p_in0)) / ctx.p_in0.size * 100
            issues['warnings'].append(f"p_in0 contains {nan_pct:.1f}% NaN values")
    
    if ctx.flow_accum is None:
        issues['missing'].append("Flow accumulation not computed")
    else:
        if np.any(ctx.flow_accum < 0):
            neg_count = np.sum(ctx.flow_accum < 0)
            issues['errors'].append(f"flow_accum contains {neg_count} negative values")
    
    if ctx.qy_path is None:
        issues['missing'].append("QY output path not set")
    
    if ctx.con_adjust is None:
        issues['missing'].append("Constraint adjustment not computed")
    
    return issues

def main():
    """Main debug workflow."""
    # Configuration (match run_converted_afinch_full_model.py)
    ROOT = Path(__file__).parent.parent
    SRC = ROOT / "src"
    
    # Create pipeline
    print(f"{Colors.OKGREEN}{Colors.BOLD}Initializing AFINCH Debug Pipeline...{Colors.ENDC}")
    pipeline = ConvertedAFinchPipeline(
        base_dir=ROOT,
        src_dir=SRC,
        ths="HillCountry",
        hsr_key="Brazos",
        wy1=2010,
        ny=1,
        logger=print,
    )
    
    # Ask user what to inspect
    print("\nDebug Options:")
    print("1. Inspect full context after complete run")
    print("2. Step-by-step inspection (choose step)")
    print("3. Validate context for issues")
    print("4. Quick diagnostic summary")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        print(f"\n{Colors.OKCYAN}Running full pipeline...{Colors.ENDC}")
        ctx = pipeline.run_all()
        inspect_context(ctx, verbose=True)
        
    elif choice == "2":
        step = input("Enter step number (1-6): ").strip()
        if step.isdigit() and 1 <= int(step) <= 6:
            inspect_step_outputs(pipeline, int(step))
        else:
            print("Invalid step number")
    
    elif choice == "3":
        print(f"\n{Colors.OKCYAN}Running full pipeline...{Colors.ENDC}")
        ctx = pipeline.run_all()
        issues = validate_context(ctx)
        
        print_section("VALIDATION REPORT")
        
        if issues['errors']:
            print(f"{Colors.FAIL}Errors ({len(issues['errors'])}):{Colors.ENDC}")
            for err in issues['errors']:
                print(f"  ✗ {err}")
        
        if issues['warnings']:
            print(f"{Colors.WARNING}Warnings ({len(issues['warnings'])}):{Colors.ENDC}")
            for warn in issues['warnings']:
                print(f"  ⚠ {warn}")
        
        if issues['missing']:
            print(f"{Colors.WARNING}Missing ({len(issues['missing'])}):{Colors.ENDC}")
            for miss in issues['missing']:
                print(f"  ○ {miss}")
        
        if not any([issues['errors'], issues['warnings'], issues['missing']]):
            print(f"{Colors.OKGREEN}✓ No issues detected!{Colors.ENDC}")
    
    elif choice == "4":
        print(f"\n{Colors.OKCYAN}Running full pipeline...{Colors.ENDC}")
        ctx = pipeline.run_all()
        
        print_section("QUICK DIAGNOSTIC")
        print(f"  ✓ Pipeline completed")
        print(f"  Water Year: {ctx.wy}")
        print(f"  Output paths set: {ctx.qy_path is not None}")
        
        if ctx.flow_accum is not None:
            print(f"  Flow accumulation: {ctx.flow_accum.shape}")
            print(f"    - Min: {np.nanmin(ctx.flow_accum):.2f}")
            print(f"    - Max: {np.nanmax(ctx.flow_accum):.2f}")
            print(f"    - Mean: {np.nanmean(ctx.flow_accum):.2f}")
        
        issues = validate_context(ctx)
        if any([issues['errors'], issues['warnings'], issues['missing']]):
            print(f"\n  {Colors.WARNING}Issues found: {len(issues['errors'])} errors, {len(issues['warnings'])} warnings{Colors.ENDC}")
        else:
            print(f"\n  {Colors.OKGREEN}No issues detected!{Colors.ENDC}")
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
