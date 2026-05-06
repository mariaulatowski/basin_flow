#!/usr/bin/env python3
"""
Compare AFINCH model output to streamgage observations.
Generates visual and statistical comparisons for validation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns
from scipy import stats

# Configuration
MODEL_OUTPUT = Path('HSR1200/Output/FlowAccum/ComIDQ12WY2018.csv')
GAGE_DATA = Path('inputData/inputs/monthly_wide_cfs.csv')
STATION_COMID_MAP = Path('HSR1200/Streamflow/StationDASqMi.csv')
OUTPUT_DIR = Path('HSR1200/Output/Comparisons')
OUTPUT_DIR.mkdir(exist_ok=True)

# Water year 2018 months (Oct 2017 - Sep 2018)
WATER_YEAR_MONTHS = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
CALENDAR_MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

def load_data():
    """Load model output and gage data."""
    print("Loading data...")
    
    # Load model output
    model_df = pd.read_csv(MODEL_OUTPUT)
    print(f"Model output: {len(model_df)} reaches")
    
    # Load gage data
    gage_df = pd.read_csv(GAGE_DATA)
    print(f"Gage data: {len(gage_df)} total records")
    
    # Load station-ComID mapping
    if STATION_COMID_MAP.exists():
        station_map = pd.read_csv(STATION_COMID_MAP)
        # Convert types and handle any whitespace
        station_map['Station'] = station_map['Station'].astype(str).str.strip()
        station_map['ComID'] = station_map['ComID'].astype(int)
        print(f"Station mapping: {len(station_map)} stations mapped")
    else:
        print(f"ERROR: {STATION_COMID_MAP} not found!")
        station_map = None
    
    return model_df, gage_df, station_map

def prepare_gage_data(gage_df, year=2018):
    """Prepare gage data for comparison (convert to water year order)."""
    print(f"\nPreparing gage data for WY{year}...")
    
    # For water year, we need Oct-Dec from previous year + Jan-Sep from current year
    # But the gage data CSV has calendar year rows, so:
    # WY2018 = Oct 2017 (from 2017 row) + Dec 2017 (from 2017 row) + Jan-Sep 2018 (from 2018 row)
    
    df_current = gage_df[gage_df['Year'] == year].copy() if year in gage_df['Year'].values else None
    df_prev = gage_df[gage_df['Year'] == year - 1].copy() if (year - 1) in gage_df['Year'].values else None
    
    if df_current is None:
        print(f"ERROR: No gage data found for Year {year}")
        return pd.DataFrame()
    
    print(f"  Current year {year}: {len(df_current)} records")
    if df_prev is not None:
        print(f"  Previous year {year-1}: {len(df_prev)} records")
    
    # Build result dataframe
    result = []
    
    for _, row_current in df_current.iterrows():
        gage_id = row_current['Gage_ID_norm']
        
        # Start with current year data
        row_dict = {
            'Gage_ID_norm': gage_id,
            'Station_Name': row_current.get('Station_Name', 'Unknown'),
            'LAT': row_current.get('LAT', np.nan),
            'LONG': row_current.get('LONG', np.nan),
        }
        
        # Add Jan-Sep from current year
        for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
            row_dict[month] = row_current.get(month, np.nan)
        
        # Try to add Oct-Dec from previous year
        if df_prev is not None:
            row_prev = df_prev[df_prev['Gage_ID_norm'] == gage_id]
            if len(row_prev) > 0:
                for month in ['OCT', 'NOV', 'DEC']:
                    row_dict[month] = row_prev.iloc[0].get(month, np.nan)
            else:
                # Can't get previous year data for this station
                row_dict['OCT'] = np.nan
                row_dict['NOV'] = np.nan
                row_dict['DEC'] = np.nan
        else:
            # Try to get from current year (some data might have full calendar)
            row_dict['OCT'] = row_current.get('OCT', np.nan)
            row_dict['NOV'] = row_current.get('NOV', np.nan)
            row_dict['DEC'] = row_current.get('DEC', np.nan)
        
        result.append(row_dict)
    
    wy_df = pd.DataFrame(result)
    wy_cols = ['Gage_ID_norm', 'Station_Name', 'LAT', 'LONG'] + WATER_YEAR_MONTHS
    wy_df = wy_df[[c for c in wy_cols if c in wy_df.columns]]
    
    return wy_df

def match_gages_to_reaches(model_df, gage_df, station_map):
    """Create mapping between gage observations and model reaches."""
    print("\nMatching gages to model reaches...")
    
    comparisons = []
    
    if station_map is None:
        print("ERROR: No mapping file available!")
        return []
    
    # Build model lookup by ComID
    model_lookup = {}
    for _, row in model_df.iterrows():
        comid = int(row['ComIDVAA'])
        model_lookup[comid] = row
    
    # For each station in the mapping, find its gage data and model reach
    for _, map_row in station_map.iterrows():
        station_id = str(map_row['Station']).strip()
        comid = int(map_row['ComID'])
        
        # Find gage data for this station in WY2018
        gage_matches = gage_df[gage_df['Gage_ID_norm'].astype(str).str.strip() == station_id]
        
        if len(gage_matches) == 0:
            continue
        
        # Use first match (there should be only one row per station per year)
        gage_row = gage_matches.iloc[0]
        
        # Find model reach
        if comid not in model_lookup:
            print(f"Warning: ComID {comid} not found in model output for station {station_id}")
            continue
        
        model_row = model_lookup[comid]
        
        comparisons.append({
            'gage_id': station_id,
            'comid': comid,
            'station_name': gage_row.get('Station_Name', 'Unknown'),
            'gage_data': gage_row,
            'model_data': model_row
        })
    
    print(f"Matched {len(comparisons)} gage-reach pairs")
    return comparisons

def compute_statistics(obs, pred):
    """Compute performance metrics."""
    # Remove NaN/Inf
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs_clean = obs[mask]
    pred_clean = pred[mask]
    
    if len(obs_clean) == 0:
        return {}
    
    residuals = pred_clean - obs_clean
    
    # Metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    pbias = 100 * np.sum(residuals) / np.sum(np.abs(obs_clean))
    
    # Nash-Sutcliffe Efficiency
    nse = 1 - (np.sum(residuals ** 2) / np.sum((obs_clean - np.mean(obs_clean)) ** 2))
    
    # Kling-Gupta Efficiency
    cc = np.corrcoef(obs_clean, pred_clean)[0, 1]
    alpha = np.std(pred_clean) / np.std(obs_clean)
    beta = np.mean(pred_clean) / np.mean(obs_clean)
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    # Correlation
    r2 = cc ** 2
    
    return {
        'n': len(obs_clean),
        'mae': mae,
        'rmse': rmse,
        'pbias': pbias,
        'nse': nse,
        'kge': kge,
        'r2': r2,
        'correlation': cc
    }

def create_comparison_plots(comparisons):
    """Create visual comparisons."""
    print("\nGenerating plots...")
    
    # Map water year months to model column names
    model_col_map = {
        'OCT': 'QAccConOct',
        'NOV': 'QAccConNov',
        'DEC': 'QAccConDec',
        'JAN': 'QAccConJan',
        'FEB': 'QAccConFeb',
        'MAR': 'QAccConMar',
        'APR': 'QAccConApr',
        'MAY': 'QAccConMay',
        'JUN': 'QAccConJun',
        'JUL': 'QAccConJul',
        'AUG': 'QAccConAug',
        'SEP': 'QAccConSep'
    }
    
    months_wy = WATER_YEAR_MONTHS
    n_gages = min(len(comparisons), 10)  # Plot top 10
    n_plot = min(5, n_gages)
    
    # 1. Time series comparison (top 5 gages)
    if n_plot > 0:
        fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot))
        if n_plot == 1:
            axes = [axes]
        fig.suptitle('Model vs. Observed Monthly Streamflow (WY2018)', fontsize=14, fontweight='bold')
        
        for idx, comp in enumerate(comparisons[:n_plot]):
            ax = axes[idx]
            gage_row = comp['gage_data']
            model_row = comp['model_data']
            
            gage_id = comp['gage_id']
            station = comp['station_name'][:30]
            
            # Extract monthly values
            obs_vals = [gage_row.get(m, np.nan) for m in months_wy]
            pred_vals = [model_row.get(model_col_map[m], np.nan) for m in months_wy]
            
            x = np.arange(len(months_wy))
            
            ax.plot(x, obs_vals, 'o-', label='Observed', linewidth=2, markersize=6, color='blue')
            ax.plot(x, pred_vals, 's--', label='Modeled', linewidth=2, markersize=6, color='orange')
            
            stats_dict = compute_statistics(np.array(obs_vals), np.array(pred_vals))
            
            ax.set_title(f'{gage_id} - {station} (NSE={stats_dict.get("nse", np.nan):.3f}, RMSE={stats_dict.get("rmse", np.nan):.2f})',
                        fontsize=10)
            ax.set_ylabel('Flow (CFS)')
            ax.set_xticks(x)
            ax.set_xticklabels(months_wy, rotation=45)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'timeseries_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  → {OUTPUT_DIR / 'timeseries_comparison.png'}")
        plt.close()
    
    # 2. Scatter plot (all gages, all months)
    all_obs = []
    all_pred = []
    
    for comp in comparisons:
        gage_row = comp['gage_data']
        model_row = comp['model_data']
        
        for i, m in enumerate(months_wy):
            obs = gage_row.get(m, np.nan)
            pred = model_row.get(model_col_map[m], np.nan)
            if np.isfinite(obs) and np.isfinite(pred):
                all_obs.append(obs)
                all_pred.append(pred)
    
    all_obs = np.array(all_obs)
    all_pred = np.array(all_pred)
    
    if len(all_obs) > 0:
        stats_overall = compute_statistics(all_obs, all_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(all_obs, all_pred, alpha=0.6, s=50, color='steelblue')
        
        # 1:1 line
        lim_min = min(all_obs.min(), all_pred.min())
        lim_max = max(all_obs.max(), all_pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel('Observed Flow (CFS)', fontsize=12)
        ax.set_ylabel('Modeled Flow (CFS)', fontsize=12)
        ax.set_title(f'Model vs. Observed Flows\n(n={len(all_obs)}, NSE={stats_overall.get("nse", np.nan):.3f}, R²={stats_overall.get("r2", np.nan):.3f})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  → {OUTPUT_DIR / 'scatter_comparison.png'}")
        plt.close()
        
        # 3. Residuals plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        residuals = all_pred - all_obs
        
        # Residuals vs observed
        ax = axes[0]
        ax.scatter(all_obs, residuals, alpha=0.6, s=50, color='steelblue')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Observed Flow (CFS)', fontsize=11)
        ax.set_ylabel('Residual (Modeled - Observed, CFS)', fontsize=11)
        ax.set_title('Residuals vs. Observed', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax = axes[1]
        ax.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(x=np.mean(residuals), color='orange', linestyle='--', linewidth=2, label=f'Mean={np.mean(residuals):.2f}')
        ax.set_xlabel('Residual (CFS)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Residuals', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'residuals_analysis.png', dpi=150, bbox_inches='tight')
        print(f"  → {OUTPUT_DIR / 'residuals_analysis.png'}")
        plt.close()
    else:
        print("  Warning: Not enough matching data points for scatter plot")
        stats_overall = {}
    
    return stats_overall

def create_statistics_table(comparisons):
    """Create detailed statistics table."""
    print("\nGenerating statistics table...")
    
    # Map water year months to model column names
    model_col_map = {
        'OCT': 'QAccConOct',
        'NOV': 'QAccConNov',
        'DEC': 'QAccConDec',
        'JAN': 'QAccConJan',
        'FEB': 'QAccConFeb',
        'MAR': 'QAccConMar',
        'APR': 'QAccConApr',
        'MAY': 'QAccConMay',
        'JUN': 'QAccConJun',
        'JUL': 'QAccConJul',
        'AUG': 'QAccConAug',
        'SEP': 'QAccConSep'
    }
    
    stats_list = []
    
    for comp in comparisons:
        gage_row = comp['gage_data']
        model_row = comp['model_data']
        
        obs_vals = np.array([gage_row.get(m, np.nan) for m in WATER_YEAR_MONTHS])
        pred_vals = np.array([model_row.get(model_col_map[m], np.nan) for m in WATER_YEAR_MONTHS])
        
        stats_dict = compute_statistics(obs_vals, pred_vals)
        
        stats_list.append({
            'Gage_ID': comp['gage_id'],
            'Station_Name': comp['station_name'][:40],
            'ComID': comp['comid'],
            'N': stats_dict.get('n', 0),
            'MAE_CFS': stats_dict.get('mae', np.nan),
            'RMSE_CFS': stats_dict.get('rmse', np.nan),
            'PBIAS_%': stats_dict.get('pbias', np.nan),
            'NSE': stats_dict.get('nse', np.nan),
            'KGE': stats_dict.get('kge', np.nan),
            'R2': stats_dict.get('r2', np.nan),
            'Correlation': stats_dict.get('correlation', np.nan)
        })
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(OUTPUT_DIR / 'model_vs_gages_statistics.csv', index=False)
    print(f"  → {OUTPUT_DIR / 'model_vs_gages_statistics.csv'}")
    
    # Print summary
    print("\n" + "="*100)
    print("OVERALL STATISTICS SUMMARY")
    print("="*100)
    print(stats_df.to_string(index=False))
    print("="*100)
    
    return stats_df

def main():
    """Main workflow."""
    print("="*80)
    print("AFINCH Model vs. Streamgage Comparison (WY2018)")
    print("="*80)
    
    # Load data
    model_df, gage_df, station_map = load_data()
    
    # Prepare gage data for WY2018
    gage_wy = prepare_gage_data(gage_df, year=2018)
    
    # Match gages to reaches
    comparisons = match_gages_to_reaches(model_df, gage_wy, station_map)
    
    if len(comparisons) == 0:
        print("\nERROR: No gage-reach pairs matched!")
        return
    
    # Create plots
    overall_stats = create_comparison_plots(comparisons)
    
    # Create statistics table
    stats_df = create_statistics_table(comparisons)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL MODEL PERFORMANCE")
    print("="*80)
    print(f"Gages compared: {len(comparisons)}")
    print(f"Monthly observations: {stats_df['N'].sum()}")
    print(f"\nAggregate metrics (all gages, all months):")
    print(f"  MAE:           {overall_stats.get('mae', np.nan):.3f} CFS")
    print(f"  RMSE:          {overall_stats.get('rmse', np.nan):.3f} CFS")
    print(f"  PBIAS:         {overall_stats.get('pbias', np.nan):.2f} %")
    print(f"  NSE:           {overall_stats.get('nse', np.nan):.3f}")
    print(f"  KGE:           {overall_stats.get('kge', np.nan):.3f}")
    print(f"  R²:            {overall_stats.get('r2', np.nan):.3f}")
    print(f"  Correlation:   {overall_stats.get('correlation', np.nan):.3f}")
    print("="*80)
    
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - timeseries_comparison.png: Time series plots for top 5 gages")
    print("  - scatter_comparison.png: Scatter plot of all observations")
    print("  - residuals_analysis.png: Residual diagnostic plots")
    print("  - model_vs_gages_statistics.csv: Detailed per-gage statistics")

if __name__ == '__main__':
    main()
