"""
AFINCH Setup Data Module

Appendix 4: AFSetupData
Appendix 5: AFReadNLCD  
Appendix 6: AFReadPrismPrec

Purpose:
- Setup data structures for water year analysis
- Read NLCD (National Land Cover Database) catchment attributes
- Read PRISM (Precipitation-elevation Regressions on Independent Slopes) data
- Intersect flowlines, catchments, and precipitation data
- Create unified data structures for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLCDReader:
    """
    Read and process NLCD (National Land Cover Database) catchment attributes.
    
    NLCD provides land cover classifications at ~30m resolution, aggregated to 
    NHD catchments. Data includes:
    - ComID: NHD HyBase Common Identifier
    - GridCode: NLCD grid cell identifier
    - NLCD classes 11-92 (water, developed, barren, forest, shrub, herb, crop, etc.)
    """
    
    # NLCD class mappings
    NLCD_CLASSES = {
        11: 'Open Water',
        12: 'Perennial Ice/Snow',
        21: 'Developed, Open Space',
        22: 'Developed, Low Intensity',
        23: 'Developed, Medium Intensity',
        24: 'Developed, High Intensity',
        31: 'Barren Rock/Sand/Clay',
        32: 'Unconsolidated Shore',
        33: 'Barren Shrub/Scrub',
        41: 'Deciduous Forest',
        42: 'Evergreen Forest',
        43: 'Mixed Forest',
        51: 'Shrub/Scrub',
        61: 'Grassland/Herbaceous',
        71: 'Sedge/Herbaceous',
        72: 'Lichens/Moss',
        81: 'Pasture/Hay',
        82: 'Cultivated Crops',
        83: 'Other Ag',
        84: 'Woody Wetlands',
        85: 'Herbaceous Wetlands',
        91: 'Woody Wetlands',
        92: 'Emergent Herbaceous Wetlands',
    }
    
    def __init__(self, nlcd_file: Path, nhdflowline_file: Path):
        """
        Initialize NLCD reader.
        
        Parameters
        ----------
        nlcd_file : Path
            Path to catchmentattributesnlcd.txt file
        nhdflowline_file : Path
            Path to nhdflowline.txt file (ComID, LengthKm, ReachCode)
        """
        self.nlcd_file = Path(nlcd_file)
        self.nhdflowline_file = Path(nhdflowline_file)
        self.nlcd_df = None
        self.flowline_df = None
    
    def read_nlcd(self) -> pd.DataFrame:
        """
        Read NLCD catchment attributes file.
        
        Returns
        -------
        pd.DataFrame
            NLCD data with ComID, GridCode, and land cover percentages
        """
        if not self.nlcd_file.exists():
            raise FileNotFoundError(f"NLCD file not found: {self.nlcd_file}")
        
        # NLCD columns: ComID, GridCode, NLCD11-92, PCTCN, PCTMX, SUMPCT
        col_names = ['ComID', 'GridCode'] + [f'NLCD_{c}' for c in self.NLCD_CLASSES.keys()] + \
                   ['PCTCN', 'PCTMX', 'SUMPCT']
        
        self.nlcd_df = pd.read_csv(
            self.nlcd_file,
            sep=',',
            header=0,
            dtype={'ComID': 'int64', 'GridCode': 'int64'}
        )
        
        logger.info(f"Read NLCD data: {len(self.nlcd_df)} records")
        return self.nlcd_df
    
    def read_nhdflowline(self) -> pd.DataFrame:
        """
        Read NHD Flowline file.
        
        Returns
        -------
        pd.DataFrame
            Flowline data with ComID, LengthKm, ReachCode
        """
        if not self.nhdflowline_file.exists():
            raise FileNotFoundError(f"Flowline file not found: {self.nhdflowline_file}")
        
        self.flowline_df = pd.read_csv(
            self.nhdflowline_file,
            sep=',',
            header=0,
            dtype={'ComID': 'int64', 'LengthKm': 'float64', 'ReachCode': 'str'}
        )
        
        logger.info(f"Read flowline data: {len(self.flowline_df)} records")
        return self.flowline_df
    
    def subset_by_reachcode(self, ths_code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract Target Hydrologic Subregion (THS) from regional data.
        
        Parameters
        ----------
        ths_code : str
            4-digit THS code (e.g., '0101')
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (filtered_flowlines, filtered_nlcd)
        """
        if self.flowline_df is None:
            self.read_nhdflowline()
        if self.nlcd_df is None:
            self.read_nlcd()
        
        # Extract THS from ReachCode (first 4 characters)
        ths_mask = self.flowline_df['ReachCode'].str.startswith(ths_code)
        flowline_ths = self.flowline_df[ths_mask].copy()
        
        logger.info(f"Selected {len(flowline_ths)} flowlines for THS {ths_code}")
        
        return flowline_ths, self.nlcd_df
    
    def intersect_and_join(self, flowline_ths: pd.DataFrame, ths_code: str) -> pd.DataFrame:
        """
        Intersect NHD flowlines and NLCD catchments for THS.
        
        Parameters
        ----------
        flowline_ths : pd.DataFrame
            Flowlines filtered to THS
        ths_code : str
            Target Hydrologic Subregion code
            
        Returns
        -------
        pd.DataFrame
            Merged data with both flowline and NLCD attributes
        """
        # Find common ComIDs between flowlines and NLCD
        common_comids = set(flowline_ths['ComID']) & set(self.nlcd_df['ComID'])
        
        flowline_subset = flowline_ths[flowline_ths['ComID'].isin(common_comids)].copy()
        nlcd_subset = self.nlcd_df[self.nlcd_df['ComID'].isin(common_comids)].copy()
        
        # Merge
        merged = flowline_subset.merge(nlcd_subset, on='ComID', how='inner')
        
        logger.info(f"Matched {len(merged)} ComIDs (flowline ∩ NLCD)")
        logger.info(f"Difference: {len(flowline_subset) - len(merged)} flowlines without NLCD data")
        
        return merged


class PRISMPrecipitationReader:
    """
    Read and process PRISM precipitation data.
    
    PRISM: Parameter-elevation Regressions on Independent Slopes Model
    - Gridded monthly precipitation at ~4 km resolution (2.5 arcmin)
    - Aggregated to NHD catchments
    - 13 months per water year (Oct-Sep): columns for Oct(month 1) through Sep(month 12) 
      plus annual average (month 13)
    """
    
    def __init__(self, prism_file: Path, gridcode_comid_file: Path):
        """
        Initialize PRISM reader.
        
        Parameters
        ----------
        prism_file : Path
            Path to monthly PRISM precipitation file (e.g., PrismPrecipWY2010.dat)
        gridcode_comid_file : Path
            Path to GridCode-ComID crosswalk file
        """
        self.prism_file = Path(prism_file)
        self.gridcode_comid_file = Path(gridcode_comid_file)
        self.prism_df = None
        self.xwalk_df = None
    
    def read_prism(self) -> pd.DataFrame:
        """
        Read PRISM precipitation file.
        
        Returns
        -------
        pd.DataFrame
            PRISM data with GridCode, Area, and 13 monthly precipitation columns
        """
        if not self.prism_file.exists():
            raise FileNotFoundError(f"PRISM file not found: {self.prism_file}")
        
        # PRISM format: GridCode, AreaSqMi, P_Oct, P_Nov, ..., P_Sep, P_Annual
        month_names = ['P_Oct', 'P_Nov', 'P_Dec', 'P_Jan', 'P_Feb', 'P_Mar',
                      'P_Apr', 'P_May', 'P_Jun', 'P_Jul', 'P_Aug', 'P_Sep', 'P_Annual']
        
        col_names = ['GridCode', 'AreaSqMi'] + month_names
        
        self.prism_df = pd.read_csv(
            self.prism_file,
            sep=r'\s+',
            header=0,
            names=col_names,
            dtype={'GridCode': 'int64', 'AreaSqMi': 'float64'}
        )
        
        logger.info(f"Read PRISM data: {len(self.prism_df)} grid cells")
        return self.prism_df
    
    def read_gridcode_comid_xwalk(self) -> pd.DataFrame:
        """Read GridCode-ComID crosswalk file."""
        if not self.gridcode_comid_file.exists():
            raise FileNotFoundError(f"GridCode-ComID file not found: {self.gridcode_comid_file}")
        
        self.xwalk_df = pd.read_csv(
            self.gridcode_comid_file,
            sep=',',
            header=0,
            dtype={'GridCode': 'int64', 'ComID': 'int64'}
        )
        
        logger.info(f"Read gridcode-comid crosswalk: {len(self.xwalk_df)} records")
        return self.xwalk_df
    
    def intersect_with_ths(self, gridcodes_ths: pd.DataFrame) -> pd.DataFrame:
        """
        Intersect PRISM grid cells with THS catchments.
        
        Parameters
        ----------
        gridcodes_ths : pd.DataFrame
            GridCodes for catchments in THS (from NLCD)
            
        Returns
        -------
        pd.DataFrame
            PRISM data matched to THS catchments
        """
        if self.prism_df is None:
            self.read_prism()
        
        # Find PRISM grid cells in THS catchments
        ths_gridcodes = set(gridcodes_ths['GridCode'].values)
        prism_ths = self.prism_df[self.prism_df['GridCode'].isin(ths_gridcodes)].copy()
        
        if len(prism_ths) < len(ths_gridcodes):
            missing = len(ths_gridcodes) - len(prism_ths)
            pct_missing = (missing / len(ths_gridcodes)) * 100
            logger.warning(f"Missing PRISM data for {missing} ({pct_missing:.1f}%) grid cells in THS")
        
        return prism_ths


class AFSetupData:
    """
    Unified workflow for setting up AFINCH analysis data.
    
    Combines NLCD, Flowlines, PRISM, and other data sources into
    integrated data structures for the water year analysis.
    """
    
    def __init__(self, base_dir: Path, ths_number: str, ths_name: str, water_year: int):
        """
        Initialize setup for given THS and water year.
        
        Parameters
        ----------
        base_dir : Path
            Base directory containing HSR subdirectories
        ths_number : str
            4-digit THS code (e.g., '0101')
        ths_name : str
            THS name
        water_year : int
            Water year (Oct [WY-1] to Sep [WY])
        """
        self.base_dir = Path(base_dir)
        self.ths_number = ths_number
        self.ths_name = ths_name
        self.water_year = water_year
        self.hsr_code = ths_number[:2]  # First 2 digits for regional directory
        self.data = {}
    
    def setup_water_year_info(self) -> Dict[str, int]:
        """
        Setup water year calendar information.
        
        Returns
        -------
        Dict[str, int]
            Days in each month for the water year
        """
        # Water year: Oct [WY-1] to Sep [WY]
        wy_start_year = self.water_year - 1
        
        # Days per month (accounting for leap years)
        days_in_month = {
            'Oct': 31,
            'Nov': 30,
            'Dec': 31,
            'Jan': 31,
            'Feb': 29 if self._is_leap_year(wy_start_year) else 28,
            'Mar': 31,
            'Apr': 30,
            'May': 31,
            'Jun': 30,
            'Jul': 31,
            'Aug': 31,
            'Sep': 30,
        }
        
        total_days = sum(days_in_month.values())
        self.data['days_in_month'] = days_in_month
        self.data['total_days_wy'] = total_days
        
        logger.info(f"\n{'='*70}")
        logger.info(f"AFINCH: Analysis of Flow in Networks of Channels")
        logger.info(f"Water Year {self.water_year} in Target Hydrologic Subregion (THS) {self.ths_number}")
        logger.info(f"{'='*70}")
        logger.info(f"Total days in WY{self.water_year}: {total_days}")
        
        return days_in_month
    
    def run_setup_workflow(self) -> Dict[str, pd.DataFrame]:
        """
        Execute full setup workflow.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Integrated data structures for analysis
        """
        logger.info("\nStep 1: Setup Water Year Calendar")
        self.setup_water_year_info()
        
        logger.info("\nStep 2: Read NLCD Data")
        self._run_read_nlcd()
        
        logger.info("\nStep 3: Read PRISM Precipitation Data")
        self._run_read_prism_prec()
        
        logger.info("\nStep 4: Generate Structure Data (from station lists)")
        # This would call AFGenStrucData
        
        logger.info("\nStep 5: Read Inflow and Water Use Data")
        # This would call AFReadInFlowWY
        
        logger.info("\nStep 6: Generate Basin Grid ComID Data")
        # This would call AFStaBasinGridComIDWY
        
        logger.info("\nSetup complete!")
        return self.data
    
    def _run_read_nlcd(self):
        """Read and process NLCD data for THS."""
        nlcd_file = self.base_dir / f"HSR{self.hsr_code}00" / "NLCD" / "catchmentattributesnlcd.txt"
        flowline_file = self.base_dir / f"HSR{self.hsr_code}00" / "Flowlines" / "nhdflowline.txt"
        
        reader = NLCDReader(nlcd_file, flowline_file)
        reader.read_nlcd()
        reader.read_nhdflowline()
        
        flowline_ths, nlcd_all = reader.subset_by_reachcode(self.ths_number)
        merged = reader.intersect_and_join(flowline_ths, self.ths_number)
        
        self.data['nlcd'] = merged
        logger.info(f"NLCD data ready: {len(merged)} ComIDs with land cover data")
    
    def _run_read_prism_prec(self):
        """Read and process PRISM precipitation data for water year."""
        prism_file = (self.base_dir / f"HSR{self.hsr_code}00" / "PRISM" / "Precipitation" /
                     f"PrismPrecipWY{self.water_year}.dat")
        gridcode_comid_file = self.base_dir / f"HSR{self.hsr_code}00" / "Flowlines" / "GridCodeComID.txt"
        
        reader = PRISMPrecipitationReader(prism_file, gridcode_comid_file)
        reader.read_prism()
        reader.read_gridcode_comid_xwalk()
        
        # Intersect with THS catchments
        if 'nlcd' in self.data:
            prism_ths = reader.intersect_with_ths(self.data['nlcd'])
            self.data['prism'] = prism_ths
            logger.info(f"PRISM data ready: {len(prism_ths)} grid cells with precipitation")
    
    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """Check if year is leap year."""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


if __name__ == '__main__':
    # Example usage
    base_dir = Path('../../HSR_Data')  # Parent directory containing HSR subdirectories
    
    setup = AFSetupData(
        base_dir=base_dir,
        ths_number='0101',
        ths_name='Upper Midwest',
        water_year=2010
    )
    
    data = setup.run_setup_workflow()
    
    print(f"\nSetup data keys: {list(data.keys())}")
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {key}: {len(df)} records")
