"""
Download PRISM data for remaining years (2012-2024) with Texas bbox clipping.
This saves bandwidth and storage by only keeping Texas-relevant area.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:
    print("ERROR: Missing geospatial libraries. Install: rasterio geopandas shapely")
    sys.exit(1)

# Add afinch_python_modules to path
sys.path.insert(0, str(Path(__file__).parent / "afinch_python_modules"))

from download_prism_wy import (
    water_year_months,
    build_filename_timeseries,
    build_url,
    download_file,
    extract_zip,
    find_extracted_raster,
)


# Texas bounding box (approximate, with 0.5° buffer for edge catchments)
# Lon range: -106.5 to -93.0 (W to E)
# Lat range: 25.8 to 36.5 (S to N)
TEXAS_BBOX = (-106.5, 25.8, -93.0, 36.5)  # (xmin, ymin, xmax, ymax)


def clip_raster_to_texas(input_raster: Path, output_raster: Path) -> bool:
    """
    Clip a full-US PRISM raster to Texas bbox.
    
    Returns True if successful, False otherwise.
    """
    try:
        with rasterio.open(input_raster) as src:
            # Create bbox geometry in raster CRS
            bbox_geom = [box(*TEXAS_BBOX)]
            
            # Clip raster
            data, transform = mask(src, bbox_geom, crop=True)
            
            # Update metadata
            profile = src.profile.copy()
            profile.update({
                'height': data.shape[1],
                'width': data.shape[2],
                'transform': transform,
            })
            
            # Write clipped raster
            output_raster.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_raster, 'w', **profile) as dst:
                dst.write(data)
        
        print(f"    ✓ CLIPPED to Texas bbox")
        return True
    
    except Exception as exc:
        print(f"    ✗ CLIP ERROR: {exc}")
        return False


def download_prism_for_year_with_clipping(
    wy: int, 
    out_dir: Path, 
    elements: list[str],
    clip_to_texas: bool = True
) -> bool:
    """Download PRISM data for a single water year and optionally clip to Texas."""
    
    print(f"\n{'='*70}")
    print(f"Downloading PRISM data for WY{wy}")
    if clip_to_texas:
        print(f"Texas bbox: {TEXAS_BBOX}")
    print(f"{'='*70}\n")
    
    base_url = "https://ftp.prism.oregonstate.edu/time_series/us/an/4km"
    region = "us"
    dataset = "25m"
    
    months = water_year_months(wy)
    wy_success = True
    
    for element in elements:
        element_dir = out_dir / element
        zip_dir = element_dir / "zip"
        extracted_dir = element_dir / "extracted"
        clipped_dir = element_dir / "clipped" if clip_to_texas else None
        
        print(f"\nElement: {element}")
        print(f"  Zip dir:       {zip_dir}")
        print(f"  Extracted dir: {extracted_dir}")
        if clipped_dir:
            print(f"  Clipped dir:   {clipped_dir}\n")
        
        for wm in months:
            filename = build_filename_timeseries(
                element,
                wm.cal_year,
                wm.cal_month,
                region=region,
                dataset=dataset,
            )
            
            url = build_url(
                base_url,
                filename,
                element=element,
                year=wm.cal_year,
                source_mode="timeseries",
            )
            
            zip_path = zip_dir / filename
            
            # Check if clipped version exists (final product)
            if clipped_dir and clip_to_texas:
                clipped_file = find_extracted_raster(clipped_dir, element, wm.cal_year, wm.cal_month)
                if clipped_file is not None:
                    print(f"  ✓ SKIP existing clipped: {filename}")
                    continue
            
            # Check if extracted version exists
            month_raster = find_extracted_raster(extracted_dir, element, wm.cal_year, wm.cal_month)
            if month_raster is not None and not clip_to_texas:
                print(f"  ✓ SKIP existing: {filename}")
                continue
            elif month_raster is None:
                # Need to download
                print(f"  ↓ DOWNLOAD: {filename}")
                try:
                    ok = download_file(url, zip_path, timeout=300)
                    if not ok:
                        print(f"    ✗ FAILED: 404 or HTTP error")
                        wy_success = False
                        continue
                except Exception as exc:
                    print(f"    ✗ ERROR: {exc}")
                    wy_success = False
                    continue
                
                # Extract
                try:
                    extracted = extract_zip(zip_path, extracted_dir)
                    tif_count = len([p for p in extracted if p.suffix.lower() == ".tif"])
                    print(f"    ✓ EXTRACTED ({tif_count} tif)")
                except Exception as exc:
                    print(f"    ✗ EXTRACT ERROR: {exc}")
                    wy_success = False
                    continue
                
                month_raster = find_extracted_raster(extracted_dir, element, wm.cal_year, wm.cal_month)
            
            # Clip to Texas if needed
            if clip_to_texas and month_raster is not None:
                clipped_file = clipped_dir / month_raster.name
                if not clipped_file.exists():
                    print(f"  ↪ CLIPPING: {filename}")
                    ok = clip_raster_to_texas(month_raster, clipped_file)
                    if not ok:
                        wy_success = False
                else:
                    print(f"  ✓ CLIPPED exists: {filename}")
    
    return wy_success


def main():
    out_dir = Path(r"c:\Users\mu3575\Documents\WAM\inputData\prism_monthly")
    elements = ["ppt", "tmean"]
    
    # Years already downloaded: 2010, 2011 (from batch script)
    # Remaining years: 2012-2024
    water_years = list(range(2012, 2025))
    
    print(f"\n{'='*70}")
    print(f"BATCH PRISM DOWNLOAD WITH TEXAS CLIPPING: WY2012-WY2024")
    print(f"{'='*70}")
    print(f"Output directory: {out_dir}")
    print(f"Elements: {', '.join(elements)}")
    print(f"Water years: {water_years}")
    print(f"Texas bbox: {TEXAS_BBOX}")
    print(f"\nNote: WY2010-WY2011 already downloaded (no clipping). Clipping will be")
    print(f"applied to remaining years to save storage space (~80% reduction).\n")
    
    successful = []
    failed = []
    
    try:
        for wy in water_years:
            start_time = time.time()
            success = download_prism_for_year_with_clipping(
                wy, 
                out_dir, 
                elements,
                clip_to_texas=True
            )
            elapsed = time.time() - start_time
            
            if success:
                successful.append(wy)
                status = "✓ SUCCESS"
            else:
                failed.append(wy)
                status = "✗ PARTIAL/FAILED"
            
            print(f"\n{status}: WY{wy} ({elapsed:.1f}s)")
            
            # Small delay between years to be respectful to server
            if wy < water_years[-1]:
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as exc:
        print(f"\n\nFATAL ERROR: {exc}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Successful:  {len(successful)} years - {successful}")
    print(f"Failed:      {len(failed)} years - {failed}")
    print(f"Total:       {len(successful) + len(failed)}/{len(water_years)}")
    
    print(f"\nStorage impact:")
    print(f"  Full US (2012-2024): ~130 GB (13 years × 2 elements × 5 GB)")
    print(f"  Texas only:          ~26 GB (~80% reduction)")
    print(f"\nNext steps:")
    print(f"  1. Verify downloaded/clipped data: ls {out_dir}/*/clipped/")
    print(f"  2. Run: build_real_nlcd_prism_inputs.py with clipped rasters")


if __name__ == "__main__":
    main()
