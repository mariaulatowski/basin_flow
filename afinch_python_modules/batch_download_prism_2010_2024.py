"""
Batch download PRISM data for years 2010-2024 for multi-year regression.
"""

import sys
import time
from pathlib import Path

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


def download_prism_for_year(wy: int, out_dir: Path, elements: list[str]) -> bool:
    """Download PRISM data for a single water year."""
    print(f"\n{'='*70}")
    print(f"Downloading PRISM data for WY{wy}")
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
        
        print(f"\nElement: {element}")
        print(f"  Zip dir:       {zip_dir}")
        print(f"  Extracted dir: {extracted_dir}\n")
        
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
            
            # Skip if already extracted
            month_raster = find_extracted_raster(extracted_dir, element, wm.cal_year, wm.cal_month)
            if month_raster is not None:
                print(f"  ✓ SKIP existing: {filename}")
                continue
            
            # Download
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
                bil_count = len([p for p in extracted if p.suffix.lower() == ".bil"])
                tif_count = len([p for p in extracted if p.suffix.lower() == ".tif"])
                print(f"    ✓ EXTRACTED ({tif_count} tif, {bil_count} bil)")
            except Exception as exc:
                print(f"    ✗ EXTRACT ERROR: {exc}")
                wy_success = False
    
    return wy_success


def main():
    out_dir = Path(r"c:\Users\mu3575\Documents\WAM\inputData\prism_monthly")
    elements = ["ppt", "tmean"]
    water_years = list(range(2010, 2025))  # 2010-2024
    
    print(f"\n{'='*70}")
    print(f"BATCH PRISM DOWNLOAD: WY2010-WY2024")
    print(f"{'='*70}")
    print(f"Output directory: {out_dir}")
    print(f"Elements: {', '.join(elements)}")
    print(f"Water years: {water_years}")
    print(f"\nStarting download...\n")
    
    successful = []
    failed = []
    
    try:
        for wy in water_years:
            start_time = time.time()
            success = download_prism_for_year(wy, out_dir, elements)
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
    print(f"\nNext step: Run build_real_nlcd_prism_inputs.py to process rasters")
    print(f"           into aggregated grid format.")


if __name__ == "__main__":
    main()
