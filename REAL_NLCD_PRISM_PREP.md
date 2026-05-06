# Real NLCD + PRISM Prep for Converted AFINCH

This document explains how to generate real-data versions of:
- `HSR1200/NLCD/catchmentattributesnlcd.txt`
- `HSR1200/PRISM/Precipitation/PrismPrecipWY<wy>.dat`
- `HSR1200/PRISM/Temperature/PrismTempAveWY<wy>.dat`

using:
- catchment polygons with COMIDs
- NLCD land-cover raster
- monthly PRISM ppt and tmean rasters

## Script

Use:
- `afinch_python_modules/build_real_nlcd_prism_inputs.py`

## Required inputs

1. Catchments vector file (GeoPackage/Shapefile/GeoJSON) with COMID field.
2. NLCD categorical raster for your target period.
3. PRISM monthly precipitation rasters (Oct..Sep for target WY).
4. PRISM monthly mean temperature rasters (Oct..Sep for target WY).
5. Existing xwalk file:
   - `HSR1200/Flowlines/GridCodeComID.txt`

## Command template

```powershell
C:/Users/mu3575/AppData/Local/anaconda3/envs/wam-model/python.exe \
  afinch_python_modules/build_real_nlcd_prism_inputs.py \
  --root C:/Users/mu3575/Documents/WAM \
  --hsr 1200 \
  --ths 1201 \
  --wy 2018 \
  --catchments C:/path/to/NHDPlusCatchment_1201.gpkg \
  --catchments-comid-field FEATUREID \
  --nlcd-raster C:/path/to/NLCD_2019_Land_Cover.tif \
  --prism-ppt-pattern "C:/path/to/prism/ppt/PRISM_ppt_stable_4kmM3_{yyyy}{mm:02d}_bil.tif" \
  --prism-tmean-pattern "C:/path/to/prism/tmean/PRISM_tmean_stable_4kmM3_{yyyy}{mm:02d}_bil.tif"
```

Notes:
- `{yyyy}` and `{mm}` are replaced automatically.
- Water-year order is Oct (WY-1) through Sep (WY).
- The script computes `PIn_13` as mean of `PIn_01..PIn_12`.

## Outputs

1. `HSR1200/NLCD/catchmentattributesnlcd.txt`
2. `HSR1200/PRISM/Precipitation/PrismPrecipWY2018.dat`
3. `HSR1200/PRISM/Temperature/PrismTempAveWY2018.dat`

## Next step: run converted core

```powershell
C:/Users/mu3575/AppData/Local/anaconda3/envs/wam-model/python.exe \
  afinch_python_modules/run_converted_afinch_core_with_adapter.py
```

This will use the updated NLCD/PRISM files in `HSR1200`.
