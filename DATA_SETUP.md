# Data Setup for Reproducible AFINCH Runs

This repository includes code and lightweight config files needed to run the AFINCH workflow.
Large geospatial datasets are intentionally excluded from git and must be downloaded locally.

## 1) Clone and create the Python environment

```powershell
git clone https://github.com/mariaulatowski/basin_flow.git
cd basin_flow
conda env create -f environment.yml
conda activate <env-name-from-environment-yml>
```

## 2) Required large input data (download separately)

Place files under `afinch_matlab_source/input_data/` using the structure below.

### A) Basin boundary shapefile

Folder:
- `afinch_matlab_source/input_data/basin/`

Required files (example):
- `TWDB_MRBs_2014.shp`
- `.dbf`, `.shx`, `.prj` sidecars

### B) NHDPlus geodatabases (Texas HU4 coverage)

Folder:
- `afinch_matlab_source/input_data/nhd/_extracted_gdb/`

Expected content:
- extracted HU4 geodatabase folders like `NHDPLUS_H_1205_HU4_GDB/`, etc.

### C) NLCD raster

Folder:
- `afinch_matlab_source/input_data/nlcd/`

Required file:
- `Annual_NLCD_LndCov_2018_CU_C1V1.tif`

### D) PRISM monthly rasters

Folders:
- `afinch_matlab_source/input_data/prism/precipitation/extracted/`
- `afinch_matlab_source/input_data/prism/temperature/extracted/`

Expected content:
- monthly precipitation TIFFs
- monthly mean temperature TIFFs

### E) Gage monthly flow CSV

Folder:
- `afinch_matlab_source/input_data/gages/`

Required file (current default):
- `monthly_wide_cfs.csv`

Minimum columns expected by the builder:
- station id: `Gage_ID_norm` or `Station`
- year: `Year` or `WY`
- monthly flow columns: `JAN`..`DEC`

### F) Optional WAM points CSV

Folder:
- `afinch_matlab_source/input_data/wam/`

Example file:
- `Brazos_new_wam_locations_nhdplus.csv`

## 3) What is auto-generated (do not download)

The Build Network step creates several derived files from your source data, including:
- `NHDFlowlineVAA.txt`
- `StationComID.csv`
- `ComIDStationDAMoAnQ{WY}.dat`
- catchment attribute outputs

Important:
- `NHDPlusCatchment_1200.gpkg` is a generated/derived catchment layer in this workflow and is not required as a source download for reproducibility.

## 4) Run from GUI

Launch:

```powershell
python afinch_comprehensive_gui.py
```

In the GUI:
1. Set Base Dir to the repo root.
2. Use Auto-Detect to populate paths.
3. Run Build Network (dry run first, then apply).
4. Run subsequent model steps.

## 5) Switching to a different Texas basin

To run a different basin:
1. Use a basin shapefile/value that identifies the target basin polygon.
2. Keep NHD/NLCD/PRISM inputs available for the corresponding spatial extent.
3. Provide matching gage monthly flow CSV for that basin.
4. Re-run Build Network so all derived files are rebuilt for the new basin.

## 6) Verification

Use the validator after build:

```powershell
python afinch_matlab_source/run/validation/validate_build_outputs.py --base-dir . --hsr HSR1200
```

A healthy build should show non-zero network connectivity and non-zero NLCD overlap.
