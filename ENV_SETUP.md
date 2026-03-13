# WAM Model Environment Setup

## 1. Create the Conda environment

In path where the model is stored:

```powershell
conda env create -f environment.yml
```

## 2. Activate the environment

```powershell
conda activate wam-model
```

## 3. Verify key packages

```powershell
python -c "import geopandas, pandas, numpy, scipy, networkx, PIL, pyogrio; print('Environment OK')"
```

## 4. Run the GUI

```powershell
python wam_gui.py
```

## 5. Run from command line (optional)

```powershell
python brazos_streamflow_model.py --help
```

## 6. Update env if dependencies change

If `environment.yml` changes:

```powershell
conda env update -f environment.yml --prune
```

## 7. Remove environment (optional)

```powershell
conda env remove -n wam-model
```
