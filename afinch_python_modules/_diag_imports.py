import subprocess
import sys

FILES = [
    "AFBoxplotExplanVar",
    "AFCallRegCheckBox",
    "AFConFlowAccum",
    "AFGenLag1Precp",
    "AFGenStrucData",
    "AFid",
    "AFImagePOAYield",
    "AFPlotAreasFlows",
    "AFPlotQmMeaEst",
    "AFPlotRegressCoeff",
    "AFQConAdjInc",
    "AFQEstAdjInc",
    "AFReadInFlowWY",
    "AFReadNLCD",
    "AFReadPrismPrec",
    "AFReadPrismTemp",
    "AFRegCheckBoxGUI",
    "AFRegressByWY",
    "AFRegressPOA",
    "AFsetupData",
    "AFStaBasinGridComIDWY",
    "AFTrendDurations",
    "AFWrtQYEstCon",
    "AFYieldAtGagesGUI",
    "AFYieldImage",
    "FKenSen",
    "gui_afinch",
    "initialize_var_AFlniAFStrct",
    "starting_afinch",
]

ROOT = r"c:\Users\mu3575\Documents\WAM\afinch_matlab_source"

for f in FILES:
    code = (
        "from importlib.machinery import SourceFileLoader as L\n"
        "from importlib.util import module_from_spec, spec_from_loader\n"
        f"p=r'{ROOT}\\{f}'\n"
        "ld=L('m', p)\n"
        "sp=spec_from_loader('m', ld)\n"
        "m=module_from_spec(sp)\n"
        "ld.exec_module(m)\n"
        "print('ok')\n"
    )
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    tag = "PASS" if cp.returncode == 0 else "FAIL"
    print(f"{tag} {f} rc={cp.returncode}")
    if cp.stdout.strip():
        print("  out:", cp.stdout.strip()[:200])
    if cp.stderr.strip():
        print("  err:", cp.stderr.strip().splitlines()[-1][:200])
