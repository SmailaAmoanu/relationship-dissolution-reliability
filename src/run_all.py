from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "00_verify_data.py",
    "01_data_audit_eda.py",
    "02_model_benchmark.py",
    "03_explainability_uncertainty.py",
    "04_robustness.py",
    "05_nested_tuning.py",
    "06_reliability_analysis.py",
    "07_make_figures.py",
]

for script in SCRIPTS:
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)

print("Analysis completed successfully.")
