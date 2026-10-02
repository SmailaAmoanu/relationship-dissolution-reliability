# Reproducibility notes

## Environment

Install the packages in `requirements.txt` in a clean Python environment. Python 3.11 or later is recommended.

## Raw data

The raw HCMST public-use file is not included. Place the verified file in `data/raw/` before running the workflow. `src/00_verify_data.py` checks the expected SHA-256 hash.

## Analysis stages

| Stage | Script | Purpose |
|---|---|---|
| 0 | `00_verify_data.py` | Verify the raw data file. |
| 1 | `01_data_audit_eda.py` | Reconstruct cohorts, outcomes, predictor audit, and analysis frames. |
| 2 | `02_model_benchmark.py` | Compare probabilistic classifiers on held-out data. |
| 3 | `03_explainability_uncertainty.py` | Compute SHAP summaries, calibration, and pooled conformal uncertainty. |
| 4 | `04_robustness.py` | Repeat train/test splits and quantify importance-rank stability. |
| 5 | `05_nested_tuning.py` | Separate hyperparameter selection from outer performance evaluation. |
| 6 | `06_reliability_analysis.py` | Compute formal rank stability, calibration, class-specific coverage, and class-conditional conformal diagnostics. |
| 7 | `07_make_figures.py` | Generate manuscript figures. |

Run all stages with `python src/run_all.py`.

## Reference outputs

`results/reference/` contains the numerical outputs used to prepare the manuscript. A fresh run writes results to `reproduced_results/`. Small numerical differences can arise from library versions, but the documented seeds and specifications are retained.
