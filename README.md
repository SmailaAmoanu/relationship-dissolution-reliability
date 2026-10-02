# Reliability of Computational Predictions in Longitudinal Social Data

This repository contains the reproducibility materials for the study **“Evaluating the Reliability of Computational Predictions in Longitudinal Social Data: Relationship Dissolution Across Forecast Horizons.”**

The study uses the 2017 wave of the **How Couples Meet and Stay Together (HCMST)** panel as a common forecast origin and evaluates relationship dissolution by approximately three years (2020) and five years (2022). The analysis examines predictive performance together with validation stability, stability of model explanations, probability calibration, and outcome-specific uncertainty.

## Study design

Only information observed in 2017 is used as predictor information. The 2020 and 2022 waves are used to construct later outcomes.

| Forecast endpoint | Usable N | Dissolutions | Continuations | Dissolution rate |
|---|---:|---:|---:|---:|
| 2020 (about 3 years) | 1,675 | 183 | 1,492 | 10.9% |
| 2022 (about 5 years) | 1,388 | 218 | 1,170 | 15.7% |

The predictive models are logistic regression, elastic-net logistic regression, random forest, and XGBoost. Reliability analyses include repeated and nested validation, SHAP rank stability, probability calibration, pooled split-conformal prediction, class-specific coverage, and class-conditional conformal prediction.

## Repository contents

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── src/
│   ├── 00_verify_data.py
│   ├── 01_data_audit_eda.py
│   ├── 02_model_benchmark.py
│   ├── 03_explainability_uncertainty.py
│   ├── 04_robustness.py
│   ├── 05_nested_tuning.py
│   ├── 06_reliability_analysis.py
│   ├── 07_make_figures.py
│   ├── common.py
│   └── run_all.py
├── notebooks/
│   └── analysis_overview.ipynb
├── data/
│   ├── raw/README.md
│   └── processed/
├── results/reference/
├── figures/
└── docs/
    ├── REPRODUCIBILITY.md
    └── OUTPUT_MAP.md
```

## Data

The analysis uses:

Rosenfeld, M. J., Thomas, R. J., & Hausen, S. (2024). *How Couples Meet and Stay Together (HCMST) 2017, 2020, 2022, United States*. ICPSR. DOI: 10.3886/ICPSR38873.v1.

The raw `.dta` file is not redistributed in this repository. Download the public-use file from ICPSR and place it in `data/raw/` as:

```text
HCMST 2017 to 2022 small public version 2.2.dta
```

The analysis copy used for the study has SHA-256:

```text
a76c1ef8e14a69951db3d71142dfb3866063c7af8d0da67942b744e2727aeb56
```

`00_verify_data.py` checks this hash before analysis.

## Reproduce the analysis

Python 3.11 or later is recommended.

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
python src/run_all.py
```

Fresh outputs are written to `reproduced_results/`. Reference outputs from the manuscript analysis are retained under `results/reference/` for comparison.

For a shorter walkthrough of the study design and principal outputs, open `notebooks/analysis_overview.ipynb`. The scripts in `src/` are the authoritative reproduction workflow.

## Interpretation

The analysis is predictive rather than causal. SHAP values describe the fitted prediction function and should not be interpreted as causal effects. Marginal conformal coverage does not imply equal coverage within each outcome class.

## Documentation

`docs/REPRODUCIBILITY.md` describes the computational stages. `docs/OUTPUT_MAP.md` maps manuscript results to the scripts that produce them.

## Author

Smaila Amoanu  
Department of Mathematics and Statistics  
University of Nevada, Reno
