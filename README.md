# Environmental Data Analysis

A reproducible Python workflow for cleaning environmental observations, exploring numeric associations, and evaluating regression models against a mean baseline. Supports your own CSV or an explicitly synthetic air-quality demonstration. No API keys or data downloads required.

## Quick start

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m environmental_analysis --demo --output outputs/demo
python -m unittest discover -s tests -v
```

The demo generates daily temperature (°C), humidity (%), wind speed (m/s), an illustrative traffic index, and PM2.5 (µg/m³). Its target is generated from a known mathematical relationship with noise. It includes missing predictor values and duplicate rows to exercise cleaning. **It is not measured data and must not be used to make environmental or health claims.**

## Analyze observations

```bash
python -m environmental_analysis --input observations.csv \
  --target pm25_ug_m3 --date-column date --output outputs/observations
```

Use one observation per CSV row, unique column names, numeric predictors, and a numeric target. Dates must be parseable and complete when `--date-column` is supplied. Other text columns are excluded. Remove identifiers, future information, and columns derived from the target before running. At least 30 usable rows and one numeric predictor are required. Without a date column the workflow uses a seeded random 80/20 split; use this only for independent observations. Chronological splits keep equal timestamps on the same side of the boundary. Repeated sites or spatially related observations require a separate grouped validation design.

## Analysis and outputs

- Removes exact duplicates and invalid/missing targets; retains outliers for inspection.
- Reports missingness, excluded columns, descriptive statistics, and Pearson correlations.
- Fits a training-mean baseline, standardized Ridge regression, and a random forest. Predictor imputation and scaling are fitted only on training rows.
- Evaluates held-out MAE, RMSE, and R²; computes random-forest permutation importance with repeated shuffles.
- Writes `report.md`, `analysis_metadata.json`, `cleaned_data.csv`, `summary_statistics.csv`, `correlations.csv`, `model_metrics.csv`, `predictions.csv`, `feature_importance.csv`, `correlations.png`, and `evaluation.png`.

`source_row` in predictions is the zero-based original CSV row index (excluding the header). Missing values in the cleaned export remain missing; imputation happens inside model pipelines. Correlations use pairwise complete observations. Importance is predictive association, not causation; correlated inputs can share importance. Scores from one holdout are exploratory, without confidence intervals. The workflow does not tune models or establish causal effects.

Use `--seed` for reproducible generation, splitting, and model randomness. Plots use a headless backend. Outputs are ignored by Git; keep sensitive observations out of commits. Dependencies are declared in `pyproject.toml`; `requirements.txt` records the versions used for validation on Python 3.12.
