"""Analysis with held-out evaluation and training-only preprocessing."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def clean_data(frame, target):
    """Remove exact duplicates and unusable targets; leave feature imputation to models."""
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    if target not in frame:
        raise ValueError(f"Target column not found: {target}")
    result = frame.drop_duplicates().copy()
    result[target] = pd.to_numeric(result[target], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    missing_targets = int(result[target].isna().sum())
    result = result.dropna(subset=[target])
    features = [c for c in result.select_dtypes(include="number").columns
                if c != target and result[c].notna().any()]
    if len(result) < 30:
        raise ValueError("At least 30 usable observations are required")
    if not features:
        raise ValueError("At least one numeric predictor with observed values is required")
    if result[target].nunique() < 2:
        raise ValueError("Target must contain at least two distinct values")
    quality = {"input_rows": len(frame), "usable_rows": len(result),
               "duplicates_removed": len(frame) - len(frame.drop_duplicates()),
               "missing_or_invalid_targets_removed": missing_targets,
               "missing_values": {c: int(result[c].isna().sum()) for c in result},
               "features": features,
               "excluded_columns": [c for c in result if c not in features and c != target]}
    return result, quality


def run_analysis(frame, target, output_dir, seed=42, date_column=None, synthetic=False):
    """Write inspectable tables, plots, metrics, and a Markdown interpretation."""
    cleaned, quality = clean_data(frame, target)
    features = quality["features"]
    if date_column:
        if date_column == target or date_column not in cleaned:
            raise ValueError("Date column must exist and differ from the target")
        dates = pd.to_datetime(cleaned[date_column], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("Date column contains missing or invalid dates")
        cleaned = cleaned.assign(**{date_column: dates}).sort_values(date_column)
        features = [c for c in features if c != date_column]
        quality["features"] = features
        if not features:
            raise ValueError("No predictors remain after excluding the date column")
        cutoff = int(len(cleaned) * .8)
        # Keep tied timestamps together to avoid leakage across the boundary.
        boundary = cleaned[date_column].iloc[cutoff]
        train = cleaned[cleaned[date_column] < boundary]
        test = cleaned[cleaned[date_column] >= boundary]
        if len(train) < 10 or len(test) < 2:
            raise ValueError("Date split requires at least 10 earlier training rows and 2 test rows")
        split = "Chronological holdout: latest approximately 20% of observations"
    else:
        train, test = train_test_split(cleaned, test_size=.2, random_state=seed)
        split = "Random 80/20 holdout; appropriate only for independent observations"
    models = {
        "mean_baseline": make_pipeline(SimpleImputer(keep_empty_features=True), DummyRegressor()),
        "ridge": make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=1)),
        "random_forest": make_pipeline(SimpleImputer(keep_empty_features=True), RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=2)),
    }
    records = []
    predictions = pd.DataFrame({"source_row": test.index, "actual": test[target].to_numpy()})
    for name, model in models.items():
        model.fit(train[features], train[target])
        predicted = model.predict(test[features])
        predictions[name] = predicted
        records.append({"model": name, "mae": mean_absolute_error(test[target], predicted),
                        "rmse": float(np.sqrt(mean_squared_error(test[target], predicted))),
                        "r2": r2_score(test[target], predicted)})
    scores = pd.DataFrame(records).sort_values("rmse")
    importance = permutation_importance(models["random_forest"], test[features], test[target],
                                        scoring="neg_mean_absolute_error", n_repeats=10,
                                        random_state=seed, n_jobs=2)
    importances = pd.DataFrame({"feature": features, "mae_increase": importance.importances_mean,
                               "std": importance.importances_std}).sort_values("mae_increase", ascending=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output / "cleaned_data.csv", index=False)
    cleaned[features + [target]].describe().to_csv(output / "summary_statistics.csv")
    correlations = cleaned[features + [target]].corr()
    correlations.to_csv(output / "correlations.csv")
    scores.to_csv(output / "model_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    importances.to_csv(output / "feature_importance.csv", index=False)
    metadata = {"quality": quality, "seed": seed, "synthetic_data": synthetic,
                "split": split, "train_rows": len(train), "test_rows": len(test), "target": target}
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlations)), correlations.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(correlations)), correlations.columns)
    fig.colorbar(matrix, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output / "correlations.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(cleaned[target], bins=25)
    axes[0].set(xlabel=target, ylabel="Observations", title="Target distribution")
    for name in ("ridge", "random_forest"):
        axes[1].scatter(predictions.actual, predictions[name], s=12, alpha=.6, label=name)
    limits = [predictions.actual.min(), predictions.actual.max()]
    axes[1].plot(limits, limits, "k--")
    axes[1].set(xlabel="Observed target", ylabel="Predicted target", title="Held-out predictions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
    lines = ["# Environmental data analysis", "",
             "**Synthetic demonstration data: these results are not environmental findings.**" if synthetic
             else "Results describe the supplied observations; verify sampling and measurement quality.",
             "", f"Target: `{target}`. Retained {len(cleaned)} of {len(frame)} rows.",
             f"{split}; {len(train)} training rows and {len(test)} test rows.",
             "Missing numeric predictors are imputed using training-set means only.", "",
             "## Held-out model performance", "", "| Model | MAE | RMSE | R² |", "|---|---:|---:|---:|"]
    for row in scores.itertuples():
        lines.append(f"| {row.model} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.3f} |")
    lines += ["", "Lower MAE/RMSE is better. Negative R² indicates performance below a test-mean reference.",
              "The mean baseline uses the training target mean. The held-out set evaluates fixed models;",
              "ranking these scores is exploratory and needs independent confirmation before deployment.",
              "", "## Associations and limitations", "",
              "See correlations.csv for pairwise Pearson associations and feature_importance.csv for",
              "random-forest permutation importance (increase in held-out MAE, with repeat standard deviation).",
              "Correlations and predictive importance do not establish causation. Correlated predictors can",
              "share importance. Outliers are retained; only exact duplicates and invalid targets are removed.",
              "Non-numeric columns are excluded. Numeric identifiers and target-derived columns must be",
              "removed before analysis. Spatial or repeated-site observations require grouped validation;",
              "use a date split for time-dependent data. No uncertainty intervals or causal effects are estimated."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata
