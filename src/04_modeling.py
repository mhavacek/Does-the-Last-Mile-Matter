"""Fit statistical and predictive models for multimodal transit trips."""

from __future__ import annotations

import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import FIGURE_DIR, PROCESSED_DIR, SEED, TABLE_DIR, write_text, LOG_DIR


DATASET = PROCESSED_DIR / "nhts_transit_last_mile_analysis.csv"
NUMERIC = ["access_time_min", "egress_time_min", "trip_distance_miles", "trip_duration_min", "commute_trip", "age", "vehicles_available"]
CATEGORICAL = ["sex", "household_income_cat", "urban_or_msa", "trip_purpose_group"]
LOGIT_CATEGORICAL = ["sex", "household_income_cat", "urban_or_msa"]


def load_model_data() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError("Processed dataset missing. Run src/02_prepare_analysis_dataset.py first.")
    return pd.read_csv(DATASET)


def fit_logit(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, dict[str, float]]:
    predictors = [c for c in NUMERIC + LOGIT_CATEGORICAL if c in df.columns]
    model_df = df[[outcome] + predictors].copy()
    model_df = model_df.dropna(subset=[outcome])
    if model_df[outcome].nunique() < 2 or len(model_df) < 20:
        raise ValueError(f"Outcome {outcome} needs at least two classes and 20 observations.")
    for col in NUMERIC:
        if col in model_df:
            model_df[col] = model_df[col].fillna(model_df[col].median())
    for col in LOGIT_CATEGORICAL:
        if col in model_df:
            model_df[col] = model_df[col].fillna("missing").astype(str)
    X = pd.get_dummies(model_df[predictors], columns=[c for c in LOGIT_CATEGORICAL if c in predictors], drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add")
    y = model_df[outcome].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.Logit(y, X).fit(disp=False, maxiter=200)
    params = result.params
    conf = result.conf_int()
    out = pd.DataFrame(
        {
            "term": params.index,
            "coefficient": params.values,
            "odds_ratio": np.exp(params.values),
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": result.pvalues.values,
        }
    )
    info = {"n_observations": float(result.nobs), "pseudo_r2": float(result.prsquared)}
    return out, info


def model_a(df: pd.DataFrame) -> None:
    warning_path = LOG_DIR / "model_a_warning.txt"
    try:
        results, info = fit_logit(df, "multimodal_transit_trip")
        results["n_observations"] = info["n_observations"]
        results["pseudo_r2"] = info["pseudo_r2"]
        results.to_csv(TABLE_DIR / "logit_multimodal_results.csv", index=False)
        warning_path.unlink(missing_ok=True)
    except Exception as exc:
        write_text(warning_path, f"Model A skipped: {exc}\n")
        pd.DataFrame([{"status": "skipped", "reason": str(exc)}]).to_csv(TABLE_DIR / "logit_multimodal_results.csv", index=False)


def model_b(df: pd.DataFrame) -> None:
    warning_path = LOG_DIR / "model_b_warning.txt"
    predictors = [c for c in NUMERIC + CATEGORICAL if c in df.columns]
    model_df = df[["multimodal_transit_trip"] + predictors].dropna(subset=["multimodal_transit_trip"]).copy()
    y = model_df["multimodal_transit_trip"].astype(int)
    if y.nunique() < 2 or len(model_df) < 30:
        reason = "Random forest needs at least two outcome classes and 30 observations."
        write_text(warning_path, reason + "\n")
        pd.DataFrame([{"metric": "status", "value": "skipped", "note": reason}]).to_csv(TABLE_DIR / "random_forest_metrics.csv", index=False)
        pd.DataFrame(columns=["feature", "importance"]).to_csv(TABLE_DIR / "random_forest_feature_importance.csv", index=False)
        return

    X = model_df[predictors]
    numeric = [c for c in NUMERIC if c in X.columns]
    categorical = [c for c in CATEGORICAL if c in X.columns]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, class_weight="balanced")
    pipe = Pipeline([("preprocess", pre), ("model", rf)])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=stratify)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1] if len(pipe.classes_) == 2 else None
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob) if prob is not None and y_test.nunique() == 2 else np.nan,
    }
    cm = confusion_matrix(y_test, pred)
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    rows.append({"metric": "confusion_matrix", "value": cm.tolist()})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "random_forest_metrics.csv", index=False)

    names = pipe.named_steps["preprocess"].get_feature_names_out()
    imps = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": names, "importance": imps}).sort_values("importance", ascending=False)
    fi.to_csv(TABLE_DIR / "random_forest_feature_importance.csv", index=False)
    top = fi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close(fig)
    warning_path.unlink(missing_ok=True)


def model_c(df: pd.DataFrame) -> None:
    definitions = {
        "any_nonwalking_access_or_egress": df["nonwalking_access_or_egress"].astype(int),
        "total_access_egress_time_gt_10": (df["total_access_egress_time_min"] > 10).astype(int),
        "total_access_egress_time_gt_15": (df["total_access_egress_time_min"] > 15).astype(int),
    }
    rows = []
    for name, outcome_values in definitions.items():
        temp = df.copy()
        temp["sensitivity_outcome"] = outcome_values
        try:
            result, info = fit_logit(temp, "sensitivity_outcome")
            key_terms = result[result["term"].isin(["access_time_min", "egress_time_min", "total_access_egress_time_min"])]
            if key_terms.empty:
                key_terms = result.head(0)
            for _, row in key_terms.iterrows():
                rows.append({"definition": name, **row.to_dict(), **info})
            if key_terms.empty:
                rows.append({"definition": name, "status": "fit", **info})
        except Exception as exc:
            rows.append({"definition": name, "status": "skipped", "reason": str(exc)})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "sensitivity_results.csv", index=False)


def main() -> None:
    df = load_model_data()
    model_a(df)
    model_b(df)
    model_c(df)
    print("Saved model outputs.")


if __name__ == "__main__":
    main()
