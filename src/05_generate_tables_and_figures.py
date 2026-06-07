"""Generate analysis notes from saved analysis outputs."""

from __future__ import annotations

import math
import shutil

import pandas as pd

from utils import FIGURE_DIR, NOTES_DIR, PROCESSED_DIR, TABLE_DIR, write_text


DATASET_ACCESS_DATE = "June 2, 2026"
DATASET_CITATION = (
    "Federal Highway Administration. (2017). 2017 National Household Travel Survey, "
    "U.S. Department of Transportation, Washington, DC. Available online: https://nhts.ornl.gov."
)


def read_table(name: str) -> pd.DataFrame | None:
    path = TABLE_DIR / name
    return pd.read_csv(path) if path.exists() else None


def fmt(value: object, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "not available"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_metadata_tables(df: pd.DataFrame) -> None:
    outcome_rows = [
        {
            "variable": "nonwalking_access_or_egress",
            "definition": "1 if any selected TRACC_* or TREGR_* access/egress mode is non-walking; 0 otherwise.",
            "interpretation_note": "Non-circular with respect to access_time_min and egress_time_min.",
        },
        {
            "variable": "multimodal_transit_trip",
            "definition": "1 if nonwalking_access_or_egress == 1 OR total_access_egress_time_min > 10; 0 otherwise.",
            "interpretation_note": "Partly definitional when access_time_min or egress_time_min are included as predictors.",
        },
    ]
    pd.DataFrame(outcome_rows).to_csv(TABLE_DIR / "outcome_definitions.csv", index=False)

    missing_rows = [
        {"item": "final analysis rows", "n": len(df), "handling": "All public-transit trip rows retained."},
        {
            "item": "usable access/egress variables",
            "n": int(
                (
                    df["access_mode_category"].ne("missing")
                    | df["egress_mode_category"].ne("missing")
                    | df["access_time_min"].notna()
                    | df["egress_time_min"].notna()
                ).sum()
            ),
            "handling": "Used in descriptive availability counts; rows without usable access/egress remain in the dataset.",
        },
        {
            "item": "access_time_min nonmissing",
            "n": int(df["access_time_min"].notna().sum()),
            "handling": "Missing numeric predictors are median-imputed inside Model A and Model B.",
        },
        {
            "item": "egress_time_min nonmissing",
            "n": int(df["egress_time_min"].notna().sum()),
            "handling": "Missing numeric predictors are median-imputed inside Model A and Model B.",
        },
        {
            "item": "total_access_egress_time_min nonmissing",
            "n": int(df["total_access_egress_time_min"].notna().sum()),
            "handling": "Computed as access + egress when either component exists; set missing when both components are missing.",
        },
        {
            "item": "categorical predictors",
            "n": len(df),
            "handling": "Missing categorical predictors are filled as an explicit 'missing' level before encoding.",
        },
    ]
    pd.DataFrame(missing_rows).to_csv(TABLE_DIR / "missing_data_handling.csv", index=False)

    test_n = math.ceil(len(df) * 0.25)
    train_n = len(df) - test_n
    model_rows = [
        {
            "model": "Model A logistic regression",
            "outcome": "multimodal_transit_trip",
            "n": len(df),
            "settings": "statsmodels Logit; numeric predictors median-imputed; categorical predictors encoded with one reference level; survey weights not applied.",
        },
        {
            "model": "Model B random forest",
            "outcome": "multimodal_transit_trip",
            "n": len(df),
            "settings": f"75/25 stratified train/test split; train n approximately {train_n}; test n approximately {test_n}; 400 trees; min_samples_leaf=5; class_weight='balanced'; random seed fixed.",
        },
    ]
    pd.DataFrame(model_rows).to_csv(TABLE_DIR / "model_metadata.csv", index=False)

    dataset_rows = [
        {
            "dataset": "2017 National Household Travel Survey public-use survey data",
            "release_year": 2017,
            "key_files": "trippub.csv, perpub.csv, hhpub.csv, vehpub.csv",
            "access_date": DATASET_ACCESS_DATE,
            "citation": DATASET_CITATION,
        }
    ]
    pd.DataFrame(dataset_rows).to_csv(TABLE_DIR / "dataset_metadata.csv", index=False)


def mirror_figures() -> None:
    target_dir = FIGURE_DIR.parents[1] / "figures"
    target_dir.mkdir(exist_ok=True)
    for source in FIGURE_DIR.glob("*.png"):
        shutil.copy2(source, target_dir / source.name)


def main() -> None:
    if not (PROCESSED_DIR / "nhts_transit_last_mile_analysis.csv").exists():
        raise FileNotFoundError("Processed dataset missing. Run earlier workflow steps first.")
    df = pd.read_csv(PROCESSED_DIR / "nhts_transit_last_mile_analysis.csv")
    write_metadata_tables(df)
    mirror_figures()
    access = read_table("descriptive_access_modes.csv")
    egress = read_table("descriptive_egress_modes.csv")
    time_summary = read_table("time_summary.csv")
    by_purpose = read_table("by_trip_purpose.csv")
    logit = read_table("logit_multimodal_results.csv")

    final_n = len(df)
    multimodal_share = df["multimodal_transit_trip"].mean() if "multimodal_transit_trip" in df else float("nan")
    top_access = "not available" if access is None or access.empty else str(access.iloc[0]["access_mode"])
    top_egress = "not available" if egress is None or egress.empty else str(egress.iloc[0]["egress_mode"])
    total_mean = "not available"
    if time_summary is not None and "total_access_egress_time_min" in set(time_summary["variable"]):
        total_mean = fmt(time_summary.loc[time_summary["variable"] == "total_access_egress_time_min", "mean"].iloc[0])

    model_sentence = "The logistic regression output was not available or was skipped because the modeling sample was insufficient."
    if logit is not None and "term" in logit.columns:
        terms = logit[logit["term"].isin(["access_time_min", "egress_time_min"])]
        if not terms.empty:
            bits = [
                f"{row.term}: OR={fmt(row.odds_ratio)}, p={fmt(row.p_value, 3)}"
                for row in terms.itertuples(index=False)
            ]
            model_sentence = "In Model A, first/last-mile time variables were estimated as follows: " + "; ".join(bits) + "."

    purpose_sentence = "Trip-purpose comparisons were not available."
    if by_purpose is not None and not by_purpose.empty:
        highest = by_purpose.sort_values("mean_total_time", ascending=False).iloc[0]
        purpose_sentence = (
            f"The highest mean total access-egress time was observed for {highest['trip_purpose_group']} trips "
            f"({fmt(highest['mean_total_time'])} minutes)."
        )

    abstract = (
        "This study examines whether first-mile and last-mile burden is associated with operationally defined multimodal or burdened public transit "
        "journeys using public-use records from the National Household Travel Survey. The analysis focuses on trips "
        "where public transit was used, because access and egress measures are observed only for transit trips. "
        "Access and egress modes are summarized, access and egress times are compared across trip purposes, and "
        "logistic regression and random forest models are used to classify whether a transit trip involved non-walking "
        "access or egress or elevated access-egress time. Because the main outcome includes a time-threshold component, "
        "model estimates for access and egress time are interpreted as validation of the operational definition rather "
        "than independent evidence that time burden predicts a separate behavioral outcome. Results should be interpreted "
        "as descriptive and associational rather than causal."
    )

    notes = f"""# Results Summary

## Dataset Description

The analysis uses public-use National Household Travel Survey trip records, merged with person and household files when available. The final analytic file is restricted to public transit trips.

Final sample size: {final_n:,} transit trips.

Dataset release: 2017 NHTS public-use survey data. Access date: {DATASET_ACCESS_DATE}.

Recommended citation: {DATASET_CITATION}

## Outcome Definition

The primary generated outcome, `multimodal_transit_trip`, is defined as:

`nonwalking_access_or_egress == 1 OR total_access_egress_time_min > 10`.

`nonwalking_access_or_egress` equals 1 when any selected access or egress mode flag indicates a non-walking access/egress mode. `total_access_egress_time_min` equals `access_time_min + egress_time_min` when either component is available and is missing when both components are missing.

Important caveat: because the primary outcome includes `total_access_egress_time_min > 10`, models that use `access_time_min` and `egress_time_min` as predictors are partly definitional. The high pseudo-R2, high random-forest AUC, and large time-threshold sensitivity odds ratios should therefore be described as internal classification/validation of the operational outcome, not as independent evidence that time burden predicts an unrelated behavioral outcome.

## Key Descriptive Findings

- Most common access category: {top_access}.
- Most common egress category: {top_egress}.
- Mean total access-egress time: {total_mean} minutes.
- Multimodal transit trip share: {fmt(multimodal_share)}.
- {purpose_sentence}

## Key Model Findings

{model_sentence}

The random forest uses a 75/25 stratified train/test split, 400 trees, `min_samples_leaf=5`, `class_weight='balanced'`, and a fixed random seed. Its test set contains approximately 2,773 trips.

Model status: current `logit_multimodal_results.csv` and `random_forest_metrics.csv` are authoritative. Warning files are generated only if a model is skipped.

## Missing-Data Handling

All {final_n:,} public-transit trip rows are retained in the processed analysis file. Access/egress availability is lower than the row count because some rows have missing or not-ascertained access/egress measures. Numeric model predictors, including access and egress times, are median-imputed inside the model pipeline. Categorical predictors are filled as an explicit `missing` level before encoding. Access and egress mode summaries retain `missing` as an explicit category.

## Interpretation

The findings should be described cautiously. The main outcome combines non-walking access/egress with an access-egress time threshold, so the strongest time-variable model results should be framed as showing that the operational definition is recoverable from its time components. The non-walking access/egress sensitivity analysis is the cleaner, less circular specification for claims about mode-based multimodality. The design supports statements about association, classification, and descriptive burden patterns, not causal effects.

## Limitations

- Access and egress variables are available only for public transit trips, so they are not used to predict transit use among all trips.
- The primary generated outcome partly includes access-egress time by definition; models including access and egress times should not be interpreted as independent prediction of a separate behavioral outcome.
- Public-use variables may be categorical, top-coded, or differently named across NHTS releases.
- `WTTRDFIN` is retained in the processed data, but the default models do not apply survey weights or full complex-survey variance estimation.
- Missing data and refused or unknown response codes may affect estimates; model runs retain all rows using median imputation for numeric predictors and explicit missing levels for categorical predictors.

## Possible Paper Titles

1. Does the Last Mile Matter? Predicting Multimodal Public Transit Trips with the NHTS
2. First- and Last-Mile Burden in Public Transit Trips
3. Door-to-Door Transit: Access, Egress, and Multimodal Travel in the NHTS
4. Predicting Multimodal Public Transport Journeys from Access and Egress Measures
5. Beyond the Transit Ride: First/Last-Mile Predictors in Household Travel Survey Data

## Draft Abstract

{abstract}

## Suggested Figure and Table List

- Table 1. Sample size flow.
- Table 2. Distribution of access and egress modes.
- Table 3. Access, egress, and total burden by trip purpose.
- Table 4. Logistic regression estimates for multimodal transit trips.
- Figure 1. Access mode distribution.
- Figure 2. Egress mode distribution.
- Figure 3. Access-egress time by trip purpose.
- Figure 4. Multimodal share by trip purpose.
- Figure 5. Random forest feature importance.

## Data and Code Availability

The input data are public-use National Household Travel Survey files available from the NHTS website. The local analysis code and generated outputs are contained in this project directory. Raw NHTS files are not redistributed by this project beyond the user's local copy.

## Reference Verification

No bibliography file is included in this analysis repository. Reference verification should be completed in the article files before submission.
"""
    write_text(NOTES_DIR / "results_summary.md", notes)
    checklist = f"""# Submission Checklist Notes

- Outcome definition: `multimodal_transit_trip = nonwalking_access_or_egress OR total_access_egress_time_min > 10`. This is now explicitly documented as partly definitional when time variables are predictors.
- Model outputs: current populated model CSVs are authoritative. Warning files are generated only if a model is skipped.
- Missing data: all {final_n:,} rows are retained; numeric model predictors are median-imputed; categorical predictors use explicit missing levels; mode summaries keep `missing`.
- Dataset: 2017 NHTS public-use survey data; access date {DATASET_ACCESS_DATE}; `WTTRDFIN` retained but not applied in default models.
- Random forest: 25% test set, approximately 2,773 trips; 400 trees; `class_weight='balanced'`; fixed seed.
- Figures: PNG files are in both `outputs/figures/` and root `figures/`.
- Reference DOI verification should be completed in the article files before submission.
"""
    write_text(NOTES_DIR / "submission_checklist_notes.md", checklist)
    print("Saved analysis notes.")


if __name__ == "__main__":
    main()
