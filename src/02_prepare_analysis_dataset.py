"""Prepare the public-transit first/last-mile analysis dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    LOG_DIR,
    PROCESSED_DIR,
    availability_report,
    cap_upper_outlier,
    clean_numeric,
    discover_column,
    is_transit_mode,
    load_raw_data,
    mode_bucket,
    purpose_bucket,
    write_text,
)


def first_available(df: pd.DataFrame, patterns: list[str]) -> str | None:
    return discover_column(df.columns, patterns)


def merge_raw(data: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    trip = data["trip"]
    if trip is None:
        raise FileNotFoundError(
            "Missing data/raw/trippub.csv. Place the 2017 NHTS public-use trip CSV in data/raw/ and rerun."
        )
    merged = trip.copy()

    person = data.get("person")
    if person is not None and {"HOUSEID", "PERSONID"}.issubset(merged.columns) and {"HOUSEID", "PERSONID"}.issubset(person.columns):
        add_cols = [c for c in person.columns if c not in merged.columns or c in ["HOUSEID", "PERSONID"]]
        merged = merged.merge(person[add_cols], on=["HOUSEID", "PERSONID"], how="left", suffixes=("", "_PER"))

    household = data.get("household")
    if household is not None and "HOUSEID" in merged.columns and "HOUSEID" in household.columns:
        add_cols = [c for c in household.columns if c not in merged.columns or c == "HOUSEID"]
        merged = merged.merge(household[add_cols], on="HOUSEID", how="left", suffixes=("", "_HH"))
    return merged


def detect_transit(df: pd.DataFrame, found: dict[str, str | None]) -> pd.Series:
    pubtrans = found.get("PUBTRANS")
    trptrans = found.get("TRPTRANS")
    if pubtrans and pubtrans in df.columns:
        raw = df[pubtrans]
        numeric = pd.to_numeric(raw, errors="coerce")
        yes = raw.astype(str).str.upper().isin(["1", "YES", "Y", "TRUE"]) | (numeric == 1)
        if yes.sum() > 0:
            return yes
    if trptrans and trptrans in df.columns:
        return df[trptrans].map(is_transit_mode)
    raise ValueError("Could not identify public transit trips: PUBTRANS and usable TRPTRANS are missing.")


def collect_mode_category(row: pd.Series, cols: list[str]) -> str:
    buckets = []
    for col in cols:
        bucket = mode_bucket(row.get(col), col)
        if bucket != "missing":
            buckets.append(bucket)
    priority = ["pov", "bus", "rail", "subway", "bike_active", "other", "walk"]
    for item in priority:
        if item in buckets:
            return item
    return "missing"


def add_mode_indicators(df: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
    category_col = f"{prefix}_mode_category"
    df[category_col] = df.apply(lambda row: collect_mode_category(row, cols), axis=1) if cols else "missing"
    for bucket in ["walk", "bus", "rail", "subway", "pov", "bike_active", "other"]:
        df[f"{prefix}_{'bike_or_active' if bucket == 'bike_active' else bucket}"] = (df[category_col] == bucket).astype(int)
    return df


def main() -> None:
    data = load_raw_data()
    found, report = availability_report(data)
    warnings = [report]
    merged = merge_raw(data)

    merged["transit_trip"] = detect_transit(merged, found).astype(int)
    all_trips = len(merged)
    transit = merged.loc[merged["transit_trip"] == 1].copy()
    warnings.append(f"All trip records: {all_trips}")
    warnings.append(f"Public transit trip records retained: {len(transit)}")

    access_time_col = found.get("ACCESS_TIME")
    egress_time_col = found.get("EGRESS_TIME")
    transit["access_time_min"] = clean_numeric(transit[access_time_col]) if access_time_col else np.nan
    transit["egress_time_min"] = clean_numeric(transit[egress_time_col]) if egress_time_col else np.nan
    transit["access_time_min"] = cap_upper_outlier(transit["access_time_min"])
    transit["egress_time_min"] = cap_upper_outlier(transit["egress_time_min"])
    transit["total_access_egress_time_min"] = transit["access_time_min"].fillna(0) + transit["egress_time_min"].fillna(0)
    transit.loc[transit[["access_time_min", "egress_time_min"]].isna().all(axis=1), "total_access_egress_time_min"] = np.nan

    access_cols = [c for c in transit.columns if c.startswith("TRACC_") or ("ACCESS" in c and "MODE" in c)]
    egress_cols = [c for c in transit.columns if c.startswith("TREGR_") or ("EGRESS" in c and "MODE" in c)]
    transit = add_mode_indicators(transit, "access", access_cols)
    transit = add_mode_indicators(transit, "egress", egress_cols)

    transit["nonwalking_access_or_egress"] = (
        transit[["access_bus", "access_rail", "access_subway", "access_pov", "access_bike_or_active", "access_other",
                 "egress_bus", "egress_rail", "egress_subway", "egress_pov", "egress_bike_or_active", "egress_other"]]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )
    transit["multimodal_transit_trip"] = (
        (transit["nonwalking_access_or_egress"] == 1)
        | (transit["total_access_egress_time_min"] > 10)
    ).astype(int)

    purpose_col = found.get("TRIPPURP")
    transit["trip_purpose_group"] = transit[purpose_col].map(purpose_bucket) if purpose_col else "other"
    transit["commute_trip"] = (transit["trip_purpose_group"] == "commuting/work").astype(int)

    distance_col = found.get("TRPMILES")
    duration_col = found.get("TRVLCMIN")
    transit["trip_distance_miles"] = clean_numeric(transit[distance_col]) if distance_col else np.nan
    transit["trip_duration_min"] = clean_numeric(transit[duration_col]) if duration_col else np.nan
    transit["trip_distance_miles"] = cap_upper_outlier(transit["trip_distance_miles"])
    transit["trip_duration_min"] = cap_upper_outlier(transit["trip_duration_min"])

    vehicles_col = found.get("VEHICLES")
    income_col = found.get("INCOME")
    age_col = found.get("AGE")
    sex_col = found.get("SEX")
    urban_col = found.get("URBAN_OR_MSA")
    weight_col = found.get("WTTRDFIN")
    transit["vehicles_available"] = clean_numeric(transit[vehicles_col]) if vehicles_col else np.nan
    transit["household_income_cat"] = transit[income_col].astype("string") if income_col else pd.NA
    transit["age"] = clean_numeric(transit[age_col]) if age_col else np.nan
    transit["sex"] = transit[sex_col].astype("string") if sex_col else pd.NA
    transit["urban_or_msa"] = transit[urban_col].astype("string") if urban_col else pd.NA
    transit["trip_weight"] = clean_numeric(transit[weight_col]) if weight_col else np.nan

    keep = [
        "HOUSEID", "PERSONID", "transit_trip", "access_time_min", "egress_time_min",
        "total_access_egress_time_min", "access_mode_category", "egress_mode_category",
        "access_walk", "access_bus", "access_rail", "access_subway", "access_pov",
        "access_bike_or_active", "access_other", "egress_walk", "egress_bus", "egress_rail",
        "egress_subway", "egress_pov", "egress_bike_or_active", "egress_other",
        "multimodal_transit_trip", "nonwalking_access_or_egress", "commute_trip",
        "trip_purpose_group", "trip_distance_miles", "trip_duration_min", "vehicles_available",
        "household_income_cat", "age", "sex", "urban_or_msa", "trip_weight",
    ]
    keep = [c for c in keep if c in transit.columns]
    out = transit[keep].copy()
    out.to_csv(PROCESSED_DIR / "nhts_transit_last_mile_analysis.csv", index=False)

    usable = (
        out["access_mode_category"].ne("missing")
        | out["egress_mode_category"].ne("missing")
        | out["access_time_min"].notna()
        | out["egress_time_min"].notna()
    ).sum()
    warnings.append(f"Transit trips with usable access/egress variables: {usable}")
    warnings.append(f"Processed file rows: {len(out)}")
    write_text(LOG_DIR / "variable_availability.txt", "\n".join(warnings) + "\n")
    print(f"Saved {len(out):,} rows to data/processed/nhts_transit_last_mile_analysis.csv")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
