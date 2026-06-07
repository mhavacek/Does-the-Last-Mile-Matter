"""Generate descriptive tables and figures for transit first/last-mile trips."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, load_raw_data


DATASET = PROCESSED_DIR / "nhts_transit_last_mile_analysis.csv"


def require_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError("Processed dataset missing. Run src/02_prepare_analysis_dataset.py first.")
    return pd.read_csv(DATASET)


def save_bar(table: pd.DataFrame, label_col: str, value_col: str, path, title: str, ylabel: str = "Share") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table[label_col].astype(str), table[value_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = require_dataset()
    raw = load_raw_data()
    all_trips = 0 if raw["trip"] is None else len(raw["trip"])
    sample_flow = pd.DataFrame(
        [
            {"stage": "all trips", "n": all_trips},
            {"stage": "trips with public transit", "n": len(df)},
            {
                "stage": "trips with usable access/egress variables",
                "n": int(
                    (
                        df["access_mode_category"].ne("missing")
                        | df["egress_mode_category"].ne("missing")
                        | df["access_time_min"].notna()
                        | df["egress_time_min"].notna()
                    ).sum()
                ),
            },
            {"stage": "final modeling sample", "n": int(df["multimodal_transit_trip"].notna().sum())},
        ]
    )
    sample_flow.to_csv(TABLE_DIR / "sample_flow.csv", index=False)

    access = df["access_mode_category"].fillna("missing").value_counts(dropna=False).rename_axis("access_mode").reset_index(name="n")
    access["share"] = access["n"] / access["n"].sum()
    egress = df["egress_mode_category"].fillna("missing").value_counts(dropna=False).rename_axis("egress_mode").reset_index(name="n")
    egress["share"] = egress["n"] / egress["n"].sum()
    access.to_csv(TABLE_DIR / "descriptive_access_modes.csv", index=False)
    egress.to_csv(TABLE_DIR / "descriptive_egress_modes.csv", index=False)

    time_vars = ["access_time_min", "egress_time_min", "total_access_egress_time_min"]
    time_summary = df[time_vars].agg(["count", "mean", "median", "std", "min", "max"]).T.reset_index(names="variable")
    time_summary.to_csv(TABLE_DIR / "time_summary.csv", index=False)

    by_purpose = (
        df.groupby("trip_purpose_group", dropna=False)
        .agg(
            n=("multimodal_transit_trip", "size"),
            multimodal_share=("multimodal_transit_trip", "mean"),
            mean_access_time=("access_time_min", "mean"),
            median_access_time=("access_time_min", "median"),
            mean_egress_time=("egress_time_min", "mean"),
            median_egress_time=("egress_time_min", "median"),
            mean_total_time=("total_access_egress_time_min", "mean"),
        )
        .reset_index()
    )
    by_purpose.to_csv(TABLE_DIR / "by_trip_purpose.csv", index=False)

    compare = (
        df.assign(group=df["multimodal_transit_trip"].map({1: "multimodal", 0: "walk-only/low-burden"}))
        .groupby("group")
        .agg(
            n=("group", "size"),
            mean_access_time=("access_time_min", "mean"),
            mean_egress_time=("egress_time_min", "mean"),
            mean_distance=("trip_distance_miles", "mean"),
            commute_share=("commute_trip", "mean"),
        )
        .reset_index()
    )
    compare.to_csv(TABLE_DIR / "multimodal_vs_walk_only.csv", index=False)

    save_bar(access, "access_mode", "share", FIGURE_DIR / "access_mode_distribution.png", "Access Mode Distribution")
    save_bar(egress, "egress_mode", "share", FIGURE_DIR / "egress_mode_distribution.png", "Egress Mode Distribution")
    save_bar(by_purpose, "trip_purpose_group", "mean_total_time", FIGURE_DIR / "access_egress_time_by_trip_purpose.png", "Mean Access and Egress Time by Purpose", "Minutes")
    save_bar(by_purpose, "trip_purpose_group", "multimodal_share", FIGURE_DIR / "multimodal_share_by_trip_purpose.png", "Multimodal Transit Share by Purpose")
    print("Saved descriptive tables and figures.")


if __name__ == "__main__":
    main()
