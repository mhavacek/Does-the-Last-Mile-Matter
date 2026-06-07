"""Shared utilities for the NHTS first/last-mile analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"
NOTES_DIR = ROOT / "outputs" / "analysis_notes"
SEED = 20260602


for directory in [RAW_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, NOTES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DiscoveryResult:
    name: str
    column: str | None
    required: bool
    patterns: list[str]
    note: str = ""


def find_file(stem: str | list[str]) -> Path | None:
    stems = [stem] if isinstance(stem, str) else stem
    normalized = [item.lower() for item in stems]
    paths = sorted(RAW_DIR.rglob("*.csv"))
    for wanted in normalized:
        for path in paths:
            if path.stem.lower() == wanted:
                return path
    return None


def read_csv_upper(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [str(col).upper().strip() for col in df.columns]
    return df


def load_raw_data() -> dict[str, pd.DataFrame | None]:
    return {
        "trip": read_csv_upper(find_file(["trippub", "trip", "trips", "tripv2pub"])),
        "person": read_csv_upper(find_file(["perpub", "person", "persons", "perv2pub"])),
        "household": read_csv_upper(find_file(["hhpub", "household", "households", "hhv2pub"])),
        "vehicle": read_csv_upper(find_file(["vehpub", "vehicle", "vehicles", "vehv2pub"])),
    }


def discover_column(columns: Iterable[str], patterns: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    cols = list(columns)
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for col in cols:
            if regex.search(col) and not any(re.search(ex, col, re.IGNORECASE) for ex in exclude):
                return col
    return None


def availability_report(data: dict[str, pd.DataFrame | None]) -> tuple[dict[str, str | None], str]:
    trip_cols = [] if data.get("trip") is None else list(data["trip"].columns)
    person_cols = [] if data.get("person") is None else list(data["person"].columns)
    hh_cols = [] if data.get("household") is None else list(data["household"].columns)
    all_cols = trip_cols + person_cols + hh_cols

    specs = [
        DiscoveryResult("HOUSEID", discover_column(all_cols, [r"^HOUSEID$"]), True, [r"^HOUSEID$"]),
        DiscoveryResult("PERSONID", discover_column(all_cols, [r"^PERSONID$"]), False, [r"^PERSONID$"]),
        DiscoveryResult("TRPTRANS", discover_column(trip_cols, [r"^TRPTRANS$", r"TRP.*MODE", r"TRANS"]), False, [r"^TRPTRANS$"]),
        DiscoveryResult("PUBTRANS", discover_column(trip_cols, [r"^PUBTRANS$", r"PUBLIC.*TRANS"]), False, [r"^PUBTRANS$"]),
        DiscoveryResult("TRIPPURP", discover_column(trip_cols, [r"^TRIPPURP$", r"^WHYTRP1S$", r"^WHYTRP90$", r"WHYTO", r"PURP"]), False, [r"^TRIPPURP$", r"^WHYTRP1S$"]),
        DiscoveryResult("TRPMILES", discover_column(trip_cols, [r"^TRPMILES$", r"MILES", r"DIST"]), False, [r"^TRPMILES$"]),
        DiscoveryResult("TRVLCMIN", discover_column(trip_cols, [r"^TRVLCMIN$", r"TRVL.*MIN", r"DURATION"]), False, [r"^TRVLCMIN$"]),
        DiscoveryResult("ACCESS_TIME", discover_column(trip_cols, [r"^TRACCTM$", r"ACC.*TIME", r"ACCESS.*MIN"]), False, [r"^TRACCTM$"]),
        DiscoveryResult("EGRESS_TIME", discover_column(trip_cols, [r"^TREGRTM$", r"EGR.*TIME", r"EGRESS.*MIN"]), False, [r"^TREGRTM$"]),
        DiscoveryResult("ACCESS_MODE", discover_column(trip_cols, [r"^TRACC_", r"ACCESS.*MODE", r"ACC.*MODE"]), False, [r"^TRACC_"]),
        DiscoveryResult("EGRESS_MODE", discover_column(trip_cols, [r"^TREGR_", r"EGRESS.*MODE", r"EGR.*MODE"]), False, [r"^TREGR_"]),
        DiscoveryResult("WTTRDFIN", discover_column(trip_cols, [r"^WTTRDFIN$", r"TRIP.*WGT", r"TRIP.*WEIGHT"]), False, [r"^WTTRDFIN$"]),
        DiscoveryResult("AGE", discover_column(person_cols + hh_cols, [r"^R_AGE$", r"^AGE$", r"AGE"]), False, [r"AGE"]),
        DiscoveryResult("SEX", discover_column(person_cols + hh_cols, [r"^R_SEX$", r"^SEX$", r"GENDER"]), False, [r"SEX"]),
        DiscoveryResult("VEHICLES", discover_column(hh_cols + person_cols, [r"^HHVEHCNT$", r"VEH.*CNT", r"VEHICLE"]), False, [r"VEH"]),
        DiscoveryResult("INCOME", discover_column(hh_cols + person_cols, [r"^HHFAMINC$", r"INCOME", r"INC"]), False, [r"INCOME"]),
        DiscoveryResult("URBAN_OR_MSA", discover_column(hh_cols + person_cols, [r"^URBAN$", r"^MSASIZE$", r"^HBHUR$", r"MSA", r"URB"]), False, [r"URBAN", r"MSA"]),
    ]

    found = {spec.name: spec.column for spec in specs}
    lines = ["Variable availability report", "============================", ""]
    for key, df in data.items():
        lines.append(f"{key}: {'loaded, ' + str(len(df)) + ' rows' if df is not None else 'missing'}")
    lines.append("")
    for spec in specs:
        status = "FOUND" if spec.column else ("MISSING REQUIRED" if spec.required else "missing optional")
        lines.append(f"{spec.name}: {status}; column={spec.column}; patterns={', '.join(spec.patterns)}")
    return found, "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.mask(values < 0)
    values = values.mask(values.isin([-1, -7, -8, -9, 97, 98, 99, 997, 998, 999]))
    return values


def cap_upper_outlier(series: pd.Series, quantile: float = 0.995) -> pd.Series:
    values = series.copy()
    if values.notna().sum() < 10:
        return values
    cap = values.quantile(quantile)
    return values.mask(values > cap)


def mode_bucket(value: object, column_name: str = "") -> str:
    text = f"{column_name} {value}".upper()
    if pd.isna(value):
        return "missing"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    flag_like = bool(re.search(r"^(TRACC_|TREGR_)", column_name, re.I)) or bool(
        re.search(r"(ACCESS|EGRESS).*(WLK|WALK|BUS|RAIL|CRL|SUB|POV|CAR|BIKE|OTH)", column_name, re.I)
    )
    if flag_like:
        if numeric != 1 and not re.search(r"^(YES|Y|TRUE)$", str(value).strip(), re.I):
            return "missing"
        col_text = column_name.upper()
        if any(token in col_text for token in ["WALK", "FOOT", "WLK"]):
            return "walk"
        if any(token in col_text for token in ["BIKE", "BICYCLE"]):
            return "bike_active"
        if any(token in col_text for token in ["POV", "CAR", "AUTO", "DRIVE", "VAN", "TAXI", "TNC"]):
            return "pov"
        if "BUS" in col_text:
            return "bus"
        if any(token in col_text for token in ["RAIL", "TRAIN", "COMMUTER", "AMTRAK", "CRL"]):
            return "rail"
        if any(token in col_text for token in ["SUBWAY", "ELEVATED", "METRO", "SUB"]):
            return "subway"
        return "other"
    if not pd.isna(numeric):
        if numeric == 1:
            return "walk"
        if numeric == 2:
            return "bike_active"
        if numeric in [3, 4, 5, 6, 7, 8, 9, 14, 15]:
            return "pov"
        if numeric in [8, 10]:
            return "bus"
        if numeric in [12, 13, 17]:
            return "rail"
        if numeric == 11:
            return "subway"
    affirmative_flag = numeric == 1 and bool(re.search(r"(WLK|WALK|BUS|RAIL|SUB|POV|CAR|BIKE|EGR|ACC)", column_name, re.I))
    if not affirmative_flag and re.search(r"^(0|2|NO|N)$", str(value).strip(), re.I):
        return "missing"
    if any(token in text for token in ["WALK", "FOOT", "WLK"]):
        return "walk"
    if any(token in text for token in ["BIKE", "BICYCLE"]):
        return "bike_active"
    if any(token in text for token in ["POV", "CAR", "AUTO", "DRIVE", "VAN", "TAXI", "TNC", "UBER", "LYFT"]):
        return "pov"
    if "BUS" in text:
        return "bus"
    if any(token in text for token in ["RAIL", "TRAIN", "COMMUTER", "AMTRAK"]):
        return "rail"
    if any(token in text for token in ["SUBWAY", "ELEVATED", "METRO"]):
        return "subway"
    return "other"


def purpose_bucket(value: object) -> str:
    text = str(value).upper()
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if any(token in text for token in ["WORK", "COMMUTE", "HBW"]) or numeric in [10, 11, 12]:
        return "commuting/work"
    if any(token in text for token in ["SCHOOL", "HBSCH", "SCH"]) or numeric in [20, 21, 22]:
        return "school"
    if any(token in text for token in ["SHOP", "ERRAND", "BUY", "MEDICAL", "HBSHOP"]) or numeric in [30, 40, 41, 42]:
        return "shopping/errands"
    if any(token in text for token in ["SOCIAL", "LEISURE", "RECREATION", "VISIT", "SOCREC", "HBSOCREC"]) or numeric in [50, 70, 80, 81, 82]:
        return "leisure/social"
    return "other"


def is_transit_mode(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).upper()
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    transit_numeric = {8, 10, 11, 12, 13, 17}
    return (
        numeric in transit_numeric
        or any(token in text for token in ["BUS", "SUBWAY", "RAIL", "TRANSIT", "STREETCAR", "TROLLEY", "FERRY"])
    )


def save_column_log(df: pd.DataFrame | None, name: str) -> None:
    path = LOG_DIR / f"columns_{name}.txt"
    if df is None:
        write_text(path, f"{name} file missing\n")
        return
    rows = [f"{col}\t{df[col].dtype}\tmissing={df[col].isna().sum()}" for col in df.columns]
    write_text(path, "\n".join(rows) + "\n")
