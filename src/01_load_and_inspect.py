"""Load raw NHTS files, inspect columns, and report variable availability."""

from __future__ import annotations

from utils import LOG_DIR, availability_report, load_raw_data, save_column_log, write_text


def main() -> None:
    data = load_raw_data()
    save_column_log(data["trip"], "trip")
    save_column_log(data["person"], "person")
    save_column_log(data["household"], "household")
    save_column_log(data["vehicle"], "vehicle")

    _, report = availability_report(data)
    write_text(LOG_DIR / "variable_availability.txt", report)
    print(report)


if __name__ == "__main__":
    main()
