# Reproducibility Review

## Folder Structure Recommendation

The repository should keep source code, raw-data instructions, and generated outputs separate:

```text
data/README.md
data/raw/                 # ignored; user-provided NHTS files
data/processed/           # generated analysis data
src/                      # analysis scripts
outputs/tables/           # generated CSV tables
outputs/figures/          # generated figures
outputs/logs/             # run logs and variable reports
figures/                  # article-ready copied PNGs
docs/                     # reproducibility and review notes
```

## Script Review

- Hardcoded local paths: no user-specific absolute paths were found in analysis scripts. Paths are built relative to the repository root in `src/utils.py`.
- Random seeds: random forest and train/test split use `SEED = 20260602` from `src/utils.py`.
- Files not in repository: raw NHTS files are required but intentionally excluded. Users must download and place them under `data/raw/2017/`.
- Magic numbers: the main outcome uses `total_access_egress_time_min > 10`; sensitivity analysis also uses `> 10` and `> 15`. These thresholds are now documented in generated outcome metadata. They remain researcher-defined operational thresholds, not externally validated cutoffs.
- Missing data: all public-transit rows are retained; numeric model predictors are median-imputed inside model code; categorical predictors are encoded with explicit missing levels. This is documented in `outputs/tables/missing_data_handling.csv`.
- Survey weights: `WTTRDFIN` is retained but not applied in the default models. This is a limitation unless weighted estimation is added.
- Outcome circularity: `multimodal_transit_trip` partly includes an access-egress time threshold. Models that include access/egress time as predictors should be interpreted as classification/validation of the operational outcome, not independent evidence that time burden predicts a separate behavioral construct.
- Generated files: `outputs.zip`, `__pycache__`, `.venv`, raw CSVs, and large spreadsheet artifacts should not be committed.

## License

Repository materials are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Raw NHTS data are not redistributed and remain under NHTS/FHWA public-use terms.
