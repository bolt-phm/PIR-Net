# Zenodo Round-2 CV Result Package (63+18)

This folder contains the merged round-2 result package for Zenodo cross-condition experiments, aggregated from servers `A, B, C, D, E, F`.

## Scope

- Clean CV train/evaluation runs: 7 experiments x 3 folds x 3 seeds = 63 runs.
  - Experiments: 401, 402, 403, 404, 501, 502, 503
- Noisy generalization runs: 2 experiments x 3 folds x 3 seeds = 18 runs.
  - Experiments: 404, 501

## Files

- `coverage_summary.json`: completeness status (expected vs found).
- `train_manifest.csv`: one row per train run (accuracy/F1 + source mapping + sha256).
- `train_summary_by_experiment.csv`: per-experiment mean/std summary over 9 runs.
- `generalization_file_manifest.csv`: one row per generalization summary file.
- `generalization_long.csv`: expanded noisy metrics (group/snr rows across folds/seeds).
- `generalization_summary_by_group_snr.csv`: aggregated noisy summary by experiment/group/snr.
- `raw/`: copied source artifacts used to generate the tables.
