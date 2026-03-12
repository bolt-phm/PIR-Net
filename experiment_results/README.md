# Experiment Results Bundle

This folder provides a structured, open-source-ready result bundle aligned with the project experiment design.

## Scope

- PIR-Net ablation experiments: 022, 122, 202, 212, 220, 221, 222
- External baselines: 301, 302, 303, 304, 305, 306, 307
- Zenodo cross-condition benchmarks: 401, 402, 403, 404

## Included Files

1. `experiment_plan.json`: experiment taxonomy and artifact policy.
2. `result_index.csv` / `result_index.json`: machine-readable artifact inventory with status and checksums.
3. `summary.json`: high-level counts of available and expected artifacts.
4. Per-experiment folders with:
   - `config_snapshot.json`
   - `metrics_summary_template.json`
   - copied available artifacts (for example confusion matrices where present)

## Current Snapshot Summary

- Total records: 100
- Available records: 53
- Expected-but-missing records: 47

Expected-but-missing entries are placeholders for runtime outputs that are not currently tracked in this repository snapshot
(for example, full training logs and model checkpoints).

## Update Workflow

To regenerate this bundle from the latest repository contents:

```bash
python tools/build_experiment_results_bundle.py
```
