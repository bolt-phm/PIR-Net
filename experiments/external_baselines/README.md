# Baseline Suite (Self-contained in `baselines/`)

This folder is standalone and does not depend on code outside `baselines/`.

## Reviewer Mapping

- R2-4 (need SOTA comparisons):
  - `305`: EfficientNet-B0 spectrogram baseline
  - `306`: Signal Transformer baseline
- R1-2 (improvement may come from ResNet):
  - `304`: ResNet-18 spectrogram CNN baseline
  - `303`: ResNet1D waveform baseline
- R2-6 (signal encoder contribution):
  - `301`: WDCNN 1D
  - `302`: InceptionTime-like 1D
  - `303`: ResNet1D
  - `306`: Signal Transformer
- R3-1 (related work completion):
  - See `RELATED_WORK_IMPACT_DETECTION.md` for impact/percussion-related references to add in manuscript.

## Important Fairness Constraint Applied

All baseline experiments here avoid PIR-specific preprocessing:

- No `smart_resample`
- No 5-channel PIR pseudo-image construction
- No PIR-specific fusion module
- No combined label-smoothing + focal loss by default

Current preprocessing options are standard and configurable per experiment:

- waveform: `linear` / `decimate` / `identity_crop`
- image: `none` / `spectrogram_rgb`

## Run

```bash
python baselines/run_baselines.py
python baselines/run_baselines.py --with_generalization
```

You can also run one experiment directly:

```bash
cd baselines/304
python train.py --exp_dir .
python generalization.py --exp_dir .
```
## Multi-server Split (4 machines)

Use scripts in `baselines/scripts`:

- `serverA.sh`: experiments `301 302`
- `serverB.sh`: experiments `303 306`
- `serverC.sh`: experiment `304`
- `serverD.sh`: experiment `305`

Typical usage on each server:

```bash
cd /path/to/baselines
chmod +x scripts/serverA.sh scripts/serverB.sh scripts/serverC.sh scripts/serverD.sh
ROOT=$(pwd) bash scripts/serverA.sh
```

After all servers finish, copy logs back under one `baselines` folder and merge:

```bash
python scripts/merge_results.py
```
