# External Baseline Suite

This folder contains self-contained baseline experiments under `experiments/external_baselines/`.

## 1. Baseline Coverage

Implemented experiment groups:

- `301`: WDCNN 1D waveform baseline
- `302`: InceptionTime-like 1D waveform baseline
- `303`: ResNet1D waveform baseline
- `304`: ResNet-18 spectrogram CNN baseline
- `305`: EfficientNet-B0 spectrogram baseline
- `306`: Signal Transformer baseline

## 2. Fairness Constraint

Baselines are configured without PIR-specific preprocessing:

- no `smart_resample`,
- no 5-channel PIR pseudo-image construction,
- no PIR-specific fusion module,
- no PIR-only combined loss by default.

Available baseline preprocessing modes:

- waveform: `linear` / `decimate` / `identity_crop`
- image: `none` / `spectrogram_rgb`

## 3. Run

All baselines:

```bash
python run_baselines.py --experiments 301 302 303 304 305 306
python run_baselines.py --experiments 301 302 303 304 305 306 --with_generalization
```

Single baseline example:

```bash
cd 304
python train.py --exp_dir .
python generalization.py --exp_dir .
```

## 4. Multi-Server Split (4 machines)

Use scripts in `scripts/`:

- `serverA.sh`: `301 302`
- `serverB.sh`: `303 306`
- `serverC.sh`: `304`
- `serverD.sh`: `305`

Merge outputs:

```bash
python scripts/merge_results.py
```