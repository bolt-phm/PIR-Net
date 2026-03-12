# External Baseline Suite (`301`-`307`)

This directory contains self-contained external baseline implementations used for standardized comparison against PIR-Net.

## 1. Baseline Inventory

1. `301`: WDCNN (1D waveform CNN)
2. `302`: InceptionTime-like (1D waveform)
3. `303`: ResNet1D (1D waveform)
4. `304`: ResNet-18 (RGB spectrogram CNN)
5. `305`: EfficientNet-B0 (RGB spectrogram CNN)
6. `306`: Signal Transformer (1D sequence model)
7. `307`: CNN-BiLSTM-Attention (1D sequence model)

## 2. Protocol Constraint

All baselines are executed under a non-PIR preprocessing protocol:

1. No physics-informed `smart_resample`.
2. No 5-channel PIR pseudo-image representation.
3. No PIR-specific fusion modules.
4. No PIR-specific custom combined loss by default.

Supported baseline preprocessing options:

1. Waveform mode: `linear`, `decimate`, `identity_crop`
2. Image mode: `none`, `spectrogram_rgb`

## 3. Execution

Run all baselines:

```bash
python run_baselines.py --experiments 301 302 303 304 305 306 307
python run_baselines.py --experiments 301 302 303 304 305 306 307 --with_generalization
```

Run a single experiment:

```bash
cd 307
python train.py --exp_dir .
python generalization.py --exp_dir .
```

## 4. Multi-Server Split

Use scripts in `scripts/`:

1. `serverA.sh`: `301 302`
2. `serverB.sh`: `303 306`
3. `serverC.sh`: `304`
4. `serverD.sh`: `307`

Collect outputs:

```bash
python scripts/merge_results.py
```

## 5. Related Notes

For impact/percussion literature taxonomy used in documentation, see:

- `RELATED_WORK_IMPACT_DETECTION.md`
