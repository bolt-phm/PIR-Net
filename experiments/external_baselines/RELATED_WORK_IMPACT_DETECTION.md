# Impact/ Percussion Detection Taxonomy

This note defines a concise taxonomy for organizing impact/percussion-based bolt-loosening studies in project documentation.

## 1. Method Families

1. Percussion excitation with handcrafted signal features.
2. Vibro-acoustic impact response for preload or loosening estimation.
3. Deep learning on impact-derived waveform or spectrogram inputs.
4. Hybrid physics-informed and data-driven diagnostic pipelines.

## 2. Mapping Guidance

When writing technical summaries or manuscript background sections:

1. Separate sensor modality (acoustic, piezoelectric, vibration) from model architecture (CNN, Transformer, hybrid).
2. Distinguish qualitative state detection from quantitative looseness estimation.
3. Report preprocessing assumptions explicitly to avoid protocol mismatch.
4. Map baseline groups (`301`-`307`) to the taxonomy above for transparent comparison.

## 3. Reproducibility Principle

All literature alignment should remain consistent with implemented code paths and disclosed experimental protocols in this repository.
