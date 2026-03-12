# BoltDetectionGUI Release Notes

## Package Information

1. Artifact: `BoltDetection_setup.exe`
2. Type: Windows installer
3. Verification file: `SHA256SUMS.txt`

## Functional Scope

This release supports:

1. Project configuration loading.
2. Python bridge invocation for inference workflows.
3. Assisted batch validation use cases.

The GUI is an auxiliary validation interface and does not replace a full production DAQ/runtime stack.

## Runtime Prerequisites

The configured project directory must include:

1. `config.json`
2. `inference_engine.py`
3. Trained model weights under `checkpoints/...`
