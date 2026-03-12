# BoltDetectionGUI Release Notes

## Package
- File: `BoltDetection_setup.exe`
- Type: Windows installer

## Integrity
- Verify checksum using `SHA256SUMS.txt` before installation.

## Scope
This validation assistant focuses on:
- project configuration,
- Python bridge execution,
- assisted batch validation/inference workflow,
- optional USB mode integration workflow.

It is designed for assisted verification and demonstration, rather than replacing a full production DAQ/runtime pipeline.

## Known prerequisite
The selected project path must contain:
- `config.json`
- `inference_engine.py`
- trained model weights in `checkpoints/...`
