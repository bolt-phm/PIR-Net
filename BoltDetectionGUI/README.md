# BoltDetectionGUI (Windows Validation Assistant)

This folder contains the optional Windows desktop validation assistant for PIR-Net. It helps users edit configs, invoke the Python validation bridge, and inspect results. It is not a standalone data-acquisition/runtime system.

## Folder Layout

- `src/`: C# WinForms source code (.NET 10, Windows)
- `release/BoltDetection_setup.exe`: installer package for end users
- `release/SHA256SUMS.txt`: checksum file for installer verification

## Quick Install (End Users)

1. Go to `release/`.
2. Verify checksum (optional but recommended):
   - PowerShell: `Get-FileHash .\BoltDetection_setup.exe -Algorithm SHA256`
   - Compare with `SHA256SUMS.txt`.
3. Run `BoltDetection_setup.exe` and complete installation.

## Build From Source (Developers)

Requirements:
- Windows 10/11
- .NET SDK 10.0

Build commands:

```powershell
cd BoltDetectionGUI/src
dotnet restore
dotnet build -c Release
```

Run directly:

```powershell
dotnet run
```

## Runtime Bridge Requirement

The GUI calls a Python bridge script named `inference_engine.py` in the selected project path.
This repository provides that script at the repository root. Make sure your GUI "Project Path"
points to a folder containing:

- `config.json`
- `inference_engine.py`
- trained weights in `checkpoints/...`
