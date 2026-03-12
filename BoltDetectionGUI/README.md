# BoltDetectionGUI (Auxiliary Validation Tool)

`BoltDetectionGUI` is a Windows desktop assistant for PIR-Net validation workflows.  
It is designed for configuration management and Python-bridge inference orchestration, not as a standalone DAQ runtime.

## 1. Contents

1. `src/`: C# WinForms source code (.NET 10).
2. `release/BoltDetection_setup.exe`: installer package.
3. `release/SHA256SUMS.txt`: checksum file for installer verification.

## 2. End-User Installation

1. Navigate to `release/`.
2. Verify installer integrity:
   - PowerShell: `Get-FileHash .\\BoltDetection_setup.exe -Algorithm SHA256`
   - Compare with `SHA256SUMS.txt`.
3. Run `BoltDetection_setup.exe`.

## 3. Build from Source

Requirements:

1. Windows 10/11
2. .NET SDK 10.0

Commands:

```powershell
cd BoltDetectionGUI/src
dotnet restore
dotnet build -c Release
```

Run:

```powershell
dotnet run
```

## 4. Runtime Bridge Requirements

The selected project path must contain:

1. `config.json`
2. `inference_engine.py`
3. Model weights under `checkpoints/...`
