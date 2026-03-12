param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,

    [string]$Title = "BoltDetectionGUI Helper Tool",
    [string]$Notes = "Windows installer for BoltDetectionGUI",
    [switch]$Draft,
    [switch]$PreRelease
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$asset = Join-Path $repoRoot 'BoltDetectionGUI\release\BoltDetection_setup.exe'
$checksum = Join-Path $repoRoot 'BoltDetectionGUI\release\SHA256SUMS.txt'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is not installed. Install it from https://cli.github.com/ and run gh auth login.'
}

if (-not (Test-Path $asset)) {
    throw "Installer not found: $asset"
}

if (-not (Test-Path $checksum)) {
    throw "Checksum file not found: $checksum"
}

$flags = @()
if ($Draft) { $flags += '--draft' }
if ($PreRelease) { $flags += '--prerelease' }

$cmd = @(
    'release', 'create', $Tag,
    $asset,
    $checksum,
    '--title', $Title,
    '--notes', $Notes
) + $flags

Write-Host "Running: gh $($cmd -join ' ')"
& gh @cmd

Write-Host 'Done. GUI release published.'
