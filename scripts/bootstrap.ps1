[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$Venv = '.venv',
    [string]$Extras = 'ui,dicom',
    [string]$CacheDir = '',
    [string]$VisionBundle = '',
    [string]$RuntimeConfig = '',
    [Alias('skip-models')]
    [switch]$SkipModels,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$EnvFile = Join-Path $ProjectRoot '.env'
$EnvExample = Join-Path $ProjectRoot '.env.example'

if ([string]::IsNullOrWhiteSpace($CacheDir)) {
    if (-not [string]::IsNullOrWhiteSpace($env:TBX_ARTIFACT_ROOT)) {
        $CacheDir = $env:TBX_ARTIFACT_ROOT
    } else {
        $CacheDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'TBX-Agent\artifacts'
    }
}
$CacheDir = [System.IO.Path]::GetFullPath($CacheDir)
$InstallBundle = -not [string]::IsNullOrWhiteSpace($VisionBundle)
if ($InstallBundle -and $SkipModels) {
    throw '-VisionBundle cannot be combined with -SkipModels.'
}
if ($InstallBundle -and [string]::IsNullOrWhiteSpace($RuntimeConfig)) {
    throw '-VisionBundle requires -RuntimeConfig pointing to an external runtime JSON path.'
}
if ($InstallBundle -and $Extras -notmatch '(^|,)vision(,|$)') {
    $Extras = "$Extras,vision"
}
if (-not [System.IO.Path]::IsPathRooted($Venv)) { $Venv = Join-Path $ProjectRoot $Venv }
$VenvPath = [System.IO.Path]::GetFullPath($Venv)
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'

function Invoke-Step {
    param([string]$Executable, [string[]]$Arguments)
    Write-Host ('> ' + $Executable + ' ' + ($Arguments -join ' '))
    if ($DryRun) { return }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Executable" }
}

if ($DryRun -or -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Invoke-Step $Python @('-m', 'venv', $VenvPath)
}
Invoke-Step $VenvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Step $VenvPython @('-m', 'pip', 'install', '-e', "$ProjectRoot[$Extras]")
if ($InstallBundle) {
    $RuntimeConfig = [System.IO.Path]::GetFullPath($RuntimeConfig)
    Invoke-Step $VenvPython @(
        (Join-Path $ProjectRoot 'scripts\install_vision_bundle.py'),
        '--bundle', [System.IO.Path]::GetFullPath($VisionBundle),
        '--artifact-root', $CacheDir, '--runtime-config', $RuntimeConfig
    )
}
if ($DryRun) {
    Write-Host 'Dry run complete; no files changed or models downloaded.'
    exit 0
}
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    Add-Content -LiteralPath $EnvFile -Value "`nTBX_ARTIFACT_ROOT=$CacheDir" -Encoding utf8
    if ($InstallBundle) {
        Add-Content -LiteralPath $EnvFile -Value "TBX_AGENT_RANK03_RUNTIME_CONFIG=$RuntimeConfig" -Encoding utf8
    }
    Write-Host "Created $EnvFile; review it before real inference."
} else {
    Write-Host 'Existing .env was preserved.'
    if ($InstallBundle) { Write-Host "Set TBX_AGENT_RANK03_RUNTIME_CONFIG=$RuntimeConfig and TBX_ARTIFACT_ROOT=$CacheDir in .env." }
}
Write-Host 'Ready for a model-free demonstration: .\scripts\run_local.ps1 -Demo'
Write-Host 'For real vision, download the published inference bundle and use -VisionBundle <zip> -RuntimeConfig <external-json>.'
Write-Host 'Install optional D-FINE source and PSPNet explicitly with bootstrap_dfine.py and bootstrap_models.py download anatomy.'
