param(
    [string]$CirCorData = "$PSScriptRoot\work\data\circor_subset",
    [string]$BmdData = "$PSScriptRoot\work\data\bmd_hs",
    [switch]$RebuildFeatures
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The project environment is missing. Run .\setup.ps1 first."
}
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot"

if (-not (Test-Path -LiteralPath $CirCorData)) {
    throw "CirCor subset not found at $CirCorData. Run scripts\download_circor_subset.py first."
}
if (-not (Test-Path -LiteralPath $BmdData)) {
    throw "BMD-HS data not found at $BmdData. See docs\RESEARCH_DIRECTION.md."
}
if ($RebuildFeatures) {
    $Cache = Join-Path $ProjectRoot "work\phase_research_features"
    if (Test-Path -LiteralPath $Cache) {
        Remove-Item -LiteralPath $Cache -Recurse -Force
    }
}

& $VenvPython "$ProjectRoot\scripts\train_phase_segmenter.py" `
    --data $CirCorData --output "$ProjectRoot\outputs\phase_model"
if ($LASTEXITCODE -ne 0) { throw "Phase segmenter training failed." }

& $VenvPython "$ProjectRoot\scripts\run_phase_research.py" `
    --dataset $BmdData `
    --phase-model "$ProjectRoot\outputs\phase_model\phase_segmenter.joblib" `
    --output "$ProjectRoot\outputs\phase_disease_model"
if ($LASTEXITCODE -ne 0) { throw "Phase-aware disease experiment failed." }

& $VenvPython "$ProjectRoot\tests\test_phases.py"
if ($LASTEXITCODE -ne 0) { throw "Phase tests failed." }
Write-Host "Research pipeline complete: $ProjectRoot\outputs\phase_disease_model"
