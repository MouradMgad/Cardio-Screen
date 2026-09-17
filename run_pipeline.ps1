param(
    [switch]$ReextractFeatures
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The project environment is missing. Run .\setup.ps1 first."
}
$PythonExe = $VenvPython
$env:PYTHONPATH = "$ProjectRoot\src"

function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArguments)
    & $PythonExe @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline command failed with exit code $LASTEXITCODE."
    }
}

$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = "$ProjectRoot\work"
$env:LOKY_MAX_CPU_COUNT = "8"

$DataDirectory = Join-Path $ProjectRoot "work\data\training"
$FeatureFile = Join-Path $ProjectRoot "work\features_v2.csv"
if (-not (Test-Path -LiteralPath $DataDirectory)) {
    throw "Dataset not found at $DataDirectory. See DATASET.md."
}

if ($ReextractFeatures -or -not (Test-Path -LiteralPath $FeatureFile)) {
    Invoke-ProjectPython "$ProjectRoot\scripts\extract_features.py" --data $DataDirectory --output $FeatureFile --jobs 6
} else {
    Write-Host "Using cached features: $FeatureFile"
    Write-Host "Use .\run_pipeline.ps1 -ReextractFeatures to rebuild them from every WAV file."
}

Invoke-ProjectPython "$ProjectRoot\scripts\tune.py" --features "$ProjectRoot\work\features_v2.csv" --output "$ProjectRoot\outputs\model_final"
Invoke-ProjectPython "$ProjectRoot\scripts\selective_eval.py" --features "$ProjectRoot\work\features_v2.csv" --model-dir "$ProjectRoot\outputs\model_final"
Invoke-ProjectPython "$ProjectRoot\scripts\train_phenotype.py" --features "$ProjectRoot\work\features_v2.csv" --output "$ProjectRoot\outputs\model_final"
Invoke-ProjectPython "$ProjectRoot\scripts\evaluate_phenotype.py" --features "$ProjectRoot\work\features_v2.csv" --model "$ProjectRoot\outputs\model_final\phenotype_model.joblib" --binary-metrics "$ProjectRoot\outputs\model_final\metrics.json" --output "$ProjectRoot\outputs\model_final\phenotype_validation.json"
Invoke-ProjectPython "$ProjectRoot\scripts\build_report.py"
Invoke-ProjectPython "$ProjectRoot\tests\test_phenotypes.py"
Invoke-ProjectPython "$ProjectRoot\tests\smoke_test.py"
Write-Host "Pipeline complete. Results: $ProjectRoot\outputs\model_final"
