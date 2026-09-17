$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The project environment is missing. Run .\setup.ps1 first."
}
$PythonExe = $VenvPython
$env:PYTHONPATH = "$ProjectRoot\src"

$ModelPath = Join-Path $ProjectRoot "outputs\model_final\heart_sound_model.joblib"
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "The trained model is missing. Run .\run_pipeline.ps1 first."
}

Write-Host "Starting CardioScreen at http://127.0.0.1:7860"
& $PythonExe "$ProjectRoot\app.py"
