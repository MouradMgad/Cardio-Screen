$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\0\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = "$ProjectRoot\work\python_packages;$ProjectRoot\src"

if (-not (Test-Path $PythonExe)) {
    throw "Bundled Python runtime was not found at $PythonExe"
}

$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = "$ProjectRoot\work"
$env:LOKY_MAX_CPU_COUNT = "8"

& $PythonExe "$ProjectRoot\scripts\extract_features.py" --output "$ProjectRoot\work\features_v2.csv" --jobs 6
& $PythonExe "$ProjectRoot\scripts\tune.py" --features "$ProjectRoot\work\features_v2.csv" --output "$ProjectRoot\outputs\model_final"
& $PythonExe "$ProjectRoot\scripts\selective_eval.py" --features "$ProjectRoot\work\features_v2.csv" --model-dir "$ProjectRoot\outputs\model_final"
& $PythonExe "$ProjectRoot\scripts\build_report.py"
& $PythonExe "$ProjectRoot\tests\smoke_test.py"
Write-Host "Pipeline complete. Results: $ProjectRoot\outputs\model_final"
