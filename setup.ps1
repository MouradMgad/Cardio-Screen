$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if ($UvCommand) {
        $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
        if (Test-Path -LiteralPath $BundledPython) {
            & $UvCommand.Source venv --python $BundledPython $VenvDirectory
        } else {
            & $UvCommand.Source venv $VenvDirectory
        }
    } elseif (Test-Path -LiteralPath $BundledPython) {
        & $BundledPython -m venv $VenvDirectory
    } else {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PyLauncher) {
            & $PyLauncher.Source -3 -m venv $VenvDirectory
        } else {
            $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if (-not $PythonCommand) {
                throw "Install Python 3.12 or newer, then run this script again."
            }
            & $PythonCommand.Source -m venv $VenvDirectory
        }
    }
}

if ($UvCommand) {
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
    & $UvCommand.Source pip install --python $VenvPython -r (Join-Path $ProjectRoot "requirements.txt")
} else {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE."
}
Write-Host "Setup complete. Run .\run_app.ps1 or .\run_pipeline.ps1."
