# Creates the virtual environment on first run, installs the package and starts the server.
# Usage (Windows PowerShell 5.1 or later):  .\start.ps1
# If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\start.ps1
#
# Dependencies are installed only when .venv was just created or pyproject.toml changed since
# the last successful install (a hash is kept in .venv\visionflow-install.sha256), so later
# starts work offline.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pyproject = Join-Path $PSScriptRoot "pyproject.toml"
$hashFile = Join-Path $PSScriptRoot ".venv\visionflow-install.sha256"
$versionCheck = "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"

function Find-BasePython {
    # Returns @(command, args...) for the first Python 3.11+ found, or $null.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += , @("py", "-3") }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @("python") }
    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $extra = @($candidate | Select-Object -Skip 1)
        & $exe @extra -c $versionCheck
        if ($LASTEXITCODE -eq 0) { return , $candidate }
    }
    return $null
}

$justCreated = $false
if (-not (Test-Path $venvPython)) {
    $base = Find-BasePython
    if ($null -eq $base) {
        Write-Host "ERROR: Python 3.11 or newer was not found (checked 'py -3' and 'python')."
        Write-Host "Install Python 3.11+ from https://www.python.org/ and try again."
        exit 1
    }
    Write-Host "Creating virtual environment (.venv) ..."
    $exe = $base[0]
    $extra = @($base | Select-Object -Skip 1)
    & $exe @extra -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: could not create the virtual environment."
        exit 1
    }
    $justCreated = $true
}

& $venvPython -c $versionCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: .venv uses a Python older than 3.11. Delete the .venv folder and run this script again."
    exit 1
}

$currentHash = (Get-FileHash -Path $pyproject -Algorithm SHA256).Hash
$storedHash = ""
if (Test-Path $hashFile) { $storedHash = (Get-Content -Path $hashFile -Raw).Trim() }

if ($justCreated -or ($storedHash -ne $currentHash)) {
    Write-Host "Installing dependencies (the first run downloads several hundred MB) ..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip upgrade failed."; exit 1 }
    & $venvPython -m pip install -e .
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: package installation failed."; exit 1 }
    Set-Content -Path $hashFile -Value $currentHash -Encoding ASCII
} else {
    Write-Host "Dependencies are up to date (pyproject.toml unchanged); skipping installation."
}

$torchInfo = & $venvPython -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host $torchInfo
if ($torchInfo -notmatch "cuda True") {
    Write-Host "NOTE: CUDA is not available, so the GPU is NOT used (CPU only, slow; yolo11n is suggested). See the 'GPU kullanimi' section in README.md to install a CUDA build of torch."
}

Write-Host ""
Write-Host "Starting the server. Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop."
Write-Host ""
& $venvPython -m uvicorn visionflow.api:create_app --factory --host 127.0.0.1 --port 8000
