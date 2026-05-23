# =============================================================================
# SpecVLM — Windows Setup Script (PowerShell)
# =============================================================================
# This script sets up the development environment for SpecVLM on Windows.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
#
# What it does:
# 1. Creates Python virtual environment
# 2. Installs PyTorch with CUDA support
# 3. Installs all SpecVLM dependencies
# 4. Verifies GPU availability
# 5. Runs a quick smoke test

param(
    [string]$EnvDir = "specvlm-env",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.ForegroundColor = "Green"
Write-Host "=== SpecVLM Environment Setup ===" -ForegroundColor Cyan
$Host.UI.RawUI.ForegroundColor = "White"

# Step 1: Check Python
Write-Host "`n[1/6] Checking Python..." -ForegroundColor Yellow
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
Write-Host "  Found: $pyVersion"

# Step 2: Create virtual environment
Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path $EnvDir) {
    Write-Host "  Environment '$EnvDir' already exists. Skipping creation."
} else {
    python -m venv $EnvDir
    Write-Host "  Created: $EnvDir"
}

# Activate environment
$activatePath = "$PWD\$EnvDir\Scripts\Activate.ps1"
if (Test-Path $activatePath) {
    & $activatePath
    Write-Host "  Environment activated"
} else {
    Write-Host "  WARNING: Could not activate. Try running: $activatePath" -ForegroundColor Yellow
}

# Step 3: Upgrade pip
Write-Host "`n[3/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Step 4: Install PyTorch with CUDA
Write-Host "`n[4/6] Installing PyTorch..." -ForegroundColor Yellow
if (-not $SkipTorch) {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} else {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Verify PyTorch
$torchTest = python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>&1
Write-Host "  $torchTest"

# Step 5: Install SpecVLM dependencies
Write-Host "`n[5/6] Installing SpecVLM dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install -e .

# Step 6: Verify installation
Write-Host "`n[6/6] Verifying installation..." -ForegroundColor Yellow
python -c "
from specvlm.config.settings import settings
from specvlm.inference.engine import InferenceEngine, EngineConfig
from specvlm.inference.visual_encoder import VisualEncoder
from specvlm.inference.kv_cache import KVCacheManager
from specvlm.inference.token_verifier import TokenVerifier

print('✓ SpecVLM modules loaded successfully')
print(f'✓ Project: {settings.project_name} v{settings.version}')
print(f'✓ CUDA available: {__import__(\"torch\").cuda.is_available()}')
"

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:"
Write-Host "  .\$EnvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run inference:"
Write-Host "  python experiments/phase1_baseline.py --prompt 'Hello'"
Write-Host ""
Write-Host "To start the server:"
Write-Host "  uvicorn specvlm.serving.api:app --reload"
