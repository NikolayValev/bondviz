# PowerShell-safe launcher that bypasses policy for this session only.
param(
  [ValidateSet("streamlit","fetch","price")] [string]$mode = "streamlit",
  [double]$coupon = 0.05,
  [double]$yield = 0.04,
  [double]$years = 10,
  [double]$face = 1000
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..  # project root

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "venv missing; running setup..." -ForegroundColor Yellow
  & scripts\win\setup_venv.bat
}

$env:PYTHONPATH = (Join-Path (Get-Location) "src")

switch ($mode) {
  "streamlit" { & $py -m streamlit run app\streamlit_app.py }
  "fetch"     { & $py scripts\fetch_treasury.py }
  "price"     { & $py scripts\price_bond.py --coupon $coupon --yield $yield --years $years --face $face }
}
