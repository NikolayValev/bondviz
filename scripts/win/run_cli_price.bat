@echo off
setlocal
cd /d %~dp0\..\..
IF NOT EXIST ".venv\Scripts\python.exe" (
  echo Virtual environment missing. Run scripts\win\setup_venv.bat
  exit /b 1
)
set PYTHONPATH=%CD%\src
REM Example: 5% coupon, 4% cont. yield, 10y, face 1000
".venv\Scripts\python.exe" scripts\price_bond.py --coupon 0.05 --yield 0.04 --years 10 --face 1000
