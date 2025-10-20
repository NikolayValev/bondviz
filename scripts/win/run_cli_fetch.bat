@echo off
setlocal
cd /d %~dp0\..\..
IF NOT EXIST ".venv\Scripts\python.exe" (
  echo Virtual environment missing. Run scripts\win\setup_venv.bat
  exit /b 1
)
set PYTHONPATH=%CD%\src
".venv\Scripts\python.exe" scripts\fetch_treasury.py
