@echo off
setlocal
cd /d %~dp0\..\..
REM Preferred: run through the venv interpreter to avoid PATH issues
IF NOT EXIST ".venv\Scripts\python.exe" (
  echo Virtual environment missing. Run scripts\win\setup_venv.bat
  exit /b 1
)
set PYTHONPATH=%CD%\src
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.runOnSave true
