@echo off
setlocal
cd /d %~dp0\..\..
REM Step 1: create venv only if it doesn't already exist
IF EXIST ".venv\Scripts\python.exe" (
  echo [1/4] Reusing existing venv...
) ELSE (
  echo [1/4] Creating venv...
  REM Prefer system launcher to avoid recreating the active venv
  where py >NUL 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv) || goto :err
)
echo [2/4] Upgrading pip/setuptools...
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel || goto :err
echo [3/4] Installing project (editable)...
.\.venv\Scripts\python.exe -m pip install -e . || goto :err
echo [4/4] Installing app deps...
.\.venv\Scripts\python.exe -m pip install streamlit matplotlib pandas requests python-dotenv pandas-datareader polygon-api-client || goto :err
echo Done.
exit /b 0
:err
echo FAILED with errorlevel=%errorlevel%
exit /b %errorlevel%
