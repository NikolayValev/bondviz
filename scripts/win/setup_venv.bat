@echo off
setlocal
cd /d %~dp0\..
echo [1/4] Creating venv...
python -m venv .venv || goto :err
echo [2/4] Upgrading pip/setuptools...
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel || goto :err
echo [3/4] Installing project (editable)...
.\.venv\Scripts\python.exe -m pip install -e . || goto :err
echo [4/4] Installing app deps...
.\.venv\Scripts\python.exe -m pip install streamlit matplotlib pandas requests python-dotenv || goto :err
echo Done.
exit /b 0
:err
echo FAILED with errorlevel=%errorlevel%
exit /b %errorlevel%
