@echo off
setlocal
cd /d %~dp0\..
echo === Interpreter ===
".venv\Scripts\python.exe" -c "import sys,platform; print(platform.python_version()); print(sys.executable)"
echo === bondviz import ===
set PYTHONPATH=%CD%\src
".venv\Scripts\python.exe" - <<PY
try:
    import bondviz, inspect, pathlib
    print("bondviz OK:", pathlib.Path(bondviz.__file__).resolve())
except Exception as e:
    print("bondviz import FAILED:", e)
PY
