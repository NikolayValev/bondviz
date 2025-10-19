#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m streamlit run app/streamlit_app.py
