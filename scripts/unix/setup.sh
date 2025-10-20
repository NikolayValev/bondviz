#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m venv .venv
. .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
pip install streamlit matplotlib pandas requests python-dotenv pandas-datareader polygon-api-client
