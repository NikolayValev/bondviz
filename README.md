# BondViz

A minimal Python project for fetching and visualizing U.S. Treasury yield curves and pricing plain-vanilla bonds under continuous compounding.

---

## Overview

BondViz provides:

- Automated Treasury yield data from the U.S. Department of Treasury XML feed  
- Continuous-compounding bond valuation model  
- Matplotlib visualizations of yield curves and discount factors  
- Streamlit dashboard for interactive exploration

---

## Project Structure

bondviz/
├── pyproject.toml
├── src/bondviz/
│ ├── init.py
│ ├── pricing.py # Continuous compounding logic
│ ├── treasury.py # Fetch & parse Treasury XML
│ └── plots.py # Matplotlib visualization
├── scripts/
│ ├── fetch_treasury.py # CLI: fetch yield data
│ ├── price_bond.py # CLI: compute PV
│ ├── win/ # Windows startup scripts
│ └── unix/ # Unix/WSL startup scripts
└── app/
└── streamlit_app.py # Web dashboard

---

## Setup

### 1. Create and install dependencies

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.venv\Scripts\python.exe -m pip install -e .
```

#### Alternatively use helper script

```bash
scripts\win\setup_venv.bat
Usage
Command-line tools
```

Fetch latest yield curve:

```bash
scripts\win\run_cli_fetch.bat
```

Price a bond:

```bash
scripts\win\run_cli_price.bat
```

## Streamlit App

```bash
scripts\win\run_streamlit.bat
```

or

```bash
python -m streamlit run app\streamlit_app.py
```

Open the local URL (default <http://localhost:8501>) to view:

Current Treasury yield curve

Continuous-compounding bond PV calculator

## Notes

Uses official Treasury XML feed (no API key required)

Handles namespace changes and missing date tags gracefully

Compatible with Windows, WSL, and macOS

### License

MIT License
