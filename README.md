# BondViz

A minimal Python project for fetching and visualizing U.S. Treasury yield curves and pricing plain-vanilla bonds under continuous compounding.

---

## Overview

BondViz provides:

- Automated Treasury yield data from the U.S. Department of Treasury XML feed  
- Continuous-compounding bond valuation model  
- Matplotlib visualizations of yield curves and discount factors  
- Streamlit dashboard for interactive exploration  
- Polygon.io stock data integration  
- Yield-curve PCA explorer (scikit-learn)  
- Optional pybind11 C++ extension for fast bond math

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
```

### 2. Native extension (optional)

The project ships with a pybind11-backed extension (`bondviz._native`) that accelerates bond pricing.

- Windows: install Microsoft Build Tools (C++ workload) before running the setup script.  
- macOS/Linux: ensure clang/gcc and headers are available (`xcode-select --install` or `build-essential`).  
- Run `pip install -e .` (or the helper script) to compile automatically. If compilation fails, the app falls back to Python implementations.

### Usage

Command-line tools

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

## Screenshots



## Notes

Uses official Treasury XML feed (no API key required)

Handles namespace changes and missing date tags gracefully

Compatible with Windows, WSL, and macOS

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app/streamlit_app.py`.
3. Streamlit installs `requirements.txt` (pure Python — no C++ build; bond math falls
   back to the Python implementation when the native extension is absent).
4. Home and Fixed Income work with no configuration. For the Stocks page, add a secret
   under **Manage app → Settings → Secrets**:

   ```toml
   POLYGON_API_KEY = "your_key_here"
   ```

## Web front-end (Vercel)

A native Next.js + D3 front-end lives in [`web/`](web/) and deploys to Vercel at a custom domain
(see [`web/README.md`](web/README.md)). It reuses the same Treasury data and is independent of the
Streamlit app.

### License

MIT License
