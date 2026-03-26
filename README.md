# U.S. Economy Snapshot

This project pulls current U.S. macroeconomic indicators, generates a chart pack, and writes a short analysis brief.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/us_economy_analysis.py
```

## Outputs

- `data/raw/`: downloaded series from public data endpoints
- `data/processed/`: merged and derived indicator tables
- `plots/`: PNG charts
- `reports/us_economy_snapshot.md`: written analysis
