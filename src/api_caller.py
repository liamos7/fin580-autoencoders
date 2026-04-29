# fetches data from Wharton Market Research and joins with existing predictors data

import sys
from pathlib import Path

import wrds
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

RAW_DIR = Path(config.RAW_DIR)


def fetch_returns():
    try:
        db = wrds.Connection()  # prompts for WRDS login

        # ── Step 1: Pull monthly returns and risk-free rate from WRDS ──
        crsp = db.raw_sql("""
            SELECT permno, date, ret
            FROM crsp.msf
            WHERE date >= '1957-03-01' AND date <= '2023-12-31'
        """)
        crsp['yyyymm'] = pd.to_datetime(crsp['date']).dt.strftime('%Y%m').astype(int)
        crsp = crsp[['permno', 'yyyymm', 'ret']]
        print(f"Returns downloaded: {len(crsp):,} rows")

        ff = db.raw_sql("""
            SELECT date, rf
            FROM ff.factors_monthly
            WHERE date >= '1957-03-01' AND date <= '2023-12-31'
        """)
        ff['yyyymm'] = pd.to_datetime(ff['date']).dt.strftime('%Y%m').astype(int)
        ff = ff[['yyyymm', 'rf']]
        print(f"Risk-free rate downloaded: {len(ff):,} rows")
    finally:
        db.close()

    # ── Step 2: Merge returns and rf with signed predictors ──
    signals_path = RAW_DIR / "signed_predictors_dl_wide.csv"
    signals = pd.read_csv(signals_path, usecols=['permno', 'yyyymm'], low_memory=False)
    print(f"Signals loaded: {len(signals):,} rows")

    panel = signals.merge(crsp, on=['permno', 'yyyymm'], how='inner')
    panel = panel.merge(ff, on='yyyymm', how='left')
    panel['ret_excess'] = panel['ret'] - panel['rf']
    print(f"Panel merged: {len(panel):,} rows, {panel['permno'].nunique():,} stocks")

    # ── Step 3: Save excess returns only ──
    out_cols = ['permno', 'yyyymm', 'ret_excess']
    out_path = RAW_DIR / "returns.csv"
    panel[out_cols].to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    fetch_returns()
