# fetches data from Wharton Market Research and joins with existing predictors data

import wrds
import pandas as pd

# ── Step 1: Pull monthly returns and risk-free rate from WRDS ──
db = wrds.Connection()  # prompts for WRDS login

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

db.close()

# ── Step 2: Merge returns and rf with signed predictors ──
# Only load permno and yyyymm — we don't need the characteristics here,
# those stay in the original CSV and get loaded by data_loader.py later.
signals = pd.read_csv('/Users/liamoshaughnessy/Documents/ECO480/final_project/data/raw/signed_predictors_dl_wide.csv',
                      usecols=['permno', 'yyyymm'], low_memory=False)
print(f"Signals loaded: {len(signals):,} rows")

panel = signals.merge(crsp, on=['permno', 'yyyymm'], how='inner')
panel = panel.merge(ff, on='yyyymm', how='left')

panel['ret_excess'] = panel['ret'] - panel['rf']
print(f"Panel merged: {len(panel):,} rows, {panel['permno'].nunique():,} stocks")

# ── Step 3: Save returns/rf only (characteristics stay in signed_predictors CSV) ──
panel.to_csv('/Users/liamoshaughnessy/Documents/ECO480/final_project/data/raw/returns.csv', index=False)
print("Saved to data/raw/returns.csv")
