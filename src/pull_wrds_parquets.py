"""
step1_pull_wrds.py  —  run this on the LOGIN NODE (has internet access)
========================================================================
Pulls raw tables from WRDS and saves them as parquets in data/checkpoints/.
Takes ~10-20 minutes. No heavy computation — just downloads.

    python src/step1_pull_wrds.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

CKPT_DIR = Path("data/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WRDS_USER = os.environ["WRDS_USER"]
WRDS_PASS = os.environ["WRDS_PASS"]

def make_engine():
    # Try the two known WRDS hostnames
    for host in ["wrds-pgdata.wharton.upenn.edu", "wrds-cloud.wharton.upenn.edu"]:
        try:
            url = (
                f"postgresql+psycopg2://{WRDS_USER}:{WRDS_PASS}"
                f"@{host}:9737/wrds?sslmode=require"
            )
            engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"  Connected via {host}")
            return engine
        except Exception as e:
            print(f"  {host} failed: {e}")
    raise RuntimeError("Could not connect to WRDS. Are you on the login node?")

def pull(engine, sql, date_cols=None):
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, parse_dates=date_cols)

print("Connecting to WRDS...")
engine = make_engine()

# ── CRSP monthly stock file ──────────────────────────────────────
ckpt = CKPT_DIR / "crsp_raw.parquet"
if ckpt.exists():
    print(f"CRSP raw already exists, skipping ({ckpt})")
else:
    print("Pulling CRSP msf (this takes ~5 min)...")
    crsp = pull(engine, """
        SELECT permno, date, ret, prc, shrout, vol
        FROM crsp.msf
        WHERE date >= '1960-01-01' AND date <= '2024-01-01'
    """, date_cols=["date"])
    crsp.to_parquet(ckpt, index=False)
    print(f"  Saved {len(crsp):,} rows -> {ckpt}")
    del crsp

# ── Fama-French factors ──────────────────────────────────────────
ckpt = CKPT_DIR / "ff_monthly.parquet"
if ckpt.exists():
    print(f"FF factors already exist, skipping")
else:
    print("Pulling FF monthly factors...")
    ff = pull(engine, """
        SELECT date, mktrf, rf
        FROM ff.factors_monthly
        WHERE date >= '1955-01-01' AND date <= '2024-01-01'
    """, date_cols=["date"])
    ff.to_parquet(ckpt, index=False)
    print(f"  Saved {len(ff):,} rows -> {ckpt}")
    del ff

# ── CCM link table ───────────────────────────────────────────────
ckpt = CKPT_DIR / "ccm.parquet"
if ckpt.exists():
    print(f"CCM already exists, skipping")
else:
    print("Pulling CCM link table...")
    ccm = pull(engine, """
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
    """, date_cols=["linkdt", "linkenddt"])
    ccm.to_parquet(ckpt, index=False)
    print(f"  Saved {len(ccm):,} rows -> {ckpt}")

# ── Compustat annual ─────────────────────────────────────────────
ckpt = CKPT_DIR / "comp_funda_raw.parquet"
if ckpt.exists():
    print(f"Compustat funda already exists, skipping")
else:
    print("Pulling Compustat funda (this takes ~5 min)...")
    comp = pull(engine, """
        SELECT gvkey, datadate,
               at, ceq, csho, lt, sale, ib, xrd, mkvalt,
               dltt, dp, ppent, rect, invt, che,
               xsga, act, lct, capx, xint, cogs, sich
        FROM comp.funda
        WHERE indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
          AND datadate >= '1960-01-01'
        ORDER BY gvkey, datadate
    """, date_cols=["datadate"])
    comp["gp"] = comp["sale"] - comp["cogs"].fillna(0)
    comp.to_parquet(ckpt, index=False)
    print(f"  Saved {len(comp):,} rows -> {ckpt}")
    del comp

# ── Compustat quarterly ──────────────────────────────────────────
ckpt = CKPT_DIR / "comp_fundq_raw.parquet"
if ckpt.exists():
    print(f"Compustat fundq already exists, skipping")
else:
    print("Pulling Compustat fundq...")
    compq = pull(engine, """
        SELECT gvkey, datadate, saleq, atq, ibq
        FROM comp.fundq
        WHERE indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
          AND fqtr IS NOT NULL
          AND datadate >= '1970-01-01'
          AND datadate <= '2024-01-01'
        ORDER BY gvkey, datadate
    """, date_cols=["datadate"])
    compq.to_parquet(ckpt, index=False)
    print(f"  Saved {len(compq):,} rows -> {ckpt}")
    del compq

engine.dispose()
print("\nAll WRDS data pulled. Now submit: sbatch submit_build.sh")