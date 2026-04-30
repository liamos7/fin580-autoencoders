"""
build_missing.py  —  run this as a SLURM job (no internet needed)
==================================================================
Reads raw parquets saved by step1_pull_wrds.py, constructs all missing
GKX characteristics, and rebuilds panel_processed.parquet.

    sbatch submit_build.sh

Assumes data/checkpoints/ already has:
    crsp_raw.parquet, ff_monthly.parquet, ccm.parquet,
    comp_funda_raw.parquet, comp_fundq_raw.parquet
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CKPT_DIR      = Path("data/checkpoints")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Final computed characteristic checkpoints
CKPT_CRSP   = CKPT_DIR / "crsp_chars.parquet"
CKPT_MVE    = CKPT_DIR / "mve_monthly.parquet"
CKPT_ANNUAL = CKPT_DIR / "comp_annual.parquet"
CKPT_QTRLY  = CKPT_DIR / "comp_quarterly.parquet"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def rolling_cumret(group, start_lag, end_lag):
    r = group["ret1"].values
    T = len(r)
    result = np.full(T, np.nan)
    n_needed = start_lag - end_lag
    for t in range(start_lag, T):
        window = r[t - start_lag : t - end_lag]
        if len(window) == n_needed and not np.any(np.isnan(window)):
            result[t] = np.prod(window) - 1
    return pd.Series(result, index=group.index, name="value")


def rolling_beta_vec(group, min_obs=24, window=60):
    exc = group["exc_ret"].values.astype(np.float64)
    mkt = group["mktrf"].values.astype(np.float64)
    T = len(exc)
    betas = np.full(T, np.nan)
    for t in range(min_obs, T):
        sl = slice(max(0, t - window), t)
        r, m = exc[sl], mkt[sl]
        valid = ~(np.isnan(r) | np.isnan(m))
        if valid.sum() < min_obs:
            continue
        r, m = r[valid], m[valid]
        m_var = m.var(ddof=1)
        if m_var > 0:
            betas[t] = np.cov(r, m, ddof=1)[0, 1] / m_var
    return pd.Series(betas, index=group.index, name="beta")


def pct_change_prev(s):
    return (s / s.shift(1).clip(lower=1e-6)) - 1


def rank_normalize_col(arr, dates):
    result = np.empty_like(arr, dtype=np.float64)
    for d in np.unique(dates):
        mask = dates == d
        block = arr[mask].astype(np.float64)
        n = mask.sum()
        if n <= 1:
            result[mask] = 0.0
            continue
        ranked = rankdata(block, method="average", nan_policy="propagate")
        result[mask] = 2.0 * ranked / (n + 1) - 1.0
    return result


# ═══════════════════════════════════════════════════════════════════
# PART 1: CRSP — mvel1, turn, mom36m, chmom, betasq
# ═══════════════════════════════════════════════════════════════════

if CKPT_CRSP.exists() and CKPT_MVE.exists():
    print("\n── Loading CRSP chars from checkpoint ──")
    crsp_chars  = pd.read_parquet(CKPT_CRSP)
    mve_monthly = pd.read_parquet(CKPT_MVE)
    print(f"  {len(crsp_chars):,} rows")
else:
    print("\n── Building CRSP characteristics ──")
    crsp = pd.read_parquet(CKPT_DIR / "crsp_raw.parquet")
    ff   = pd.read_parquet(CKPT_DIR / "ff_monthly.parquet")

    crsp["yyyymm"] = pd.to_datetime(crsp["date"]).dt.to_period("M").dt.to_timestamp()
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    crsp["mve"]   = crsp["prc"].abs() * crsp["shrout"]
    crsp["mvel1"] = np.log(crsp["mve"].clip(lower=1e-6))
    crsp["turn"]  = (crsp["vol"] / crsp["shrout"].clip(lower=1)).clip(lower=0)

    ff["yyyymm"] = pd.to_datetime(ff["date"]).dt.to_period("M").dt.to_timestamp()
    ff = ff.set_index("yyyymm")[["mktrf", "rf"]]
    crsp = crsp.merge(ff, left_on="yyyymm", right_index=True, how="left")
    crsp["exc_ret"] = crsp["ret"] - crsp["rf"]

    crsp_mom = crsp[["permno", "yyyymm", "ret"]].copy()
    crsp_mom["ret1"] = 1 + crsp_mom["ret"].fillna(0)

    print("  Computing mom36m...")
    mom36m = (
        crsp_mom.groupby("permno", group_keys=False)[["yyyymm", "ret1"]]
        .apply(lambda g: rolling_cumret(g, start_lag=36, end_lag=13),
               include_groups=False)
    )
    crsp["mom36m"] = mom36m.values

    print("  Computing chmom...")
    mom6m_raw = (
        crsp_mom.groupby("permno", group_keys=False)[["yyyymm", "ret1"]]
        .apply(lambda g: rolling_cumret(g, start_lag=6, end_lag=1),
               include_groups=False)
    )
    crsp["mom6m_tmp"] = mom6m_raw.values
    crsp["chmom"] = crsp.groupby("permno")["mom6m_tmp"].transform(
        lambda x: x - x.shift(6)
    )
    crsp.drop(columns=["mom6m_tmp"], inplace=True)

    print("  Computing betasq...")
    betas = (
        crsp.sort_values(["permno", "yyyymm"])
        .groupby("permno", group_keys=False)[["yyyymm", "exc_ret", "mktrf"]]
        .apply(rolling_beta_vec, include_groups=False)
    )
    crsp["betasq"] = betas.values ** 2

    crsp["date"] = crsp["yyyymm"].dt.strftime("%Y%m").astype(int)
    crsp_chars  = crsp[["permno", "date", "mvel1", "turn", "mom36m", "chmom", "betasq"]]
    mve_monthly = crsp[["permno", "date", "mve"]].rename(columns={"date": "yyyymm_int"})

    crsp_chars.to_parquet(CKPT_CRSP, index=False)
    mve_monthly.to_parquet(CKPT_MVE, index=False)
    del crsp, crsp_mom, mom36m, mom6m_raw, betas, ff
    print(f"  Done: {len(crsp_chars):,} rows")


# ═══════════════════════════════════════════════════════════════════
# PART 2: Compustat annual — 28 characteristics
# ═══════════════════════════════════════════════════════════════════

annual_char_cols = [
    "egr", "chcsho", "rd_mve", "sgr", "lgr", "depr",
    "secured", "securedind", "roic", "salerec", "salecash",
    "saleinv", "pchsaleinv", "currat", "quick", "pchquick",
    "cinvest", "pchdepr", "grgma", "pchgm_pchsale",
    "pchsale_pchxsga", "pchsale_pchinvt", "pchsale_pchrect",
    "bm_ia", "mve_ia", "cfp_ia", "chpmia", "chatoia",
]

if CKPT_ANNUAL.exists():
    print("\n── Loading Compustat annual chars from checkpoint ──")
    comp_annual = pd.read_parquet(CKPT_ANNUAL)
    ccm         = pd.read_parquet(CKPT_DIR / "ccm.parquet")
    print(f"  {len(comp_annual):,} rows")
else:
    print("\n── Building Compustat annual characteristics ──")
    comp = pd.read_parquet(CKPT_DIR / "comp_funda_raw.parquet")
    ccm  = pd.read_parquet(CKPT_DIR / "ccm.parquet")
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    comp = comp.merge(ccm, on="gvkey", how="inner")
    comp = comp[
        (comp["datadate"] >= comp["linkdt"]) &
        (comp["datadate"] <= comp["linkenddt"])
    ]

    comp["use_date"]   = comp["datadate"] + pd.DateOffset(months=6)
    comp["yyyymm_int"] = comp["use_date"].dt.year * 100 + comp["use_date"].dt.month
    comp = comp.sort_values(["permno", "yyyymm_int", "datadate"])
    comp = comp.drop_duplicates(subset=["permno", "yyyymm_int"], keep="last")

    comp = comp.merge(mve_monthly, on=["permno", "yyyymm_int"], how="left")
    comp["mve"] = comp["mve"].fillna(comp["mkvalt"].fillna(0) * 1000).clip(lower=1)

    comp["sic2"] = comp["sich"].fillna(0).astype(int).astype(str).str[:2].str.zfill(2)
    comp = comp.sort_values(["permno", "datadate"]).reset_index(drop=True)

    g = comp.groupby("permno")

    comp["egr"]        = g["ceq"].transform(pct_change_prev)
    comp["chcsho"]     = np.log(
        (comp["csho"] / g["csho"].transform(lambda x: x.shift(1))).clip(lower=1e-6)
    )
    comp["sgr"]        = g["sale"].transform(pct_change_prev)
    comp["lgr"]        = g["lt"].transform(pct_change_prev)
    comp["rd_mve"]     = comp["xrd"].fillna(0) / comp["mve"]
    comp["depr"]       = comp["dp"].fillna(0) / comp["ppent"].clip(lower=1)
    comp["secured"]    = comp["dltt"].fillna(0) / comp["at"].clip(lower=1)
    comp["securedind"] = (comp["secured"] > 0).astype(float)
    comp["roic"]       = (
        (comp["ib"].fillna(0) + comp["xint"].fillna(0) * 0.65)
        / (comp["ceq"].fillna(0) + comp["dltt"].fillna(0)).clip(lower=1)
    )
    comp["salerec"]    = comp["sale"] / comp["rect"].clip(lower=1)
    comp["salecash"]   = comp["sale"] / comp["che"].clip(lower=1)
    comp["saleinv"]    = comp["sale"] / comp["invt"].clip(lower=1)
    comp["pchsaleinv"] = g["saleinv"].transform(pct_change_prev)
    comp["currat"]     = comp["act"] / comp["lct"].clip(lower=1)
    comp["quick"]      = (comp["act"] - comp["invt"].fillna(0)) / comp["lct"].clip(lower=1)
    comp["pchquick"]   = g["quick"].transform(pct_change_prev)

    ind_capx        = comp.groupby(["sic2", "datadate"])["capx"].transform("mean")
    comp["cinvest"] = (comp["capx"].fillna(0) - ind_capx.fillna(0)) / comp["sale"].clip(lower=1)

    comp["pchdepr"]           = g["depr"].transform(pct_change_prev)
    comp["gm"]                = comp["gp"] / comp["sale"].clip(lower=1)
    comp["grgma"]             = g["gm"].transform(pct_change_prev)
    comp["pchgm_pchsale"]     = comp["grgma"] - comp["sgr"]
    comp["pchsale_pchxsga"]   = comp["sgr"] - g["xsga"].transform(pct_change_prev)
    comp["pchsale_pchinvt"]   = comp["sgr"] - g["invt"].transform(pct_change_prev)
    comp["pchsale_pchrect"]   = comp["sgr"] - g["rect"].transform(pct_change_prev)

    comp["log_mve"]  = np.log(comp["mve"])
    comp["book_mve"] = comp["ceq"].fillna(0) / comp["mve"]
    comp["cf_p"]     = (comp["ib"].fillna(0) + comp["dp"].fillna(0)) / comp["mve"]

    for raw_col, ia_col in [("book_mve", "bm_ia"), ("log_mve", "mve_ia"), ("cf_p", "cfp_ia")]:
        ind_mean     = comp.groupby(["sic2", "datadate"])[raw_col].transform("mean")
        comp[ia_col] = comp[raw_col] - ind_mean

    comp["pm"]      = comp["ib"].fillna(0) / comp["sale"].clip(lower=1)
    comp["chpm"]    = g["pm"].transform(pct_change_prev)
    ind_mean_chpm   = comp.groupby(["sic2", "datadate"])["chpm"].transform("mean")
    comp["chpmia"]  = comp["chpm"] - ind_mean_chpm

    comp["at_turn"] = comp["sale"] / comp["at"].clip(lower=1)
    comp["chat"]    = g["at_turn"].transform(pct_change_prev)
    ind_mean_chat   = comp.groupby(["sic2", "datadate"])["chat"].transform("mean")
    comp["chatoia"] = comp["chat"] - ind_mean_chat

    comp_annual = comp[["permno", "yyyymm_int"] + annual_char_cols].rename(
        columns={"yyyymm_int": "date"}
    )
    comp_annual.to_parquet(CKPT_ANNUAL, index=False)
    del comp, g
    print(f"  Done: {len(comp_annual):,} rows")


# ═══════════════════════════════════════════════════════════════════
# PART 3: Compustat quarterly — rsup, roavol
# ═══════════════════════════════════════════════════════════════════

quarterly_char_cols = ["rsup", "roavol"]

if CKPT_QTRLY.exists():
    print("\n── Loading Compustat quarterly chars from checkpoint ──")
    comp_quarterly = pd.read_parquet(CKPT_QTRLY)
    print(f"  {len(comp_quarterly):,} rows")
else:
    print("\n── Building Compustat quarterly characteristics ──")
    compq = pd.read_parquet(CKPT_DIR / "comp_fundq_raw.parquet")
    ccm_q = ccm[["gvkey", "permno", "linkdt", "linkenddt"]].copy()

    compq = compq.merge(ccm_q, on="gvkey", how="inner")
    compq = compq[
        (compq["datadate"] >= compq["linkdt"]) &
        (compq["datadate"] <= compq["linkenddt"])
    ]
    compq["use_date"]   = compq["datadate"] + pd.DateOffset(months=4)
    compq["yyyymm_int"] = compq["use_date"].dt.year * 100 + compq["use_date"].dt.month
    compq = compq.sort_values(["permno", "yyyymm_int", "datadate"])
    compq = compq.drop_duplicates(subset=["permno", "yyyymm_int"], keep="last")
    compq = compq.merge(mve_monthly, on=["permno", "yyyymm_int"], how="left")

    gq = compq.groupby("permno")
    compq["saleq_lag4"] = gq["saleq"].transform(lambda x: x.shift(4))
    compq["rsup"]       = (compq["saleq"] - compq["saleq_lag4"]) / compq["mve"].clip(lower=1)
    compq["roa_q"]      = compq["ibq"] / compq["atq"].clip(lower=1)
    compq["roavol"]     = gq["roa_q"].transform(lambda x: x.rolling(8, min_periods=4).std())

    comp_quarterly = compq[["permno", "yyyymm_int"] + quarterly_char_cols].rename(
        columns={"yyyymm_int": "date"}
    )
    comp_quarterly.to_parquet(CKPT_QTRLY, index=False)
    del compq, gq
    print(f"  Done: {len(comp_quarterly):,} rows")


# ═══════════════════════════════════════════════════════════════════
# PART 4: Merge into processed panel and save
# ═══════════════════════════════════════════════════════════════════

print("\n── Merging new characteristics into panel ──")
panel = pd.read_parquet(PROCESSED_DIR / "panel_processed.parquet")
print(f"  Existing panel: {len(panel):,} rows, {len(panel.columns)} cols")

new_char_names = (
    ["mvel1", "turn", "mom36m", "chmom", "betasq"]
    + annual_char_cols
    + quarterly_char_cols
)
panel.drop(columns=[c for c in new_char_names if c in panel.columns], inplace=True)

panel = panel.merge(crsp_chars, on=["permno", "date"], how="left")
panel = panel.merge(comp_annual, on=["permno", "date"], how="left")
panel = panel.sort_values(["permno", "date"])
for col in annual_char_cols:
    if col in panel.columns:
        panel[col] = panel.groupby("permno")[col].transform(lambda x: x.ffill(limit=12))

panel = panel.merge(comp_quarterly, on=["permno", "date"], how="left")
for col in quarterly_char_cols:
    if col in panel.columns:
        panel[col] = panel.groupby("permno")[col].transform(lambda x: x.ffill(limit=4))

actually_added = [c for c in new_char_names if c in panel.columns]
print(f"\n  Added {len(actually_added)} characteristics:")
for c in actually_added:
    pct = panel[c].notna().mean() * 100
    print(f"    {c:28s}  {pct:.1f}% non-missing")

print("\n  Imputing with cross-sectional medians...")
for col in actually_added:
    medians = panel.groupby("date")[col].transform("median")
    panel[col] = panel[col].where(panel[col].notna(), medians)

print("  Rank-normalizing...")
dates_arr = panel["date"].values
for col in actually_added:
    panel[col] = rank_normalize_col(panel[col].values, dates_arr)

panel = panel.dropna(subset=["ret"])
n_chars = len([c for c in panel.columns if c not in ("permno", "date", "ret")])
print(f"\n  Final panel: {len(panel):,} rows, {n_chars} characteristics")

panel.to_parquet(PROCESSED_DIR / "panel_processed.parquet", index=False)
print(f"  Saved to {PROCESSED_DIR / 'panel_processed.parquet'}")
print("\nDone. Resubmit your training SLURM job.")