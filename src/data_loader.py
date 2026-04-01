"""
Data loading and preprocessing pipeline.

Handles:
- Loading raw stock-month panel (returns + characteristics)
- Rank normalization of characteristics to (-1, 1)
- Missing value imputation (cross-sectional median)
- Characteristic lag alignment
- Train/validation/test splitting with rolling windows
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

import config


# ─────────────────────────────────────────────────────────────────
# Core column names — adjust these to match your dataset
# ─────────────────────────────────────────────────────────────────

# If using Chen-Zimmermann (openassetpricing.com):
#   - download "Firm Level Characteristics / CSV / Full Sets"
#   - the file has columns: permno, date (YYYYMM), ret, and ~200 characteristics
#
# If using WRDS directly:
#   - merge CRSP msf (monthly stock file) with Compustat funda/fundq
#   - construct characteristics following Gu, Kelly, Xiu (2020) Appendix

COL_PERMNO = "permno"       # stock identifier
COL_DATE = "date"           # YYYYMM integer
COL_RETURN = "ret"          # monthly excess return (already in excess of rf)

# The 94 characteristics from Gu, Kelly, Xiu (2020).
# Below is the full list. Your dataset may use slightly different names —
# map them here. Characteristics you can't find can be dropped; the model
# still works with fewer, just note it in your report.
GKX_CHARACTERISTICS = [
    # Price trend
    "mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom", "maxret",
    # Liquidity
    "turn", "std_turn", "mvel1", "dolvol", "ill", "zerotrade", "baspread",
    # Risk
    "retvol", "idiovol", "beta", "betasq",
    # Value
    "bm", "bm_ia", "ep", "cashpr", "dy", "lev", "sp", "roaq",
    # Profitability
    "roic", "roeq", "salerec", "salecash", "saleinv", "pchsaleinv",
    "cashdebt", "rd_sale", "operprof", "ps",
    # Investment
    "agr", "invest", "egr", "grcapx", "grGMA", "chcsho", "chinv",
    "pchcapx_ia", "hire",
    # Intangibles
    "orgcap", "rd", "rd_mve", "acc", "absacc", "pctacc",
    # Trading friction
    "nincr", "stdcf", "std_dolvol", "convind", "ear", "ms",
    "pricedelay", "age",
    # Size
    "mve_ia", "chpmia",
    # Leverage / solvency
    "secured", "securedind", "depr", "sgr",
    # Miscellaneous / other
    "aeavol", "cash", "chtx", "cinvest", "currat", "herf", "lgr",
    "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchxsga",
    "pchsale_pchinvt", "pchsale_pchrect", "quick", "realestate",
    "roavol", "rsup", "sin", "cfp", "cfp_ia", "chatoia",
    "divi", "divo", "gma", "grltnoa", "salerecm", "tang", "tb",
]

# Classify update frequency for lag convention
MONTHLY_CHARS = [
    "mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom", "maxret",
    "turn", "std_turn", "dolvol", "ill", "zerotrade", "baspread",
    "retvol", "idiovol", "beta", "betasq", "mvel1", "std_dolvol",
    "pricedelay",
]
QUARTERLY_CHARS = [
    "roaq", "roeq", "salerec", "salecash", "saleinv", "pchsaleinv",
    "cashdebt", "stdcf", "ear", "rsup", "cfp",
    "nincr", "ms",
]
# Everything else is annual


def load_raw_panel(filepath: str) -> pd.DataFrame:
    """
    Load the raw stock-month panel.
    
    Expected format: CSV with columns [permno, date, ret, char1, char2, ...].
    Date should be integer YYYYMM or convertible to one.
    Returns should already be excess returns. If not, subtract the risk-free
    rate here (merge with FF rf from Ken French's site).
    """
    print(f"Loading raw data from {filepath}...")
    
    # Detect format
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure date is integer YYYYMM
    if df[COL_DATE].dtype == "object" or df[COL_DATE].dtype == "<M8[ns]":
        df[COL_DATE] = pd.to_datetime(df[COL_DATE]).dt.strftime("%Y%m").astype(int)
    
    print(f"  Raw panel: {len(df):,} stock-months, "
          f"{df[COL_PERMNO].nunique():,} unique stocks, "
          f"{df[COL_DATE].nunique()} months")
    
    return df


def select_characteristics(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the subset of GKX characteristics present in the dataset.
    Returns the filtered dataframe and the list of available characteristic names.
    """
    available = [c for c in GKX_CHARACTERISTICS if c in df.columns]
    missing = [c for c in GKX_CHARACTERISTICS if c not in df.columns]
    
    if missing:
        print(f"  WARNING: {len(missing)} characteristics not found in dataset:")
        print(f"    {missing[:10]}{'...' if len(missing) > 10 else ''}")
    
    print(f"  Using {len(available)} of {len(GKX_CHARACTERISTICS)} characteristics")
    
    keep_cols = [COL_PERMNO, COL_DATE, COL_RETURN] + available
    return df[keep_cols].copy(), available


def impute_missing(df: pd.DataFrame, char_cols: List[str]) -> pd.DataFrame:
    """
    Replace missing characteristics with cross-sectional median for that month.
    This matches the paper's approach (Section 3.1).
    """
    print("  Imputing missing values with cross-sectional medians...")
    
    for col in char_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df.groupby(COL_DATE)[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Drop any remaining rows with NaN returns
    before = len(df)
    df = df.dropna(subset=[COL_RETURN])
    print(f"  Dropped {before - len(df)} rows with missing returns")
    
    return df


def rank_normalize(df: pd.DataFrame, char_cols: List[str]) -> pd.DataFrame:
    """
    Rank-normalize characteristics to (-1, 1) cross-sectionally each month.
    
    Following the paper: rank / (n+1) * 2 - 1, where n is the number of
    non-missing observations for that characteristic in that month.
    This maps the smallest value to near -1 and the largest to near +1.
    """
    print("  Rank-normalizing characteristics to (-1, 1)...")
    
    def _rank_norm(x: pd.Series) -> pd.Series:
        ranked = x.rank(method="average")
        n = x.notna().sum()
        if n <= 1:
            return x * 0.0  # degenerate case
        return 2.0 * ranked / (n + 1) - 1.0
    
    for col in char_cols:
        df[col] = df.groupby(COL_DATE)[col].transform(_rank_norm)
    
    return df


def align_characteristic_lags(
    df: pd.DataFrame, char_cols: List[str]
) -> pd.DataFrame:
    """
    Align characteristics with their appropriate lags.
    
    The paper matches returns at month t with:
    - Monthly chars: most recent at t-1
    - Quarterly chars: most recent at t-4  
    - Annual chars: most recent at t-6
    
    In practice, if your dataset already aligns chars with the month they
    were known (as Chen-Zimmermann does), you just need to lag by 1 month
    for all characteristics. The extra quarterly/annual lags are to account
    for publication delay, which Chen-Zimmermann already handles.
    """
    
    print("  Aligning characteristic lags...")
    
    # Sort for proper lagging
    df = df.sort_values([COL_PERMNO, COL_DATE])
    
    # For pre-processed datasets (Chen-Zimmermann), a single 1-month lag suffices.
    # The characteristics at date t are used to predict returns at date t+1.
    # We implement this by shifting characteristics forward by 1 month within
    # each stock, so that row (permno, t) has chars from t-1 and return from t.
    
    # Create lagged characteristics
    for col in char_cols:
        df[col] = df.groupby(COL_PERMNO)[col].shift(1)
    
    # Drop the first observation per stock (no lagged chars available)
    df = df.dropna(subset=char_cols, how="all")
    
    return df


def build_panel(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full preprocessing pipeline: load -> select chars -> lag -> impute -> normalize.
    
    Returns:
        df: Cleaned panel with columns [permno, date, ret, char1, ..., charP]
            where chars at each row are lagged (known at t-1) and rank-normalized.
        char_cols: List of characteristic column names.
    """
    df = load_raw_panel(filepath)
    df, char_cols = select_characteristics(df)
    df = align_characteristic_lags(df, char_cols)
    df = impute_missing(df, char_cols)
    df = rank_normalize(df, char_cols)
    
    # Final cleanup: drop any remaining NaN rows
    df = df.dropna()
    
    print(f"\n  Final panel: {len(df):,} stock-months, "
          f"{df[COL_PERMNO].nunique():,} stocks, "
          f"{df[COL_DATE].nunique()} months")
    
    return df, char_cols


def split_panel(
    df: pd.DataFrame,
    train_end: int,
    val_end: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split panel into train/validation/test by date.
    
    Args:
        train_end: Last YYYYMM in training set
        val_end: Last YYYYMM in validation set
        Everything after val_end is test.
    """
    train = df[df[COL_DATE] <= train_end].copy()
    val = df[(df[COL_DATE] > train_end) & (df[COL_DATE] <= val_end)].copy()
    test = df[df[COL_DATE] > val_end].copy()
    
    print(f"  Train: {len(train):,} obs ({train[COL_DATE].min()}-{train[COL_DATE].max()})")
    print(f"  Val:   {len(val):,} obs ({val[COL_DATE].min()}-{val[COL_DATE].max()})")
    print(f"  Test:  {len(test):,} obs ({test[COL_DATE].min()}-{test[COL_DATE].max()})")
    
    return train, val, test


def get_rolling_splits(
    df: pd.DataFrame,
    initial_train_end: int = config.TRAIN_END_INIT,
    val_years: int = config.VAL_YEARS,
    test_start: int = config.TEST_START,
    test_end: int = config.TEST_END,
    refit_freq: int = config.REFIT_FREQ,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, int]]:
    """
    Generate rolling train/val splits for annual refitting.
    
    The paper expands training by 1 year each refit, rolls validation forward.
    
    Returns list of (train_df, val_df, test_year_start, test_year_end) tuples.
    """
    all_dates = sorted(df[COL_DATE].unique())
    
    # Convert test_start/test_end to indices in date list
    test_dates = [d for d in all_dates if test_start <= d <= test_end]
    
    splits = []
    current_train_end = initial_train_end
    
    for year_offset in range(0, len(test_dates), refit_freq):
        if year_offset + refit_freq > len(test_dates):
            break
        
        test_block_start = test_dates[year_offset]
        test_block_end = test_dates[min(year_offset + refit_freq - 1, len(test_dates) - 1)]
        
        # Validation: val_years * 12 months before test_block_start
        val_dates = [d for d in all_dates if d < test_block_start]
        val_start_idx = max(0, len(val_dates) - val_years * 12)
        val_period = val_dates[val_start_idx:]
        
        if len(val_period) == 0:
            continue
        
        val_start = val_period[0]
        val_end = val_period[-1]
        
        # Training: everything before validation
        train = df[df[COL_DATE] < val_start]
        val = df[(df[COL_DATE] >= val_start) & (df[COL_DATE] <= val_end)]
        
        if len(train) > 0 and len(val) > 0:
            splits.append((train, val, test_block_start, test_block_end))
    
    print(f"  Generated {len(splits)} rolling train/val splits")
    return splits


def prepare_tensors(
    df: pd.DataFrame,
    char_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a panel DataFrame to numpy arrays suitable for model input.
    
    Returns:
        returns: (T, N_max) padded return matrix (NaN where stock absent)
        chars: (T, N_max, P) padded characteristic tensor
        mask: (T, N_max) boolean mask (True where stock present)
        dates: (T,) array of dates
    """
    dates = sorted(df[COL_DATE].unique())
    T = len(dates)
    P = len(char_cols)
    
    # Find max stocks in any month
    counts = df.groupby(COL_DATE)[COL_PERMNO].count()
    N_max = counts.max()
    
    returns = np.full((T, N_max), np.nan)
    chars = np.full((T, N_max, P), np.nan)
    mask = np.zeros((T, N_max), dtype=bool)
    
    for t, date in enumerate(dates):
        month_df = df[df[COL_DATE] == date]
        n = len(month_df)
        
        returns[t, :n] = month_df[COL_RETURN].values
        chars[t, :n, :] = month_df[char_cols].values
        mask[t, :n] = True
    
    return returns, chars, mask, np.array(dates)


if __name__ == "__main__":
    """Quick test: run the pipeline on your data file."""
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else config.RAW_DIR + "panel.csv"
    
    df, char_cols = build_panel(filepath)
    train, val, test = split_panel(df, train_end=197412, val_end=198612)
    
    returns, chars, mask, dates = prepare_tensors(train, char_cols)
    print(f"\n  Tensor shapes: returns {returns.shape}, chars {chars.shape}, mask {mask.shape}")
