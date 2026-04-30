"""
Data loading and preprocessing pipeline.

Handles:
- Loading raw stock-month panel (returns + characteristics)
- Rank normalization of characteristics to (-1, 1)
- Missing value imputation (cross-sectional median)
- Characteristic lag alignment
- Train/validation/test splitting with rolling windows
"""

import sys
from pathlib import Path

# Allow imports from the project root (one level up from src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
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

# Mapping from Chen-Zimmermann column names (lowercase) -> GKX names used here.
# Only covers the GKX chars that exist in CZ under a different name.
# Chars absent from CZ entirely are simply dropped (noted at runtime).
CZ_RENAME = {
    "yyyymm":           "date",
    # ── Price trend ──
    "mrreversal":       "mom1m",        # short-term reversal (1-month)
    "mom6m":            "mom6m",
    "mom12m":           "mom12m",
    "momreversal":      "mom36m",       # long-term reversal (36-month)
    "chmom":            "chmom",        # change in 6-month momentum
    "indmom":           "indmom",
    "maxret":           "maxret",
    # ── Liquidity ──
    "turn":             "turn",         # share turnover
    "turnvolatility":   "std_turn",     # turnover volatility
    "std_turn":         "std_turn",
    "dolvol":           "dolvol",
    "illiquidity":      "ill",          # Amihud illiquidity
    "zerotrade":        "zerotrade",
    "zerotrade1m":      "zerotrade",    # alternative CZ name
    "bidaskspread":     "baspread",
    "baspread":         "baspread",
    "volsd":            "std_dolvol",
    "std_dolvol":       "std_dolvol",
    # ── Risk ──
    "realizedvol":      "retvol",       # total return volatility
    "retvol":           "retvol",
    "idiovol":          "idiovol",
    "idiovol3f":        "idiovol",
    "beta":             "beta",
    "betasq":           "betasq",
    # ── Value ──
    "bm":               "bm",
    "bm_ia":            "bm_ia",        # book-to-market industry adjusted
    "bmia":             "bm_ia",
    "ep":               "ep",
    "cfp":              "cfp",
    "cfp_ia":           "cfp_ia",
    "cfpia":            "cfp_ia",
    "leverage":         "lev",
    "lev":              "lev",
    "sp":               "sp",           # sales-to-price
    "cashprod":         "cashpr",
    "cashpr":           "cashpr",
    "divyield":         "dy",
    "divyieldst":       "dy",
    "dy":               "dy",
    # ── Profitability ──
    "roaq":             "roaq",
    "roic":             "roic",
    "roe":              "roeq",
    "roeq":             "roeq",
    "salerec":          "salerec",      # sales / receivables
    "salecash":         "salecash",
    "saleinv":          "saleinv",
    "pchsaleinv":       "pchsaleinv",   # pct change sales/inventory
    "operprof":         "operprof",
    "ps":               "ps",           # financial statements score
    "gp":               "gma",          # gross profitability
    "gma":              "gma",
    "rd_sale":          "rd_sale",
    "rds":              "rd_sale",
    "cashdebt":         "cashdebt",
    "cf":               "cashdebt",     # alt CZ name: cash flow to debt
    # ── Investment ──
    "assetgrowth":      "agr",
    "agr":              "agr",
    "investment":       "invest",
    "invest":           "invest",
    "egr":              "egr",          # equity growth
    "grcapx":           "grcapx",
    "grgma":            "grgma",        # growth in gross margin (CZ may use GrGMA)
    "chcsho":           "chcsho",       # change in shares outstanding
    "chinv":            "chinv",
    "chinvia":          "pchcapx_ia",
    "pchcapx_ia":       "pchcapx_ia",
    "hire":             "hire",
    # ── Intangibles ──
    "orgcap":           "orgcap",
    "rd":               "rd",
    "rdmve":            "rd_mve",       # R&D to market equity
    "rd_mve":           "rd_mve",
    "accruals":         "acc",
    "acc":              "acc",
    "abnormalaccruals":  "absacc",
    "absacc":           "absacc",
    "pctacc":           "pctacc",
    # ── Trading friction / earnings ──
    "numearnincrease":  "nincr",
    "nincr":            "nincr",
    "varcf":            "stdcf",
    "stdcf":            "stdcf",
    "convdebt":         "convind",
    "convind":          "convind",
    "announcementreturn": "ear",
    "ear":              "ear",
    "ms":               "ms",           # financial statements score
    "pricedelayslope":  "pricedelay",
    "pricedelay":       "pricedelay",
    "firmage":          "age",
    "age":              "age",
    "revenuessurprise": "rsup",
    "rsup":             "rsup",
    "sinalgo":          "sin",          # sin stock indicator
    "sin":              "sin",
    # ── Size ──
    "mve_ia":           "mve_ia",       # market equity industry adjusted
    "mveia":            "mve_ia",
    "chpmia":           "chpmia",       # change in profit margin (IA)
    "mvel1":            "mvel1",        # log market equity
    # ── Leverage / solvency ──
    "secured":          "secured",
    "securedind":       "securedind",
    "depr":             "depr",
    "sgr":              "sgr",          # sales growth
    "salesgrowth":      "sgr",
    # ── Miscellaneous ──
    "aeavol":           "aeavol",       # abnormal earnings announcement volume
    "cash":             "cash",
    "chtax":            "chtx",
    "chtx":             "chtx",
    "cinvest":          "cinvest",      # corporate investment
    "currat":           "currat",       # current ratio
    "herf":             "herf",         # industry Herfindahl
    "lgr":              "lgr",          # liability growth
    "pchdepr":          "pchdepr",
    "pchgm_pchsale":    "pchgm_pchsale",
    "pchquick":         "pchquick",
    "pchsale_pchxsga":  "pchsale_pchxsga",
    "pchsale_pchinvt":  "pchsale_pchinvt",
    "pchsale_pchrect":  "pchsale_pchrect",
    "quick":            "quick",
    "realestate":       "realestate",
    "roavol":           "roavol",
    "tang":             "tang",
    "chassetturnoever": "chatoia",
    "chatoia":          "chatoia",
    "divinit":          "divi",
    "divi":             "divi",
    "divomit":          "divo",
    "divo":             "divo",
    "grltnoa":          "grltnoa",
    "salerecm":         "salerecm",     # not in CZ but keep for completeness
    "tax":              "tb",
    "tb":               "tb",
}

# The 94 characteristics from Gu, Kelly, Xiu (2020), using GKX names.
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
    "agr", "invest", "egr", "grcapx", "grgma", "chcsho", "chinv",
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
    Load the raw stock-month panel, reading only the columns we need.

    Supports the Chen-Zimmermann "signed_predictors_dl_wide" format where:
      - date column is named 'yyyymm' (renamed to 'date')
      - returns ('ret') are absent — rows will have ret=NaN; merge returns
        separately before calling build_panel, or build_panel will drop them.
      - characteristic names differ from GKX — CZ_RENAME handles the mapping.

    For any other CSV/parquet, columns are lowercased and matched as-is.
    """
    print(f"Loading raw data from {filepath}...")

    # Build the set of CZ source columns we want to load
    # (CZ name -> GKX name for all chars + permno/date/ret)
    cz_to_gkx = {k: v for k, v in CZ_RENAME.items()}
    # Inverse: gkx -> cz (for chars that share the same name)
    gkx_names = set(GKX_CHARACTERISTICS)
    # All CZ source names to keep
    needed_cz = set(cz_to_gkx.keys())
    # Also keep any column whose lowercase name is already a GKX name
    # (will be resolved after loading)

    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower().str.strip()
    elif filepath.endswith(".csv"):
        # Read only the header to determine which columns exist (preserve original case)
        header = pd.read_csv(filepath, nrows=0)
        orig_cols = list(header.columns)                          # original case
        lower_cols = [c.lower().strip() for c in orig_cols]      # lowercase version
        lower_to_orig = dict(zip(lower_cols, orig_cols))          # lowercase -> original

        keep_lower = set()
        keep_lower.add("permno")
        for c in ("date", "yyyymm", "ret"):
            if c in lower_cols:
                keep_lower.add(c)
        for c in needed_cz:
            if c in lower_cols:
                keep_lower.add(c)
        for c in lower_cols:
            if c in gkx_names:
                keep_lower.add(c)

        # usecols must use the original case names as they appear in the file
        usecols = [lower_to_orig[c] for c in lower_cols if c in keep_lower]
        print(f"  Reading {len(usecols)} of {len(orig_cols)} columns...")
        df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    # Apply CZ -> GKX renames
    df = df.rename(columns=cz_to_gkx)

    # If no 'ret' column, add a NaN placeholder so the pipeline can run;
    # caller must merge in returns before splitting/training.
    if COL_RETURN not in df.columns:
        print("  WARNING: no 'ret' column found — ret set to NaN. "
              "Merge CRSP returns before training.")
        df[COL_RETURN] = np.nan

    # Ensure date is integer YYYYMM
    if df[COL_DATE].dtype == object or str(df[COL_DATE].dtype) == "datetime64[ns]":
        df[COL_DATE] = pd.to_datetime(df[COL_DATE]).dt.strftime("%Y%m").astype(int)
    else:
        df[COL_DATE] = df[COL_DATE].astype(int)

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
        print(f"  WARNING: {len(missing)} characteristics not found after renaming:")
        for m in missing:
            print(f"    - {m}")
    
    print(f"  Using {len(available)} of {len(GKX_CHARACTERISTICS)} characteristics")
    
    keep_cols = [COL_PERMNO, COL_DATE, COL_RETURN] + available
    return df[keep_cols].copy(), available


def impute_missing(df: pd.DataFrame, char_cols: List[str]) -> pd.DataFrame:
    """
    Replace missing characteristics with cross-sectional median for that month.
    This matches the paper's approach (Section 3.1).

    Vectorized: compute all column medians per month in one groupby, then
    fill NaNs from the resulting median matrix.
    """
    print("  Imputing missing values with cross-sectional medians...")

    char_df = df[char_cols]
    medians = char_df.groupby(df[COL_DATE]).transform("median")
    df = df.copy()
    df[char_cols] = char_df.where(char_df.notna(), medians)

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

    Vectorized: all columns are ranked in one groupby pass using scipy,
    which is ~50x faster than looping column-by-column with transform.
    """
    from scipy.stats import rankdata

    print("  Rank-normalizing characteristics to (-1, 1)...")

    char_arr = df[char_cols].values.astype(np.float64)  # (N, P)
    dates = df[COL_DATE].values
    result = np.empty_like(char_arr)

    for date in np.unique(dates):
        mask = dates == date
        block = char_arr[mask]          # (n_stocks, P)
        n = block.shape[0]
        if n <= 1:
            result[mask] = 0.0
            continue
        # rankdata over axis=0 handles NaN implicitly via nan_policy
        ranked = np.apply_along_axis(
            lambda col: rankdata(col, method="average", nan_policy="propagate"), 0, block
        )
        result[mask] = 2.0 * ranked / (n + 1) - 1.0

    df = df.copy()
    df[char_cols] = result
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


def build_panel(
    filepath: Optional[str] = None,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full preprocessing pipeline: load -> select chars -> lag -> impute -> normalize.

    If filepath is None, searches config.RAW_DIR for a .parquet or .csv file.
    If save_processed is True, saves the cleaned panel to config.PROCESSED_DIR.

    Returns:
        df: Cleaned panel with columns [permno, date, ret, char1, ..., charP]
            where chars at each row are lagged (known at t-1) and rank-normalized.
        char_cols: List of characteristic column names.
    """
    if filepath is None:
        raw_dir = Path(config.RAW_DIR)
        candidates = list(raw_dir.glob("*.parquet")) + list(raw_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No .parquet or .csv file found in {raw_dir}")
        filepath = str(candidates[0])
        if len(candidates) > 1:
            print(f"  WARNING: multiple raw files found, using {filepath}")

    df = load_raw_panel(filepath)

    # Merge in returns and risk-free rate if not already present
    if df[COL_RETURN].isna().all():
        returns_path = Path(config.RAW_DIR) / "returns.csv"
        if returns_path.exists():
            print(f"  Merging returns from {returns_path}...")
            returns = pd.read_csv(returns_path, usecols=['permno', 'yyyymm', 'ret_excess'])
            returns = returns.rename(columns={'yyyymm': COL_DATE, 'ret_excess': COL_RETURN})
            # Drop the NaN placeholder before merging to avoid collision
            df = df.drop(columns=[COL_RETURN])
            df = df.merge(returns[['permno', COL_DATE, COL_RETURN]],
                          on=[COL_PERMNO, COL_DATE], how='inner')
        else:
            raise FileNotFoundError(
                f"No returns found. Run src/api_caller.py to generate {returns_path}"
            )

    df, char_cols = select_characteristics(df)
    df = align_characteristic_lags(df, char_cols)
    df = impute_missing(df, char_cols)
    df = rank_normalize(df, char_cols)

    # Final cleanup: drop any remaining NaN rows
    df = df.dropna()

    print(f"\n  Final panel: {len(df):,} stock-months, "
          f"{df[COL_PERMNO].nunique():,} stocks, "
          f"{df[COL_DATE].nunique()} months")

    if save_processed:
        processed_dir = Path(config.PROCESSED_DIR)
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "panel_processed.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved processed panel to {out_path}")

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
    """
    Run with:
        python src/data_loader.py                         # build panel from auto-detected file
        python src/data_loader.py path/to/raw.csv         # build from specific file
        python src/data_loader.py --diagnose path/to/raw.csv  # just check column mapping
    """
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--diagnose":
        # Diagnostic mode: show which CZ columns map to GKX chars and which don't
        filepath = sys.argv[2]
        header = pd.read_csv(filepath, nrows=0)
        cz_cols_lower = set(c.lower().strip() for c in header.columns)

        gkx_set = set(GKX_CHARACTERISTICS)
        cz_to_gkx_lower = {k.lower(): v for k, v in CZ_RENAME.items()}

        matched = {}     # gkx_name -> cz_source_name
        unmatched = []   # gkx names with no mapping

        for gkx in GKX_CHARACTERISTICS:
            # Direct match (column already named as GKX)
            if gkx in cz_cols_lower:
                matched[gkx] = gkx
                continue
            # Check CZ_RENAME inverse: find a CZ key that maps to this GKX name
            found = False
            for cz_key, gkx_val in cz_to_gkx_lower.items():
                if gkx_val == gkx and cz_key in cz_cols_lower:
                    matched[gkx] = cz_key
                    found = True
                    break
            if not found:
                unmatched.append(gkx)

        print(f"\n  CZ file has {len(header.columns)} columns")
        print(f"  GKX characteristics matched: {len(matched)} / {len(GKX_CHARACTERISTICS)}")
        print(f"\n  ── Matched ({len(matched)}) ──")
        for gkx, cz in sorted(matched.items()):
            tag = "" if gkx == cz else f"  (CZ: {cz})"
            print(f"    {gkx}{tag}")
        print(f"\n  ── Unmatched ({len(unmatched)}) ──")
        for gkx in unmatched:
            print(f"    {gkx}")

        # Show CZ columns that weren't matched to anything (potential candidates)
        used_cz = set(matched.values()) | set(cz_to_gkx_lower.keys())
        unused_cz = sorted(cz_cols_lower - used_cz - {"permno", "yyyymm", "date", "ret"})
        print(f"\n  ── Unused CZ columns ({len(unused_cz)}) ──")
        for c in unused_cz[:30]:
            print(f"    {c}")
        if len(unused_cz) > 30:
            print(f"    ... and {len(unused_cz) - 30} more")
    else:
        filepath = sys.argv[1] if len(sys.argv) > 1 else None
        df, char_cols = build_panel(filepath, save_processed=True)
        train, val, test = split_panel(df, train_end=197412, val_end=198612)
        returns, chars, mask, dates = prepare_tensors(train, char_cols)
        print(f"\n  Tensor shapes: returns {returns.shape}, chars {chars.shape}, mask {mask.shape}")