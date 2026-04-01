"""
Managed portfolio construction following Eq. (16) of Gu, Kelly, Xiu (2021).

x_t = (Z_{t-1}' Z_{t-1})^{-1} Z_{t-1}' r_t

Each element of x_t is a characteristic-managed long-short portfolio return.
We also append an equal-weighted market portfolio (corresponding to a
constant column in Z), yielding P+1 managed portfolios.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from src.data_loader import COL_PERMNO, COL_DATE, COL_RETURN


def compute_managed_portfolios_single_month(
    returns: np.ndarray,
    chars: np.ndarray,
    regularize: float = 1e-6,
) -> np.ndarray:
    """
    Compute managed portfolio returns for a single month.
    
    Args:
        returns: (N,) vector of stock returns at time t
        chars: (N, P) matrix of characteristics at time t-1
            (already lagged in the data pipeline)
        regularize: Ridge regularization for (Z'Z) inversion stability
    
    Returns:
        x_t: (P+1,) vector of managed portfolio returns
            First P elements: characteristic-managed portfolios
            Last element: equal-weighted market portfolio
    """
    N, P = chars.shape
    
    # Z'Z with ridge regularization for numerical stability
    ZtZ = chars.T @ chars + regularize * np.eye(P)
    
    # x_t = (Z'Z)^{-1} Z' r_t
    # Solve via Cholesky for speed and stability
    try:
        L = np.linalg.cholesky(ZtZ)
        Ztr = chars.T @ returns
        x_char = np.linalg.solve(L.T, np.linalg.solve(L, Ztr))
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if Cholesky fails
        x_char = np.linalg.lstsq(ZtZ, chars.T @ returns, rcond=None)[0]
    
    # Equal-weighted market portfolio
    x_mkt = np.mean(returns)
    
    # Concatenate: (P+1,) vector
    x_t = np.append(x_char, x_mkt)
    
    return x_t


def compute_managed_portfolios_panel(
    df: pd.DataFrame,
    char_cols: List[str],
    regularize: float = 1e-6,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute managed portfolio returns for the full panel.
    
    Args:
        df: Panel with columns [permno, date, ret, char1, ..., charP].
            Characteristics should already be lagged and rank-normalized.
        char_cols: List of characteristic column names.
        regularize: Ridge parameter for (Z'Z) inversion.
    
    Returns:
        mp_df: DataFrame with columns [date, x_1, ..., x_P, x_mkt]
        mp_array: (T, P+1) numpy array of managed portfolio returns
    """
    dates = sorted(df[COL_DATE].unique())
    P = len(char_cols)
    T = len(dates)
    
    mp_array = np.zeros((T, P + 1))
    
    for t, date in enumerate(dates):
        month = df[df[COL_DATE] == date]
        
        ret = month[COL_RETURN].values
        Z = month[char_cols].values
        
        # Drop stocks with any NaN in characteristics this month
        valid = ~np.any(np.isnan(Z), axis=1) & ~np.isnan(ret)
        ret = ret[valid]
        Z = Z[valid, :]
        
        if len(ret) < P + 10:
            # Too few stocks to estimate — use pseudoinverse
            mp_array[t, :P] = np.linalg.lstsq(Z, ret, rcond=None)[0]
            mp_array[t, P] = np.mean(ret)
        else:
            mp_array[t, :] = compute_managed_portfolios_single_month(
                ret, Z, regularize
            )
    
    # Build DataFrame
    mp_cols = [f"x_{c}" for c in char_cols] + ["x_mkt"]
    mp_df = pd.DataFrame(mp_array, columns=mp_cols)
    mp_df.insert(0, COL_DATE, dates)
    
    print(f"  Managed portfolios: {T} months × {P+1} portfolios")
    print(f"  Mean abs return: {np.mean(np.abs(mp_array)):.4f}")
    print(f"  Std of portfolios: {np.std(mp_array, axis=0).mean():.4f}")
    
    return mp_df, mp_array


def compute_managed_portfolios_from_tensors(
    returns: np.ndarray,
    chars: np.ndarray,
    mask: np.ndarray,
    regularize: float = 1e-6,
) -> np.ndarray:
    """
    Compute managed portfolios from pre-built tensor format.
    
    Args:
        returns: (T, N_max) return matrix (NaN where absent)
        chars: (T, N_max, P) characteristic tensor (NaN where absent)
        mask: (T, N_max) boolean mask
        regularize: Ridge parameter
    
    Returns:
        mp: (T, P+1) managed portfolio return array
    """
    T, N_max = returns.shape
    P = chars.shape[2]
    mp = np.zeros((T, P + 1))
    
    for t in range(T):
        valid = mask[t]
        ret_t = returns[t, valid]
        Z_t = chars[t, valid, :]
        
        # Additional NaN check
        finite = np.all(np.isfinite(Z_t), axis=1) & np.isfinite(ret_t)
        ret_t = ret_t[finite]
        Z_t = Z_t[finite, :]
        
        if len(ret_t) > P:
            mp[t, :] = compute_managed_portfolios_single_month(
                ret_t, Z_t, regularize
            )
        else:
            mp[t, :P] = np.linalg.lstsq(Z_t, ret_t, rcond=None)[0]
            mp[t, P] = np.mean(ret_t)
    
    return mp


if __name__ == "__main__":
    """Quick test with synthetic data."""
    np.random.seed(42)
    
    N, P, T = 500, 10, 60
    
    # Simulate characteristics and returns
    Z = np.random.randn(N, P)
    true_betas = np.random.randn(P, 3) * 0.5
    true_factors = np.random.randn(T, 3) * 0.02 + 0.005
    
    print("Testing managed portfolio construction...")
    for t in range(T):
        # Simulated returns: r = Z @ beta @ f + noise
        betas = Z @ true_betas[:, :3]  # (N, 3)
        r = betas @ true_factors[t, :] + np.random.randn(N) * 0.1
        
        x = compute_managed_portfolios_single_month(r, Z)
        assert x.shape == (P + 1,), f"Expected shape ({P+1},), got {x.shape}"
    
    print(f"  All {T} months computed successfully, shape per month: ({P+1},)")
