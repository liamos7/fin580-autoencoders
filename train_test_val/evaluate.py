"""
Evaluation metrics for Autoencoder Asset Pricing Models.

Implements Tables 1-4 from the paper:
- R²_total: contemporaneous factor explanatory power (Eq. 20)
- R²_pred: predictive power via lagged conditional betas (Eq. 21)
- Long-short portfolio Sharpe ratios (Table 3)
- Factor tangency portfolio Sharpe ratios (Table 4)
- Pricing errors / alphas (Fig. 3)
- Variable importance (Fig. 4)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from train_test_val.models import ConditionalAutoencoder
from train_test_val.train import AssetPricingDataset, ensemble_predict_month


# ─────────────────────────────────────────────────────────────────
# Factor history collection (for seeding the test-period λ̂)
# ─────────────────────────────────────────────────────────────────

def collect_factor_history(
    models: list,
    dataset: AssetPricingDataset,
) -> List[np.ndarray]:
    """
    Extract factor estimates f̂_t for every month in a dataset.

    Call this on the TRAINING dataset after training is complete.
    The returned list can then be passed as factor_history to
    evaluate_model, compute_long_short_sharpe, and the backtest
    so that λ̂ is well-estimated from the very first test month.
    """
    factors = []
    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            continue
        _, _, f_t = ensemble_predict_month(models, chars_t, mp_t)
        factors.append(f_t.cpu().numpy())
    return factors


# ─────────────────────────────────────────────────────────────────
# Core R² metrics
# ─────────────────────────────────────────────────────────────────

def compute_r2_total(
    models: list,
    dataset: AssetPricingDataset,
) -> float:
    """
    Out-of-sample total R² (Eq. 20):
    
    R²_total = 1 - Σ(r_{i,t} - β̂'_{i,t-1} f̂_t)² / Σ r²_{i,t}
    
    Measures how well contemporaneous factors explain return variation.
    """
    ss_res = 0.0
    ss_tot = 0.0
    
    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            continue
        
        r_hat, _, _ = ensemble_predict_month(models, chars_t, mp_t)
        
        ss_res += ((returns_t - r_hat) ** 2).sum().item()
        ss_tot += (returns_t ** 2).sum().item()
    
    return 1.0 - ss_res / max(ss_tot, 1e-10)


def compute_r2_pred(
    models: list,
    dataset: AssetPricingDataset,
    factor_history: Optional[np.ndarray] = None,
) -> float:
    """
    Out-of-sample predictive R² (Eq. 21):
    
    R²_pred = 1 - Σ(r_{i,t} - β̂'_{i,t-1} λ̂_{t-1})² / Σ r²_{i,t}
    
    where λ̂_{t-1} is the prevailing mean of estimated factors up to t-1.
    
    Measures how well the model predicts expected returns cross-sectionally.
    """
    # First pass: collect all estimated factors
    all_factors = []
    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            all_factors.append(None)
            continue
        _, _, f_t = ensemble_predict_month(models, chars_t, mp_t)
        all_factors.append(f_t.cpu().numpy())
    
    # Prepend any historical factors from training period
    if factor_history is not None:
        historical = [factor_history[i] for i in range(len(factor_history))]
    else:
        historical = []
    
    # Second pass: compute predictive errors using prevailing mean
    ss_res = 0.0
    ss_tot = 0.0
    
    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            continue
        
        # Prevailing mean of factors up to t-1
        past_factors = historical + [f for f in all_factors[:t] if f is not None]
        if len(past_factors) == 0:
            continue
        
        lambda_prev = np.mean(past_factors, axis=0)
        lambda_prev = torch.tensor(lambda_prev, dtype=torch.float32, device=chars_t.device)
        
        # Predicted return: β̂'_{i,t-1} λ̂_{t-1}
        r_hat_pred, beta, _ = ensemble_predict_month(models, chars_t, mp_t)
        # Override r_hat with beta @ lambda (not beta @ f_t)
        r_hat_pred = (beta * lambda_prev.unsqueeze(0)).sum(dim=1)
        
        ss_res += ((returns_t - r_hat_pred) ** 2).sum().item()
        ss_tot += (returns_t ** 2).sum().item()
    
    return 1.0 - ss_res / max(ss_tot, 1e-10)


# ─────────────────────────────────────────────────────────────────
# Sharpe ratios
# ─────────────────────────────────────────────────────────────────

def compute_long_short_sharpe(
    models: list,
    dataset: AssetPricingDataset,
    returns_panel: np.ndarray = None,
    chars_panel: np.ndarray = None,
    mask_panel: np.ndarray = None,
    mp_panel: np.ndarray = None,
    n_deciles: int = 10,
    value_weight: bool = False,
    mvel_idx: Optional[int] = None,
    return_series: bool = False,
    predictive: bool = True,
    factor_history: Optional[List[np.ndarray]] = None,
    lambda_window: int = 60,
) -> float:
    """
    Long-short decile spread portfolio Sharpe ratio (Table 3).

    Sort stocks into deciles by predicted return each month.
    Buy top decile (10), sell bottom decile (1). Compute annualized SR.

    When predictive=True (default), uses a ROLLING window of past factors
    λ̂_{t-1} instead of contemporaneous factors f̂_t to form predictions.
    This gives a genuinely tradable Sharpe ratio: the sorting signal at t
    uses only information available at t-1.

    Parameters
    ----------
    factor_history : list of np.ndarray, optional
        Factor estimates from the training period. Prepended to the test
        factors so that λ̂ is well-estimated even at the start of the test
        set. Without this, the first ~24 test months use a noisy λ̂.
    lambda_window : int
        Rolling window (in months) for estimating λ̂. Default 60 (5 years).
        This prevents λ̂ from locking in and ensures the sorting signal
        has genuine time variation. Set to 0 for expanding window.
    """
    monthly_returns = []

    # If predictive, first collect all factor estimates
    if predictive:
        all_factors = []
        for t in range(dataset.T):
            chars_t, returns_t, mp_t = dataset.get_month_data(t)
            if len(returns_t) == 0:
                all_factors.append(None)
                continue
            _, _, f_t = ensemble_predict_month(models, chars_t, mp_t)
            all_factors.append(f_t.cpu().numpy())

        # Prepend training factor history
        if factor_history is not None:
            historical = [f for f in factor_history if f is not None]
        else:
            historical = []

    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        n_stocks = len(returns_t)
        if n_stocks < n_deciles * 5:
            continue

        if predictive:
            # All available factors up to t-1 (historical + test[:t])
            past_factors = historical + [f for f in all_factors[:t] if f is not None]
            if len(past_factors) < 2:
                continue

            # Rolling window: use only the last lambda_window observations
            if lambda_window > 0 and len(past_factors) > lambda_window:
                past_factors = past_factors[-lambda_window:]

            lambda_prev = np.mean(past_factors, axis=0)
            lambda_prev_t = torch.tensor(
                lambda_prev, dtype=torch.float32, device=chars_t.device
            )
            # Get betas from the model, but use λ̂_{t-1} instead of f̂_t
            _, beta, _ = ensemble_predict_month(models, chars_t, mp_t)
            r_hat = (beta * lambda_prev_t.unsqueeze(0)).sum(dim=1)
        else:
            # Contemporaneous: r̂ = β' f̂_t (old behavior)
            r_hat, _, _ = ensemble_predict_month(models, chars_t, mp_t)

        r_hat_np = r_hat.cpu().numpy()
        r_actual = returns_t.cpu().numpy()

        # Assign each stock to a decile
        try:
            decile_labels = pd.qcut(
                r_hat_np, q=n_deciles, labels=False, duplicates="drop"
            )
        except ValueError:
            continue

        top_mask = decile_labels == decile_labels.max()
        bot_mask = decile_labels == decile_labels.min()

        if top_mask.sum() == 0 or bot_mask.sum() == 0:
            continue

        if value_weight and mvel_idx is not None:
            weights_t = chars_t[:, mvel_idx].cpu().numpy()
            weights_t = weights_t - weights_t.min() + 1e-6
            top_ret = np.average(r_actual[top_mask],
                                 weights=weights_t[top_mask])
            bot_ret = np.average(r_actual[bot_mask],
                                 weights=weights_t[bot_mask])
        else:
            top_ret = r_actual[top_mask].mean()
            bot_ret = r_actual[bot_mask].mean()

        monthly_returns.append(top_ret - bot_ret)

    monthly_returns = np.array(monthly_returns)

    if len(monthly_returns) == 0:
        return (0.0, monthly_returns) if return_series else 0.0

    sr = monthly_returns.mean() / max(monthly_returns.std(ddof=1), 1e-10)
    sr_annual = sr * np.sqrt(12)

    return (sr_annual, monthly_returns) if return_series else sr_annual


def compute_chars_only_sharpe(
    dataset: AssetPricingDataset,
    char_indices: Optional[List[int]] = None,
    n_deciles: int = 10,
    return_series: bool = False,
) -> float:
    """
    Characteristics-only benchmark Sharpe ratio.

    Sorts stocks by a simple composite of firm characteristics each month,
    WITHOUT any autoencoder. This isolates the Sharpe ratio you get from
    the cross-sectional characteristic sort alone — the "floor" that the
    autoencoder's predictive SR should be compared against.

    By default uses all characteristics (equal-weight average of ranks).
    If char_indices is provided, uses only those characteristics
    (e.g., [0, 3, 7] for mom1m, mom12m, bm).

    The composite score is the cross-sectional mean of the (already
    rank-normalized) characteristics for each stock. Stocks with high
    composite scores are longed, low scores are shorted.
    """
    monthly_returns = []

    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        n_stocks = len(returns_t)
        if n_stocks < n_deciles * 5:
            continue

        z = chars_t.cpu().numpy()  # (N, P) — already rank-normalized to [-1,1]
        r_actual = returns_t.cpu().numpy()

        # Composite score: average of selected (or all) characteristics
        if char_indices is not None:
            composite = z[:, char_indices].mean(axis=1)
        else:
            composite = z.mean(axis=1)

        try:
            decile_labels = pd.qcut(
                composite, q=n_deciles, labels=False, duplicates="drop"
            )
        except ValueError:
            continue

        top_mask = decile_labels == decile_labels.max()
        bot_mask = decile_labels == decile_labels.min()

        if top_mask.sum() == 0 or bot_mask.sum() == 0:
            continue

        top_ret = r_actual[top_mask].mean()
        bot_ret = r_actual[bot_mask].mean()
        monthly_returns.append(top_ret - bot_ret)

    monthly_returns = np.array(monthly_returns)

    if len(monthly_returns) == 0:
        return (0.0, monthly_returns) if return_series else 0.0

    sr = monthly_returns.mean() / max(monthly_returns.std(ddof=1), 1e-10)
    sr_annual = sr * np.sqrt(12)

    return (sr_annual, monthly_returns) if return_series else sr_annual


def compute_tangency_sharpe(
    models: list,
    dataset: AssetPricingDataset,
) -> float:
    """
    Factor tangency portfolio Sharpe ratio (Table 4).
    
    Form the mean-variance efficient portfolio of the K estimated factors.
    Out-of-sample: use mean/cov estimated through t, track return at t+1.
    Target 1% monthly volatility (as in the paper).
    """
    # Collect factor returns
    factor_returns = []
    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            continue
        _, _, f_t = ensemble_predict_month(models, chars_t, mp_t)
        factor_returns.append(f_t.cpu().numpy())
    
    factor_returns = np.array(factor_returns)  # (T, K)
    T, K = factor_returns.shape
    
    if T < K + 2:
        return 0.0
    
    # Rolling tangency portfolio
    tangency_returns = []
    target_vol = 0.01  # 1% monthly
    
    for t in range(K + 1, T):
        # Estimate mean and covariance from history up to t
        mu = factor_returns[:t].mean(axis=0)  # (K,)
        cov = np.cov(factor_returns[:t].T)     # (K, K)
        
        if K == 1:
            cov = np.array([[cov]]) if np.isscalar(cov) else cov.reshape(1, 1)
        
        try:
            cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(K))
            w = cov_inv @ mu  # tangency weights
            
            # Scale to target volatility
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol > 1e-10:
                w = w * target_vol / port_vol
            
            # Out-of-sample return at t
            tangency_ret = w @ factor_returns[t]
            tangency_returns.append(tangency_ret)
        except np.linalg.LinAlgError:
            continue
    
    tangency_returns = np.array(tangency_returns)
    
    if len(tangency_returns) == 0:
        return 0.0
    
    sr = tangency_returns.mean() / max(tangency_returns.std(ddof=1), 1e-10)
    return sr * np.sqrt(12)


# ─────────────────────────────────────────────────────────────────
# Pricing errors (alphas)
# ─────────────────────────────────────────────────────────────────

def compute_pricing_errors(
    models: list,
    dataset: AssetPricingDataset,
    mp_panel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Out-of-sample pricing errors (alphas) for managed portfolios (Fig. 3).
    
    α_j = E[x_{j,t} - x̂_{j,t}]

    The managed portfolio return for characteristic j at time t is:
        x_{j,t} = (1/N_t) Σ_i z_{i,j,t-1} * r_{i,t}
    
    The model-implied managed portfolio return is:
        x̂_{j,t} = (1/N_t) Σ_i z_{i,j,t-1} * r̂_{i,t}

    The residual x_{j,t} - x̂_{j,t} = (1/N_t) Σ_i z_{i,j,t-1} * ε_{i,t}
    aggregates stock-level pricing errors weighted by characteristics.
    
    Returns:
        alphas: (P+1,) average pricing errors
        t_stats: (P+1,) t-statistics
        mean_returns: (P+1,) average returns of managed portfolios
    """
    T = dataset.T
    P_plus_1 = mp_panel.shape[1]
    
    residuals = np.zeros((T, P_plus_1))
    valid_months = np.zeros(T, dtype=bool)
    
    for t in range(T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            continue
        
        r_hat, beta, f = ensemble_predict_month(models, chars_t, mp_t)
        
        # Stock-level residuals
        eps = (returns_t - r_hat).cpu().numpy()  # (N_t,)
        z = chars_t.cpu().numpy()                # (N_t, P)
        N_t = len(eps)
        
        # Managed portfolio residual for the market (equal-weighted):
        residuals[t, 0] = eps.mean()
        
        # Managed portfolio residual for each characteristic j:
        # x̂_{j,t} - x_{j,t} = (1/N_t) Σ_i z_{i,j} * ε_i
        P = min(z.shape[1], P_plus_1 - 1)
        residuals[t, 1:P+1] = (z[:, :P] * eps[:, None]).mean(axis=0)
        
        valid_months[t] = True
    
    n_valid = valid_months.sum()
    residuals_valid = residuals[valid_months]
    
    alphas = residuals_valid.mean(axis=0)
    t_stats = alphas / (residuals_valid.std(axis=0, ddof=1) / np.sqrt(n_valid) + 1e-10)
    mean_returns = mp_panel.mean(axis=0) if mp_panel.ndim == 2 else mp_panel
    
    return alphas, t_stats, mean_returns


# ─────────────────────────────────────────────────────────────────
# Variable importance
# ─────────────────────────────────────────────────────────────────

def compute_variable_importance(
    models: list,
    dataset: AssetPricingDataset,
    char_cols: List[str],
    baseline_r2: float,
) -> pd.DataFrame:
    """
    Variable importance (Fig. 4): reduction in R²_total when each
    characteristic is set to zero.
    """
    importances = {}
    P = len(char_cols)
    
    for j, col_name in enumerate(char_cols):
        # Create modified dataset with characteristic j zeroed out
        modified_chars = dataset.chars.clone()
        modified_chars[:, j] = 0.0
        
        # Compute R² with modified data
        ss_res = 0.0
        ss_tot = 0.0
        
        for t in range(dataset.T):
            month_mask = dataset.months == t
            if month_mask.sum() == 0:
                continue
            
            chars_t = modified_chars[month_mask]
            returns_t = dataset.returns[month_mask]
            mp_t = dataset.mp_by_month[t]
            
            r_hat, _, _ = ensemble_predict_month(models, chars_t, mp_t)
            ss_res += ((returns_t - r_hat) ** 2).sum().item()
            ss_tot += (returns_t ** 2).sum().item()
        
        r2_without_j = 1.0 - ss_res / max(ss_tot, 1e-10)
        importances[col_name] = baseline_r2 - r2_without_j
    
    # Normalize to sum to 1
    total = sum(max(v, 0) for v in importances.values())
    if total > 0:
        importances = {k: max(v, 0) / total for k, v in importances.items()}
    
    df = pd.DataFrame({
        "characteristic": list(importances.keys()),
        "importance": list(importances.values()),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    return df


# ─────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────

def evaluate_model(
    models: list,
    test_data: AssetPricingDataset,
    model_name: str = "",
    char_cols: Optional[List[str]] = None,
    verbose: bool = True,
    factor_history: Optional[List[np.ndarray]] = None,
    lambda_window: int = 60,
) -> Dict:
    """
    Run all evaluation metrics for a trained ensemble.

    Parameters
    ----------
    factor_history : list of np.ndarray, optional
        Factor estimates from the training period (from collect_factor_history).
    lambda_window : int
        Rolling window for λ̂ estimation (default 60 months = 5 years).
    
    Returns dict with all metrics.
    """
    results = {"model": model_name}
    
    # R²_total
    r2_total = compute_r2_total(models, test_data)
    results["r2_total"] = r2_total
    
    # R²_pred
    r2_pred = compute_r2_pred(models, test_data, factor_history=factor_history)
    results["r2_pred"] = r2_pred
    
    # Predictive Sharpe ratio (tradable — uses rolling λ̂_{t-1})
    sr_pred = compute_long_short_sharpe(
        models, test_data, predictive=True,
        factor_history=factor_history,
        lambda_window=lambda_window,
    )
    results["sharpe_pred"] = sr_pred
    
    # Contemporaneous Sharpe (factor model fit quality — NOT tradable)
    sr_contemp = compute_long_short_sharpe(
        models, test_data, predictive=False,
    )
    results["sharpe_contemp"] = sr_contemp
    
    # Characteristics-only benchmark Sharpe
    sr_chars = compute_chars_only_sharpe(test_data)
    results["sharpe_chars_only"] = sr_chars
    
    # Tangency Sharpe
    sr_tangency = compute_tangency_sharpe(models, test_data)
    results["sharpe_tangency"] = sr_tangency
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"  {model_name} Evaluation Results")
        print(f"{'='*50}")
        print(f"  R²_total:            {r2_total*100:.2f}%")
        print(f"  R²_pred:             {r2_pred*100:.2f}%")
        print(f"  Sharpe (pred, EW):   {sr_pred:.2f}   ← tradable")
        print(f"  Sharpe (chars only): {sr_chars:.2f}   ← benchmark")
        print(f"  Sharpe (contemp.):   {sr_contemp:.2f}   ← factor fit")
        print(f"  Sharpe (tang.):      {sr_tangency:.2f}")
        if sr_chars != 0:
            print(f"  Incremental SR:      {sr_pred - sr_chars:+.2f}   "
                  f"(AE vs chars-only)")
    
    return results


def test_hl_significance(
    monthly_hl_returns: np.ndarray,
    newey_west_lags: int = 6,
) -> pd.DataFrame:
    """
    Test whether the H-L portfolio mean return is significantly different from zero.

    Runs three tests:
      1. OLS t-test with Newey-West HAC standard errors (standard in asset pricing)
      2. Simple i.i.d. t-test (scipy)
      3. Wilcoxon signed-rank test (nonparametric, robust to fat tails)

    Parameters
    ----------
    monthly_hl_returns : array of shape (T,)
        Monthly H-L (or L-H) spread returns.
    newey_west_lags : int
        Bandwidth for Newey-West HAC correction (rule of thumb: floor(T^(1/4))).

    Returns
    -------
    DataFrame with columns: test, mean_monthly, t_stat, p_value, significant_5pct
    """
    import statsmodels.api as sm
    from scipy import stats as scipy_stats

    T = len(monthly_hl_returns)
    mean = monthly_hl_returns.mean()

    rows = []

    # 1. Newey-West
    X = np.ones((T, 1))
    res = sm.OLS(monthly_hl_returns, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags}
    )
    rows.append({
        "test": f"Newey-West (lags={newey_west_lags})",
        "mean_monthly": mean,
        "t_stat": float(res.tvalues[0]),
        "p_value": float(res.pvalues[0]),
    })

    # 2. i.i.d. t-test
    t_stat, p_val = scipy_stats.ttest_1samp(monthly_hl_returns, 0)
    rows.append({
        "test": "i.i.d. t-test",
        "mean_monthly": mean,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    })

    # 3. Wilcoxon
    stat, p_val_w = scipy_stats.wilcoxon(monthly_hl_returns)
    rows.append({
        "test": "Wilcoxon signed-rank",
        "mean_monthly": mean,
        "t_stat": float(stat),   # test statistic (not a t-stat)
        "p_value": float(p_val_w),
    })

    df = pd.DataFrame(rows)
    df["significant_5pct"] = df["p_value"] < 0.05
    return df


def compare_models(results_list: List[Dict]) -> pd.DataFrame:
    """Format evaluation results into a comparison table."""
    df = pd.DataFrame(results_list)
    df = df.set_index("model")
    
    # Format percentages
    for col in ["r2_total", "r2_pred"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)
    
    for col in ["sharpe_pred", "sharpe_chars_only", "sharpe_contemp", "sharpe_tangency"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    return df