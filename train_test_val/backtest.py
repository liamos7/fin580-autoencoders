"""
Backtest infrastructure for autoencoder long-short portfolios.

Generates:
- Cumulative PnL time series (long-short vs S&P 500)
- Drawdown analysis (max drawdown, Calmar ratio, drawdown duration)
- Risk-adjusted performance (Sharpe, Sortino, annualized return/vol)
- Factor alpha regressions (CAPM alpha, FF3 alpha)
- Monthly return distribution and hit rate
- Top/bottom holdings identification by permno

All portfolio returns are computed using the PREDICTIVE signal
(β̂_{i,t-1}' λ̂_{t-1}), ensuring no look-ahead bias.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from train_test_val.train import AssetPricingDataset, ensemble_predict_month


# ═══════════════════════════════════════════════════════════════════
# Core backtest engine
# ═══════════════════════════════════════════════════════════════════

def run_backtest(
    models: list,
    dataset: AssetPricingDataset,
    panel_df: pd.DataFrame,
    n_deciles: int = 10,
    market_returns: Optional[np.ndarray] = None,
    factor_history: Optional[List[np.ndarray]] = None,
    lambda_window: int = 60,
) -> Dict:
    """
    Run a full backtest of the predictive long-short strategy.

    Uses a ROLLING window of past estimated factors λ̂_{t-1} to form
    cross-sectional predictions each month. Stocks are sorted into
    deciles by predicted return; the strategy goes long the top decile
    and short the bottom decile (equal-weighted).

    Also runs a characteristics-only benchmark portfolio for comparison.

    Parameters
    ----------
    models : list
        Trained ensemble of ConditionalAutoencoder models.
    dataset : AssetPricingDataset
        Test-period dataset.
    panel_df : pd.DataFrame
        Raw test panel with columns [permno, date, ret, ...].
    n_deciles : int
        Number of portfolio buckets (default 10 = deciles).
    market_returns : np.ndarray, optional
        Monthly market excess returns aligned to test dates.
    factor_history : list of np.ndarray, optional
        Factor estimates from training period (from collect_factor_history).
    lambda_window : int
        Rolling window in months for λ̂ estimation (default 60).
        Set to 0 for expanding window.

    Returns
    -------
    dict with keys:
        'monthly_returns' : pd.DataFrame
        'summary' : pd.DataFrame of performance statistics
        'drawdown' : pd.DataFrame of drawdown time series
        'holdings' : pd.DataFrame of top/bottom holdings per month
    """
    dates = sorted(panel_df["date"].unique())
    T = dataset.T

    # ── Step 1: collect all factor estimates ──
    all_factors = []
    for t in range(T):
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

    # ── Step 2: build permno mapping per month ──
    permnos_by_month = []
    for date in dates:
        permnos_by_month.append(
            panel_df[panel_df["date"] == date]["permno"].values
        )

    # ── Step 3: monthly portfolio construction ──
    records = []
    holdings_records = []

    for t in range(T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        n_stocks = len(returns_t)
        if n_stocks < n_deciles * 5:
            continue

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

        # Predictive return: β̂_{i,t-1}' λ̂_{t-1}
        _, beta, _ = ensemble_predict_month(models, chars_t, mp_t)
        r_pred = (beta * lambda_prev_t.unsqueeze(0)).sum(dim=1)
        r_pred_np = r_pred.cpu().numpy()
        r_actual = returns_t.cpu().numpy()

        # Decile assignment
        try:
            decile_labels = pd.qcut(
                r_pred_np, q=n_deciles, labels=False, duplicates="drop"
            )
        except ValueError:
            continue

        top_mask = decile_labels == decile_labels.max()
        bot_mask = decile_labels == decile_labels.min()

        if top_mask.sum() == 0 or bot_mask.sum() == 0:
            continue

        long_ret = r_actual[top_mask].mean()
        short_ret = r_actual[bot_mask].mean()
        ls_ret = long_ret - short_ret

        # Characteristics-only benchmark: sort on mean of all chars
        z = chars_t.cpu().numpy()
        composite = z.mean(axis=1)
        try:
            char_deciles = pd.qcut(
                composite, q=n_deciles, labels=False, duplicates="drop"
            )
            char_top = char_deciles == char_deciles.max()
            char_bot = char_deciles == char_deciles.min()
            chars_ls_ret = r_actual[char_top].mean() - r_actual[char_bot].mean()
        except ValueError:
            chars_ls_ret = 0.0

        # Market return
        if market_returns is not None and t < len(market_returns):
            mkt_ret = market_returns[t]
        else:
            mkt_ret = r_actual.mean()

        date_val = dates[t] if t < len(dates) else t

        records.append({
            "date": date_val,
            "t": t,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "chars_ls_ret": chars_ls_ret,
            "market_ret": mkt_ret,
            "n_long": int(top_mask.sum()),
            "n_short": int(bot_mask.sum()),
            "n_total": n_stocks,
        })

        # Track holdings
        if t < len(permnos_by_month):
            permnos_t = permnos_by_month[t][:n_stocks]
            for idx in np.where(top_mask)[0]:
                if idx < len(permnos_t):
                    holdings_records.append({
                        "date": date_val,
                        "permno": permnos_t[idx],
                        "side": "long",
                        "predicted_ret": r_pred_np[idx],
                        "actual_ret": r_actual[idx],
                    })
            for idx in np.where(bot_mask)[0]:
                if idx < len(permnos_t):
                    holdings_records.append({
                        "date": date_val,
                        "permno": permnos_t[idx],
                        "side": "short",
                        "predicted_ret": r_pred_np[idx],
                        "actual_ret": r_actual[idx],
                    })

    monthly_df = pd.DataFrame(records)
    holdings_df = pd.DataFrame(holdings_records)

    if len(monthly_df) == 0:
        return {
            "monthly_returns": monthly_df,
            "summary": pd.DataFrame(),
            "drawdown": pd.DataFrame(),
            "holdings": holdings_df,
        }

    # ── Step 4: compute performance statistics ──
    summary = compute_performance_summary(monthly_df)
    drawdown_df = compute_drawdown_series(monthly_df)

    return {
        "monthly_returns": monthly_df,
        "summary": summary,
        "drawdown": drawdown_df,
        "holdings": holdings_df,
    }


# ═══════════════════════════════════════════════════════════════════
# Performance statistics
# ═══════════════════════════════════════════════════════════════════

def compute_performance_summary(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a comprehensive performance summary for the long-short
    strategy and the market benchmark.

    Returns a DataFrame with one row per strategy and columns for
    annualized return, volatility, Sharpe, Sortino, max drawdown,
    Calmar ratio, hit rate, skewness, kurtosis.
    """
    strategies = {
        "Long-Short (AE)": monthly_df["ls_ret"].values,
        "Chars-Only L/S": monthly_df["chars_ls_ret"].values if "chars_ls_ret" in monthly_df.columns else np.zeros(len(monthly_df)),
        "Long Only": monthly_df["long_ret"].values,
        "Short Only": monthly_df["short_ret"].values,
        "Market (EW)": monthly_df["market_ret"].values,
    }

    rows = []
    for name, rets in strategies.items():
        ann_ret = rets.mean() * 12
        ann_vol = rets.std(ddof=1) * np.sqrt(12)

        # Sharpe
        sr = ann_ret / max(ann_vol, 1e-10)

        # Sortino (downside deviation)
        downside = rets[rets < 0]
        downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(12) if len(downside) > 0 else 1e-10
        sortino = ann_ret / max(downside_vol, 1e-10)

        # Drawdown
        cum = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = dd.min()

        # Calmar
        calmar = ann_ret / max(abs(max_dd), 1e-10)

        # Hit rate
        hit_rate = (rets > 0).mean()

        # Higher moments
        from scipy import stats as sp_stats
        skew = sp_stats.skew(rets)
        kurt = sp_stats.kurtosis(rets)

        rows.append({
            "strategy": name,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sr,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "n_months": len(rets),
            "mean_monthly": rets.mean(),
            "std_monthly": rets.std(ddof=1),
        })

    return pd.DataFrame(rows)


def compute_drawdown_series(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drawdown time series for long-short and market strategies.
    """
    dd_records = []

    for col, label in [("ls_ret", "Long-Short"), ("market_ret", "Market")]:
        rets = monthly_df[col].values
        cum = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max

        for i, row in monthly_df.iterrows():
            dd_records.append({
                "date": row["date"],
                "strategy": label,
                "cumulative_return": cum[i] - 1,
                "drawdown": dd[i],
                "wealth": cum[i],
            })

    return pd.DataFrame(dd_records)


# ═══════════════════════════════════════════════════════════════════
# Factor alpha regressions
# ═══════════════════════════════════════════════════════════════════

def compute_factor_alphas(
    monthly_df: pd.DataFrame,
    ff_factors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Regress long-short returns on factor benchmarks.

    If ff_factors is None, runs CAPM alpha using the EW market return
    from the panel. If ff_factors is provided (with columns
    [date, mktrf, smb, hml]), also runs FF3 alpha.

    All t-statistics use Newey-West HAC standard errors with
    lag = floor(T^{1/4}).
    """
    import statsmodels.api as sm
    import math

    ls_ret = monthly_df["ls_ret"].values
    mkt_ret = monthly_df["market_ret"].values
    T = len(ls_ret)
    nw_lags = math.floor(T ** 0.25)

    rows = []

    # ── CAPM alpha ──
    X_capm = sm.add_constant(mkt_ret)
    res_capm = sm.OLS(ls_ret, X_capm).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    rows.append({
        "model": "CAPM",
        "alpha_monthly": res_capm.params[0],
        "alpha_annual": res_capm.params[0] * 12,
        "alpha_tstat": res_capm.tvalues[0],
        "alpha_pval": res_capm.pvalues[0],
        "market_beta": res_capm.params[1],
        "r_squared": res_capm.rsquared,
    })

    # ── FF3 alpha (if factors available) ──
    if ff_factors is not None:
        # Merge on date
        merged = monthly_df[["date", "ls_ret"]].merge(
            ff_factors, on="date", how="inner"
        )
        if len(merged) > 10:
            y = merged["ls_ret"].values
            X_ff3 = sm.add_constant(
                merged[["mktrf", "smb", "hml"]].values
            )
            nw_ff = math.floor(len(y) ** 0.25)
            res_ff3 = sm.OLS(y, X_ff3).fit(
                cov_type="HAC", cov_kwds={"maxlags": nw_ff}
            )
            rows.append({
                "model": "FF3",
                "alpha_monthly": res_ff3.params[0],
                "alpha_annual": res_ff3.params[0] * 12,
                "alpha_tstat": res_ff3.tvalues[0],
                "alpha_pval": res_ff3.pvalues[0],
                "market_beta": res_ff3.params[1],
                "r_squared": res_ff3.rsquared,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Holdings analysis
# ═══════════════════════════════════════════════════════════════════

def summarize_holdings(
    holdings_df: pd.DataFrame,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify the most frequently held stocks in the long and short legs.

    Returns:
        long_freq: top_n most frequently longed permnos
        short_freq: top_n most frequently shorted permnos
    """
    if len(holdings_df) == 0:
        empty = pd.DataFrame(columns=["permno", "n_months", "mean_pred_ret", "mean_actual_ret"])
        return empty, empty

    def _freq_table(side_df: pd.DataFrame) -> pd.DataFrame:
        grouped = side_df.groupby("permno").agg(
            n_months=("date", "count"),
            mean_pred_ret=("predicted_ret", "mean"),
            mean_actual_ret=("actual_ret", "mean"),
        ).reset_index()
        return grouped.sort_values("n_months", ascending=False).head(top_n)

    long_freq = _freq_table(holdings_df[holdings_df["side"] == "long"])
    short_freq = _freq_table(holdings_df[holdings_df["side"] == "short"])

    return long_freq, short_freq


# ═══════════════════════════════════════════════════════════════════
# Rolling performance
# ═══════════════════════════════════════════════════════════════════

def compute_rolling_sharpe(
    monthly_df: pd.DataFrame,
    window: int = 36,
) -> pd.DataFrame:
    """
    Compute rolling annualized Sharpe ratio for the long-short strategy.
    """
    ls_ret = monthly_df["ls_ret"].values
    dates = monthly_df["date"].values

    rolling_sr = []
    for i in range(window, len(ls_ret)):
        window_rets = ls_ret[i - window:i]
        sr = window_rets.mean() / max(window_rets.std(ddof=1), 1e-10) * np.sqrt(12)
        rolling_sr.append({
            "date": dates[i],
            "rolling_sharpe_36m": sr,
        })

    return pd.DataFrame(rolling_sr)


# ═══════════════════════════════════════════════════════════════════
# Turnover analysis
# ═══════════════════════════════════════════════════════════════════

def compute_turnover(
    holdings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute monthly portfolio turnover for the long and short legs.

    Turnover is the fraction of holdings that change from one month
    to the next. A turnover of 0.0 means the portfolio is completely
    static; 1.0 means 100% of holdings are replaced.

    Also computes the Jaccard similarity (intersection / union) between
    consecutive months' holdings as a complementary measure.

    Returns a DataFrame with one row per month and columns:
        date, long_turnover, short_turnover, long_jaccard, short_jaccard,
        n_long, n_short
    """
    if len(holdings_df) == 0:
        return pd.DataFrame()

    dates = sorted(holdings_df["date"].unique())
    records = []

    for side in ["long", "short"]:
        side_df = holdings_df[holdings_df["side"] == side]

        prev_permnos = None
        for date in dates:
            month_permnos = set(
                side_df[side_df["date"] == date]["permno"].values
            )
            if len(month_permnos) == 0:
                prev_permnos = month_permnos
                continue

            if prev_permnos is not None and len(prev_permnos) > 0:
                intersection = month_permnos & prev_permnos
                union = month_permnos | prev_permnos
                # Turnover: fraction of current holdings NOT in previous
                turnover = 1.0 - len(intersection) / len(month_permnos)
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
            else:
                turnover = 1.0
                jaccard = 0.0

            records.append({
                "date": date,
                "side": side,
                "turnover": turnover,
                "jaccard": jaccard,
                "n_holdings": len(month_permnos),
            })

            prev_permnos = month_permnos

    df = pd.DataFrame(records)

    # Pivot to wide format
    if len(df) == 0:
        return df

    long_df = df[df["side"] == "long"][["date", "turnover", "jaccard", "n_holdings"]].rename(
        columns={"turnover": "long_turnover", "jaccard": "long_jaccard", "n_holdings": "n_long"}
    )
    short_df = df[df["side"] == "short"][["date", "turnover", "jaccard", "n_holdings"]].rename(
        columns={"turnover": "short_turnover", "jaccard": "short_jaccard", "n_holdings": "n_short"}
    )

    merged = long_df.merge(short_df, on="date", how="outer").sort_values("date").reset_index(drop=True)
    return merged


def summarize_turnover(turnover_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for portfolio turnover.

    Returns a DataFrame with mean, median, std, min, max turnover
    for the long and short legs, plus holding period estimates.
    """
    if len(turnover_df) == 0:
        return pd.DataFrame()

    rows = []
    for side, col in [("Long", "long_turnover"), ("Short", "short_turnover")]:
        if col not in turnover_df.columns:
            continue
        t = turnover_df[col].dropna()
        if len(t) == 0:
            continue

        mean_to = t.mean()
        # Average holding period ≈ 1 / turnover (in months)
        avg_holding = 1.0 / max(mean_to, 1e-10)

        rows.append({
            "side": side,
            "mean_turnover": mean_to,
            "median_turnover": t.median(),
            "std_turnover": t.std(),
            "min_turnover": t.min(),
            "max_turnover": t.max(),
            "avg_holding_months": avg_holding,
            "mean_jaccard": turnover_df[col.replace("turnover", "jaccard")].mean(),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Full backtest runner (integrates with run.py)
# ═══════════════════════════════════════════════════════════════════

def run_full_backtest(
    models: list,
    dataset: AssetPricingDataset,
    panel_df: pd.DataFrame,
    output_dir: str = config.TABLE_DIR,
    ff_factors: Optional[pd.DataFrame] = None,
    factor_history: Optional[List[np.ndarray]] = None,
    lambda_window: int = 60,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end backtest: compute returns, performance, drawdowns,
    factor alphas, and holdings. Save all tables to output_dir.

    Parameters
    ----------
    factor_history : list of np.ndarray, optional
        Factor estimates from training period.
    lambda_window : int
        Rolling window for λ̂ estimation (default 60 months).
    """
    if verbose:
        print("\n  Running predictive long-short backtest...")
        print(f"  λ̂ window: {lambda_window} months "
              f"({'rolling' if lambda_window > 0 else 'expanding'})")
        if factor_history:
            print(f"  Training factor history: {len(factor_history)} months")

    results = run_backtest(
        models, dataset, panel_df,
        factor_history=factor_history,
        lambda_window=lambda_window,
    )
    monthly_df = results["monthly_returns"]
    holdings_df = results["holdings"]

    if len(monthly_df) == 0:
        print("  WARNING: No valid months in backtest. Skipping.")
        return results

    # ── Performance summary ──
    summary = results["summary"]
    if verbose:
        print("\n  Performance Summary:")
        print(summary.to_string(index=False))

    summary.to_csv(output_dir + "backtest_performance.csv", index=False)

    # ── Drawdown ──
    dd_df = results["drawdown"]
    dd_df.to_csv(output_dir + "backtest_drawdown.csv", index=False)

    ls_dd = dd_df[dd_df["strategy"] == "Long-Short"]
    if len(ls_dd) > 0:
        max_dd = ls_dd["drawdown"].min()
        max_dd_date = ls_dd.loc[ls_dd["drawdown"].idxmin(), "date"]
        if verbose:
            print(f"\n  Max drawdown: {max_dd*100:.1f}% at {max_dd_date}")

    # ── Factor alphas ──
    alphas = compute_factor_alphas(monthly_df, ff_factors)
    if verbose:
        print("\n  Factor Alphas:")
        print(alphas.to_string(index=False))
    alphas.to_csv(output_dir + "backtest_alphas.csv", index=False)

    # ── Monthly returns ──
    monthly_df.to_csv(output_dir + "backtest_monthly_returns.csv", index=False)

    # ── Rolling Sharpe ──
    rolling_sr = compute_rolling_sharpe(monthly_df)
    if len(rolling_sr) > 0:
        rolling_sr.to_csv(output_dir + "backtest_rolling_sharpe.csv", index=False)

    # ── Holdings ──
    long_freq, short_freq = summarize_holdings(holdings_df)
    if verbose and len(long_freq) > 0:
        print(f"\n  Most frequently longed stocks (by permno):")
        print(long_freq.head(10).to_string(index=False))
        print(f"\n  Most frequently shorted stocks (by permno):")
        print(short_freq.head(10).to_string(index=False))

    long_freq.to_csv(output_dir + "backtest_long_holdings.csv", index=False)
    short_freq.to_csv(output_dir + "backtest_short_holdings.csv", index=False)

    # ── Turnover analysis ──
    turnover_df = compute_turnover(holdings_df)
    turnover_summary = summarize_turnover(turnover_df)

    if verbose and len(turnover_summary) > 0:
        print("\n  Portfolio Turnover:")
        print(turnover_summary.to_string(index=False))

    if len(turnover_df) > 0:
        turnover_df.to_csv(output_dir + "backtest_turnover.csv", index=False)
        turnover_summary.to_csv(output_dir + "backtest_turnover_summary.csv", index=False)

    results["alphas"] = alphas
    results["rolling_sharpe"] = rolling_sr
    results["long_freq"] = long_freq
    results["short_freq"] = short_freq
    results["turnover"] = turnover_df
    results["turnover_summary"] = turnover_summary

    if verbose:
        print("\n  Backtest tables saved to", output_dir)

    return results