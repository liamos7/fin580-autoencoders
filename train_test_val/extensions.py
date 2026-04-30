"""
Extensions to the autoencoder asset pricing model.

Plan A: Energy-based anomaly scoring on reconstruction residuals
Plan B: Energy-based regularization of the latent factor space

Both draw on ideas from energy-based autoencoders used in HEP trigger
systems, adapted to the asset pricing context.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from train_test_val.models import ConditionalAutoencoder, build_model
from train_test_val.train import AssetPricingDataset, ensemble_predict_month


# ═══════════════════════════════════════════════════════════════════
# PLAN A: Anomaly Scoring on Residuals
# ═══════════════════════════════════════════════════════════════════

def compute_anomaly_scores(
    models: list,
    dataset: AssetPricingDataset,
    window: int = config.ANOMALY_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-stock-month reconstruction errors (anomaly scores).
    
    The energy score E_i = (1/T_w) Σ ε²_{i,s} over a trailing window
    measures how well the factor model explains each stock.
    
    High-energy stocks = poorly explained by latent factor structure.
    In the trigger analogy: these are the "anomalous events" the
    standard model fails to reconstruct.
    
    Returns:
        residuals: list of (N_t,) arrays of ε_{i,t} per month
        squared_residuals: list of (N_t,) arrays of ε²_{i,t}
        predicted_returns: list of (N_t,) arrays of r̂_{i,t}
        actual_returns: list of (N_t,) arrays of r_{i,t}
    """
    residuals = []
    squared_residuals = []
    predicted_returns = []
    actual_returns = []
    
    for t in range(dataset.T):
        chars_t, returns_t, mp_t = dataset.get_month_data(t)
        if len(returns_t) == 0:
            residuals.append(np.array([]))
            squared_residuals.append(np.array([]))
            predicted_returns.append(np.array([]))
            actual_returns.append(np.array([]))
            continue
        
        r_hat, _, _ = ensemble_predict_month(models, chars_t, mp_t)
        
        r = returns_t.cpu().numpy()
        rh = r_hat.cpu().numpy()
        eps = r - rh
        
        residuals.append(eps)
        squared_residuals.append(eps ** 2)
        predicted_returns.append(rh)
        actual_returns.append(r)
    
    return residuals, squared_residuals, predicted_returns, actual_returns


def portfolio_sort_analysis(
    dataset: AssetPricingDataset,
    residuals: list,
    actual_returns: list,
    n_quantiles: int = config.N_QUANTILES,
    return_hl_series: bool = False,
):
    """
    Plan A, Test 1: Portfolio sorts by anomaly score.
    
    Each month, sort stocks into quintiles by |ε_{i,t}|.
    Examine next-month return spread between high and low quintiles.
    
    If high-anomaly stocks earn abnormal returns, the autoencoder's
    residuals contain economic signal — mispricing or regime change
    that the factor model misses.
    """
    # We need to look at return at t+1 for stocks sorted at t
    T = len(residuals)
    
    quintile_returns = {q: [] for q in range(n_quantiles)}
    
    for t in range(T - 1):
        eps_t = residuals[t]
        r_next = actual_returns[t + 1]
        
        if len(eps_t) == 0 or len(r_next) == 0:
            continue
        
        # Use the minimum of stocks present in both months
        n = min(len(eps_t), len(r_next))
        if n < n_quantiles * 5:
            continue
        
        # Anomaly score: absolute residual
        anomaly = np.abs(eps_t[:n])
        r = r_next[:n]
        
        # Sort into quintiles
        percentiles = np.percentile(anomaly, np.linspace(0, 100, n_quantiles + 1))
        
        for q in range(n_quantiles):
            lo = percentiles[q]
            hi = percentiles[q + 1]
            if q == n_quantiles - 1:
                mask = anomaly >= lo
            else:
                mask = (anomaly >= lo) & (anomaly < hi)
            
            if mask.sum() > 0:
                quintile_returns[q].append(r[mask].mean())
    
    # Compute summary statistics
    results = []
    for q in range(n_quantiles):
        rets = np.array(quintile_returns[q])
        if len(rets) > 0:
            results.append({
                "quintile": q + 1,
                "label": "Low anomaly" if q == 0 else ("High anomaly" if q == n_quantiles - 1 else f"Q{q+1}"),
                "mean_return_monthly": rets.mean(),
                "std_return": rets.std(),
                "sharpe_annual": rets.mean() / max(rets.std(), 1e-10) * np.sqrt(12),
                "n_months": len(rets),
            })
    
    df = pd.DataFrame(results)
    
    # Add long-short (high anomaly - low anomaly)
    ls_ret = np.array([])
    if len(df) >= 2:
        high = np.array(quintile_returns[n_quantiles - 1])
        low = np.array(quintile_returns[0])
        n_common = min(len(high), len(low))
        ls_ret = high[:n_common] - low[:n_common]
        df = pd.concat([df, pd.DataFrame([{
            "quintile": 0,
            "label": "High - Low",
            "mean_return_monthly": ls_ret.mean(),
            "std_return": ls_ret.std(),
            "sharpe_annual": ls_ret.mean() / max(ls_ret.std(), 1e-10) * np.sqrt(12),
            "n_months": len(ls_ret),
        }])], ignore_index=True)

    return (df, ls_ret) if return_hl_series else df


def predictive_regression(
    dataset: AssetPricingDataset,
    residuals: list,
    predicted_returns: list,
    actual_returns: list,
) -> Dict:
    """
    Plan A, Test 2: Predictive regressions.
    
    Regress r_{i,t+1} on anomaly score E_{i,t}, controlling for
    the model's predicted return r̂_{i,t+1} and characteristics.
    
    If E_{i,t} has a significant coefficient, reconstruction error
    adds predictive power beyond the model itself.
    """
    # Collect panel for regression
    y_list = []       # r_{i,t+1}
    anomaly_list = [] # |ε_{i,t}|
    rhat_list = []    # r̂_{i,t+1}
    
    T = len(residuals)
    for t in range(T - 1):
        eps_t = residuals[t]
        r_next = actual_returns[t + 1]
        rhat_next = predicted_returns[t + 1]
        
        n = min(len(eps_t), len(r_next), len(rhat_next))
        if n < 10:
            continue
        
        y_list.extend(r_next[:n].tolist())
        anomaly_list.extend(np.abs(eps_t[:n]).tolist())
        rhat_list.extend(rhat_next[:n].tolist())
    
    y = np.array(y_list)
    anomaly = np.array(anomaly_list)
    rhat = np.array(rhat_list)
    
    # OLS: r_{t+1} = a + b1 * anomaly_t + b2 * rhat_{t+1} + error
    X = np.column_stack([np.ones(len(y)), anomaly, rhat])
    
    try:
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta_ols
        
        # Standard errors (OLS, not Newey-West — for a class project this is fine)
        sigma2 = (residual ** 2).mean()
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
        t_stats = beta_ols / se
        
        return {
            "intercept": beta_ols[0],
            "anomaly_coeff": beta_ols[1],
            "rhat_coeff": beta_ols[2],
            "anomaly_tstat": t_stats[1],
            "rhat_tstat": t_stats[2],
            "r_squared": 1 - (residual ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "n_obs": len(y),
        }
    except np.linalg.LinAlgError:
        return {"error": "Regression failed"}


def transition_analysis(
    dataset: AssetPricingDataset,
    residuals: list,
    actual_returns: list,
    panel_df: pd.DataFrame,
    lookback: int = 6,
    lookahead: int = 6,
    n_quantiles: int = config.N_QUANTILES,
) -> Dict:
    """
    Plan A, Test 3: Transition analysis.

    Track stocks that move from the bottom anomaly quintile into the top
    quintile (low-to-high transition). Examine their returns over the
    subsequent lookahead window vs. stocks that stay low-anomaly.

    Analogous to the trigger detecting a transition from background to a
    new-physics event: a sudden jump in reconstruction error flags a
    structural break in a stock's factor exposure.

    Args:
        dataset:        AssetPricingDataset for the test period.
        residuals:      list of (N_t,) residual arrays from compute_anomaly_scores.
        actual_returns: list of (N_t,) actual return arrays.
        panel_df:       raw test DataFrame with columns [permno, date, ...],
                        sorted by [permno, date] — used to recover stock identities.
        lookback:       months of history used to classify a stock as "low anomaly".
        lookahead:      months over which post-transition returns are measured.
    """
    from collections import defaultdict

    dates = sorted(panel_df["date"].unique())
    T = len(dates)

    # Build permno list per month (matches the order in residuals/actual_returns,
    # since prepare_tensors iterates month_df in the panel's row order)
    permnos_by_month = []
    for date in dates:
        permnos_by_month.append(panel_df[panel_df["date"] == date]["permno"].values)

    # ── Step 1: compute anomaly score (|ε|) per stock-month ──
    # Store as {permno: [(t, score), ...]}
    stock_scores: dict = defaultdict(list)
    for t in range(T):
        eps = residuals[t]
        permnos = permnos_by_month[t]
        n = min(len(eps), len(permnos))
        for i in range(n):
            stock_scores[permnos[i]].append((t, abs(eps[i])))

    # ── Step 2: for each month t ≥ lookback, classify each stock ──
    # A stock is "previously low" if its mean |ε| over [t-lookback, t-1]
    # fell in the bottom quintile of the cross-section at t-1.
    # A "transition" occurs when its score at t is in the top quintile.

    # Pre-compute cross-sectional quintile thresholds per month
    lo_thresh = {}   # bottom-quintile upper bound at month t
    hi_thresh = {}   # top-quintile lower bound at month t
    for t in range(T):
        eps = np.abs(residuals[t])
        if len(eps) >= n_quantiles * 5:
            lo_thresh[t] = np.percentile(eps, 100 / n_quantiles)
            hi_thresh[t] = np.percentile(eps, 100 * (n_quantiles - 1) / n_quantiles)

    # ── Step 3: identify transitions and collect forward returns ──
    transition_fwd_returns = []   # post-transition cumulative returns
    control_fwd_returns = []      # stocks that stay low-anomaly

    for permno, history in stock_scores.items():
        t_vals = [h[0] for h in history]
        s_vals = np.array([h[1] for h in history])
        t_index = {t: i for i, t in enumerate(t_vals)}

        for idx, (t, score) in enumerate(history):
            if t < lookback or t + lookahead >= T:
                continue
            if t not in hi_thresh or t not in lo_thresh:
                continue

            # Check lookback window: was stock consistently low-anomaly?
            lookback_scores = [
                s_vals[t_index[s]] for s in range(t - lookback, t)
                if s in t_index and s in lo_thresh
            ]
            if len(lookback_scores) < lookback // 2:
                continue

            was_low = np.mean(lookback_scores) <= lo_thresh[t - 1] if (t - 1) in lo_thresh else False

            # Collect lookahead returns for this stock
            fwd = [
                s_vals[t_index[s]] for s in range(t + 1, t + lookahead + 1)
                if s in t_index
            ]
            # Use actual returns instead of anomaly scores for forward performance
            fwd_rets = []
            for s in range(t + 1, t + lookahead + 1):
                permnos_s = permnos_by_month[s] if s < T else []
                pos = np.where(permnos_s == permno)[0]
                if len(pos) > 0 and pos[0] < len(actual_returns[s]):
                    fwd_rets.append(actual_returns[s][pos[0]])

            if len(fwd_rets) < lookahead // 2:
                continue

            cum_ret = np.prod(1 + np.array(fwd_rets)) - 1

            if was_low and score >= hi_thresh[t]:
                # Transition: low → high anomaly
                transition_fwd_returns.append(cum_ret)
            elif was_low and score <= lo_thresh[t]:
                # Control: stayed low
                control_fwd_returns.append(cum_ret)

    # ── Step 4: summarise ──
    def summarise(rets: list, label: str) -> dict:
        if len(rets) == 0:
            return {"group": label, "n": 0, "mean_cum_ret": np.nan,
                    "std": np.nan, "sharpe_annual": np.nan}
        r = np.array(rets)
        ann_sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(12 / lookahead)
        return {
            "group": label,
            "n": len(r),
            "mean_cum_ret": r.mean(),
            "std": r.std(),
            "sharpe_annual": ann_sharpe,
        }

    rows = [
        summarise(transition_fwd_returns, f"Low→High anomaly (transition)"),
        summarise(control_fwd_returns,    f"Low→Low anomaly (control)"),
    ]

    spread = np.nan
    if transition_fwd_returns and control_fwd_returns:
        n = min(len(transition_fwd_returns), len(control_fwd_returns))
        diff = np.array(transition_fwd_returns[:n]) - np.array(control_fwd_returns[:n])
        spread_sharpe = diff.mean() / (diff.std() + 1e-10) * np.sqrt(12 / lookahead)
        rows.append({
            "group": "Transition − Control (spread)",
            "n": n,
            "mean_cum_ret": diff.mean(),
            "std": diff.std(),
            "sharpe_annual": spread_sharpe,
        })

    results_df = pd.DataFrame(rows)

    return {
        "summary": results_df,
        "n_transitions": len(transition_fwd_returns),
        "n_controls": len(control_fwd_returns),
        "lookback_months": lookback,
        "lookahead_months": lookahead,
    }


# ═══════════════════════════════════════════════════════════════════
# PLAN B: Energy-Regularized Autoencoder
# ═══════════════════════════════════════════════════════════════════

class EnergyRegularizedAutoencoder(ConditionalAutoencoder):
    """
    Conditional autoencoder with energy-based regularization.
    
    Adds two penalty terms to the standard reconstruction loss:
    
    1. Energy penalty: Penalizes latent factor configurations that imply
       implausibly high Sharpe ratios (arbitrage opportunities).
       
       L_energy = λ * max(0, SR(f) - κ)²
       
       This is motivated by the Hansen-Jagannathan bound: in an
       arbitrage-free economy, the maximum attainable Sharpe ratio
       is bounded. Factors that violate this bound are likely
       overfitting to transient patterns.
    
    2. Disentanglement penalty: Penalizes correlation between latent
       factors to encourage each factor to capture a distinct risk source.
       
       L_disentangle = γ * ||off-diag(Corr(f))||²_F
    
    Total loss: L_recon + L_energy + L_disentangle + L_l1
    """
    
    def __init__(
        self,
        n_characteristics: int,
        n_managed_portfolios: int,
        n_factors: int,
        hidden_layers: list,
        energy_lambda: float = config.ENERGY_LAMBDA_DEFAULT,
        sharpe_threshold: float = config.SHARPE_THRESHOLD,
        disentangle_gamma: float = config.DISENTANGLE_GAMMA_DEFAULT,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            n_characteristics, n_managed_portfolios, n_factors,
            hidden_layers, use_batch_norm
        )
        
        self.energy_lambda = energy_lambda
        self.sharpe_threshold = sharpe_threshold
        self.disentangle_gamma = disentangle_gamma
        
        # Rolling buffer for factor returns (for Sharpe estimation)
        self.factor_buffer = []
        self.buffer_maxlen = 120  # 10 years of monthly data
    
    def compute_energy_penalty(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute energy penalty based on implied Sharpe ratio.
        
        Uses a rolling buffer of factor realizations to estimate
        the maximum Sharpe ratio of the tangency portfolio.
        
        L_energy = λ * max(0, SR² - κ²)
        
        where SR² = μ' Σ⁻¹ μ is the squared Sharpe of the tangency portfolio.
        """
        # Add current factor to buffer
        self.factor_buffer.append(f.detach().cpu().numpy())
        if len(self.factor_buffer) > self.buffer_maxlen:
            self.factor_buffer.pop(0)
        
        if len(self.factor_buffer) < self.n_factors + 2:
            return torch.tensor(0.0, device=f.device)
        
        # Estimate factor mean and covariance from buffer
        F = np.array(self.factor_buffer)  # (T_buf, K)
        mu = F.mean(axis=0)
        cov = np.cov(F.T)
        
        if self.n_factors == 1:
            cov = np.array([[cov]]) if np.isscalar(cov) else cov.reshape(1, 1)
        
        try:
            cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(self.n_factors))
            sr_squared = mu @ cov_inv @ mu  # annualize: * 12
            sr_squared_annual = sr_squared * 12
            
            kappa_squared = self.sharpe_threshold ** 2
            
            # Hinge loss: only penalize if SR exceeds threshold
            penalty = max(0, sr_squared_annual - kappa_squared)
            
            return torch.tensor(
                self.energy_lambda * penalty,
                dtype=torch.float32,
                device=f.device,
            )
        except np.linalg.LinAlgError:
            return torch.tensor(0.0, device=f.device)
    
    def compute_disentanglement_penalty(self, f: torch.Tensor) -> torch.Tensor:
        """
        Penalize off-diagonal correlations between latent factors.
        
        L_disentangle = γ * ||off-diag(Corr(f))||²_F
        
        This encourages each factor to capture a distinct source of risk,
        analogous to separating signal channels in the detector.
        """
        if len(self.factor_buffer) < self.n_factors + 2:
            return torch.tensor(0.0, device=f.device)
        
        F = torch.tensor(
            np.array(self.factor_buffer),
            dtype=torch.float32,
            device=f.device,
        )
        
        # Compute correlation matrix
        F_centered = F - F.mean(dim=0, keepdim=True)
        stds = F_centered.std(dim=0, keepdim=True) + 1e-8
        F_norm = F_centered / stds
        corr = (F_norm.T @ F_norm) / (F_norm.shape[0] - 1)
        
        # Off-diagonal Frobenius norm
        mask = 1.0 - torch.eye(self.n_factors, device=f.device)
        off_diag = corr * mask
        penalty = (off_diag ** 2).sum()
        
        return self.disentangle_gamma * penalty
    
    def compute_total_loss(
        self,
        r: torch.Tensor,
        r_hat: torch.Tensor,
        f: torch.Tensor,
        l1_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Total loss = L_recon + L_energy + L_disentangle + L_l1
        
        Returns total loss and a dict of individual components for logging.
        """
        recon_loss = ((r - r_hat) ** 2).mean()
        energy_loss = self.compute_energy_penalty(f)
        disentangle_loss = self.compute_disentanglement_penalty(f)
        l1_loss = l1_lambda * self.get_l1_penalty() if l1_lambda > 0 else torch.tensor(0.0)
        
        total = recon_loss + energy_loss + disentangle_loss + l1_loss
        
        components = {
            "recon": recon_loss.item(),
            "energy": energy_loss.item(),
            "disentangle": disentangle_loss.item(),
            "l1": l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss,
            "total": total.item(),
        }
        
        return total, components
    
    def reset_buffer(self):
        """Clear the factor buffer (call between training runs)."""
        self.factor_buffer = []


def build_energy_model(
    architecture: str,
    n_characteristics: int,
    n_managed_portfolios: int,
    n_factors: int,
    energy_lambda: float = config.ENERGY_LAMBDA_DEFAULT,
    disentangle_gamma: float = config.DISENTANGLE_GAMMA_DEFAULT,
    sharpe_threshold: float = config.SHARPE_THRESHOLD,
) -> EnergyRegularizedAutoencoder:
    """Factory for energy-regularized models."""
    
    hidden_layers = config.ARCHITECTURES.get(architecture, [32, 16])
    
    return EnergyRegularizedAutoencoder(
        n_characteristics=n_characteristics,
        n_managed_portfolios=n_managed_portfolios,
        n_factors=n_factors,
        hidden_layers=hidden_layers,
        energy_lambda=energy_lambda,
        sharpe_threshold=sharpe_threshold,
        disentangle_gamma=disentangle_gamma,
    )


# ═══════════════════════════════════════════════════════════════════
# Ablation grid runners
# ═══════════════════════════════════════════════════════════════════

def run_energy_ablation(
    architecture: str,
    n_characteristics: int,
    n_managed_portfolios: int,
    n_factors: int,
    train_data: AssetPricingDataset,
    val_data: AssetPricingDataset,
    test_data: AssetPricingDataset,
    energy_lambdas: List[float] = config.ENERGY_LAMBDA_GRID,
    disentangle_gammas: List[float] = config.DISENTANGLE_GAMMA_GRID,
) -> pd.DataFrame:
    """
    Systematic ablation over energy and disentanglement penalty weights.
    
    For each (λ, γ) combination, train the model and evaluate OOS metrics.
    This generates the results table for the report's Section 4.
    """
    results = []
    
    for lam in energy_lambdas:
        for gam in disentangle_gammas:
            print(f"\n  Energy λ={lam}, Disentangle γ={gam}")
            
            model = build_energy_model(
                architecture, n_characteristics, n_managed_portfolios,
                n_factors, energy_lambda=lam, disentangle_gamma=gam,
            ).to(config.DEVICE)
            
            # Train (simplified: single seed for ablation speed)
            from train_test_val.train import train_single_model
            model, history = train_single_model(
                model, train_data, val_data, verbose=False
            )

            # Evaluate
            from train_test_val.evaluate import compute_r2_total, compute_r2_pred
            r2_total = compute_r2_total([model], test_data)
            r2_pred = compute_r2_pred([model], test_data)
            
            results.append({
                "energy_lambda": lam,
                "disentangle_gamma": gam,
                "r2_total": r2_total,
                "r2_pred": r2_pred,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    """Quick test of extension components."""
    
    print("Testing Plan B: EnergyRegularizedAutoencoder\n")
    
    P, K, N = 20, 3, 100
    MP = P + 1
    
    model = build_energy_model("CA2", P, MP, K)
    
    z = torch.randn(N, P)
    x = torch.randn(MP)
    
    r_hat, beta, f = model(z, x)
    
    # Simulate several months to fill buffer
    for _ in range(20):
        f_sim = torch.randn(K)
        model.factor_buffer.append(f_sim.numpy())
    
    # Test loss computation
    r = torch.randn(N)
    total_loss, components = model.compute_total_loss(r, r_hat, f)
    
    print(f"  Loss components:")
    for k, v in components.items():
        print(f"    {k:15s}: {v:.6f}")
    
    print("\n  Energy regularization test passed.")
