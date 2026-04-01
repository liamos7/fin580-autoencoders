"""
Training loop for Conditional Autoencoder Asset Pricing Models.

Implements:
- Adam optimizer with the paper's hyperparameters
- L1 (LASSO) regularization
- Early stopping on validation loss (Algorithm 1)
- Ensemble training over multiple random seeds
- Annual rolling refitting with expanding training window
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import config
from models import build_model, ConditionalAutoencoder


class AssetPricingDataset:
    """
    Wraps panel data for mini-batch SGD.
    
    The paper's SGD draws random (stock, month) observations into batches.
    We flatten the panel into individual (z_i, r_i, x_t) tuples and sample
    from that pool, which handles the unbalanced panel naturally.
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        chars: np.ndarray,
        managed_portfolios: np.ndarray,
        mask: np.ndarray,
        device: str = config.DEVICE,
    ):
        """
        Args:
            returns: (T, N_max) return panel
            chars: (T, N_max, P) characteristics panel
            managed_portfolios: (T, P+1) managed portfolio returns
            mask: (T, N_max) boolean mask
        """
        self.device = device
        
        # Flatten valid observations into lists
        # Each observation: (z_{i,t-1}, r_{i,t}, x_t, month_index)
        T, N_max = returns.shape
        P = chars.shape[2]
        
        obs_chars = []
        obs_returns = []
        obs_mp = []
        obs_month = []
        
        for t in range(T):
            valid_idx = np.where(mask[t])[0]
            for i in valid_idx:
                r = returns[t, i]
                z = chars[t, i, :]
                if np.isfinite(r) and np.all(np.isfinite(z)):
                    obs_chars.append(z)
                    obs_returns.append(r)
                    obs_mp.append(managed_portfolios[t, :])
                    obs_month.append(t)
        
        self.chars = torch.tensor(np.array(obs_chars), dtype=torch.float32, device=device)
        self.returns = torch.tensor(np.array(obs_returns), dtype=torch.float32, device=device)
        self.mp = torch.tensor(np.array(obs_mp), dtype=torch.float32, device=device)
        self.months = np.array(obs_month)
        self.n_obs = len(self.returns)
        
        # Also store month-level data for factor network
        self.mp_by_month = torch.tensor(
            managed_portfolios, dtype=torch.float32, device=device
        )
        self.T = T
    
    def get_batches(self, batch_size: int, shuffle: bool = True):
        """Yield mini-batches of (chars, returns, managed_portfolios)."""
        indices = np.arange(self.n_obs)
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.n_obs, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                self.chars[idx],       # (B, P)
                self.returns[idx],     # (B,)
                self.mp[idx],          # (B, P+1) — same x_t for all stocks in same month
            )
    
    def get_month_data(self, t: int):
        """Get all stocks for a specific month (for evaluation)."""
        month_mask = self.months == t
        return (
            self.chars[month_mask],
            self.returns[month_mask],
            self.mp_by_month[t],
        )


def compute_loss(
    model: nn.Module,
    chars: torch.Tensor,
    returns: torch.Tensor,
    mp: torch.Tensor,
    l1_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reconstruction loss + L1 penalty.
    
    Loss = (1/B) Σ (r_i - β_i' f)^2 + λ Σ|θ|
    
    Note: In each mini-batch, stocks may come from different months.
    Since the factor network takes x_t as input, we need to handle
    the fact that different stocks in the batch have different x_t.
    We compute f for each unique x_t in the batch.
    """
    r_hat, beta, f = model(chars, mp[0])  # simplified: use first month's mp
    
    # More correct approach: group by month within batch
    # For efficiency, we approximate by computing f from the mean mp in the batch
    # or we restructure to process month-by-month.
    # The paper's SGD pools observations, so using the associated x_t per obs is correct.
    
    # Since mp is (B, P+1) and may vary across batch elements,
    # we need to handle this carefully:
    unique_mp, inverse = torch.unique(mp, dim=0, return_inverse=True)
    
    # Compute factors for each unique managed portfolio vector
    all_r_hat = torch.zeros(len(returns), device=returns.device)
    
    for j in range(len(unique_mp)):
        mp_mask = (inverse == j)
        if mp_mask.sum() == 0:
            continue
        batch_chars = chars[mp_mask]
        batch_mp = unique_mp[j]
        
        r_hat_j, _, _ = model(batch_chars, batch_mp)
        all_r_hat[mp_mask] = r_hat_j
    
    # Reconstruction loss
    recon_loss = ((returns - all_r_hat) ** 2).mean()
    
    # L1 penalty
    l1_penalty = model.get_l1_penalty() if l1_lambda > 0 else torch.tensor(0.0)
    
    total_loss = recon_loss + l1_lambda * l1_penalty
    
    return total_loss, recon_loss


def compute_loss_by_month(
    model: nn.Module,
    dataset: AssetPricingDataset,
    l1_lambda: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute loss month-by-month (exact, for validation/evaluation).
    This is slower but correct — each month uses its own x_t.
    """
    model.eval()
    total_recon = 0.0
    total_obs = 0
    
    with torch.no_grad():
        for t in range(dataset.T):
            chars_t, returns_t, mp_t = dataset.get_month_data(t)
            if len(returns_t) == 0:
                continue
            
            r_hat, _, _ = model(chars_t, mp_t)
            total_recon += ((returns_t - r_hat) ** 2).sum().item()
            total_obs += len(returns_t)
    
    recon_loss = total_recon / max(total_obs, 1)
    l1_penalty = model.get_l1_penalty().item() if l1_lambda > 0 else 0.0
    
    return recon_loss + l1_lambda * l1_penalty, recon_loss


def train_single_model(
    model: nn.Module,
    train_data: AssetPricingDataset,
    val_data: AssetPricingDataset,
    l1_lambda: float = config.L1_LAMBDA,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.PATIENCE,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a single model with early stopping (Algorithm 1).
    
    Returns:
        best_model: Model with best validation loss
        history: Dict with training metrics
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS,
    )
    
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(max_epochs):
        # ── Training ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for chars, returns, mp in train_data.get_batches(batch_size):
            optimizer.zero_grad()
            loss, _ = compute_loss(model, chars, returns, mp, l1_lambda)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        train_loss = epoch_loss / max(n_batches, 1)
        
        # ── Validation ──
        val_loss, val_recon = compute_loss_by_month(model, val_data, l1_lambda)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if verbose and (epoch % 10 == 0 or epochs_without_improvement == patience):
            print(f"    Epoch {epoch:3d} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} (recon: {val_recon:.6f}) | "
                  f"Patience: {epochs_without_improvement}/{patience}")
        
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def train_ensemble(
    architecture: str,
    n_characteristics: int,
    n_managed_portfolios: int,
    n_factors: int,
    train_data: AssetPricingDataset,
    val_data: AssetPricingDataset,
    n_seeds: int = config.N_ENSEMBLE_SEEDS,
    l1_lambda: float = config.L1_LAMBDA,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[nn.Module]:
    """
    Train an ensemble of models with different random seeds.
    
    The paper averages predictions across 10 seeds for stability.
    
    Returns:
        models: List of trained models
    """
    models = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n  Seed {seed + 1}/{n_seeds}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = build_model(
            architecture, n_characteristics, n_managed_portfolios, n_factors
        ).to(config.DEVICE)
        
        model, history = train_single_model(
            model, train_data, val_data, l1_lambda, verbose=verbose
        )
        
        models.append(model)
        
        if save_dir:
            path = Path(save_dir) / f"{architecture}_K{n_factors}_seed{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
    
    return models


def ensemble_predict_month(
    models: List[nn.Module],
    chars: torch.Tensor,
    mp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Average predictions across ensemble members for one month.
    
    Returns:
        r_hat: (N,) averaged predicted returns
        beta: (N, K) averaged factor loadings
        f: (K,) averaged factors
    """
    all_r_hat = []
    all_beta = []
    all_f = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            r_hat, beta, f = model(chars, mp)
            all_r_hat.append(r_hat)
            all_beta.append(beta)
            all_f.append(f)
    
    return (
        torch.stack(all_r_hat).mean(dim=0),
        torch.stack(all_beta).mean(dim=0),
        torch.stack(all_f).mean(dim=0),
    )


def tune_l1(
    architecture: str,
    n_characteristics: int,
    n_managed_portfolios: int,
    n_factors: int,
    train_data: AssetPricingDataset,
    val_data: AssetPricingDataset,
    l1_grid: List[float] = config.L1_LAMBDA_GRID,
    n_seeds: int = 3,  # fewer seeds for tuning efficiency
    verbose: bool = True,
) -> float:
    """
    Select optimal L1 penalty via validation loss.
    Uses fewer ensemble seeds for speed during tuning.
    """
    best_lambda = l1_grid[0]
    best_val_loss = float("inf")
    
    for l1_lambda in l1_grid:
        if verbose:
            print(f"\n  Tuning λ = {l1_lambda}")
        
        models = train_ensemble(
            architecture, n_characteristics, n_managed_portfolios, n_factors,
            train_data, val_data, n_seeds=n_seeds, l1_lambda=l1_lambda,
            verbose=False,
        )
        
        # Average validation loss across ensemble
        val_losses = []
        for model in models:
            vl, _ = compute_loss_by_month(model, val_data, l1_lambda=0)
            val_losses.append(vl)
        avg_val = np.mean(val_losses)
        
        if verbose:
            print(f"    Val loss: {avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_lambda = l1_lambda
    
    if verbose:
        print(f"\n  Best λ = {best_lambda} (val loss = {best_val_loss:.6f})")
    
    return best_lambda


if __name__ == "__main__":
    """Smoke test with synthetic data."""
    
    print("Testing training pipeline with synthetic data...\n")
    
    P = 20       # characteristics
    K = 3        # factors
    N = 100      # stocks
    T_train = 60
    T_val = 24
    MP = P + 1
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    returns_train = np.random.randn(T_train, N) * 0.1
    chars_train = np.random.randn(T_train, N, P)
    mp_train = np.random.randn(T_train, MP) * 0.05
    mask_train = np.ones((T_train, N), dtype=bool)
    
    returns_val = np.random.randn(T_val, N) * 0.1
    chars_val = np.random.randn(T_val, N, P)
    mp_val = np.random.randn(T_val, MP) * 0.05
    mask_val = np.ones((T_val, N), dtype=bool)
    
    train_ds = AssetPricingDataset(returns_train, chars_train, mp_train, mask_train)
    val_ds = AssetPricingDataset(returns_val, chars_val, mp_val, mask_val)
    
    print(f"  Train: {train_ds.n_obs} observations")
    print(f"  Val:   {val_ds.n_obs} observations")
    
    # Train a single CA1 model
    model = build_model("CA1", P, MP, K).to(config.DEVICE)
    model, history = train_single_model(
        model, train_ds, val_ds, max_epochs=50, patience=5, verbose=True
    )
    
    print(f"\n  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")
    print("  Training test passed.")
