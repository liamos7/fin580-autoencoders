"""
Model definitions for Conditional Autoencoder Asset Pricing.

Architecture (from Fig. 2 of the paper):

    Beta Network                    Factor Network
    ───────────                     ──────────────
    z_{i,t-1}  (P×1)               x_t  (P+1 × 1)
        │                               │
    [Hidden layers with ReLU]       [Single linear layer]
        │                               │
    β_{i,t-1}  (K×1)               f_t  (K×1)
        │                               │
        └──────── dot product ──────────┘
                     │
                r̂_{i,t} = β' f   (scalar)

Implements: CA0 (linear), CA1 (1 hidden), CA2 (2 hidden), CA3 (3 hidden).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

import config


class BetaNetwork(nn.Module):
    """
    Maps firm characteristics z_{i,t-1} ∈ R^P to factor loadings β_{i,t-1} ∈ R^K.
    
    Architecture depends on hidden_layers:
        CA0: z -> Linear -> β                    (no hidden layers)
        CA1: z -> Linear(32) -> ReLU -> BN -> Linear -> β
        CA2: z -> Linear(32) -> ReLU -> BN -> Linear(16) -> ReLU -> BN -> Linear -> β
        CA3: z -> Linear(32) -> ReLU -> BN -> Linear(16) -> ReLU -> BN -> 
                  Linear(8) -> ReLU -> BN -> Linear -> β
    """
    
    def __init__(
        self,
        n_characteristics: int,
        n_factors: int,
        hidden_layers: List[int],
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.n_characteristics = n_characteristics
        self.n_factors = n_factors
        
        layers = []
        in_dim = n_characteristics
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        # Final layer: output K factor loadings (no activation — betas are unconstrained)
        layers.append(nn.Linear(in_dim, n_factors))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, P) characteristics
        Returns:
            beta: (batch_size, K) factor loadings
        """
        return self.network(z)


class FactorNetwork(nn.Module):
    """
    Maps managed portfolio returns x_t ∈ R^{P+1} to latent factors f_t ∈ R^K.
    
    Always a single linear layer (no activation), preserving the economic
    interpretation that factors are portfolios (linear combinations of returns).
    """
    
    def __init__(
        self,
        n_managed_portfolios: int,
        n_factors: int,
    ):
        super().__init__()
        
        self.n_managed_portfolios = n_managed_portfolios
        self.n_factors = n_factors
        
        # Single linear layer: x_t -> f_t
        self.linear = nn.Linear(n_managed_portfolios, n_factors, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (P+1,) or (1, P+1) managed portfolio returns for one month
        Returns:
            f: (K,) or (1, K) factor returns
        """
        return self.linear(x)


class ConditionalAutoencoder(nn.Module):
    """
    Full conditional autoencoder: β(z)' f = r̂.
    
    Combines BetaNetwork and FactorNetwork with a dot product.
    
    The key insight: β is computed per-stock from characteristics (left network),
    while f is computed once per month from managed portfolios (right network).
    The reconstructed return is their inner product.
    """
    
    def __init__(
        self,
        n_characteristics: int,
        n_managed_portfolios: int,
        n_factors: int,
        hidden_layers: List[int],
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.n_factors = n_factors
        self.beta_network = BetaNetwork(
            n_characteristics, n_factors, hidden_layers, use_batch_norm
        )
        self.factor_network = FactorNetwork(n_managed_portfolios, n_factors)
    
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (N_t, P) characteristics for all stocks at time t-1
            x: (P+1,) managed portfolio returns at time t
        
        Returns:
            r_hat: (N_t,) predicted returns
            beta: (N_t, K) estimated factor loadings
            f: (K,) estimated factor returns
        """
        # Beta network: per-stock
        beta = self.beta_network(z)  # (N_t, K)
        
        # Factor network: per-month
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.factor_network(x).squeeze(0)  # (K,)
        
        # Dot product: r̂_{i,t} = β_{i,t-1}' f_t
        r_hat = (beta * f.unsqueeze(0)).sum(dim=1)  # (N_t,)
        
        return r_hat, beta, f
    
    def get_l1_penalty(self) -> torch.Tensor:
        """Compute L1 (LASSO) penalty over all weight parameters."""
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l1 += param.abs().sum()
        return l1


class IPCA(nn.Module):
    """
    Instrumented PCA baseline (Kelly, Pruitt, Su 2019).
    
    This is equivalent to CA0 but estimated with the exact IPCA algorithm.
    We implement it as a special case of the autoencoder for fair comparison:
    β = Z Γ (linear, no hidden layers, no activation).
    """
    
    def __init__(
        self,
        n_characteristics: int,
        n_managed_portfolios: int,
        n_factors: int,
    ):
        super().__init__()
        
        self.n_factors = n_factors
        
        # Γ: P × K mapping from characteristics to betas
        self.gamma = nn.Linear(n_characteristics, n_factors, bias=False)
        
        # Factor network (same as CA)
        self.factor_network = FactorNetwork(n_managed_portfolios, n_factors)
    
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same interface as ConditionalAutoencoder."""
        beta = self.gamma(z)  # (N_t, K)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.factor_network(x).squeeze(0)  # (K,)
        
        r_hat = (beta * f.unsqueeze(0)).sum(dim=1)
        
        return r_hat, beta, f
    
    def get_l1_penalty(self) -> torch.Tensor:
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l1 += param.abs().sum()
        return l1


def build_model(
    architecture: str,
    n_characteristics: int,
    n_managed_portfolios: int,
    n_factors: int,
    use_batch_norm: bool = config.USE_BATCH_NORM,
) -> nn.Module:
    """
    Factory function to build a model by name.
    
    Args:
        architecture: One of "IPCA", "CA0", "CA1", "CA2", "CA3"
        n_characteristics: P (number of characteristics)
        n_managed_portfolios: P+1
        n_factors: K
    
    Returns:
        model: nn.Module
    """
    if architecture == "IPCA":
        return IPCA(n_characteristics, n_managed_portfolios, n_factors)
    
    if architecture not in config.ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: IPCA, {list(config.ARCHITECTURES.keys())}"
        )
    
    hidden_layers = config.ARCHITECTURES[architecture]
    
    return ConditionalAutoencoder(
        n_characteristics=n_characteristics,
        n_managed_portfolios=n_managed_portfolios,
        n_factors=n_factors,
        hidden_layers=hidden_layers,
        use_batch_norm=use_batch_norm,
    )


if __name__ == "__main__":
    """Smoke test: verify forward pass shapes for all architectures."""
    
    P = 94   # characteristics
    K = 5    # factors
    N = 200  # stocks in a month
    MP = P + 1  # managed portfolios
    
    print("Testing model architectures:\n")
    
    for arch_name in ["IPCA", "CA0", "CA1", "CA2", "CA3"]:
        model = build_model(arch_name, P, MP, K)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        
        # Forward pass
        z = torch.randn(N, P)
        x = torch.randn(MP)
        
        r_hat, beta, f = model(z, x)
        
        assert r_hat.shape == (N,), f"{arch_name}: r_hat shape {r_hat.shape}"
        assert beta.shape == (N, K), f"{arch_name}: beta shape {beta.shape}"
        assert f.shape == (K,), f"{arch_name}: f shape {f.shape}"
        
        print(f"  {arch_name:5s}: {n_params:>7,} params | "
              f"r_hat {tuple(r_hat.shape)}, beta {tuple(beta.shape)}, f {tuple(f.shape)}")
    
    print("\n  All architecture tests passed.")
