# Autoencoder Asset Pricing

Replication and extension of [Gu, Kelly, and Xiu (2021), "Autoencoder Asset Pricing Models"](https://doi.org/10.1016/j.jeconom.2020.07.009). Princeton FIN 580 Final Project — Liam O'Shaughnessy & John Giess.

We rebuild the conditional autoencoder (CA) framework, extend the test period through 2023, and investigate whether reconstruction error constitutes a tradable anomaly signal.

---

## Results Summary

| Model | R²_total | R²_pred | SR (predictive) | SR (chars-only) |
|-------|----------|---------|-----------------|-----------------|
| IPCA  | 6.3%     | 0.3%    | 2.46 (median)   | 1.24            |
| CA0   | 15.5%    | 0.4%    | 2.25 (median)   | 1.24            |
| CA1   | 23.0%    | 0.8%    | 3.79 (median)   | 1.24            |
| CA2   | 25.1%    | 0.7%    | 2.56 (median)   | 1.24            |
| CA3   | 25.0%    | 1.3%    | 4.85 (median)   | 1.24            |

**Best model (CA2, K=5, 10-seed ensemble, test 2011–2023):**
- Total R² = 25.05%, Predictive R² = 0.65%
- Predictive Sharpe = **2.13** vs characteristics-only benchmark of **1.24**
- CAPM alpha = 47.5 bps/month (t = 7.76), market beta = 1.58
- Max drawdown = −25.3% (Sep 2011), mean monthly turnover = 30%

Reconstruction error (anomaly score) produces a monotonic return gradient across quintiles but is **not a statistically significant predictor** of future returns (NW t = −1.07, p = 0.28).

---

## Repository Structure

```
fin580-autoencoders/
├── run.py                          # Main pipeline entry point
├── config.py                       # Hyperparameters and paths
├── requirements.txt
├── submit.sh                       # Adroit SLURM job script
├── submit_build.sh                 # SLURM job for data preprocessing
├── plot_comparison.py              # Standalone plotting for model comparison
│
├── src/
│   ├── data_loader.py              # Panel construction, preprocessing, splits
│   ├── build_missing.py            # Fill missing characteristics
│   ├── pull_wrds_parquets.py       # Pull CRSP/WRDS data
│   ├── api_caller.py               # WRDS API wrapper
│   └── plot_results.py             # Figure generation utilities
│
├── train_test_val/
│   ├── models.py                   # ConditionalAutoencoder architectures (CA0–CA3, IPCA)
│   ├── train.py                    # Training loop, ensemble, AssetPricingDataset
│   ├── managed_portfolios.py       # Characteristic managed portfolio construction
│   ├── evaluate.py                 # R², Sharpe, pricing errors, chars-only benchmark
│   ├── extensions.py               # Plan A (anomaly) and Plan B (ablation) analyses
│   └── backtest.py                 # Long-short backtest, turnover, factor alphas
│
└── outputs/
    ├── figures/                    # Generated plots
    ├── tables/                     # CSV outputs (model_comparison, backtest, etc.)
    └── logs/                       # SLURM stdout/stderr logs
```

---

## Data

Data is not included in this repository and must be sourced independently.

**Required:**
- **CRSP** monthly stock returns and firm characteristics via WRDS (`src/pull_wrds_parquets.py`)
- **Chen-Zimmermann** open-source characteristics panel ([link](https://www.openassetpricing.com/)) — the `signed_predictors_dl_wide` dataset

**Processing:**
```bash
# Pull from WRDS (requires Princeton WRDS credentials)
python src/pull_wrds_parquets.py

# Build and merge characteristic panel
python src/build_missing.py

# Or if you already have a processed parquet:
# pass --processed path/to/panel_processed.parquet to run.py
```

The processed panel should be a parquet file with columns: `permno`, `date`, `ret`, `ret_excess`, and one column per characteristic (91 in our case). Characteristics are rank-normalized cross-sectionally to [−1, 1] each month during preprocessing.

**Sample:** 4,221,335 stock-months, 91 characteristics, 1972–2023.  
**Splits:** Train 1972–2005 | Val 2006–2010 | Test 2011–2023.

---

## Setup

```bash
git clone https://github.com/liamos7/fin580-autoencoders
cd fin580-autoencoders
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA optional but recommended for the full comparison run.

Additional dependencies used in `evaluate.py` and `backtest.py` (not in requirements.txt):
```bash
pip install statsmodels scipy
```

---

## Usage

### Single model

```bash
python run.py \
  --processed data/processed/panel_processed.parquet \
  --architecture CA2 \
  --K 5 \
  --seeds 10
```

### With anomaly extensions (Plan A) and ablation (Plan B)

```bash
python run.py \
  --processed data/processed/panel_processed.parquet \
  --architecture CA2 \
  --K 5 \
  --seeds 10 \
  --run-extensions
```

### With full backtest

```bash
python run.py \
  --processed data/processed/panel_processed.parquet \
  --architecture CA2 \
  --K 5 \
  --seeds 10 \
  --run-extensions \
  --run-backtest
```

### Full architecture comparison (all 30 configs)

```bash
python run.py \
  --processed data/processed/panel_processed.parquet \
  --compare-all \
  --seeds 10 \
  --run-backtest
```

### Tune L1 penalty on validation set

```bash
python run.py \
  --processed data/processed/panel_processed.parquet \
  --architecture CA2 \
  --K 5 \
  --tune-l1
```

### Princeton Adroit (SLURM)

Edit `submit.sh` to set your paths, then:
```bash
sbatch submit.sh
```

The job requests 1 GPU, 4 CPUs, 32GB RAM, and a 12-hour wall time, which is sufficient for the full 10-seed CA2 ensemble plus extensions and backtest.

---

## Key Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--processed` | `None` | Path to preprocessed `.parquet`; skips raw data build |
| `--architecture` | `CA2` | Model depth: `IPCA`, `CA0`, `CA1`, `CA2`, `CA3` |
| `--K` | `5` | Number of latent factors |
| `--seeds` | `10` | Ensemble size (predictions averaged across seeds) |
| `--compare-all` | `False` | Train all 30 architecture × K configurations |
| `--run-extensions` | `False` | Run Plan A (anomaly) and Plan B (ablation) |
| `--run-backtest` | `False` | Run predictive long-short backtest |
| `--tune-l1` | `False` | Grid-search L1 penalty on validation set |

---

## Model Architecture

The conditional autoencoder follows Gu et al. (2021). For each month $t$:

**Beta network:** maps firm characteristics $\mathbf{z}_{i,t-1} \in \mathbb{R}^P$ to factor loadings $\hat{\boldsymbol{\beta}}_i \in \mathbb{R}^K$ through $s$ dense layers with batch normalization and ReLU.

**Factor network:** projects $P+1$ characteristic-managed portfolio returns $\mathbf{x}_t$ onto $K$ latent factors via a single linear layer. The managed portfolios are the GLS projection $\mathbf{x}_t = (\mathbf{Z}_{t-1}^\top \mathbf{Z}_{t-1})^{-1} \mathbf{Z}_{t-1}^\top \mathbf{r}_t$.

**Predicted return:** $\hat{r}_{i,t} = \hat{\boldsymbol{\beta}}_i^\top \hat{\mathbf{f}}_t$.

**Loss:** MSE + L1 weight penalty + (optionally) Sharpe cap + factor correlation penalty.

| Architecture | Hidden layers | Parameters (K=5) |
|---|---|---|
| IPCA | 0 (fully linear) | ~470 |
| CA0 | 0 (linear beta net) | ~470 |
| CA1 | [32] | ~3,200 |
| CA2 | [32, 16] | ~3,700 |
| CA3 | [32, 16, 8] | ~3,830 |

---

## Outputs

After a full run, `outputs/tables/` contains:

| File | Contents |
|------|----------|
| `model_comparison.csv` | R², Sharpe for all 30 configurations |
| `robustness_summary.csv` | Median/IQR of predictive Sharpe across configs |
| `plan_a_portfolio_sorts.csv` | Quintile returns by anomaly score |
| `plan_a_hl_significance_anomaly.csv` | NW/t-test/Wilcoxon for anomaly H-L spread |
| `plan_a_hl_significance_pred.csv` | Same for predicted-return decile spread |
| `plan_a_transitions.csv` | Transition analysis (low→high anomaly) |
| `plan_b_ablation.csv` | 5×5 energy × disentangle grid |
| `backtest_performance.csv` | Ann. return, vol, Sharpe, Sortino, drawdown, hit rate |
| `backtest_monthly_returns.csv` | Month-by-month L/S, long, short, chars, market returns |
| `backtest_drawdown.csv` | Cumulative return and drawdown time series |
| `backtest_alphas.csv` | CAPM (and optional FF3) alpha with NW SEs |
| `backtest_rolling_sharpe.csv` | 36-month rolling Sharpe |
| `backtest_turnover.csv` | Monthly turnover and Jaccard similarity |
| `backtest_turnover_summary.csv` | Mean/median/std turnover, avg holding period |
| `backtest_long_holdings.csv` | Most frequently longed stocks by permno |
| `backtest_short_holdings.csv` | Most frequently shorted stocks by permno |

---

## Extensions

### Plan A — Anomaly Detection

Tests whether reconstruction error $|\varepsilon_{i,t}| = |r_{i,t} - \hat{r}_{i,t}|$ predicts next-month returns. Three tests:

1. **Portfolio sorts:** quintile sort on anomaly score, examine next-month spread
2. **Predictive regression:** $r_{i,t+1} = \alpha + \beta_1 |\varepsilon_{i,t}| + \beta_2 \hat{r}_{i,t+1} + u$ with month-clustered SEs
3. **Transition analysis:** stocks moving from low to high anomaly quintile vs. control

**Finding:** Monotonic return gradient (low anomaly earns more than high anomaly) but spread is not statistically significant (NW t = −1.07). Autoencoder residuals are noise, not mispricing.

### Plan B — Regularization Ablation

5×5 grid over energy penalty $\lambda$ (Sharpe cap) and factor disentanglement penalty $\gamma$. Best cell varies across seeds, so results are indicative rather than conclusive without multi-seed replication.

### Backtest

Predictive long-short strategy using rolling 60-month factor mean $\hat{\boldsymbol{\lambda}}_{t-1}$ seeded with training-period factor history. Includes CAPM alpha (NW SEs), turnover analysis, and characteristics-only benchmark for comparison.

---

## Citation

```
Gu, S., Kelly, B., & Xiu, D. (2021). Autoencoder asset pricing models.
Journal of Econometrics, 222(1), 429–450.
```

```
Chen, A. Y., & Zimmermann, T. (2022). Open source cross-sectional asset pricing.
Critical Finance Review, 11(2), 207–264.
```
