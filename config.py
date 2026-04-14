"""
Configuration for Autoencoder Asset Pricing Models
Replication of Gu, Kelly, Xiu (2021) with energy-based extensions.
"""

import torch

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent

DATA_DIR      = str(_ROOT / "data")        + "/"
RAW_DIR       = str(_ROOT / "data/raw")    + "/"
PROCESSED_DIR = str(_ROOT / "data/processed") + "/"
OUTPUT_DIR    = str(_ROOT / "outputs")     + "/"
MODEL_DIR     = str(_ROOT / "outputs/models")   + "/"
FIGURE_DIR    = str(_ROOT / "outputs/figures")  + "/"
TABLE_DIR     = str(_ROOT / "outputs/tables")   + "/"

# Characteristic lag conventions (months)
LAG_MONTHLY = 1
LAG_QUARTERLY = 4
LAG_ANNUAL = 6

# ─────────────────────────────────────────────
# Train / Validation / Test splits
# ─────────────────────────────────────────────
# Paper: train 1957-1974, val 1975-1986, test 1987-2016
# With rolling refitting: train expands by 1yr, val rolls forward 12mo
TRAIN_START = 195703
TRAIN_END_INIT = 197412    # initial training end
VAL_YEARS = 12             # validation window in years
TEST_START = 198701
TEST_END = 201612
REFIT_FREQ = 12            # refit every 12 months

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────
# Number of latent factors
K_FACTORS = [1, 2, 3, 4, 5, 6]
K_DEFAULT = 5

# Beta network hidden layer configs (neurons per layer)
ARCHITECTURES = {
    "CA0": [],              # linear: no hidden layers
    "CA1": [32],            # one hidden layer, 32 neurons
    "CA2": [32, 16],        # two hidden layers
    "CA3": [32, 16, 8],     # three hidden layers
}

# Factor network: always one linear layer (no hidden layers)
# Input dim = P+1 (94 characteristics + 1 market portfolio)
N_CHARACTERISTICS = 94
N_MANAGED_PORTFOLIOS = N_CHARACTERISTICS + 1  # +1 for equal-weighted market

# Activation function
ACTIVATION = "relu"  # ReLU throughout

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 500
PATIENCE = 10              # early stopping patience

# Regularization
L1_LAMBDA = 1e-5           # LASSO penalty weight (tuned on val)
L1_LAMBDA_GRID = [0, 1e-6, 1e-5, 1e-4, 1e-3]

# Ensemble
N_ENSEMBLE_SEEDS = 10      # number of random seeds for ensemble averaging

# Adam optimizer parameters (from paper's Algorithm 2)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Batch normalization
USE_BATCH_NORM = True

# ─────────────────────────────────────────────
# Extensions (Plans A and B)
# ─────────────────────────────────────────────
# Plan B: Energy-based regularization
ENERGY_LAMBDA_GRID = [0, 1e-4, 1e-3, 1e-2, 1e-1]
ENERGY_LAMBDA_DEFAULT = 1e-2
SHARPE_THRESHOLD = 2.0     # kappa: annualized Sharpe cap

# Plan B: Disentanglement penalty
DISENTANGLE_GAMMA_GRID = [0, 1e-4, 1e-3, 1e-2, 1e-1]
DISENTANGLE_GAMMA_DEFAULT = 1e-3

# Plan A: Anomaly scoring
ANOMALY_WINDOW = 12        # rolling window for energy score (months)
N_QUANTILES = 5            # quintile sorts for portfolio analysis
