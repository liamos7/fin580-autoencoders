"""
run.py — End-to-end pipeline for autoencoder asset pricing.

Usage:
    python run.py --data data/raw/panel.csv --architecture CA2 --K 5

Steps:
    1. Load and preprocess data
    2. Construct managed portfolios
    3. Train model (with ensemble and early stopping)
    4. Evaluate out-of-sample
    5. Run extensions (Plans A and B)
    6. Save results
"""

import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
from src.data_loader import build_panel, split_panel, prepare_tensors
from train_test_val.managed_portfolios import compute_managed_portfolios_from_tensors
from train_test_val.models import build_model
from train_test_val.train import AssetPricingDataset, train_ensemble, tune_l1
from train_test_val.evaluate import evaluate_model, compare_models


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Asset Pricing")
    parser.add_argument("--data", type=str, default=config.RAW_DIR + "panel.csv",
                        help="Path to raw stock-month panel CSV")
    parser.add_argument("--processed", type=str, default=None,
                        help="Path to already-processed parquet (skips build_panel)")
    parser.add_argument("--architecture", type=str, default="CA2",
                        choices=["IPCA", "CA0", "CA1", "CA2", "CA3"],
                        help="Model architecture")
    parser.add_argument("--K", type=int, default=config.K_DEFAULT,
                        help="Number of latent factors")
    parser.add_argument("--tune-l1", action="store_true",
                        help="Tune L1 penalty on validation set")
    parser.add_argument("--run-extensions", action="store_true",
                        help="Run Plan A and Plan B extensions")
    parser.add_argument("--compare-all", action="store_true",
                        help="Train and compare all architectures")
    parser.add_argument("--seeds", type=int, default=config.N_ENSEMBLE_SEEDS,
                        help="Number of ensemble seeds")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directories
    for d in [config.OUTPUT_DIR, config.MODEL_DIR, config.FIGURE_DIR, config.TABLE_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # ── Step 1: Load and preprocess data ──
    print("\n" + "=" * 60)
    print("  STEP 1: Data Loading and Preprocessing")
    print("=" * 60)
    
    if args.processed:
        print(f"  Loading processed panel from {args.processed}...")
        df = pd.read_parquet(args.processed)
        char_cols = [c for c in df.columns if c not in ("permno", "date", "ret", "ret_excess")]
        print(f"  {len(df):,} stock-months, {len(char_cols)} characteristics")
    else:
        df, char_cols = build_panel(args.data)
    P = len(char_cols)
    
    # Split into train / val / test
    train_df, val_df, test_df = split_panel(df, train_end=200512, val_end=201012)
    
    # Convert to tensor format
    print("\n  Converting to tensors...")
    ret_train, chr_train, msk_train, dates_train = prepare_tensors(train_df, char_cols)
    ret_val, chr_val, msk_val, dates_val = prepare_tensors(val_df, char_cols)
    ret_test, chr_test, msk_test, dates_test = prepare_tensors(test_df, char_cols)
    
    # ── Step 2: Managed portfolios ──
    print("\n" + "=" * 60)
    print("  STEP 2: Managed Portfolio Construction")
    print("=" * 60)
    
    mp_train = compute_managed_portfolios_from_tensors(ret_train, chr_train, msk_train)
    mp_val = compute_managed_portfolios_from_tensors(ret_val, chr_val, msk_val)
    mp_test = compute_managed_portfolios_from_tensors(ret_test, chr_test, msk_test)
    
    MP = mp_train.shape[1]  # P + 1
    
    # Build datasets
    train_data = AssetPricingDataset(ret_train, chr_train, mp_train, msk_train)
    val_data = AssetPricingDataset(ret_val, chr_val, mp_val, msk_val)
    test_data = AssetPricingDataset(ret_test, chr_test, mp_test, msk_test)
    
    print(f"\n  Dataset sizes: train={train_data.n_obs:,}, "
          f"val={val_data.n_obs:,}, test={test_data.n_obs:,}")
    
    # ── Step 3: Train model ──
    print("\n" + "=" * 60)
    print(f"  STEP 3: Training {args.architecture} with K={args.K}")
    print("=" * 60)
    
    if args.compare_all:
        # Train all architectures for comparison
        all_results = []
        
        for arch in ["IPCA", "CA0", "CA1", "CA2", "CA3"]:
            for K in config.K_FACTORS:
                print(f"\n  Training {arch} K={K}...")
                
                models = train_ensemble(
                    arch, P, MP, K, train_data, val_data,
                    n_seeds=args.seeds, verbose=False,
                )
                
                results = evaluate_model(
                    models, test_data,
                    model_name=f"{arch}_K{K}",
                    verbose=False,
                )
                results["architecture"] = arch
                results["K"] = K
                all_results.append(results)
                
                print(f"    R²_total={results['r2_total']*100:.2f}%, "
                      f"R²_pred={results['r2_pred']*100:.2f}%, "
                      f"SR={results['sharpe_ew']:.2f}")
        
        comparison = pd.DataFrame(all_results)
        comparison.to_csv(config.TABLE_DIR + "model_comparison.csv", index=False)
        print(f"\n  Comparison saved to {config.TABLE_DIR}model_comparison.csv")

        # Pick best model by R²_total for extensions
        best = max(all_results, key=lambda r: r["r2_total"])
        best_arch, best_K = best["architecture"], best["K"]
        print(f"\n  Best model: {best_arch} K={best_K} "
              f"(R²_total={best['r2_total']*100:.2f}%)")

        if args.run_extensions:
            print(f"\n  Retraining {best_arch} K={best_K} for extensions...")
            models = train_ensemble(
                best_arch, P, MP, best_K, train_data, val_data,
                n_seeds=args.seeds, verbose=False,
            )
            args.architecture = best_arch
            args.K = best_K

    else:
        # Train single architecture
        l1_lambda = config.L1_LAMBDA
        if args.tune_l1:
            print("\n  Tuning L1 penalty...")
            l1_lambda = tune_l1(
                args.architecture, P, MP, args.K,
                train_data, val_data,
            )
        
        models = train_ensemble(
            args.architecture, P, MP, args.K,
            train_data, val_data,
            n_seeds=args.seeds,
            l1_lambda=l1_lambda,
            save_dir=config.MODEL_DIR,
            verbose=args.verbose,
        )
        
        # ── Step 4: Evaluate ──
        print("\n" + "=" * 60)
        print("  STEP 4: Out-of-Sample Evaluation")
        print("=" * 60)

        evaluate_model(
            models, test_data,
            model_name=f"{args.architecture}_K{args.K}",
            char_cols=char_cols,
            verbose=args.verbose,
        )

    # ── Step 5: Extensions (Plans A & B) ──
    if args.run_extensions:
        print("\n" + "=" * 60)
        print("  STEP 5: Extensions (Plans A & B)")
        print(f"  Model: {args.architecture} K={args.K}")
        print("=" * 60)

        from train_test_val.extensions import (
            compute_anomaly_scores,
            portfolio_sort_analysis,
            predictive_regression,
            transition_analysis,
            run_energy_ablation,
        )

        # Plan A: Anomaly scoring
        print("\n  Plan A: Anomaly scoring on residuals...")
        residuals, sq_res, pred_ret, act_ret = compute_anomaly_scores(
            models, test_data
        )

        sort_results = portfolio_sort_analysis(
            test_data, residuals, act_ret
        )
        print("\n  Portfolio sort results:")
        print(sort_results.to_string(index=False))
        sort_results.to_csv(config.TABLE_DIR + "plan_a_portfolio_sorts.csv", index=False)

        reg_results = predictive_regression(
            test_data, residuals, pred_ret, act_ret
        )
        print(f"\n  Predictive regression:")
        for k, v in reg_results.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        print("\n  Transition analysis (low→high anomaly)...")
        trans_results = transition_analysis(
            test_data, residuals, act_ret, test_df
        )
        print(f"  Transitions found: {trans_results['n_transitions']}, "
              f"Controls: {trans_results['n_controls']}")
        print(trans_results["summary"].to_string(index=False))
        trans_results["summary"].to_csv(
            config.TABLE_DIR + "plan_a_transitions.csv", index=False
        )

        # Plan B: Energy regularization ablation
        print("\n  Plan B: Energy regularization ablation...")
        ablation_results = run_energy_ablation(
            args.architecture, P, MP, args.K,
            train_data, val_data, test_data,
        )
        print("\n  Ablation results:")
        print(ablation_results.to_string(index=False))
        ablation_results.to_csv(config.TABLE_DIR + "plan_b_ablation.csv", index=False)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
