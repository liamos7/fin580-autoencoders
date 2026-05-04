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
from train_test_val.evaluate import evaluate_model, compare_models, compute_long_short_sharpe, test_hl_significance


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
    parser.add_argument("--run-backtest", action="store_true",
                        help="Run predictive long-short backtest vs market")
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

                # Collect factor history from training set
                from train_test_val.evaluate import collect_factor_history
                fh = collect_factor_history(models, train_data)
                
                results = evaluate_model(
                    models, test_data,
                    model_name=f"{arch}_K{K}",
                    verbose=False,
                    factor_history=fh,
                )
                results["architecture"] = arch
                results["K"] = K
                all_results.append(results)
                
                print(f"    R²_total={results['r2_total']*100:.2f}%, "
                      f"R²_pred={results['r2_pred']*100:.2f}%, "
                      f"SR_pred={results['sharpe_pred']:.2f}, "
                      f"SR_chars={results['sharpe_chars_only']:.2f}, "
                      f"SR_contemp={results['sharpe_contemp']:.2f}")
        
        comparison = pd.DataFrame(all_results)
        comparison.to_csv(config.TABLE_DIR + "model_comparison.csv", index=False)
        print(f"\n  Comparison saved to {config.TABLE_DIR}model_comparison.csv")

        # ── Robustness summary: median/IQR across all configurations ──
        pred_srs = comparison["sharpe_pred"].values
        chars_sr = comparison["sharpe_chars_only"].iloc[0]  # same for all configs
        r2_totals = comparison["r2_total"].values * 100

        print(f"\n  {'─'*50}")
        print(f"  Robustness Summary ({len(pred_srs)} configurations)")
        print(f"  {'─'*50}")
        print(f"  Predictive Sharpe:")
        print(f"    Median:  {np.median(pred_srs):.2f}")
        print(f"    Mean:    {np.mean(pred_srs):.2f}")
        print(f"    IQR:     [{np.percentile(pred_srs, 25):.2f}, "
              f"{np.percentile(pred_srs, 75):.2f}]")
        print(f"    Range:   [{np.min(pred_srs):.2f}, {np.max(pred_srs):.2f}]")
        print(f"    Configs with SR > chars-only ({chars_sr:.2f}): "
              f"{(pred_srs > chars_sr).sum()}/{len(pred_srs)}")
        print(f"  Chars-only benchmark SR: {chars_sr:.2f}")
        print(f"  R²_total:")
        print(f"    Median:  {np.median(r2_totals):.2f}%")
        print(f"    Best:    {np.max(r2_totals):.2f}%")

        # Save robustness summary
        robustness = pd.DataFrame([{
            "n_configs": len(pred_srs),
            "sr_pred_median": np.median(pred_srs),
            "sr_pred_mean": np.mean(pred_srs),
            "sr_pred_q25": np.percentile(pred_srs, 25),
            "sr_pred_q75": np.percentile(pred_srs, 75),
            "sr_pred_min": np.min(pred_srs),
            "sr_pred_max": np.max(pred_srs),
            "sr_chars_only": chars_sr,
            "n_configs_above_chars": int((pred_srs > chars_sr).sum()),
            "r2_total_median": np.median(r2_totals),
            "r2_total_max": np.max(r2_totals),
        }])
        robustness.to_csv(config.TABLE_DIR + "robustness_summary.csv", index=False)

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

        # Collect factor history from training period for seeding λ̂
        from train_test_val.evaluate import collect_factor_history
        print("  Collecting training-period factor history...")
        factor_history = collect_factor_history(models, train_data)
        print(f"  Factor history: {len(factor_history)} months from training set")

        evaluate_model(
            models, test_data,
            model_name=f"{args.architecture}_K{args.K}",
            char_cols=char_cols,
            verbose=args.verbose,
            factor_history=factor_history,
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

        import math

        sort_results, anomaly_hl_series = portfolio_sort_analysis(
            test_data, residuals, act_ret, panel_df=test_df, return_hl_series=True
        )
        print("\n  Portfolio sort results:")
        print(sort_results.to_string(index=False))
        sort_results.to_csv(config.TABLE_DIR + "plan_a_portfolio_sorts.csv", index=False)

        # H-L significance: anomaly-score quintile sort
        print("\n  H-L significance (anomaly quintile sort)...")
        if len(anomaly_hl_series) > 0:
            nw_lags = math.floor(len(anomaly_hl_series) ** 0.25)
            sig_anomaly = test_hl_significance(anomaly_hl_series, newey_west_lags=nw_lags)
            print(sig_anomaly.to_string(index=False))
            sig_anomaly.to_csv(config.TABLE_DIR + "plan_a_hl_significance_anomaly.csv", index=False)

        # H-L significance: predicted-return decile sort
        print("\n  H-L significance (predicted-return decile sort)...")
        _, pred_hl_series = compute_long_short_sharpe(models, test_data, return_series=True)
        if len(pred_hl_series) > 0:
            nw_lags = math.floor(len(pred_hl_series) ** 0.25)
            sig_pred = test_hl_significance(pred_hl_series, newey_west_lags=nw_lags)
            print(sig_pred.to_string(index=False))
            sig_pred.to_csv(config.TABLE_DIR + "plan_a_hl_significance_pred.csv", index=False)

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

    # ── Step 6: Backtest ──
    if args.run_backtest:
        print("\n" + "=" * 60)
        print("  STEP 6: Predictive Long-Short Backtest")
        print(f"  Model: {args.architecture} K={args.K}")
        print("=" * 60)

        from train_test_val.backtest import run_full_backtest

        # Ensure factor_history is available
        if 'factor_history' not in dir():
            from train_test_val.evaluate import collect_factor_history
            print("  Collecting training-period factor history...")
            factor_history = collect_factor_history(models, train_data)

        backtest_results = run_full_backtest(
            models, test_data, test_df,
            output_dir=config.TABLE_DIR,
            factor_history=factor_history,
            lambda_window=60,
            verbose=args.verbose,
        )

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()