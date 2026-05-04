import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# Add src and train_test_val to path
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('train_test_val'))

import config
from src.data_loader import split_panel, prepare_tensors
from train_test_val.managed_portfolios import compute_managed_portfolios_from_tensors
from train_test_val.models import build_model
from train_test_val.train import ensemble_predict_month

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="CA2", help="Model architecture")
    parser.add_argument("--k", type=int, default=5, help="Number of factors")
    parser.add_argument("--processed", type=str, default=config.PROCESSED_DIR + "panel_processed.parquet", help="Path to processed parquet")
    args = parser.parse_args()
    
    print(f"\n--- Generating Reproduction Errors for {args.arch} K={args.k} ---")
    
    # 1. Load Data
    print(f"\n1. Loading data panel from {args.processed}...")
    df = pd.read_parquet(args.processed)
    char_cols = [c for c in df.columns if c not in ("permno", "date", "ret")]
    P = len(char_cols)
    MP = P + 1

    # 2. Split Data
    print("2. Splitting test set & freeing memory...")
    _, _, test_df = split_panel(df, train_end=200512, val_end=201012)
    
    # Free memory immediately
    del df
    import gc
    gc.collect()

    # 3. Convert to Tensors
    print("3. Converting to tensors (this is exactly how run.py does it)...")
    ret_test, chr_test, msk_test, dates_arr = prepare_tensors(test_df, char_cols)
    mp_test = compute_managed_portfolios_from_tensors(ret_test, chr_test, msk_test)
    
    # 4. Load Models
    print(f"4. Loading {config.N_ENSEMBLE_SEEDS} models to {config.DEVICE}...")
    models = []
    for seed in range(config.N_ENSEMBLE_SEEDS):
        model = build_model(args.arch, P, MP, args.k).to(config.DEVICE)
        model_path = f"{config.MODEL_DIR}{args.arch}_K{args.k}_seed{seed}.pt"
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        models.append(model)
        
    # 5. Generate Predictions
    print("5. Generating predictions...")
    records = []
    
    for t in tqdm(range(len(dates_arr))):
        date = dates_arr[t]
        valid = msk_test[t]
        if not np.any(valid): continue
        
        chars_t = torch.tensor(chr_test[t, valid], dtype=torch.float32, device=config.DEVICE)
        ret_t = torch.tensor(ret_test[t, valid], dtype=torch.float32, device=config.DEVICE)
        mp_t = torch.tensor(mp_test[t], dtype=torch.float32, device=config.DEVICE)
        
        with torch.no_grad():
            r_hat, _, _ = ensemble_predict_month(models, chars_t, mp_t)
            
        r_hat = r_hat.cpu().numpy()
        ret_actual = ret_t.cpu().numpy()
        residual = ret_actual - r_hat
        
        month_df = test_df[test_df['date'] == date]
        permnos = month_df['permno'].values
        
        for i in range(len(permnos)):
            records.append({
                'permno': permnos[i],
                'date': date,
                'ret': ret_actual[i],
                'predicted_ret': r_hat[i],
                'residual': residual[i],
                'abs_residual': np.abs(residual[i])
            })
            
    # 6. Formatting
    print("6. Formatting dataframe and aligning forward returns...")
    errors_df = pd.DataFrame(records)
    errors_df = errors_df.sort_values(['permno', 'date'])
    errors_df['ret_forward'] = errors_df.groupby('permno')['ret'].shift(-1)
    errors_df = errors_df.dropna(subset=['ret_forward'])
    
    out_file = f"{config.OUTPUT_DIR}reproduction_errors_{args.arch}_K{args.k}.parquet"
    errors_df.to_parquet(out_file, index=False)
    print(f"\nDone! Saved {len(errors_df)} records to {out_file}")

if __name__ == "__main__":
    main()
