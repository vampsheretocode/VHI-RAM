import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import logging
import argparse
import sys
import os

# Ensure modules are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.blend.verification.metrics import evaluate_models
from src.blend.weights.static import learn_static_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Run on a small sample for testing')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    log_file = 'logs/phase3_baseline.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info("Starting Phase 3 Baseline Evaluation")
    dataset_path = 'data/verification/verification_2020.parquet'
    
    # Load required columns
    cols = ['lead_time_hours', 'hres_forecast', 'pangu_forecast', 'observation_value', 'split']
    logging.info(f"Loading {cols} from {dataset_path}...")
    
    # Use PyArrow to read specific columns efficiently
    table = pq.read_table(dataset_path, columns=cols)
    df = table.to_pandas()
    
    if args.sample:
        logging.info("Running in SAMPLE mode (100k rows)")
        df = df.head(100000)
    else:
        logging.info(f"Loaded full dataset: {len(df)} rows")
        
    logging.info("Splitting data by 'split' column (Train/Val/Test)...")
    
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    logging.info(f"Train size: {len(df_train)}")
    logging.info(f"Val size: {len(df_val)}")
    logging.info(f"Test size: {len(df_test)}")

    logging.info("1. Calculating static constrained weights on TRAIN only...")
    static_weights = learn_static_weights(df_train)
    logging.info(f"Static weights learned: {static_weights}")
    
    with open('artifacts/phase3_static_weights.json', 'w') as f:
        json.dump(static_weights, f, indent=2)

    w_hres = static_weights['HRES']

    logging.info("2. Evaluating baselines...")
    metrics = {
        'overall': {},
        'per_lead_time': {}
    }

    # Evaluate Overall
    logging.info("Evaluating OVERALL validation and test sets...")
    metrics['overall']['val'] = evaluate_models(df_val, w_hres)
    metrics['overall']['test'] = evaluate_models(df_test, w_hres)
    
    # Evaluate Per Lead Time
    leads = sorted(df['lead_time_hours'].unique())
    for split_name, split_df in [('val', df_val), ('test', df_test)]:
        metrics['per_lead_time'][split_name] = {}
        for lead in leads:
            df_lead = split_df[split_df['lead_time_hours'] == lead]
            metrics['per_lead_time'][split_name][str(lead)] = evaluate_models(df_lead, w_hres)

    logging.info("3. Saving metrics...")
    with open('artifacts/phase3_baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logging.info("4. Generating Markdown Report...")
    report = f"""# Phase 3 Baseline Evaluation Report

## Dataset Details
- Total evaluated rows: {len(df)}
- Train cases: {len(df_train)}
- Validation cases: {len(df_val)}
- Test cases: {len(df_test)}

## Static Constrained Blend Weights
- **HRES Weight:** {static_weights['HRES']:.4f}
- **PANGU Weight:** {static_weights['PANGU']:.4f}
*Weights were computed strictly on the TRAIN partition.*

## Leakage Protection Verification
- Static weights were frozen before evaluation on TEST.
- Training baseline fit did NOT use validation or test rows.
- Only forecast information available at initialization time was evaluated.

## Test Period Performance

### HRES
- MAE: {metrics['overall']['test']['HRES']['MAE']:.4f}
- RMSE: {metrics['overall']['test']['HRES']['RMSE']:.4f}
- Bias: {metrics['overall']['test']['HRES']['Bias']:.4f}

### PANGU
- MAE: {metrics['overall']['test']['PANGU']['MAE']:.4f}
- RMSE: {metrics['overall']['test']['PANGU']['RMSE']:.4f}
- Bias: {metrics['overall']['test']['PANGU']['Bias']:.4f}

### Equal Weight Blend
- MAE: {metrics['overall']['test']['Equal Weight']['MAE']:.4f}
- RMSE: {metrics['overall']['test']['Equal Weight']['RMSE']:.4f}
- Bias: {metrics['overall']['test']['Equal Weight']['Bias']:.4f}

### Static Blend
- MAE: {metrics['overall']['test']['Static Blend']['MAE']:.4f}
- RMSE: {metrics['overall']['test']['Static Blend']['RMSE']:.4f}
- Bias: {metrics['overall']['test']['Static Blend']['Bias']:.4f}

*Per lead-time metrics and validation set performance can be found in `artifacts/phase3_baseline_metrics.json`.*
"""
    with open('reports/phase3_baseline_report.md', 'w') as f:
        f.write(report)
        
    logging.info("Phase 3 complete.")

if __name__ == "__main__":
    main()
