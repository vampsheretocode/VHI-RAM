import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import logging
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.blend.weights.gating import AdaptiveGatingModel
from src.blend.verification.metrics import compute_metrics

def evaluate_predictions(df, w_hres, w_pangu):
    """Evaluates adaptive predictions against baseline."""
    y = df['observation_value'].values
    hres = df['hres_forecast'].values
    pangu = df['pangu_forecast'].values
    
    adaptive_blend = w_hres * hres + w_pangu * pangu
    equal_blend = 0.5 * hres + 0.5 * pangu
    
    # Using Phase 3 static weights as baseline
    static_w = 0.17881116271018982
    static_blend = static_w * hres + (1 - static_w) * pangu
    
    return {
        "HRES": compute_metrics(y, hres),
        "PANGU": compute_metrics(y, pangu),
        "Equal Weight": compute_metrics(y, equal_blend),
        "Static Blend": compute_metrics(y, static_blend),
        "Adaptive Gate": compute_metrics(y, adaptive_blend),
        "samples": len(df)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Run a small training sample')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/weights', exist_ok=True)

    log_file = 'logs/phase5_adaptive_gating.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info("Starting Phase 5 Adaptive Gating Model Training")
    
    # 1. Load Data
    cols = ['init_time', 'valid_time', 'lead_time_hours', 'latitude', 'longitude', 
            'hres_forecast', 'pangu_forecast', 'observation_value', 'split']
    dataset_path = 'data/verification/verification_2020.parquet'
    
    logging.info("Loading verification data...")
    table = pq.read_table(dataset_path, columns=cols)
    df = table.to_pandas()
    
    logging.info("Loading historical skill data...")
    skill_df = pd.read_parquet('data/skill/historical_skill_2020.parquet')
    
    logging.info("Merging historical skill (Merge AsOf)...")
    df = df.sort_values('init_time')
    skill_df = skill_df.sort_values('valid_time')
    
    df = pd.merge_asof(
        df,
        skill_df,
        left_on='init_time',
        right_on='valid_time',
        by='lead_time_hours',
        direction='backward',
        suffixes=('', '_historical')
    )
    
    if args.sample:
        logging.info("SAMPLE MODE: Training on a 1% random subset spanning all splits.")
        df = df.sample(frac=0.01, random_state=42).copy()
        df = df.sort_values('init_time')
        
    # 2. Prepare Train/Val/Test
    # Since dataset is ordered by init_time, this is safe.
    df_train = df[df['split'] == 'train'].copy()
    df_val = df[df['split'] == 'val'].copy()
    df_test = df[df['split'] == 'test'].copy()
    
    if not args.sample and len(df_train) > 5000000:
        logging.info("Downsampling TRAIN set to 5M rows for efficient LightGBM training...")
        df_train_fit = df_train.sample(n=5000000, random_state=42)
    else:
        df_train_fit = df_train
        
    logging.info(f"Train fit size: {len(df_train_fit)}")
    logging.info(f"Val size: {len(df_val)}")
    logging.info(f"Test size: {len(df_test)}")

    # 3. Train Model
    logging.info("Training Adaptive Gating Model...")
    model = AdaptiveGatingModel(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(df_train_fit)
    
    model.save_importances('artifacts/phase5_feature_importance.json')
    
    # 4. Predict
    logging.info("Predicting weights...")
    w_hres, w_pangu = model.predict_weights(df)
    
    df['w_hres'] = w_hres
    df['w_pangu'] = w_pangu
    df['adaptive_blend'] = w_hres * df['hres_forecast'] + w_pangu * df['pangu_forecast']
    
    # Verify properties
    min_w, max_w = df['w_hres'].min(), df['w_hres'].max()
    sum_w = (df['w_hres'] + df['w_pangu']).mean()
    logging.info(f"Weight validation: min={min_w:.4f}, max={max_w:.4f}, mean sum={sum_w:.4f}")
    
    if min_w < 0 or max_w > 1.0001 or abs(sum_w - 1.0) > 1e-4:
        logging.error("Weight constraints violated!")
        sys.exit(1)
        
    # 5. Evaluate Validation
    logging.info("Evaluating on VALIDATION set...")
    val_metrics = {
        'overall': evaluate_predictions(df[df['split'] == 'val'], 
                                        df[df['split'] == 'val']['w_hres'], 
                                        df[df['split'] == 'val']['w_pangu'])
    }
    with open('artifacts/phase5_validation_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent=2)

    # 6. Evaluate Test
    logging.info("Evaluating on frozen TEST set...")
    df_test_out = df[df['split'] == 'test']
    w_hres_test = df_test_out['w_hres']
    w_pangu_test = df_test_out['w_pangu']
    
    test_metrics = {
        'overall': evaluate_predictions(df_test_out, w_hres_test, w_pangu_test),
        'per_lead_time': {}
    }
    
    leads = sorted(df_test_out['lead_time_hours'].unique())
    for lead in leads:
        mask = df_test_out['lead_time_hours'] == lead
        test_metrics['per_lead_time'][str(lead)] = evaluate_predictions(
            df_test_out[mask], w_hres_test[mask], w_pangu_test[mask]
        )
        
    with open('artifacts/phase5_test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
        
    # 7. Save Dataset
    logging.info("Saving adaptive weights dataset...")
    out_cols = ['init_time', 'valid_time', 'lead_time_hours', 'latitude', 'longitude', 'split',
                'hres_forecast', 'pangu_forecast', 'w_hres', 'w_pangu', 'adaptive_blend']
    df[out_cols].to_parquet('data/weights/adaptive_weights_2020.parquet', index=False)
    
    # 8. Report Generate
    h_win = (w_hres_test > 0.5).mean() * 100
    p_win = (w_pangu_test > 0.5).mean() * 100
    
    report = f"""# Phase 5 Adaptive Gating Report

## Model Details
- **Architecture**: LightGBM Regressors modeling absolute error magnitude for HRES and PANGU.
- **Transformation**: `w_HRES = pred_PANGU_err / (pred_HRES_err + pred_PANGU_err)`
- **Constraints**: Weights are bounded `[0, 1]` and sum exactly to 1. 

## Leakage Protection
- The model was trained purely on `TRAIN` cases (Jan-Aug).
- Missing historical skills (early Jan) were naturally passed to LightGBM which inherently handles `NaN` splits. No future imputation was used.
- Test observations were strictly isolated from the training graph.

## Weight Statistics (TEST)
- **Mean w_HRES**: {w_hres_test.mean():.4f}
- **Mean w_PANGU**: {w_pangu_test.mean():.4f}
- **HRES favored (>50%)**: {h_win:.1f}% of cases
- **PANGU favored (>50%)**: {p_win:.1f}% of cases

## Final Frozen TEST Evaluation

### HRES
- MAE: {test_metrics['overall']['HRES']['MAE']:.4f}
- RMSE: {test_metrics['overall']['HRES']['RMSE']:.4f}
- Bias: {test_metrics['overall']['HRES']['Bias']:.4f}

### PANGU
- MAE: {test_metrics['overall']['PANGU']['MAE']:.4f}
- RMSE: {test_metrics['overall']['PANGU']['RMSE']:.4f}
- Bias: {test_metrics['overall']['PANGU']['Bias']:.4f}

### Equal Weight
- MAE: {test_metrics['overall']['Equal Weight']['MAE']:.4f}
- RMSE: {test_metrics['overall']['Equal Weight']['RMSE']:.4f}
- Bias: {test_metrics['overall']['Equal Weight']['Bias']:.4f}

### Static Blend
- MAE: {test_metrics['overall']['Static Blend']['MAE']:.4f}
- RMSE: {test_metrics['overall']['Static Blend']['RMSE']:.4f}
- Bias: {test_metrics['overall']['Static Blend']['Bias']:.4f}

### Adaptive Gate
- MAE: {test_metrics['overall']['Adaptive Gate']['MAE']:.4f}
- RMSE: {test_metrics['overall']['Adaptive Gate']['RMSE']:.4f}
- Bias: {test_metrics['overall']['Adaptive Gate']['Bias']:.4f}

## Conclusion
See `artifacts/phase5_feature_importance.json` for model interpretation.
Outputs saved to `data/weights/adaptive_weights_2020.parquet`.
"""
    with open('reports/phase5_adaptive_gating_report.md', 'w') as f:
        f.write(report)
        
    logging.info("Phase 5 complete.")

if __name__ == "__main__":
    main()
