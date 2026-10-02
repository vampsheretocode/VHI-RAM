import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.blend.probabilistic.calibration import ResidualCalibrator

def main():
    print("PHASE 6B: SMALL SAMPLE RUN")
    
    # We load 1000 rows from verification and weights
    # We just need some TRAIN rows and some TEST rows.
    # Because datasets are sorted by init_time, head() will only be TRAIN (Jan).
    # To get a mix, we'll read the whole table with specific columns, then sample.
    
    # But loading 52M rows just to sample 1000 is fast enough in PyArrow, 
    # but let's just do it directly.
    cols_verif = ['init_time', 'lead_time_hours', 'latitude', 'longitude', 'observation_value', 'split']
    df_verif = pq.read_table('data/verification/verification_2020.parquet', columns=cols_verif).to_pandas()
    
    cols_weights = ['init_time', 'lead_time_hours', 'latitude', 'longitude', 
                    'hres_forecast', 'pangu_forecast', 'w_hres', 'w_pangu', 'adaptive_blend']
    df_weights = pq.read_table('data/weights/adaptive_weights_2020.parquet', columns=cols_weights).to_pandas()
    
    # Merge observation
    df = pd.merge(df_weights, df_verif, on=['init_time', 'lead_time_hours', 'latitude', 'longitude'])
    
    # Take a 10,000 row sample to ensure we get a decent residual distribution
    df_sample = df.sample(n=10000, random_state=42)
    
    # Fit calibrator on train split of sample
    df_train = df_sample[df_sample['split'] == 'train']
    
    calibrator = ResidualCalibrator(quantiles=[0.1, 0.9])
    calibrator.fit(df_train)
    
    # Predict on test split of sample
    df_test = df_sample[df_sample['split'] == 'test'].copy()
    
    if len(df_test) == 0:
        print("No test rows in sample, predicting on train instead.")
        df_test = df_train.head(10).copy()
        
    df_pred = calibrator.predict(df_test)
    df_pred['residual'] = df_pred['observation_value'] - df_pred['p50']
    
    display_cols = [
        'hres_forecast', 'pangu_forecast', 'w_hres', 'w_pangu', 
        'p50', 'p10', 'p90', 'observation_value', 'residual', 'forecast_disagreement', 'disagreement_bin'
    ]
    
    print("\n--- SAMPLE ROWS OUTPUT ---")
    print(df_pred[display_cols].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
