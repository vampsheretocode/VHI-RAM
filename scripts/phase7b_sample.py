import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import lightgbm as lgb
import os

def main():
    print("PHASE 7B: EXTREME EVENT SMALL SAMPLE RUN")
    
    # 1. Load a sample from probabilistic forecasts
    file_path = 'data/probabilistic/probabilistic_forecasts_2020.parquet'
    
    # To get a good mix of splits, we'll read specific columns for the whole dataset, then sample
    cols = [
        'init_time', 'lead_time_hours', 'split', 'hres_forecast', 'pangu_forecast',
        'p10', 'p50', 'p90', 'interval_width', 'forecast_disagreement', 'observation_value'
    ]
    df = pq.read_table(file_path, columns=cols).to_pandas()
    
    # Sample 50,000 rows to ensure we get some extreme events
    df_sample = df.sample(n=50000, random_state=42).copy()
    
    # 2. Calculate Threshold (95th percentile of TRAIN observations)
    df_train_temp = df_sample[df_sample['split'] == 'train']
    threshold = np.percentile(df_train_temp['observation_value'].dropna(), 95)
    print(f"TRAIN-derived Extreme Event Threshold (95th Percentile): {threshold:.4f} K")
    
    # 3. Create observed_event target
    df_sample['event_threshold'] = threshold
    df_sample['observed_event'] = (df_sample['observation_value'] >= threshold).astype(int)
    
    # 4. Extract TRAIN split again
    df_train = df_sample[df_sample['split'] == 'train'].copy()
    
    # 5. Define features
    features = [
        'p50', 'p90', 'p10', 'interval_width', 'forecast_disagreement', 'lead_time_hours',
        'hres_forecast', 'pangu_forecast'
    ]
    
    X_train = df_train[features]
    y_train = df_train['observed_event']
    
    print(f"Training on {len(X_train)} rows...")
    print(f"Extreme event frequency in TRAIN: {y_train.mean():.4f}")
    
    # 6. Train simple LightGBM Classifier
    model = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        class_weight='balanced', # Important for imbalanced extreme events
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 7. Predict on TEST split
    df_test = df_sample[df_sample['split'] == 'test'].copy()
    
    X_test = df_test[features]
    probs = model.predict_proba(X_test)[:, 1] # Probability of class 1 (Extreme)
    
    df_test['predicted_extreme_probability'] = probs
    # Use 0.5 as initial decision threshold (since we balanced the classes)
    df_test['predicted_event'] = (probs > 0.5).astype(int)
    
    print(f"Evaluating on {len(df_test)} TEST rows...")
    
    # 8. Show output
    display_cols = [
        'p50', 'p10', 'p90',
        'event_threshold', 'observation_value', 'observed_event',
        'predicted_extreme_probability', 'predicted_event'
    ]
    
    # Sort to show some extreme events first
    df_show = df_test.sort_values('observation_value', ascending=False)
    
    print("\n--- SAMPLE OUTPUT ROWS (High Obs First) ---")
    print(df_show[display_cols].head(5).to_string(index=False))
    
    print("\n--- SAMPLE OUTPUT ROWS (Random) ---")
    print(df_test[display_cols].sample(5, random_state=42).to_string(index=False))

if __name__ == "__main__":
    main()
