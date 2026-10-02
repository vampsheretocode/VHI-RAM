import pyarrow.parquet as pq
import pandas as pd
import numpy as np

def main():
    print("--- PHASE 7B AUDIT CALCULATION ---")
    cols = ['init_time', 'split', 'observation_value', 'latitude', 'longitude', 'region']
    df = pq.read_table('data/verification/verification_2020.parquet', columns=cols).to_pandas()
    
    df_train = df[df['split'] == 'train'].copy()
    
    threshold = np.percentile(df_train['observation_value'].dropna(), 95)
    
    total_obs = len(df_train)
    df_train['is_extreme'] = (df_train['observation_value'] >= threshold).astype(int)
    extreme_count = df_train['is_extreme'].sum()
    event_rate = df_train['is_extreme'].mean()
    unique_cells = df_train[['latitude', 'longitude']].drop_duplicates().shape[0]
    
    print(f"Total TRAIN obs: {total_obs}")
    print(f"TRAIN extreme-event count: {extreme_count}")
    print(f"TRAIN event rate: {event_rate:.4f}")
    print(f"Unique grid cells: {unique_cells}")
    
    print("\n--- Event rate by region ---")
    if 'region' in df_train.columns:
        print(df_train.groupby('region')['is_extreme'].mean())
        print("\n--- 95th Percentile by region ---")
        print(df_train.groupby('region')['observation_value'].apply(lambda x: np.percentile(x.dropna(), 95)))
    else:
        print("Region not available.")
        
    print("\n--- Event rate by month ---")
    df_train['month'] = df_train['init_time'].dt.month
    print(df_train.groupby('month')['is_extreme'].mean())
    
    print("\n--- 95th Percentile by month ---")
    print(df_train.groupby('month')['observation_value'].apply(lambda x: np.percentile(x.dropna(), 95)))

if __name__ == "__main__":
    main()
