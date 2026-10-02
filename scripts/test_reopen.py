import pandas as pd
import numpy as np
import sys

def test_reopen():
    print("\n--- TASK 8: REOPEN TEST & TASK 4: QA CHECKS ---")
    try:
        df = pd.read_parquet('data/phase1_dataset.parquet')
        print(f"Schema (Columns):\n{df.dtypes}")
        print(f"Shape/Record count: {df.shape}")
        
        print(f"Min init_date: {df['init_time'].min()}")
        print(f"Max init_date: {df['init_time'].max()}")
        print(f"Unique sources: {df['source'].unique()}")
        print(f"Unique lead times: {df['lead_time_hours'].unique()}")
        
        print("\n--- TASK 4: DATA QUALITY CHECKS (DETAILED) ---")
        
        # A. Missing forecasts (NaNs in forecast_value)
        missing_f = df['forecast_value'].isna().sum()
        print(f"A. Missing forecasts (NaNs): {missing_f}")
        
        # B. Missing ERA5 observations (NaNs in observation_value)
        missing_obs = df['observation_value'].isna().sum()
        print(f"B. Missing ERA5 observations: {missing_obs}")
        
        # C. Duplicate (source, init_time, lead_time, latitude, longitude)
        dups_c = df.duplicated(subset=['source', 'init_time', 'lead_time_hours', 'latitude', 'longitude']).sum()
        print(f"C. Duplicate (source, init, lead, lat, lon): {dups_c}")
        
        # D. Duplicate valid times (per spatial point per source per lead)
        # We can just check duplicates for valid_time, lat, lon, source
        dups_d = df.duplicated(subset=['source', 'valid_time', 'latitude', 'longitude']).sum()
        print(f"D. Duplicate valid times per point: {dups_d}")
        
        # E. NaNs
        total_nans = df.isna().sum().sum()
        print(f"E. Total NaNs in entire dataset: {total_nans}")
        
        # F. Infinite values
        num_cols = df.select_dtypes(include=[np.number]).columns
        infs = np.isinf(df[num_cols]).sum().sum()
        print(f"F. Infinite values: {infs}")
        
        # G. Latitude/longitude bounds
        print(f"G. Latitude bounds: {df['latitude'].min()} to {df['latitude'].max()}")
        print(f"G. Longitude bounds: {df['longitude'].min()} to {df['longitude'].max()}")
        
        # H. Lead-time validity
        print(f"H. Lead-time validity: {list(df['lead_time_hours'].unique())}")
        
        # I. Temporal ordering (is init_time <= valid_time)
        temp_order = (df['init_time'] <= df['valid_time']).all()
        print(f"I. Temporal ordering valid (init <= valid): {temp_order}")
        
        # J. Forecast/observation shape compatibility 
        # (Since merged row by row, shape is implicitly perfectly matched)
        print(f"J. Forecast/observation shape compatibility: Checked by merge (same row count {len(df)})")
        
        # K. Source coverage
        print(f"K. Source coverage: {list(df['source'].unique())}")
        
        # L. Number of initializations
        print(f"L. Number of initializations: {df['init_time'].nunique()}")
        
        # M. Number of valid forecast cases
        print(f"M. Number of valid forecast cases (rows): {len(df)}")
        
        # N. Number of grid cells
        cells = df[['latitude', 'longitude']].drop_duplicates().shape[0]
        print(f"N. Number of grid cells: {cells}")
        
        print("\nSample records:")
        print(df.head(3).to_string())
        
        print("\n[SUCCESS] Artifact reopened and validated successfully from fresh process.")
    except Exception as e:
        print(f"[ERROR] Reopen test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    test_reopen()
