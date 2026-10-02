import pandas as pd
import yaml
import sys
import numpy as np
import os

def run_audit(artifact_path, manifest_path):
    print("--- INDEPENDENT LEAKAGE & VALIDATION AUDIT ---")
    
    if not os.path.exists(artifact_path) or not os.path.exists(manifest_path):
        print(f"ERROR: Artifact or manifest not found.")
        sys.exit(1)
        
    print(f"Reading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
        
    print(f"Reading artifact from {artifact_path}...")
    try:
        df = pd.read_parquet(artifact_path)
    except Exception as e:
        print(f"ERROR: Could not read artifact: {e}")
        sys.exit(1)
        
    errors = []
    
    # 1 & 2. Historical period and Leakage
    start_date = manifest.get('climatology_start', '')
    end_date = manifest.get('climatology_end', '')
    print(f"Documented Period: {start_date} to {end_date}")
    if '2020' in start_date or '2020' in end_date:
        errors.append("LEAKAGE DETECTED: 2020 data referenced in manifest dates.")
    if end_date > '2019-12-31':
        errors.append("LEAKAGE DETECTED: end_date extends beyond 2019-12-31.")
        
    # 3. Expected India grid
    lats = df['latitude'].unique()
    lons = df['longitude'].unique()
    expected_lats = 117
    expected_lons = 121
    if len(lats) != expected_lats or len(lons) != expected_lons:
        errors.append(f"Grid shape mismatch. Found {len(lats)}x{len(lons)}, expected {expected_lats}x{expected_lons}")
        
    # 4. Every required day_of_year
    doys = df['day_of_year'].unique()
    if len(doys) != 366 or min(doys) != 1 or max(doys) != 366:
        errors.append(f"DOY representation incomplete. Found {len(doys)} unique DOYs.")
        
    # 5. No duplicates
    expected_rows = expected_lats * expected_lons * 366
    if len(df) != expected_rows:
        errors.append(f"Row count mismatch. Found {len(df)}, expected {expected_rows}")
    if df.duplicated(subset=['latitude', 'longitude', 'day_of_year']).any():
        errors.append("Duplicate coordinate combinations found.")
        
    # 6. No null thresholds
    if df['threshold_95'].isnull().any():
        errors.append("Null threshold values found.")
        
    # 7. Sample count positive
    if (df['sample_count'] <= 0).any():
        errors.append("Found non-positive sample counts.")
        
    # 8. Numerically plausible (2m_temperature in Kelvin is generally 250 to 330)
    min_temp = df['threshold_95'].min()
    max_temp = df['threshold_95'].max()
    print(f"Threshold Range: {min_temp:.2f}K to {max_temp:.2f}K")
    if min_temp < 200 or max_temp > 340:
        errors.append(f"Thresholds out of physically plausible range for 2m temperature: {min_temp}K to {max_temp}K")
        
    # 14. Manifest matches artifact
    if manifest.get('final_row_count') != len(df):
        errors.append("Manifest final_row_count does not match artifact row count.")
        
    if len(errors) > 0:
        print("\nAUDIT FAILED with the following errors:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("\nAUDIT PASSED: No leakage detected. Methodology structurally validated.")

if __name__ == "__main__":
    run_audit(
        "data/extreme_climatology/era5_smooth_doy_95th.parquet",
        "data/extreme_climatology/manifest.yml"
    )
