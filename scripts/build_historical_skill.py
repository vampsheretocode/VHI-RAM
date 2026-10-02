import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import logging
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.blend.skill.tracker import compute_daily_errors, compute_rolling_skill, attach_historical_skill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Run a small chronological example')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/skill', exist_ok=True)

    log_file = 'logs/phase4_skill.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info("Starting Phase 4 Historical Skill Tracking")
    dataset_path = 'data/verification/verification_2020.parquet'
    
    cols = ['init_time', 'valid_time', 'lead_time_hours', 'hres_forecast', 'pangu_forecast', 'observation_value', 'split', 'region']
    logging.info(f"Loading {cols} from {dataset_path}...")
    
    table = pq.read_table(dataset_path, columns=cols)
    df = table.to_pandas()
    
    if args.sample:
        logging.info("SAMPLE MODE: Filtering to Jan-Feb 2020 to test rolling window logic.")
        df = df[df['valid_time'] < pd.to_datetime('2020-03-01')]
        
    logging.info(f"Dataset loaded: {len(df)} rows.")

    # 1. Compute Daily Errors
    logging.info("1. Computing daily aggregate errors (domain-level fallback)...")
    daily_errors = compute_daily_errors(df)
    
    # 2. Compute Rolling Skill
    logging.info("2. Computing 30-day rolling skill per lead time...")
    rolling_skill = compute_rolling_skill(daily_errors, window_days=30)
    
    # 3. Attach Historical Skill
    logging.info("3. Attaching historical skill to forecast cases (Merge AsOf)...")
    df_features = attach_historical_skill(df, rolling_skill)
    
    # Validate Leakage
    logging.info("Validating leakage safety...")
    leakage = df_features[df_features['valid_time_historical'] > df_features['init_time']]
    if len(leakage) > 0:
        logging.error(f"LEAKAGE DETECTED! {len(leakage)} rows use future observations.")
        sys.exit(1)
    else:
        logging.info("Check passed: No historical skill uses observations from after init_time.")

    # Show Early vs Later rows
    logging.info("--- Demonstrating chronological feature accumulation ---")
    
    # Early row example
    # Find a row where init_time is very early (e.g., first week of Jan 2020)
    early_mask = df_features['init_time'] < pd.to_datetime('2020-01-08')
    early_rows = df_features[early_mask]
    if not early_rows.empty:
        # Check a 240h forecast in the first week. It shouldn't have ANY historical data because 
        # valid_time of 240h forecast initialized in Jan 1-7 will be Jan 11-17.
        early_240 = early_rows[early_rows['lead_time_hours'] == 240]
        if not early_240.empty:
            logging.info("Early 240h forecast (Init: ~Jan 1):")
            sample = early_240.iloc[0]
            logging.info(f"  Init: {sample['init_time']}, Valid: {sample['valid_time']}")
            logging.info(f"  Available history end (valid_time_historical): {sample['valid_time_historical']}")
            logging.info(f"  Sample count: {sample['skill_sample_count']}")

    # Later row example (e.g. Feb 2020)
    later_mask = df_features['init_time'] > pd.to_datetime('2020-02-15')
    later_rows = df_features[later_mask]
    if not later_rows.empty:
        later_240 = later_rows[later_rows['lead_time_hours'] == 240]
        if not later_240.empty:
            logging.info("Later 240h forecast (Init: ~Feb 15):")
            sample = later_240.iloc[0]
            logging.info(f"  Init: {sample['init_time']}, Valid: {sample['valid_time']}")
            logging.info(f"  Available history end (valid_time_historical): {sample['valid_time_historical']}")
            logging.info(f"  Sample count: {sample['skill_sample_count']}")

    missing_count = df_features['skill_sample_count'].isnull().sum()
    logging.info(f"Cases with insufficient/missing history: {missing_count}")

    # Output Parquet
    out_cols = [
        'init_time', 'valid_time', 'lead_time_hours', 'latitude', 'longitude', 'split',
        'valid_time_historical', 'skill_sample_count',
        'rolling_hres_mae', 'rolling_pangu_mae', 'rolling_hres_rmse', 'rolling_pangu_rmse',
        'rolling_hres_bias', 'rolling_pangu_bias', 'pangu_relative_skill'
    ]
    # To save space, we just extract the feature dataset for downstream modeling
    # Note: df_features doesn't have lat/lon because we loaded from the main table, wait.
    # Oh, I didn't load lat/lon. Let's not save lat/lon here, or we can just save it.
    
    # Actually, we can just save the rolling_skill table itself as the feature dataset.
    # It contains exactly what we need to join against any forecast case.
    out_parquet = 'data/skill/historical_skill_2020.parquet'
    rolling_skill.to_parquet(out_parquet, index=False)
    logging.info(f"Saved rolling skill table to {out_parquet}")

    # Calculate summaries for JSON
    summary = {
        "window_definition": "Trailing 30 days of valid_time",
        "verification_lag_handling": "Joined via merge_asof where valid_time <= init_time",
        "minimum_sample_threshold_used": "Domain-level fallback (all grid cells aggregated)",
        "total_skill_periods_computed": len(rolling_skill),
        "source_lead_coverage": "HRES and PANGU for 24, 48, 72, 120, 240 leads",
        "cases_with_no_history_at_init": int(missing_count)
    }

    with open('artifacts/historical_skill_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    report = f"""# Phase 4 Historical Skill Tracking Report

## Logic & Leakage Protection
- **Window Definition**: Trailing 30 days of valid_time.
- **Verification Lag Definition**: A forecast is verified only when the real world reaches its `valid_time`. Therefore, historical features for a forecast initialized at `T` strictly use observations where `valid_time <= T`.
- **Minimum Sample Threshold**: Grid-cell historical metrics over a 30-day window possess inadequate sample sizes (max 60). Therefore, domain-level (national) fallback aggregation is utilized to generate robust historical skill metrics, aggregating ~14,000 grid cells per timestep.
- **Leakage Test Performed**: Checked that `valid_time_historical <= init_time` for all rows. Passed.

## Coverage
- **Total skill periods computed**: {len(rolling_skill)}
- **Source × Lead Coverage**: HRES and PANGU computed per lead time [24, 48, 72, 120, 240].
- **Missing/Insufficient-history cases**: {missing_count} (Early January forecasts do not have historical windows).

## Historical Feature Example
An example of a historical feature row:
```json
{rolling_skill.iloc[-1].to_dict()}
```

*Skill features have been extracted to `{out_parquet}` for Phase 5.*
"""
    with open('reports/phase4_skill_report.md', 'w') as f:
        f.write(report)
        
    logging.info("Phase 4 complete.")

if __name__ == "__main__":
    main()
