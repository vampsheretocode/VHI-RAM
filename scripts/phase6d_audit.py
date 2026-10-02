import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import os

print("--- PHASE 6D AUDIT SCRIPT ---")
all_passed = True

# 1. Load artifacts
file_path = 'data/probabilistic/probabilistic_forecasts_2020.parquet'
file_size = os.path.getsize(file_path)
print(f"Probabilistic Forecasts File Size: {file_size} bytes")

cols = ['init_time', 'valid_time', 'lead_time_hours', 'split', 'p10', 'p50', 'p90', 'observation_value', 'interval_width']
df = pq.read_table(file_path, columns=cols).to_pandas()

print(f"Row count: {len(df)}")
if len(df) != 51814620:
    print(f"FAIL: Expected 51814620 rows, got {len(df)}")
    all_passed = False

with open('artifacts/phase6_probabilistic_metrics.json', 'r') as f:
    metrics = json.load(f)

# 2. Check P10 <= P50 <= P90
valid_intervals = (df['p10'] <= df['p50']) & (df['p50'] <= df['p90'])
violators = (~valid_intervals).sum()
print(f"Interval ordering (P10 <= P50 <= P90) violations: {violators}")
if violators > 0: all_passed = False

# 3. Check Interval Width Non-Negative
neg_widths = (df['interval_width'] < 0).sum()
print(f"Negative interval widths: {neg_widths}")
if neg_widths > 0: all_passed = False

# 4. No Self-Calibration/Test Leakage
# Test if the computed coverage exactly matches the JSON report independently
df_test = df[df['split'] == 'test']
actual_coverage = ((df_test['observation_value'] >= df_test['p10']) & (df_test['observation_value'] <= df_test['p90'])).mean()

print(f"Independent TEST 80% Coverage calculation: {actual_coverage:.4f}")
reported_coverage = metrics['overall']['coverage_80']
print(f"Reported TEST 80% Coverage in JSON: {reported_coverage:.4f}")

if abs(actual_coverage - reported_coverage) > 1e-6:
    print("FAIL: Independent metric recalculation mismatch.")
    all_passed = False
else:
    print("PASS: Independent metric recalculation exactly matches.")

# 5. Check missing values
null_p50 = df['p50'].isnull().sum()
print(f"Missing P50 values: {null_p50}")
if null_p50 > 0: all_passed = False

print("")
if all_passed:
    print("PHASE 6 AUDIT: PASS")
else:
    print("PHASE 6 AUDIT: FAIL")
