import pandas as pd
import yaml
import os
import sys

parquet_path = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\extreme_climatology\era5_smooth_doy_95th.parquet"
manifest_path = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\extreme_climatology\manifest.yml"

print("--- 1. FILE EXISTENCE ---")
p_exists = os.path.exists(parquet_path)
m_exists = os.path.exists(manifest_path)
print(f"Parquet exists at exact path: {p_exists}")
print(f"Manifest exists at exact path: {m_exists}")

if not p_exists or not m_exists:
    sys.exit(1)

print("\n--- 2. PARQUET METADATA ---")
df = pd.read_parquet(parquet_path)
row_count = len(df)
cols = list(df.columns)
lats = df['latitude'].nunique()
lons = df['longitude'].nunique()
doys = df['day_of_year'].nunique()
t_min = df['threshold_95'].min()
t_max = df['threshold_95'].max()
nulls = df['threshold_95'].isnull().sum()
dups = df.duplicated(subset=['latitude', 'longitude', 'day_of_year']).sum()

print(f"Row count: {row_count}")
print(f"Columns: {cols}")
print(f"Latitude unique count: {lats}")
print(f"Longitude unique count: {lons}")
print(f"Day_of_year unique count: {doys}")
print(f"Threshold_95 minimum: {t_min}")
print(f"Threshold_95 maximum: {t_max}")
print(f"Null threshold count: {nulls}")
print(f"Duplicate (lat, lon, doy) count: {dups}")

print("\n--- 3. DATA VERIFICATION ---")
print(f"Expected rows = 5,181,462: {'PASS' if row_count == 5181462 else 'FAIL'}")
print(f"No null thresholds: {'PASS' if nulls == 0 else 'FAIL'}")
print(f"No duplicate spatial/DOY keys: {'PASS' if dups == 0 else 'FAIL'}")

print("\n--- 4. MANIFEST DOCUMENTATION ---")
with open(manifest_path, 'r') as f:
    manifest = yaml.safe_load(f)
print(f"Historical period: {manifest.get('climatology_start')} to {manifest.get('climatology_end')}")
print(f"Variable: {manifest.get('variable')}")
print(f"Spatial domain: lats {manifest.get('latitude_bounds')}, lons {manifest.get('longitude_bounds')}, res {manifest.get('grid_resolution')}")
print(f"Percentile: {manifest.get('percentile')}")
print(f"Smoothing method/window: {manifest.get('window_days')} days, {manifest.get('calendar_handling')} calendar")
print(f"Artifact information: {manifest.get('source_dataset')}, {manifest.get('creation_timestamp')}")

print("\n--- 5. COMPARISON VS COLAB ---")
print(f"Rows match (5,181,462): {'PASS' if row_count == 5181462 else 'FAIL'}")
print(f"Latitudes match (117): {'PASS' if lats == 117 else 'FAIL'}")
print(f"Longitudes match (121): {'PASS' if lons == 121 else 'FAIL'}")
print(f"DOYs match (366): {'PASS' if doys == 366 else 'FAIL'}")
expected_min, expected_max = 249.7819, 320.3997
print(f"Threshold range matches approx (249.78 to 320.40): {'PASS' if abs(t_min - expected_min) < 1.0 and abs(t_max - expected_max) < 1.0 else 'FAIL'}")
print(f"Null thresholds match (0): {'PASS' if nulls == 0 else 'FAIL'}")
print(f"Duplicate keys match (0): {'PASS' if dups == 0 else 'FAIL'}")
