import xarray as xr
import gcsfs
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import os
import yaml
from datetime import datetime

MONTHLY_DIR = 'data/phase1_monthly'
FINAL_OUT = 'data/phase1_dataset.parquet'
MONTHLY_MANIFEST = os.path.join(MONTHLY_DIR, 'manifest_2020.yml')
FINAL_MANIFEST = 'data/phase1_manifest.yml'

def check_reopen_and_validate(path):
    if not os.path.exists(path):
        return False, None
    try:
        df = pd.read_parquet(path)
        # minimal QA to ensure it's not corrupt
        if len(df) == 0:
            return False, None
        return True, df
    except Exception as e:
        print(f"Validation failed for {path}: {e}")
        return False, None

def run_monthly_qa(df, month, src_hres=True, src_pangu=True):
    # Print examples
    print(f"\n--- QA MONTH {month} ---")
    print(f"File size: {df.memory_usage(deep=True).sum()} bytes (in-memory)")
    print(f"Total row count: {len(df)}")
    
    hres_count = len(df[df['source'] == 'HRES'])
    pangu_count = len(df[df['source'] == 'PANGU'])
    print(f"HRES row count: {hres_count}")
    print(f"PANGU row count: {pangu_count}")
    
    print(f"Unique initialization count: {df['init_time'].nunique()}")
    print(f"Unique valid-time count: {df['valid_time'].nunique()}")
    print(f"Unique lead times: {df['lead_time_hours'].unique().tolist()}")
    
    grid_cells = df[['latitude', 'longitude']].drop_duplicates().shape[0]
    print(f"Grid-cell count: {grid_cells}")
    
    missing_f = df['forecast_value'].isna().sum()
    missing_o = df['observation_value'].isna().sum()
    nans = df.isna().sum().sum()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    infs = np.isinf(df[num_cols]).sum().sum()
    
    print(f"Missing forecasts: {missing_f}")
    print(f"Missing observations: {missing_o}")
    print(f"NaNs: {nans}")
    print(f"Infinite values: {infs}")
    
    dups = df.duplicated(subset=['source', 'init_time', 'lead_time_hours', 'latitude', 'longitude']).sum()
    print(f"Duplicate keys: {dups}")
    
    print(f"Invalid coordinates: Lat {df['latitude'].min()} to {df['latitude'].max()}, Lon {df['longitude'].min()} to {df['longitude'].max()}")
    print(f"Invalid lead times (outside expected): {[l for l in df['lead_time_hours'].unique() if l not in [24, 48, 72, 120, 240]]}")
    
    print(f"Temperature - Min: {df['forecast_value'].min():.2f} K")
    print(f"Temperature - Max: {df['forecast_value'].max():.2f} K")
    print(f"Temperature - Mean: {df['forecast_value'].mean():.2f} K")
    print(f"Temperature - Median: {df['forecast_value'].median():.2f} K")
    
    # Temporal validation examples
    print("\nTemporal Validation Examples:")
    if hres_count > 0:
        ex_hres = df[df['source'] == 'HRES'].iloc[0]
        print(f"HRES: init_time = {ex_hres['init_time']}, lead_time = {ex_hres['lead_time_hours']}h, valid_time = {ex_hres['valid_time']}")
    if pangu_count > 0:
        ex_pangu = df[df['source'] == 'PANGU'].iloc[0]
        print(f"PANGU: init_time = {ex_pangu['init_time']}, lead_time = {ex_pangu['lead_time_hours']}h, valid_time = {ex_pangu['valid_time']}")
        
    return {
        "row_count": int(len(df)),
        "hres_count": int(hres_count),
        "pangu_count": int(pangu_count),
        "dups": int(dups),
        "missing_f": int(missing_f),
        "missing_o": int(missing_o),
        "nans": int(nans),
        "infs": int(infs)
    }

def process_month(month, fs, ds_era5, ds_hres, ds_pangu, leads, lon_slice):
    out_file = os.path.join(MONTHLY_DIR, f'phase1_2020_{month:02d}.parquet')
    tmp_file = os.path.join(MONTHLY_DIR, f'phase1_2020_{month:02d}.tmp.parquet')
    
    valid, df_existing = check_reopen_and_validate(out_file)
    if valid:
        print(f"\nMonth {month} already successfully processed and validated. Skipping.")
        qa_stats = run_monthly_qa(df_existing, month)
        qa_stats['file_size'] = os.path.getsize(out_file)
        return out_file, qa_stats
        
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        
    start_date = f"2020-{month:02d}-01T00:00:00"
    if month == 12:
        end_date = f"2020-12-31T12:00:00"
    else:
        next_month = np.datetime64(f"2020-{month+1:02d}-01")
        end_date = str(next_month - np.timedelta64(12, 'h'))
        
    print(f"\nProcessing Month {month}: {start_date} to {end_date}...")
    
    # 1. HRES
    hres_lat = slice(37.0, 8.0) if ds_hres.latitude[0] > ds_hres.latitude[-1] else slice(8.0, 37.0)
    ds_hres_sub = ds_hres.sel(time=slice(start_date, end_date), latitude=hres_lat, longitude=lon_slice)
    hours = ds_hres_sub['time'].dt.hour
    ds_hres_sub = ds_hres_sub.isel(time=((hours == 0) | (hours == 12)))
    
    # 2. PANGU
    pangu_lat = slice(37.0, 8.0) if ds_pangu.latitude[0] > ds_pangu.latitude[-1] else slice(8.0, 37.0)
    ds_pangu_sub = ds_pangu.sel(time=slice(start_date, end_date), latitude=pangu_lat, longitude=lon_slice)
    hours_p = ds_pangu_sub['time'].dt.hour
    ds_pangu_sub = ds_pangu_sub.isel(time=((hours_p == 0) | (hours_p == 12)))
    
    dfs = []
    
    for lead in leads:
        # HRES
        try:
            dh_lead = ds_hres_sub.sel(prediction_timedelta=lead)
        except KeyError:
            dh_lead = ds_hres_sub.sel(prediction_timedelta=np.timedelta64(lead, 'h'))
            
        df_h = dh_lead['2m_temperature'].to_dataframe().reset_index()
        df_h['valid_time'] = df_h['time'] + pd.to_timedelta(lead, unit='h')
        df_h = df_h.rename(columns={'time': 'init_time', '2m_temperature': 'forecast_value'})
        df_h['source'] = 'HRES'
        
        # PANGU
        try:
            dp_lead = ds_pangu_sub.sel(prediction_timedelta=lead)
        except KeyError:
            dp_lead = ds_pangu_sub.sel(prediction_timedelta=np.timedelta64(lead, 'h'))
            
        df_p = dp_lead['2m_temperature'].to_dataframe().reset_index()
        df_p['valid_time'] = df_p['time'] + pd.to_timedelta(lead, unit='h')
        df_p = df_p.rename(columns={'time': 'init_time', '2m_temperature': 'forecast_value'})
        df_p['source'] = 'PANGU'
        
        df_combined = pd.concat([df_h, df_p], ignore_index=True)
        if df_combined.empty:
            continue
            
        # ERA5
        unique_v = df_combined['valid_time'].unique()
        min_v = unique_v.min()
        max_v = unique_v.max()
        
        era5_lat = slice(37.0, 8.0) if ds_era5.latitude[0] > ds_era5.latitude[-1] else slice(8.0, 37.0)
        ds_era5_sub = ds_era5.sel(time=slice(min_v, max_v), latitude=era5_lat, longitude=lon_slice)
        df_o = ds_era5_sub['2m_temperature'].to_dataframe().reset_index()
        df_o = df_o.rename(columns={'time': 'valid_time', '2m_temperature': 'observation_value'})
        
        # Merge
        df_combined['latitude'] = df_combined['latitude'].astype(np.float32).round(2)
        df_combined['longitude'] = df_combined['longitude'].astype(np.float32).round(2)
        df_o['latitude'] = df_o['latitude'].astype(np.float32).round(2)
        df_o['longitude'] = df_o['longitude'].astype(np.float32).round(2)
        
        df_merged = pd.merge(df_combined, df_o, on=['valid_time', 'latitude', 'longitude'], how='inner')
        df_merged['lead_time_hours'] = lead
        df_merged['variable'] = '2m_temperature'
        df_merged['resolution'] = '0.25'
        df_merged['region'] = 'India'
        df_merged['absolute_error'] = np.abs(df_merged['forecast_value'] - df_merged['observation_value'])
        
        cols = ['source', 'init_time', 'prediction_timedelta', 'lead_time_hours', 'valid_time', 'latitude', 'longitude', 
                'forecast_value', 'observation_value', 'variable', 'resolution', 'region', 'absolute_error']
        # prediction_timedelta usually comes from the model coord, we'll ensure it exists
        if 'prediction_timedelta' not in df_merged.columns:
            df_merged['prediction_timedelta'] = lead
            
        df_merged = df_merged[cols]
        dfs.append(df_merged)
        
    df_month = pd.concat(dfs, ignore_index=True)
    
    # Write to temp
    table = pa.Table.from_pandas(df_month)
    writer = pq.ParquetWriter(tmp_file, table.schema)
    writer.write_table(table)
    writer.close()
    
    # Rename
    os.rename(tmp_file, out_file)
    print(f"Month {month} successfully saved to {out_file}")
    
    # Reopen and QA
    v, df_re = check_reopen_and_validate(out_file)
    if not v:
        raise RuntimeError(f"Failed to reopen finalized file {out_file}!")
        
    qa_stats = run_monthly_qa(df_re, month)
    qa_stats['file_size'] = os.path.getsize(out_file)
    
    return out_file, qa_stats

def main():
    os.makedirs(MONTHLY_DIR, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token='anon')
    
    era5_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
    hres_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
    pangu_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
    
    print("Opening Zarr stores lazily...")
    ds_era5 = xr.open_zarr(fs.get_mapper(era5_path), consolidated=True)
    ds_hres = xr.open_zarr(fs.get_mapper(hres_path), consolidated=True)
    ds_pangu = xr.open_zarr(fs.get_mapper(pangu_path), consolidated=True)
    
    lon_slice = slice(68.0, 98.0)
    leads = [24, 48, 72, 120, 240]
    
    manifest_data = {}
    if os.path.exists(MONTHLY_MANIFEST):
        with open(MONTHLY_MANIFEST, 'r') as f:
            manifest_data = yaml.safe_load(f) or {}
            
    all_monthly_files = []
    for m in range(1, 13):
        out_f, qas = process_month(m, fs, ds_era5, ds_hres, ds_pangu, leads, lon_slice)
        all_monthly_files.append(out_f)
        manifest_data[f"month_{m}"] = {
            "file": out_f,
            "qa": qas,
            "status": "PASS"
        }
        with open(MONTHLY_MANIFEST, 'w') as f:
            yaml.dump(manifest_data, f)
            
    print("\n================ FINAL COMBINATION ================")
    if os.path.exists(FINAL_OUT):
        os.remove(FINAL_OUT)
        
    dfs = []
    for f in all_monthly_files:
        dfs.append(pd.read_parquet(f))
        
    df_final = pd.concat(dfs, ignore_index=True)
    table_final = pa.Table.from_pandas(df_final)
    writer_final = pq.ParquetWriter(FINAL_OUT, table_final.schema)
    writer_final.write_table(table_final)
    writer_final.close()
    
    print("Final Parquet writer closed.")
    
    print("\n--- FINAL REOPEN TEST ---")
    df_reopen = pd.read_parquet(FINAL_OUT)
    print(f"Final Total rows: {len(df_reopen)}")
    print(f"Final Sources: {df_reopen['source'].unique().tolist()}")
    print(f"Final Date coverage: {df_reopen['init_time'].min()} to {df_reopen['init_time'].max()}")
    print(f"Final Leads: {df_reopen['lead_time_hours'].unique().tolist()}")
    
    final_manifest = {
        "dataset_sources": ["HRES", "PANGU", "ERA5"],
        "variable": "2m_temperature",
        "domain": "8-37N, 68-98E",
        "resolution": "0.25 deg",
        "initialization_period": "2020-01-01 to 2020-12-31",
        "leads": leads,
        "row_count": len(df_reopen),
        "file_size": os.path.getsize(FINAL_OUT),
        "monthly_artifacts": all_monthly_files,
        "status": "PASS"
    }
    with open(FINAL_MANIFEST, 'w') as f:
        yaml.dump(final_manifest, f)
        
    print(f"Final Manifest written to {FINAL_MANIFEST}")
    
if __name__ == '__main__':
    main()
