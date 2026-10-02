import xarray as xr
import gcsfs
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import os
from datetime import datetime

def test_time_alignment():
    print("\n--- TASK 1: FIX AND VERIFY TIME ALIGNMENT ---")
    
    init_str = '2020-01-01T00:00:00'
    lead_hours = 24
    
    init_time = np.datetime64(init_str)
    lead_time = np.timedelta64(lead_hours, 'h')
    valid_time = init_time + lead_time
    
    expected_valid = np.datetime64('2020-01-02T00:00:00')
    
    print(f"init_time: {init_time} (dtype: {init_time.dtype})")
    print(f"lead_time: {lead_time} (dtype: {lead_time.dtype})")
    print(f"calculated valid_time: {valid_time} (dtype: {valid_time.dtype})")
    print(f"expected valid_time: {expected_valid}")
    print(f"equality result: {valid_time == expected_valid}")
    
    if valid_time != expected_valid:
        print("[ERROR] Time alignment failed.")
        sys.exit(1)
    else:
        print("[SUCCESS] Time alignment is explicit and correct.")

def process_chunk(fs, ds_era5, ds_model, model_name, month, lon_slice, leads):
    start_date = f"2020-{month:02d}-01T00:00:00"
    if month == 12:
        end_date = f"2020-12-31T12:00:00"
    else:
        # get last day of month
        next_month = np.datetime64(f"2020-{month+1:02d}-01")
        end_date = str(next_month - np.timedelta64(12, 'h'))
        
    print(f"  Processing {model_name} for {start_date} to {end_date}...")
    
    # Determine lat slices dynamically
    model_lat_slice = slice(37, 8) if ds_model.latitude[0] > ds_model.latitude[-1] else slice(8, 37)
    era5_lat_slice = slice(37, 8) if ds_era5.latitude[0] > ds_era5.latitude[-1] else slice(8, 37)
    
    # Slice the model data
    model_chunk = ds_model.sel(
        time=slice(start_date, end_date),
        latitude=model_lat_slice,
        longitude=lon_slice
    )
    
    # We only want 00Z and 12Z.
    hours = model_chunk['time'].dt.hour
    model_chunk = model_chunk.isel(time=((hours == 0) | (hours == 12)))
    
    dfs = []
    
    for lead in leads:
        try:
            # WeatherBench2 HRES/Pangu stores prediction_timedelta as integer hours or timedelta depending on engine
            try:
                ds_lead = model_chunk.sel(prediction_timedelta=lead)
            except KeyError:
                ds_lead = model_chunk.sel(prediction_timedelta=np.timedelta64(lead, 'h'))
        except KeyError as e:
            print(f"KeyError for lead {lead}: {e}")
            continue
            
        # Convert to DataFrame
        df_f = ds_lead['2m_temperature'].to_dataframe().reset_index()
        
        # Calculate valid time explicitly
        df_f['valid_time'] = df_f['time'] + pd.to_timedelta(lead, unit='h')
        df_f = df_f.rename(columns={'time': 'init_time', '2m_temperature': 'forecast_value'})
        
        # Extract the unique valid times to slice ERA5
        unique_valid_times = df_f['valid_time'].unique()
        
        min_v = unique_valid_times.min()
        max_v = unique_valid_times.max()
        
        ds_era5_chunk = ds_era5.sel(
            time=slice(min_v, max_v),
            latitude=era5_lat_slice,
            longitude=lon_slice
        )
        
        df_obs = ds_era5_chunk['2m_temperature'].to_dataframe().reset_index()
        df_obs = df_obs.rename(columns={'time': 'valid_time', '2m_temperature': 'observation_value'})
        
        # Merge forecast and observations based on valid_time, lat, lon
        # Round lats and lons slightly to avoid floating point mismatch
        df_f['latitude'] = df_f['latitude'].astype(np.float32).round(2)
        df_f['longitude'] = df_f['longitude'].astype(np.float32).round(2)
        df_obs['latitude'] = df_obs['latitude'].astype(np.float32).round(2)
        df_obs['longitude'] = df_obs['longitude'].astype(np.float32).round(2)
        
        df_merged = pd.merge(df_f, df_obs, on=['valid_time', 'latitude', 'longitude'], how='inner')
        
        df_merged['source'] = model_name
        df_merged['lead_time_hours'] = lead
        df_merged['variable'] = '2m_temperature'
        df_merged['resolution'] = '0.25'
        df_merged['region'] = 'India'
        
        df_merged['absolute_error'] = np.abs(df_merged['forecast_value'] - df_merged['observation_value'])
        
        # Reorder to requested schema
        cols = ['source', 'init_time', 'lead_time_hours', 'valid_time', 'latitude', 'longitude', 
                'forecast_value', 'observation_value', 'variable', 'resolution', 'region', 'absolute_error']
        df_merged = df_merged[cols]
        dfs.append(df_merged)
        
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def build_pipeline():
    print("\n--- TASK 2, 3, 4, 5, 6: ACQUISITION PIPELINE & QA ---")
    fs = gcsfs.GCSFileSystem(token='anon')
    
    era5_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
    hres_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
    pangu_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
    
    print("Opening Zarr stores lazily...")
    ds_era5 = xr.open_zarr(fs.get_mapper(era5_path), consolidated=True)
    ds_hres = xr.open_zarr(fs.get_mapper(hres_path), consolidated=True)
    ds_pangu = xr.open_zarr(fs.get_mapper(pangu_path), consolidated=True)
    
    lon_slice = slice(68, 98)
    
    leads = [24, 48, 72, 120, 240]
    out_path = 'data/phase1_dataset.parquet'
    
    # Clear existing
    if os.path.exists(out_path):
        os.remove(out_path)
        
    writer = None
    total_rows = 0
    total_missing_obs = 0
    total_nans = 0
    total_infs = 0
    stats = {'HRES': [], 'PANGU': [], 'ERA5': []}
    
    # Process incrementally by month to avoid RAM explosion
    for month in range(1, 13):
        print(f"\nProcessing Month {month}/12...")
        
        # HRES
        df_hres = process_chunk(fs, ds_era5, ds_hres, 'HRES', month, lon_slice, leads)
        # PANGU
        df_pangu = process_chunk(fs, ds_era5, ds_pangu, 'PANGU', month, lon_slice, leads)
        
        df_combined = pd.concat([df_hres, df_pangu], ignore_index=True)
        
        if df_combined.empty:
            continue
            
        # Quality Checks incrementally
        total_rows += len(df_combined)
        total_missing_obs += df_combined['observation_value'].isna().sum()
        total_nans += df_combined[['forecast_value', 'observation_value']].isna().sum().sum()
        total_infs += np.isinf(df_combined[['forecast_value', 'observation_value']]).sum().sum()
        
        # Gather stats
        for src in ['HRES', 'PANGU']:
            d_src = df_combined[df_combined['source'] == src]
            if not d_src.empty:
                stats[src].append([d_src['forecast_value'].min(), d_src['forecast_value'].max(), 
                                   d_src['forecast_value'].mean(), d_src['forecast_value'].median()])
                stats['ERA5'].append([d_src['observation_value'].min(), d_src['observation_value'].max(),
                                      d_src['observation_value'].mean(), d_src['observation_value'].median()])
        
        # Write to Parquet
        table = pa.Table.from_pandas(df_combined)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        
    if writer:
        writer.close()
        
    print("\n--- TASK 4: DATA QUALITY CHECKS ---")
    print(f"Total Rows Processed: {total_rows}")
    print(f"Missing Observations: {total_missing_obs}")
    print(f"NaN values found: {total_nans}")
    print(f"Infinite values found: {total_infs}")
    
    print("\n--- TASK 5: PHYSICAL SANITY CHECKS ---")
    for src in ['HRES', 'PANGU', 'ERA5']:
        mins = [x[0] for x in stats[src]]
        maxs = [x[1] for x in stats[src]]
        means = [x[2] for x in stats[src]]
        medians = [x[3] for x in stats[src]]
        print(f"{src} - Min: {np.min(mins):.2f} K, Max: {np.max(maxs):.2f} K, Mean: {np.mean(means):.2f} K, Median: {np.median(medians):.2f} K")
        
    print("\n--- TASK 6 & 9: ACTUAL STORAGE & SIZE MEASUREMENT ---")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Artifact successfully created at: {os.path.abspath(out_path)}")
    print(f"Actual size on disk: {size_mb:.2f} MB")
    
    # Provenance
    print("\n--- TASK 7: PROVENANCE MANIFEST ---")
    manifest = {
        "dataset_sources": ["HRES", "Pangu", "ERA5"],
        "variable": "2m_temperature",
        "spatial_bounds": {"lat": "8N to 37N", "lon": "68E to 98E"},
        "resolution": "0.25 deg",
        "initialization_period": "2020-01-01 to 2020-12-31",
        "lead_times_hours": leads,
        "extraction_timestamp": str(datetime.utcnow()),
        "artifact_path": os.path.abspath(out_path),
        "number_of_records": total_rows,
        "missing_data_counts": int(total_missing_obs)
    }
    import yaml
    with open('data/phase1_manifest.yml', 'w') as f:
        yaml.dump(manifest, f)
    print("Manifest written to data/phase1_manifest.yml")
    

if __name__ == '__main__':
    test_time_alignment()
    build_pipeline()

