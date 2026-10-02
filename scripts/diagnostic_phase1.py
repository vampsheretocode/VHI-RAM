import xarray as xr
import gcsfs
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import os

def run_diagnostic():
    print("DIAGNOSTIC STARTED")
    
    fs = gcsfs.GCSFileSystem(token='anon')
    
    era5_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
    hres_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
    pangu_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
    
    print("Opening ERA5 lazily...")
    ds_era5 = xr.open_zarr(fs.get_mapper(era5_path), consolidated=True)
    print("Opening HRES lazily...")
    ds_hres = xr.open_zarr(fs.get_mapper(hres_path), consolidated=True)
    print("Opening PANGU lazily...")
    ds_pangu = xr.open_zarr(fs.get_mapper(pangu_path), consolidated=True)
    
    print("\nSuccessful-open messages complete.")
    
    # Tiny India subset: e.g. Lat 20.0 to 19.0, Lon 75.0 to 76.0 (approx 4x4 grid = 16 cells)
    lon_slice = slice(75.0, 76.0)
    
    init_str = '2020-01-01T00:00:00'
    lead_h = 24
    
    print(f"\nProcessing init_time: {init_str}, lead_time: {lead_h} hours")
    
    # 1. HRES 
    hres_lat_slice = slice(20.0, 19.0) if ds_hres.latitude[0] > ds_hres.latitude[-1] else slice(19.0, 20.0)
    ds_hres_sub = ds_hres.sel(time=init_str, latitude=hres_lat_slice, longitude=lon_slice)
    try:
        ds_hres_lead = ds_hres_sub.sel(prediction_timedelta=lead_h)
    except KeyError:
        ds_hres_lead = ds_hres_sub.sel(prediction_timedelta=np.timedelta64(lead_h, 'h'))
        
    df_hres = ds_hres_lead['2m_temperature'].to_dataframe().reset_index()
    df_hres['valid_time'] = df_hres['time'] + pd.to_timedelta(lead_h, unit='h')
    df_hres = df_hres.rename(columns={'time': 'init_time', '2m_temperature': 'forecast_value'})
    df_hres['source'] = 'HRES'
    
    # 2. PANGU
    pangu_lat_slice = slice(20.0, 19.0) if ds_pangu.latitude[0] > ds_pangu.latitude[-1] else slice(19.0, 20.0)
    ds_pangu_sub = ds_pangu.sel(time=init_str, latitude=pangu_lat_slice, longitude=lon_slice)
    try:
        ds_pangu_lead = ds_pangu_sub.sel(prediction_timedelta=lead_h)
    except KeyError:
        ds_pangu_lead = ds_pangu_sub.sel(prediction_timedelta=np.timedelta64(lead_h, 'h'))
        
    df_pangu = ds_pangu_lead['2m_temperature'].to_dataframe().reset_index()
    df_pangu['valid_time'] = df_pangu['time'] + pd.to_timedelta(lead_h, unit='h')
    df_pangu = df_pangu.rename(columns={'time': 'init_time', '2m_temperature': 'forecast_value'})
    df_pangu['source'] = 'PANGU'
    
    print(f"HRES rows: {len(df_hres)}")
    print(f"PANGU rows: {len(df_pangu)}")
    
    # Explicit time print
    v_time = df_hres['valid_time'].iloc[0]
    print(f"\nTime Alignment:")
    print(f"init_time: {init_str}")
    print(f"lead_time: {lead_h} hours")
    print(f"calculated valid_time: {v_time}")
    
    df_f = pd.concat([df_hres, df_pangu], ignore_index=True)
    
    # 3. ERA5
    # Since we are predicting for valid_time
    era5_lat_slice = slice(20.0, 19.0) if ds_era5.latitude[0] > ds_era5.latitude[-1] else slice(19.0, 20.0)
    ds_era5_sub = ds_era5.sel(time=v_time, latitude=era5_lat_slice, longitude=lon_slice)
    df_era5 = ds_era5_sub['2m_temperature'].to_dataframe().reset_index()
    df_era5 = df_era5.rename(columns={'time': 'valid_time', '2m_temperature': 'observation_value'})
    
    # Merge
    df_f['latitude'] = df_f['latitude'].astype(np.float32).round(2)
    df_f['longitude'] = df_f['longitude'].astype(np.float32).round(2)
    df_era5['latitude'] = df_era5['latitude'].astype(np.float32).round(2)
    df_era5['longitude'] = df_era5['longitude'].astype(np.float32).round(2)
    
    df_merged = pd.merge(df_f, df_era5, on=['valid_time', 'latitude', 'longitude'], how='inner')
    df_merged['lead_time_hours'] = lead_h
    df_merged['absolute_error'] = np.abs(df_merged['forecast_value'] - df_merged['observation_value'])
    
    print("\n--- TINY SUBSET RESULTS ---")
    for src in ['HRES', 'PANGU']:
        subset = df_merged[df_merged['source'] == src].head(2)
        print(f"\nSource: {src}")
        for _, row in subset.iterrows():
            print(f"Lat: {row['latitude']}, Lon: {row['longitude']} | Forecast: {row['forecast_value']:.2f} K | Obs (ERA5): {row['observation_value']:.2f} K | Error: {row['absolute_error']:.2f} K")
            
    # Write Parquet
    out_path = 'data/phase1_diagnostic.parquet'
    if os.path.exists(out_path):
        os.remove(out_path)
        
    table = pa.Table.from_pandas(df_merged)
    writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    writer.close()
    
    print(f"\nExplicitly closed Parquet writer.")
    size = os.path.getsize(out_path)
    print(f"File {out_path} created. Size: {size} bytes")

if __name__ == '__main__':
    run_diagnostic()
    print("SUCCESSFUL EXIT")
    sys.exit(0)
