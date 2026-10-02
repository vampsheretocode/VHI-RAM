import argparse
import yaml
import xarray as xr
import pandas as pd
import numpy as np
import time
import os
import sys
import psutil
from datetime import datetime
import gc

def get_ram_usage():
    mem = psutil.virtual_memory()
    return f"{mem.percent}% ({mem.used / 1024**3:.2f} GB)"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_computation(config, is_benchmark):
    print(f"[{datetime.now().time()}] Starting Cloud Climatology Computation (Benchmark: {is_benchmark})", flush=True)
    print(f"[{datetime.now().time()}] Initial RAM: {get_ram_usage()}", flush=True)
    
    url = config['data_source']['zarr_url']
    var_name = config['data_source']['variable']
    start_date = config['historical_period']['start_date']
    end_date = config['historical_period']['end_date']
    lat_max = config['spatial_domain']['latitude_max']
    lat_min = config['spatial_domain']['latitude_min']
    lon_min = config['spatial_domain']['longitude_min']
    lon_max = config['spatial_domain']['longitude_max']
    
    if is_benchmark:
        # Benchmark mode: just 1 month
        end_date = '1991-01-31'
        
    print(f"[{datetime.now().time()}] Period: {start_date} to {end_date}", flush=True)
    
    # Open dataset
    print(f"[{datetime.now().time()}] GCS ACCESS MODE: ANONYMOUS PUBLIC", flush=True)
    storage_options = {'token': 'anon'}
    try:
        ds = xr.open_zarr(url, consolidated=True, storage_options=storage_options)
        print(f"[{datetime.now().time()}] Dataset metadata accessed successfully.", flush=True)
    except Exception as e:
        print(f"ERROR accessing GCS anonymously: {e}", flush=True)
        sys.exit(1)
    temp = ds[var_name].sel(time=slice(start_date, end_date))
    # Note: ERA5 lat is 90 to -90, so slice from max to min
    temp_india = temp.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    total_time_steps = len(temp_india.time)
    lats = temp_india.latitude.values
    lons = temp_india.longitude.values
    
    print(f"[{datetime.now().time()}] Target Shape: {total_time_steps} times, {len(lats)} lats, {len(lons)} lons", flush=True)
    
    # Pre-allocate or accumulate locally
    # To avoid RAM spikes, we will download data year by year and store it locally
    years = np.unique(temp_india.time.dt.year)
    all_data = []
    times = []
    
    start_time = time.time()
    chunks_accessed = 0
    bytes_transferred_estimate = 0
    chunk_size_bytes = 1 * 721 * 1440 * 4 # 4.15 MB per time step globally
    
    for y in years:
        print(f"[{datetime.now().time()}] Downloading year {y}...", flush=True)
        year_data_lazy = temp_india.sel(time=str(y))
        
        # Trigger download
        year_values = year_data_lazy.values
        year_times = year_data_lazy.time.values
        
        all_data.append(year_values)
        times.append(year_times)
        
        num_steps = len(year_times)
        chunks_accessed += num_steps
        bytes_transferred_estimate += num_steps * chunk_size_bytes
        
        print(f"[{datetime.now().time()}] Year {y} complete. RAM: {get_ram_usage()}", flush=True)
        gc.collect()
        
    elapsed_dl = time.time() - start_time
    
    print(f"[{datetime.now().time()}] Download complete. Concatenating...", flush=True)
    india_cube = np.concatenate(all_data, axis=0)
    time_index = pd.DatetimeIndex(np.concatenate(times, axis=0))
    del all_data, times
    gc.collect()
    
    india_bytes = india_cube.nbytes
    print(f"[{datetime.now().time()}] India subset size: {india_bytes / 1024**2:.2f} MB", flush=True)
    print(f"[{datetime.now().time()}] Download Elapsed Time: {elapsed_dl:.2f} sec", flush=True)
    
    if is_benchmark:
        extrap_factor = 42368 / chunks_accessed
        print(f"\n--- BENCHMARK RESULTS ---")
        print(f"Elapsed Time: {elapsed_dl:.2f} sec")
        print(f"Peak RAM: {get_ram_usage()}")
        print(f"Chunks Accessed: {chunks_accessed}")
        print(f"Bytes Transferred: {bytes_transferred_estimate / 1024**3:.2f} GB")
        print(f"Processing Throughput: {chunks_accessed / elapsed_dl:.2f} chunks/sec")
        print(f"Estimated Full-Run Runtime (Download): {(elapsed_dl * extrap_factor) / 3600:.2f} hours")
        print(f"Estimated Full-Run Data Transfer: {(bytes_transferred_estimate * extrap_factor) / 1024**3:.2f} GB")
        print("-------------------------\n")
        
        import json
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/phase7b3_cloud_benchmark.json', 'w') as f:
            json.dump({
                "elapsed_sec": elapsed_dl,
                "chunks_accessed": chunks_accessed,
                "throughput_chunks_per_sec": chunks_accessed / elapsed_dl,
                "estimated_full_runtime_hr": (elapsed_dl * extrap_factor) / 3600,
                "estimated_full_transfer_gb": (bytes_transferred_estimate * extrap_factor) / 1024**3
            }, f, indent=2)
        print("Benchmark JSON saved. Exiting.")
        return

    print(f"[{datetime.now().time()}] Computing Climatology Percentiles...", flush=True)
    # Day of year arrays
    doy_array = time_index.dayofyear
    window_days = config['climatology']['smoothing_window_days']
    percentile = config['climatology']['percentile'] * 100
    
    # Handle DOY 1-366 circular distance
    def get_circular_mask(target_doy):
        d1 = np.abs(doy_array - target_doy)
        d2 = np.abs(doy_array - target_doy + 366)
        d3 = np.abs(doy_array - target_doy - 366)
        return np.minimum.reduce([d1, d2, d3]) <= window_days

    results = []
    
    # Compute per DOY
    for target_doy in range(1, 367):
        mask = get_circular_mask(target_doy)
        window_data = india_cube[mask, :, :]
        
        # calculate percentile across time (axis 0)
        q95 = np.nanpercentile(window_data, percentile, axis=0)
        
        sample_count = np.sum(~np.isnan(window_data), axis=0)
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                results.append({
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "day_of_year": int(target_doy),
                    "threshold_95": float(q95[i, j]),
                    "sample_count": int(sample_count[i, j])
                })
    
    df_out = pd.DataFrame(results)
    
    # Save
    out_path = config['output']['artifact_path']
    manifest_path = config['output']['manifest_path']
    temp_path = out_path.replace(".parquet", "_temp.parquet")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[{datetime.now().time()}] Writing output to {temp_path}", flush=True)
    df_out.to_parquet(temp_path, index=False)
    
    # Validation
    df_verify = pd.read_parquet(temp_path)
    if len(df_verify) == len(df_out):
        os.rename(temp_path, out_path)
        print(f"[{datetime.now().time()}] Artifact successfully generated and renamed to {out_path}", flush=True)
    else:
        print("CLIMATOLOGY COMPUTATION NOT COMPLETED - Reopen validation failed.", flush=True)
        sys.exit(1)
        
    # Manifest
    manifest = {
        "source_dataset": url,
        "variable": var_name,
        "climatology_start": start_date,
        "climatology_end": end_date,
        "latitude_bounds": [lat_max, lat_min],
        "longitude_bounds": [lon_min, lon_max],
        "grid_resolution": 0.25,
        "percentile": config['climatology']['percentile'],
        "window_days": window_days,
        "calendar_handling": config['climatology']['calendar_type'],
        "creation_timestamp": datetime.now().isoformat(),
        "final_row_count": len(df_verify),
        "validation_status": "PASS"
    }
    
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f)
    print(f"[{datetime.now().time()}] Manifest written to {manifest_path}", flush=True)
    print("CLIMATOLOGY COMPUTATION COMPLETED SUCCESSFULLY", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase7b3_climatology.yml")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    
    config = load_config(args.config)
    try:
        run_computation(config, args.benchmark)
    except Exception as e:
        print(f"CLIMATOLOGY COMPUTATION NOT COMPLETED. Error: {e}", flush=True)
        sys.exit(1)
