import xarray as xr
import numpy as np
import pandas as pd
import psutil
import os
import sys
import gc

def memory_usage():
    mem = psutil.virtual_memory()
    return mem.percent, f"{mem.percent}% ({mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB)"

print("1. Dataset opened...", flush=True)
url = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

try:
    ds = xr.open_zarr(url, consolidated=True)
    print("Dataset opened successfully.", flush=True)
except Exception as e:
    print(f"Error opening dataset: {e}", flush=True)
    sys.exit(1)

print("2. Variable selected: 2m_temperature", flush=True)
temp = ds['2m_temperature']

print("3. Climatology period: 1991-2019", flush=True)
ds_hist = temp.sel(time=slice('1991-01-01', '2019-12-31'))

print("4. Spatial bounds: lat 37 to 8 N, lon 68 to 98 E", flush=True)
# Ensure slicing respects descending latitudes
ds_india = ds_hist.sel(latitude=slice(37, 8), longitude=slice(68, 98))

lats = ds_india.latitude.values
lons = ds_india.longitude.values
times = pd.DatetimeIndex(ds_india.time.values)

print(f"5. Grid dimensions: {len(lats)} lats x {len(lons)} lons x {len(times)} times", flush=True)

# Tiling strategy
lat_chunk = 20
lon_chunk = 20
num_lat_tiles = int(np.ceil(len(lats) / lat_chunk))
num_lon_tiles = int(np.ceil(len(lons) / lon_chunk))
total_tiles = num_lat_tiles * num_lon_tiles

print(f"6. Tile size: {lat_chunk} x {lon_chunk} grid cells", flush=True)
print(f"7. Number of tiles: {total_tiles}", flush=True)

# Estimate memory per tile: 20 * 20 * 42368 * 4 bytes = ~67 MB.
print(f"Estimated memory per tile: ~67 MB", flush=True)

# Prepare circular DOY indexing
doy_array = times.dayofyear
def get_circular_mask(target_doy, window_days=15):
    # Circular distance on a 366-day calendar
    # A day is in window if its shortest distance to target_doy is <= 15
    dist1 = np.abs(doy_array - target_doy)
    dist2 = np.abs(doy_array - target_doy + 366)
    dist3 = np.abs(doy_array - target_doy - 366)
    min_dist = np.minimum.reduce([dist1, dist2, dist3])
    return min_dist <= window_days

print("Precomputing day-of-year masks...", flush=True)
doy_masks = {d: get_circular_mask(d) for d in range(1, 367)}

def process_tile(lat_start, lat_end, lon_start, lon_end, tile_idx):
    mem_pct, mem_str = memory_usage()
    print(f"8. RAM before tile {tile_idx}: {mem_str}", flush=True)
    if mem_pct > 80:
        print("RAM exceeds 80%. Running garbage collection...", flush=True)
        gc.collect()
        mem_pct, mem_str = memory_usage()
        if mem_pct > 85:
            print("RAM exceeds 85%. SAFELY ABORTING.", flush=True)
            sys.exit(1)

    print(f"9. Processing tile {tile_idx} (lat: {lat_start}-{lat_end}, lon: {lon_start}-{lon_end})", flush=True)
    
    # Extract data lazily
    tile_data = ds_india.isel(latitude=slice(lat_start, lat_end), longitude=slice(lon_start, lon_end))
    
    # Materialize the tile into RAM
    # Since we make HTTP requests here, we'll time it.
    print(f"   Downloading tile data into RAM...", flush=True)
    tile_values = tile_data.values # This triggers computation
    
    tile_lats = tile_data.latitude.values
    tile_lons = tile_data.longitude.values
    
    results = []
    
    for i, lat_val in enumerate(tile_lats):
        for j, lon_val in enumerate(tile_lons):
            ts = tile_values[:, i, j]
            for target_doy in range(1, 367):
                mask = doy_masks[target_doy]
                window_data = ts[mask]
                
                # Filter NaNs if any
                window_data = window_data[~np.isnan(window_data)]
                sample_count = len(window_data)
                
                if sample_count > 0:
                    q95 = np.percentile(window_data, 95)
                else:
                    q95 = np.nan
                    
                results.append({
                    "latitude": lat_val,
                    "longitude": lon_val,
                    "day_of_year": target_doy,
                    "threshold_95": float(q95),
                    "sample_count": int(sample_count)
                })
                
    df = pd.DataFrame(results)
    
    os.makedirs("data/extreme_climatology", exist_ok=True)
    temp_path = f"data/extreme_climatology/climatology_tile_{tile_idx:03d}_temp.parquet"
    final_path = f"data/extreme_climatology/climatology_tile_{tile_idx:03d}.parquet"
    
    df.to_parquet(temp_path, index=False)
    
    # Reopen to verify
    try:
        df_read = pd.read_parquet(temp_path)
        if len(df_read) == len(df):
            os.rename(temp_path, final_path)
            print(f"10. Tile {tile_idx} complete and validated.", flush=True)
        else:
            print(f"Error validating tile {tile_idx}", flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"Error reopening tile {tile_idx}: {e}", flush=True)
        sys.exit(1)
        
    mem_pct, mem_str = memory_usage()
    print(f"11. RAM after tile {tile_idx}: {mem_str}", flush=True)

# PILOT RUN ONLY
print("\n--- STARTING PILOT RUN ---", flush=True)
process_tile(0, lat_chunk, 0, lon_chunk, 1)

print("\n13. Final validation status: Pilot tile completed successfully.", flush=True)
sys.exit(0)
