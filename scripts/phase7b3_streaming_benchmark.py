import xarray as xr
import numpy as np
import psutil
import time
import os
import sys

def memory_usage():
    mem = psutil.virtual_memory()
    return mem.percent, f"{mem.percent}% ({mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB)"

print("--- STARTING STREAMING BENCHMARK ---", flush=True)
mem_pct, mem_str = memory_usage()
print(f"RAM before opening data: {mem_str}", flush=True)

if mem_pct > 90:
    print("Baseline RAM is >90%. Aborting safely.", flush=True)
    sys.exit(1)

url = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
print(f"Opening dataset metadata from: {url}", flush=True)

ds = xr.open_zarr(url, consolidated=True)
mem_pct, mem_str = memory_usage()
print(f"RAM after opening metadata: {mem_str}", flush=True)

# Select variable and historical period
temp = ds['2m_temperature']
temp_hist = temp.sel(time=slice('1991-01-01', '1991-01-31'))
time_steps = len(temp_hist.time)
print(f"Benchmark period: 1991-01-01 to 1991-01-31 ({time_steps} time steps)", flush=True)

# Slice India
# ERA5 lat is 90 to -90, so we use slice(37, 8)
lat_slice = slice(37, 8)
lon_slice = slice(68, 98)
temp_india = temp_hist.sel(latitude=lat_slice, longitude=lon_slice)

india_shape = temp_india.shape
print(f"India subset shape: {india_shape} (expected {time_steps}, 117, 121)", flush=True)

# Streaming extraction
print("\nStarting streaming read...", flush=True)
start_time = time.time()

# Accumulator list
subset_chunks = []
peak_mem = mem_pct

for i in range(time_steps):
    step_start = time.time()
    
    # Extract the single time step. Note: using .values forces download of that slice.
    # Because zarr chunking is (1, 721, 1440), slicing lat/lon still downloads the whole global chunk
    # before slicing locally in xarray.
    chunk_data = temp_india.isel(time=i).values
    subset_chunks.append(chunk_data)
    
    step_time = time.time() - step_start
    
    current_mem, current_mem_str = memory_usage()
    if current_mem > peak_mem:
        peak_mem = current_mem
        
    if (i + 1) % 10 == 0 or (i + 1) == time_steps:
        print(f"Processed chunk {i+1}/{time_steps}. Last step time: {step_time:.3f}s. RAM: {current_mem_str}", flush=True)
        
    if current_mem > 85:
        print("RAM reached 85%. HARD ABORTING benchmark.", flush=True)
        sys.exit(1)

elapsed = time.time() - start_time
print(f"\nStreaming complete in {elapsed:.2f} seconds.", flush=True)

# Stack to form the final array
print("\nAccumulating and verifying shape...", flush=True)
stacked_india = np.stack(subset_chunks, axis=0)
print(f"Accumulated shape: {stacked_india.shape}", flush=True)

mem_pct, mem_str = memory_usage()
print(f"RAM after releasing temporary arrays: {mem_str}", flush=True)

# Calculate metrics
global_chunk_size = 1 * 721 * 1440 * 4
total_bytes_read = time_steps * global_chunk_size
india_chunk_size = 1 * 117 * 121 * 4
total_india_bytes = time_steps * india_chunk_size

print("\n--- BENCHMARK RESULTS ---", flush=True)
print(f"Chunks accessed: {time_steps}")
print(f"Total global bytes downloaded: {total_bytes_read / 1024**2:.2f} MB")
print(f"Total elapsed time: {elapsed:.2f} seconds")
print(f"Average time per chunk: {elapsed / time_steps:.3f} seconds")
print(f"India subset size materialized: {total_india_bytes / 1024**2:.2f} MB")
print(f"Peak RAM during streaming: {peak_mem}%")

# Extrapolate for 1991-2019 (29 years)
total_steps_29y = 42368
extrap_time = (elapsed / time_steps) * total_steps_29y
extrap_dl = total_steps_29y * global_chunk_size
extrap_disk = total_steps_29y * india_chunk_size

print("\n--- EXTRAPOLATION FOR 1991-2019 (42,368 steps) ---", flush=True)
print(f"Estimated Runtime: {extrap_time / 3600:.2f} hours")
print(f"Estimated Network Transfer: {extrap_dl / 1024**3:.2f} GB")
print(f"Estimated Local Disk Requirement (India subset only): {extrap_disk / 1024**3:.2f} GB")

sys.exit(0)
