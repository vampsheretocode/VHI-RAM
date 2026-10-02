import xarray as xr
import sys
import psutil
import os
import zarr
import gcsfs

def memory_usage():
    mem = psutil.virtual_memory()
    return f"{mem.percent}% ({mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB)"

print("1. Dataset opened...", flush=True)
url = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

try:
    ds = xr.open_zarr(url, consolidated=True)
    print("Dataset opened successfully.", flush=True)
except Exception as e:
    print(f"Error opening dataset: {e}", flush=True)
    sys.exit(1)

print("2. Coordinates/shape inspected...", flush=True)
print("Dimensions:", ds.dims, flush=True)
print("Coordinates:", list(ds.coords), flush=True)

if '2m_temperature' not in ds:
    print("2m_temperature not found in dataset.", flush=True)
    sys.exit(1)

print("2m_temperature shape:", ds['2m_temperature'].shape, flush=True)
print("Latitudes:", ds.latitude.values.min(), "to", ds.latitude.values.max(), flush=True)
print("Longitudes:", ds.longitude.values.min(), "to", ds.longitude.values.max(), flush=True)
print("Time range:", ds.time.values.min(), "to", ds.time.values.max(), flush=True)
print(f"10. Memory/resource status: {memory_usage()}", flush=True)
print("Probe completed.", flush=True)
