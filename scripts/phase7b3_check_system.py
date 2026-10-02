import psutil
import zarr
import xarray as xr

mem = psutil.virtual_memory()
print(f"Current RAM Usage: {mem.percent}% ({mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB)")

if mem.percent > 80:
    print("WARNING: RAM is > 80%")
    # Identify top memory consumers
    processes = sorted(psutil.process_iter(['pid', 'name', 'memory_info']), key=lambda p: p.info['memory_info'].rss, reverse=True)[:5]
    print("Top 5 memory consuming processes:")
    for p in processes:
        print(f"PID: {p.info['pid']}, Name: {p.info['name']}, RAM: {p.info['memory_info'].rss / 1024**2:.2f} MB")

url = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
print(f"\nChecking Zarr chunk geometry for {url}")
try:
    ds = xr.open_zarr(url, consolidated=True)
    temp = ds['2m_temperature']
    print(f"Variable shape: {temp.shape}")
    print(f"Dimensions: {temp.dims}")
    if hasattr(temp.data, 'chunksize'):
        print(f"Dask Chunks: {temp.data.chunksize}")
    if 'chunks' in temp.encoding:
        print(f"Zarr Encoding Chunks: {temp.encoding['chunks']}")
except Exception as e:
    print(f"Error accessing zarr: {e}")
