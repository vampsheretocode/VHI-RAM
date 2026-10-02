import xarray as xr
from datetime import datetime
import sys

print("--- LOCAL METADATA / GCS ACCESS TEST ---")
url = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

print(f"[{datetime.now().time()}] GCS ACCESS MODE: ANONYMOUS PUBLIC")
storage_options = {'token': 'anon'}
try:
    ds = xr.open_zarr(url, consolidated=True, storage_options=storage_options)
    print(f"[{datetime.now().time()}] Dataset metadata accessed successfully.")
    
    # Just read metadata, do not slice or download
    print(f"Dataset variables: {list(ds.data_vars)}")
    temp = ds['2m_temperature']
    print(f"Variable shape: {temp.shape}")
    print("SUCCESS: Anonymous access verified locally.")
    sys.exit(0)
except Exception as e:
    print(f"ERROR accessing GCS anonymously: {e}")
    sys.exit(1)
