import xarray as xr
import gcsfs
import sys

def main():
    try:
        print("Checking access to gs://weatherbench2...")
        fs = gcsfs.GCSFileSystem(token='anon')
        
        # Verify datasets exist
        dirs = fs.ls('gs://weatherbench2/datasets')
        print(f"Found {len(dirs)} items in gs://weatherbench2/datasets")
        
        # Inspect ERA5 at 1 degree (climatology/truth)
        era5_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
        
        # Try to open just the metadata without downloading data
        ds = xr.open_zarr(fs.get_mapper(era5_path), consolidated=True)
        print("\n--- ERA5 1959-2022 6h 64x32 (Small sample for quick test) ---")
        print("Dimensions:", ds.dims)
        print("Variables (subset):", list(ds.data_vars.keys())[:5])
        
        # Determine temporal coverage (metadata)
        print("Time range:", ds.time.values[0], "to", ds.time.values[-1])
        print("Latitude range:", ds.latitude.values.min(), "to", ds.latitude.values.max())
        print("Longitude range:", ds.longitude.values.min(), "to", ds.longitude.values.max())
        
        print("\n[VERIFIED] WeatherBench 2 access is successful.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to access WeatherBench 2: {e}")
        print("[CAUSE] Network issue, missing dependencies, or authentication (though anon is used).")
        print("[NEXT OPTION] Check internet connectivity or GCS access policies.")
        sys.exit(1)

if __name__ == '__main__':
    main()
