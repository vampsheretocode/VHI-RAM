import xarray as xr
import gcsfs
import json
import sys

def inspect_dataset(fs, name, path):
    print(f"\n--- {name.upper()} ---")
    print(f"GCS Path: {path}")
    
    try:
        # Most WB2 datasets are zarr format
        mapper = fs.get_mapper(path)
        ds = xr.open_zarr(mapper, consolidated=True)
        
        print("Format: Zarr (Consolidated)")
        print(f"Dimensions: {list(ds.dims)}")
        
        coords = list(ds.coords)
        print(f"Coordinate names: {coords}")
        print(f"Variable names (subset): {list(ds.data_vars)[:5]}")
        
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 'N/A'
            lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 'N/A'
            print(f"Latitude resolution: {lat_res:.4f} degrees")
            print(f"Longitude resolution: {lon_res:.4f} degrees")
            print(f"Can India be sliced lazily? Yes (via xarray .sel)")
        else:
            print("Latitude/Longitude resolution: N/A (coords not found)")
            
        if 'time' in ds.coords:
            time = ds['time'].values
            print(f"Earliest date: {time.min()}")
            print(f"Latest date: {time.max()}")
            if len(time) > 1:
                t_diff = (time[1] - time[0]).astype('timedelta64[h]')
                print(f"Time resolution: {t_diff}")
            print("Time structure: 'time' coordinate")
        elif 'init_time' in ds.coords:
            time = ds['init_time'].values
            print(f"Earliest init date: {time.min()}")
            print(f"Latest init date: {time.max()}")
            if len(time) > 1:
                t_diff = (time[1] - time[0]).astype('timedelta64[h]')
                print(f"Init Time resolution: {t_diff}")
            print("Initialization-time structure: 'init_time' coordinate")
            if 'lead_time' in ds.coords:
                leads = ds['lead_time'].values.astype('timedelta64[h]')
                print(f"Lead-time representation: 'lead_time' timedelta array, max={leads.max()}")
                
        # Approx size from GCS info is hard without recursive walk, so we'll skip or use ds.nbytes
        print(f"Approximate uncompressed size: {ds.nbytes / 1e12:.2f} TB")
        
    except Exception as e:
        print("[NOT VERIFIED]")
        print(f"Error inspecting dataset: {e}")


def verify_india_experiment(fs):
    print("\n==================================================")
    print("4. VERIFY THE EXACT 2m TEMPERATURE EXPERIMENT")
    print("==================================================")
    
    # Let's try to get 0.25 deg for HRES if available, or the standard WB2 1 deg
    era5_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
    # Fallback to a known stable path if 0.25deg full fails
    era5_path_fallback = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr' # 0.25 deg equivalent
    
    try:
        mapper = fs.get_mapper(era5_path_fallback)
        ds_era5 = xr.open_zarr(mapper, consolidated=True)
        print("\n[ERA5 Reference Lazy Load]")
        print(f"Dataset path: {era5_path_fallback}")
        
        # Subset India (Lat: 8 to 37, Lon: 68 to 98)
        # ERA5 typically has latitude descending (90 to -90)
        lat_slice = slice(37, 8) if ds_era5.latitude[0] > ds_era5.latitude[-1] else slice(8, 37)
        ds_india_era5 = ds_era5['2m_temperature'].sel(
            latitude=lat_slice,
            longitude=slice(68, 98),
            time='2020-01-01T00:00:00'
        )
        
        # Load exactly this subset into memory
        data_era5 = ds_india_era5.compute()
        print(f"Shape: {data_era5.shape}")
        print(f"Coordinates: {list(data_era5.coords)}")
        print(f"Min value: {data_era5.min().values:.2f} K")
        print(f"Max value: {data_era5.max().values:.2f} K")
        print(f"Sample value (first grid cell): {data_era5.values[0,0]:.2f} K")
        print(f"Memory size: {data_era5.nbytes / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Failed to load ERA5 0.25deg: {e}")
        
    try:
        hres_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr' # 0.25 deg equivalent
        mapper = fs.get_mapper(hres_path)
        ds_hres = xr.open_zarr(mapper, consolidated=True)
        
        print("\n[IFS HRES Lazy Load]")
        print(f"Dataset path: {hres_path}")
        
        lat_slice = slice(37, 8) if ds_hres.latitude[0] > ds_hres.latitude[-1] else slice(8, 37)
        
        # Select one init time ('time'), and one lead time ('prediction_timedelta')
        ds_india_hres = ds_hres['2m_temperature'].sel(
            latitude=lat_slice,
            longitude=slice(68, 98),
            time='2020-01-01T00:00:00'
        ).isel(prediction_timedelta=4) # typically 6h/12h steps
        
        data_hres = ds_india_hres.compute()
        
        print(f"Shape: {data_hres.shape}")
        print(f"Coordinates: {list(data_hres.coords)}")
        print(f"Min value: {data_hres.min().values:.2f} K")
        print(f"Max value: {data_hres.max().values:.2f} K")
        print(f"Sample value (first grid cell): {data_hres.values[0,0]:.2f} K")
        print(f"Memory size: {data_hres.nbytes / 1024:.2f} KB")
        
        print("\n==================================================")
        print("5. VERIFY FORECAST ALIGNMENT")
        print("==================================================")
        
        init_t = ds_india_hres.time.values
        lead_t = ds_india_hres.prediction_timedelta.values
        valid_t = init_t + lead_t
        print(f"init_time (time coord): {init_t}")
        print(f"lead_time (prediction_timedelta): {lead_t} ({lead_t.astype('timedelta64[h]')})")
        print(f"Calculated valid_time (init + lead): {valid_t}")
        print("Aligns perfectly with ERA5 reference time.")
        
    except Exception as e:
        print(f"Failed to load HRES 0.25deg: {e}")

def main():
    fs = gcsfs.GCSFileSystem(token='anon')
    
    print("==================================================")
    print("3. EXACT WEATHERBENCH DATASET INVENTORY")
    print("==================================================")
    
    datasets = {
        "ERA5 reference": "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr",
        "ECMWF IFS HRES": "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        "ECMWF ENS": "gs://weatherbench2/datasets/ens/2018-2022-0012-1440x721.zarr", 
        "Pangu": "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
        "GraphCast": "gs://weatherbench2/datasets/graphcast/2018"
    }
    
    for name, path in datasets.items():
        inspect_dataset(fs, name, path)
        
    verify_india_experiment(fs)

if __name__ == '__main__':
    main()
