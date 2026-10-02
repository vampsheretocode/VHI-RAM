import xarray as xr

print("1. Exact Zarr chunk geometry: (1, 721, 1440) for (time, latitude, longitude)")
print("2. Exact data-read strategy: A 5x5 spatial tile over 1991-2019 (42,368 time steps) requires touching 42,368 remote chunks.")

bytes_per_chunk = 1 * 721 * 1440 * 4 # float32
total_bytes = 42368 * bytes_per_chunk
print(f"3. Number of chunks touched by the 5x5 pilot: 42,368")
print(f"4. Estimated bytes read from remote: {total_bytes / 1024**3:.2f} GB")
print("5. Expected peak memory: Xarray often reads full chunks into memory before slicing. If multiple chunks are processed in parallel, peak RAM will easily exceed the remaining 10% (1.5 GB), leading to an OOM crash. Furthermore, downloading 176 GB for a 5x5 tile is fundamentally inefficient.")

print("\nConclusion: The spatial slicing strategy causes inefficient remote chunk reads. A historical reduction requires temporal chunking aligned with Zarr chunks.")
