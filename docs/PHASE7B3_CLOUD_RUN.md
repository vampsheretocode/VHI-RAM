# Phase 7B.3 Cloud Climatology Execution Guide

This document provides exact, reproducible step-by-step instructions for computing the 1991–2019 ERA5 extreme heat climatology thresholds in a clean Google Colab environment. 

**Local computation was verified as not practical due to network bottlenecks and memory constraints caused by the remote Zarr chunk geometry. This calculation MUST be performed in a high-bandwidth cloud environment.**

## Prerequisites
- A Google Account
- Access to Google Colab (Free tier is usually sufficient, though a high-RAM runtime is recommended).

## Step-by-Step Instructions

### 1. Create a New Colab Notebook
Navigate to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### 2. Install Required Dependencies
In the first cell, install the necessary Python packages:
```python
!pip install xarray zarr gcsfs fsspec pandas numpy psutil pyyaml fastparquet
```

### 3. Verify Python/Runtime Versions
In the next cell, verify the environment and available RAM:
```python
import sys
import psutil
print(f"Python Version: {sys.version}")
mem = psutil.virtual_memory()
print(f"Available RAM: {mem.total / 1024**3:.2f} GB")
if mem.total / 1024**3 < 12.0:
    print("WARNING: Less than 12GB of RAM available. Consider switching to a High-RAM runtime.")
```

### 4. Verify GCS Access
Verify that the Colab instance can read the public WeatherBench2 bucket:
```python
import xarray as xr
try:
    ds = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr", consolidated=True)
    print("GCS Access Verified. Dataset Shape:", ds['2m_temperature'].shape)
except Exception as e:
    print("GCS Access Failed:", e)
```

### 5. Copy the Required Project Scripts & Configuration
Create the required directory structure and copy the contents of the scripts. You can copy the code directly from your local project into Colab cells or upload the files.
If uploading:
- Upload `configs/phase7b3_climatology.yml` to `/content/configs/`
- Upload `scripts/phase7b3_cloud_climatology.py` to `/content/scripts/`
- Upload `scripts/audit_phase7b3_climatology.py` to `/content/scripts/`

Create directories:
```bash
!mkdir -p configs scripts data/extreme_climatology artifacts
```

### 6. Run the Small Real-Data Benchmark
**DO NOT skip this step.** Run the benchmark to extract 1 month of data and verify memory/network throughput.
```bash
!python -u scripts/phase7b3_cloud_climatology.py --config configs/phase7b3_climatology.yml --benchmark
```

### 7. Inspect Benchmark Results
Read the generated `artifacts/phase7b3_cloud_benchmark.json`. 
- Ensure Peak RAM did not exceed safe limits (e.g., >85%).
- Evaluate the `estimated_full_runtime_hr`.
- *Note: Cloud network throughput varies. Do not expect a fixed runtime.*

### 8. Decide on Full Computation
If the `estimated_full_runtime_hr` is practical (e.g., < 2 hours) and memory was stable, proceed. Otherwise, halt and provision a stronger compute node.

### 9. Run the Full 1991–2019 Computation
Run the full 29-year reduction:
```bash
!python -u scripts/phase7b3_cloud_climatology.py --config configs/phase7b3_climatology.yml
```
This will generate `data/extreme_climatology/era5_smooth_doy_95th.parquet` and the manifest.

### 10. Run the Independent Audit
Verify that no leakage occurred and the methodology is structurally sound:
```bash
!python -u scripts/audit_phase7b3_climatology.py
```

### 11. Verify the Final Artifact and Manifest
Ensure both files exist and the audit passed without errors.

### 12. Download the Threshold Artifacts
Download exactly two files to your local machine:
1. `data/extreme_climatology/era5_smooth_doy_95th.parquet`
2. `data/extreme_climatology/manifest.yml`

### 13. Local Integration
Place the downloaded files into your local project exactly at:
- `<PROJECT_ROOT>/data/extreme_climatology/era5_smooth_doy_95th.parquet`
- `<PROJECT_ROOT>/data/extreme_climatology/manifest.yml`

This natively fulfills the `docs/cloud_climatology_adapter_spec.md` contract.
