# Cloud-Generated Threshold Artifact Specification & Adapter

## Phase 7B.3 Status: PAUSED (REMOTE DATA-ACCESS BOTTLENECK)

Local computation of the ERA5 climatology thresholds was halted because it was verified to be strictly network-bound and practically impossible on the current laptop machine.

### Findings Report
- **Zarr Chunks**: `(1, 721, 1440)` for `(time, latitude, longitude)`
- **Historical Period**: 1991–2019
- **Estimated Remote Transfer**: ~175.8 GB (full global temporal chunks)
- **India Extracted Storage**: ~2.4 GB (India spatial bounds only)
- **Local Computation**: NOT PRACTICAL under current machine/network constraints
- **Exact Climatology Methodology**: REMAINS UNCHANGED

### Required Scientific Method for Cloud Execution
The cloud compute script MUST implement this exact sequence:
1. Load `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr` (2m_temperature).
2. Filter strictly to the 1991–2019 historical period.
3. Spatially constrain to India (lat 8–37N, lon 68–98E).
4. Do NOT materialize the 1991–2019 sequence in local RAM simultaneously.
5. Compute the **smoothed day-of-year climatology**:
   - For each DOY (1–366) and for each grid cell:
   - Identify all dates within a **±15-day circular window** across the 29-year record.
   - Calculate the **95th percentile** of these temporally local observations.
6. Return the threshold artifact as a Parquet or Zarr file.

### Adapter Specification
When the cloud-generated threshold artifact is completed, it must be downloaded to:
`data/extreme_climatology/cloud_thresholds_95.parquet`

The adapter script in the blending system must expect:
- **Columns**: `latitude`, `longitude`, `day_of_year`, `threshold_95`, `sample_count`
- **Data Types**: Float32 for temperatures, Int16 for DOY and sample_count.
- **Null handling**: Grid cells over oceans or missing ERA5 values must be `NaN` safely.

Until this cloud artifact is injected, Phase 7C (Extreme Classifier Training) cannot proceed.
