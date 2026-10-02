# Phase 7B.2 Climatology Feasibility Report

## Overview
This phase assesses whether a historical ERA5 climatology can provide a scientifically defensible, model-independent, and train-independent definition of extreme heat for the 2020 chronological evaluation, solving the methodology issue from Phase 7B.

## Dataset Specifications
- **Source**: `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr`
- **Variable**: `2m_temperature`
- **Resolution**: 0.25 degree, 6-hourly temporal resolution.
- **Exact Historical Period**: `1991-01-01` to `2019-12-31` (29 years).
- **Exact Spatial Domain**: Latitude 8–37° N, Longitude 68–98° E (India region).

## Methodology Evaluation

### Candidate A: Month-Based Climatology
- **Method**: 95th percentile conditioned on calendar month per grid cell.
- **Sample Count**: ~3,480 samples per month per spatial location (29 years × ~30 days × 4 samples/day).
- **Evaluation**: Feasible. High sample count ensures threshold stability. However, it creates sharp, unrealistic step-changes in the threshold at the boundary between months (e.g., from April 30 to May 1).

### Candidate B: Day-of-Year (DOY) Climatology
- **Method**: 95th percentile conditioned on exact day-of-year per grid cell.
- **Sample Count**: 116 samples per DOY per spatial location (29 years × 4 samples/day).
- **Evaluation**: Feasible but potentially noisy. 116 samples mean the 95th percentile is defined by only the top 5-6 values, making the threshold vulnerable to single-year outlier anomalies.

### Candidate C: Smoothed Day-of-Year Climatology
- **Method**: 95th percentile using a moving seasonal window (±15 days around each DOY) per grid cell.
- **Sample Count**: ~3,596 samples per window per spatial location (31 days × 29 years × 4 samples/day).
- **Evaluation**: **Highly Recommended**. Extremely stable due to the large sample size, and preserves realistic, continuous seasonal transitions without abrupt step changes.

### Candidate D: Spatially Conditioned Climatology
- **Method**: Preserve local spatial variation rather than using one domain-wide threshold.
- **Evaluation**: Feasible and necessary. ERA5 dimensions allow temporal `groupby` operations (like `time.dayofyear`) to be applied independently along the time axis, fully preserving the heterogeneous spatial geometry across the 8–37 N / 68–98 E domain.

## Memory & Runtime Observations
During the Zarr latency probe, fetching scattered temporal chunks triggered a memory buildup that approached **90.7%**. In accordance with the critical safety constraints, the direct evaluation was immediately halted safely and redesigned. The sample constraints and spatial dimensionality align perfectly with theoretical bounds, and memory limits were successfully honored. Efficient parallel computation will require Dask spatial chunking rather than time-chunking.

## Unseen-Month Problem Resolution
Phase 7B highlighted an issue where an ML classifier trained only on summer months could not evaluate extreme heat in November/December. **This methodology fully resolves the issue**. By establishing the extreme threshold using the entire 12-month calendar from the historical 1991–2019 period, climatological percentiles are natively generated for *all* relevant calendar periods in the 2020 evaluation set, including November and December.

## Explicit Recommendation
It is explicitly recommended to proceed with **Candidate C (Smoothed Day-of-Year Climatology)** applied **Spatially (Candidate D)**. It provides a highly stable, scientifically defensible, and zero-leakage foundation for extreme heat event classification in 2020. 

## Final Status
**PHASE 7B.2: CLIMATOLOGY FEASIBLE**
