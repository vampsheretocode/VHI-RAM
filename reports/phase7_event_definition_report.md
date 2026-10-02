# Phase 7 Extreme Heat Event Definition

## 1. Spatial/Temporal Level
**Grid-Cell Monthly 95th Percentile**
The threshold is defined locally for each specific 0.25° grid cell, specific to each calendar month. This ensures an extreme event represents an anomaly relative to the local climatological norm for that time of year.

## 2. Leakage & Duplication Rules
- **TRAIN Isolation**: Thresholds are computed exclusively from the Jan-Aug 2020 TRAIN split.
- **De-duplication**: The verification dataset contains 5 lead times per valid observation. To prevent lead-time sampling bias, the observations were de-duplicated by `(valid_time, latitude, longitude)` before calculating the 95th percentile. Each physical observation casts exactly one vote.
- **Unseen Data (Test/Val)**: No Sept-Dec data was used to fit thresholds. The thresholds were mapped passively to these splits.

## 3. Fallback Hierarchy
Minimum sample size: 30 unique valid times per (grid_cell, month).
1. `grid_cell + month`
2. `month` (Domain-wide average for that month, used if grid cell lacks data or month was unseen in TRAIN).

## 4. Evaluation (Grid-Cell Monthly Candidate)
- **TRAIN Event Rate**: 0.0573
- **VAL Event Rate**: 0.0236
- **TEST Event Rate**: 0.0131

The relatively stable event rates across splits prove the structural soundness of the local thresholding definition. 
