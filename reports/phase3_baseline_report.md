# Phase 3 Baseline Evaluation Report

## Dataset Details
- Total evaluated rows: 51814620
- Train cases: 34543080
- Validation cases: 8635770
- Test cases: 8635770

## Static Constrained Blend Weights
- **HRES Weight:** 0.1788
- **PANGU Weight:** 0.8212
*Weights were computed strictly on the TRAIN partition.*

## Leakage Protection Verification
- Static weights were frozen before evaluation on TEST.
- Training baseline fit did NOT use validation or test rows.
- Only forecast information available at initialization time was evaluated.

## Test Period Performance

### HRES
- MAE: 1.6702
- RMSE: 2.5102
- Bias: -0.3047

### PANGU
- MAE: 1.0657
- RMSE: 1.6450
- Bias: -0.0450

### Equal Weight Blend
- MAE: 1.1737
- RMSE: 1.7866
- Bias: -0.1748

### Static Blend
- MAE: 1.0414
- RMSE: 1.6058
- Bias: -0.0914

*Per lead-time metrics and validation set performance can be found in `artifacts/phase3_baseline_metrics.json`.*
