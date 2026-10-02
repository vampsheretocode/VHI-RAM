# Phase 5 Adaptive Gating Report

## Model Details
- **Architecture**: LightGBM Regressors modeling absolute error magnitude for HRES and PANGU.
- **Transformation**: `w_HRES = pred_PANGU_err / (pred_HRES_err + pred_PANGU_err)`
- **Constraints**: Weights are bounded `[0, 1]` and sum exactly to 1. 

## Leakage Protection
- The model was trained purely on `TRAIN` cases (Jan-Aug).
- Missing historical skills (early Jan) were naturally passed to LightGBM which inherently handles `NaN` splits. No future imputation was used.
- Test observations were strictly isolated from the training graph.

## Weight Statistics (TEST)
- **Mean w_HRES**: 0.4221
- **Mean w_PANGU**: 0.5779
- **HRES favored (>50%)**: 20.4% of cases
- **PANGU favored (>50%)**: 79.6% of cases

## Final Frozen TEST Evaluation

### HRES
- MAE: 1.6702
- RMSE: 2.5102
- Bias: -0.3047

### PANGU
- MAE: 1.0657
- RMSE: 1.6450
- Bias: -0.0450

### Equal Weight
- MAE: 1.1737
- RMSE: 1.7866
- Bias: -0.1748

### Static Blend
- MAE: 1.0414
- RMSE: 1.6058
- Bias: -0.0914

### Adaptive Gate
- MAE: 1.0588
- RMSE: 1.6245
- Bias: -0.1153

## Conclusion
See `artifacts/phase5_feature_importance.json` for model interpretation.
Outputs saved to `data/weights/adaptive_weights_2020.parquet`.
