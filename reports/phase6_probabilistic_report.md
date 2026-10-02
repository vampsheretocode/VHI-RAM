# Phase 6 Probabilistic Forecast Report

## 1. Objective
Convert the Phase 5 adaptive deterministic blend into a fully calibrated probabilistic forecast providing P10, P50, and P90 percentiles, with uncertainty dynamically conditioned on model disagreement and lead time.

## 2. Data
- `data/verification/verification_2020.parquet`
- `data/weights/adaptive_weights_2020.parquet`

## 3. Method
- **Deterministic Center**: `P50` = `w_hres * HRES + w_pangu * PANGU` (using Phase 5 weights).
- **Residual Construction**: Empirical error `Observation - P50`.
- **Historical Calibration**: Quantiles (0.10, 0.90) extracted strictly from the Jan-Aug TRAIN split.
- **Disagreement Conditioning**: Residuals partitioned into Low/Medium/High disagreement bins based on exact `|HRES - PANGU|` terciles learned per lead-time from TRAIN data.
- **Leakage Prevention**: Validation/Test observations are completely isolated from calibration.

## 4. Calibration Parameters
See `artifacts/phase6_calibration.json`. All parameters were fitted on the 34.5M TRAIN split.

## 5. Results (Frozen TEST Split)

### Overall Probabilistic Metrics
- **MAE**: 1.0588
- **RMSE**: 1.6245
- **Bias**: -0.1153
- **80% Coverage**: 78.8%
- **Mean Interval Width**: 3.1160
- **CRPS**: NOT_COMPUTED (The empirical P10/P50/P90 discrete bounds do not uniquely define a continuous CDF).

### Disagreement Analysis
| Bin | Count | Mean Disagreement | Mean Abs Error | Mean Width | 80% Coverage |
|---|---|---|---|---|---|
| low | 2570297 | 0.2453 | 0.6880 | 1.9388 | 77.7% |
| medium | 2829672 | 0.9337 | 0.9114 | 2.7869 | 79.5% |
| high | 3235801 | 3.1528 | 1.4822 | 4.3388 | 79.0% |

## 6. Limitations
- **Reference**: ERA5 is used as the ground truth.
- **Data Limits**: Only 2020 data utilized.
- **Representation**: Only three quantiles are represented (P10, P50, P90). True CRPS cannot be robustly computed without parametric assumptions.

## 7. Honest Conclusion
The empirical disagreement-conditioned calibration successfully maps the adaptive blend into a probabilistic framework. Disagreement analysis proves that higher inter-model spread strictly correlates with higher realized absolute error, confirming the usefulness of disagreement as an uncertainty signal. The 80% coverage empirically aligns near the target on the TEST set, validating the isolation methodology.
