# PHASE 7C DESIGN REPORT: EXTREME-HEAT EVALUATION

## 1. Available Data & Artifact Alignment
The foundation for Phase 7C is rigorously supported by two existing verified artifacts:
1. **Forecast & Verification Data** (`data/probabilistic/probabilistic_forecasts_2020.parquet`):
   - **Scale:** 51.8M rows (10.3M per lead time).
   - **Features:** `hres_forecast`, `pangu_forecast`, `adaptive_blend`, `p10`, `p50`, `p90`, `interval_width`, `forecast_disagreement`.
   - **Reference:** `observation_value` (ERA5 truth).
   - **Metadata:** `init_time`, `valid_time`, `lead_time_hours` (24, 48, 72, 120, 240), `latitude`, `longitude`, `split` (train/val/test).
2. **Climatological Thresholds** (`data/extreme_climatology/era5_smooth_doy_95th.parquet`):
   - **Scale:** 5.18M rows (117 Lats × 121 Lons × 366 DOYs).
   - **Features:** `threshold_95`.

**Alignment:** The datasets will be precisely merged using `latitude`, `longitude`, and `day_of_year` (extracted dynamically via `valid_time.dt.dayofyear`).

## 2. Proposed Event Definition
The extreme-heat event is defined dynamically at the grid-cell level:
```python
is_extreme_observed = (observation_value >= threshold_95).astype(int)
```
This is a binary target (1 = extreme heat day, 0 = normal day). 

## 3. Temporal Split & Evaluation Horizons
- **Split:** We will strictly reuse the existing `split` column from the Phase 1–6 pipeline (`train`, `val`, `test`), ensuring that the final evaluation remains strictly frozen on the out-of-sample **2020 chronological test set**.
- **Horizons:** The evaluation will be executed independently across all 5 supported lead times: **24h, 48h, 72h, 120h, 240h**.

## 4. Modeling Architecture & Features
While the P10/P50/P90 interval provides uncertainty bounds, extracting an exact exceedance probability $P(X \ge threshold)$ from non-parametric quantile bounds requires assumptions about the tail distribution. 
**Proposed Architecture:** A dedicated Extreme-Event Classifier (e.g., Logistic Regression or a shallow LightGBM classifier).
- **Features:**
  - `margin` (Distance to threshold): `adaptive_blend - threshold_95`
  - `p90_margin` (Upper bound distance): `p90 - threshold_95`
  - `interval_width` (Uncertainty proxy)
  - `forecast_disagreement`
- **Target:** `is_extreme_observed`
- **Output:** Calibrated probability of exceedance (0.0 to 1.0).

## 5. Scientifically Supportable Metrics
Since extreme events are highly imbalanced (~5% climatological baseline), standard regression metrics (MAE/RMSE) or raw accuracy are invalid. We will calculate meteorological contingency metrics on the 2020 test set:
- **Base Rate:** The observed frequency of events in 2020 (expected to be slightly >5% due to warming trends).
- **Brier Score (BS):** Measures the accuracy of the probabilistic predictions.
- **Probability of Detection (POD) / Recall:** True Positives / (True Positives + False Negatives).
- **False Alarm Ratio (FAR):** False Positives / (True Positives + False Positives).
- **Critical Success Index (CSI):** True Positives / (True Positives + False Positives + False Negatives) — the standard metric for severe weather forecasting.

## 6. Leakage Controls
- **Threshold Leakage:** The 95th percentile was verified via cloud-audit to be strictly limited to 1991–2019 data (0% 2020 leakage).
- **Information Leakage:** The `probabilistic_forecasts_2020.parquet` features are already strictly bounded by `init_time`, preventing forward-looking bias. 
- **Evaluation Leakage:** Metrics will be calculated exclusively where `split == "test"`.

## 7. Expected Artifacts
- `artifacts/phase7c_extreme_metrics.json`: The rigorous metric dictionary by lead time.
- `models/extreme_event_classifier.model`: The serialized classifier.
- `data/extreme_climatology/extreme_predictions_test.parquet`: The test-set predictions for dashboard integration.

## 8. Known Limitations
1. **Non-stationarity:** Because the thresholds are defined on 1991–2019, the true occurrence rate in the 2020 test set will likely exceed 5%. The model must be calibrated to the modern baseline.
2. **Deterministic Fallback:** A naive baseline (`adaptive_blend >= threshold_95`) will be evaluated as a control to prove the value of the probabilistic classifier. 

**STATUS: DESIGN COMPLETE. WAITING FOR APPROVAL TO IMPLEMENT PHASE 7C.**
