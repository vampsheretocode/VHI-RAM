# PHASE 7C IMPLEMENTATION REPORT: EXTREME-HEAT CLASSIFIER

## 1. Architecture & Features
- **Exact Model:** LightGBM Classifier (`LGBMClassifier`) with 100 estimators, balanced class weights, mapped via Isotonic Regression. Five independent models were trained, one for each lead time (24h, 48h, 72h, 120h, 240h).
- **Features:** 
  1. `adaptive_blend_margin` (adaptive_blend - threshold_95)
  2. `p90_margin` (p90 - threshold_95)
  3. `interval_width`
  4. `forecast_disagreement`
- **Target Event:** `observation_value >= threshold_95` (using the rigorously derived 1991–2019 Phase 7B.3 cloud artifact).
- **Leakage Check:** Features depend strictly on prior forecast outputs. Thresholds depend strictly on 1991–2019 climatology. No target or verification data bled into the feature calculations.

## 2. Temporal Splits & Sample Counts
The data was strictly partitioned by the predefined `split` column, guaranteeing the chronological integrity of the 2020 test set.
For each of the 5 lead times:
- **Train Samples (2018–2019):** ~6,908,616
- **Validation Samples:** ~1,727,154
- **Test Samples (2020):** ~1,727,154

## 3. Event Prevalence (Base Rate)
The historically-defined 5% extreme event rate is subject to recent climate non-stationarity. Observed 2020 (Test) event rates were measured dynamically:
- **24h Lead:** 4.28%
- **48h Lead:** 4.27%
- **72h Lead:** 4.36%
- **120h Lead:** 4.65%
- **240h Lead:** 5.66%

## 4. Calibration & Decision Procedure
- **Calibration:** Because `CalibratedClassifierCV` with `cv="prefit"` is deprecated in modern scikit-learn, probabilities were manually and strictly calibrated using **Isotonic Regression** fitted exclusively on the `val` dataset.
- **Decision Threshold:** The binary decision threshold was optimized by maximizing the Critical Success Index (CSI) exclusively on the `val` set. No test data was consulted for thresholding.
  - Selected Thresholds: 24h (0.35), 48h (0.30), 72h (0.30), 120h (0.25), 240h (0.15).

## 5. Baselines
The probabilistic classifier was evaluated against two strict baselines:
1. **Climatological Baseline:** Assigning the historical training base rate as the probability for all test events.
2. **Deterministic Baseline:** The naive threshold exceedance condition (`adaptive_blend >= threshold_95`).

## 6. Frozen 2020 Test-Set Metrics
The classifier systematically outperformed both baselines across every single lead time.

| Lead Time | Brier (Climatology) | Brier (Deterministic) | **Brier (Classifier)** | CSI (Deterministic) | **CSI (Classifier)** |
|-----------|--------------------|-----------------------|-----------------------|---------------------|----------------------|
| **24h**   | 0.0412             | 0.0312                | **0.0236**            | 0.4026              | **0.4354**           |
| **48h**   | 0.0411             | 0.0354                | **0.0276**            | 0.3273              | **0.3635**           |
| **72h**   | 0.0419             | 0.0383                | **0.0308**            | 0.2795              | **0.3216**           |
| **120h**  | 0.0444             | 0.0431                | **0.0355**            | 0.2270              | **0.2719**           |
| **240h**  | 0.0534             | 0.0567*               | **0.0473**            | 0.1347              | **0.1861**           |

*(Note: At 240h, the naive deterministic baseline is actively worse than blind climatology in terms of Brier score [0.0567 > 0.0534]. The probabilistic classifier successfully rescues skill, yielding a highly superior 0.0473.)*

## 7. Artifacts Produced
- `artifacts/phase7c_extreme_metrics.json`: Detailed evaluation metrics (Brier, POD, FAR, CSI).
- `models/extreme_event_classifier.model`: The serialized dictionary of LightGBM + Isotonic models per lead time.
- `data/extreme_climatology/extreme_predictions_test.parquet`: The aggregated 2020 test predictions for subsequent dashboard injection.
- `data/extreme_climatology/phase7c_manifest.yml`: Traceable metadata and validation logs.

**STATUS: TRAINING AND EVALUATION COMPLETE. STOPPING FOR REVIEW BEFORE DASHBOARD INTEGRATION.**
