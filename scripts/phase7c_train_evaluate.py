import pandas as pd
import pyarrow.dataset as ds
import numpy as np
import os
import gc
import json
import yaml
import joblib
from datetime import datetime
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

print("--- PHASE 7C IMPLEMENTATION ---")

# Paths
PROB_PATH = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\probabilistic\probabilistic_forecasts_2020.parquet"
CLIM_PATH = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\extreme_climatology\era5_smooth_doy_95th.parquet"

ARTIFACTS_DIR = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\artifacts"
MODELS_DIR = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\models"
DATA_OUT_DIR = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\extreme_climatology"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# 1. Load Climatology
print("Loading climatology thresholds...")
df_clim = pd.read_parquet(CLIM_PATH)
# Rename for easier merging
df_clim = df_clim[['latitude', 'longitude', 'day_of_year', 'threshold_95']]

# 2. Setup Dataset
dataset = ds.dataset(PROB_PATH, format="parquet")
lead_times = [24, 48, 72, 120, 240]

metrics_out = {}
models_out = {}
manifest_info = {
    "lead_times": lead_times,
    "model_type": "LightGBM + Isotonic Calibration",
    "features": ["adaptive_blend_margin", "p90_margin", "interval_width", "forecast_disagreement"],
    "target": "is_extreme_observed",
    "baselines": ["Climatological Event Rate", "Deterministic Adaptive Blend"],
    "creation_timestamp": datetime.now().isoformat()
}

all_test_predictions = []

def calculate_contingency(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "POD": float(pod),
        "FAR": float(far),
        "CSI": float(csi),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }

for lead in lead_times:
    print(f"\n========== PROCESSING LEAD TIME {lead}h ==========")
    
    # Load specific lead time to manage memory
    print("Loading probabilistic data...")
    df_lead = dataset.to_table(filter=(ds.field('lead_time_hours') == lead)).to_pandas()
    
    # Calculate DOY from valid_time
    # valid_time is Unix epoch in nanoseconds
    df_lead['valid_datetime'] = pd.to_datetime(df_lead['valid_time'], unit='ns')
    df_lead['day_of_year'] = df_lead['valid_datetime'].dt.dayofyear
    
    # Merge with climatology
    print("Merging climatology...")
    df = df_lead.merge(df_clim, on=['latitude', 'longitude', 'day_of_year'], how='inner')
    
    del df_lead
    gc.collect()
    
    # Create Features
    print("Engineering features...")
    df['adaptive_blend_margin'] = df['adaptive_blend'] - df['threshold_95']
    df['p90_margin'] = df['p90'] - df['threshold_95']
    # 'interval_width' and 'forecast_disagreement' are already present
    
    # Create Targets
    df['is_extreme_observed'] = (df['observation_value'] >= df['threshold_95']).astype(int)
    
    # Create Deterministic Baseline
    df['baseline_det_pred'] = (df['adaptive_blend'] >= df['threshold_95']).astype(int)
    
    # Split
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    train_base_rate = df_train['is_extreme_observed'].mean()
    val_base_rate = df_val['is_extreme_observed'].mean()
    test_base_rate = df_test['is_extreme_observed'].mean()
    
    print(f"Base Rates - Train: {train_base_rate:.4f} | Val: {val_base_rate:.4f} | Test: {test_base_rate:.4f}")
    
    features = ['adaptive_blend_margin', 'p90_margin', 'interval_width', 'forecast_disagreement']
    
    X_train = df_train[features]
    y_train = df_train['is_extreme_observed']
    
    X_val = df_val[features]
    y_val = df_val['is_extreme_observed']
    
    X_test = df_test[features]
    y_test = df_test['is_extreme_observed']
    
    # Train Classifier
    print("Training LightGBM Classifier...")
    base_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1)
    base_clf.fit(X_train, y_train)
    
    # Calibrate on Validation Set
    print("Calibrating probabilities via Isotonic Regression on Val set...")
    val_raw_probs = base_clf.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_raw_probs, y_val)
    
    # Determine optimal decision threshold on Validation Set (maximize CSI)
    print("Selecting decision threshold on Validation set...")
    val_probs = calibrator.predict(val_raw_probs)
    
    best_csi = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        val_preds = (val_probs >= thresh).astype(int)
        csi = calculate_contingency(y_val, val_preds)['CSI']
        if csi > best_csi:
            best_csi = csi
            best_thresh = thresh
            
    print(f"Selected Threshold: {best_thresh:.2f} (Val CSI: {best_csi:.4f})")
    
    # Predict on Test (FROZEN EVALUATION)
    print("Evaluating strictly on TEST set...")
    test_raw_probs = base_clf.predict_proba(X_test)[:, 1]
    test_probs = calibrator.predict(test_raw_probs)
    test_preds = (test_probs >= best_thresh).astype(int)
    
    df_test = df_test.copy()
    df_test['classifier_prob'] = test_probs
    df_test['classifier_pred'] = test_preds
    df_test['baseline_clim_prob'] = train_base_rate
    
    # Metrics
    # 1. Climatological Baseline
    bs_clim = brier_score_loss(y_test, df_test['baseline_clim_prob'])
    
    # 2. Deterministic Adaptive Baseline
    # Has no calibrated prob, so BS is calculated on 0/1 predictions
    bs_det = brier_score_loss(y_test, df_test['baseline_det_pred'])
    cont_det = calculate_contingency(y_test, df_test['baseline_det_pred'])
    
    # 3. Classifier
    bs_clf = brier_score_loss(y_test, test_probs)
    cont_clf = calculate_contingency(y_test, test_preds)
    
    print(f"Brier Scores - Clim: {bs_clim:.4f} | Det: {bs_det:.4f} | Clf: {bs_clf:.4f}")
    print(f"CSI - Det: {cont_det['CSI']:.4f} | Clf: {cont_clf['CSI']:.4f}")
    
    metrics_out[str(lead)] = {
        "train_samples": int(len(df_train)),
        "val_samples": int(len(df_val)),
        "test_samples": int(len(df_test)),
        "train_base_rate": float(train_base_rate),
        "val_base_rate": float(val_base_rate),
        "test_base_rate": float(test_base_rate),
        "decision_threshold_selected_on_val": float(best_thresh),
        "baselines": {
            "climatological_brier_score": float(bs_clim),
            "deterministic_adaptive": {
                "brier_score": float(bs_det),
                **cont_det
            }
        },
        "classifier": {
            "brier_score": float(bs_clf),
            **cont_clf
        }
    }
    
    # Save Model (save both base clf and calibrator)
    models_out[str(lead)] = {'clf': base_clf, 'calibrator': calibrator, 'threshold': best_thresh}
    
    # Append test predictions for artifact (lightweight columns)
    cols_to_save = ['valid_time', 'latitude', 'longitude', 'lead_time_hours', 'observation_value', 'threshold_95', 'adaptive_blend', 'baseline_det_pred', 'classifier_prob', 'classifier_pred', 'is_extreme_observed']
    all_test_predictions.append(df_test[cols_to_save])
    
    del df, df_train, df_val, df_test, X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

# Save Artifacts
print("\nSaving Models...")
joblib.dump(models_out, os.path.join(MODELS_DIR, "extreme_event_classifier.model"))

print("Saving Metrics...")
with open(os.path.join(ARTIFACTS_DIR, "phase7c_extreme_metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)
    
print("Saving Test Predictions...")
df_all_test = pd.concat(all_test_predictions, ignore_index=True)
df_all_test.to_parquet(os.path.join(DATA_OUT_DIR, "extreme_predictions_test.parquet"), index=False)

print("Saving Manifest...")
manifest_info["metrics_summary"] = {k: {"clf_csi": v["classifier"]["CSI"], "det_csi": v["baselines"]["deterministic_adaptive"]["CSI"]} for k, v in metrics_out.items()}
with open(os.path.join(DATA_OUT_DIR, "phase7c_manifest.yml"), "w") as f:
    yaml.dump(manifest_info, f)

print("PHASE 7C IMPLEMENTATION COMPLETE.")
