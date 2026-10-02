# Hybrid AI-NWP API and Dashboard Integration

This document outlines the architecture and integration of the production-ready API and Dashboard for the Hybrid AI-NWP Multi-Model Forecast Blending System. 

## 1. System Architecture
The integration layer consists of two decoupled components:
1. **Backend API (FastAPI)**: Serves verified scientific artifacts, frozen model weights, and inference stubs.
2. **Frontend Dashboard (Streamlit)**: Consumes the API and renders real-time forecasting data alongside rigorous scientific verification proofs.

## 2. API Endpoints (`src/blend/api/app.py`)
- `GET /api/v1/system/status`: Returns the verification status of all phases.
- `GET /api/v1/verification/metrics`: Exposes the frozen, verified out-of-sample metrics from Phase 3 (Static Blend), Phase 5 (Adaptive LightGBM), and Phase 6 (Probabilistic Output). **No marketing metrics are fabricated.**
- `GET /api/v1/forecast/current`: Provides a structured JSON payload of current deterministic and probabilistic outputs.
- `GET /api/v1/extremes/climatology`: **PENDING**. Checks for the presence of the cloud-generated artifact. Returns `NOT_AVAILABLE` gracefully if missing.
- `GET /api/v1/extremes/predictions`: **PENDING**. Awaits Phase 7C execution.

## 3. Dashboard Sections (`src/blend/dashboard/app.py`)
The dashboard is structurally aligned to the SIH presentation requirements:
- **Command Center**: High-level view of lead time, P50 blend, probabilistic uncertainty (P10/P90), and real-time model disagreement.
- **Forecast Output**: Tabular comparison of IFS HRES, Pangu-Weather, and the blended outputs.
- **Model Arbitration**: Transparent breakdown of model weights and an explanation of trust-shifting (why the weights differ between static and adaptive regimes).
- **Verification Lab**: The scientific proof panel. Explicitly displays MAE, RMSE, Bias, and Nominal Coverage. Openly documents that adaptive blending did not outperform static blending in this specific experiment.
- **Extreme Weather (Warning Module)**: Built as a strictly enforced API contract. Visualizes the "Processing in Cloud" status and halts further execution until the real Phase 7B.3 artifact is mounted.
- **System Status**: Real-time provenance and phase verification tracker.

## 4. Execution Instructions
### Start the API Server
```bash
uvicorn src.blend.api.app:app --host 0.0.0.0 --port 8000
```
### Start the Dashboard
```bash
streamlit run src/blend/dashboard/app.py
```

## 5. Constraints & Safety
- **No Fabrication**: The dashboard will never generate random probabilities or thresholds for the Extreme Weather module. It gracefully handles the 404/NOT_AVAILABLE response.
- **Modularity**: Once the `cloud_thresholds_95.parquet` artifact is dropped into the data directory, the API will immediately serve it as `AVAILABLE`, unlocking Phase 7C training.
