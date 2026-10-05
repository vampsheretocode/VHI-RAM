from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from pathlib import Path
from typing import Dict, Any

app = FastAPI(
    title="Hybrid AI-NWP Multi-Model Forecast Blending API",
    description="Production API exposing verified Phase 3-6 artifacts and serving as the data contract for the integration dashboard.",
    version="1.1.0"
)

ARTIFACT_DIR = Path(__file__).parent.parent.parent.parent / "artifacts"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

@app.get("/")
def read_root():
    return {"status": "online", "message": "Hybrid Blending API is running."}

@app.get("/api/v1/system/status")
def get_system_status():
    return {
        "status": "online",
        "phases_verified": ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"],
        "phases_pending": ["Phase 7B.3 (Cloud execution)", "Phase 7C"],
        "data_provenance": "WeatherBench2 ERA5, IFS HRES, Pangu-Weather",
        "last_verification_status": "Phase 6 Probabilistic Output Verified"
    }

@app.get("/api/v1/forecast/trajectory")
def get_forecast_trajectory():
    """
    Returns the real verified MAE and Interval Width across lead times to construct the Explorer chart.
    """
    try:
        with open(ARTIFACT_DIR / "phase5_test_metrics.json", "r") as f:
            p5 = json.load(f)["per_lead_time"]
        with open(ARTIFACT_DIR / "phase6_probabilistic_metrics.json", "r") as f:
            p6 = json.load(f)["lead_time"]
            
        trajectory = []
        for lt in ["24", "48", "72", "120", "240"]:
            trajectory.append({
                "lead_time": int(lt),
                "HRES": p5[lt]["HRES"]["MAE"],
                "PANGU": p5[lt]["PANGU"]["MAE"],
                "Equal Weight": p5[lt]["Equal Weight"]["MAE"],
                "Static Blend": p5[lt]["Static Blend"]["MAE"],
                "Adaptive Blend": p5[lt]["Adaptive Gate"]["MAE"],
                "Interval Width": p6[lt]["mean_interval_width"]
            })
        return {"status": "success", "data": trajectory}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/verification/metrics")
def get_verification_metrics():
    """
    Returns the explicitly verified Phase 3-6 metrics for the scientific audit panel.
    """
    return {
        "static_blend": {
            "hres_weight": 0.1788111627,
            "pangu_weight": 0.8211888373,
            "test_mae": 1.0414,
            "test_rmse": 1.6058,
            "test_bias": -0.0914
        },
        "adaptive_blend": {
            "mean_hres_weight": 0.4221,
            "mean_pangu_weight": 0.5779,
            "test_mae": 1.0588,
            "test_rmse": 1.6245,
            "test_bias": -0.1153,
            "note": "Adaptive did NOT outperform static."
        },
        "probabilistic": {
            "nominal_coverage": 0.80,
            "actual_coverage": 0.7878,
            "mean_interval_width": 3.1160,
            "test_mae": 1.0588,
            "test_rmse": 1.6245,
            "test_bias": -0.1153,
            "note": "Higher disagreement was associated with higher realized error and wider intervals."
        }
    }

from fastapi import Query
import functools
import pyarrow.dataset as ds
import pandas as pd

@functools.lru_cache(maxsize=100)
def fetch_reference_forecast(lat: float, lon: float, init_time: str, lead_time_hours: int):
    # Lazy read and pushdown filter to prevent loading the 3.5GB parquet into memory
    dataset = ds.dataset(DATA_DIR / "probabilistic" / "probabilistic_forecasts_test.parquet", format="parquet")
    table = dataset.to_table(filter=(
        (ds.field("split") == "test") &
        (ds.field("init_time") == pd.Timestamp(init_time)) &
        (ds.field("lead_time_hours") == lead_time_hours) &
        (ds.field("latitude") == lat) &
        (ds.field("longitude") == lon)
    ))
    rows = table.to_pylist()
    return rows[0] if rows else None

@app.get("/api/v1/forecast/current")
def get_current_forecast(
    lat: float = Query(17.75, description="Latitude"),
    lon: float = Query(68.25, description="Longitude"),
    init_time: str = Query("2020-11-01 00:00:00", description="Initialization time"),
    lead_time: int = Query(120, description="Lead time in hours")
):
    """
    Returns a verified Phase 6 probabilistic forecast case drawn from the frozen 2020 evaluation set.
    This serves as the data contract for the dashboard's Verified 2020 Reference Case.
    """
    try:
        sample = fetch_reference_forecast(lat, lon, init_time, lead_time)
        if not sample:
            return {"status": "error", "message": "No precomputed forecast is available for this selection."}
        
        hres_val = float(sample['hres_forecast'])
        pangu_val = float(sample['pangu_forecast'])
        
        # Load static weights from artifact
        with open(ARTIFACT_DIR / "phase3_static_weights.json", "r") as f:
            static_weights = json.load(f)
        static_p50 = (hres_val * static_weights["HRES"]) + (pangu_val * static_weights["PANGU"])

        return {
            "lead_time": f"{int(sample['lead_time_hours'])}h",
            "init_time": str(sample['init_time']),
            "valid_time": str(sample['valid_time']),
            "latitude": float(sample['latitude']),
            "longitude": float(sample['longitude']),
            "hres_forecast": round(hres_val, 2),
            "pangu_forecast": round(pangu_val, 2),
            "static_blend_p50": round(static_p50, 2),
            "adaptive_blend_p50": round(float(sample['adaptive_blend']), 2),
            "disagreement": round(float(sample['forecast_disagreement']), 2),
            "probabilistic": {
                "p10": round(float(sample['p10']), 2),
                "p90": round(float(sample['p90']), 2),
                "interval_width": round(float(sample['interval_width']), 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/extremes/climatology")
def get_extreme_climatology():
    cloud_artifact = DATA_DIR / "extreme_climatology" / "era5_smooth_doy_95th.parquet"
    if cloud_artifact.exists():
        return {"status": "AVAILABLE", "message": "Phase 7B.3 ERA5 Extreme Heat Climatology artifact is mounted."}
    else:
        return {
            "status": "NOT_AVAILABLE", 
            "message": "Phase 7B.3 threshold artifact is pending."
        }

@app.get("/api/v1/extremes/evaluation")
def get_extreme_evaluation():
    try:
        with open(ARTIFACT_DIR / "phase7c_extreme_metrics.json", "r") as f:
            metrics = json.load(f)
        with open(DATA_DIR / "extreme_climatology" / "phase7c_manifest.yml", "r") as f:
            import yaml
            manifest = yaml.safe_load(f)
        return {
            "status": "success",
            "metrics": metrics,
            "manifest": manifest
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load Phase 7C artifacts: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
