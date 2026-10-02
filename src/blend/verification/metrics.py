import numpy as np
import pandas as pd

def compute_metrics(y_true, y_pred):
    """Computes MAE, RMSE, and Mean Bias for given true and predicted values."""
    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    bias = np.mean(error)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Bias": float(bias)
    }

def evaluate_models(df, static_w):
    """
    Evaluates all models on a given dataframe.
    static_w: learned weight for HRES.
    Returns a dict with metrics.
    """
    y = df['observation_value'].values
    hres = df['hres_forecast'].values
    pangu = df['pangu_forecast'].values
    
    equal_blend = 0.5 * hres + 0.5 * pangu
    static_blend = static_w * hres + (1 - static_w) * pangu
    
    return {
        "HRES": compute_metrics(y, hres),
        "PANGU": compute_metrics(y, pangu),
        "Equal Weight": compute_metrics(y, equal_blend),
        "Static Blend": compute_metrics(y, static_blend),
        "samples": len(df)
    }
