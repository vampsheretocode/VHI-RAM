import numpy as np

def learn_static_weights(df_train):
    """
    Learns non-negative weights that sum to 1.
    Minimizes MSE(w*HRES + (1-w)*PANGU, OBS)
    """
    # y = w * x1 + (1-w) * x2
    # w * (x1 - x2) = y - x2
    x1 = df_train['hres_forecast'].values
    x2 = df_train['pangu_forecast'].values
    y = df_train['observation_value'].values
    
    # Filter out NaNs if any (we already proved there are 0 NaNs, but good practice)
    valid = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    x1 = x1[valid]
    x2 = x2[valid]
    y = y[valid]

    diff = x1 - x2
    target = y - x2
    
    numerator = np.sum(target * diff)
    denominator = np.sum(diff**2)
    
    if denominator == 0:
        w = 0.5
    else:
        w = numerator / denominator
        
    w = float(np.clip(w, 0.0, 1.0))
    
    return {
        "HRES": w,
        "PANGU": 1.0 - w
    }
