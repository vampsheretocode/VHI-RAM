import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.blend.probabilistic.calibration import ResidualCalibrator

def test_probabilistic_module():
    # 1. Create a dummy TRAIN dataset
    df_train = pd.DataFrame({
        'hres_forecast': [10.0, 20.0, 30.0, 40.0],
        'pangu_forecast': [12.0, 18.0, 35.0, 42.0],
        'w_hres': [0.5, 0.2, 0.8, 0.5],
        'w_pangu': [0.5, 0.8, 0.2, 0.5],
        'observation_value': [10.5, 19.5, 30.0, 41.5],
        'lead_time_hours': [24, 24, 48, 48]
    })
    
    calibrator = ResidualCalibrator(quantiles=[0.1, 0.9])
    calibrator.fit(df_train)
    
    # 2. Predict on TEST dataset
    df_test = pd.DataFrame({
        'hres_forecast': [15.0, 25.0],
        'pangu_forecast': [16.0, 22.0],
        'w_hres': [0.5, 0.5],
        'w_pangu': [0.5, 0.5],
        'observation_value': [15.2, 23.0], # Used only for checking, not calibration
        'lead_time_hours': [24, 48]
    })
    
    res = calibrator.predict(df_test)
    
    # Verify P50 Reconstruction
    expected_p50 = df_test['w_hres'] * df_test['hres_forecast'] + df_test['w_pangu'] * df_test['pangu_forecast']
    np.testing.assert_allclose(res['p50'], expected_p50)
    print("WEIGHT/BLEND RECONSTRUCTION: PASS")
    
    # Verify Quantile Ordering
    assert (res['p10'] <= res['p50']).all()
    assert (res['p50'] <= res['p90']).all()
    print("QUANTILE ORDERING: PASS")
    
    # Verify Interval Width
    assert (res['p90'] - res['p10'] >= 0).all()
    print("INTERVAL WIDTH: PASS")
    
    # Verify Artifact Serialization
    artifact = calibrator.get_calibration_artifact()
    assert 'methodology' in artifact
    assert 'quantile_parameters' in artifact
    print("CALIBRATION ARTIFACT: PASS")

if __name__ == "__main__":
    test_probabilistic_module()
