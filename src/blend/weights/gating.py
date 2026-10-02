import numpy as np
import lightgbm as lgb
import pandas as pd
import json
import os

class AdaptiveGatingModel:
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=6):
        self.hres_model = lgb.LGBMRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.pangu_model = lgb.LGBMRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.features = [
            'lead_time_hours', 'latitude', 'longitude',
            'rolling_hres_rmse', 'rolling_pangu_rmse',
            'rolling_hres_mae', 'rolling_pangu_mae',
            'rolling_hres_bias', 'rolling_pangu_bias',
            'pangu_relative_skill', 'skill_sample_count',
            'forecast_diff', 'month_sin', 'month_cos'
        ]

    def _prepare_features(self, df):
        """Constructs features while explicitly handling missing history and avoiding leakage."""
        X = pd.DataFrame()
        X['lead_time_hours'] = df['lead_time_hours']
        X['latitude'] = df['latitude']
        X['longitude'] = df['longitude']
        
        # Historical skill features (can be NaN, LightGBM handles natively)
        X['rolling_hres_rmse'] = df['rolling_hres_rmse']
        X['rolling_pangu_rmse'] = df['rolling_pangu_rmse']
        X['rolling_hres_mae'] = df['rolling_hres_mae']
        X['rolling_pangu_mae'] = df['rolling_pangu_mae']
        X['rolling_hres_bias'] = df['rolling_hres_bias']
        X['rolling_pangu_bias'] = df['rolling_pangu_bias']
        X['pangu_relative_skill'] = df['pangu_relative_skill']
        X['skill_sample_count'] = df['skill_sample_count']
        
        # Current forecast context
        X['forecast_diff'] = np.abs(df['hres_forecast'] - df['pangu_forecast'])
        
        # Month cyclical encoding
        months = df['init_time'].dt.month
        X['month_sin'] = np.sin(months * (2. * np.pi / 12))
        X['month_cos'] = np.cos(months * (2. * np.pi / 12))
        
        return X

    def fit(self, df_train):
        X_train = self._prepare_features(df_train)
        
        y_hres_err = np.abs(df_train['hres_forecast'] - df_train['observation_value'])
        y_pangu_err = np.abs(df_train['pangu_forecast'] - df_train['observation_value'])
        
        # Handle outliers in target (weather extremes) by capping at 99.9th percentile for stability
        cap_h = np.percentile(y_hres_err.dropna(), 99.9)
        cap_p = np.percentile(y_pangu_err.dropna(), 99.9)
        y_hres_err = np.clip(y_hres_err, 0, cap_h)
        y_pangu_err = np.clip(y_pangu_err, 0, cap_p)
        
        self.hres_model.fit(X_train, y_hres_err)
        self.pangu_model.fit(X_train, y_pangu_err)
        
        # Feature importance
        self.feature_importances = {
            'HRES_Model': dict(zip(self.features, self.hres_model.feature_importances_.tolist())),
            'PANGU_Model': dict(zip(self.features, self.pangu_model.feature_importances_.tolist()))
        }

    def predict_weights(self, df):
        X = self._prepare_features(df)
        
        pred_hres_err = self.hres_model.predict(X)
        pred_pangu_err = self.pangu_model.predict(X)
        
        # Prevent negative predictions and zero division
        pred_hres_err = np.clip(pred_hres_err, 1e-6, None)
        pred_pangu_err = np.clip(pred_pangu_err, 1e-6, None)
        
        # If model predicts PANGU error is high, HRES weight should be high
        w_hres = pred_pangu_err / (pred_hres_err + pred_pangu_err)
        w_pangu = 1.0 - w_hres
        
        return w_hres, w_pangu

    def save_importances(self, path):
        with open(path, 'w') as f:
            json.dump(self.feature_importances, f, indent=2)
