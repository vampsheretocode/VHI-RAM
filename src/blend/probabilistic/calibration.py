import pandas as pd
import numpy as np

class ResidualCalibrator:
    def __init__(self, quantiles=[0.1, 0.9]):
        self.quantiles = quantiles
        self.q_params = {}
        self.disagreement_thresholds = {}

    def fit(self, df_train):
        """
        Fits the residual quantiles and disagreement bins on TRAIN data.
        """
        df = df_train.copy()
        
        # Calculate adaptive blend P50
        df['p50'] = df['w_hres'] * df['hres_forecast'] + df['w_pangu'] * df['pangu_forecast']
        
        # Calculate residual
        df['residual'] = df['observation_value'] - df['p50']
        
        # Calculate disagreement
        df['forecast_disagreement'] = np.abs(df['hres_forecast'] - df['pangu_forecast'])
        
        # Determine disagreement bins per lead time
        for lead in df['lead_time_hours'].unique():
            lead_df = df[df['lead_time_hours'] == lead]
            
            # Using 33rd and 66th percentiles to define low/medium/high
            t33 = np.percentile(lead_df['forecast_disagreement'].dropna(), 33)
            t66 = np.percentile(lead_df['forecast_disagreement'].dropna(), 66)
            
            # Avoid duplicate edges
            if t33 == t66:
                t66 += 1e-6
                
            self.disagreement_thresholds[lead] = (float(t33), float(t66))
            
            # Assign bins
            def get_bin(val):
                if pd.isna(val): return 'medium'
                if val <= t33: return 'low'
                elif val <= t66: return 'medium'
                return 'high'
            
            lead_df = lead_df.copy()
            lead_df['bin'] = lead_df['forecast_disagreement'].apply(get_bin)
            
            # Calculate residual quantiles per bin
            for b in ['low', 'medium', 'high']:
                bin_df = lead_df[lead_df['bin'] == b]
                if not bin_df.empty:
                    q_vals = np.percentile(bin_df['residual'].dropna(), [q * 100 for q in self.quantiles])
                    self.q_params[f"{lead}_{b}"] = {
                        str(self.quantiles[0]): float(q_vals[0]),
                        str(self.quantiles[1]): float(q_vals[1])
                    }
                else:
                    q_vals = np.percentile(lead_df['residual'].dropna(), [q * 100 for q in self.quantiles])
                    self.q_params[f"{lead}_{b}"] = {
                        str(self.quantiles[0]): float(q_vals[0]),
                        str(self.quantiles[1]): float(q_vals[1])
                    }

    def get_calibration_artifact(self):
        return {
            "methodology": "Static quantiles of historical empirical residuals",
            "training_period": "TRAIN split (Jan-Aug)",
            "disagreement_bins": {str(k): [float(x) for x in v] for k, v in self.disagreement_thresholds.items()},
            "quantile_parameters": self.q_params
        }

    def predict(self, df):
        """
        Applies fitted calibration to produce P10, P50, P90.
        """
        # Vectorized implementation for speed on 50M rows
        df = df.copy()
        
        if 'p50' not in df.columns:
            df['p50'] = df['w_hres'] * df['hres_forecast'] + df['w_pangu'] * df['pangu_forecast']
            
        df['forecast_disagreement'] = np.abs(df['hres_forecast'] - df['pangu_forecast'])
        
        # Prepare arrays for vectorized mapping
        leads = df['lead_time_hours'].values
        disagg = df['forecast_disagreement'].values
        
        p10 = np.zeros(len(df))
        p90 = np.zeros(len(df))
        bins = np.empty(len(df), dtype=object)
        
        # Map per unique lead time
        for lead in np.unique(leads):
            mask = leads == lead
            disagg_lead = disagg[mask]
            
            if lead in self.disagreement_thresholds:
                t33, t66 = self.disagreement_thresholds[lead]
            else:
                t33, t66 = 0.5, 1.0 # fallback
                
            low_mask = disagg_lead <= t33
            med_mask = (disagg_lead > t33) & (disagg_lead <= t66)
            high_mask = disagg_lead > t66
            
            # Map bins
            bins_lead = np.empty(len(disagg_lead), dtype=object)
            bins_lead[low_mask] = 'low'
            bins_lead[med_mask] = 'medium'
            bins_lead[high_mask] = 'high'
            
            bins[mask] = bins_lead
            
            # Map quantiles
            p10_lead = np.zeros(len(disagg_lead))
            p90_lead = np.zeros(len(disagg_lead))
            
            for b_name, b_mask in [('low', low_mask), ('medium', med_mask), ('high', high_mask)]:
                if not np.any(b_mask): continue
                
                key = f"{lead}_{b_name}"
                if key in self.q_params:
                    q10_val = self.q_params[key][str(self.quantiles[0])]
                    q90_val = self.q_params[key][str(self.quantiles[1])]
                else:
                    q10_val, q90_val = -1.0, 1.0
                    
                p10_lead[b_mask] = q10_val
                p90_lead[b_mask] = q90_val
                
            p10[mask] = p10_lead
            p90[mask] = p90_lead
            
        df['disagreement_bin'] = bins
        df['p10'] = df['p50'] + p10
        df['p90'] = df['p50'] + p90
        
        return df
