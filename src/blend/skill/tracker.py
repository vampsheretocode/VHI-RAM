import pandas as pd
import numpy as np

def compute_daily_errors(df):
    """
    Computes daily aggregate errors per lead_time and valid_time.
    We aggregate spatially over the domain because 60 samples/month per grid cell 
    is insufficient for stable RMSE tracking of extremes.
    """
    df = df.copy()
    df['hres_err'] = df['hres_forecast'] - df['observation_value']
    df['pangu_err'] = df['pangu_forecast'] - df['observation_value']
    
    agg_funcs = {
        'hres_err': ['mean', lambda x: np.mean(np.abs(x)), lambda x: np.mean(x**2), 'count'],
        'pangu_err': ['mean', lambda x: np.mean(np.abs(x)), lambda x: np.mean(x**2)]
    }
    
    daily = df.groupby(['valid_time', 'lead_time_hours']).agg(agg_funcs).reset_index()
    daily.columns = [
        'valid_time', 'lead_time_hours',
        'hres_bias', 'hres_mae', 'hres_mse', 'sample_count',
        'pangu_bias', 'pangu_mae', 'pangu_mse'
    ]
    return daily

def compute_rolling_skill(daily_errors, window_days=30):
    """
    Computes rolling 30-day skill metrics ending ON the valid_time.
    """
    daily_errors = daily_errors.sort_values('valid_time')
    
    results = []
    leads = daily_errors['lead_time_hours'].unique()
    for lead in leads:
        lead_df = daily_errors[daily_errors['lead_time_hours'] == lead].set_index('valid_time')
        rolling = lead_df.rolling('30D', min_periods=1).mean()
        rolling_counts = lead_df['sample_count'].rolling('30D', min_periods=1).sum()
        
        rolling['skill_sample_count'] = rolling_counts
        rolling['rolling_hres_rmse'] = np.sqrt(rolling['hres_mse'])
        rolling['rolling_pangu_rmse'] = np.sqrt(rolling['pangu_mse'])
        rolling['rolling_hres_mae'] = rolling['hres_mae']
        rolling['rolling_pangu_mae'] = rolling['pangu_mae']
        rolling['rolling_hres_bias'] = rolling['hres_bias']
        rolling['rolling_pangu_bias'] = rolling['pangu_bias']
        
        rolling['pangu_relative_skill'] = rolling['hres_mae'] / (rolling['hres_mae'] + rolling['pangu_mae'] + 1e-9)
        
        rolling = rolling.drop(columns=['hres_mse', 'pangu_mse', 'hres_mae', 'pangu_mae', 'hres_bias', 'pangu_bias', 'sample_count'])
        rolling = rolling.reset_index()
        rolling['lead_time_hours'] = lead
        results.append(rolling)
        
    return pd.concat(results, ignore_index=True)

def attach_historical_skill(df, rolling_skill):
    """
    Attaches the rolling skill to the forecast dataset without leakage.
    For a forecast initialized at T, we can only see observations up to T.
    Thus, we look for the rolling skill computed over valid_times <= T.
    """
    df = df.sort_values('init_time')
    rolling_skill = rolling_skill.sort_values('valid_time')
    
    merged = pd.merge_asof(
        df,
        rolling_skill,
        left_on='init_time',
        right_on='valid_time',
        by='lead_time_hours',
        direction='backward',
        suffixes=('', '_historical')
    )
    return merged
