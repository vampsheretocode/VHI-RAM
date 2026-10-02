import pandas as pd
import json

path = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\extreme_climatology\extreme_predictions_test.parquet"
df = pd.read_parquet(path)

rows = len(df)
cols = list(df.columns)
nulls = df.isnull().sum().to_dict()
dups = df.duplicated(subset=['valid_time', 'latitude', 'longitude', 'lead_time_hours']).sum()
prob_min = df['classifier_prob'].min()
prob_max = df['classifier_prob'].max()
leads = df['lead_time_hours'].unique().tolist()

res = {
    "rows": rows,
    "columns": cols,
    "nulls": nulls,
    "duplicates": int(dups),
    "prob_min": float(prob_min),
    "prob_max": float(prob_max),
    "leads": leads
}
print(json.dumps(res, indent=2))
