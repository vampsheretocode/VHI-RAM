import pandas as pd
import pyarrow.parquet as pq

p = r"c:\Users\LENOVO\OneDrive\Desktop\SIH (VHI-RAM)\data\probabilistic\probabilistic_forecasts_2020.parquet"
df_meta = pq.read_metadata(p)
print(f"Total rows: {df_meta.num_rows}")

# we can read just the columns we need to count unique values to avoid massive memory
df_part = pd.read_parquet(p, columns=['split', 'lead_time_hours'])
print("Splits:", df_part['split'].value_counts())
print("Lead times:", df_part['lead_time_hours'].value_counts())
