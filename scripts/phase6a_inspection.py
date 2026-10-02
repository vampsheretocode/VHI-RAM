import pyarrow.parquet as pq
import pandas as pd
import logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/phase6_probabilistic.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def inspect_parquet(name, path):
    logging.info(f"--- Inspecting {name} ---")
    meta = pq.read_metadata(path)
    logging.info(f"Row count: {meta.num_rows}")
    
    schema = pq.read_schema(path)
    logging.info(f"Columns: {schema.names}")
    for name_col in schema.names:
        logging.info(f"  {name_col}: {schema.field(name_col).type}")
        
    # Read a sample to show split/lead distribution if applicable
    # We will read specific columns to be efficient
    cols_to_read = [c for c in schema.names if c in ['split', 'lead_time_hours']]
    if cols_to_read:
        df = pq.read_table(path, columns=cols_to_read).to_pandas()
        if 'split' in df.columns:
            logging.info(f"Split distribution:\n{df['split'].value_counts()}")
        if 'lead_time_hours' in df.columns:
            logging.info(f"Lead distribution:\n{df['lead_time_hours'].value_counts()}")
            
    # Check nulls for relevant fields (weights)
    cols_to_check = [c for c in schema.names if c in ['w_hres', 'w_pangu', 'adaptive_blend']]
    if cols_to_check:
        df_nulls = pq.read_table(path, columns=cols_to_check).to_pandas()
        for c in cols_to_check:
            logging.info(f"Nulls in {c}: {df_nulls[c].isnull().sum()}")
            
    logging.info("-" * 40)

if __name__ == "__main__":
    logging.info("STARTING PHASE 6A INSPECTION")
    
    inspect_parquet("Verification Data", 'data/verification/verification_2020.parquet')
    inspect_parquet("Historical Skill Data", 'data/skill/historical_skill_2020.parquet')
    inspect_parquet("Adaptive Weights Data", 'data/weights/adaptive_weights_2020.parquet')
    
    logging.info("INSPECTION COMPLETE")
