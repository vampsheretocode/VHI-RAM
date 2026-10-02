import pandas as pd
import sys
import os

def test_reopen():
    print("\n--- DIAGNOSTIC REOPEN TEST ---")
    path = 'data/phase1_diagnostic.parquet'
    if not os.path.exists(path):
        print("[ERROR] Parquet file not found!")
        sys.exit(1)
        
    try:
        df = pd.read_parquet(path)
        print(f"Schema (Columns):\n{df.dtypes}")
        print(f"Row count: {len(df)}")
        print("\nFirst few rows:")
        print(df.head(3).to_string())
        print(f"\nFile size: {os.path.getsize(path)} bytes")
        
        print("\n[SUCCESS] Parquet footer/magic bytes successfully verified.")
    except Exception as e:
        print(f"[ERROR] Reopen test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    test_reopen()
    sys.exit(0)
