import subprocess
import sys
import platform

def check_compatibility():
    print("=== PYTHON COMPATIBILITY CHECK ===")
    print(f"Python version: {platform.python_version()}")
    
    # Packages we currently have
    current_packages = ["xarray", "zarr", "gcsfs", "fsspec", "dask", "numpy", "pandas", "pyyaml"]
    future_packages = ["scipy", "scikit-learn", "lightgbm", "xskillscore", "properscoring"]
    
    print("\nChecking future packages using pip dry-run:")
    
    try:
        # Run pip install --dry-run
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", *future_packages],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[SUCCESS] All planned future packages can be resolved for Python 3.14.")
        else:
            print("[WARNING] Pip failed to resolve some future packages for Python 3.14.")
            print(result.stderr)
            print("[RECOMMENDATION] Python 3.14 may be too new for stable ML packages like lightgbm or xskillscore. Recommend Python 3.11 or 3.12.")
    except Exception as e:
        print(f"Error running pip: {e}")

if __name__ == "__main__":
    check_compatibility()
