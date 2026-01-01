import squigglepy as sq
import numpy as np
import sys
import os
import multiprocessing as mp
from datetime import datetime, timedelta
import typing

# Add current directory to path
sys.path.append(os.getcwd())

from model import get_agi_task_length, run_model

def debug_sampling():
    print(f"Squigglepy version: {sq.__version__}")
    
    # Set seed like in run_model
    np.random.seed(42)
    
    n_samples = 1000
    
    print("Getting AGI task length from model...")
    agi_dist = get_agi_task_length()
    print(f"Distribution type: {type(agi_dist)}")
    
    try:
        print(f"Attempting sq.sample(agi_dist, n={n_samples})...")
        samples = sq.sample(agi_dist, n=n_samples)
        print(f"Success! Sample shape: {samples.shape}")
        print(f"First 5 samples: {samples[:5]}")
    except Exception as e:
        print(f"sq.sample failed: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 20)
    print("Attempting run_model(n_samples=100)...")
    try:
        days, dates = run_model(n_samples=100, seed=42)
        print("run_model success!")
    except Exception as e:
        print(f"run_model failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sampling()
