import pytest
import numpy as np
from datetime import datetime
from model import calculate_doubling_time, run_model, O3_LAUNCH_DATE

def test_calculate_doubling_time_simple():
    # Simple case: start=1, agi=2, doubling=100, accel=1
    # Should take exactly 100 days
    days = calculate_doubling_time(1, 2, 100, 1)
    assert np.isclose(days, 100)

def test_calculate_doubling_time_accel():
    # Acceleration case
    # start=1, agi=4 (2 doublings), doubling=100
    # accel=0.5 (superexponential)
    # Formula: T * (1 - a^d) / (1 - a)
    # 100 * (1 - 0.5^2) / (1 - 0.5) = 100 * 0.75 / 0.5 = 150 days
    days = calculate_doubling_time(1, 4, 100, 0.5)
    assert np.isclose(days, 150)

def test_run_model_smoke():
    # Smoke test to ensure it runs without error
    days, dates = run_model(n_samples=100, seed=42)
    assert len(days) == 100
    assert len(dates) == 100
    assert not np.any(np.isnan(days))

def test_run_model_deterministic():
    # Test with fixed values
    days, dates = run_model(
        n_samples=10,
        start_task_length=1.0,
        agi_task_length=2.0,
        doubling_time=100.0,
        acceleration=1.0,
        shift=0.0,
        seed=42
    )
    # Should be close to 100 days (plus measurement error)
    # Measurement error is lognormal(0, 0.1), so mean is exp(0.1^2/2) ~= 1.005
    # So roughly 100 days
    assert np.all(days > 50) and np.all(days < 150)
