import numpy as np
import squigglepy as sq
from datetime import datetime, timedelta
import multiprocessing as mp

# --- Constants ---
O3_LAUNCH_DATE = datetime(2025, 4, 16)
CLAUDE_3P7_LAUNCH_DATE = datetime(2025, 2, 24)
# Add latest model release dates for better reference
GPT4O_LAUNCH_DATE = datetime(2024, 5, 13)
CLAUDE_3_OPUS_LAUNCH_DATE = datetime(2024, 3, 4)

# --- Model Functions ---
import typing
import sys

print("DEBUG: Loading model.py...", flush=True)

def calculate_doubling_time(
    start_task_length: float | np.ndarray | object, 
    agi_task_length: float | np.ndarray | object, 
    doubling_time: float | np.ndarray | object, 
    acceleration: float | np.ndarray | object = 1
) -> float | np.ndarray:
    """
    Calculate the time needed to reach AGI capability based on current capability and growth parameters.
    
    Args:
        start_task_length (float): Hours for current AI to complete reference task
        agi_task_length (float): Hours defining AGI-level task completion
        doubling_time (float): Days for AI capability to double
        acceleration (float, optional): Growth rate modifier. Defaults to 1 (exponential).
            <1: superexponential growth, >1: subexponential growth
    
    Returns:
        float: Days until AGI is reached
    """
    # Input validation
    for name, value in [
        ("start_task_length", start_task_length),
        ("agi_task_length", agi_task_length),
        ("doubling_time", doubling_time),
        ("acceleration", acceleration)
    ]:
        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            raise ValueError(f"{name} contains invalid values (NaN or Inf)")

    # Calculate capability ratio and required doublings
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.maximum(agi_task_length / start_task_length, 1e-10)  # Avoid division by zero
        doublings_needed = np.log(ratio) / np.log(2)
        
        # Handle acceleration = 1 (exponential) case efficiently
        if isinstance(acceleration, (int, float)) and acceleration == 1:
            return doublings_needed * doubling_time
            
        # Handle array input for acceleration
        if isinstance(acceleration, np.ndarray):
            # Create result array
            result = np.zeros_like(doublings_needed)
            
            # Find indices where acceleration is effectively 1
            is_one = np.isclose(acceleration, 1.0)
            
            # Calculate for acceleration == 1
            if np.any(is_one):
                result[is_one] = doublings_needed[is_one] * doubling_time[is_one]
                
            # Calculate for acceleration != 1
            not_one = ~is_one
            if np.any(not_one):
                accel_not_one = acceleration[not_one]
                doublings_not_one = doublings_needed[not_one]
                dt_not_one = doubling_time[not_one]
                
                power_term = np.power(accel_not_one, doublings_not_one)
                result[not_one] = dt_not_one * (1 - power_term) / (1 - accel_not_one)
                
            return np.clip(np.nan_to_num(result, nan=365*50, posinf=365*100, neginf=0), 0, 365*100)

        # Handle general case with acceleration (scalar != 1)
        power_term = np.power(acceleration, doublings_needed)
        result = doubling_time * (1 - power_term) / (1 - acceleration)
        
        # Handle edge cases
        result = np.nan_to_num(result, nan=365*50, posinf=365*100, neginf=0)
        return np.clip(result, 0, 365*100)  # Cap at 100 years
    # If acceleration is a constant 1, use the simple formula
    # Otherwise, use the general formula
    # This works for both scalars and squigglepy distributions

    # Try to detect if acceleration is exactly 1 (float or sq.Constant)
    is_one = False
    try:
        # If it's a squigglepy constant
        if hasattr(acceleration, 'value'):
            is_one = float(acceleration.value) == 1.0
        else:
            is_one = float(acceleration) == 1.0
    except Exception:
        pass

    ratio = agi_task_length / start_task_length
    doublings_needed = sq.dist_log(ratio) / np.log(2)

    if is_one:
        return doublings_needed * doubling_time
    else:
        # Use ** for exponentiation with distributions
        power_term = acceleration ** doublings_needed
        return doubling_time * (1 - power_term) / (1 - acceleration)

def samples_to_date(samples, index_date=O3_LAUNCH_DATE):
    # Convert samples to numpy array if it's not already
    samples_array = np.asarray(samples)
    
    # Replace any NaN values with a reasonable default (5 years)
    samples_array = np.nan_to_num(samples_array, nan=365*5)
    
    # Clip extremely large values to prevent date overflow
    # (100 years should be enough for most forecasts)
    max_days = 365 * 100  # 100 years
    samples_array = np.clip(samples_array, 0, max_days)
    
    # Round up to nearest integer day and convert to int safely
    days = np.ceil(samples_array).astype(np.int64)
    
    # Vectorized date conversion
    date_converter = np.vectorize(lambda x: index_date + timedelta(days=int(x)))
    return date_converter(days)

# --- Default Distributions (can be overridden) ---
def get_start_task_length(n_samples=100_000):
    current_best = 1.75  # o3 task length at 50% reliability?
    elicitation_boost = sq.mixture([[0.3, 1], [0.4, 1.2], [0.3, 1.5]])
    reliability_needed = sq.mixture([[0.3, 0.5], [0.4, 0.8], [0.2, 0.9], [0.1, 0.99]])
    def reliability_count_to_penalty(r):
        r = np.asarray(r)
        reliability_levels = np.array([0.50, 0.80, 0.90, 0.95, 0.99])
        penalty = np.array([1.0, 0.25, 0.25**2, 0.25**3, 0.25**4])
        idx = np.abs(r[..., None] - reliability_levels).argmin(axis=-1)
        return penalty[idx]
    task_type_penalty = sq.mixture([[0.2, 1], [0.4, 1 / sq.lognorm(5, 20)], [0.4, 1 / sq.lognorm(10, 1000)]])
    task_type_penalty = sq.mixture([[0.2, 1], [0.4, 1 / sq.lognorm(5, 20)], [0.4, 1 / sq.lognorm(10, 1000)]])
    
    # Sample everything to avoid mixing distributions and arrays
    start_task_length_samples = sq.sample(current_best * elicitation_boost, n=n_samples)
    reliability_samples = sq.sample(reliability_needed, n=n_samples)
    penalty_samples = reliability_count_to_penalty(reliability_samples)
    task_type_penalty_samples = sq.sample(task_type_penalty, n=n_samples)
    
    # Combine samples
    start_task_length = start_task_length_samples * penalty_samples * task_type_penalty_samples
    
    # Apply max constraint
    start_task_length = np.maximum(30/60/60, start_task_length)
    return start_task_length

def get_agi_task_length():
    # Mixture of fast (~167h) and slow (~400h) AGI task length scenarios
    fast = sq.lognorm(lognorm_mean=167, lognorm_sd=400, credibility=80, lclip=40)
    slow = sq.lognorm(lognorm_mean=400, lognorm_sd=1000, credibility=80, lclip=40)
    return sq.mixture([[0.7, fast], [0.3, slow]])

def get_doubling_time():
    return sq.mixture([[0.4, 212], [0.2, 118], [0.1, 320], [0.3, sq.lognorm(lognorm_mean=126, lognorm_sd=40)]])

def get_acceleration():
    # Mixture: superexponential (<1), neutral (1), slowdown (>1) scenarios
    superexp = 1 - sq.lognorm(lognorm_mean=0.005, lognorm_sd=0.1, credibility=80)
    slow = 1 + sq.lognorm(lognorm_mean=0.005, lognorm_sd=0.1, credibility=80)
    return sq.mixture([[0.3, superexp], [0.4, 1], [0.3, slow]])

def get_shift():
    # Mixture: typical small (~30d) and larger (~120d) shift scenarios
    small = sq.norm(mean=30, sd=10, lclip=0)
    large = sq.norm(mean=120, sd=30, lclip=0)
    return sq.mixture([[0.6, small], [0.4, large]])

def adapted_metr_model(start_task_length, agi_task_length, doubling_time, acceleration, shift):
    # Calculate the scaling factor more carefully
    # Use standard ** operator for exponentiation with distributions
    start_task_length = sq.const(start_task_length) if isinstance(start_task_length, (int, float)) else start_task_length
    agi_task_length = sq.const(agi_task_length) if isinstance(agi_task_length, (int, float)) else agi_task_length
    doubling_time = sq.const(doubling_time) if isinstance(doubling_time, (int, float)) else doubling_time
    acceleration = sq.const(acceleration) if isinstance(acceleration, (int, float)) else acceleration
    shift = sq.const(shift) if isinstance(shift, (int, float)) else shift
    
    # Use ** for exponentiation
    scaling_factor = 2 ** (shift / doubling_time)
    
    start_task_length_adjusted = start_task_length * scaling_factor
    days = calculate_doubling_time(start_task_length_adjusted, agi_task_length, doubling_time, acceleration)
    measurement_error_variance = sq.invlognorm(0.8, 1.5)
    return days * measurement_error_variance

def _sample_helper(f, n):
    return sq.sample(f(n), n=n) if callable(f) else sq.sample(f, n=n)

def run_model(
    n_samples=100_000,
    start_task_length=None,
    agi_task_length=None,
    doubling_time=None,
    acceleration=None,
    shift=None,
    index_date=O3_LAUNCH_DATE,
    correlated=False,
    use_parallel=False,
    seed=None,
    progress_callback=None,  # Add progress callback for UI updates
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the AGI timeline model with enhanced error handling and progress reporting.
    
    Args:
        n_samples (int): Number of Monte Carlo samples
        start_task_length: Distribution or value for start task length
        agi_task_length: Distribution or value for AGI task length
        doubling_time: Distribution or value for capability doubling time
        acceleration: Distribution or value for acceleration factor
        shift: Distribution or value for capability shift in days
        index_date (datetime): Reference date for predictions
        correlated (bool): Whether to correlate parameters
        use_parallel (bool): Use parallel processing for large samples
        seed (int): Random seed for reproducibility
        progress_callback (callable): Optional callback for progress updates (0-100)
        
    Returns:
        tuple: (days_until_agi, agi_dates)
    """
    # Input validation and normalization
    n_samples = int(np.clip(n_samples, 1_000, 1_000_000))
    if progress_callback:
        progress_callback(5)  # Initial progress

    try:
        # Set random state for reproducibility
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
            
        # Sample parameters with progress updates
        if progress_callback:
            progress_callback(10)

        # Parameter sampling with better defaults
        raw_params = {
            'start_task_length': get_start_task_length(n_samples) if start_task_length is None else start_task_length,
            'agi_task_length': get_agi_task_length() if agi_task_length is None else agi_task_length,
            'doubling_time': get_doubling_time() if doubling_time is None else doubling_time,
            'acceleration': get_acceleration() if acceleration is None else acceleration,
            'shift': get_shift() if shift is None else shift,
        }

        params = {}
        # Sample any distributions and convert to numpy arrays
        for k, v in raw_params.items():
            print(f"DEBUG: Processing parameter {k}...", flush=True)
            try:
                # Try to sample if it's a distribution or something sampleable
                if isinstance(v, np.ndarray):
                     val = v
                elif hasattr(v, 'sample'): 
                     # Use .sample() method directly if available
                     # Ensure n is a python int
                     n_int = int(n_samples)
                     try:
                         val = v.sample(n=n_int)
                     except Exception as e_sample:
                         print(f"DEBUG: v.sample() failed for {k}: {e_sample}")
                         # Try sq.sample as fallback
                         val = sq.sample(v, n=n_int)
                elif hasattr(v, 'type'):
                     # It's a distribution but maybe no sample method? Use sq.sample
                     n_int = int(n_samples)
                     val = sq.sample(v, n=n_int)
                else:
                     val = np.full(n_samples, v)
                
                # Ensure it's a flat float array
                val = np.asarray(val, dtype=float)
                if val.shape != (n_samples,):
                    val = np.resize(val, n_samples)
                params[k] = val
            except Exception as e:
                import traceback
                print(f"DEBUG: CRITICAL FAILURE sampling {k}")
                print(f"DEBUG: Value type: {type(v)}")
                print(f"DEBUG: Exception: {e}")
                traceback.print_exc()
                
                # If sampling fails, we can't proceed with this parameter as a distribution
                # But if it's a scalar wrapped in something weird, try float conversion
                try:
                    scalar_val = float(v)
                    params[k] = np.full(n_samples, scalar_val)
                except:
                    raise RuntimeError(f"Failed to sample parameter '{k}' ({type(v)}): {e}")
                
        if progress_callback:
            progress_callback(30)
            
        # Apply correlations if needed
        if correlated:
            u = np.random.uniform(0, 1, n_samples)
            params['doubling_time'] = 60 + 340 * u  # 60-400 day range
            params['acceleration'] = 1 - 0.2 * (1 - u)  # Correlated acceleration
            
        if progress_callback:
            progress_callback(50)
            
        # Calculate days to AGI with progress updates
        days_to_agi = calculate_doubling_time(
            start_task_length=params['start_task_length'],
            agi_task_length=params['agi_task_length'],
            doubling_time=params['doubling_time'],
            acceleration=params['acceleration']
        )
        
        # Apply shift and measurement error
        scaling_factor = 2 ** (params['shift'] / params['doubling_time'])
        days_to_agi = days_to_agi * scaling_factor
        
        # Add lognormal measurement error (10% std dev)
        measurement_error = np.random.lognormal(0, 0.1, n_samples)
        days_to_agi = days_to_agi * measurement_error
        
        # Clip to reasonable range
        days_to_agi = np.clip(days_to_agi, 0, 365*100)  # Cap at 100 years
        
        if progress_callback:
            progress_callback(90)
            
        # Convert to dates
        agi_dates = samples_to_date(days_to_agi, index_date=index_date)
        
        if progress_callback:
            progress_callback(100)
            
        return days_to_agi, agi_dates
        
    except Exception as e:
        error_msg = f"Error in model execution: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    """
    Run the AGI timeline model.
    If correlated=True, samples doubling_time and acceleration with negative correlation (lower doubling_time -> lower acceleration).
    If use_parallel=True and n_samples > 20,000, parallelize the sampling step for speed.
    """
    n_samples = min(max(1000, n_samples), 200_000)
    # Set random seed for reproducibility if provided
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
    try:
        if use_parallel and n_samples > 20000:
            with mp.Pool(4) as pool:
                results = pool.starmap(
                    _sample_helper,
                    [
                        (get_start_task_length if start_task_length is None else start_task_length, n_samples),
                        (get_agi_task_length if agi_task_length is None else agi_task_length, n_samples),
                        (get_doubling_time if doubling_time is None else doubling_time, n_samples),
                        (get_acceleration if acceleration is None else acceleration, n_samples),
                        (get_shift if shift is None else shift, n_samples),
                    ]
                )
            start_task_length, agi_task_length, doubling_time, acceleration, shift = results
            if correlated:
                u = np.random.uniform(0, 1, n_samples)
                doubling_time = 60 + 340 * u
                acceleration = 1 - 0.2 * (1 - u)
        else:
            start_task_length = sq.sample(get_start_task_length(n_samples) if start_task_length is None else start_task_length, n=n_samples)
            agi_task_length = sq.sample(get_agi_task_length() if agi_task_length is None else agi_task_length, n=n_samples)
            if correlated:
                u = np.random.uniform(0, 1, n_samples)
                doubling_time = 60 + 340 * u
                acceleration = 1 - 0.2 * (1 - u)
            else:
                doubling_time = sq.sample(get_doubling_time() if doubling_time is None else doubling_time, n=n_samples)
                acceleration = sq.sample(get_acceleration() if acceleration is None else acceleration, n=n_samples)
            shift = sq.sample(get_shift() if shift is None else shift, n=n_samples)

        # Convert to numpy arrays
        start_task_length = np.asarray(start_task_length)
        agi_task_length = np.asarray(agi_task_length)
        doubling_time = np.asarray(doubling_time)
        acceleration = np.asarray(acceleration)
        shift = np.asarray(shift)

        # Safety checks to avoid NaNs
        # Ensure all values are positive where needed
        start_task_length = np.maximum(start_task_length, 1e-10)  # Avoid log(0)
        agi_task_length = np.maximum(agi_task_length, 1e-9)  # Avoid log(0)
        doubling_time = np.maximum(doubling_time, 0.1)  # Avoid division by 0
        
        # Handle special cases in acceleration
        # Acceleration exactly 1.0 needs special handling
        accel_special = np.isclose(acceleration, 1.0, rtol=1e-10, atol=1e-14)
        # Clip acceleration to valid range (avoid NaNs from powers)
        acceleration = np.clip(acceleration, 0.5, 1.5)

        # Now do all calculations in numpy
        scaling_factor = 2 ** (shift / doubling_time)
        start_task_length_adjusted = start_task_length * scaling_factor

        # Calculate doublings needed - avoid log(negative)
        ratio = np.maximum(agi_task_length / start_task_length_adjusted, 1e-10)
        doublings_needed = np.log(ratio) / np.log(2)

        # Calculate days to AGI
        days = np.zeros_like(doublings_needed)
        
        # Handle acceleration = 1 case 
        days[accel_special] = doublings_needed[accel_special] * doubling_time[accel_special]
        
        # Handle acceleration != 1 case safely
        non_special = ~accel_special
        if np.any(non_special):
            # Safe calculation avoiding division by zero and invalid powers
            accel_non_special = acceleration[non_special]
            doublings_non_special = doublings_needed[non_special]
            dt_non_special = doubling_time[non_special]
            
            # Calculate power term safely (avoid negative bases with fractional exponents)
            # Use exp(log(a) * b) instead of a ** b for negative a
            power_term = np.zeros_like(doublings_non_special)
            
            # Valid powers (acceleration > 0, real doublings)
            valid_power = (accel_non_special > 0) & np.isfinite(doublings_non_special)
            power_term[valid_power] = accel_non_special[valid_power] ** doublings_non_special[valid_power]
            
            # Safe division
            denominator = 1 - accel_non_special
            # Avoid division by zero
            safe_denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
            
            days[non_special] = dt_non_special * (1 - power_term) / safe_denom

        # Add measurement error
        measurement_error_variance = sq.sample(sq.invlognorm(0.8, 1.5), n=n_samples)
        days = days * measurement_error_variance

        # Clean up any remaining NaNs or infinities
        days = np.nan_to_num(days, nan=365*5, posinf=365*50, neginf=0)
        
        # Filter out extreme values
        days = np.clip(days, 0, 365*100)

        # Convert to dates
        samples_dates = samples_to_date(days, index_date=index_date)
        return days, samples_dates
        
    except Exception as e:
        # If sampling fails, provide a helpful error message
        error_msg = f"Error in model calculations: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e 

DEFAULT_PARAMS = {
    "start_task_length": 1.75,
    "agi_task_length": 167.0,
    "doubling_time": 212.0,
    "acceleration": 1.0,
    "shift": 90,
    "correlated": False,
    "use_parallel": False,
    "elicitation_boost": 1.0,
    "reliability_needed": 0.5,
    "task_type_penalty": 1.0,
    "reference_date": O3_LAUNCH_DATE,
} 

# Parameter presets with descriptions
PARAMETER_PRESETS = {
    "Default": {
        "params": DEFAULT_PARAMS,
        "description": "Balanced parameters based on current AI trends from METR study. Uses 1.75 hours start task length (o3's level), 167 hours AGI task length (month of work), 212 days doubling time, and standard growth assumptions."
    },
    "Conservative": {
        "params": {
            "start_task_length": 0.8,  # Lower capability assessment
            "agi_task_length": 2000,   # Full work year
            "doubling_time": 320,      # Pessimistic trend
            "acceleration": 1.1,       # Progress slows down
            "shift": 30,               # Conservative shift
            "elicitation_boost": 0.8,  # Less scaffolding improvement
            "reliability_needed": 0.8, # Higher reliability needed
            "task_type_penalty": 4.0,  # AGI tasks much harder than METR tasks
            "reference_date": O3_LAUNCH_DATE,
            "correlated": False,
            "use_parallel": True,
        },
        "description": "Pessimistic timeline with more challenging AGI requirements. Uses lower current capability assessment, full work year AGI task (2000 hours), slower progress rate (320 days doubling), and assumption that AGI tasks are much harder than benchmarks."
    },
    "Aggressive": {
        "params": {
            "start_task_length": 2.5,  # Higher capability assessment
            "agi_task_length": 80,     # Just 2 work weeks
            "doubling_time": 118,      # 2024-2025 trend (faster)
            "acceleration": 0.9,       # Superexponential progress
            "shift": 120,              # Significant shift
            "elicitation_boost": 1.5,  # Better scaffolding
            "reliability_needed": 0.5, # METR standard
            "task_type_penalty": 0.5,  # AGI tasks easier than METR tasks
            "reference_date": O3_LAUNCH_DATE, 
            "correlated": True,        # Correlated sampling
            "use_parallel": True,
        },
        "description": "Optimistic timeline with faster progress. Uses higher capability assessment, 80-hour AGI threshold (2 weeks of work), fast doubling time (118 days), and superexponential progress assumptions, along with significant lead time from private models."
    },
    "Middle Ground": {
        "params": {
            "start_task_length": 1.75, # Default o3 value
            "agi_task_length": 400,    # Few months of work
            "doubling_time": 160,      # Between default and aggressive
            "acceleration": 1.0,       # Exponential progress
            "shift": 60,               # Moderate shift
            "elicitation_boost": 1.0,  # No adjustment
            "reliability_needed": 0.7, # Moderate reliability
            "task_type_penalty": 1.5,  # AGI tasks somewhat harder
            "reference_date": O3_LAUNCH_DATE,
            "correlated": False,
            "use_parallel": True,
        },
        "description": "Moderate forecasting assumptions between Default and Aggressive. Uses 400-hour AGI task length (10 weeks), 160-day doubling time, and moderate reliability requirements."
    }
} 

# Model task time dictionary (in hours)
# Add new utility function for quick estimation
def estimate_agi_date(
    start_task_length=1.75,
    agi_task_length=167,
    doubling_time=212,
    acceleration=1.0,
    shift=90,
    index_date=O3_LAUNCH_DATE,
):
    """
    Quick estimation of AGI date without full Monte Carlo simulation.
    Returns the median estimate and 80% confidence interval.
    """
    # Calculate median estimate
    scaling_factor = 2 ** (shift / doubling_time)
    start_task_length_adj = start_task_length * scaling_factor
    
    if acceleration == 1:
        days = (np.log(agi_task_length / start_task_length_adj) / np.log(2)) * doubling_time
    else:
        power_term = acceleration ** (np.log(agi_task_length / start_task_length_adj) / np.log(2))
        days = doubling_time * (1 - power_term) / (1 - acceleration)
    
    agi_date = index_date + timedelta(days=days)
    
    # Simple uncertainty estimate (could be enhanced)
    lower_bound = agi_date - timedelta(days=days*0.3)  # 30% earlier
    upper_bound = agi_date + timedelta(days=days*0.5)  # 50% later
    
    return {
        'median_date': agi_date,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'days_estimate': days
    }

# METR Benchmark Data Fetching
_METR_BENCHMARK_URL = "https://metr.org/assets/benchmark_results.yaml"
_metr_cache = None

def fetch_metr_benchmark_data():
    """
    Fetch and parse benchmark results from METR's public YAML file.
    Returns a dict mapping model names to their p50_horizon_length (in hours).
    Data is cached after first fetch.
    """
    global _metr_cache
    if _metr_cache is not None:
        return _metr_cache
    
    try:
        import urllib.request
        import yaml
        
        with urllib.request.urlopen(_METR_BENCHMARK_URL, timeout=10) as response:
            yaml_content = response.read().decode('utf-8')
        
        data = yaml.safe_load(yaml_content)
        results = data.get('results', {})
        
        model_task_times = {}
        for model_key, model_data in results.items():
            metrics = model_data.get('metrics', {})
            p50_horizon = metrics.get('p50_horizon_length', {})
            estimate = p50_horizon.get('estimate')
            release_date = model_data.get('release_date')
            
            if estimate is not None:
                # Create display-friendly name
                display_name = _format_model_name(model_key)
                model_task_times[display_name] = {
                    'p50_horizon_length': float(estimate),
                    'release_date': release_date,
                    'raw_key': model_key
                }
        
        _metr_cache = model_task_times
        return model_task_times
        
    except Exception as e:
        print(f"Warning: Failed to fetch METR benchmark data: {e}")
        # Return fallback data if fetch fails
        return _get_fallback_task_times()

def _format_model_name(model_key):
    """Convert YAML model keys to display-friendly names."""
    name_mapping = {
        'o3': 'o3',
        'o4-mini': 'o4-mini',
        'o1_preview': 'o1 Preview',
        'o1_elicited': 'o1',
        'claude_3_7_sonnet': 'Claude 3.7 Sonnet',
        'claude_3_5_sonnet': 'Claude 3.5 Sonnet (old)',
        'claude_3_5_sonnet_20241022': 'Claude 3.5 Sonnet (new)',
        'claude_3_opus': 'Claude 3 Opus',
        'claude_4_opus': 'Claude 4 Opus',
        'claude_4_1_opus': 'Claude 4.1 Opus',
        'claude_4_sonnet': 'Claude 4 Sonnet',
        'claude_sonnet_4_5': 'Claude Sonnet 4.5',
        'claude_opus_4_5': 'Claude Opus 4.5',
        'gpt_4o': 'GPT-4o',
        'gpt_4': 'GPT-4',
        'gpt_4_turbo': 'GPT-4 Turbo',
        'gpt_4_0125': 'GPT-4 0125',
        'gpt_4_1106': 'GPT-4 1106',
        'gpt_3_5_turbo_instruct': 'GPT-3.5 Turbo',
        'gpt_5': 'GPT-5',
        'gpt_5_1_codex_max': 'GPT-5.1 Codex Max',
        'gpt2': 'GPT-2',
        'deepseek_r1': 'DeepSeek R1',
        'deepseek_r1_0528': 'DeepSeek R1 0528',
        'deepseek_v3': 'DeepSeek V3',
        'deepseek_v3_0324': 'DeepSeek V3 0324',
        'gemini_2_5_pro_preview': 'Gemini 2.5 Pro Preview',
        'grok_4': 'Grok 4',
        'kimi_k2_thinking': 'Kimi K2 Thinking',
        'qwen_2_5_72b': 'Qwen 2.5 72B',
        'qwen_2_72b': 'Qwen 2 72B',
        'davinci_002': 'Davinci 002',
        'gpt-oss-120b': 'GPT-OSS 120B',
    }
    return name_mapping.get(model_key, model_key.replace('_', ' ').title())

def _get_fallback_task_times():
    """Fallback data if METR fetch fails."""
    return {
        'o3': {'p50_horizon_length': 94.0, 'release_date': '2025-04-16'},
        'o4-mini': {'p50_horizon_length': 78.6, 'release_date': '2025-04-16'},
        'Claude 3.7 Sonnet': {'p50_horizon_length': 56.1, 'release_date': '2025-02-24'},
        'o1': {'p50_horizon_length': 41.1, 'release_date': '2024-12-05'},
        'Claude 3.5 Sonnet (new)': {'p50_horizon_length': 29.6, 'release_date': '2024-10-22'},
        'DeepSeek R1': {'p50_horizon_length': 26.9, 'release_date': '2025-01-20'},
        'o1 Preview': {'p50_horizon_length': 22.0, 'release_date': '2024-09-12'},
        'Claude 3.5 Sonnet (old)': {'p50_horizon_length': 18.7, 'release_date': '2024-06-20'},
        'GPT-4o': {'p50_horizon_length': 9.2, 'release_date': '2024-05-13'},
        'Claude 3 Opus': {'p50_horizon_length': 6.4, 'release_date': '2024-03-04'},
        'GPT-3.5 Turbo': {'p50_horizon_length': 0.6, 'release_date': '2022-03-15'},
    }

def get_model_task_times():
    """
    Get model task times (p50_horizon_length in hours) from METR benchmark data.
    Returns dict: {model_name: p50_horizon_length_hours}
    """
    data = fetch_metr_benchmark_data()
    return {name: info['p50_horizon_length'] for name, info in data.items()}

def get_model_release_dates():
    """
    Get model release dates from METR benchmark data.
    Returns dict: {model_name: release_date_string}
    """
    data = fetch_metr_benchmark_data()
    return {name: info['release_date'] for name, info in data.items()}

# Legacy compatibility: MODEL_TASK_TIMES as a simple dict
# This will be populated on first access
MODEL_TASK_TIMES = get_model_task_times() 