import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import run_model, get_start_task_length, get_agi_task_length, get_doubling_time, get_acceleration, get_shift, O3_LAUNCH_DATE, CLAUDE_3P7_LAUNCH_DATE, DEFAULT_PARAMS, PARAMETER_PRESETS, MODEL_TASK_TIMES
import squigglepy as sq
from datetime import datetime, timedelta
import io
import pandas as pd
import plots  # Import the new plots module

st.set_page_config(page_title="AGI Timelines Model", layout="wide")

st.title("AGI Timelines Model (METR)")
st.markdown("""
Interactively forecast AGI timelines using the METR model based on ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499).

Adjust the parameters and run the model to see updated timelines.
""")

# Sidebar for variable selection
st.sidebar.header("Model Parameters")

with st.sidebar.expander("ℹ️ Parameter Explanations (click to expand)"):
    st.markdown("""
    **Start task length (hours):**
    The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for improvements from scaffolding and compute (elicitation boost), higher reliability requirements (reliability penalty), and the possibility that AGI tasks are harder than METR's self-contained software tasks (task type penalty). This is not just the raw model benchmark, but a composite reflecting how close we are to AGI-level tasks.
    
    **AGI task length (hours):**
    The human time required to complete a task that would count as 'AGI-level' (e.g., a month- or year-long project). This is a subjective threshold, often set to something like 167 hours (a month of full-time work) or up to 2000 hours (a work year).
    
    **Doubling time (days):**
    The number of days it takes for the maximum task length that AI can do at 50% reliability to double. This is fit to historical trends (e.g., 212 days for 2019-2024, 118 days for 2024-2025, or a mixture).
    
    **Acceleration:**
    If <1, the doubling time itself shrinks over time (superexponential progress); if >1, it grows (progress slows); if 1, progress is exponential.
    
    **Shift (days):**
    The number of days to shift the forecast earlier to account for internal model capabilities before public release (e.g., 30-150 days).
    
    **Elicitation boost:**
    Factor for improvements from better scaffolding or more compute. >1 means faster progress.
    
    **Reliability needed:**
    The required reliability for AGI-level performance. Higher reliability means a harder bar for AGI and increases the penalty on start task length.
    
    **Task type complexity factor:**
    Adjusts for AGI tasks being harder than the METR benchmark tasks. >1 = AGI tasks are harder.
    
    **Reference date:**
    The date from which to start counting days until AGI. Usually the launch date of the reference model.
    """)

# Parameter presets
st.sidebar.subheader("Parameter Presets")

# Add a callback to set session state when selection changes
if 'current_preset' not in st.session_state:
    st.session_state.current_preset = "Default"

# Initialize selected_preset from session state if it exists
if 'selected_preset' not in st.session_state:
    st.session_state.selected_preset = "Default"

def on_preset_change():
    # Update the current preset for tooltip
    st.session_state.current_preset = st.session_state.preset_selectbox
    # Store the selected preset
    st.session_state.selected_preset = st.session_state.preset_selectbox

# Use the stored selection as the default index
default_index = list(PARAMETER_PRESETS.keys()).index(st.session_state.selected_preset) if st.session_state.selected_preset in PARAMETER_PRESETS else 0

# Get tooltips from PARAMETER_PRESETS
preset_tooltips = {preset_name: preset_data["description"] for preset_name, preset_data in PARAMETER_PRESETS.items()}

selected_preset = st.sidebar.selectbox(
    "Choose a parameter preset",
    list(PARAMETER_PRESETS.keys()),
    index=default_index,  # Use the stored selection
    help=preset_tooltips.get(st.session_state.current_preset, ""),  # Get help text from session state
    key="preset_selectbox",
    on_change=on_preset_change
)

# Apply preset button
apply_preset = st.sidebar.button("Apply Preset")

# Save preset values in session state so they persist
if "preset_applied" not in st.session_state:
    st.session_state.preset_applied = False
    
if apply_preset or (not st.session_state.preset_applied and selected_preset != "Default"):
    # Apply the selected preset
    st.session_state.preset_applied = True
    st.session_state.preset_params = PARAMETER_PRESETS[selected_preset]["params"]
    # Force a model run by clearing the cached results
    if "results" in st.session_state:
        del st.session_state["results"]
    st.sidebar.success(f"Applied {selected_preset} preset")
    # Force a rerun to apply the preset values
    st.rerun()

# Fixed number of samples (always use 100,000)
n_samples = 100_000

# Model section
st.sidebar.subheader("Model Configuration")

# Advanced mode toggle
advanced_mode = st.sidebar.checkbox("Advanced Mode (More Parameters)")

# Reference model selection
if advanced_mode:
    # Create a copy of MODEL_TASK_TIMES and add "Custom" option
    reference_models = dict(MODEL_TASK_TIMES)
    reference_models["Custom"] = DEFAULT_PARAMS["start_task_length"]
    
    # Sort models by their task length (descending) for a more logical order
    sorted_models = ["Custom"] + sorted(MODEL_TASK_TIMES.keys(), key=lambda x: -MODEL_TASK_TIMES[x])
    
    reference_model = st.sidebar.selectbox(
        "Reference model for start task length",
        sorted_models,
        format_func=lambda x: f"{x} ({reference_models[x]:.3f} hrs)" if x != "Custom" else "Custom"
    )

    # Set task length based on model selection
    if reference_model != "Custom":
        start_task_length = MODEL_TASK_TIMES[reference_model]
    else:
        # Use preset parameter if one has been applied
        default_value = st.session_state.preset_params["start_task_length"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["start_task_length"]
        start_task_length = st.sidebar.number_input(
            "Start task length (hours)", 
            min_value=float(0.01), 
            max_value=float(10.0), 
            value=float(default_value), 
            step=float(0.01),
            help="The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for scaffolding, reliability, and task type penalties. Default: 1.75 hours (o3's task length)"
        )
else:
    # Use preset parameter if one has been applied
    default_value = st.session_state.preset_params["start_task_length"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["start_task_length"]
    start_task_length = st.sidebar.number_input(
        "Start task length (hours)", 
        min_value=float(0.01), 
        max_value=float(10.0), 
        value=float(default_value), 
        step=float(0.01),
        help="The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for scaffolding, reliability, and task type penalties. Default: 1.75 hours (o3's task length)"
    )

# AGI task length
# Use preset parameter if one has been applied
agi_default = st.session_state.preset_params["agi_task_length"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["agi_task_length"]
agi_task_length = st.sidebar.number_input(
    "AGI task length (hours)", 
    min_value=float(40.0), 
    max_value=float(5000.0), 
    value=float(agi_default), 
    step=float(1.0),
    help="The human time required to complete a task that would count as 'AGI-level' (e.g., a month- or year-long project). Default: 167 hours (month-long tasks)"
)

# Growth parameters
st.sidebar.subheader("Growth Parameters")

doubling_default = st.session_state.preset_params["doubling_time"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["doubling_time"]
doubling_time = st.sidebar.number_input(
    "Doubling time (days)", 
    min_value=float(60.0), 
    max_value=float(400.0), 
    value=float(doubling_default), 
    step=float(1.0),
    help="The number of days it takes for the maximum task length that AI can do at 50% reliability to double. Default: 212 days (METR 2019-2024 trend)"
)

if advanced_mode:
    # Add doubling time model selection with presets
    doubling_time_model = st.sidebar.selectbox(
        "Doubling time model",
        ["Custom", "METR 2019-2024 Trend (212 days)", "METR 2024-2025 Trend (118 days)", 
         "Pessimistic Trend (320 days)", "Mixture Model"]
    )
    
    if doubling_time_model == "METR 2019-2024 Trend (212 days)":
        doubling_time = DEFAULT_PARAMS["doubling_time"]
    elif doubling_time_model == "METR 2024-2025 Trend (118 days)":
        doubling_time = 118.0
    elif doubling_time_model == "Pessimistic Trend (320 days)":
        doubling_time = 320.0
    elif doubling_time_model == "Mixture Model":
        st.sidebar.markdown("Using default METR mixture model for doubling time")
        doubling_time = -1  # Special value to use the mixture model

accel_default = st.session_state.preset_params["acceleration"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["acceleration"]
acceleration = st.sidebar.slider(
    "Acceleration", 
    min_value=float(0.8), 
    max_value=float(1.2), 
    value=float(accel_default), 
    step=float(0.01),
    help="If <1, the doubling time itself shrinks over time (superexponential progress); if >1, it grows (progress slows); if 1, progress is exponential."
)

shift_default = st.session_state.preset_params["shift"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["shift"]
shift = st.sidebar.number_input(
    "Shift (days)", 
    min_value=float(0), 
    max_value=float(250), 
    value=float(shift_default), 
    step=float(1),
    help="The number of days to shift the forecast earlier to account for internal model capabilities before public release. Default: 90 days"
)

# Advanced parameters in expanded section
if advanced_mode:
    st.sidebar.subheader("Advanced Adjustments")
    
    with st.sidebar.expander("Additional Parameters"):
        # Elicitation boost factor
        elicit_default = st.session_state.preset_params["elicitation_boost"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["elicitation_boost"]
        elicitation_boost = st.slider(
            "Elicitation boost", 
            min_value=float(0.5), 
            max_value=float(2.0), 
            value=float(elicit_default), 
            step=float(0.1),
            help="Boost factor from better scaffolding and increased compute. Default: 1.0 (no adjustment)"
        )
        
        # Reliability settings
        reliability_default = st.session_state.preset_params["reliability_needed"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["reliability_needed"]
        reliability_needed = st.slider(
            "Reliability needed", 
            min_value=float(0.5), 
            max_value=float(0.99), 
            value=float(reliability_default), 
            step=float(0.01),
            help="Required reliability level. Default: 0.5 (METR's standard)"
        )
        
        # Task type penalty factor
        penalty_default = st.session_state.preset_params["task_type_penalty"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["task_type_penalty"]
        task_type_penalty = st.slider(
            "Task type complexity factor", 
            min_value=float(0.1), 
            max_value=float(10.0), 
            value=float(penalty_default), 
            step=float(0.1),
            help="Adjustment for AGI tasks being harder than METR's self-contained software tasks. >1 = harder, <1 = easier. Default: 1.0 (no adjustment)"
        )
        
        # Reference date selection
        # Get default reference date from preset if available
        if hasattr(st.session_state, "preset_params") and "reference_date" in st.session_state.preset_params:
            ref_date_default = "o3 Launch (2025-04-16)" if st.session_state.preset_params["reference_date"] == O3_LAUNCH_DATE else "Claude 3.7 Launch (2025-02-24)"
        else:
            ref_date_default = "o3 Launch (2025-04-16)"
            
        reference_date_option = st.selectbox(
            "Reference date",
            ["o3 Launch (2025-04-16)", "Claude 3.7 Launch (2025-02-24)", "Custom Date"],
            index=0 if ref_date_default == "o3 Launch (2025-04-16)" else 1
        )
        
        if reference_date_option == "o3 Launch (2025-04-16)":
            reference_date = DEFAULT_PARAMS["reference_date"]
        elif reference_date_option == "Claude 3.7 Launch (2025-02-24)":
            reference_date = CLAUDE_3P7_LAUNCH_DATE
        else:
            reference_date = st.date_input(
                "Custom reference date",
                DEFAULT_PARAMS["reference_date"].date(),
                help="The date from which to start counting days until AGI"
            )
            # Convert to datetime
            reference_date = datetime.combine(reference_date, datetime.min.time())
    
    # Only show the additional parameter adjustments if advanced mode is on
    use_adjusted_start_task = False
    if elicitation_boost != DEFAULT_PARAMS["elicitation_boost"] or reliability_needed != DEFAULT_PARAMS["reliability_needed"] or task_type_penalty != DEFAULT_PARAMS["task_type_penalty"]:
        use_adjusted_start_task = True
        # Calculate adjusted start task length
        reliability_penalty = 1.0
        if reliability_needed > 0.5:
            # Simple approximation of the reliability penalty function
            if reliability_needed <= 0.8:
                reliability_penalty = 0.25
            elif reliability_needed <= 0.9:
                reliability_penalty = 0.25**2
            elif reliability_needed <= 0.95:
                reliability_penalty = 0.25**3
            else:
                reliability_penalty = 0.25**4
        
        adjusted_start_task_length = start_task_length * elicitation_boost * reliability_penalty * task_type_penalty
        st.sidebar.markdown(f"Adjusted start task length: **{adjusted_start_task_length:.3f} hours**")
        start_task_length = adjusted_start_task_length
else:
    # Default reference date when not in advanced mode
    reference_date = DEFAULT_PARAMS["reference_date"]
    
    # Set these variables to their defaults or preset values for use in other parts of the code
    if hasattr(st.session_state, "preset_params"):
        elicitation_boost = st.session_state.preset_params.get("elicitation_boost", DEFAULT_PARAMS["elicitation_boost"])
        reliability_needed = st.session_state.preset_params.get("reliability_needed", DEFAULT_PARAMS["reliability_needed"])
        task_type_penalty = st.session_state.preset_params.get("task_type_penalty", DEFAULT_PARAMS["task_type_penalty"])
    else:
        elicitation_boost = DEFAULT_PARAMS["elicitation_boost"]
        reliability_needed = DEFAULT_PARAMS["reliability_needed"]
        task_type_penalty = DEFAULT_PARAMS["task_type_penalty"]

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Model")

# Prepare variables for model
kwargs = {}
if start_task_length > 0:
    kwargs["start_task_length"] = sq.const(start_task_length)
if agi_task_length > 0:
    kwargs["agi_task_length"] = sq.const(agi_task_length)
if doubling_time > 0:
    kwargs["doubling_time"] = sq.const(doubling_time)
else:
    # If doubling_time is -1, use the mixture model
    kwargs["doubling_time"] = None  # Will use the default mixture model in model.py
if acceleration != DEFAULT_PARAMS["acceleration"]:
    kwargs["acceleration"] = sq.const(acceleration)
if shift > 0:
    kwargs["shift"] = sq.const(shift)
# Add reference date to kwargs
kwargs["index_date"] = reference_date

if advanced_mode:
    # Get correlated value from preset if available
    correlated_default = st.session_state.preset_params.get("correlated", DEFAULT_PARAMS["correlated"]) if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["correlated"]
    correlated = st.sidebar.checkbox(
        "Correlate doubling time and acceleration (advanced)",
        value=correlated_default,
        help="If checked, samples doubling time and acceleration with negative correlation (lower doubling time → lower acceleration, i.e., faster progress)."
    )
    
    # Get parallel value from preset if available
    parallel_default = st.session_state.preset_params.get("use_parallel", DEFAULT_PARAMS["use_parallel"]) if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["use_parallel"]
    use_parallel = st.sidebar.checkbox(
        "Parallelize sampling (advanced, for large n_samples)",
        value=parallel_default,
        help="If checked, uses multiprocessing to parallelize sampling for large sample sizes. May speed up model runs on large datasets."
    )
else:
    # Set from preset if available, otherwise use defaults
    if hasattr(st.session_state, "preset_params"):
        correlated = st.session_state.preset_params.get("correlated", DEFAULT_PARAMS["correlated"])
        use_parallel = st.session_state.preset_params.get("use_parallel", DEFAULT_PARAMS["use_parallel"])
    else:
        correlated = DEFAULT_PARAMS["correlated"]
        use_parallel = DEFAULT_PARAMS["use_parallel"]

kwargs["correlated"] = correlated
kwargs["use_parallel"] = use_parallel

@st.cache_data(show_spinner=False)
def cached_run_model(n_samples, _start_task_length, _agi_task_length, _doubling_time, _acceleration, _shift, index_date, correlated=False, use_parallel=False):
    return run_model(
        n_samples=n_samples,
        start_task_length=_start_task_length,
        agi_task_length=_agi_task_length,
        doubling_time=_doubling_time,
        acceleration=_acceleration,
        shift=_shift,
        index_date=index_date,
        correlated=correlated,
        use_parallel=use_parallel,
    )

if run_button or "results" not in st.session_state:
    with st.spinner("Running model..."):
        try:
            samples, samples_dates = cached_run_model(
                n_samples=n_samples,
                _start_task_length=kwargs.get("start_task_length"),
                _agi_task_length=kwargs.get("agi_task_length"),
                _doubling_time=kwargs.get("doubling_time"),
                _acceleration=kwargs.get("acceleration"),
                _shift=kwargs.get("shift"),
                index_date=kwargs.get("index_date"),
                correlated=kwargs.get("correlated", False),
                use_parallel=kwargs.get("use_parallel", False),
            )
            st.session_state["results"] = (samples, samples_dates)
            st.success("Model run completed successfully!")
        except Exception as e:
            st.error(f"Error running model: {str(e)}")
            st.info("Try adjusting parameters or reducing the number of samples.")
            if "results" not in st.session_state:
                # If we have no previous results, stop here
                st.stop()
            else:
                # Use previous results
                samples, samples_dates = st.session_state["results"]
                st.warning("Showing previous model results.")
else:
    samples, samples_dates = st.session_state["results"]

# Clean samples
samples_clean = samples[~np.isnan(samples)]
valid_dates = [d for d in samples_dates if d is not None and not np.isnan(d.year)]
years = np.array([d.year + d.timetuple().tm_yday / 365 for d in valid_dates]) if valid_dates else []

# Calculate percentiles
if len(samples_clean) > 0:
    percentiles = sq.get_percentiles(samples_clean, digits=0)
else:
    percentiles = {}

# Tabs for results
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Plots", "Sensitivity", "Raw Data"])

with tab1:
    st.header("Summary")
    
    if len(valid_dates) > 0:
        median_idx = int(np.median(np.arange(len(valid_dates))))
        sorted_dates = np.sort(valid_dates)
        median_date = sorted_dates[median_idx]
        st.subheader("Model's Computed Median AGI Date")
        st.info(f"**{median_date.strftime('%A, %-d %B %Y')}**")
        
        # Probability table
        st.subheader("AGI Arrival Probabilities")
        years_array = np.array([d.year for d in valid_dates])
        current_year = datetime.now().year
        max_year = int(np.max(years_array))
        display_years = list(range(current_year, current_year+6)) + list(range(current_year+5, min(max_year+1, current_year+20), 5))
        
        results = []
        for year in display_years:
            prob = np.mean(years_array <= year) * 100
            results.append(f"By EOY {year}: {prob:.1f}%")
        
        col1, col2 = st.columns(2)
        half = len(results) // 2 + len(results) % 2
        with col1:
            for r in results[:half]: st.text(r)
        with col2:
            for r in results[half:]: st.text(r)
            
    else:
        st.warning("No valid results to display.")

with tab2:
    st.header("Plots")
    
    if len(valid_dates) > 0:
        # Histogram
        st.subheader("Distribution of AGI Arrival Dates")
        fig_hist = plots.plot_agi_arrival_distribution(years)
        st.pyplot(fig_hist)
        
        # Confidence Intervals
        st.subheader("Confidence Intervals")
        
        # Calculate percentile data for plotting
        p_levels = [5, 10, 25, 50, 75, 90, 95]
        p_values = []
        p_dates = []
        sorted_years = np.sort(years)
        sorted_dates_arr = np.sort(valid_dates)
        
        for p in p_levels:
            idx = int(p/100 * len(sorted_years))
            p_values.append(sorted_years[idx])
            p_dates.append(sorted_dates_arr[idx])
            
        fig_ci = plots.plot_confidence_intervals(years, p_levels, p_values, p_dates)
        st.pyplot(fig_ci)
        
        st.markdown(f"""
        **Confidence Interval Interpretation:**
        - There is a **5%** chance AGI will arrive before **{p_dates[0].strftime('%B %Y')}**
        - There is a **50%** chance AGI will arrive before **{p_dates[3].strftime('%B %Y')}** (median projection)
        - There is a **95%** chance AGI will arrive before **{p_dates[6].strftime('%B %Y')}**
        """)
        
        # Growth Plot
        st.subheader("Exponential Growth in Task Length")
        fig_growth = plots.plot_task_length_growth(start_task_length, agi_task_length, doubling_time)
        st.pyplot(fig_growth)
        
        # Timeline
        st.subheader("AI Capability Timeline")
        
        # Calculate milestones
        days_per_quarter = 91.25
        quarters = np.arange(0, 44)
        x = quarters
        n_quarters = len(quarters)
        
        milestones = [
            {"name": "Start (Current)", "hours": start_task_length, "description": "Current strongest model capabilities"},
            {"name": "Week-Long Project", "hours": 40, "description": "Models can complete full week-long software projects"},
            {"name": "Two-Week Project", "hours": 80, "description": "Models can build and deploy complex systems over two weeks"},
            {"name": "Month-Long Project", "hours": 167, "description": "Models can autonomously develop month-long projects"},
            {"name": "Quarter-Long Project", "hours": 500, "description": "Models can manage projects spanning a quarter"},
            {"name": "Half-Year Project", "hours": 1000, "description": "Models can execute multi-quarter planning and projects"},
            {"name": "Year-Long Project", "hours": 2000, "description": "AGI level: Models can execute year-long development cycles"}
        ]
        
        doubling_days = 212 if doubling_time <= 0 else doubling_time
        
        for i, milestone in enumerate(milestones):
            if i == 0:
                milestone["date"] = reference_date
                continue
                
            milestone_threshold = milestone["hours"]
            base_rate = start_task_length
            y = base_rate * np.power(2, (x * days_per_quarter) / doubling_days)
            cross_idx = np.argmax(y >= milestone_threshold) if np.any(y >= milestone_threshold) else n_quarters
            relative_days = cross_idx * days_per_quarter
            milestone_date = reference_date + timedelta(days=relative_days)
            milestone["date"] = milestone_date
            
            if milestone["hours"] >= 167:
                fast_y = base_rate * np.power(2, (x * days_per_quarter) / (doubling_days * 0.7))
                fast_cross = np.argmax(fast_y >= milestone_threshold) if np.any(fast_y >= milestone_threshold) else n_quarters
                fast_date = reference_date + timedelta(days=fast_cross * days_per_quarter)
                milestone["early_date"] = fast_date
                
                slow_y = base_rate * np.power(2, (x * days_per_quarter) / (doubling_days * 1.5))
                slow_cross = np.argmax(slow_y >= milestone_threshold) if np.any(slow_y >= milestone_threshold) else n_quarters
                slow_date = reference_date + timedelta(days=slow_cross * days_per_quarter)
                milestone["late_date"] = slow_date
        
        fig_timeline = plots.plot_timeline(milestones, reference_date)
        st.pyplot(fig_timeline)
        
    else:
        st.warning("No valid results to plot.")

with tab3:
    st.header("Sensitivity Analysis")
    
    # Sensitivity Analysis Setup
    SENS_N = 10000
    param_names = [
        ("start_task_length", "Start task length (hours)", float(0.01), float(10.0)),
        ("agi_task_length", "AGI task length (hours)", float(40.0), float(5000.0)),
        ("doubling_time", "Doubling time (days)", float(60.0), float(400.0)),
        ("acceleration", "Acceleration", float(0.8), float(1.2)),
        ("shift", "Shift (days)", float(0), float(250)),
    ]
    
    fixed_params = {
        "start_task_length": start_task_length,
        "agi_task_length": agi_task_length,
        "doubling_time": doubling_time,
        "acceleration": acceleration,
        "shift": shift,
        "index_date": reference_date,
        "correlated": correlated,
        "use_parallel": False,
    }
    
    def get_median_agi_year(params):
        try:
            days, dates = run_model(
                n_samples=SENS_N,
                start_task_length=sq.const(params["start_task_length"]),
                agi_task_length=sq.const(params["agi_task_length"]),
                doubling_time=sq.const(params["doubling_time"]),
                acceleration=sq.const(params["acceleration"]),
                shift=sq.const(params["shift"]),
                index_date=params["index_date"],
                correlated=params.get("correlated", False),
                use_parallel=False,
            )
            valid_dates = [d for d in dates if d is not None]
            if valid_dates:
                years = np.array([d.year + d.timetuple().tm_yday / 365 for d in valid_dates])
                return float(np.median(years))
            else:
                return np.nan
        except Exception:
            return np.nan

    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Calculating tornado plot..."):
            impacts = []
            labels = []
            low_yrs = []
            high_yrs = []
            
            for key, label, vmin, vmax in param_names:
                params_low = fixed_params.copy()
                params_high = fixed_params.copy()
                params_low[key] = vmin
                params_high[key] = vmax
                low_year = get_median_agi_year(params_low)
                high_year = get_median_agi_year(params_high)
                
                labels.append(label)
                low_yrs.append(low_year)
                high_yrs.append(high_year)
                impacts.append(abs(high_year - low_year))
            
            # Sort by impact
            order = np.argsort(impacts)[::-1]
            labels = np.array(labels)[order]
            low_yrs = np.array(low_yrs)[order]
            high_yrs = np.array(high_yrs)[order]
            
            fig_tornado = plots.plot_tornado(labels, low_yrs, high_yrs)
            st.pyplot(fig_tornado)
    
    # Interactive Sensitivity
    st.subheader("Interactive Sensitivity Analysis")
    
    def quick_estimate_agi_date(params):
        base_rate = params["start_task_length"]
        agi_threshold = params["agi_task_length"]
        doubling_days = params["doubling_time"]
        accel = params["acceleration"]
        shift_days = params["shift"]
        
        base_rate = base_rate * (2 ** (shift_days / doubling_days))
        ratio = agi_threshold / base_rate
        doublings_needed = np.log2(ratio)
        
        if accel == 1.0:
            days_to_agi = doublings_needed * doubling_days
        else:
            try:
                power_term = accel ** doublings_needed
                days_to_agi = doubling_days * (1 - power_term) / (1 - accel)
            except:
                days_to_agi = doublings_needed * doubling_days
        
        agi_date = params["index_date"] + timedelta(days=days_to_agi)
        return agi_date, days_to_agi

    col1, col2 = st.columns([3, 1])
    interactive_params = fixed_params.copy()
    
    with col1:
        st.markdown("**Adjust parameters**")
        param_col1, param_col2 = st.columns(2)
        for i, (key, label, vmin, vmax) in enumerate(param_names):
            with param_col1 if i % 2 == 0 else param_col2:
                step = 0.01 if key == "acceleration" else (1.0 if key == "shift" else (vmax - vmin) / 100)
                interactive_params[key] = st.slider(
                    label, float(vmin), float(vmax), float(fixed_params[key]), float(step), key=f"interactive_{key}"
                )
                
    with col2:
        st.markdown("### Estimated Date")
        est_date, days = quick_estimate_agi_date(interactive_params)
        st.markdown(f"<h2 style='color:#4F8DFD;'>{est_date.strftime('%b %Y')}</h2>", unsafe_allow_html=True)
        st.markdown(f"Days to AGI: {days:.0f}")

with tab4:
    st.header("Raw Data")
    
    if len(valid_dates) > 0:
        # Create DataFrame for download
        df = pd.DataFrame({
            "Simulation Index": range(len(valid_dates)),
            "AGI Date": valid_dates,
            "AGI Year": years
        })
        
        st.dataframe(df.head(100))
        st.markdown(f"*Showing first 100 of {len(df)} simulations*")
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Simulation Results (CSV)",
            csv,
            "agi_simulation_results.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.warning("No data to download.")

st.markdown("---")
st.markdown("""
*Model code adapted from the METR paper ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499) 
and [Forecaster Reacts to METR's bombshell](https://peterwildeford.substack.com/p/forecaster-reacts-metrs-bombshell)*
""")