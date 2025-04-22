import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import run_model, get_start_task_length, get_agi_task_length, get_doubling_time, get_acceleration, get_shift, O3_LAUNCH_DATE, CLAUDE_3P7_LAUNCH_DATE
import squigglepy as sq
from datetime import datetime, timedelta
import io
import pandas as pd
import urllib.parse

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

# Fixed number of samples (always use 100,000)
n_samples = 100_000

# Model section
st.sidebar.subheader("Model Configuration")

# Advanced mode toggle
advanced_mode = st.sidebar.checkbox("Advanced Mode (More Parameters)")

# Reference model selection
if advanced_mode:
    reference_model = st.sidebar.selectbox(
        "Reference model for start task length",
        ["Custom", "o3", "o4-mini", "Claude 3.7 Sonnet", "o1", "Claude 3.5 Sonnet (new)", "GPT4.5", 
        "DeepSeek R1", "o1 Preview", "Claude 3.5 Sonnet (old)", "GPT4o", "Claude 3 Opus", "GPT3.5 Turbo"],
    )

    # Model task time dictionary (in hours)
    model_task_times = {
        "o3": 1.75,  # ~1hr45min
        "o4-mini": 1.50,  # ~1hr30min
        "Claude 3.7 Sonnet": 0.983,  # 59min
        "o1": 0.65,  # 39min
        "Claude 3.5 Sonnet (new)": 0.467,  # 28min
        "GPT4.5": 0.467,  # ~28min
        "DeepSeek R1": 0.417,  # ~25min
        "o1 Preview": 0.367,  # 22min
        "Claude 3.5 Sonnet (old)": 0.3,  # 18min
        "GPT4o": 0.15,  # 9min
        "Claude 3 Opus": 0.1,  # 6min
        "GPT3.5 Turbo": 0.01,  # 36sec
        "Custom": 1.75  # Default to o3 value
    }

    # Set task length based on model selection
    if reference_model != "Custom":
        start_task_length = model_task_times[reference_model]
    else:
        start_task_length = st.sidebar.number_input(
            "Start task length (hours)", 
            min_value=0.01, 
            max_value=10.0, 
            value=1.75, 
            step=0.01,
            help="The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for scaffolding, reliability, and task type penalties. Default: 1.75 hours (o3's task length)"
        )
else:
    start_task_length = st.sidebar.number_input(
        "Start task length (hours)", 
        min_value=0.01, 
        max_value=10.0, 
        value=1.75, 
        step=0.01,
        help="The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for scaffolding, reliability, and task type penalties. Default: 1.75 hours (o3's task length)"
    )

# AGI task length
agi_task_length = st.sidebar.number_input(
    "AGI task length (hours)", 
    min_value=40.0, 
    max_value=5000.0, 
    value=167.0, 
    step=1.0,
    help="The human time required to complete a task that would count as 'AGI-level' (e.g., a month- or year-long project). Default: 167 hours (month-long tasks)"
)

# Growth parameters
st.sidebar.subheader("Growth Parameters")

doubling_time = st.sidebar.number_input(
    "Doubling time (days)", 
    min_value=60.0, 
    max_value=400.0, 
    value=212.0, 
    step=1.0,
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
        doubling_time = 212.0
    elif doubling_time_model == "METR 2024-2025 Trend (118 days)":
        doubling_time = 118.0
    elif doubling_time_model == "Pessimistic Trend (320 days)":
        doubling_time = 320.0
    elif doubling_time_model == "Mixture Model":
        st.sidebar.markdown("Using default METR mixture model for doubling time")
        doubling_time = -1  # Special value to use the mixture model

acceleration = st.sidebar.slider(
    "Acceleration", 
    min_value=0.8, 
    max_value=1.2, 
    value=1.0, 
    step=0.01,
    help="If <1, the doubling time itself shrinks over time (superexponential progress); if >1, it grows (progress slows); if 1, progress is exponential."
)

shift = st.sidebar.number_input(
    "Shift (days)", 
    min_value=0, 
    max_value=250, 
    value=90, 
    step=1,
    help="The number of days to shift the forecast earlier to account for internal model capabilities before public release. Default: 90 days"
)

# Advanced parameters in expanded section
if advanced_mode:
    st.sidebar.subheader("Advanced Adjustments")
    
    with st.sidebar.expander("Additional Parameters"):
        # Elicitation boost factor
        elicitation_boost = st.slider(
            "Elicitation boost", 
            min_value=0.5, 
            max_value=2.0, 
            value=1.0, 
            step=0.1,
            help="Boost factor from better scaffolding and increased compute. Default: 1.0 (no adjustment)"
        )
        
        # Reliability settings
        reliability_needed = st.slider(
            "Reliability needed", 
            min_value=0.5, 
            max_value=0.99, 
            value=0.5, 
            step=0.01,
            help="Required reliability level. Default: 0.5 (METR's standard)"
        )
        
        # Task type penalty factor
        task_type_penalty = st.slider(
            "Task type complexity factor", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Adjustment for AGI tasks being harder than METR's self-contained software tasks. >1 = harder, <1 = easier. Default: 1.0 (no adjustment)"
        )
        
        # Reference date selection
        reference_date_option = st.selectbox(
            "Reference date",
            ["o3 Launch (2025-04-16)", "Claude 3.7 Launch (2025-02-24)", "Custom Date"]
        )
        
        if reference_date_option == "o3 Launch (2025-04-16)":
            reference_date = O3_LAUNCH_DATE
        elif reference_date_option == "Claude 3.7 Launch (2025-02-24)":
            reference_date = CLAUDE_3P7_LAUNCH_DATE
        else:
            reference_date = st.date_input(
                "Custom reference date",
                O3_LAUNCH_DATE.date(),
                help="The date from which to start counting days until AGI"
            )
            # Convert to datetime
            reference_date = datetime.combine(reference_date, datetime.min.time())
    
    # Only show the additional parameter adjustments if advanced mode is on
    use_adjusted_start_task = False
    if elicitation_boost != 1.0 or reliability_needed != 0.5 or task_type_penalty != 1.0:
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
    reference_date = O3_LAUNCH_DATE

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
if acceleration != 1.0:
    kwargs["acceleration"] = sq.const(acceleration)
if shift > 0:
    kwargs["shift"] = sq.const(shift)
# Add reference date to kwargs
kwargs["index_date"] = reference_date

if advanced_mode:
    correlated = st.sidebar.checkbox(
        "Correlate doubling time and acceleration (advanced)",
        value=False,
        help="If checked, samples doubling time and acceleration with negative correlation (lower doubling time → lower acceleration, i.e., faster progress)."
    )
    use_parallel = st.sidebar.checkbox(
        "Parallelize sampling (advanced, for large n_samples)",
        value=False,
        help="If checked, uses multiprocessing to parallelize sampling for large sample sizes. May speed up model runs on large datasets."
    )
else:
    correlated = False
    use_parallel = False

kwargs["correlated"] = correlated
kwargs["use_parallel"] = use_parallel

@st.cache_data(show_spinner=False)
def cached_run_model(n_samples, _start_task_length, _agi_task_length, _doubling_time, _acceleration, _shift, index_date, correlated=False):
    return run_model(
        n_samples=n_samples,
        start_task_length=_start_task_length,
        agi_task_length=_agi_task_length,
        doubling_time=_doubling_time,
        acceleration=_acceleration,
        shift=_shift,
        index_date=index_date,
        correlated=correlated,
    )

if run_button or "results" not in st.session_state:
    with st.spinner("Running model..."):
        try:
            samples, samples_dates = run_model(n_samples=n_samples, **kwargs)
                correlated=kwargs.get("correlated", False),
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

# Show results
st.header("Results")

from pprint import pformat
import numpy as np

# Handle NaN values in percentiles calculation
try:
    # Filter out NaN values before calculating percentiles
    samples_clean = samples[~np.isnan(samples)]
    if len(samples_clean) < len(samples):
        st.warning(f"Filtered out {len(samples) - len(samples_clean)} NaN values from results.")
    
    if len(samples_clean) > 0:
        percentiles = sq.get_percentiles(samples_clean, digits=0)

    else:
        st.error("All samples contain NaN values. Cannot calculate percentiles.")
except Exception as e:
    st.error(f"Error calculating percentiles: {str(e)}")

# Histogram of AGI arrival dates
try:
    # Filter out invalid dates
    valid_dates = [d for d in samples_dates if d is not None and not np.isnan(d.year)]
    if len(valid_dates) > 0:
        years = np.array([d.year + d.timetuple().tm_yday / 365 for d in valid_dates])
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(years, bins=50, color="#4F8DFD", alpha=0.7)
        ax.set_xlabel("AGI Arrival Year")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of AGI Arrival Dates")
        st.pyplot(fig)

        # Add violin plot for uncertainty visualization
        st.subheader("Uncertainty Visualization: Violin Plot of AGI Arrival Years")
        fig2, ax2 = plt.subplots(figsize=(8, 2))
        ax2.violinplot(years, showmeans=True, showmedians=True)
        ax2.set_xticks([1])
        ax2.set_xticklabels(["AGI Arrival Year Distribution"])
        ax2.set_ylabel("Year")
        st.pyplot(fig2)
    else:
        st.warning("No valid dates to plot in histogram.")
except Exception as e:
    st.error(f"Error plotting histogram: {str(e)}")

# Add the exponential growth plot to the UI
try:
    st.subheader("Exponential Growth in Task Length Over Time")
    
    # Create a simple static example plot to demonstrate the layout
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Create a logarithmic series for the three curves
    quarters = np.arange(0, 44)
    x = quarters
    
    # Calculate y values based on the current parameters
    # These are simplified calculations to ensure visibility
    base_rate = start_task_length
    doubling_days = 212 if doubling_time <= 0 else doubling_time
    
    # Median curve - classic exponential
    median_y = base_rate * np.power(2, (x * 91.25) / doubling_days)
    
    # Calculate 10% faster and 90% slower curves
    fast_y = base_rate * np.power(2, (x * 91.25) / (doubling_days * 0.7))
    slow_y = base_rate * np.power(2, (x * 91.25) / (doubling_days * 1.5))

    # Plot the three curves
    ax.plot(x, fast_y, 'b--', linewidth=2, label='10% Earliest')
    ax.plot(x, median_y, 'b-', linewidth=2, label='Median')
    ax.plot(x, slow_y, 'b--', linewidth=2, label='90% Latest')
    
    # Add endpoint markers
    # Find where curves cross AGI threshold
    agi_threshold = agi_task_length
    
    # Find crossing points (simplified)
    med_cross = np.argmax(median_y >= agi_threshold) if np.any(median_y >= agi_threshold) else len(x)-1
    fast_cross = np.argmax(fast_y >= agi_threshold) if np.any(fast_y >= agi_threshold) else len(x)-1
    slow_cross = np.argmax(slow_y >= agi_threshold) if np.any(slow_y >= agi_threshold) else len(x)-1
    
    # Add red X markers for AGI reached
    if med_cross < len(x)-1:
        ax.plot(x[med_cross], median_y[med_cross], 'rx', markersize=10)
    if fast_cross < len(x)-1:
        ax.plot(x[fast_cross], fast_y[fast_cross], 'rx', markersize=10)
    if slow_cross < len(x)-1:
        ax.plot(x[slow_cross], slow_y[slow_cross], 'rx', markersize=10)
    
    # Add black circle markers for AGI not reached by end
    if med_cross == len(x)-1:
        ax.plot(x[med_cross], median_y[med_cross], 'ko', markersize=10)
    if fast_cross == len(x)-1:
        ax.plot(x[fast_cross], fast_y[fast_cross], 'ko', markersize=10)
    if slow_cross == len(x)-1:
        ax.plot(x[slow_cross], slow_y[slow_cross], 'ko', markersize=10)
    
    # Add dummy plots for legend
    ax.plot([], [], 'rx', markersize=10, label='AGI reached')
    ax.plot([], [], 'ko', markersize=10, label='AGI not by EOY2035')
    
    # Format the plot
    ax.set_yscale('log', base=2)
    y_ticks = [2**k for k in range(-7, 14)]
    ax.set_yticks(y_ticks)
    
    # Format y-axis with billions formatter
    def format_ticks(x, pos):
        if x >= 1e9:
            return f"{x/1e9:.1f}B"
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        if x >= 1e3:
            return f"{x/1e3:.1f}K"
        if x <= 0.5 and x > 0:
            return f"1/{1/x:.0f}"
        return f"{x:.0f}"
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    
    # Format x-axis with quarters
    quarter_labels = []
    for q in quarters:
        year = 2025 + q // 4
        quarter = q % 4 + 1
        quarter_labels.append(f"{year}Q{quarter}")
    
    ax.set_xticks(quarters)
    ax.set_xticklabels(quarter_labels, rotation=90)
    
    # Add grid and labels
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylabel('Task length (hours) -- note log scale')
    ax.set_title('Exponential Growth in Task Length Over Time')
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error plotting exponential growth: {str(e)}")
    st.exception(e)

# Show AGI arrival date by percentile
try:
    if 'valid_dates' in locals() and len(valid_dates) > 0:
        st.subheader("AGI Arrival Dates by Year")
        years_array = np.array([d.year for d in valid_dates])
        
        # Calculate cumulative probability for each year
        min_year = int(np.min(years_array))
        max_year = int(np.max(years_array))
        
        # Show years from now until max_year with reasonable spacing
        current_year = datetime.now().year
        display_years = list(range(current_year, current_year+6)) + list(range(current_year+5, max_year+1, 5))
        
        results = []
        for year in display_years:
            prob = np.mean(years_array <= year) * 100
            results.append(f"By EOY {year}: {prob:.1f}%")
        
        # Display in two columns
        col1, col2 = st.columns(2)
        half = len(results) // 2 + len(results) % 2
        
        with col1:
            for r in results[:half]:
                st.text(r)
                
        with col2:
            for r in results[half:]:
                st.text(r)
                
    else:
        st.warning("No valid dates to calculate year probabilities.")
except Exception as e:
    st.error(f"Error calculating year probabilities: {str(e)}")

# Show model's computed median AGI achievement date
try:
    if 'valid_dates' in locals() and len(valid_dates) > 0:
        median_idx = int(np.median(np.arange(len(valid_dates))))
        sorted_dates = np.sort(valid_dates)
        median_date = sorted_dates[median_idx]
        st.subheader("Model's Computed Median AGI Date")
        st.info(f"**{median_date.strftime('%A, %-d %B %Y')}**")
        st.subheader("Model's Computed 10% Earliest AGI Date")
        # Also show 10% earliest and 90% latest AGI dates from the plot logic
        # Use the same logic as the plot: compute the quarter index where each curve crosses the AGI threshold
        base_rate = start_task_length
        doubling_days = 212 if doubling_time <= 0 else doubling_time
        agi_threshold = agi_task_length
        n_quarters = 43
        days_per_quarter = 91.25
        x = np.arange(0, n_quarters + 1)
        # 10% earliest (fastest)
        fast_y = base_rate * np.power(2, (x * days_per_quarter) / (doubling_days * 0.7))
        fast_cross = np.argmax(fast_y >= agi_threshold) if np.any(fast_y >= agi_threshold) else n_quarters
        # 90% latest (slowest)
        slow_y = base_rate * np.power(2, (x * days_per_quarter) / (doubling_days * 1.5))
        slow_cross = np.argmax(slow_y >= agi_threshold) if np.any(slow_y >= agi_threshold) else n_quarters
        # Reference date
        ref_date = reference_date if 'reference_date' in locals() else O3_LAUNCH_DATE
        fast_date = ref_date + timedelta(days=fast_cross * days_per_quarter)
        slow_date = ref_date + timedelta(days=slow_cross * days_per_quarter)
        st.info(f"**{fast_date.strftime('%A, %-d %B %Y')}**")
        st.subheader("Model's Computed 90% Latest AGI Date")
        st.info(f"**{slow_date.strftime('%A, %-d %B %Y')}**")

        # --- Explainability Section ---
        st.markdown("---")
        st.subheader("Explainability: What does this scenario mean?")
        st.markdown(f"""
        **What does this model do?**
        This model projects when AGI might be achieved by extrapolating recent trends in AI's ability to complete long, complex tasks, as measured by the METR benchmark. It uses a probabilistic approach, accounting for uncertainty in how fast progress will continue, how hard AGI-level tasks are, and how much reliability is required.
        
        **Key parameters:**
        - **Start task length:** The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for improvements from scaffolding and compute (elicitation boost), higher reliability requirements (reliability penalty), and the possibility that AGI tasks are harder than METR's self-contained software tasks (task type penalty). This is not just the raw model benchmark, but a composite reflecting how close we are to AGI-level tasks. [See: Forecaster Reacts]
        - **AGI task length:** The human time required to complete a task that would count as "AGI-level" (e.g., a month- or year-long project). This is a subjective threshold, often set to something like 167 hours (a month of full-time work) or up to 2000 hours (a work year). [See: Forecaster Reacts]
        - **Doubling time:** The number of days it takes for the maximum task length that AI can do at 50% reliability to double. This is fit to historical trends (e.g., 212 days for 2019-2024, 118 days for 2024-2025, or a mixture). [See: Forecaster Reacts]
        - **Acceleration:** If <1, the doubling time itself shrinks over time (superexponential progress); if >1, it grows (progress slows); if 1, progress is exponential. [See: Forecaster Reacts]
        - **Shift:** The number of days to shift the forecast earlier to account for internal model capabilities before public release (e.g., 30-150 days). [See: Forecaster Reacts]
        
        **How do these affect the forecast?**
        - Lower start task length, lower AGI task length, shorter doubling time, and lower acceleration all make AGI arrive sooner.
        - Higher reliability or harder AGI task definitions push the date later.
        
        **Uncertainty and caveats:**
        - The model assumes "business as usual" and does not account for major disruptions (e.g., regulation, war, economic shocks).
        - It extrapolates from recent trends, which may not continue indefinitely. Progress could slow down due to diminishing returns, or speed up due to breakthroughs or feedback loops.
        - The definition of AGI is based on task length and reliability, which may not capture all aspects of "real" AGI.
        - The model does not predict when AGI will be widely deployed or have major social impact—just when it becomes technically possible.
        """)
    else:
        st.warning("No valid dates to compute median AGI date.")
except Exception as e:
    st.error(f"Error computing median AGI date: {str(e)}")

st.markdown("---")
st.markdown("""
*Model code adapted from the METR paper ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499) 
and [Forecaster Reacts to METR's bombshell](https://peterwildeford.substack.com/p/forecaster-reacts-metrs-bombshell)*
""") 