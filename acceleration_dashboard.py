import streamlit as st
import numpy as np
import pandas as pd
import squigglepy as sq
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize, Bounds

from libs import (
    estimate_growth_parameters,
    estimate_doubling_time_only,
    bootstrap_growth_parameters,
    bootstrap_doubling_time_fixed,
    sliding_window_analysis,
)
from model_data import model_data

st.set_page_config(page_title="AGI Acceleration Tracker", layout="wide")
st.title("AGI Acceleration Tracker")

st.markdown("""
This dashboard analyzes AI capability growth based on the [METR task completion benchmarks](https://arxiv.org/abs/2503.14499).
It helps estimate how quickly AI systems are improving and when they might reach AGI-level capabilities.

**Key concepts:**
- **Doubling time**: How many days it takes for AI task completion capability to double
- **Acceleration**: Whether the doubling time itself is changing (< 1 means speeding up, = 1 means constant, > 1 means slowing down)
- **Task length**: The duration of tasks an AI can reliably complete (longer = more capable)
""")

# Build observed_models from model_data
observed_models = [
    (model['name'], model['launch_date'], model['performance_50p'], model['performance_80p'])
    for model in model_data.values()
    if model['performance_50p'] is not None
]

model_names = [m[0] for m in observed_models]

# Sidebar configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("""
Select models and parameters to analyze AI capability growth trends.
""")

tab1, tab2, tab3 = st.tabs(["Parameter Estimation", "Simple Models", "Acceleration Projections"])

# ============ TAB 1: Parameter Estimation ============
with tab1:
    st.header("Growth Parameter Estimation")

    st.markdown("""
    This section estimates the **doubling time** and **acceleration** parameters from historical model data.

    - Select a range of models to fit the growth curve
    - The algorithm finds parameters that best explain the observed capability improvements
    - Use bootstrap analysis to understand uncertainty in these estimates
    """)

    col1, col2 = st.columns(2)
    with col1:
        start_model_idx = st.selectbox(
            "Start model",
            range(len(model_names)),
            format_func=lambda i: model_names[i],
            index=0
        )
    with col2:
        end_model_idx = st.selectbox(
            "End model",
            range(len(model_names)),
            format_func=lambda i: model_names[i],
            index=len(model_names) - 1
        )

    reliability = st.radio("Reliability level", ["50%", "80%"], horizontal=True,
                          help="50% = tasks the model completes half the time; 80% = tasks completed 80% of the time (harder threshold)")

    if start_model_idx >= end_model_idx:
        st.error("Start model must come before end model")
    else:
        selected_models = observed_models[start_model_idx:end_model_idx + 1]

        # Estimate parameters - both with and without acceleration
        params = estimate_growth_parameters(selected_models, reliability_level=reliability)
        doubling_time, acceleration = params

        doubling_time_fixed = estimate_doubling_time_only(selected_models, reliability_level=reliability)

        st.subheader("Estimated Parameters")

        st.markdown("""
        Two models are fit to the data:
        1. **With acceleration**: Allows the doubling time to change over time
        2. **Fixed acceleration (=1)**: Assumes constant exponential growth
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Doubling Time (with accel)", f"{doubling_time} days")
            st.metric("Doubling Time (fixed accel=1)", f"{doubling_time_fixed} days")
        with col2:
            st.metric("Acceleration", f"{acceleration}")
            accel_desc = "speeding up" if acceleration < 1 else ("constant" if acceleration == 1 else "slowing down")
            st.caption(f"Growth is {accel_desc}")
        with col3:
            n_models = len(selected_models)
            time_span = (selected_models[-1][1] - selected_models[0][1]).days
            st.metric("Models / Time Span", f"{n_models} models / {time_span} days")

        # ============ HINGE MODEL COMPARISON ============
        st.subheader("Hinge Model: Pre-2024 vs 2024+ Comparison")

        st.markdown("""
        **Question**: Has AI progress accelerated since 2024?

        This section fits separate models to pre-2024 and 2024+ data, then compares
        the fit quality to a single trend across all data. If the two-regime model
        fits substantially better, it suggests a structural change in progress rate.
        """)

        hinge_date = datetime(2024, 1, 1)
        pre_2024_models = [m for m in selected_models if m[1] < hinge_date]
        post_2024_models = [m for m in selected_models if m[1] >= hinge_date]

        if len(pre_2024_models) >= 3 and len(post_2024_models) >= 3:
            # Fit separate models
            pre_params = estimate_growth_parameters(pre_2024_models, reliability_level=reliability)
            post_params = estimate_growth_parameters(post_2024_models, reliability_level=reliability)

            pre_doubling_fixed = estimate_doubling_time_only(pre_2024_models, reliability_level=reliability)
            post_doubling_fixed = estimate_doubling_time_only(post_2024_models, reliability_level=reliability)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Pre-2024**")
                st.write(f"Models: {len(pre_2024_models)}")
                st.write(f"Doubling time: {pre_params[0]} days (accel={pre_params[1]})")
                st.write(f"Doubling time (fixed): {pre_doubling_fixed} days")

            with col2:
                st.markdown("**2024+**")
                st.write(f"Models: {len(post_2024_models)}")
                st.write(f"Doubling time: {post_params[0]} days (accel={post_params[1]})")
                st.write(f"Doubling time (fixed): {post_doubling_fixed} days")

            with col3:
                st.markdown("**Comparison**")
                speedup = pre_doubling_fixed / post_doubling_fixed if post_doubling_fixed > 0 else 0
                st.write(f"Speedup factor: {speedup:.2f}x")
                if speedup > 1.5:
                    st.success("Progress appears significantly faster post-2024")
                elif speedup > 1.1:
                    st.info("Progress appears somewhat faster post-2024")
                else:
                    st.warning("No clear speedup detected")

            # Calculate MSE for both models
            def calc_mse(models, doubling_time, acceleration=1.0):
                if len(models) < 2:
                    return float('inf')
                baseline_date = models[0][1]
                hours_idx = 2 if reliability == "50%" else 3
                baseline_hours = models[0][hours_idx]

                total_mse = 0
                for m in models:
                    doublings = np.log(m[hours_idx] / baseline_hours) / np.log(2)
                    actual_days = (m[1] - baseline_date).days
                    if acceleration == 1.0:
                        predicted_days = doublings * doubling_time
                    else:
                        predicted_days = doubling_time * (1 - acceleration**doublings) / (1 - acceleration)
                    total_mse += (actual_days - predicted_days) ** 2
                return total_mse / len(models)

            # Single model MSE
            single_mse = calc_mse(selected_models, doubling_time_fixed, 1.0)

            # Two-regime MSE (weighted average)
            pre_mse = calc_mse(pre_2024_models, pre_doubling_fixed, 1.0)
            post_mse = calc_mse(post_2024_models, post_doubling_fixed, 1.0)
            two_regime_mse = (pre_mse * len(pre_2024_models) + post_mse * len(post_2024_models)) / len(selected_models)

            st.markdown("**Model Fit Comparison (MSE)**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Single trend MSE", f"{single_mse:.0f}")
            with col2:
                st.metric("Two-regime MSE", f"{two_regime_mse:.0f}")
                improvement = (single_mse - two_regime_mse) / single_mse * 100 if single_mse > 0 else 0
                if improvement > 10:
                    st.caption(f"Two-regime model fits {improvement:.0f}% better")
        else:
            st.info(f"Need at least 3 models in each period. Pre-2024: {len(pre_2024_models)}, Post-2024: {len(post_2024_models)}")

        # ============ BOOTSTRAP ANALYSIS ============
        st.subheader("Bootstrap Confidence Intervals")

        st.markdown("""
        Bootstrap resampling provides confidence intervals for the parameter estimates.
        This helps understand how uncertain we should be about the doubling time.

        - **Censoring adjustment**: Accounts for the fact that we haven't seen a new model release yet
          (if the true doubling time were very short, we'd expect to have seen more progress by now)
        """)

        include_censoring = st.checkbox("Account for time since last model (censoring)", value=True)
        current_date = datetime.today() if include_censoring else None
        n_bootstrap = st.slider("Number of bootstrap samples", 500, 5000, 1000, step=500)

        with st.spinner("Running bootstrap analysis..."):
            bootstrap_results = bootstrap_growth_parameters(
                selected_models,
                reliability_level=reliability,
                current_date=current_date,
                n_bootstrap=n_bootstrap
            )

        if bootstrap_results:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Median estimate:**")
                st.write(f"Doubling time: {bootstrap_results['median'][0]} days")
                st.write(f"Acceleration: {bootstrap_results['median'][1]}")
            with col2:
                st.write("**95% Confidence Intervals:**")
                ci = bootstrap_results['ci_95']
                st.write(f"Doubling time: {ci['doubling_time'][0]} - {ci['doubling_time'][1]} days")
                st.write(f"Acceleration: {ci['acceleration'][0]} - {ci['acceleration'][1]}")

            # Additional bootstrap for future predictions
            st.markdown("---")
            st.markdown("**Projected Future Doubling Time**")
            st.markdown("""
            Based on the bootstrap samples, here's the distribution of what the effective
            doubling time might be over the next 1-3 years:
            """)

            # Use bootstrap results to project
            median_doubling = bootstrap_results['median'][0]
            median_accel = bootstrap_results['median'][1]
            ci_low = ci['doubling_time'][0]
            ci_high = ci['doubling_time'][1]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current estimate", f"{median_doubling} days",
                         help=f"95% CI: {ci_low}-{ci_high} days")
            with col2:
                # Project 2 doublings ahead
                future_doubling = median_doubling * (median_accel ** 2)
                st.metric("After 2 more doublings", f"{int(future_doubling)} days",
                         delta=f"{int(future_doubling - median_doubling)} days" if future_doubling != median_doubling else None)
            with col3:
                # Project 4 doublings ahead
                future_doubling_4 = median_doubling * (median_accel ** 4)
                st.metric("After 4 more doublings", f"{int(future_doubling_4)} days",
                         delta=f"{int(future_doubling_4 - median_doubling)} days" if future_doubling_4 != median_doubling else None)
        else:
            st.warning("Bootstrap analysis did not produce results. Try selecting more models.")

        # ============ BOOTSTRAP WITH FIXED ACCELERATION ============
        st.subheader("Bootstrap Confidence Intervals (Acceleration = 1)")

        st.markdown("""
        This analysis assumes **pure exponential growth** (acceleration fixed at 1.0).
        This is a simpler model that may be more robust when data is limited.
        """)

        with st.spinner("Running fixed-acceleration bootstrap..."):
            bootstrap_fixed = bootstrap_doubling_time_fixed(
                selected_models,
                reliability_level=reliability,
                current_date=current_date,
                n_bootstrap=n_bootstrap
            )

        if bootstrap_fixed:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Median Doubling Time", f"{bootstrap_fixed['median']} days")
                st.write(f"Mean: {bootstrap_fixed['mean']} days (SD: {bootstrap_fixed['std']})")
            with col2:
                st.write("**Confidence Intervals:**")
                st.write(f"95% CI: {bootstrap_fixed['ci_95'][0]} - {bootstrap_fixed['ci_95'][1]} days")
                st.write(f"90% CI: {bootstrap_fixed['ci_90'][0]} - {bootstrap_fixed['ci_90'][1]} days")
                st.write(f"80% CI: {bootstrap_fixed['ci_80'][0]} - {bootstrap_fixed['ci_80'][1]} days")
        else:
            st.warning("Fixed-acceleration bootstrap did not produce results.")

        # ============ QUARTERLY HORIZON PROJECTIONS 2026-2030 ============
        st.subheader("Projected METR Horizon Length by Quarter (2026-2030)")

        st.markdown("""
        These projections show the **expected task completion capability** (METR horizon length in hours)
        at each quarter, with uncertainty bounds based on the bootstrap distribution of doubling times.

        **Methodology:**
        - Uses the latest model's capability as the starting point
        - Projects forward using sampled doubling times from bootstrap
        - Shows median projection with 10th-90th percentile uncertainty range
        """)

        # Get the latest model for starting point
        latest_model_data = observed_models[-1]
        latest_model_name = latest_model_data[0]
        latest_date = latest_model_data[1]
        start_task_50 = latest_model_data[2]  # 50% reliability
        start_task_80 = latest_model_data[3]  # 80% reliability

        st.write(f"**Starting from:** {latest_model_name} ({latest_date.strftime('%Y-%m-%d')})")
        st.write(f"- 50% reliability: {start_task_50:.2f} hours ({start_task_50 * 60:.1f} min)")
        st.write(f"- 80% reliability: {start_task_80:.2f} hours ({start_task_80 * 60:.1f} min)")

        # Run bootstrap for both reliability levels
        n_proj_samples = 5000

        with st.spinner("Calculating quarterly projections..."):
            # Bootstrap for 50% reliability
            bootstrap_50 = bootstrap_doubling_time_fixed(
                selected_models,
                reliability_level="50%",
                current_date=current_date,
                n_bootstrap=n_proj_samples
            )

            # Bootstrap for 80% reliability
            bootstrap_80 = bootstrap_doubling_time_fixed(
                selected_models,
                reliability_level="80%",
                current_date=current_date,
                n_bootstrap=n_proj_samples
            )

        if bootstrap_50 and bootstrap_80 and 'samples' in bootstrap_50 and 'samples' in bootstrap_80:
            doubling_samples_50 = bootstrap_50['samples']
            doubling_samples_80 = bootstrap_80['samples']

            def format_hours(hours):
                """Format hours into readable string."""
                if hours < 1:
                    return f"{hours * 60:.1f} min"
                elif hours < 24:
                    return f"{hours:.1f} hr"
                elif hours < 168:  # less than a week
                    return f"{hours / 24:.1f} days"
                elif hours < 720:  # less than a month
                    return f"{hours / 168:.1f} wk"
                elif hours < 8760:  # less than a year
                    return f"{hours / 720:.1f} mo"
                else:
                    return f"{hours / 8760:.1f} yr"

            # Generate quarters from 2026 Q1 to 2030 Q4
            quarters = []
            for year in range(2026, 2031):
                for q in range(1, 5):
                    quarter_start = datetime(year, (q - 1) * 3 + 1, 1)
                    quarters.append((f"{year} Q{q}", quarter_start))

            # Calculate projected task length for each quarter
            results_data = []
            for quarter_label, quarter_date in quarters:
                days_elapsed = (quarter_date - latest_date).days
                if days_elapsed < 0:
                    continue

                # Calculate doublings that occur in this time for each bootstrap sample
                # doublings = days_elapsed / doubling_time
                doublings_50 = days_elapsed / doubling_samples_50
                doublings_80 = days_elapsed / doubling_samples_80

                # Project task length: start * 2^doublings
                projected_50 = start_task_50 * (2 ** doublings_50)
                projected_80 = start_task_80 * (2 ** doublings_80)

                # Get percentiles
                p10_50, p50_50, p90_50 = np.percentile(projected_50, [10, 50, 90])
                p10_80, p50_80, p90_80 = np.percentile(projected_80, [10, 50, 90])

                results_data.append({
                    'Quarter': quarter_label,
                    'Days': days_elapsed,
                    '50% rel (10th)': format_hours(p10_50),
                    '50% rel (median)': format_hours(p50_50),
                    '50% rel (90th)': format_hours(p90_50),
                    '80% rel (10th)': format_hours(p10_80),
                    '80% rel (median)': format_hours(p50_80),
                    '80% rel (90th)': format_hours(p90_80),
                    # Store raw values for plotting
                    '_p50_50': p50_50,
                    '_p10_50': p10_50,
                    '_p90_50': p90_50,
                    '_p50_80': p50_80,
                    '_p10_80': p10_80,
                    '_p90_80': p90_80,
                })

            # Display by year
            display_cols = ['Quarter', 'Days', '50% rel (10th)', '50% rel (median)', '50% rel (90th)',
                           '80% rel (10th)', '80% rel (median)', '80% rel (90th)']

            for year in range(2026, 2031):
                st.markdown(f"**{year}**")
                year_data = [r for r in results_data if r['Quarter'].startswith(str(year))]
                if year_data:
                    year_df = pd.DataFrame(year_data)[display_cols]
                    st.dataframe(year_df, hide_index=True, use_container_width=True)

            # Visualization
            st.markdown("---")
            st.markdown("**Projected METR Horizon Length Over Time**")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            quarter_labels = [r['Quarter'] for r in results_data]
            x = range(len(quarter_labels))

            # 50% reliability plot
            median_50 = [r['_p50_50'] for r in results_data]
            p10_50 = [r['_p10_50'] for r in results_data]
            p90_50 = [r['_p90_50'] for r in results_data]

            ax1.semilogy(x, median_50, 'b-o', label='Median', markersize=6)
            ax1.fill_between(x, p10_50, p90_50, alpha=0.3, color='blue', label='10th-90th percentile')
            ax1.axhline(y=167, color='r', linestyle='--', label='AGI threshold (167 hr)')
            ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='1 hour')
            ax1.axhline(y=24, color='gray', linestyle=':', alpha=0.5, label='1 day')

            ax1.set_xticks(x)
            ax1.set_xticklabels(quarter_labels, rotation=45, ha='right')
            ax1.set_ylabel('Task Length (hours, log scale)')
            ax1.set_title('50% Reliability')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # 80% reliability plot
            median_80 = [r['_p50_80'] for r in results_data]
            p10_80 = [r['_p10_80'] for r in results_data]
            p90_80 = [r['_p90_80'] for r in results_data]

            ax2.semilogy(x, median_80, 'r-o', label='Median', markersize=6)
            ax2.fill_between(x, p10_80, p90_80, alpha=0.3, color='red', label='10th-90th percentile')
            ax2.axhline(y=167, color='r', linestyle='--', label='AGI threshold (167 hr)')
            ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='1 hour')
            ax2.axhline(y=24, color='gray', linestyle=':', alpha=0.5, label='1 day')

            ax2.set_xticks(x)
            ax2.set_xticklabels(quarter_labels, rotation=45, ha='right')
            ax2.set_ylabel('Task Length (hours, log scale)')
            ax2.set_title('80% Reliability')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Key milestones
            st.markdown("---")
            st.markdown("**Key Milestone Projections**")

            milestones = [
                (1, "1 hour tasks"),
                (8, "Full workday (8 hr)"),
                (24, "1 day tasks"),
                (168, "1 week tasks"),
                (720, "1 month tasks"),
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**50% Reliability**")
                for target, label in milestones:
                    if target > start_task_50:
                        # Find when median projection crosses this threshold
                        for r in results_data:
                            if r['_p50_50'] >= target:
                                st.write(f"- {label}: ~{r['Quarter']} (median)")
                                break
                        else:
                            st.write(f"- {label}: After 2030")

            with col2:
                st.markdown("**80% Reliability**")
                for target, label in milestones:
                    if target > start_task_80:
                        for r in results_data:
                            if r['_p50_80'] >= target:
                                st.write(f"- {label}: ~{r['Quarter']} (median)")
                                break
                        else:
                            st.write(f"- {label}: After 2030")

        else:
            st.warning("Could not generate quarterly projections. Bootstrap may have failed.")

        # Sliding window analysis
        st.subheader("Parameter Stability by Window Size")

        st.markdown("""
        This analysis slides a window across the model timeline to see how stable
        the parameter estimates are. High variance suggests the parameters may be
        changing over time or are sensitive to which models are included.
        """)

        df = sliding_window_analysis(selected_models, reliability_level=reliability)
        if not df.empty:
            summary = df.groupby("window")[["doubling_time", "acceleration"]].agg(["mean", "std"])
            summary.columns = ['Doubling Time (mean)', 'Doubling Time (std)',
                             'Acceleration (mean)', 'Acceleration (std)']
            st.dataframe(summary.round(2))
        else:
            st.info("Not enough models for sliding window analysis")

# ============ TAB 2: Simple Models ============
with tab2:
    st.header("Simple AGI Timeline Models")

    st.markdown("""
    Run simple timeline projections using different parameter assumptions.

    **How it works:**
    1. Start with a model's current task completion capability
    2. Apply exponential growth with the specified doubling time
    3. Calculate when capability reaches AGI-level (167 hours = ~1 month of work)

    The measurement error option adds realistic uncertainty about the true capability level.
    """)

    model_type = st.selectbox(
        "Model Type",
        ["METR Paper Baseline", "From Specific Model", "Custom Parameters"]
    )

    if model_type == "METR Paper Baseline":
        st.info("Using METR paper parameters: start=1hr, AGI=167hrs, doubling=212 days")
        start_task = 1.0
        agi_task = 167.0
        doubling = 212
        accel = 1.0
        reference_model = 'claude_3p7_sonnet'

    elif model_type == "From Specific Model":
        col1, col2 = st.columns(2)
        with col1:
            reference_model = st.selectbox(
                "Reference model",
                [k for k in model_data.keys() if model_data[k]['performance_50p'] is not None],
                format_func=lambda k: model_data[k]['name']
            )
            rel_level = st.radio("Use reliability level", ["50%", "80%"], horizontal=True, key="simple_rel")

        with col2:
            # Get estimated doubling time for recent models
            recent_models = [m for m in observed_models if m[1] >= datetime(2024, 1, 1)]
            if recent_models:
                recent_params = estimate_growth_parameters(recent_models, reliability_level=rel_level)
                suggested_doubling = recent_params[0]
            else:
                suggested_doubling = 120

            doubling = st.number_input("Doubling time (days)", value=suggested_doubling, min_value=1)
            accel = st.number_input("Acceleration", value=1.0, min_value=0.8, max_value=1.2, step=0.01)

        if rel_level == "50%":
            start_task = model_data[reference_model]['performance_50p']
        else:
            start_task = model_data[reference_model]['performance_80p']
        agi_task = 167.0

        st.write(f"**Start task length:** {start_task:.4f} hours ({start_task * 60:.2f} minutes)")

    else:  # Custom Parameters
        col1, col2 = st.columns(2)
        with col1:
            start_task = st.number_input("Start task length (hours)", value=0.5, min_value=0.001, format="%.4f")
            agi_task = st.number_input("AGI task length (hours)", value=167.0, min_value=1.0)
        with col2:
            doubling = st.number_input("Doubling time (days)", value=165, min_value=1)
            accel = st.number_input("Acceleration", value=1.0, min_value=0.8, max_value=1.2, step=0.01)
        reference_model = 'o3'

    shift = st.number_input("Shift (days to subtract)", value=0, min_value=0,
                           help="Account for elapsed time since the reference model's release")
    n_samples = st.slider("Number of samples", 1000, 100000, 10000, step=1000)

    add_error = st.checkbox("Add measurement error (lognormal)", value=True,
                           help="Adds realistic uncertainty about true capability levels")

    if st.button("Run Model", key="run_simple"):
        with st.spinner("Running model..."):
            doublings_needed = np.log(agi_task / start_task) / np.log(2)
            if accel == 1:
                days_base = doublings_needed * doubling
            else:
                days_base = doubling * (1 - accel**doublings_needed) / (1 - accel)

            if add_error:
                error = sq.sample(sq.lognorm(lognorm_mean=1, lognorm_sd=0.3), n=n_samples)
                days_samples = days_base * error - shift
            else:
                days_samples = np.full(n_samples, days_base - shift)

            index_date = model_data[reference_model]['launch_date']

            percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            pct_values = np.percentile(days_samples, percentiles)

            st.subheader("Results")

            st.write(f"**Doublings needed:** {doublings_needed:.1f}")
            st.write(f"**Base days to AGI:** {days_base:.0f} days ({days_base/365:.1f} years)")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Days to AGI (percentiles):**")
                results_df = pd.DataFrame({
                    'Percentile': [f"{p}%" for p in percentiles],
                    'Days': [int(v) for v in pct_values],
                    'Date': [(index_date + timedelta(days=int(v))).strftime('%Y-%m-%d') for v in pct_values]
                })
                st.dataframe(results_df, hide_index=True)

            with col2:
                st.write("**Probability by year:**")
                years_list = list(range(2025, 2036))
                probs = []
                for year in years_list:
                    year_start = datetime(year, 1, 1)
                    days_to_year = (year_start - index_date).days
                    prob = np.mean(days_samples <= days_to_year) * 100
                    probs.append(prob)

                year_df = pd.DataFrame({
                    'By EOY': years_list,
                    'Probability': [f"{p:.1f}%" for p in probs]
                })
                st.dataframe(year_df, hide_index=True)

            st.subheader("Distribution of AGI Arrival")
            fig, ax = plt.subplots(figsize=(10, 5))

            years = days_samples / 365 + index_date.year + (index_date.timetuple().tm_yday / 365)
            years_clean = years[(years > 2024) & (years < 2100)]

            ax.hist(years_clean, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.median(years_clean), color='red', linestyle='--', label=f'Median: {np.median(years_clean):.1f}')
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

# ============ TAB 3: Acceleration Projections ============
with tab3:
    st.header("Acceleration Projections")

    st.markdown("""
    **What is acceleration?**

    Acceleration models whether progress is speeding up or slowing down over time.
    - **Acceleration = 1.0**: Constant exponential growth (doubling time stays fixed)
    - **Acceleration < 1.0**: Super-exponential growth (doubling time shrinks each generation)
    - **Acceleration > 1.0**: Sub-exponential growth (doubling time increases, progress slows)

    For example, with acceleration = 0.95 and initial doubling time of 200 days:
    - 1st doubling: 200 days
    - 2nd doubling: 190 days
    - 3rd doubling: 180 days
    - ... and so on

    This compounding effect can dramatically change AGI timelines.
    """)

    col1, col2 = st.columns(2)

    with col1:
        start_model_key = st.selectbox(
            "Starting model",
            [k for k in model_data.keys() if model_data[k]['performance_50p'] is not None],
            format_func=lambda k: model_data[k]['name'],
            key="proj_start"
        )
        initial_doubling = st.number_input("Initial doubling time (days)", value=260, min_value=1, key="proj_doubling")

    with col2:
        proj_acceleration = st.number_input("Acceleration factor", value=0.95, min_value=0.8, max_value=1.2, step=0.01, key="proj_accel")
        target_hours = st.number_input("Target AGI task length (hours)", value=167.0, min_value=1.0, key="proj_target")

    if st.button("Generate Projection", key="run_proj"):
        start_task = model_data[start_model_key]['performance_50p']
        start_date = model_data[start_model_key]['launch_date']

        current_task = start_task
        days_elapsed = 0.0
        tau = initial_doubling
        step = 0

        rows = []

        def pretty_time(hours):
            if hours >= 1:
                return f"{hours:.2f}hr"
            minutes = hours * 60
            if minutes >= 1:
                return f"{minutes:.2f}min"
            seconds = minutes * 60
            return f"{seconds:.0f}sec"

        while current_task < target_hours and step < 50:
            date = start_date + timedelta(days=days_elapsed)
            rows.append({
                'Step': step,
                'Date': date.strftime('%Y-%m-%d'),
                'Days': int(days_elapsed),
                'Task Length': pretty_time(current_task),
                'Task (hours)': round(current_task, 4),
                'Doubling Time': round(tau, 1)
            })

            current_task *= 2
            days_elapsed += tau
            tau *= proj_acceleration
            step += 1

        date = start_date + timedelta(days=days_elapsed)
        rows.append({
            'Step': step,
            'Date': date.strftime('%Y-%m-%d'),
            'Days': int(days_elapsed),
            'Task Length': pretty_time(current_task) + " (TARGET)",
            'Task (hours)': round(current_task, 4),
            'Doubling Time': round(tau, 1)
        })

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.subheader("Capability Growth Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))

        dates = [start_date + timedelta(days=r['Days']) for r in rows]
        tasks = [r['Task (hours)'] for r in rows]

        ax.semilogy(dates, tasks, 'bo-', markersize=8)
        ax.axhline(y=target_hours, color='r', linestyle='--', label=f'AGI target: {target_hours}hr')

        ax.set_xlabel("Date")
        ax.set_ylabel("Task Length (hours, log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        final_date = start_date + timedelta(days=days_elapsed)
        st.success(f"**Projected AGI arrival: {final_date.strftime('%Y-%m-%d')}** ({int(days_elapsed)} days, {days_elapsed/365:.1f} years from {start_date.strftime('%Y-%m-%d')})")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard analyzes AI capability growth based on METR task completion benchmarks. "
    "See [METR paper](https://arxiv.org/abs/2503.14499) for methodology."
)
st.sidebar.markdown("""
**References:**
- [METR: Measuring AI Ability to Complete Long Tasks](https://arxiv.org/abs/2503.14499)
- [Forecaster Reacts to METR's bombshell](https://peterwildeford.substack.com/p/forecaster-reacts-metrs-bombshell)
""")
