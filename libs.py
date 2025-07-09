import numpy as np
import pandas as pd
import squigglepy as sq
import matplotlib.pyplot as plt

from pprint import pprint
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize, Bounds


DAYS_PER_QUARTER = 365 / 4


def run_model(model, index_date, cores=1):
    samples = sq.sample(model, n=100_000, verbose=True, cores=cores)
    pprint(sq.get_percentiles(samples, digits=0))
    print("\n-\n")
    samples_ = sq.get_percentiles(samples_to_date(samples, index_date=index_date))
    samples_ = {k: v.strftime("%Y %b %d") for k, v in samples_.items()}
    pprint(samples_)
    return samples


def samples_to_date(samples, index_date):
    date_converter = np.vectorize(
        lambda x: index_date + timedelta(days=int(np.ceil(x)))
    )
    return date_converter(samples)


def calculate_doubling_time(
    start_task_length, agi_task_length, doubling_time, acceleration=1
):
    """
    Parameters
    ----------
    start_task_length : scalar or distribution
        Current hours needed for the reference task.
    agi_task_length : scalar or distribution
        Hours required for the task at AGI.
    initial_doubling_time : scalar or distribution (days)
        Doubling time at the *current* capability level.
    acceleration : scalar or distribution
        Multiplicative factor applied to the doubling time *after every doubling*.
        • 1.0  → constant exponential growth (baseline).
        • <1.0 → doubling time shrinks, giving super‑exponential growth.
        • >1.0 → growth slows over time.
    """
    doublings_needed = sq.dist_log(agi_task_length / start_task_length) / np.log(2)
    if acceleration == 1:
        return doublings_needed * doubling_time
    else:
        return doubling_time * (1 - acceleration**doublings_needed) / (1 - acceleration)


def _pretty_time(hours: float) -> str:
    """Return a string with value + unit, choosing h / min / s."""
    if hours >= 1:
        return f"{hours:6.2f}hr"
    minutes = hours * 60
    if minutes >= 1:
        return f"{minutes:6.2f}min"
    seconds = minutes * 60
    return f"{seconds:6.0f}sec"


def test_acceleration(
    start_task_length: float,
    agi_task_length: float,
    initial_doubling_time: float,
    acceleration: float = 1.0,
    start_date: str | datetime | None = None,
    date_fmt: str = "%Y‑%m‑%d",
):
    # Anchor date
    if start_date is None:
        start_date = datetime.today()
    elif isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)

    current_task = start_task_length
    days_elapsed = 0.0
    tau = initial_doubling_time
    step = 0

    header = f"{'Step':>4} | {'Date':^10} | {'Day':>6} | {'Task':>10} | τ (d)"
    print(header)
    print("-" * len(header))

    while current_task < agi_task_length:
        date = start_date + timedelta(days=days_elapsed)
        print(
            f"{step:4d} | {date.strftime(date_fmt)} | "
            f"{int(days_elapsed):6d} | {_pretty_time(current_task):>10} | {tau:5.1f}"
        )

        current_task *= 2  # actual doubling
        days_elapsed += tau
        tau *= acceleration  # super‑/sub‑exponential effect
        step += 1

    # final line after exceeding target
    date = start_date + timedelta(days=days_elapsed)
    print(
        f"{step:4d} | {date.strftime(date_fmt)} | "
        f"{int(days_elapsed):6d} | {_pretty_time(current_task):>10} | {tau:5.1f}  <-- reached target"
    )


def estimate_growth_parameters(
    observations, baseline_date=None, baseline_task_hours=None, reliability_level="50%"
):
    """
    Returns (initial_doubling_time, acceleration) where:
    - initial_doubling_time = doubling time in days at baseline capability
    - acceleration = multiplicative change after each doubling
    """
    # Pick the right column based on reliability level
    if reliability_level == "50%":
        hours_idx = 2
    elif reliability_level == "80%":
        hours_idx = 3
    else:
        raise ValueError("reliability_level must be '50%' or '80%'")

    clean_data = [
        (name, date, obs[hours_idx])
        for obs in observations
        for name, date in [(obs[0], obs[1])]
    ]

    if baseline_date is None:
        baseline_date = clean_data[0][1]
    if baseline_task_hours is None:
        baseline_task_hours = clean_data[0][2]

    doublings = np.log(
        [hours / baseline_task_hours for _, _, hours in clean_data]
    ) / np.log(2)
    elapsed_days = np.array(
        [(date - baseline_date).days for _, date, _ in clean_data], dtype=float
    )

    def mse_loss(params):
        doubling_time, accel = params
        if doubling_time <= 0 or not 0 < accel < 2:
            return np.inf
        if np.isclose(accel, 1.0):
            prediction = doublings * doubling_time
        else:
            prediction = doubling_time * (1 - accel**doublings) / (1 - accel)
        return np.mean((prediction - elapsed_days) ** 2)

    bounds = Bounds([1e-6, 0.9], [np.inf, 1.0])
    result = minimize(mse_loss, x0=[260.0, 0.95], method="L-BFGS-B", bounds=bounds)
    doubling_time, acceleration = result.x
    return round(doubling_time), round(acceleration, 3)


def print_estimation(data, reliability_level="50%"):
    start = data[0][0]
    end = data[-1][0]
    params = estimate_growth_parameters(data, reliability_level=reliability_level)
    print(f"{start} to {end} ({reliability_level}): {params}")


def bootstrap_growth_parameters(
    observations,
    n_bootstrap=1000,
    reliability_level="50%",
    min_models=5,
    recent_weight=2.0,
):
    """
    Bootstrap confidence intervals for growth parameters with time-based weighting.
    """
    n_obs = len(observations)
    results = []

    # Weight more recent observations higher
    weights = np.array([recent_weight ** (i / n_obs) for i in range(n_obs)])
    weights /= weights.sum()

    for _ in range(n_bootstrap):
        # Sample with replacement, weighted by recency
        indices = np.random.choice(n_obs, size=n_obs, replace=True, p=weights)
        bootstrap_sample = [observations[i] for i in sorted(indices)]

        # Only fit if we have enough unique models and reasonable time span
        if len(set(indices)) >= min_models:
            params = estimate_growth_parameters(
                bootstrap_sample, reliability_level=reliability_level
            )
            if params[0] < 1000:  # Filter out degenerate fits
                results.append(params)

    if not results:
        return None

    # Calculate percentiles
    results = np.array(results)
    percentiles = np.percentile(results, [2.5, 50, 97.5], axis=0)

    return {
        "median": (round(percentiles[1, 0]), round(percentiles[1, 1], 3)),
        "ci_95": {
            "doubling_time": (round(percentiles[0, 0]), round(percentiles[2, 0])),
            "acceleration": (round(percentiles[0, 1], 3), round(percentiles[2, 1], 3)),
        },
        "mean": (round(results[:, 0].mean()), round(results[:, 1].mean(), 3)),
        "std": (round(results[:, 0].std()), round(results[:, 1].std(), 3)),
    }


def sliding_window_analysis(
    observations, window_sizes=[6, 8, 10, 12], reliability_level="50%"
):
    """
    Test different time windows to see parameter stability.
    """
    results = []

    for window in window_sizes:
        if window <= len(observations):
            # Try all possible windows of this size
            for start in range(len(observations) - window + 1):
                subset = observations[start : start + window]
                params = estimate_growth_parameters(
                    subset, reliability_level=reliability_level
                )

                start_date = subset[0][1]
                end_date = subset[-1][1]
                time_span = (end_date - start_date).days

                results.append(
                    {
                        "window": window,
                        "start_model": subset[0][0],
                        "end_model": subset[-1][0],
                        "time_span_days": time_span,
                        "doubling_time": params[0],
                        "acceleration": params[1],
                    }
                )

    return pd.DataFrame(results)


def billions_formatter(x, pos):
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.1f}K"
    if x <= 0.5:
        return f"1/{1/x:.0f}"
    return f"{x:.0f}"


def _quarter_labels(n: int, start_year: int = 2025) -> list[str]:
    return [f"{start_year + q // 4}Q{q % 4 + 1}" for q in range(n + 1)]


def _y_ticks(lo: int, hi: int) -> list[int]:
    return [2**k for k in range(lo, hi + 1)]


def _first_curve(order, traj, reference, above):
    cmp = np.greater_equal if above else np.less_equal
    for idx in order:
        if np.all(cmp(traj[idx], reference)):
            return idx
    return order[0]


def plot_exponential_growth(
    doubling_time_days,
    starting_hours,
    agi_task_length,
    shift=0,
    acceleration=1,
    n_quarters: int = 40,
    n_samples: int = 10_000,
    n_traces: int = 100,
    max_task_power: int = 13,
    min_y_power: int = -8,
) -> None:
    max_task_hours = 2**max_task_power
    tau0 = sq.sample(doubling_time_days, n=n_samples)
    accel = sq.sample(acceleration, n=n_samples)
    shift = sq.sample(shift, n=n_samples)
    agi = sq.sample(sq.dist_min(max_task_hours, agi_task_length), n=n_samples)
    start = sq.sample(starting_hours, n=n_samples) * 2 ** (shift / tau0)

    quarters = np.arange(n_quarters + 1)
    traj = np.zeros((n_samples, len(quarters)))
    clip_idx = np.full(n_samples, len(quarters), dtype=int)

    for i in range(n_samples):
        tau = tau0[i]
        val = start[i]
        for j in range(len(quarters)):
            if val >= max_task_hours:
                traj[i, j:] = max_task_hours
                clip_idx[i] = j
                break
            traj[i, j] = val
            val *= 2 ** (DAYS_PER_QUARTER / tau)
            tau *= accel[i]

    reached = traj >= agi[:, None]
    first_hit = np.argmax(reached, axis=1)
    first_hit[np.all(~reached, axis=1)] = len(quarters)

    order = np.argsort(first_hit)
    median_idx = order[len(order) // 2]
    median_curve = traj[median_idx]

    idx10 = _first_curve(order[int(0.10 * n_samples) :], traj, median_curve, above=True)
    idx90 = _first_curve(
        order[: int(0.90 * n_samples)][::-1], traj, median_curve, above=False
    )

    highlights = {
        "10 % earliest": (traj[idx10], first_hit[idx10], clip_idx[idx10], "b--"),
        "Median": (median_curve, first_hit[median_idx], clip_idx[median_idx], "b-"),
        "90 % latest": (traj[idx90], first_hit[idx90], clip_idx[idx90], "b--"),
    }

    plt.figure(figsize=(11, 6))
    rng = np.random.default_rng()

    for i in rng.choice(n_samples, min(n_traces, n_samples), replace=False):
        end_q = min(first_hit[i], clip_idx[i], len(quarters) - 1)
        plt.plot(
            quarters[: end_q + 1],
            traj[i, : end_q + 1],
            color="tab:blue",
            lw=0.3,
            alpha=0.25,
        )
        marker = "rx" if first_hit[i] < len(quarters) else "ko"
        plt.plot(quarters[end_q], traj[i, end_q], marker, ms=4, alpha=0.6)

    for label, (curve, hit_q, clip_q, style) in highlights.items():
        end_q = min(hit_q, clip_q, len(quarters) - 1)
        plt.plot(quarters[: end_q + 1], curve[: end_q + 1], style, lw=2, label=label)
        marker = "rx" if hit_q < len(quarters) else "ko"
        plt.plot(quarters[end_q], curve[end_q], marker, ms=7)

    plt.plot([], [], "rx", ms=7, label="AGI reached")
    plt.plot([], [], "ko", ms=7, label=f"AGI not by {_quarter_labels(n_quarters)[-1]}")
    plt.yscale("log", base=2)

    nonzero_values = traj[traj > 0]
    percentile_0p1 = np.percentile(nonzero_values, 0.1)
    min_y_data = int(np.floor(np.log2(percentile_0p1)))
    y_lo = max(min_y_power, min_y_data)

    plt.yticks(_y_ticks(lo=y_lo, hi=max_task_power))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(billions_formatter))
    plt.xticks(quarters, _quarter_labels(n_quarters), rotation=90)
    plt.grid(ls="--", alpha=0.7)
    plt.ylabel("Task length (hours) -- note log scale")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.show()
    return None
