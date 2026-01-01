import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import timedelta

def plot_agi_arrival_distribution(years):
    """Plots the histogram of AGI arrival years."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(years, bins=50, color="#4F8DFD", alpha=0.7)
    ax.set_xlabel("AGI Arrival Year")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of AGI Arrival Dates")
    return fig

def plot_confidence_intervals(years, percentiles, percentile_values, percentile_dates):
    """Plots the boxplot for confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.boxplot(years, vert=False, widths=0.7, 
                  whis=[5, 95],  # 5-95 percentile whiskers
                  medianprops=dict(color='red', linewidth=2),
                  boxprops=dict(linewidth=2),
                  whiskerprops=dict(linewidth=2),
                  capprops=dict(linewidth=2))
    
    # Add a rug plot
    ax.plot(years, np.random.normal(1, 0.04, size=len(years)), '|', 
              alpha=0.5, color='blue', markersize=8)
    
    # Annotate percentiles
    for i, p in enumerate(percentiles):
        if p in [5, 25, 50, 75, 95]:
            ax.text(percentile_values[i], 0.8, f"{p}%", 
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    # Add vertical lines
    for i, p in enumerate(percentiles):
        if p in [10, 50, 90]:
            linestyle = '-' if p == 50 else '--'
            ax.axvline(x=percentile_values[i], color='gray', linestyle=linestyle, alpha=0.6)
    
    ax.set_yticks([])
    ax.set_xlabel('AGI Arrival Year')
    ax.set_title('Confidence Intervals for AGI Arrival')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    # Set x-axis limits
    min_display = min(percentile_values) - 0.5
    max_display = min(max(percentile_values) + 0.5, 2050)
    ax.set_xlim(min_display, max_display)
    
    return fig

def plot_task_length_growth(start_task_length, agi_task_length, doubling_time, quarters=None):
    """Plots the exponential growth of task length."""
    if quarters is None:
        quarters = np.arange(0, 44)
    x = quarters
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    base_rate = start_task_length
    doubling_days = 212 if doubling_time <= 0 else doubling_time
    
    # Median curve
    median_y = base_rate * np.power(2, (x * 91.25) / doubling_days)
    
    # 10% faster and 90% slower curves
    fast_y = base_rate * np.power(2, (x * 91.25) / (doubling_days * 0.7))
    slow_y = base_rate * np.power(2, (x * 91.25) / (doubling_days * 1.5))

    ax.plot(x, fast_y, 'b--', linewidth=2, label='10% Earliest')
    ax.plot(x, median_y, 'b-', linewidth=2, label='Median')
    ax.plot(x, slow_y, 'b--', linewidth=2, label='90% Latest')
    
    # AGI threshold markers
    agi_threshold = agi_task_length
    
    med_cross = np.argmax(median_y >= agi_threshold) if np.any(median_y >= agi_threshold) else len(x)-1
    fast_cross = np.argmax(fast_y >= agi_threshold) if np.any(fast_y >= agi_threshold) else len(x)-1
    slow_cross = np.argmax(slow_y >= agi_threshold) if np.any(slow_y >= agi_threshold) else len(x)-1
    
    if med_cross < len(x)-1: ax.plot(x[med_cross], median_y[med_cross], 'rx', markersize=10)
    if fast_cross < len(x)-1: ax.plot(x[fast_cross], fast_y[fast_cross], 'rx', markersize=10)
    if slow_cross < len(x)-1: ax.plot(x[slow_cross], slow_y[slow_cross], 'rx', markersize=10)
    
    if med_cross == len(x)-1: ax.plot(x[med_cross], median_y[med_cross], 'ko', markersize=10)
    if fast_cross == len(x)-1: ax.plot(x[fast_cross], fast_y[fast_cross], 'ko', markersize=10)
    if slow_cross == len(x)-1: ax.plot(x[slow_cross], slow_y[slow_cross], 'ko', markersize=10)
    
    # Dummy plots for legend
    ax.plot([], [], 'rx', markersize=10, label='AGI reached')
    ax.plot([], [], 'ko', markersize=10, label='AGI not by EOY2035')
    
    ax.set_yscale('log', base=2)
    
    # Y-axis formatting
    min_y_value = min(min(fast_y), min(median_y), min(slow_y))
    max_y_value = max(max(fast_y), max(median_y), max(slow_y), agi_threshold*1.5)
    
    min_power = max(int(np.floor(np.log2(min_y_value))), -10)
    max_power = min(int(np.ceil(np.log2(max_y_value))), 20)
    
    major_ticks = [2**k for k in range(min_power, max_power+1, 2)]
    minor_ticks = [2**k for k in range(min_power, max_power+1) if k % 2 == 1]
    
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    ax.grid(True, axis='y', which='major', linestyle='-', alpha=0.3)
    ax.grid(True, axis='y', which='minor', linestyle='--', alpha=0.15)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    def format_ticks(x, pos):
        days = x / 8
        if days >= 365: return f"{days/365:.1f} years"
        if days >= 30: return f"{days/30:.1f} months"
        if days >= 7: return f"{days/7:.1f} weeks"
        if days >= 1: return f"{days:.1f} days"
        return f"{x:.1f} hrs"
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    
    # Reference lines
    reference_tasks = [
        (0.25, "15 min task", "gray", "dotted"),
        (1, "1 hour task", "gray", "dotted"),
        (8, "1 day task", "gray", "dashed"),
        (40, "1 week task", "gray", "dashed"),
        (160, "1 month task", "gray", "solid"),
        (2000, "1 year task", "gray", "solid")
    ]
    
    for hours, label, color, style in reference_tasks:
        if min_y_value <= hours <= max_y_value * 1.2:
            ax.axhline(y=hours, color=color, linestyle=style, alpha=0.5)
            ax.text(len(x)*1.02, hours, label, va='center', fontsize=8, alpha=0.7)
    
    # AGI threshold line
    ax.axhline(y=agi_threshold, color='red', linestyle='-.', alpha=0.7)
    ax.text(len(x)*1.02, agi_threshold, f"AGI threshold: {agi_threshold:.0f} hrs", 
           va='center', color='red', fontsize=9, fontweight='bold')
    
    # X-axis formatting
    quarter_labels = [f"{2025 + q // 4}Q{q % 4 + 1}" for q in quarters]
    ax.set_xticks(quarters[::4])
    ax.set_xticklabels(quarter_labels[::4], rotation=45, ha='right')
    ax.set_xticks(quarters, minor=True)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Task Length (work hours)')
    ax.set_title('Projected Task Length Growth Over Time', fontsize=14)
    ax.legend(loc='upper left')
    
    ax.set_ylim(min_y_value * 0.9, max_y_value * 1.1)
    max_x = min(max(20, med_cross + 8, fast_cross + 4, slow_cross + 12), len(x) - 1)
    ax.set_xlim(-1, max_x)
    
    plt.tight_layout()
    return fig

def plot_timeline(milestones, reference_date):
    """Plots the AI capability timeline."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    base_date_num = mdates.date2num(reference_date)
    milestone_dates = [m["date"] for m in milestones]
    milestone_date_nums = [mdates.date2num(d) if d is not None else base_date_num for d in milestone_dates]
    
    y_positions = list(range(len(milestones), 0, -1))
    
    ax.plot(milestone_date_nums, y_positions, 'o-', markersize=12, color='#4F8DFD', linewidth=2)
    
    for i, m in enumerate(milestones):
        if "early_date" in m and "late_date" in m:
            early_num = mdates.date2num(m["early_date"])
            late_num = mdates.date2num(m["late_date"])
            ax.plot([early_num, late_num], [y_positions[i], y_positions[i]], 'b-', linewidth=3, alpha=0.3)
            ax.plot([early_num], [y_positions[i]], 'b|', markersize=10)
            ax.plot([late_num], [y_positions[i]], 'b|', markersize=10)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{m['name']} ({m['hours']}h)" for m in milestones])
    
    date_format = mdates.DateFormatter('%b %Y')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.subplots_adjust(bottom=0.2)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Projected Date')
    ax.set_title('AI Capability Timeline Based on Model Parameters')
    
    plt.tight_layout()
    return fig

def plot_tornado(labels, low_yrs, high_yrs):
    """Plots the tornado plot for sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (l, lo, hi) in enumerate(zip(labels, low_yrs, high_yrs)):
        ax.plot([lo, hi], [i, i], 'o-', color='#4F8DFD', lw=6, alpha=0.7)
        ax.plot([lo], [i], 'o', color='red', label='Low value' if i==0 else "")
        ax.plot([hi], [i], 'o', color='green', label='High value' if i==0 else "")
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Median AGI Year")
    ax.set_title("Tornado Plot: Sensitivity of Median AGI Year")
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend()
    return fig
