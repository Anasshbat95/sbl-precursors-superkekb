"""
last_change_before_sbl.py - For every SBL event, find the last row before it in a
block, and which feature was the last one to actually change value before the
event (not just the last row overall - many EPICS-style channels only report a
new value when it changes, so most columns are unchanged/NaN most of the time).

Functions:
    last_row_before_sbl             Find the single last row before each SBL event.
    last_feature_change_before_sbl  Find the last changed row for every feature
                                    before each SBL event, plus the feature that
                                    changed most recently overall.
    plot_feature_sbl_stats          Aggregate and plot statistics by last-changed
                                    feature.
    plot_sbl_distance_histogram     Plot a histogram of distances before SBL events.

Usage:
    from last_change_before_sbl import (
        last_row_before_sbl,
        last_feature_change_before_sbl,
        plot_feature_sbl_stats,
        plot_sbl_distance_histogram,
    )

    sbl_times = SBL_timestamps_Germany["timestamp"]
    last_rows = last_row_before_sbl(BEAM_clean, "timestamp", sbl_times)
    changes = last_feature_change_before_sbl(
        BEAM_clean, "timestamp", feature_cols, sbl_times
    )

    stats_df, _ = plot_feature_sbl_stats(changes)
    distance_stats, _ = plot_sbl_distance_histogram(changes)
"""
import numpy as np
import pandas as pd


def last_row_before_sbl(df: pd.DataFrame, ts_col: str, sbl_times) -> pd.DataFrame:
    """For every SBL event: index (in the original df), timestamp and distance (seconds) of the
    last row strictly before that SBL. One row per event, in the order of sbl_times."""
    ts = pd.to_datetime(df[ts_col]).to_numpy()
    order = np.argsort(ts)
    ts_sorted = ts[order]
    orig_index = df.index.to_numpy()[order]

    rows = []
    for sbl in pd.DatetimeIndex(pd.to_datetime(sbl_times)):
        pos = np.searchsorted(ts_sorted, np.datetime64(sbl), side="left") - 1
        if pos < 0:
            rows.append((sbl, np.nan, pd.NaT, np.nan))
            continue
        row_index = orig_index[pos]
        row_time = ts_sorted[pos]
        dist_sec = (sbl - pd.Timestamp(row_time)).total_seconds()
        rows.append((sbl, row_index, pd.Timestamp(row_time), dist_sec))

    return pd.DataFrame(rows, columns=["sbl_time", "row_index", "row_timestamp", "distance_sec"])


def last_feature_change_before_sbl(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times) -> pd.DataFrame:
    """For every SBL event and every feature: the row index, timestamp and distance (seconds) of
    the last row before the event where that feature's value actually changed (differs from the
    row before it; NaN-to-NaN does not count as a change, a value appearing after NaN does).

    Also adds, per event, 'last_changed_feature' = the feature whose change was closest to the
    SBL (i.e. the one that "moved last" before the event) and 'last_changed_distance_sec'.

    Returns one row per (event, feature) pair, so group by 'sbl_time' to see all features for
    one event, or filter to the 'last_changed_feature' row per event for the single closest one.
    """
    df_sorted = df.sort_values(ts_col).reset_index()  # keeps the original index in column 'index'
    ts = pd.to_datetime(df_sorted[ts_col]).to_numpy()
    orig_index = df_sorted["index"].to_numpy()

    # for each feature, the positions where its value changed from the previous row
    changed_positions = {}
    for col in feature_cols:
        vals = df_sorted[col]
        same_as_prev = vals.eq(vals.shift()) | (vals.isna() & vals.shift().isna())
        changed_positions[col] = np.where(~same_as_prev.to_numpy())[0]
        changed_positions[col] = changed_positions[col][changed_positions[col] > 0]  # drop the first row itself

    rows = []
    for sbl in pd.DatetimeIndex(pd.to_datetime(sbl_times)):
        sbl_pos = np.searchsorted(ts, np.datetime64(sbl), side="left")
        per_feature = []
        for col in feature_cols:
            pos_arr = changed_positions[col]
            idx_before = pos_arr[pos_arr < sbl_pos]
            if len(idx_before) == 0:
                per_feature.append((sbl, col, np.nan, pd.NaT, np.nan))
                continue
            p = idx_before[-1]
            row_index = orig_index[p]
            row_time = pd.Timestamp(ts[p])
            dist_sec = (sbl - row_time).total_seconds()
            per_feature.append((sbl, col, row_index, row_time, dist_sec))

        event_df = pd.DataFrame(per_feature, columns=["sbl_time", "feature", "row_index", "row_timestamp",
                                                       "distance_sec"])
        valid = event_df.dropna(subset=["distance_sec"])
        if valid.empty:
            event_df["last_changed_feature"] = np.nan
            event_df["last_changed_distance_sec"] = np.nan
        else:
            closest = valid.loc[valid["distance_sec"].idxmin()]
            event_df["last_changed_feature"] = closest["feature"]
            event_df["last_changed_distance_sec"] = closest["distance_sec"]
        rows.append(event_df)

    return pd.concat(rows, ignore_index=True)


import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_sbl_stats(
    changes_df, title='mean time before SBL per feature',
    feature_col="last_changed_feature",
    distance_col="last_changed_distance_sec",
    event_col="sbl_time",
    figsize=(10, 5),
    show_plot=True,
):
    """Group SBL events by feature and visualize frequency (count)

    and average time before the SBL (mean) in a dual-axis plot.

    Parameters:
    -----------
    changes_df : pd.DataFrame
        Input DataFrame containing event and feature data.
    feature_col : str, optional
        Name of the feature column (default: 'last_changed_feature').
    distance_col : str, optional
        Name of the time-distance column in seconds (default:
        'last_changed_distance_sec').
    event_col : str, optional
        Name of the unique SBL event ID/time column (default: 'sbl_time').
    figsize : tuple, optional
        Plot size as (width, height) in inches (default: (10, 5)).
    show_plot : bool, optional
        Whether to call plt.show() immediately (default: True).

    Returns:
    --------
    feature_stats : pd.DataFrame
        Aggregated statistics table (count, mean, median, std).
    fig, ax1, ax2 : matplotlib Figure und Axes Objekte
    """
    # 1. Remove duplicates, keeping one row per SBL event.
    closest = changes_df.drop_duplicates(event_col)

    # 2. Group and aggregate.
    feature_stats = (
        closest.groupby(feature_col)[distance_col]
        .agg(
            count="count",
            mean_sec="mean",
            median_sec="median",
            std_sec="std",
        )
        .sort_values(by="count", ascending=False)
        .reset_index()
    )

    # 3. Create the plot.
    fig, ax1 = plt.subplots(figsize=figsize)

    # Primary axis: frequency (bars).
    bar_color = "#2badd9"
    bars = ax1.bar(
        feature_stats[feature_col],
        feature_stats["count"],
        color=bar_color,
        alpha=0.8,
        edgecolor="black",
        width=0.5,
    )
    ax1.set_xlabel("Feature (last_changed_feature)", fontweight="bold")
    ax1.set_ylabel("Frequency (Count)", color=bar_color, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.tick_params(axis="x", labelrotation=45)

    # Display values above the bars.
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Secondary axis: average time (line and points).
    ax2 = ax1.twinx()
    line_color = "#d9534f"

    ax2.plot(
        feature_stats[feature_col],
        feature_stats["mean_sec"],
        color=line_color,
        marker="o",
        linewidth=2,
        markersize=8,
    )

    # Add labels next to the points.
    for i, value in enumerate(feature_stats["mean_sec"]):
        ax2.annotate(
            f"{value:.1f}s",
            (i + 0.2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="left",
            color=line_color,
            fontweight="bold",
            fontsize=9,
        )

    ax2.set_ylabel(
        "Mean time before SBL [s]",
        color=line_color,
        fontweight="bold",
    )
    ax2.tick_params(axis="y", labelcolor=line_color)

    # Title and layout.
    plt.title(
        title, pad=15
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Adjust the y-axis limits to make room for labels.
    if not feature_stats["count"].empty:
        ax1.set_ylim(0, feature_stats["count"].max() * 1.15)

    plt.tight_layout()

    if show_plot:
        plt.show()

    return feature_stats, (fig, ax1, ax2)




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sbl_distance_histogram(
    changes_df,
    title="Last Change before SBL (Beam)",
    distance_col="last_changed_distance_sec",
    event_col="sbl_time",
    bins=30,
    figsize=(10, 6),
    show_plot=True,
):
    """Plots a histogram of the time distances before SBL events, highlighting the

    mean, median, and most frequent range (peak bin).

    Parameters:
    -----------
    changes_df : pd.DataFrame
        DataFrame containing the change events.
    distance_col : str, optional
        Name of the column with time distance in seconds (default is
        'last_changed_distance_sec').
    event_col : str, optional
        Name of the column with unique SBL event timestamps (default is
        'sbl_time').
    bins : int, optional
        Number of histogram bins (default is 30).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default is (10, 6)).
    show_plot : bool, optional
        Whether to automatically display the plot with plt.show() (default is
        True).

    Returns:
    --------
    stats_dict : dict
        Dictionary containing summary statistics (mean, median, count,
        peak_range).
    fig, ax : matplotlib Figure and Axes objects
    """
    # Keep one row per SBL event.
    closest = changes_df.drop_duplicates(event_col)

    # Calculate summary statistics.
    data = closest[distance_col].dropna()

    if data.empty:
        print(f"No data available in column '{distance_col}'.")
        return {}, (None, None)

    mean_val = data.mean()
    median_val = data.median()

    fig, ax = plt.subplots(figsize=figsize)

    # Draw the histogram.
    counts, bin_edges, _ = ax.hist(
        data,
        bins=bins,
        color="#2badd9",
        edgecolor="black",
        alpha=0.75,
    )

    # Highlight the most frequent bin.
    max_idx = int(np.argmax(np.asarray(counts)))
    ax.patches[max_idx].set_facecolor("#ff7f0e")

    peak_start = bin_edges[max_idx]
    peak_end = bin_edges[max_idx + 1]

    # Draw mean and median lines.
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.1f} s",
    )

    ax.axvline(
        median_val,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_val:.1f} s",
    )

    # Configure the plot.
    ax.set_xlabel("Seconds before SBL")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add a statistics box.
    stats_str = (
        f"Number of events: {len(data)}\n"
        f"Mean: {mean_val:.1f} s\n"
        f"Median: {median_val:.1f} s\n"
        f"Most frequent range: {peak_start:.1f}–{peak_end:.1f} s"
    )

    ax.text(
        0.95,
        0.95,
        stats_str,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.8,
        ),
    )

    ax.legend(loc="upper left")
    plt.tight_layout()

    if show_plot:
        plt.show()

    # Create a dictionary with statistics for programmatic use.
    stats_dict = {
        "count": len(data),
        "mean": mean_val,
        "median": median_val,
        "peak_range_start": peak_start,
        "peak_range_end": peak_end,
    }

    return stats_dict, (fig, ax)