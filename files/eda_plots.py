"""
eda_plots.py - Exploratory plots for the SBL project.

Usage:
    from eda_plots import (
        plot_timeseries, plot_timeseries_multi_block,
        plot_feature_distributions, plot_missing_value_overview,
        plot_sbl_events_per_day, plot_sbl_centered,
        plot_sbl_overlay, plot_sbl_summary,
    )

    # ALL events in one panel per feature (one thin line per event) ...
    figs = plot_sbl_overlay(blocks, sbl_times, before="30min", after="30min")
    # ... and the same panels, but median (+ IQR band) over the events instead of every line:
    figs, table = plot_sbl_summary(blocks, sbl_times, before="30min", after="30min",
                                   statistic="median", normalize="baseline")

    # one SBL event in the middle (red line), window before/after, one panel per block:
    fig, medians = plot_sbl_centered(
        {"beam": ds_beam, "vacuum": ds_vacuum, "background": ds_background},
        sbl_times, sbl=0, before="30min", after="30min",
    )
"""
import warnings
from itertools import cycle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler


def plot_timeseries(df, start_date=None, end_date=None, title_prefix="Data", n_cols=1,
                    sbl_datetimes=None, y_zoom=True, y_margin=0.05,
                    y_quantile_low=0.01, y_quantile_high=0.99):
    """Plot every numeric column of a single block, one panel per column, with optional SBL markers."""
    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
    start_dt = df_plot["timestamp"].min() if start_date is None else pd.to_datetime(start_date)
    end_dt = df_plot["timestamp"].max() if end_date is None else pd.to_datetime(end_date)

    df_f = df_plot[(df_plot["timestamp"] >= start_dt) & (df_plot["timestamp"] <= end_dt)].sort_values("timestamp")
    if df_f.empty:
        print(f"No data between {start_dt} and {end_dt}.")
        return

    cols = [c for c in df_f.columns if c != "timestamp"]
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18 if n_rows > 1 else 10, 3.5 * n_rows if n_rows > 1 else 4),
                             sharex=True)
    axes = np.ravel(axes)

    sbl_in_window = []
    if sbl_datetimes is not None:
        s = pd.to_datetime(pd.Series(sbl_datetimes))
        sbl_in_window = s[(s >= start_dt) & (s <= end_dt)].tolist()

    for ax, col in zip(axes, cols):
        data = df_f[col].dropna()
        ax.plot(df_f["timestamp"], df_f[col], linewidth=0.8, alpha=0.9)
        if y_zoom and not data.empty:
            y_lo, y_hi = data.quantile(y_quantile_low), data.quantile(y_quantile_high)
            margin = (y_hi - y_lo) * y_margin if y_hi != y_lo else abs(y_lo) * 0.05 + 1e-9
            ax.set_ylim(y_lo - margin, y_hi + margin)
        for t in sbl_in_window:
            ax.axvline(t, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="SBL event")
        ax.set_xlim(start_dt, end_dt)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    if sbl_in_window:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend([handles[0]], [labels[0]], loc="upper right", bbox_to_anchor=(0.99, 0.99))
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(f"{title_prefix} - {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_timeseries_multi_block(dfs, start_date=None, end_date=None, title_prefix="Comparison",
                                sbl_datetimes=None, y_zoom=True, y_margin=0.05,
                                y_quantile_low=0.01, y_quantile_high=0.99, n_cols=1,
                                sbl_color="lightcoral", sbl_alpha=0.4, dpi=150, colormap="tab20",
                                log_scale=False):
    """Plot several blocks side by side (one panel per block, all its columns overlaid), with SBL markers."""
    if isinstance(dfs, pd.DataFrame):
        dfs = {title_prefix: dfs}
    processed = {k: v.assign(timestamp=pd.to_datetime(v["timestamp"])) for k, v in dfs.items()}
    start_dt = min(d["timestamp"].min() for d in processed.values()) if start_date is None else pd.to_datetime(start_date)
    end_dt = max(d["timestamp"].max() for d in processed.values()) if end_date is None else pd.to_datetime(end_date)

    filtered = {k: d[(d["timestamp"] >= start_dt) & (d["timestamp"] <= end_dt)].sort_values("timestamp")
               for k, d in processed.items()}
    filtered = {k: d for k, d in filtered.items() if not d.empty}
    if not filtered:
        print(f"No data between {start_dt} and {end_dt}.")
        return

    labels = list(filtered.keys())
    n_rows = (len(labels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 4 * n_rows), squeeze=False, dpi=dpi)
    axes = axes.ravel()

    sbl_in_window = []
    if sbl_datetimes is not None:
        s = pd.to_datetime(pd.Series(sbl_datetimes))
        sbl_in_window = s[(s >= start_dt) & (s <= end_dt)].tolist()

    for ax, label in zip(axes, labels):
        f = filtered[label]
        feature_cols = [c for c in f.columns if c != "timestamp"]
        colors = plt.get_cmap(colormap, max(len(feature_cols), 1))
        y_lo_all, y_hi_all = [], []
        for i, col in enumerate(feature_cols):
            data = f[col].where(f[col] > 0).dropna() if log_scale else f[col].dropna()
            values = f[col].where(f[col] > 0) if log_scale else f[col]
            ax.plot(f["timestamp"], values, linewidth=1.0, alpha=0.9, color=colors(i), label=col)
            if y_zoom and not data.empty:
                y_lo_all.append(data.quantile(y_quantile_low))
                y_hi_all.append(data.quantile(y_quantile_high))
        if y_zoom and y_lo_all:
            y_lo, y_hi = min(y_lo_all), max(y_hi_all)
            margin = (y_hi - y_lo) * y_margin if y_hi != y_lo else abs(y_lo) * 0.05 + 1e-9
            ax.set_ylim(y_lo - margin, y_hi + margin)
        if log_scale:
            ax.set_yscale("log")
        for t in sbl_in_window:
            ax.axvline(t, color=sbl_color, linestyle="--", linewidth=1.0, alpha=sbl_alpha)
        ax.set_xlim(start_dt, end_dt)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    for ax in axes[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(f"{title_prefix} - {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, dataset_name, distance_threshold=300, scale_data=True, ax=None,
                               show_peak_value=False):
    """KDE of every numeric feature, restricted to rows within `distance_threshold` seconds of an SBL.

    Requires the df to already carry 'dist_to_sbl_sec' (see sbl_labeling.add_sbl_features_and_label,
    which calls the equivalent column 'min_abs_dist' -- rename/alias it to 'dist_to_sbl_sec' beforehand
    if you use this function directly).
    """
    dist_col = "dist_to_sbl_sec" if "dist_to_sbl_sec" in df.columns else "min_abs_dist"
    df_f = df[df[dist_col] <= distance_threshold]
    exclude_cols = ["target", "dist_to_sbl_sec", "min_abs_dist", "sbl_diff_sec", "sbl_index"]
    df_num = df_f.select_dtypes(include=["float64", "int64"]).drop(columns=exclude_cols, errors="ignore").dropna()

    if df_num.empty:
        print(f"No suitable numeric features found in {dataset_name}.")
        if ax is not None:
            ax.set_visible(False)
        return

    if scale_data:
        df_plot = pd.DataFrame(StandardScaler().fit_transform(df_num), columns=df_num.columns)
        x_label = "Z-score"
    else:
        df_plot, x_label = df_num, "Value"

    peak_heights = {}
    for col in df_plot.columns:
        vals = df_plot[col].values
        peak_heights[col] = gaussian_kde(vals)(np.linspace(vals.min(), vals.max(), 200)).max() if np.std(vals) > 0 else 0
    sorted_features = sorted(peak_heights, key=peak_heights.get, reverse=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    palette = cycle(sns.color_palette("tab20", n_colors=min(len(sorted_features), 20)))
    linestyles = ["-", "--", "-.", ":"]
    handles, labels = [], []
    for i, col in enumerate(sorted_features):
        line = sns.kdeplot(data=df_plot, x=col, ax=ax, color=next(palette),
                           linestyle=linestyles[i // 20 % len(linestyles)], linewidth=1.0, common_norm=False)
        label = f"{col} ({peak_heights[col]:.2f})" if show_peak_value else col
        handles.append(line.lines[-1])
        labels.append(label)

    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset_name} (dist <= {distance_threshold}s)", fontsize=8, pad=4, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.legend(handles=handles, labels=labels, fontsize=5, title="Feature (sorted by peak height)",
             title_fontsize=6, framealpha=0.6, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_feature_distributions_grid(datasets: dict, distance_threshold=300, save_path=None):
    """plot_feature_distributions for several blocks at once, arranged in a grid."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    for ax, (name, df) in zip(axes, datasets.items()):
        plot_feature_distributions(df, name, distance_threshold=distance_threshold, ax=ax, show_peak_value=True)
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_missing_value_overview(blocks: dict):
    """Bar chart per block: number of non-null values per column (quick data-quality check)."""
    n = len(blocks)
    n_rows = (n + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
    axes = np.ravel(axes)
    for ax, (name, df) in zip(axes, blocks.items()):
        df.drop(columns=["timestamp"], errors="ignore").notna().sum().plot(ax=ax, kind="barh", title=name.upper())
    for ax in axes[len(blocks):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_sbl_events_per_day(sbl_times):
    """Bar chart: number of SBL events per calendar day."""
    counts = pd.to_datetime(pd.Series(sbl_times)).dt.to_period("D").value_counts().sort_index()
    counts.plot(kind="bar", figsize=(18, 6), title="Distribution of SBL events over time",
               color="steelblue", edgecolor="black")
    plt.xlabel("Day")
    plt.ylabel("Number of SBL events")
    plt.tight_layout()
    plt.show()


def _window_seconds(value, block):
    """before/after may be one string for all blocks or a {block: string} dict."""
    v = value[block] if isinstance(value, dict) else value
    return pd.Timedelta(v).total_seconds()


def _symlog_threshold(values):
    pos = np.abs(values[np.isfinite(values) & (values != 0)])
    return float(pos.min()) if len(pos) else 1.0


def plot_sbl_centered(blocks: dict, sbl_times, sbl=0, before="30min", after="30min",
                      ts_col="timestamp", features=None, show_median=True, yscale="linear",
                      x_unit="auto", n_cols=1, dpi=300, colormap="tab20", title=None):
    """Plot ONE SBL event: x = time relative to the SBL (red line at 0), one panel per block, every
    numeric column of the block as a line, legend outside the plot area.

    blocks:   {"beam": df, "vacuum": df, ...} - every df gets its own panel.
    sbl_times: all SBL timestamps (Series / DatetimeIndex / single-column DataFrame).
    sbl:      which event: an integer position in the chronologically sorted SBL list, or a timestamp.
    before, after: window before / after the SBL, e.g. "30min". One value for all blocks, or a dict
              {"beam": "10min", "vacuum": "2h"} to choose the window per block. If before != after
              the SBL is not in the middle of the panel.
    features: optional {"beam": ["her_current", "ler_current"], ...} to plot only some columns.
    show_median: dashed horizontal line per feature at the median of its values BEFORE the SBL (left
              of the red line) and AFTER the SBL (right of it) - a jump between the two is the change
              at the SBL. Medians are also returned as a table.
    yscale:   "linear", "log" or "symlog" (symlog also shows zeros, e.g. current after an abort).
    x_unit:   "auto" (default: seconds / minutes / hours chosen per panel from its window), or force
              "s", "min" or "h" for all panels.
    dpi / colormap: figure resolution (default 300) and colormap (default "tab20").

    Returns (fig, table) - table has median_before / median_after / n_before / n_after per feature.
    """
    sbl_s = sbl_times.iloc[:, 0] if isinstance(sbl_times, pd.DataFrame) else sbl_times
    sbl_s = pd.to_datetime(pd.Series(sbl_s)).sort_values().reset_index(drop=True)
    if isinstance(sbl, (int, np.integer)):
        if not 0 <= sbl < len(sbl_s):
            raise IndexError(f"sbl={sbl} is out of range, valid positions are 0..{len(sbl_s) - 1}.")
        t, sbl_label = sbl_s.iloc[sbl], f"index {sbl}"
    else:
        t, sbl_label = pd.Timestamp(sbl), "given timestamp"

    names = list(blocks)

    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 3.3 * n_rows), dpi=dpi,
                             squeeze=False, layout="constrained")
    axes = axes.ravel()
    cmap = plt.get_cmap(colormap)
    n_colors = cmap.N if cmap.N <= 20 else 20
    linestyles = ["-", "--", "-.", ":"]

    rows = []
    for ax, name in zip(axes, names):
        df = blocks[name]
        ts = pd.to_datetime(df[ts_col])
        if (ts.dt.tz is None) != (t.tzinfo is None):
            raise ValueError(f"Block '{name}' and the SBL time differ in time-zone awareness - convert "
                             "both to the same (e.g. both tz-naive UTC) before plotting.")
        b_s, a_s = _window_seconds(before, name), _window_seconds(after, name)
        unit = x_unit            # "auto" = pick the unit that suits THIS panel's window
        if unit == "auto":
            w = max(b_s, a_s)
            unit = "h" if w >= 2 * 3600 else "min" if w >= 180 else "s"
        div = {"s": 1.0, "min": 60.0, "h": 3600.0}[unit]
        mask = (ts >= t - pd.Timedelta(seconds=b_s)) & (ts <= t + pd.Timedelta(seconds=a_s))
        win = df.loc[mask].copy()
        x = ((ts[mask] - t).dt.total_seconds() / div).to_numpy()
        cols = (features or {}).get(name) or [c for c in df.select_dtypes("number").columns if c != ts_col]

        ax.set_title(f"{name}   (-{before[name] if isinstance(before, dict) else before} / "
                     f"+{after[name] if isinstance(after, dict) else after})", fontsize=10, fontweight="bold")
        ax.set_xlim(-b_s / div, a_s / div)
        ax.set_xlabel(f"time relative to SBL ({unit})")
        ax.axvline(0, color="red", lw=1.6, zorder=5, label="SBL")
        ax.grid(alpha=0.3)

        if win.empty:
            ax.text(0.5, 0.5, "no data in this window", transform=ax.transAxes, ha="center", va="center",
                    color="gray")
            continue

        order = np.argsort(x)
        marker = "." if len(win) < 400 else None
        all_vals = []
        for i, c in enumerate(cols):
            y = win[c].to_numpy(float)
            color = cmap(i % n_colors)
            ax.plot(x[order], y[order], color=color, lw=0.9, marker=marker, ms=3,
                    ls=linestyles[(i // n_colors) % len(linestyles)], label=c)
            all_vals.append(y)
            yb, ya = y[x < 0], y[x >= 0]
            med_b = np.nanmedian(yb) if np.isfinite(yb).any() else np.nan
            med_a = np.nanmedian(ya) if np.isfinite(ya).any() else np.nan
            rows.append(dict(block=name, feature=c, median_before=med_b, median_after=med_a,
                             n_before=int(np.isfinite(yb).sum()), n_after=int(np.isfinite(ya).sum())))
            if show_median:
                if np.isfinite(med_b):
                    ax.hlines(med_b, -b_s / div, 0, colors=[color], linestyles="--", lw=0.9, alpha=0.9)
                if np.isfinite(med_a):
                    ax.hlines(med_a, 0, a_s / div, colors=[color], linestyles="--", lw=0.9, alpha=0.9)

        if yscale == "log":
            ax.set_yscale("log", nonpositive="mask")
        elif yscale == "symlog":
            ax.set_yscale("symlog", linthresh=_symlog_threshold(np.concatenate(all_vals)))

        handles, labels = ax.get_legend_handles_labels()
        if show_median:
            handles.append(plt.Line2D([], [], color="gray", ls="--", lw=0.9))
            labels.append("median before / after")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
                  borderaxespad=0, frameon=False)

    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title or f"SBL at {t}  ({sbl_label})", fontsize=12)
    plt.show()
    return fig, pd.DataFrame(rows)


# ====================================================================================================
# all SBL events on one time axis: overlay of every event, and summary (median / mean) over the events
# ====================================================================================================
_UNIT_DIV = {"s": 1.0, "min": 60.0, "h": 3600.0}


def _auto_unit(window_s, x_unit):
    if x_unit != "auto":
        return x_unit
    return "h" if window_s >= 2 * 3600 else "min" if window_s >= 180 else "s"


def _sbl_sorted(sbl_times):
    s_ = sbl_times.iloc[:, 0] if isinstance(sbl_times, pd.DataFrame) else sbl_times
    return pd.to_datetime(pd.Series(s_)).sort_values().reset_index(drop=True)


def _check_tz(ts, sbl_s, name):
    if (ts.dt.tz is None) != (sbl_s.dt.tz is None):
        raise ValueError(f"Block '{name}' and the SBL times differ in time-zone awareness - convert both "
                         "to the same (e.g. both tz-naive UTC) first.")


def _event_windows(df, ts_col, sbl_s, indices, b_s, a_s, cols):
    """Cut the window [SBL - b_s, SBL + a_s] out of one block for every selected event.
    Returns a list of (event_index, x_seconds_relative_to_SBL, values[rows x features])."""
    d = df.sort_values(ts_col)
    ts = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
    vals = d[cols].to_numpy(float)
    out = []
    for i in indices:
        t = np.datetime64(sbl_s.iloc[i], "ns").astype("int64")
        lo = np.searchsorted(ts, t - int(b_s * 1e9), side="left")
        hi = np.searchsorted(ts, t + int(a_s * 1e9), side="right")
        if hi > lo:
            out.append((i, (ts[lo:hi] - t) / 1e9, vals[lo:hi]))
    return out


def _block_setup(blocks, sbl_times, before, after, sbl_indices, features, ts_col):
    sbl_s = _sbl_sorted(sbl_times)
    indices = list(range(len(sbl_s))) if sbl_indices is None else [i for i in sbl_indices if 0 <= i < len(sbl_s)]
    if not indices:
        raise ValueError("No valid sbl_indices.")
    setup = {}
    for name, df in blocks.items():
        _check_tz(pd.to_datetime(df[ts_col]), sbl_s, name)
        cols = (features or {}).get(name) or [c for c in df.select_dtypes("number").columns if c != ts_col]
        b_s, a_s = _window_seconds(before, name), _window_seconds(after, name)
        setup[name] = dict(cols=cols, b_s=b_s, a_s=a_s,
                           events=_event_windows(df, ts_col, sbl_s, indices, b_s, a_s, cols))
    return sbl_s, indices, setup


def _grid(n_panels, n_cols):
    n_cols = max(1, min(n_cols, n_panels))
    return int(np.ceil(n_panels / n_cols)), n_cols


def _outside_legend(fig, handles, labels):
    try:
        fig.legend(handles, labels, loc="outside right upper", fontsize=8, frameon=False)
    except ValueError:                                   # older matplotlib: no "outside" locations
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)


def plot_sbl_overlay(blocks: dict, sbl_times, before="30min", after="30min", sbl_indices=None,
                     features=None, ts_col="timestamp", yscale="linear", y_quantiles=(0.5, 99.5),
                     n_cols=3, dpi=300, colormap="tab20", x_unit="auto") -> dict:
    """ALL selected SBL events in ONE panel per feature: x = time relative to the SBL (red line at 0),
    one thin transparent line per event at the real sample times (no interpolation). One figure per block.

    blocks: {"beam": df, ...}.  sbl_indices: positions in the sorted SBL list (None = all events).
    before / after: window around the SBL, one string for all blocks or a {block: string} dict.
    yscale: "linear", "log" or "symlog".  y_quantiles: the y-axis is limited to these percentiles of
    the pooled values (+5% margin) so a single extreme event cannot flatten the rest; None = full range.
    The line style is the same before and after the SBL. Legend sits outside the plot (right).
    Returns {block: figure}."""
    sbl_s, indices, setup = _block_setup(blocks, sbl_times, before, after, sbl_indices, features, ts_col)
    cmap = plt.get_cmap(colormap)
    n_colors = min(cmap.N, 20)
    figs = {}
    for name, st in setup.items():
        cols, events = st["cols"], st["events"]
        if not events:
            print(f"[{name}] no data in any event window - skipped.")
            continue
        unit = _auto_unit(max(st["b_s"], st["a_s"]), x_unit)
        div = _UNIT_DIV[unit]
        n_rows, ncol = _grid(len(cols), n_cols)
        fig, axes = plt.subplots(n_rows, ncol, figsize=(4.6 * ncol + 1.8, 2.9 * n_rows), dpi=dpi,
                                 squeeze=False, layout="constrained")
        axes = axes.ravel()
        alpha = float(np.clip(12 / len(events), 0.18, 0.65))
        for j, (ax, c) in enumerate(zip(axes, cols)):
            color = cmap(j % n_colors)
            segs, pooled = [], []
            for _, x, Y in events:
                y = Y[:, j]
                ok = np.isfinite(y)
                if ok.sum() >= 2:
                    segs.append(np.column_stack([x[ok] / div, y[ok]]))
                pooled.append(y[ok])
            if segs:
                ax.add_collection(LineCollection(segs, colors=[color], linewidths=1.0, alpha=alpha))
            ax.set_xlim(-st["b_s"] / div, st["a_s"] / div)
            allv = np.concatenate(pooled) if pooled else np.array([])
            if len(allv):
                lo, hi = (np.percentile(allv, y_quantiles) if y_quantiles else (allv.min(), allv.max()))
                m = (hi - lo) * 0.05 if hi > lo else (abs(lo) * 0.05 + 1e-9)
                ax.set_ylim(lo - m, hi + m)
                if yscale == "log":
                    ax.set_yscale("log", nonpositive="mask")
                elif yscale == "symlog":
                    ax.set_yscale("symlog", linthresh=_symlog_threshold(allv))
            ax.axvline(0, color="red", lw=1.4, zorder=5)
            ax.set_title(c, fontsize=9, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(f"time relative to SBL ({unit})", fontsize=8)
        for ax in axes[len(cols):]:
            ax.set_visible(False)
        handles = [plt.Line2D([], [], color="red", lw=1.4),
                   plt.Line2D([], [], color="gray", lw=1.0, alpha=0.7)]
        _outside_legend(fig, handles, ["SBL", f"one line per event ({len(events)} events with data)"])
        fig.suptitle(f"{name}: all SBL events overlaid  (-{before[name] if isinstance(before, dict) else before}"
                     f" / +{after[name] if isinstance(after, dict) else after})", fontsize=11)
        figs[name] = fig
    plt.show()
    return figs


def _bin_events(events, n_cols_feat, b_s, a_s, bin_s):
    """Median per event and time bin. Returns (edges_s, M[feature] = events x bins array, event_ids)."""
    n_neg = int(np.ceil(b_s / bin_s))
    n_pos = int(np.ceil(a_s / bin_s))
    edges = np.arange(-n_neg, n_pos + 1) * bin_s          # a bin edge always sits exactly at the SBL (0)
    n_bins = len(edges) - 1
    M = np.full((n_cols_feat, len(events), n_bins), np.nan)
    for e, (_, x, Y) in enumerate(events):
        b = np.clip(np.floor((x - edges[0]) / bin_s).astype(int), 0, n_bins - 1)
        frame = pd.DataFrame(Y)
        frame["bin"] = b
        med = frame.groupby("bin").median()
        for j in range(n_cols_feat):
            M[j, e, med.index.to_numpy()] = med[j].to_numpy()
    return edges, M


def plot_sbl_summary(blocks: dict, sbl_times, before="30min", after="30min", sbl_indices=None,
                     features=None, ts_col="timestamp", statistic="median", normalize="baseline",
                     bin_width="auto", min_events=5, baseline_frac=0.5, normal_margin="15min",
                     n_boot=300, yscale="linear", n_cols=3, dpi=300, colormap="tab20", x_unit="auto"):
    """Same panels as plot_sbl_overlay (one per feature, red line = SBL), but instead of every event a
    SUMMARY over the events: the time axis is cut into bins, each event first gets ONE value per bin
    (its median in that bin, so an event with many samples does not dominate), then over the events:
        statistic="median": median line + band from the 25th to the 75th percentile (spread of events)
        statistic="mean":   mean line + 95% bootstrap confidence interval of the mean
    Bins reached by fewer than `min_events` events are hidden. A strip below the panels shows how many
    events contribute per bin.

    normalize: "baseline" (default) = value minus the event's OWN baseline (median of the earliest
                   `baseline_frac` of the before-window), divided by the typical fluctuation of the feature
                   around its baseline (robust sigma) -> events and features become comparable, 0 = no change;
               "zscore" = robust z-score against all normal rows of the block (farther than normal_margin
                   from every SBL);
               "none" = raw values.
    bin_width: "auto" (at least 2 median row spacings, about 100 bins over the window) or e.g. "1min".
    Returns ({block: figure}, summary table: block, feature, time_s, n_events, center, low, high)."""
    if statistic not in ("median", "mean"):
        raise ValueError("statistic must be 'median' or 'mean'.")
    if normalize not in ("baseline", "zscore", "none"):
        raise ValueError("normalize must be 'baseline', 'zscore' or 'none'.")
    sbl_s, indices, setup = _block_setup(blocks, sbl_times, before, after, sbl_indices, features, ts_col)
    cmap = plt.get_cmap(colormap)
    n_colors = min(cmap.N, 20)
    rng = np.random.default_rng(0)
    figs, rows = {}, []

    for name, st in setup.items():
        cols, events, b_s, a_s = st["cols"], st["events"], st["b_s"], st["a_s"]
        if not events:
            print(f"[{name}] no data in any event window - skipped.")
            continue
        df = blocks[name]
        spacing = pd.to_datetime(df[ts_col]).sort_values().diff().dt.total_seconds().median()
        bin_s = (pd.Timedelta(bin_width).total_seconds() if bin_width != "auto"
                 else max(2 * spacing, (b_s + a_s) / 100))
        edges, M = _bin_events(events, len(cols), b_s, a_s, bin_s)
        n_bins = M.shape[2]
        centers = (edges[:-1] + edges[1:]) / 2

        # ---- normalisation ----
        if normalize == "baseline":
            for j in range(len(cols)):
                devs = []
                for e, (_, x, Y) in enumerate(events):
                    sel = (x >= -b_s) & (x <= -b_s * (1 - baseline_frac)) & np.isfinite(Y[:, j])
                    if sel.sum() == 0:
                        M[j, e, :] = np.nan                  # no baseline for this event -> cannot be normalised
                        continue
                    base = np.median(Y[sel, j])
                    devs.append(Y[sel, j] - base)
                    M[j, e, :] = M[j, e, :] - base
                pooled = np.concatenate(devs) if devs else np.array([])
                scale = 1.4826 * np.median(np.abs(pooled - np.median(pooled))) if len(pooled) >= 20 else np.nan
                if not scale or not np.isfinite(scale) or scale <= 0:
                    scale = pooled.std() if len(pooled) >= 20 and pooled.std() > 0 else 1.0
                M[j] = M[j] / scale
            ylabel = "deviation from own baseline (robust sigma)"
        elif normalize == "zscore":
            ts_all = pd.to_datetime(df[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
            sbl_ns = np.sort(sbl_s.to_numpy(dtype="datetime64[ns]").astype("int64"))
            k = np.clip(np.searchsorted(sbl_ns, ts_all), 1, len(sbl_ns) - 1) if len(sbl_ns) > 1 else None
            dist = (np.minimum(np.abs(ts_all - sbl_ns[k - 1]), np.abs(sbl_ns[k] - ts_all)) / 1e9
                    if k is not None else np.abs(ts_all - sbl_ns[0]) / 1e9)
            normal = df.sort_values(ts_col)[cols].to_numpy(float)[dist > pd.Timedelta(normal_margin).total_seconds()]
            for j in range(len(cols)):
                v = normal[np.isfinite(normal[:, j]), j]
                if len(v) < 20:
                    M[j] = np.nan
                    continue
                q1, med_, q3 = np.percentile(v, [25, 50, 75])
                sc = (q3 - q1) / 1.349 or v.std() or 1.0
                M[j] = (M[j] - med_) / sc
            ylabel = "robust z-score vs. normal data"
        else:
            ylabel = "value"

        # ---- statistic over events, per bin ----
        counts = np.isfinite(M).sum(axis=1)                      # features x bins
        center = np.full((len(cols), n_bins), np.nan)
        low, high = center.copy(), center.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(len(cols)):
                valid = counts[j] >= min_events
                if not valid.any():
                    continue
                Mj = M[j][:, valid]
                if statistic == "median":
                    low[j, valid], center[j, valid], high[j, valid] = np.nanpercentile(Mj, [25, 50, 75], axis=0)
                else:
                    center[j, valid] = np.nanmean(Mj, axis=0)
                    boots = np.empty((n_boot, Mj.shape[1]))
                    for b in range(n_boot):
                        boots[b] = np.nanmean(Mj[rng.integers(0, Mj.shape[0], Mj.shape[0])], axis=0)
                    low[j, valid], high[j, valid] = np.nanpercentile(boots, [2.5, 97.5], axis=0)

        for j, c in enumerate(cols):
            for bidx in range(n_bins):
                rows.append(dict(block=name, feature=c, time_s=centers[bidx], n_events=int(counts[j, bidx]),
                                 center=center[j, bidx], low=low[j, bidx], high=high[j, bidx]))
        if not np.isfinite(center).any():
            print(f"[{name}] no bin is reached by >= {min_events} events (rows are about {spacing:.0f} s apart): "
                  "use a wider window, a larger bin_width, or lower min_events.")

        # ---- figure ----
        unit = _auto_unit(max(b_s, a_s), x_unit)
        div = _UNIT_DIV[unit]
        n_rows, ncol = _grid(len(cols), n_cols)
        fig = plt.figure(figsize=(4.6 * ncol + 1.8, 2.9 * n_rows + 1.3), dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(n_rows + 1, ncol, height_ratios=[1] * n_rows + [0.4])
        xc = centers / div
        for j, c in enumerate(cols):
            ax = fig.add_subplot(gs[j // ncol, j % ncol])
            color = cmap(j % n_colors)
            ax.fill_between(xc, low[j], high[j], color=color, alpha=0.25, lw=0)
            ax.plot(xc, center[j], color=color, lw=1.6)
            if normalize != "none":
                ax.axhline(0, color="gray", lw=0.6)
            ax.axvline(0, color="red", lw=1.4, zorder=5)
            ax.set_xlim(-b_s / div, a_s / div)
            if yscale == "symlog" and np.isfinite(center[j]).any():
                ax.set_yscale("symlog", linthresh=_symlog_threshold(center[j][np.isfinite(center[j])]))
            ax.set_title(c, fontsize=9, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if j % ncol == 0:
                ax.set_ylabel(ylabel, fontsize=7)
        axc = fig.add_subplot(gs[n_rows, :])
        n_med = np.median(counts, axis=0)
        axc.bar(xc, n_med, width=bin_s / div, color="gray", alpha=0.7)
        axc.axhline(min_events, color="black", ls="--", lw=0.8)
        axc.axvline(0, color="red", lw=1.0)
        axc.set_xlim(-b_s / div, a_s / div)
        axc.set_ylabel("events / bin", fontsize=8)
        axc.set_xlabel(f"time relative to SBL ({unit}),  bin width {bin_s / div:.3g} {unit}", fontsize=8)
        axc.tick_params(labelsize=7)

        band = "IQR (25-75%)" if statistic == "median" else "95% bootstrap CI"
        handles = [plt.Line2D([], [], color="red", lw=1.4), plt.Line2D([], [], color="gray", lw=1.6),
                   plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.25),
                   plt.Line2D([], [], color="black", ls="--", lw=0.8)]
        _outside_legend(fig, handles, ["SBL", f"{statistic} over events", band,
                                       f"min. events per bin ({min_events})"])
        fig.suptitle(f"{name}: {statistic} over {len(events)} SBL events with data  (normalize: {normalize})",
                     fontsize=11)
        figs[name] = fig
    plt.show()
    return figs, pd.DataFrame(rows)
