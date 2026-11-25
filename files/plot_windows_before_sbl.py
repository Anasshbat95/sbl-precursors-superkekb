"""
plot_windows_before_sbl.py - For noisy/irregular data where only the time right
before each SBL matters: cut out everything except the last `window` (e.g. 5
minutes) before each selected SBL, and draw those segments one after another
in a single plot per feature - no other time shown, no anomaly points.

Usage:
    from plot_windows_before_sbl import plot_feature_windows_before_sbl

    # one feature, all 128 SBL events, last 5 min before each
    plot_feature_windows_before_sbl(ds_beam, "timestamp", "her_current", sbl_times,
                                    window="5min", df_name="beam")

    # several features stacked, only a few chosen events (by index into the sorted SBL list)
    plot_feature_windows_before_sbl(ds_beam, "timestamp", ["her_current", "her_sigma_x"], sbl_times,
                                    window="5min", sbl_indices=[0, 1, 2, 50, 127], df_name="beam")
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from files.anomaly_detection_plot import _make_model


def _as_sbl_series(sbl_times) -> pd.Series:
    """Accepts a Series, 1D array/list, or single-column DataFrame and always returns a
    1D datetime Series, sorted."""
    if isinstance(sbl_times, pd.DataFrame):
        if sbl_times.shape[1] != 1:
            raise ValueError(
                f"sbl_times is a DataFrame with {sbl_times.shape[1]} columns - expected exactly 1. "
                f"Pass a single column instead, e.g. sbl_times['{sbl_times.columns[0]}']."
            )
        sbl_times = sbl_times.iloc[:, 0]
    return pd.to_datetime(pd.Series(sbl_times).reset_index(drop=True)).sort_values().reset_index(drop=True)


def plot_feature_windows_before_sbl(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times,
                                    window="5min", sbl_indices=None, gap_fraction=0.08,
                                    df_name="data", marker_size=6,
                                    show_anomalies=True, anomaly_feature_cols=None,
                                    method="isolation_forest", contamination=0.05) -> pd.DataFrame:
    """Keep only the `window` right before each selected SBL, discard everything else, and plot
    those segments back to back (one subplot per feature if feature_cols is a list), with
    anomalous rows highlighted in orange.

    feature_cols: a single column name, or a list of column names (one subplot each).
    sbl_indices: which SBL events to include, as a list of positions (0-based) into the
                 chronologically sorted SBL list. Default None = all events.
    window: how much time right before each SBL to keep, e.g. "5min", "30s", "1h".
    gap_fraction: visual gap between segments, as a fraction of `window` (purely cosmetic).
    show_anomalies: if True (default), fit an unsupervised anomaly detector (same methods as
                    anomaly_detection_plot.py: "isolation_forest", "lof", "one_class_svm",
                    "one_class_svm_sgd", "robust_covariance") on the KEPT data only (i.e. only
                    using these pre-SBL windows, not the full time series) and mark flagged rows
                    with orange points on every feature's line.
    anomaly_feature_cols: which columns the detector is fit on (can be more/other features than
                          you plot). Default None = use `feature_cols` itself.

    A red solid line marks the SBL moment itself at the end of each segment; a gray dashed line
    marks the start of the window. X-axis ticks show the SBL's real timestamp per segment.

    Returns a DataFrame with the kept rows, a 'sbl_index' column (which event each row belongs
    to), a 'plot_x' column (the position used on the concatenated x-axis), and - if
    show_anomalies=True - an 'is_anomaly' column.
    """
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    sbl_sorted = _as_sbl_series(sbl_times)
    indices_given = sbl_indices is not None   # decides the tick label style further down
    if sbl_indices is None:
        sbl_indices = list(range(len(sbl_sorted)))
    else:
        out_of_range = [i for i in sbl_indices if not (0 <= i < len(sbl_sorted))]
        if out_of_range:
            print(f"Ignoring {len(out_of_range)} out-of-range index/indices (valid range is "
                 f"0..{len(sbl_sorted) - 1}): {out_of_range}")
        sbl_indices = [i for i in sbl_indices if 0 <= i < len(sbl_sorted)]
        if not sbl_indices:
            raise ValueError(f"None of the given sbl_indices are valid (valid range is "
                            f"0..{len(sbl_sorted) - 1}).")

    win_sec = pd.Timedelta(window).total_seconds()
    gap_sec = win_sec * gap_fraction

    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col)
    ts_all = d[ts_col].to_numpy(dtype="datetime64[ns]")

    kept_rows = []
    offset = 0.0
    segment_info = []  # (sbl_index, sbl_time, x_start, x_end) for drawing lines/ticks
    for i in sbl_indices:
        sbl_t = sbl_sorted.iloc[i]
        w_start = sbl_t - pd.Timedelta(window)
        mask = (d[ts_col] >= w_start) & (d[ts_col] < sbl_t)
        seg = d.loc[mask].copy()
        if seg.empty:
            segment_info.append((i, sbl_t, offset, offset + win_sec))
            offset += win_sec + gap_sec
            continue
        rel_sec = (seg[ts_col] - w_start).dt.total_seconds()
        seg["plot_x"] = offset + rel_sec
        seg["sbl_index"] = i
        seg["sbl_time"] = sbl_t
        kept_rows.append(seg)
        segment_info.append((i, sbl_t, offset, offset + win_sec))
        offset += win_sec + gap_sec

    empty_indices = [i for i in sbl_indices if i not in {s["sbl_index"].iloc[0] for s in kept_rows}] \
        if kept_rows else list(sbl_indices)
    if empty_indices:
        df_start, df_end = d[ts_col].min(), d[ts_col].max()
        print(f"No data in the {window} window before {len(empty_indices)} of {len(sbl_indices)} "
             f"requested event(s): {empty_indices}")
        print(f"  df time range: {df_start} to {df_end}")
        for i in empty_indices[:10]:
            sbl_t = sbl_sorted.iloc[i]
            inside_range = df_start <= sbl_t <= df_end
            print(f"  index {i}: SBL at {sbl_t}  "
                 f"({'inside' if inside_range else 'OUTSIDE'} df's time range)")
        if len(empty_indices) > 10:
            print(f"  ... and {len(empty_indices) - 10} more")
        print("  Likely cause: these events fall outside df's coverage, or in a period your "
             "filtering (e.g. BEAM_ON-only / shutdown removal) cut out. Widen `window`, check "
             "df's filtering, or drop these indices.")

    if not kept_rows:
        raise ValueError("No data found in ANY of the requested windows (see diagnosis above) - nothing to plot.")
    kept = pd.concat(kept_rows, ignore_index=True)

    if show_anomalies:
        fit_cols = anomaly_feature_cols or feature_cols
        X = kept[fit_cols]
        valid = X.notna().all(axis=1)
        if valid.sum() < 20:
            print(f"Only {int(valid.sum())} rows have every feature in {fit_cols} non-NaN across "
                 "all kept windows - too few to fit a model, skipping anomaly detection.")
            kept["is_anomaly"] = False
        else:
            Xs = StandardScaler().fit_transform(X[valid])
            model = _make_model(method, contamination, n_samples=valid.sum())
            pred = model.fit_predict(Xs) if method == "lof" else model.fit(Xs).predict(Xs)
            kept["is_anomaly"] = False
            kept.loc[X[valid].index, "is_anomaly"] = (pred == -1)

    n = len(feature_cols)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 0.18 * len(segment_info)), 2.3 * n),
                             squeeze=False, sharex=True)
    axes = axes[:, 0]

    for ax, col in zip(axes, feature_cols):
        ax.plot(kept["plot_x"], kept[col], color="#377eb8", lw=0.8, marker=".", markersize=marker_size,
               alpha=0.85, label="data")
        if show_anomalies and "is_anomaly" in kept.columns:
            anom = kept[kept["is_anomaly"]]
            ax.scatter(anom["plot_x"], anom[col], color="#ff7f00", s=24, zorder=5,
                      label="predicted anomaly")
        for idx, sbl_t, x_start, x_end in segment_info:
            ax.axvline(x_end, color="red", linewidth=1.2, alpha=0.8)       # the SBL moment itself
            ax.axvline(x_start, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)  # window start
        ax.set_ylabel(col, fontsize=9)

    if show_anomalies:
        axes[0].legend(loc="upper right", fontsize=8)

    tick_pos = [x_end for _, _, _, x_end in segment_info]
    if indices_given:
        # specific events requested -> show the date, so you can tell them apart
        tick_labels = [f"#{idx}\n{sbl_t:%Y-%m-%d}" for idx, sbl_t, _, _ in segment_info]
    else:
        # all events (default) -> date would be unreadable with ~128 ticks, show only the id
        tick_labels = [f"#{idx}" for idx, sbl_t, _, _ in segment_info]
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel("SBL index / time (red line = SBL moment, gray dashed = window start)")

    anomaly_note = f" - {method}, {int(kept['is_anomaly'].sum())} anomalous rows (orange)" if show_anomalies else ""
    fig.suptitle(f"[{df_name}] last {window} before each of {len(segment_info)} selected SBL event(s) "
                f"- all other time cut out{anomaly_note}", y=1.0, fontsize=11)
    plt.tight_layout()
    plt.show()

    return kept
