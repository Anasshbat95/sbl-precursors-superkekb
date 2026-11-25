"""
enrichment_analysis.py - For every feature, compare how many values fall inside a short window around each
SBL event ("N_fe") versus outside all such windows ("N_fe_outside"). The ratio of the two RATES (per second
of time WITH DATA) shows which features report more often near an SBL than normally.

    side="before"  window [SBL - w, SBL)   -> possible precursor (default for compute_enrichment_ratio)
    side="after"   window [SBL, SBL + w)   -> reaction to the SBL / abort (old behaviour)
    count="values"  non-null values (use RAW, not forward-filled data: after a forward fill every row is
                    non-null and all features of a block get the same number)
    count="changes" rows where the value changed (also works on forward-filled data)
    max_gap         a gap between two rows counts at most this long, so removed periods (shutdown) and data
                    holes are not counted as "time without reports" (they would inflate the ratio)

Note: plot_probability_per_feature keeps the OLD formula P = N_fe / (128 * N_sbls) / N_total only for
comparison (window after the SBL, constant 128 of unknown origin, no time normalisation). Use
compute_enrichment_ratio for results. Features are named "BLOCK:feature" in the enrichment ratio.

Usage:
    from enrichment_analysis import (
        count_values_in_sbl_windows, count_values_outside_sbl_windows,
        plot_probability_per_feature, compute_enrichment_ratio, plot_enrichment_ratio,
    )

    blocks = {"BEAM": beam, "BACKGROUND": background, "INJECTION": injection,
              "VACUUM": vacuum, "POSITION": position}

    n_fe = {name: count_values_in_sbl_windows(sbl_times, df) for name, df in blocks.items()}
    n_fe_outside = {name: count_values_outside_sbl_windows(sbl_times, df) for name, df in blocks.items()}
    fig, ax, p_all, group_of_feature = plot_probability_per_feature(sbl_times, blocks)
    ratio, group_of_feature = compute_enrichment_ratio(sbl_times, blocks)
    plot_enrichment_ratio(ratio, group_of_feature)
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_GROUP_COLORS = {
    "BEAM": "steelblue",
    "BACKGROUND": "darkorange",
    "INJECTION": "seagreen",
    "VACUUM": "firebrick",
    "POSITION": "purple",
}


def _color(group_colors, group):
    g = {k.upper(): v for k, v in group_colors.items()}
    return g.get(str(group).upper(), "gray")


def _ts_col(df):
    return "Timestamp" if "Timestamp" in df.columns else "timestamp"


def _sbl_series(sbl_times):
    if isinstance(sbl_times, pd.DataFrame):
        if "timestamp" in sbl_times.columns:
            sbl_times = sbl_times["timestamp"]
        elif "Timestamp" in sbl_times.columns:
            sbl_times = sbl_times["Timestamp"]
        elif sbl_times.shape[1] == 1:
            sbl_times = sbl_times.iloc[:, 0]
        else:
            raise ValueError("sbl_times DataFrame must contain one timestamp column.")
    return pd.to_datetime(pd.Series(sbl_times)).dropna()


def _windows(sbl_times, window_seconds, side, buffer_seconds=0):
    """Merged, sorted list of (start, end) windows. side="before": [SBL - w, SBL - buffer)."""
    w, b = pd.Timedelta(seconds=window_seconds), pd.Timedelta(seconds=buffer_seconds)
    iv = sorted((t - w, t - b) if side == "before" else (t + b, t + w) for t in sbl_times)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _covered_seconds(ts, windows, max_gap):
    """Seconds with data: every row covers the time until the next row, at most max_gap.
    Returns (covered time inside the windows, covered time in total)."""
    t = np.sort(pd.to_datetime(ts).to_numpy(dtype="datetime64[ns]").astype("int64"))
    if len(t) < 2:
        return 0.0, 0.0
    S = t[:-1]
    E = S + np.minimum(np.diff(t), pd.Timedelta(max_gap).value)
    total = float((E - S).sum()) / 1e9
    inside = 0.0
    for a, b in windows:
        a, b = pd.Timestamp(a).value, pd.Timestamp(b).value
        i0, i1 = np.searchsorted(E, a, "right"), np.searchsorted(S, b, "left")
        if i1 > i0:
            inside += float((np.minimum(E[i0:i1], b) - np.maximum(S[i0:i1], a)).clip(min=0).sum()) / 1e9
    return inside, total


def _in_window_mask_side(sbl_times, ts, window_seconds, side, buffer_seconds=0):
    m = pd.Series(False, index=ts.index)
    for a, b in _windows(sbl_times, window_seconds, side, buffer_seconds):
        m |= (ts >= a) & (ts < b)
    return m


def _counts_per_feature(df, mask, count):
    d = df.sort_values(_ts_col(df))
    cols = [c for c in d.select_dtypes("number").columns if c not in ("timestamp", "Timestamp")]
    m = mask.reindex(d.index).to_numpy()
    if count == "changes":
        v = d[cols]
        hit = (v.ne(v.shift()) & v.notna()).to_numpy()
        hit[0] = False
    else:
        hit = d[cols].notna().to_numpy()
    return {c: int(hit[m, j].sum()) for j, c in enumerate(cols)}, {c: int(hit[~m, j].sum()) for j, c in enumerate(cols)}


def _in_window_mask(sbl_times, ts, window_seconds):
    """True for rows inside any [SBL, SBL + window_seconds) window; overlapping windows count once."""
    in_window = pd.Series(False, index=ts.index)
    for t in sbl_times:
        in_window |= (ts >= t) & (ts < t + pd.Timedelta(seconds=window_seconds))
    return in_window


def _window_seconds_total(sbl_times, start, end, window_seconds):
    """Length in seconds of the union of all SBL windows, clipped to [start, end]."""
    total, cur_start, cur_end = 0.0, None, None
    for t in sorted(sbl_times):
        s, e = max(t, start), min(t + pd.Timedelta(seconds=window_seconds), end)
        if s >= e:
            continue
        if cur_end is not None and s <= cur_end:
            cur_end = max(cur_end, e)
            continue
        if cur_end is not None:
            total += (cur_end - cur_start).total_seconds()
        cur_start, cur_end = s, e
    if cur_end is not None:
        total += (cur_end - cur_start).total_seconds()
    return total


def count_values_in_sbl_windows(sbl_times, df, window_seconds=60) -> dict:
    """Number of non-null values per feature, inside any [SBL, SBL + window_seconds) window."""
    sbl_times = _sbl_series(sbl_times)
    ts = pd.to_datetime(df[_ts_col(df)])
    feature_cols = df.columns.drop(_ts_col(df))
    inside = df.loc[_in_window_mask(sbl_times, ts, window_seconds), feature_cols]
    return {c: int(inside[c].notna().sum()) for c in feature_cols}


def count_values_outside_sbl_windows(sbl_times, df, window_seconds=60) -> dict:
    """Number of non-null values per feature, outside every [SBL, SBL + window_seconds) window."""
    sbl_times = _sbl_series(sbl_times)
    ts = pd.to_datetime(df[_ts_col(df)])
    feature_cols = df.columns.drop(_ts_col(df))
    outside = df.loc[~_in_window_mask(sbl_times, ts, window_seconds), feature_cols]
    return {c: int(outside[c].notna().sum()) for c in feature_cols}


def _probability_per_feature(sbl_times, blocks: dict, count_fn, window_seconds=60):
    sbl_times = _sbl_series(sbl_times)
    n_sbls = len(sbl_times)
    n_counted, n_total, group_of_feature = {}, {}, {}
    for group, df in blocks.items():
        counts = count_fn(sbl_times, df, window_seconds=window_seconds)
        totals = df.drop(columns=[_ts_col(df)], errors="ignore").notna().sum()
        for feat, v in counts.items():
            n_counted[feat] = v
            n_total[feat] = totals[feat]
            group_of_feature[feat] = group
    n_counted = pd.Series(n_counted)
    n_total = pd.Series(n_total)
    p = (n_counted / (128 * n_sbls)) * (1 / n_total)
    return p.dropna(), group_of_feature


def plot_probability_per_feature(sbl_times, blocks: dict, window_seconds=60, title="P per feature",
                                 group_colors=None, save_path=None, show_plot=True):
    """P = (N_fe / (128 * N_sbls)) * (1 / N_total) per feature, as a horizontal bar chart.
    Returns (fig, ax, P, group_of_feature)."""
    group_colors = group_colors or DEFAULT_GROUP_COLORS
    p_all, group_of_feature = _probability_per_feature(sbl_times, blocks, count_values_in_sbl_windows, window_seconds)
    p_all = p_all.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [_color(group_colors, group_of_feature.get(f)) for f in p_all.index]
    ax.barh(p_all.index, p_all.values, color=colors)
    ax.set_xlabel("P (probability)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{title}  (P = N_fe / (128 * N_sbls) * 1 / N_total)")
    ax.legend(handles=[mpatches.Patch(color=c, label=g) for g, c in group_colors.items()],
             loc="lower right", title="Dataset")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show_plot:
        plt.show()
    return fig, ax, p_all, group_of_feature


def compute_enrichment_ratio(sbl_times, blocks: dict, window_seconds=60, side="before", count="changes",
                             max_gap="1h", buffer_seconds=0, return_counts=False):
    """Rate inside the SBL windows / rate outside them, per feature, both per second of time WITH DATA.
    1 = as often as normally, > 1 = more often near the SBLs, 0 = never inside the windows.
    buffer_seconds: leave out the last seconds before the SBL (window [SBL - w, SBL - buffer)). If the ratio
    collapses with a buffer of a few seconds, the extra reports come from the beam loss itself (or a small
    clock offset), not from a precursor.
    Returns (ratio, group_of_feature) with feature names "BLOCK:feature"."""
    if side not in ("before", "after"):
        raise ValueError("side must be 'before' or 'after'.")
    sbl_times = _sbl_series(sbl_times)
    wins = _windows(sbl_times, window_seconds, side, buffer_seconds)
    ratio, group_of_feature, counts = {}, {}, {}
    for group, df in blocks.items():
        ts = pd.to_datetime(df[_ts_col(df)])
        t_in, t_all = _covered_seconds(ts, wins, max_gap)
        t_out = t_all - t_in
        if t_in <= 0 or t_out <= 0:
            print(f"  {group}: no time with data inside the windows - skipped")
            continue
        n_in, n_out = _counts_per_feature(df, _in_window_mask_side(sbl_times, ts, window_seconds, side,
                                                                    buffer_seconds), count)
        for feat in n_in:
            key = f"{group}:{feat}"
            group_of_feature[key] = group
            rate_out = n_out[feat] / t_out
            ratio[key] = (n_in[feat] / t_in) / rate_out if rate_out > 0 else float("nan")
            counts[key] = dict(n_in=n_in[feat], n_out=n_out[feat], hours_in=round(t_in / 3600, 2),
                               hours_out=round(t_out / 3600, 1))
    r = pd.Series(ratio, dtype=float).dropna()
    if return_counts:
        return r, group_of_feature, pd.DataFrame(counts).T.assign(ratio=pd.Series(ratio))
    return r, group_of_feature


def enrichment_vs_buffer(sbl_times, blocks: dict, window_seconds=300, buffers=(0, 5, 10, 30, 60), **kw):
    """Enrichment ratio for several buffers (seconds left out right before the SBL). One column per buffer.
    A real precursor keeps its ratio when the last seconds are left out; a reaction to the beam loss (or a
    clock offset of a few seconds) disappears."""
    cols = {}
    for b in buffers:
        if b >= window_seconds:
            continue
        r, _ = compute_enrichment_ratio(sbl_times, blocks, window_seconds=window_seconds, buffer_seconds=b, **kw)
        cols[f"buffer {b}s"] = r
    t = pd.DataFrame(cols)
    return t.sort_values(t.columns[0], ascending=False)


def plot_enrichment_ratio(ratio: pd.Series, group_of_feature: dict, group_colors=None,
                          title="Enrichment per feature: SBL window vs. reference baseline"):
    """Bar chart of the per-feature enrichment ratio (log scale). Values > 1 react more near an SBL."""
    group_colors = group_colors or DEFAULT_GROUP_COLORS
    zero = ratio[ratio == 0]
    if len(zero):
        print(f"  {len(zero)} features never reported inside the windows (ratio 0, not drawn on the log axis): "
              f"{list(zero.index)}")
    ratio = ratio[ratio > 0].sort_values(ascending=False)
    colors = [_color(group_colors, group_of_feature.get(f)) for f in ratio.index]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(ratio.index, ratio.values, color=colors)
    ax.set_xscale("log")
    ax.axvline(1, color="black", linestyle="--", linewidth=1.2, label="No difference (=1)")
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.5)
    for bar, val in zip(bars, ratio.values):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Rate in SBL window / rate outside (per second with data, log scale)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=handles, loc="lower right", title="Dataset")
    plt.tight_layout()
    plt.show()
    return fig, ax, ratio