"""
feature_change_ranking.py - Which feature changes most in a chosen time before the SBLs?
And a heatmap of ALL features over the time before the SBLs.

rank_features_by_change(df, sbl_times, window="5min")
    For every SBL one number per feature: how much the feature changed in the `window` right before
    the SBL compared with its own baseline just before that window:
        measure="level"    : median(window) - median(baseline)                    (level shift)
        measure="activity" : log2( mean|row-to-row change| in window / in baseline )  (more or less jumpy)
    The change is expressed in units of the normal fluctuation of that feature (robust sigma of the
    same quantity at random times in normal operation), so features with different scales are
    comparable. A feature is ranked by the MEDIAN |change| over the SBL events and compared with the
    same statistic for equally many RANDOM normal-operation times (drawn many times):
        effect_ratio = SBL value / typical value at random times (1 = nothing special)
        p_empirical  = share of random draws that reach the SBL value or more
    The unit of this comparison is the event (not the row), so the correlated rows inside one window
    do not inflate the evidence. The p-values are not corrected for the many features and windows - use
    them for the ranking, not as formal significance.

plot_feature_heatmap(df, sbl_times, before="30min")
    Rows = features, columns = time before the SBL, colour = median over the events of the deviation from
    each event's own baseline (robust sigma). Rows are sorted by their strongest deviation.

Usage:
    from feature_change_ranking import rank_features_by_change, plot_feature_heatmap

    table = rank_features_by_change(ds_beam, sbl_times, window="5min", df_name="beam")
    table = rank_features_by_change(blocks, sbl_times, window=["1min", "5min", "30min"])   # all blocks, several windows
    figs, tables = plot_feature_heatmap(ds_beam, sbl_times, before="30min", df_name="beam")

Use data that are already restricted to beam ON (otherwise "random normal times" include beam-off times).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_plots import (_UNIT_DIV, _auto_unit, _bin_events, _block_setup, _normalize_to_baseline,
                       _sbl_sorted, _window_seconds)


def _nearest_dist_s(ts_ns, sbl_ns):
    i = np.searchsorted(sbl_ns, ts_ns)
    prev = np.abs(ts_ns - sbl_ns[np.clip(i - 1, 0, len(sbl_ns) - 1)])
    nxt = np.abs(sbl_ns[np.clip(i, 0, len(sbl_ns) - 1)] - ts_ns)
    return np.minimum(prev, nxt) / 1e9


def _change(ts, vals, absdiff, eps, t, w_ns, b_ns, measure, min_rows, sbl_ns):
    """Change of all features at time t: window [t-W, t) versus baseline [t-W-B, t-W). NaN if either
    period has too few rows or contains another SBL (the aftermath of an earlier event is no baseline)."""
    out = np.full(vals.shape[1], np.nan)
    if np.searchsorted(sbl_ns, t, side="left") - np.searchsorted(sbl_ns, t - w_ns - b_ns, side="left") > 0:
        return out
    lo_w, hi_w = np.searchsorted(ts, t - w_ns, side="left"), np.searchsorted(ts, t, side="left")
    lo_b = np.searchsorted(ts, t - w_ns - b_ns, side="left")
    if hi_w - lo_w < min_rows or lo_w - lo_b < min_rows:
        return out
    if measure == "level":
        return np.nanmedian(vals[lo_w:hi_w], axis=0) - np.nanmedian(vals[lo_b:lo_w], axis=0)
    a_w, a_b = np.nanmean(absdiff[lo_w:hi_w], axis=0), np.nanmean(absdiff[lo_b:lo_w], axis=0)
    return np.log2((a_w + eps) / (a_b + eps))


def rank_features_by_change(blocks, sbl_times, window="5min", baseline="auto", measure="level",
                            ts_col="timestamp", features=None, df_name="data", min_rows=2, big=3.0,
                            n_random=2000, normal_margin="15min", n_resample=2000, seed=0,
                            top=25, plot=True, dpi=300) -> pd.DataFrame:
    """Rank the features by their change in the `window` before the SBLs (see the module docstring).

    blocks: one DataFrame (named df_name) or {"beam": df, ...}; every numeric column is a feature, or give
            features={"beam": ["her_current", ...]}.
    window: one time ("5min") or a list; baseline: the period before the window it is compared with
            ("auto" = max(30 min, 2 x window)); min_rows: rows needed in the window and in the baseline.
    big:    a change counts as "big" if it exceeds this many normal fluctuations (column frac_big).
    n_random / n_resample: random normal-operation times to build the reference, and how many equally
            sized random sets are drawn from them for the p_empirical.
    Returns one row per block, feature and window, sorted by window and effect_ratio."""
    if measure not in ("level", "activity"):
        raise ValueError("measure must be 'level' or 'activity'.")
    if isinstance(blocks, pd.DataFrame):
        blocks = {df_name: blocks}
    windows = [window] if isinstance(window, str) else list(window)
    sbl_ns = np.sort(_sbl_sorted(sbl_times).to_numpy(dtype="datetime64[ns]").astype("int64"))
    margin_s = pd.Timedelta(normal_margin).total_seconds()
    rng = np.random.default_rng(seed)
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, df in blocks.items():
            d = df.sort_values(ts_col)
            cols = (features or {}).get(name) or [c for c in d.select_dtypes("number").columns if c != ts_col]
            if not cols:
                continue
            ts = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
            vals = d[cols].to_numpy(float)
            absdiff = np.abs(np.diff(vals, axis=0, prepend=np.nan))
            mean_ad = np.nanmean(absdiff, axis=0)
            eps = np.where(mean_ad > 0, 0.01 * mean_ad, 1.0)
            cand = np.where(_nearest_dist_s(ts, sbl_ns) > margin_s)[0]
            pick = rng.choice(cand, size=min(n_random, len(cand)), replace=False) if len(cand) else np.array([], int)
            t_rand = ts[pick]

            for w in windows:
                w_ns = pd.Timedelta(w).value
                b_ns = pd.Timedelta(baseline).value if baseline != "auto" else max(pd.Timedelta("30min").value, 2 * w_ns)
                C = np.array([_change(ts, vals, absdiff, eps, t, w_ns, b_ns, measure, min_rows, sbl_ns)
                              for t in sbl_ns]).reshape(len(sbl_ns), len(cols))
                R = np.array([_change(ts, vals, absdiff, eps, t, w_ns, b_ns, measure, min_rows, sbl_ns)
                              for t in t_rand]).reshape(len(t_rand), len(cols))
                for j, c in enumerate(cols):
                    r_valid = R[np.isfinite(R[:, j]), j]
                    if len(r_valid) >= 20:
                        scale = 1.4826 * np.median(np.abs(r_valid - np.median(r_valid)))
                        scale = scale if scale > 0 else (r_valid.std() if r_valid.std() > 0 else 1.0)
                    else:
                        scale = 1.0
                    c_valid = C[np.isfinite(C[:, j]), j] / scale
                    k = len(c_valid)
                    row = dict(block=name, feature=c, window=w, measure=measure, n_events=k,
                               median_change=np.nan, median_abs_change=np.nan, frac_big=np.nan,
                               null_median_abs_change=np.nan, effect_ratio=np.nan, p_empirical=np.nan)
                    if k >= 3 and len(r_valid) >= 20:
                        pool = np.abs(r_valid / scale)
                        obs = float(np.median(np.abs(c_valid)))
                        null = np.median(pool[rng.integers(0, len(pool), (n_resample, k))], axis=1)
                        null_med = float(np.median(null))
                        row.update(median_change=float(np.median(c_valid)), median_abs_change=obs,
                                   frac_big=float(np.mean(np.abs(c_valid) > big)), null_median_abs_change=null_med,
                                   effect_ratio=obs / null_med if null_med > 0 else np.nan,
                                   p_empirical=float((1 + (null >= obs).sum()) / (1 + n_resample)))
                    rows.append(row)

    table = pd.DataFrame(rows)
    order = {w: i for i, w in enumerate(windows)}
    table = (table.assign(_w=table["window"].map(order))
                  .sort_values(["_w", "effect_ratio"], ascending=[True, False]).drop(columns="_w")
                  .reset_index(drop=True))
    if plot and len(table):
        _plot_ranking(table, windows, measure, top, dpi, multi_block=len(blocks) > 1)
    return table


def _plot_ranking(table, windows, measure, top, dpi, multi_block):
    label = table["block"] + ": " + table["feature"] if multi_block else table["feature"]
    table = table.assign(label=label)
    if len(windows) == 1:
        t = table.dropna(subset=["effect_ratio"]).head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1.8), dpi=dpi, layout="constrained")
        colors = np.where(t["p_empirical"] >= 0.05, "#bdbdbd",
                          np.where(t["median_change"] >= 0, "#d62728", "#1f77b4"))
        ax.barh(t["label"], t["effect_ratio"], color=colors, edgecolor="black", lw=0.4)
        ax.axvline(1, color="black", ls="--", lw=1)
        for y, (er, p) in enumerate(zip(t["effect_ratio"], t["p_empirical"])):
            ax.text(er, y, f"  p={p:.3f}", va="center", fontsize=7)
        up, down = ("rise", "fall") if measure == "level" else ("more jumpy", "less jumpy")
        ax.set_xlabel(f"effect ratio = median |change| before the SBLs / at random normal times   (red: {up}, blue: {down}, grey: p >= 0.05)",
                      fontsize=8)
        ax.set_title(f"Change in the last {windows[0]} before the SBLs  (measure: {measure}, "
                     f"{int(t['n_events'].max())} events)", fontsize=10)
        ax.tick_params(labelsize=7)
        plt.show()
        return
    piv = table.pivot_table(index="label", columns="window", values="effect_ratio")[windows]
    pv = table.pivot_table(index="label", columns="window", values="p_empirical")[windows]
    keep = piv.max(axis=1).sort_values(ascending=False).head(top).index
    piv, pv = piv.loc[keep], pv.loc[keep]
    fig, ax = plt.subplots(figsize=(1.0 * len(windows) + 4, 0.3 * len(piv) + 1.8), dpi=dpi, layout="constrained")
    im = ax.imshow(piv.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}" + ("*" if pv.iat[i, j] < 0.05 else ""), ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="effect ratio (1 = nothing special)")
    ax.set_xlabel("time window before the SBLs")
    ax.set_title(f"Effect ratio per feature and window (measure: {measure}; * = p_empirical < 0.05, uncorrected)",
                 fontsize=9)
    plt.show()


def plot_feature_heatmap(blocks, sbl_times, before="30min", after="0s", sbl_indices=None, features=None,
                         ts_col="timestamp", df_name="data", bin_width="auto", min_events=5, baseline_frac=0.5,
                         sort="strength", top=None, vmax=None, cmap="RdBu_r", x_unit="auto", dpi=300):
    """Heatmap of all features of a block: x = time before the SBL, one row per feature, colour = median over
    the SBL events of the deviation from the event's OWN baseline in robust sigmas (0 = no change; the baseline
    is the earliest `baseline_frac` of the `before` window). Bins reached by fewer than `min_events` events are grey.

    blocks:   one DataFrame (named df_name) or {"beam": df, ...} - one figure per block.
    before / after: the window; after="0s" (default) shows only the time before the SBL (the red line is the
              right edge). A dict {block: "10min"} sets it per block.
    The heatmap shows LEVEL changes only. A feature that just becomes more jumpy without a level shift stays
    pale here - find those with rank_features_by_change(..., measure="activity").
    sort:     "strength" = strongest |deviation| in the window on top, "none" = original column order.
    top:      show only the `top` strongest features.  vmax: colour range +-vmax (default: 98th percentile).
    Returns ({block: figure}, {block: DataFrame features x time bins})."""
    if isinstance(blocks, pd.DataFrame):
        blocks = {df_name: blocks}
    sbl_s, _, setup = _block_setup(blocks, sbl_times, before, after, sbl_indices, features, ts_col)
    figs, tables = {}, {}
    for name, st in setup.items():
        cols, events, b_s, a_s = st["cols"], st["events"], st["b_s"], st["a_s"]
        if not events:
            print(f"[{name}] no data in any event window - skipped.")
            continue
        spacing = pd.to_datetime(blocks[name][ts_col]).sort_values().diff().dt.total_seconds().median()
        bin_s = (pd.Timedelta(bin_width).total_seconds() if bin_width != "auto"
                 else max(2 * spacing, (b_s + a_s) / 100))
        edges, M = _bin_events(events, len(cols), b_s, a_s, bin_s)
        _normalize_to_baseline(M, events, b_s, baseline_frac)
        counts = np.isfinite(M).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.where(counts >= min_events, np.nanmedian(M, axis=1), np.nan)
        centers = (edges[:-1] + edges[1:]) / 2

        strength = np.nanmax(np.abs(center), axis=1) if np.isfinite(center).any() else np.zeros(len(cols))
        strength = np.where(np.isfinite(strength), strength, -1)
        order = np.argsort(-strength) if sort == "strength" else np.arange(len(cols))
        if top:
            order = order[:top]
        mat = center[order]
        names = [cols[i] for i in order]
        if not np.isfinite(mat).any():
            print(f"[{name}] no bin is reached by >= {min_events} events - widen the window, use a larger "
                  "bin_width or a smaller min_events.")
            continue
        v = vmax or max(float(np.nanpercentile(np.abs(mat), 98)), 1e-9)

        unit = _auto_unit(max(b_s, a_s), x_unit)
        div = _UNIT_DIV[unit]
        fig, ax = plt.subplots(figsize=(11, 0.28 * len(names) + 2.2), dpi=dpi, layout="constrained")
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#e6e6e6")
        mesh = ax.pcolormesh(edges / div, np.arange(len(names) + 1), np.ma.masked_invalid(mat), cmap=cm,
                             vmin=-v, vmax=v, shading="flat")
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="red", lw=1.6)
        ax.set_xlabel(f"time relative to the SBL ({unit}),  bin width {bin_s / div:.3g} {unit}", fontsize=9)
        fig.colorbar(mesh, ax=ax, label="median deviation (robust sigma)")
        ax.set_title(f"{name}: all features, median over {len(events)} SBL events with data  "
                     f"(baseline = first {baseline_frac:.0%} of the window; grey = fewer than {min_events} events)",
                     fontsize=9)
        plt.show()
        figs[name] = fig
        tables[name] = pd.DataFrame(mat, index=names, columns=np.round(centers, 3))
    return figs, tables
