"""
diff_distribution_before_sbl.py - Check whether a feature changes more sharply
right before an SBL than it normally does, across several blocks and several
time windows, plus correlation between features.

Main pieces:
    plot_diff_distributions   histogram of |diff| (all data vs. just before SBL),
                               one feature per column. Axes are the SAME for the
                               same feature across repeated calls (pass `ranges`
                               back in) so different window sizes stay comparable;
                               different features keep their own natural scale.
    plot_window_sweep         runs the comparison over several window sizes at
                               once and plots ratio_mean / KS-significance vs.
                               window size, per feature -> shows which feature
                               reacts most strongly, and at what lead time.
    correlation_matrix        correlation heatmap between all features of one df.
    correlation_before_sbl    same heatmap, but only using rows inside a chosen
                               window before each SBL.

Usage:
    from diff_distribution_before_sbl import (
        plot_diff_distributions, plot_window_sweep,
        correlation_matrix, correlation_before_sbl,
    )

    feature_cols = [c for c in BEAM_clean.columns if c != "timestamp"]
    sbl = SBL_timestamps_Germany["timestamp"]

    # same x-axis per feature across both calls, because `ranges` is reused:
    result_5m, ranges = plot_diff_distributions(BEAM_clean, "timestamp", feature_cols, sbl,
                                                time_before="5min", df_name="BEAM")
    result_30m, _ = plot_diff_distributions(BEAM_clean, "timestamp", feature_cols, sbl,
                                            time_before="30min", df_name="BEAM", ranges=ranges)

    sweep = plot_window_sweep(BEAM_clean, "timestamp", feature_cols, sbl,
                              windows=["30s","1min","5min","10min","30min","1h","2h"], df_name="BEAM")

    corr_all = correlation_matrix(BEAM_clean, feature_cols, df_name="BEAM")
    corr_near = correlation_before_sbl(BEAM_clean, "timestamp", feature_cols, sbl,
                                       time_before="10min", df_name="BEAM")

    # the two empirical distribution functions behind the KS test, side by side (F_Fenster vs. F_Gesamt):
    fig, table = plot_ecdf_window_vs_all(
        {"injection": ds_injection, "background": ds_background}, "timestamp",
        ["her_inj_efficiency", "ler_inj_efficiency", "her_tune_x", "her_tune_y", "ler_tune_x", "ler_tune_y"],
        sbl, time_before="5min")
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def add_diff_columns(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Add a '<feature>_diff' column (row-to-row change) for every feature. Returns a copy."""
    df = df.copy()
    for col in feature_cols:
        df[f"{col}_diff"] = df[col].diff()
    return df


def _as_sbl_series(sbl_times) -> pd.Series:
    """Accepts a Series, 1D array/list, or a single-column DataFrame (e.g. the whole
    SBL_timestamps_Germany DataFrame instead of SBL_timestamps_Germany['timestamp']) and
    always returns a 1D datetime Series."""
    if isinstance(sbl_times, pd.DataFrame):
        if sbl_times.shape[1] != 1:
            raise ValueError(
                f"sbl_times is a DataFrame with {sbl_times.shape[1]} columns - expected exactly 1. "
                f"Pass a single column instead, e.g. sbl_times['{sbl_times.columns[0]}']."
            )
        sbl_times = sbl_times.iloc[:, 0]
    return pd.to_datetime(pd.Series(sbl_times).reset_index(drop=True))


def time_before_next_sbl(df: pd.DataFrame, ts_col: str, sbl_times) -> np.ndarray:
    """Seconds from each row to the next SBL at or after it (always >= 0). Rows AFTER the last SBL
    have no next SBL and get NaN, so a comparison such as `seconds <= window` is False for them.
    (An earlier version clipped the index instead, which gave negative seconds there and counted every
    row after the last SBL as 'inside the window before an SBL'.)"""
    ts = df[ts_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    sbl = np.sort(_as_sbl_series(sbl_times).to_numpy(dtype="datetime64[ns]").astype("int64"))
    idx = np.searchsorted(sbl, ts)                       # first SBL at or after the row
    out = np.full(len(ts), np.nan)
    has_next = idx < len(sbl)
    out[has_next] = (sbl[idx[has_next]] - ts[has_next]) / 1e9
    return out


# ---------------------------------------------------------------- diff distributions
def plot_diff_distributions(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times, time_before="5min",
                            bins=60, use_abs=True, log_y=True, df_name="data", ranges=None) -> tuple:
    """For every feature: top panel = overall diff distribution, bottom panel = diff distribution
    only in the `time_before` window right before each SBL.

    ranges: optional {feature: (lo, hi)} dict for the x-axis. If None, computed from this call's
    full (overall) data and RETURNED, so you can pass the same `ranges` into later calls with a
    different `time_before` - that keeps the x-axis for a given feature identical across window
    sizes, while different features keep their own natural scale.

    Returns (result_df, ranges).
    """
    df = add_diff_columns(df, feature_cols)
    seconds_to_next_sbl = time_before_next_sbl(df, ts_col, sbl_times)
    before_window_sec = pd.Timedelta(time_before).total_seconds()
    in_window = seconds_to_next_sbl <= before_window_sec

    if ranges is None:
        ranges = {}
        for col in feature_cols:
            vals = df[f"{col}_diff"].dropna()
            if use_abs:
                vals = vals.abs()
            ranges[col] = (vals.quantile(0.001), vals.quantile(0.999))

    n = len(feature_cols)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)

    rows = []
    for j, col in enumerate(feature_cols):
        diff_col = f"{col}_diff"
        all_vals = df[diff_col].dropna()
        window_vals = df.loc[in_window, diff_col].dropna()
        if use_abs:
            all_vals, window_vals = all_vals.abs(), window_vals.abs()

        lo, hi = ranges[col]
        bin_edges = np.linspace(lo, hi, bins)

        ax_top, ax_bot = axes[0, j], axes[1, j]
        ax_top.hist(all_vals, bins=bin_edges, color="steelblue")
        ax_top.set_xlim(lo, hi)
        ax_top.set_title(f"[{df_name}] {col}\nall data (n={len(all_vals)})", fontsize=9)
        if log_y:
            ax_top.set_yscale("log")

        ax_bot.hist(window_vals, bins=bin_edges, color="darkorange")
        ax_bot.set_xlim(lo, hi)
        ax_bot.set_title(f"[{df_name}] {col}\nlast {time_before} before SBL (n={len(window_vals)})", fontsize=9)
        ax_bot.set_xlabel("|diff|" if use_abs else "diff")
        if log_y:
            ax_bot.set_yscale("log")

        if len(all_vals) > 10 and len(window_vals) > 10:
            stat, p = ks_2samp(window_vals, all_vals)
        else:
            stat, p = np.nan, np.nan
        rows.append(dict(feature=col, n_all=len(all_vals), n_window=len(window_vals),
                         mean_all=all_vals.mean(), mean_window=window_vals.mean(),
                         ratio_mean=window_vals.mean() / all_vals.mean() if all_vals.mean() else np.nan,
                         ks_stat=stat, ks_pvalue=p))

    plt.tight_layout()
    plt.show()

    result = pd.DataFrame(rows).sort_values("ks_pvalue")
    print(result.to_string(index=False))
    return result, ranges


# ---------------------------------------------------------------- window-size sweep
def compute_window_stats(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times, time_before,
                         use_abs=True) -> pd.DataFrame:
    """Same stats as plot_diff_distributions, but no plot - used by plot_window_sweep to check
    many window sizes quickly. One row per feature."""
    df = add_diff_columns(df, feature_cols)
    seconds_to_next_sbl = time_before_next_sbl(df, ts_col, sbl_times)
    win_sec = pd.Timedelta(time_before).total_seconds()
    in_window = seconds_to_next_sbl <= win_sec

    rows = []
    for col in feature_cols:
        diff_col = f"{col}_diff"
        all_vals = df[diff_col].dropna()
        window_vals = df.loc[in_window, diff_col].dropna()
        if use_abs:
            all_vals, window_vals = all_vals.abs(), window_vals.abs()
        if len(all_vals) > 10 and len(window_vals) > 10:
            stat, p = ks_2samp(window_vals, all_vals)
        else:
            stat, p = np.nan, np.nan
        rows.append(dict(feature=col, time_before=time_before, time_before_sec=win_sec,
                         n_window=len(window_vals), mean_all=all_vals.mean(),
                         mean_window=window_vals.mean(),
                         ratio_mean=window_vals.mean() / all_vals.mean() if all_vals.mean() else np.nan,
                         ks_pvalue=p))
    return pd.DataFrame(rows)


def plot_window_sweep(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times, windows,
                      use_abs=True, df_name="data") -> pd.DataFrame:
    """Run compute_window_stats for every window size in `windows` (e.g.
    ["30s","1min","5min","10min","30min","1h","2h"]) and plot, per feature:
    left  = ratio_mean (window / overall) vs. window size
    right = -log10(KS p-value) vs. window size (above the red line = statistically significant)
    This shows which feature reacts most strongly to an upcoming SBL, and at what lead time.
    Returns the full per-window, per-feature stats table."""
    full = pd.concat([compute_window_stats(df, ts_col, feature_cols, sbl_times, w, use_abs)
                      for w in windows], ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for col in feature_cols:
        sub = full[full["feature"] == col].sort_values("time_before_sec")
        ax1.plot(sub["time_before_sec"], sub["ratio_mean"], marker="o", label=col)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("window before SBL (s)")
    ax1.set_ylabel("ratio_mean (window / overall)")
    ax1.set_title(f"[{df_name}] Change ratio vs. window size")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for col in feature_cols:
        sub = full[full["feature"] == col].sort_values("time_before_sec")
        ax2.plot(sub["time_before_sec"], -np.log10(sub["ks_pvalue"].clip(lower=1e-300)), marker="o", label=col)
    ax2.axhline(-np.log10(0.05), color="red", linestyle="--", label="p = 0.05")
    ax2.set_xscale("log")
    ax2.set_xlabel("window before SBL (s)")
    ax2.set_ylabel("-log10(p-value)")
    ax2.set_title(f"[{df_name}] KS significance vs. window size")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return full


# ---------------------------------------------------------------- plot from already-collected results
def _window_to_sec(window_label: str) -> float:
    return pd.Timedelta(window_label).total_seconds()


def plot_ratio_pvalue_vs_window(results: dict, feature: str, df_name="data"):
    """Plot ratio_mean (left axis) and ks_pvalue (right axis, log) vs. window size, for ONE feature,
    using the `results` dict you already collected by calling plot_diff_distributions in a loop:

        results = {}
        for window in windows:
            result, ranges = plot_diff_distributions(df, "timestamp", feature_cols, sbl,
                                                      time_before=window, df_name="BEAM")
            results[window] = result   # or (result, ranges) - both are accepted here

    Accepts results[window] being either the result DataFrame itself, or the
    (result_df, ranges) tuple that plot_diff_distributions returns.
    """
    windows = sorted(results.keys(), key=_window_to_sec)
    sec, ratio, pval = [], [], []
    for w in windows:
        r = results[w]
        df_res = r[0] if isinstance(r, tuple) else r   # unwrap (result, ranges) if needed
        row = df_res[df_res["feature"] == feature]
        if row.empty:
            continue
        sec.append(_window_to_sec(w))
        ratio.append(row["ratio_mean"].iloc[0])
        pval.append(row["ks_pvalue"].iloc[0])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sec, ratio, marker="o", color="steelblue", label="ratio_mean")
    ax1.axhline(1.0, color="steelblue", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("window before SBL (s)")
    ax1.set_ylabel("ratio_mean (window / overall)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(sec, pval, marker="s", color="firebrick", label="ks_pvalue")
    ax2.axhline(0.05, color="firebrick", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("ks_pvalue (log scale)", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")

    plt.title(f"[{df_name}] {feature}: ratio_mean & ks_pvalue vs. window size")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({"window": windows, "window_sec": sec, "ratio_mean": ratio, "ks_pvalue": pval})


def plot_ratio_pvalue_all_features(results: dict, feature_cols, df_name="data"):
    """Same as plot_ratio_pvalue_vs_window, but one subplot per feature in a grid, so you can
    compare all features at once instead of calling the single-feature version repeatedly."""
    windows = sorted(results.keys(), key=_window_to_sec)
    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax1, feature in zip(axes, feature_cols):
        sec, ratio, pval = [], [], []
        for w in windows:
            r = results[w]
            df_res = r[0] if isinstance(r, tuple) else r
            row = df_res[df_res["feature"] == feature]
            if row.empty:
                continue
            sec.append(_window_to_sec(w))
            ratio.append(row["ratio_mean"].iloc[0])
            pval.append(row["ks_pvalue"].iloc[0])

        ax1.plot(sec, ratio, marker="o", color="steelblue")
        ax1.axhline(1.0, color="steelblue", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_xscale("log")
        ax1.set_xlabel("window (s)")
        ax1.set_ylabel("ratio_mean", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_title(feature, fontsize=10)

        ax2 = ax1.twinx()
        ax2.plot(sec, pval, marker="s", color="firebrick")
        ax2.axhline(0.05, color="firebrick", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_yscale("log")
        ax2.set_ylabel("ks_pvalue", color="firebrick")
        ax2.tick_params(axis="y", labelcolor="firebrick")

    fig.suptitle(f"[{df_name}] ratio_mean & ks_pvalue vs. window size, all features")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- correlation
def _plot_corr_heatmap(corr: pd.DataFrame, title: str):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.6 * n + 2, 0.6 * n + 2))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def correlation_matrix(df: pd.DataFrame, feature_cols, df_name="data", method="pearson") -> pd.DataFrame:
    """Correlation between all features of `df` (all rows), plotted as a heatmap titled with df_name."""
    corr = df[feature_cols].corr(method=method)
    _plot_corr_heatmap(corr, f"Correlation - {df_name}")
    return corr


def correlation_before_sbl(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times, time_before,
                           df_name="data", method="pearson") -> pd.DataFrame:
    """Correlation between all features of `df`, using only rows inside `time_before` before each
    SBL. Plotted as a heatmap titled with df_name and the window used."""
    seconds_to_next_sbl = time_before_next_sbl(df, ts_col, sbl_times)
    win_sec = pd.Timedelta(time_before).total_seconds()
    window_df = df.loc[seconds_to_next_sbl <= win_sec]
    corr = window_df[feature_cols].corr(method=method)
    _plot_corr_heatmap(corr, f"Correlation ({time_before} before SBL) - {df_name}")
    return corr


# ---------------------------------------------------------------- the distribution functions behind the KS test
def _find_feature(blocks: dict, name: str):
    """Locate a feature in the blocks. Exact column name first; otherwise a column with the same
    underscore-separated words in another order (e.g. 'inj_efficiency_ler' -> 'ler_inj_efficiency')."""
    for b, df in blocks.items():
        if name in df.columns:
            return b, name
    words = frozenset(name.split("_"))
    hits = [(b, c) for b, df in blocks.items() for c in df.columns if frozenset(str(c).split("_")) == words]
    if len(hits) == 1:
        print(f"  note: '{name}' not found literally, using column '{hits[0][1]}' of block '{hits[0][0]}'")
        return hits[0]
    raise KeyError(f"Feature '{name}' not found in the given blocks" +
                   (f" (ambiguous: {hits})" if hits else "") + ".")


def _dist_to_nearest_sbl(df, ts_col, sbl_times) -> np.ndarray:
    """Seconds from each row to the nearest SBL, before or after it."""
    t = df[ts_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    sb = np.sort(_as_sbl_series(sbl_times).to_numpy(dtype="datetime64[ns]").astype("int64"))
    i = np.searchsorted(sb, t)
    prev = np.abs(t - sb[np.clip(i - 1, 0, len(sb) - 1)])
    nxt = np.abs(sb[np.clip(i, 0, len(sb) - 1)] - t)
    return np.minimum(prev, nxt) / 1e9


def plot_ecdf_window_vs_all(blocks: dict, ts_col: str, features, sbl_times, time_before="5min",
                            quantity="diff", kind="ecdf", reference="all", normal_margin="15min",
                            use_abs=True, xscale=None, bins=60, n_cols=3, dpi=300,
                            window_colormap="Dark2") -> tuple:
    """Distribution of a feature for the WINDOW before the SBLs versus a REFERENCE, both in one panel.

    quantity: "diff"  = the row-to-row change |v(t_i) - v(t_(i-1))| - exactly the values of the KS test
                        in plot_diff_distributions (F_Fenster vs. F_Gesamt);
              "value" = the feature VALUES v(t_i) themselves.
    kind:     "ecdf" = empirical distribution functions (the KS statistic D is the largest vertical
                        distance, drawn as a black line for a single window);
              "hist" = density histograms with shared bins (only for quantity="value").
    reference: "all"    = every row of the block (F_Gesamt; this contains the window rows too);
               "normal" = only rows farther than `normal_margin` from EVERY SBL, i.e. Normalbetrieb
                          without the windows themselves. Choose normal_margin >= the window.
               If your data still contain times without beam, "all" mixes in "no beam" - then use
               "normal" on data that are already filtered to BEAM_ON.
    time_before: one window ("5min") or a list (["1min", "5min", "30min"]) - one curve per window.
    xscale:   None = "symlog" for quantity="diff" (shows exact zeros and heavy tails) and "linear" for
              "value"; "log" is also possible. Ignored for kind="hist".
    The values are the rows of the block; the window selects rows whose time to the next SBL is <= window.
    NOTE: for quantity="value" the p-value in the table is not meaningful (values change slowly, so
    neighbouring rows are almost identical); only D is shown in the title.
    Returns (figure, table).
    """
    if quantity not in ("diff", "value"):
        raise ValueError("quantity must be 'diff' or 'value'.")
    if kind not in ("ecdf", "hist"):
        raise ValueError("kind must be 'ecdf' or 'hist'.")
    if reference not in ("all", "normal"):
        raise ValueError("reference must be 'all' or 'normal'.")
    if kind == "hist" and quantity == "diff":
        raise ValueError("kind='hist' is only offered for quantity='value' (the change distribution is a spike "
                         "at 0 plus a long tail - use kind='ecdf' there).")
    use_abs = use_abs and quantity == "diff"
    xscale = xscale or ("symlog" if quantity == "diff" else "linear")
    margin_s = pd.Timedelta(normal_margin).total_seconds()
    windows = [time_before] if isinstance(time_before, str) else list(time_before)
    feats = list(dict.fromkeys(features))                       # drop duplicates, keep the order
    located = [(f,) + _find_feature(blocks, f) for f in feats]

    n_rows = int(np.ceil(len(located) / n_cols))
    n_cols = min(n_cols, len(located))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.9 * n_rows), dpi=dpi,
                             squeeze=False, layout="constrained")
    axes = axes.ravel()
    cmap = plt.get_cmap(window_colormap)
    ref_name = "F_Gesamt" if reference == "all" else "F_Normal"
    ref_desc = "alle Zeilen" if reference == "all" else f">{normal_margin} von jedem SBL"
    rows, cache = [], {}

    for ax, (feat, block, col) in zip(axes, located):
        if block not in cache:                                  # distances once per block
            df = blocks[block]
            if quantity == "diff":
                df = add_diff_columns(df, [c for c in df.select_dtypes("number").columns if c != ts_col])
            cache[block] = (df, time_before_next_sbl(df, ts_col, sbl_times), _dist_to_nearest_sbl(df, ts_col, sbl_times))
        d, sec, dist = cache[block]
        series = d[f"{col}_diff"] if quantity == "diff" else d[col]
        if use_abs:
            series = series.abs()
        ref_vals = (series if reference == "all" else series[dist > margin_s]).dropna()
        a_sorted = np.sort(ref_vals.to_numpy())
        if len(a_sorted) < 2:
            ax.text(0.5, 0.5, "too few reference values", transform=ax.transAxes, ha="center", color="gray")
            ax.set_title(feat, fontsize=10)
            continue
        ref_label = f"{ref_name} ({ref_desc}, m={len(a_sorted)})"

        win_sorted = []
        for w in windows:
            wv = series[sec <= pd.Timedelta(w).total_seconds()].dropna()
            win_sorted.append(np.sort(wv.to_numpy()))
        pooled = np.concatenate([a_sorted] + win_sorted)

        if kind == "hist":
            lo, hi = np.percentile(pooled, [0.5, 99.5])
            if not hi > lo:
                lo, hi = float(pooled.min()), float(pooled.max()) + 1e-12
            edges = np.linspace(lo, hi, bins + 1)
            ax.hist(a_sorted, bins=edges, density=True, histtype="stepfilled", color="#377eb8", alpha=0.3)
            ax.hist(a_sorted, bins=edges, density=True, histtype="step", color="#377eb8", lw=1.6, label=ref_label)
            ax.set_xlim(lo, hi)
            ax.set_ylabel("Dichte", fontsize=8)
        else:
            ax.step(np.r_[a_sorted[0], a_sorted], np.r_[0, np.arange(1, len(a_sorted) + 1) / len(a_sorted)],
                    where="post", color="#377eb8", lw=2.0, label=ref_label, zorder=3)
            ax.set_ylabel("Verteilungsfunktion F(x)", fontsize=8)
            ax.set_ylim(-0.02, 1.02)

        for k, (w, w_sorted) in enumerate(zip(windows, win_sorted)):
            if len(w_sorted) < 2:
                rows.append(dict(feature=feat, block=block, window=w, n_window=len(w_sorted), n_all=len(a_sorted)))
                ax.text(0.5, 0.5 - 0.08 * k, f"{w}: too few values ({len(w_sorted)})", transform=ax.transAxes,
                        ha="center", color="gray")
                continue
            stat, p = ks_2samp(w_sorted, a_sorted)
            xs = np.unique(np.concatenate([w_sorted, a_sorted]))
            gap = np.abs(np.searchsorted(w_sorted, xs, side="right") / len(w_sorted)
                         - np.searchsorted(a_sorted, xs, side="right") / len(a_sorted))
            x_at_d = float(xs[int(np.argmax(gap))])
            color = "#e6550d" if len(windows) == 1 else cmap(k)
            w_label = (f"Fenster (letzte {w} vor SBL, n={len(w_sorted)})" if len(windows) == 1
                       else f"Fenster {w} (n={len(w_sorted)}, D={stat:.2f})")
            if kind == "hist":
                ax.hist(w_sorted, bins=edges, density=True, histtype="stepfilled", color=color, alpha=0.3)
                ax.hist(w_sorted, bins=edges, density=True, histtype="step", color=color, lw=1.6, label=w_label)
            else:
                ax.step(np.r_[w_sorted[0], w_sorted], np.r_[0, np.arange(1, len(w_sorted) + 1) / len(w_sorted)],
                        where="post", color=color, lw=1.8, zorder=4, label="F_" + w_label)
                if len(windows) == 1:
                    lo_y = np.searchsorted(a_sorted, x_at_d, side="right") / len(a_sorted)
                    hi_y = np.searchsorted(w_sorted, x_at_d, side="right") / len(w_sorted)
                    ax.vlines(x_at_d, min(lo_y, hi_y), max(lo_y, hi_y), color="black", lw=1.6, zorder=5)
                    ax.annotate(f"D = {stat:.3f}", (x_at_d, (lo_y + hi_y) / 2), xytext=(6, 0),
                                textcoords="offset points", fontsize=8, va="center", zorder=6)
            rows.append(dict(feature=feat, block=block, column=col, quantity=quantity, reference=reference,
                             window=w, n_window=len(w_sorted), n_all=len(a_sorted), D=stat, p_value=p,
                             x_at_D=x_at_d, frac_zero_window=float((w_sorted == 0).mean()),
                             frac_zero_all=float((a_sorted == 0).mean()),
                             median_window=float(np.median(w_sorted)), median_all=float(np.median(a_sorted))))

        if kind == "ecdf":
            pos = pooled[pooled > 0]
            if xscale == "symlog" and len(pos):
                ax.set_xscale("symlog", linthresh=float(pos.min()))
            elif xscale == "log" and len(pos):
                ax.set_xscale("log")
        mine = [r for r in rows if r["feature"] == feat and "D" in r]
        title = feat if col == feat else f"{feat}  ({col})"
        if len(windows) == 1 and mine:
            title += (f"\nD = {mine[0]['D']:.3f},  p = {mine[0]['p_value']:.2g}" if quantity == "diff"
                      else f"\nD = {mine[0]['D']:.3f}")
        ax.set_title(title, fontsize=10)
        if quantity == "diff":
            ax.set_xlabel(r"$|\Delta v| = |v(t_i) - v(t_{i-1})|$  (Änderung zwischen zwei Zeilen)" if use_abs
                          else r"$\Delta v = v(t_i) - v(t_{i-1})$  (Änderung zwischen zwei Zeilen)", fontsize=8)
        else:
            ax.set_xlabel(f"Wert von {col}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right" if kind == "ecdf" else "upper right")

    for ax in axes[len(located):]:
        ax.set_visible(False)
    what = "der Änderungen zwischen zwei Zeilen (nicht der Werte selbst)" if quantity == "diff" else "der Werte"
    how = "Verteilungsfunktionen" if kind == "ecdf" else "Histogramme (Dichte)"
    fig.suptitle(f"{how} {what}: Fenster vor SBL vs. {'alle Zeilen' if reference == 'all' else 'Normalbetrieb'}"
                 + ("   (D = größter senkrechter Abstand = KS-Statistik)" if kind == "ecdf" else ""), fontsize=10, wrap=True)
    plt.show()
    return fig, pd.DataFrame(rows)


# same function, name that fits the value / histogram use as well
plot_feature_distribution_window_vs_all = plot_ecdf_window_vs_all
