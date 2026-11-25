"""
determine_sbl_windows.py - Empirically estimate risk_start, buffer_start,
recovery_end and normal_margin from your own data, instead of guessing numbers.

One analysis per parameter:
    risk_start     Onset detection: rolling z-score vs. baseline, per event -> how many
                   seconds before SBL does the signal start deviating?
    buffer_start   Artifact detection: short-window rolling std vs. baseline std, per event
                   -> how many seconds before SBL does the signal become unreliable?
    recovery_end   Distribution comparison: slide a window after SBL, Kolmogorov-Smirnov
                   test against the normal-data distribution -> when does it stop differing?
    normal_margin  Decay analysis: mean deviation from baseline as a function of distance
                   to the nearest SBL -> where does it flatten to the noise floor?

IMPORTANT - sampling rate: all four methods need several rows inside the windows they
scan. If your block's median time between rows is coarser than the window you ask for
(e.g. a 20s window but rows every 15 min), the window is almost always empty and nothing
can be detected - this is a data-resolution limit, not a bug in the method. Call
diagnose_sampling() first, or just use estimate_all(), which measures the median spacing
itself and auto-scales every window to it unless you pass your own value.

Usage:
    from determine_sbl_windows import estimate_all, plot_estimates, diagnose_sampling

    feature_cols = [c for c in BEAM_clean.columns if c != "timestamp"]
    diagnose_sampling(BEAM_clean, "timestamp", SBL_timestamps_Germany["timestamp"])
    result = estimate_all(BEAM_clean, "timestamp", feature_cols, SBL_timestamps_Germany["timestamp"])
    print(result["suggested"])
    plot_estimates(result)
    
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ---------------------------------------------------------------- shared helpers
def _dist_to_sbl(ts, sbl):
    t = np.asarray(ts, dtype="datetime64[ns]").astype("int64")
    s = np.sort(np.asarray(sbl, dtype="datetime64[ns]").astype("int64"))
    i = np.searchsorted(s, t)
    prev = np.abs(t - s[np.clip(i - 1, 0, len(s) - 1)])
    nxt = np.abs(s[np.clip(i, 0, len(s) - 1)] - t)
    return np.minimum(prev, nxt) / 1e9


def median_spacing_sec(df, ts_col) -> float:
    """Median time (seconds) between consecutive rows - the key number that decides which
    window sizes are even possible to evaluate for this block."""
    ts = pd.to_datetime(df[ts_col]).sort_values()
    return float(ts.diff().dt.total_seconds().median())


def diagnose_sampling(df, ts_col, sbl_times, requested_windows_sec=None):
    """Print the block's median row spacing and how many rows typical windows would contain.
    requested_windows_sec: optional dict like {"risk_start": 300, "buffer_start": 30, ...}
    to check specific window sizes; otherwise checks a few common ones."""
    spacing = median_spacing_sec(df, ts_col)
    n_rows = len(df)
    span_days = (pd.to_datetime(df[ts_col]).max() - pd.to_datetime(df[ts_col]).min()).days
    print(f"{n_rows} rows over ~{span_days} days -> median spacing: {spacing:.1f}s "
         f"({spacing/60:.1f} min)")

    windows = requested_windows_sec or {"30s window": 30, "60s window": 60, "5min window": 300,
                                        "10min window": 600, "20min window": 1200}
    for label, sec in windows.items():
        rows_per_window = sec / spacing if spacing > 0 else float("nan")
        flag = "  <-- almost always empty, cannot be resolved at this sampling rate" if rows_per_window < 3 else ""
        print(f"  {label} ({sec}s) -> ~{rows_per_window:.2f} rows on average{flag}")

    if spacing > 60:
        print("\nNote: with row spacing above 1 minute, sub-minute windows (buffer, short-term "
             "noise spikes) cannot be resolved at all - only effects lasting several times the "
             "spacing can be detected. estimate_all() below widens its windows automatically to "
             "account for this, but the result is only as precise as the data allows.")
    return spacing


def baseline_stats(df, ts_col, feature_cols, sbl_times, normal_margin_guess_sec=900):
    """Mean/std per feature, computed only on rows far from every SBL (a reasonable first guess,
    e.g. 900s = 15min). Used to z-score the signal in the other estimators below."""
    dist = _dist_to_sbl(df[ts_col], sbl_times)
    normal = df.loc[dist > normal_margin_guess_sec, feature_cols]
    return normal.mean(), normal.std().replace(0, np.nan)


def _zscore_matrix(df, feature_cols, mean, std):
    return (df[feature_cols] - mean) / std


# ---------------------------------------------------------------- 1. risk_start
def estimate_risk_start(df, ts_col, feature_cols, sbl_times, search_window=None,
                        z_threshold=3.0, persist_points=None, mean=None, std=None,
                        min_rows_per_window=5):
    """Per SBL: walking backward from the event, find the earliest point still part of a
    continuous run of above-threshold z-scores ending at the event. That point is the onset
    -> lead time = SBL - onset. Returns one row per event (NaN if no onset was found.

    search_window / persist_points default to values scaled from the block's own median row
    spacing (so the window always contains enough rows), unless you set them explicitly."""
    if mean is None:
        mean, std = baseline_stats(df, ts_col, feature_cols, sbl_times)
    spacing = median_spacing_sec(df, ts_col)
    if search_window is None:
        search_window = pd.Timedelta(seconds=max(1200, min_rows_per_window * 20 * spacing))
    else:
        search_window = pd.Timedelta(search_window)
    if persist_points is None:
        persist_points = 1 if spacing > 60 else 3

    z = _zscore_matrix(df, feature_cols, mean, std).abs()
    combined = z.max(axis=1)  # worst-case feature at each timestamp
    ts = pd.DatetimeIndex(df[ts_col])

    rows = []
    for sbl in pd.to_datetime(pd.Series(sbl_times)):
        w_start = sbl - search_window
        mask = (ts >= w_start) & (ts < sbl)
        seg = combined[mask]
        seg_ts = ts[mask]
        if len(seg) < persist_points + 1:
            rows.append((sbl, np.nan, np.nan, len(seg)))
            continue
        above = (seg.to_numpy() > z_threshold)
        # scan backward from the event for the earliest point of a run of >=persist_points
        # above-threshold values that is still "connected" back to the event
        onset_i = None
        i = len(above) - 1
        if above[i]:
            below_run = 0
            onset_i = i
            while i >= 0:
                if above[i]:
                    below_run = 0
                    onset_i = i
                else:
                    below_run += 1
                    if below_run >= persist_points:
                        break
                i -= 1
        if onset_i is None:
            rows.append((sbl, np.nan, np.nan, len(seg)))
        else:
            onset_time = seg_ts[onset_i]
            rows.append((sbl, onset_time, (sbl - onset_time).total_seconds(), len(seg)))
    return pd.DataFrame(rows, columns=["sbl_time", "onset_time", "lead_time_sec", "rows_in_window"])


# ---------------------------------------------------------------- 2. buffer_start
def estimate_buffer_start(df, ts_col, feature_cols, sbl_times, search_window=None,
                          short_window=None, spike_factor=3.0, mean=None, std=None,
                          min_rows_per_window=3):
    """Per SBL: short-window rolling std (relative to baseline std) in the last `search_window`
    before the event; buffer_start = how long before SBL it first exceeds `spike_factor` times
    the typical short-window std seen in normal data.

    search_window / short_window default to values scaled from the block's own median row
    spacing, unless set explicitly."""
    if std is None:
        mean, std = baseline_stats(df, ts_col, feature_cols, sbl_times)
    spacing = median_spacing_sec(df, ts_col)
    if short_window is None:
        short_window = pd.Timedelta(seconds=max(20, min_rows_per_window * spacing))
    else:
        short_window = pd.Timedelta(short_window)
    if search_window is None:
        search_window = pd.Timedelta(seconds=max(120, 5 * short_window.total_seconds()))
    else:
        search_window = pd.Timedelta(search_window)

    ts = pd.DatetimeIndex(df[ts_col])
    data = df.set_index(ts)[feature_cols]
    roll_std = data.rolling(short_window, min_periods=min_rows_per_window).std() / std
    combined = roll_std.max(axis=1)

    dist = _dist_to_sbl(df[ts_col], sbl_times)
    quiet_level = combined[(dist > 900)].median()

    rows = []
    for sbl in pd.to_datetime(pd.Series(sbl_times)):
        w_start = sbl - search_window
        seg = combined[(combined.index >= w_start) & (combined.index < sbl)]
        if seg.dropna().empty or pd.isna(quiet_level) or quiet_level == 0:
            rows.append((sbl, np.nan, len(seg.dropna())))
            continue
        spike = seg.index[seg > spike_factor * quiet_level]
        rows.append((sbl, (sbl - spike[0]).total_seconds() if len(spike) else np.nan, len(seg.dropna())))
    return pd.DataFrame(rows, columns=["sbl_time", "buffer_start_sec", "rows_in_window"])


# ---------------------------------------------------------------- 3. recovery_end
def estimate_recovery_end(df, ts_col, feature_cols, sbl_times, search_window=None,
                          step=None, win=None, p_threshold=0.05, consecutive=2,
                          mean=None, std=None, min_rows_per_window=5):
    """Per SBL: slide a window of length `win` forward from the event (step `step`), KS-test each
    feature's values against the normal-data distribution. recovery_end = time of the first window
    where the test stops rejecting "same distribution" (p > p_threshold) for `consecutive` windows
    in a row, for every feature.

    step / win / search_window default to values scaled from the block's own median row
    spacing, unless set explicitly."""
    spacing = median_spacing_sec(df, ts_col)
    if win is None:
        win = pd.Timedelta(seconds=max(60, min_rows_per_window * spacing))
    else:
        win = pd.Timedelta(win)
    if step is None:
        step = pd.Timedelta(seconds=max(30, win.total_seconds() / 2))
    else:
        step = pd.Timedelta(step)
    if search_window is None:
        search_window = pd.Timedelta(seconds=max(1200, 15 * win.total_seconds()))
    else:
        search_window = pd.Timedelta(search_window)

    dist = _dist_to_sbl(df[ts_col], sbl_times)
    normal = df.loc[dist > 900]
    ts = pd.DatetimeIndex(df[ts_col])

    rows = []
    for sbl in pd.to_datetime(pd.Series(sbl_times)):
        w_end = sbl + search_window
        starts = pd.date_range(sbl, w_end, freq=step)
        ok_streak = 0
        recovery_time = np.nan
        max_rows_seen = 0
        for s in starts:
            seg = df[(ts >= s) & (ts < s + win)]
            max_rows_seen = max(max_rows_seen, len(seg))
            if len(seg) < min_rows_per_window:
                ok_streak = 0
                continue
            all_ok = True
            for c in feature_cols:
                ref = normal[c].dropna()
                samp = seg[c].dropna()
                if len(ref) < 10 or len(samp) < 3:
                    continue
                _, p = ks_2samp(samp, ref)
                if p <= p_threshold:
                    all_ok = False
                    break
            ok_streak = ok_streak + 1 if all_ok else 0
            if ok_streak >= consecutive:
                recovery_time = s - step * (consecutive - 1)
                break
        rows.append((sbl, recovery_time,
                    (recovery_time - sbl).total_seconds() if pd.notna(recovery_time) else np.nan,
                    max_rows_seen))
    return pd.DataFrame(rows, columns=["sbl_time", "recovery_end_time", "recovery_end_sec", "max_rows_per_window"])


# ---------------------------------------------------------------- 4. normal_margin
def estimate_normal_margin(df, ts_col, feature_cols, sbl_times, max_dist_sec=None,
                           n_bins=40, noise_tolerance=1.2, mean=None, std=None,
                           min_rows_per_bin=5):
    """Mean |z-score| as a function of distance to the nearest SBL, binned. normal_margin is the
    smallest distance beyond which the curve stays within `noise_tolerance` x the noise floor
    (estimated from the farthest, best-populated bins) for every subsequent bin.

    max_dist_sec defaults to a span wide enough to contain several hundred rows per bin,
    based on the block's median row spacing."""
    spacing = median_spacing_sec(df, ts_col)
    if max_dist_sec is None:
        max_dist_sec = max(3600, n_bins * min_rows_per_bin * spacing)
    if mean is None:
        mean, std = baseline_stats(df, ts_col, feature_cols, sbl_times, normal_margin_guess_sec=max_dist_sec)
    z = _zscore_matrix(df, feature_cols, mean, std).abs().max(axis=1)
    dist = _dist_to_sbl(df[ts_col], sbl_times)

    mask = dist <= max_dist_sec
    bins = np.linspace(0, max_dist_sec, n_bins + 1)
    bin_idx = np.digitize(dist[mask], bins) - 1
    grouped = pd.Series(z[mask].to_numpy()).groupby(bin_idx)
    curve = grouped.mean().reindex(range(n_bins))
    counts = grouped.count().reindex(range(n_bins)).fillna(0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # only use well-populated bins to find the noise floor and the margin
    valid = counts >= min_rows_per_bin
    if valid.sum() < 5:
        return pd.DataFrame({"distance_sec": bin_centers, "mean_abs_z": curve.to_numpy(),
                             "n_rows": counts.to_numpy()}), None

    valid_curve = curve[valid]
    noise_floor = valid_curve.iloc[-max(5, len(valid_curve) // 10):].mean()
    threshold = noise_floor * noise_tolerance
    margin_sec = None
    valid_idx = np.where(valid.to_numpy())[0]
    for pos, i in enumerate(valid_idx):
        remaining = curve.to_numpy()[valid_idx[pos:]]
        if np.nanmax(remaining) <= threshold:
            margin_sec = bin_centers[i]
            break
    return pd.DataFrame({"distance_sec": bin_centers, "mean_abs_z": curve.to_numpy(),
                         "n_rows": counts.to_numpy()}), margin_sec


# ---------------------------------------------------------------- main entry point
def estimate_all(df, ts_col, feature_cols, sbl_times, summary_quantile=0.75, verbose=True):
    """Run all four estimators (auto-scaled to the block's own sampling rate) and suggest one
    value per parameter (the given quantile of the per-event estimates - a higher quantile is
    more conservative / gives a wider window)."""
    spacing = median_spacing_sec(df, ts_col)
    mean, std = baseline_stats(df, ts_col, feature_cols, sbl_times)

    risk = estimate_risk_start(df, ts_col, feature_cols, sbl_times, mean=mean, std=std)
    buf = estimate_buffer_start(df, ts_col, feature_cols, sbl_times, mean=mean, std=std)
    rec = estimate_recovery_end(df, ts_col, feature_cols, sbl_times, mean=mean, std=std)
    curve, normal_margin_sec = estimate_normal_margin(df, ts_col, feature_cols, sbl_times, mean=mean, std=std)

    def q(series):
        s = series.dropna()
        return float(s.quantile(summary_quantile)) if len(s) else np.nan

    suggested = {
        "risk_start_sec": q(risk["lead_time_sec"]),
        "buffer_start_sec": q(buf["buffer_start_sec"]),
        "recovery_end_sec": q(rec["recovery_end_sec"]),
        "normal_margin_sec": normal_margin_sec,
    }

    if verbose:
        n_events = len(pd.Series(sbl_times))
        print(f"Median row spacing: {spacing:.1f}s ({spacing/60:.1f} min)")
        print(f"Events analysed: {n_events}")
        print(f"  risk_start    : {len(risk['lead_time_sec'].dropna())}/{n_events} events had a detectable onset "
             f"(median rows available per search window: {risk['rows_in_window'].median():.1f})")
        print(f"  buffer_start  : {len(buf['buffer_start_sec'].dropna())}/{n_events} events had a detectable spike "
             f"(median rows available per search window: {buf['rows_in_window'].median():.1f})")
        print(f"  recovery_end  : {len(rec['recovery_end_sec'].dropna())}/{n_events} events recovered within the search window "
             f"(median max rows seen per test window: {rec['max_rows_per_window'].median():.1f})")
        print(f"\nSuggested parameters ({summary_quantile:.0%} quantile across events):")
        for k, v in suggested.items():
            print(f"  {k}: {v:.1f}s" if v is not None and not np.isnan(v) else f"  {k}: could not be estimated")

        low_n = {"risk_start_sec": len(risk["lead_time_sec"].dropna()),
                "buffer_start_sec": len(buf["buffer_start_sec"].dropna()),
                "recovery_end_sec": len(rec["recovery_end_sec"].dropna())}
        for k, n in low_n.items():
            if 0 < n < max(10, 0.05 * n_events):
                print(f"  NOTE: '{k}' is based on only {n} event(s) - too few to trust this number, "
                     f"treat it only as a rough order of magnitude.")

        if spacing > 60 and (np.isnan(suggested["buffer_start_sec"]) or risk["rows_in_window"].median() < 3):
            print("\nWith this block's sampling rate (~{:.1f} min between rows), sub-minute effects "
                 "cannot be resolved - 'could not be estimated' here most likely means the effect is "
                 "simply below this block's time resolution, not that it doesn't exist. Consider a "
                 "block with finer sampling (e.g. BEAM instead of POSITION) for buffer_start, or "
                 "accept a wider, less precise window.".format(spacing / 60))

    return dict(risk=risk, buffer=buf, recovery=rec, normal_margin_curve=curve, suggested=suggested,
               median_spacing_sec=spacing)


def plot_estimates(result):
    """Histograms of the per-event estimates, plus the normal_margin decay curve."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(result["risk"]["lead_time_sec"].dropna(), bins=20, color="steelblue")
    axes[0, 0].set_title("risk_start: seconds before SBL where the signal starts deviating")
    axes[0, 0].set_xlabel("seconds before SBL")

    axes[0, 1].hist(result["buffer"]["buffer_start_sec"].dropna(), bins=20, color="darkorange")
    axes[0, 1].set_title("buffer_start: seconds before SBL where short-term noise spikes")
    axes[0, 1].set_xlabel("seconds before SBL")

    axes[1, 0].hist(result["recovery"]["recovery_end_sec"].dropna(), bins=20, color="seagreen")
    axes[1, 0].set_title("recovery_end: seconds after SBL until distribution matches normal")
    axes[1, 0].set_xlabel("seconds after SBL")

    curve = result["normal_margin_curve"]
    well_populated = curve["n_rows"] >= 5 if "n_rows" in curve else slice(None)
    axes[1, 1].plot(curve.loc[well_populated, "distance_sec"], curve.loc[well_populated, "mean_abs_z"],
                    color="firebrick")
    tail = curve.loc[well_populated, "mean_abs_z"]
    if len(tail):
        axes[1, 1].axhline(tail.iloc[-max(1, len(tail)//10):].mean(), color="gray", linestyle="--",
                           label="noise floor")
    if result["suggested"]["normal_margin_sec"] is not None:
        axes[1, 1].axvline(result["suggested"]["normal_margin_sec"], color="black", linestyle=":",
                           label="suggested normal_margin")
    axes[1, 1].set_title("normal_margin: mean |z-score| vs. distance to nearest SBL")
    axes[1, 1].set_xlabel("distance to nearest SBL (s)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
