"""
pipeline_validation.py - Step 5 of the SBL pipeline: validation.

Rules that respect the time dependence and the rarity of SBL events:
    * only events AFTER the calibration period are evaluated (chronological hold-out)
    * metrics are per EVENT, not per row: was there an alarm in [SBL - lookback, SBL - buffer]?
      how many seconds before the SBL was the first one?
    * false alarms are counted only in normal test time (farther than normal_margin from every
      SBL, which also excludes buffer and recovery) and reported per hour of covered time
    * thresholds are quantiles of the calibration-normal scores (no normal distribution assumed)
    * det_ci95 = Wilson interval for the detection rate (stays wide with few test events, and is not
      degenerate at 10/10); lead_ci95 = bootstrap interval for the median lead time
    * "chance_det_rate" = detection rate a random alarm process with the same false-alarm rate
      would reach by luck in the same window; your detection rate must clearly exceed it

Optional injection veto: samples shortly after an injection are ignored (the SBL study you
shared names injection oscillations as the main false-alarm source).
"""
import numpy as np
import pandas as pd

from files.sbl_anomaly import dist_to_sbl


def injection_veto(inj_active, grid_index: pd.DatetimeIndex, grid: str, cfg: dict):
    """Boolean Series on the grid: True = inside an injection veto window. Returns None (and says
    why) if there is no injection signal or if the veto would remove most of the data."""
    if inj_active is None:
        print("  Injection veto skipped: no injection-rate column found.")
        return None
    a = inj_active.astype(float).resample(grid).max()
    a = a.reindex(grid_index, method="ffill", tolerance=pd.Timedelta(cfg["max_gap"])).fillna(0)
    veto = a.rolling(pd.Timedelta(cfg["inj_veto"]), min_periods=1).max() > 0
    if veto.mean() > 0.5:
        print(f"  Injection veto skipped: it would remove {veto.mean():.0%} of the time "
              "(injection is almost always active) - use a different activity definition.")
        return None
    return veto


def _wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n - unlike a bootstrap it stays wide for small n
    and is not degenerate at 0/n or n/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _episodes(times: pd.DatetimeIndex, cooldown: pd.Timedelta) -> int:
    if len(times) == 0:
        return 0
    return 1 + int((pd.Series(times).diff() > cooldown).sum())


def _covered_hours(times: pd.DatetimeIndex, cap: pd.Timedelta) -> float:
    if len(times) < 2:
        return 0.0
    d = pd.Series(times).diff().dropna().clip(upper=cap)
    return float(d.dt.total_seconds().sum()) / 3600


def evaluate_series(fused: pd.Series, sbl, cfg: dict, split: dict, veto=None):
    """Returns (summary per threshold, per-event table)."""
    s = fused.copy()
    if veto is not None:
        s[veto.reindex(s.index, fill_value=False).to_numpy()] = np.nan
    idx = s.index
    margin = pd.Timedelta(cfg["normal_margin"]).total_seconds()
    normal = dist_to_sbl(idx, sbl) > margin
    cal = s[normal & (idx >= split["train_end"]) & (idx < split["calib_end"])].dropna()
    test_norm = s[normal & (idx >= split["calib_end"])].dropna()
    if len(cal) < 20 or len(test_norm) < 20:
        raise ValueError(f"Too few normal points (calibration={len(cal)}, test={len(test_norm)}).")

    lookback, buffer = pd.Timedelta(cfg["lookback"]), pd.Timedelta(cfg["buffer"])
    cooldown, cap = pd.Timedelta(cfg["cooldown"]), pd.Timedelta(cfg["max_gap"])
    events = [t for t in sbl if split["calib_end"] <= t <= idx.max()]
    cover_h = _covered_hours(test_norm.index, cap)
    window_h = (lookback - buffer).total_seconds() / 3600
    rng = np.random.default_rng(cfg["random_state"])

    summary, per_event = [], []
    for q in cfg["thr_quantiles"]:
        thr = float(cal.quantile(q))
        n_fa = _episodes(test_norm.index[(test_norm > thr).to_numpy()], cooldown)
        fa_h = n_fa / cover_h if cover_h > 0 else np.nan

        det, lead = [], []
        n_nodata = 0
        for t in events:
            w = s[t - lookback: t - buffer].dropna()
            if w.empty:
                n_nodata += 1
                continue
            hit = w.index[(w > thr).to_numpy()]
            det.append(len(hit) > 0)
            lead.append((t - hit[0]).total_seconds() if len(hit) else np.nan)
            per_event.append(dict(quantile=q, sbl_time=t, detected=len(hit) > 0, lead_s=lead[-1]))
        det, lead = np.array(det, bool), np.array(lead, float)

        lo, hi = _wilson(int(det.sum()), len(det))
        det_leads = lead[np.isfinite(lead)]
        lead_ci = "n/a"
        if len(det_leads) >= 3:                      # bootstrap over the detected events' lead times
            meds = [np.median(det_leads[rng.integers(0, len(det_leads), len(det_leads))])
                    for _ in range(cfg["n_boot"])]
            lead_ci = "{:.0f}-{:.0f}".format(*np.percentile(meds, [2.5, 97.5]))
        summary.append(dict(
            quantile=q, thr=round(thr, 2), events=len(det), no_data=n_nodata,
            detected=int(det.sum()), det_rate=round(det.mean(), 3) if len(det) else np.nan,
            det_ci95=f"{lo:.2f}-{hi:.2f}" if not np.isnan(lo) else "n/a",
            chance_det_rate=round(1 - np.exp(-fa_h * window_h), 3) if not np.isnan(fa_h) else np.nan,
            median_lead_s=round(float(np.nanmedian(lead)), 1) if np.isfinite(lead).any() else np.nan,
            lead_ci95=lead_ci,
            false_alarms=n_fa, fa_per_h=round(fa_h, 3) if not np.isnan(fa_h) else np.nan,
            covered_h=round(cover_h, 1)))
    return pd.DataFrame(summary), pd.DataFrame(per_event)


def plot_event_locked(fused: pd.Series, sbl, cfg: dict, split: dict, quantile=0.999,
                      lookback="30min", max_events=60):
    """Fused score vs. seconds before SBL for the test events (thin gray = one event, red = median
    over events in 1-minute bins, dashed = alarm threshold). This is the plot that shows from which
    time before an SBL the score starts to rise."""
    import matplotlib.pyplot as plt

    idx = fused.index
    margin = pd.Timedelta(cfg["normal_margin"]).total_seconds()
    normal = dist_to_sbl(idx, sbl) > margin
    cal = fused[normal & (idx >= split["train_end"]) & (idx < split["calib_end"])].dropna()
    thr = cal.quantile(quantile)
    lb = pd.Timedelta(lookback)
    events = [t for t in sbl if split["calib_end"] <= t <= idx.max()][:max_events]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pooled = []
    for t in events:
        w = fused[t - lb: t].dropna()
        if w.empty:
            continue
        x = (w.index - t).total_seconds()
        ax.plot(x, w.values, color="gray", alpha=0.25, lw=0.7)
        pooled.append(pd.Series(w.values, index=np.floor(x / 60) * 60))
    if pooled:
        med = pd.concat(pooled).groupby(level=0).median()
        ax.plot(med.index, med.values, color="firebrick", lw=2, label="median over events")
    ax.axhline(thr, color="black", linestyle="--", label=f"threshold (q={quantile})")
    ax.axvline(0, color="red")
    ax.set_xlabel("seconds relative to SBL (0 = SBL moment)")
    ax.set_ylabel("fused score (1 = 99th pct. of normal)")
    ax.set_title(f"Fused anomaly score before {len(pooled)} test SBL events")
    ax.legend()
    plt.tight_layout()
    plt.show()
