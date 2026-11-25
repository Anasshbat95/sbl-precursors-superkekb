"""
sbl_vs_control.py - Steps 4 to 9: every SBL compared with ITS OWN control times.

Control times of an SBL at t: t - 1 h, ..., t - 6 h (same day, same operating conditions, same minute of the
hour). A control time is dropped if it lies within `guard` (default 30 min) before or after ANY SBL, so it is
neither in a pre-SBL window nor in the recovery after an earlier SBL.

    4  paired_comparison   per block, feature and measure: in how many SBLs is the value HIGHER than the
                           SBL's own control values? Per SBL: score = share of its control values below the SBL
                           value (ties 1/2); the SBL counts as "higher" if score > 0.5, "lower" if < 0.5.
                           About 50 % = nothing special, also for counts with ties.
                           measures (computed from the last n rows before the time, per feature):
                             level    median of the n values
                             trend    slope over the n rows (per hour)
                             unrest   standard deviation of the non-zero row-to-row changes (size of jumps)
                             changes  number of non-zero row-to-row changes (how often it reports)
                             stale    minutes since the feature last changed its value ("freeze")
    5  ring_groups         split the SBLs by the ring that lost its beam (current_around_sbl) and run step 4
                           per group with compare_groups
    6  plot_freeze         the "stale" measure of step 4: median minutes since last change, SBL vs. control
    7  last_channel        which feature changed LAST before the time: share at SBLs vs. at control times
    8  classify            can a simple model tell "last n rows before an SBL" from "before a control time"?
                           out-of-sample, folds = blocks of consecutive SBLs (each SBL with its controls),
                           AUC with bootstrap interval over SBLs, permutation importance
    9  robustness          step 4 again with n = 5 / 10 / 20, without full-hour SBLs, without SBLs closer than
                           `min_gap` to the previous SBL, and for the 1st / 2nd half of the run
    run_all                everything in one go, all figures saved to save_dir

All p- and q-values are descriptive (many tests, thresholds chosen by us).

Usage:
    from sbl_vs_control import run_all
    out = run_all(blocks_on, sbl, beam_block="beam", save_dir="Abbildung/sbl_vs_control")
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

SHIFTS = ("1h", "2h", "3h", "4h", "5h", "6h")
MEASURES = ("level", "trend", "unrest", "changes", "stale")
SBL_COLOR, CTRL_COLOR = "#c0392b", "#7f8c8d"
STALE_MAX = "2h"          # 'stale' is capped here, and control times must be this much cleaner than after_sbl


# ----------------------------------------------------------------------------------------------- helpers
def _ns(x):
    return np.sort(pd.to_datetime(pd.Series(x)).to_numpy(dtype="datetime64[ns]").astype("int64"))


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        r = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty(len(pv))
        out[order] = np.minimum(r, 1)
        q[ok] = out
    return q


def _err(p, lo, hi):
    p, lo, hi = (np.asarray(x, float) for x in (p, lo, hi))
    return [np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)]


def _save(fig, save_dir, name):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches="tight")


def control_times(sbl_ns, shifts=SHIFTS, guard="30min", iv=None, after_sbl=None, clean=STALE_MAX):
    """Returns (times, group) where group = index of the SBL a control time belongs to.
    iv: optional interval table from operation_state.operation_intervals; then only control times in
    state BEAM_ON are kept (otherwise a control time inside a beam-off gap looks 'frozen')."""
    g = pd.Timedelta(guard).value
    times, group = [], []
    for k, sh in enumerate(shifts):
        c = sbl_ns - pd.Timedelta(sh).value
        times.append(c)
        group.append(np.arange(len(sbl_ns)))
    times, group = np.concatenate(times), np.concatenate(group)
    i = np.searchsorted(sbl_ns, times)
    nxt = np.abs(sbl_ns[np.clip(i, 0, len(sbl_ns) - 1)] - times)
    prv = np.abs(times - sbl_ns[np.clip(i - 1, 0, len(sbl_ns) - 1)])
    keep = np.minimum(nxt, prv) > g
    if after_sbl:
        # a control time must be as 'clean' as a typical SBL: at least after_sbl + clean after the previous SBL
        kp = np.searchsorted(sbl_ns, times, side="left") - 1
        dist_prev = np.where(kp >= 0, times - sbl_ns[np.clip(kp, 0, None)], np.inf)
        keep &= dist_prev > pd.Timedelta(after_sbl).value + pd.Timedelta(clean).value
    if iv is not None:
        idx = pd.IntervalIndex.from_arrays(pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"]), closed="left")
        pos = idx.get_indexer(pd.to_datetime(times))
        state = np.where(pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA")
        keep &= state == "BEAM_ON"
    return times[keep], group[keep]


def beam_present(blocks, times, spec, ts_col="timestamp"):
    """True where the beam is on at the time, decided directly from the beam block (no operation_state):
    the LAST row before the time must have at least one of spec["features"] above spec["thr"], and must not be
    older than spec["max_age"]. spec e.g. {"block": "beam", "features": ("her_current", "ler_current"), "thr": 50}."""
    d = blocks[spec["block"]].sort_values(ts_col)
    t = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
    on = (d[list(spec.get("features", ("her_current", "ler_current")))].to_numpy(float) > spec.get("thr", 50.0)).any(axis=1)
    k = np.searchsorted(t, times, side="left") - 1
    age = np.where(k >= 0, times - t[np.clip(k, 0, None)], np.inf)
    return (k >= 0) & on[np.clip(k, 0, None)] & (age <= pd.Timedelta(spec.get("max_age", "1h")).value)


def _block_measures(df, times, n, ts_col, sbl_ns=None, after_ns=0, stale_max=STALE_MAX):
    """Per time and feature: level, trend, unrest, changes (from the last n rows) and stale.
    If sbl_ns is given, only data AFTER (previous SBL + after_ns) are used: a window whose n rows reach back
    into the time right after an earlier SBL gets NaN, and 'stale' counts at most back to that limit.
    Returns {measure: DataFrame (times x features)} and the feature names."""
    d = df.sort_values(ts_col)
    cols = [c for c in d.select_dtypes("number").columns if c != ts_col]
    t = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
    v = d[cols].to_numpy(float)
    end = np.searchsorted(t, times, side="left")
    idx = end[:, None] - n + np.arange(n)[None, :]               # oldest ... newest
    full = idx[:, 0] >= 0
    limit = np.full(len(times), np.iinfo(np.int64).min)
    if sbl_ns is not None and len(sbl_ns):
        kp = np.searchsorted(sbl_ns, times, side="left") - 1     # previous SBL strictly before the time
        limit = np.where(kp >= 0, sbl_ns[np.clip(kp, 0, None)] + after_ns, limit)
        full &= t[np.clip(idx[:, 0], 0, None)] > limit
    idx = np.clip(idx, 0, None)
    vals = v[idx]                                                # times x n x f
    vals[~full] = np.nan
    tm = (t[idx] - times[:, None]) / 3.6e12                      # hours relative to the time (negative)
    tm = np.where(full[:, None], tm, np.nan)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["level"] = np.nanmedian(vals, axis=1)
        x = tm[:, :, None] - np.nanmean(tm, axis=1)[:, None, None]
        y = vals - np.nanmean(vals, axis=1)[:, None, :]
        varx = np.nansum(x * x, axis=1)
        out["trend"] = np.where(varx > 0, np.nansum(x * y, axis=1) / np.where(varx > 0, varx, 1), np.nan)
        dif = np.diff(vals, axis=1)
        nz = np.where(dif != 0, np.abs(dif), np.nan)
        cnt = np.sum(np.isfinite(nz), axis=1).astype(float)
        out["unrest"] = np.where(cnt >= 2, np.nanstd(nz, axis=1), np.nan)
        out["changes"] = np.where(full[:, None], cnt, np.nan)
    # stale: minutes since the last value change of each feature (not limited to n rows)
    stale = np.full((len(times), len(cols)), np.nan)
    for j in range(len(cols)):
        col = v[:, j]
        ch = np.r_[False, (col[1:] != col[:-1]) & np.isfinite(col[1:])]
        tc = t[ch]
        k = np.searchsorted(tc, times, side="left") - 1
        # minutes since the last change, capped at stale_max for EVERY time (SBL and control alike);
        # NaN if [time - stale_max, time] reaches into the time right after an earlier SBL
        cap = pd.Timedelta(stale_max).value
        since = np.where(k >= 0, times - tc[np.clip(k, 0, None)], np.inf)
        clean = (times - cap) > limit
        stale[:, j] = np.where(clean & np.isfinite(since), np.minimum(since, cap) / 6e10,
                               np.where(clean, cap / 6e10, np.nan))
    out["stale"] = stale
    return {m: pd.DataFrame(a, columns=cols) for m, a in out.items()}, cols


# ----------------------------------------------------------------------------------------------- step 4
def _paired_stats(a, cvals, grp, n_sbl, dev_c, ref, rng, n_perm, min_pairs, n_boot=1000):
    """a: value per SBL; cvals/grp: control values and their SBL. Returns a dict of statistics."""
    out = dict(n_sbl=0, higher=0, lower=0, ties=0, mean_score=np.nan, frac_higher=np.nan, ci_low=np.nan,
               ci_high=np.nan, p_perm=np.nan, median_effect=np.nan, median_sbl=np.nan, median_control=np.nan)
    okc = np.isfinite(cvals)
    members = {}
    for g in np.unique(grp[okc]):
        if np.isfinite(a[g]):
            members[g] = np.r_[a[g], cvals[okc & (grp == g)]]          # first entry = the SBL
    if len(members) < min_pairs:
        out["n_sbl"] = len(members)
        return out
    m_max = max(len(v) for v in members.values())
    G = np.full((len(members), m_max), np.nan)                         # score of every member vs. the others
    for r, v in enumerate(members.values()):
        d = np.sign(v[:, None] - v[None, :])
        sc = (0.5 + 0.5 * d)
        np.fill_diagonal(sc, np.nan)
        G[r, :len(v)] = np.nanmean(sc, axis=1)
    sizes = np.array([len(v) for v in members.values()])
    obs = G[:, 0]
    T = float(np.mean(obs))
    pick = (rng.random((n_perm, len(sizes))) * sizes).astype(int)       # random 'SBL' per group
    null = G[np.arange(len(sizes))[None, :], pick].mean(axis=1)
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(T - 0.5) - 1e-12)) / (1 + n_perm)
    boots = obs[rng.integers(0, len(obs), (n_boot, len(obs)))].mean(axis=1)
    keys = np.array(list(members.keys()))
    aa, bb = a[keys], ref[keys]
    dc = dev_c[np.isfinite(dev_c)]
    sc_ = 1.4826 * np.median(np.abs(dc - np.median(dc))) if len(dc) >= 20 else np.nan
    if not (sc_ > 0):
        sc_ = np.std(dc) if len(dc) >= 20 and np.std(dc) > 0 else np.nan
    ok = np.isfinite(bb)
    out.update(n_sbl=len(members), higher=int((obs > 0.5).sum()), lower=int((obs < 0.5).sum()),
               ties=int((obs == 0.5).sum()), mean_score=T, frac_higher=T,
               ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)), p_perm=float(p),
               median_effect=float(np.median((aa[ok] - bb[ok]) / sc_)) if np.isfinite(sc_) and ok.any() else np.nan,
               median_sbl=float(np.median(aa)), median_control=float(np.nanmedian(bb)))
    return out


def paired_comparison(blocks, sbl_times, n=10, shifts=SHIFTS, guard="30min", ts_col="timestamp",
                      measures=MEASURES, min_pairs=10, return_raw=False, iv=None, match=None, adjust=None,
                      require_beam=None, after_sbl="2h", n_perm=2000, seed=0, verbose=False):
    """iv: keep only BEAM_ON control times (see control_times).
    match: e.g. {"block": "beam", "feature": "ler_current", "tol": 0.05} keeps only control times whose
    level of that feature (median of the last n rows) is within +-tol (relative) of the SBL's own level,
    i.e. compares at the SAME beam current. Removes effects that only follow the current.
    adjust: e.g. {"block": "beam", "feature": "ler_current", "degree": 2, "measures": ("level", "unrest")}: before the
    comparison, the part of every feature that is explained by the beam current (polynomial fitted on the
    control times only) is subtracted, so 'higher before SBL' means 'higher than expected at this current'.
    Keeps all SBLs (unlike match). The adjusting feature itself is skipped.
    require_beam: keep only control times with beam, decided from the beam current itself (see beam_present),
    e.g. {"block": "beam", "features": ("her_current", "ler_current"), "thr": 50}. Use this instead of iv.
    after_sbl: only the time BEFORE SBLs is analysed - data from the first `after_sbl` after any SBL (abort,
    refill) never enter a window, neither before an SBL nor at a control time (None = no limit).
    Statistic per SBL: score = share of its own control values below the SBL value (ties 1/2).
    mean_score over SBLs: 0.5 = nothing special. Interval: bootstrap over SBLs. p: permutation test that
    randomly swaps which value of each SBL group is 'the SBL' (exact, also with ties and counts)."""
    rng = np.random.default_rng(seed)
    after_ns = pd.Timedelta(after_sbl).value if after_sbl else None
    s = _ns(sbl_times)
    ct, grp = control_times(s, shifts, guard, iv, after_sbl)
    if require_beam:
        keep = beam_present(blocks, ct, require_beam, ts_col)
        ct, grp = ct[keep], grp[keep]
    n_before = len(ct)
    if match:
        mb = match["block"]
        ms_, _ = _block_measures(blocks[mb], s, n, ts_col, s if after_ns is not None else None, after_ns or 0)
        mc_, _ = _block_measures(blocks[mb], ct, n, ts_col, s if after_ns is not None else None, after_ns or 0)
        a = ms_[match.get("measure", "level")][match["feature"]].to_numpy()[grp]
        b = mc_[match.get("measure", "level")][match["feature"]].to_numpy()
        keep = np.isfinite(a) & np.isfinite(b) & (np.abs(b - a) <= match.get("tol", 0.05) * np.abs(a))
        ct, grp = ct[keep], grp[keep]
    if verbose:
        print(f"  control times: {len(ct)} kept" + (f" ({n_before} before matching)" if match else "")
              + f", SBLs with >= 1 control: {len(np.unique(grp))}/{len(s)}")
    cov_s = cov_c = None
    if adjust:
        a_s, _ = _block_measures(blocks[adjust["block"]], s, n, ts_col, s if after_ns is not None else None, after_ns or 0)
        a_c, _ = _block_measures(blocks[adjust["block"]], ct, n, ts_col, s if after_ns is not None else None, after_ns or 0)
        cov_s = a_s["level"][adjust["feature"]].to_numpy()
        cov_c = a_c["level"][adjust["feature"]].to_numpy()
    rows, raw = [], {}
    for name, df in blocks.items():
        ms, cols = _block_measures(df, s, n, ts_col, s if after_ns is not None else None, after_ns or 0)
        mc, _ = _block_measures(df, ct, n, ts_col, s if after_ns is not None else None, after_ns or 0)
        if adjust:
            deg = adjust.get("degree", 2)
            for m in adjust.get("measures", ("level", "unrest")):
                for c in cols:
                    if name == adjust["block"] and c == adjust["feature"]:
                        ms[m][c] = np.nan
                        mc[m][c] = np.nan
                        continue
                    y = mc[m][c].to_numpy()
                    ok = np.isfinite(y) & np.isfinite(cov_c)
                    if ok.sum() < 5 * (deg + 1) or np.ptp(cov_c[ok]) == 0:
                        continue
                    x0, xs = np.mean(cov_c[ok]), np.std(cov_c[ok])
                    coef = np.polyfit((cov_c[ok] - x0) / xs, y[ok], deg)
                    mc[m][c] = y - np.polyval(coef, (cov_c - x0) / xs)
                    ms[m][c] = ms[m][c].to_numpy() - np.polyval(coef, (cov_s - x0) / xs)
        raw[name] = (ms, mc)
        for m in measures:
            S, C = ms[m].to_numpy(), mc[m].to_numpy()
            ref = pd.DataFrame(C).groupby(grp).median().reindex(range(len(s))).to_numpy()   # per SBL
            dev_c = C - ref[grp]
            for j, c in enumerate(cols):
                a, b = S[:, j], ref[:, j]
                row = _paired_stats(a, C[:, j], grp, len(s), dev_c[:, j], b, rng, n_perm, min_pairs)
                row = dict(block=name, feature=c, measure=m, **row)
                rows.append(row)
    t = pd.DataFrame(rows)
    t["q_bh"] = _bh(t.get("p_perm", pd.Series(np.nan, index=t.index)))
    t["clear"] = (t["q_bh"] < 0.05) & ((t["ci_low"] > 0.5) | (t["ci_high"] < 0.5))
    t = t.sort_values("q_bh").reset_index(drop=True)
    return (t, raw, ct, grp) if return_raw else t


def plot_paired(t, top=12, title="Step 4", save_dir=None, name="step4_paired_comparison.png", show=True):
    ms = [m for m in MEASURES if m in set(t["measure"])]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 0.33 * top + 2), layout="constrained")
    axes = np.atleast_1d(axes)
    words = {"level": ("higher", "lower"), "trend": ("rising more", "falling more"), "unrest": ("bigger jumps", "smaller"),
             "changes": ("more changes", "fewer"), "stale": ("longer frozen", "shorter")}
    for ax, m in zip(axes, ms):
        sub = t[(t["measure"] == m) & t["frac_higher"].notna()].copy()
        sub["dist"] = (sub["frac_higher"] - 0.5).abs()
        sub = sub.sort_values(["q_bh", "dist"], ascending=[True, False]).head(top).iloc[::-1]
        y = np.arange(len(sub))
        colors = np.where(sub["clear"], SBL_COLOR, "#95a5a6")
        e = _err(sub["frac_higher"], sub["ci_low"], sub["ci_high"])
        for k, col in enumerate(colors):
            ax.errorbar(sub["frac_higher"].iat[k], y[k], xerr=[[e[0][k]], [e[1][k]]], fmt="none", ecolor=col, capsize=3)
        ax.scatter(sub["frac_higher"], y, c=colors, zorder=3)
        ax.axvline(0.5, color="black", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{b}: {f}" for b, f in zip(sub["block"], sub["feature"])], fontsize=7)
        ax.set_xlim(0, 1)
        up, down = words[m]
        ax.set_xlabel(f"score: {up}\nthan own control times", fontsize=8)
        ax.set_title(m, fontsize=11)
    fig.suptitle(f"{title}: each SBL vs. its own control times (dot = mean score = chance that the SBL value is "
                 "above a control value; bar = 95 % bootstrap over SBLs; red = clear after BH; dashed 0.5 = nothing special)",
                 fontsize=9)
    _save(fig, save_dir, name)
    plt.show() if show else plt.close(fig)


# ----------------------------------------------------------------------------------------------- step 5
def ring_groups(beam, sbl_times, ts_col="timestamp", her_col="her_current", ler_col="ler_current", thr=50.0):
    """{group: SBL times} from current_around_sbl.ring_lost ('HER', 'LER', 'both', 'none', 'unknown')."""
    from files.current_around_sbl import current_around_sbl
    tab = current_around_sbl(beam, ts_col, sbl_times, her_col=her_col, ler_col=ler_col, thr=thr)
    return {g: tab.loc[tab["ring_lost"] == g, "sbl_time"] for g in tab["ring_lost"].unique()}


def compare_groups(blocks, groups, min_events=15, top=12, save_dir=None, show=True, **kw):
    out = []
    for g, times in groups.items():
        if len(times) < min_events:
            print(f"  group '{g}': only {len(times)} SBLs - skipped (min_events={min_events})")
            continue
        out.append(paired_comparison(blocks, times, **kw).assign(group=g, n_group=len(times)))
    if not out:
        return pd.DataFrame()
    t = pd.concat(out, ignore_index=True)
    best = (t.dropna(subset=["frac_higher"]).assign(d=lambda x: (x["frac_higher"] - 0.5).abs())
             .sort_values("q_bh").drop_duplicates(["block", "feature", "measure"]).head(top))
    keys = list(zip(best["block"], best["feature"], best["measure"]))
    gs = list(t["group"].unique())
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(keys) + 2), layout="constrained")
    cmap = plt.get_cmap("tab10")
    for k, g in enumerate(gs):
        sub = t[t["group"] == g].set_index(["block", "feature", "measure"]).reindex(keys)
        y = np.arange(len(keys)) + (k - (len(gs) - 1) / 2) * 0.25
        ax.errorbar(sub["frac_higher"], y, xerr=_err(sub["frac_higher"], sub["ci_low"], sub["ci_high"]),
                    fmt="o", color=cmap(k), capsize=2, ms=4, label=f"{g} (n={int(sub['n_group'].max()) if sub['n_group'].notna().any() else 0})")
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{b}: {f} ({m})" for b, f, m in keys], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score: chance that the SBL value is above a control value (0.5 = nothing special)")
    ax.set_title("Step 5: the same comparison per group of SBLs (e.g. ring that lost the beam)", fontsize=10)
    ax.legend(fontsize=8, title="group")
    _save(fig, save_dir, "step5_groups.png")
    plt.show() if show else plt.close(fig)
    return t


# ----------------------------------------------------------------------------------------------- step 6
def plot_freeze(t, top=15, save_dir=None, show=True):
    sub = t[(t["measure"] == "stale") & t["frac_higher"].notna()].copy()
    sub = sub.sort_values("q_bh").head(top).iloc[::-1]
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(sub) + 1.8), layout="constrained")
    ax.barh(y + 0.2, sub["median_sbl"], 0.4, color=SBL_COLOR, label="before SBL")
    ax.barh(y - 0.2, sub["median_control"], 0.4, color=CTRL_COLOR, label="own control times (median)")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b}: {f}  (score {h:.2f})" for b, f, h in zip(sub["block"], sub["feature"], sub["frac_higher"])],
                       fontsize=7)
    ax.set_xlabel("minutes since the value last changed (median over SBLs)")
    ax.set_title("Step 6: does a channel 'freeze' before an SBL?", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, save_dir, "step6_freeze.png")
    plt.show() if show else plt.close(fig)
    return sub.iloc[::-1]


# ----------------------------------------------------------------------------------------------- step 7
def last_channel(blocks, sbl_times, shifts=SHIFTS, guard="30min", ts_col="timestamp", save_dir=None, show=True,
                 iv=None, require_beam=None, after_sbl="2h"):
    """Per block: which feature changed last before the time. Ties (several features changed in the same row,
    e.g. the hourly snapshot) are counted as 'tie'."""
    s = _ns(sbl_times)
    ct, _ = control_times(s, shifts, guard, iv, after_sbl)
    if require_beam:
        ct = ct[beam_present(blocks, ct, require_beam, ts_col)]
    rows = []
    for name, df in blocks.items():
        a_ns = pd.Timedelta(after_sbl).value if after_sbl else None
        ms, cols = _block_measures(df, s, 2, ts_col, s if a_ns is not None else None, a_ns or 0)
        mc, _ = _block_measures(df, ct, 2, ts_col, s if a_ns is not None else None, a_ns or 0)
        res = {}
        for label, st in (("sbl", ms["stale"].to_numpy()), ("ctrl", mc["stale"].to_numpy())):
            lab = np.full(len(st), "none", dtype=object)
            ok = np.isfinite(st).any(axis=1)
            mn = np.nanmin(np.where(np.isfinite(st), st, np.inf), axis=1)
            nmin = (np.isclose(st, mn[:, None], atol=1e-9)).sum(axis=1)
            arg = np.argmin(np.where(np.isfinite(st), st, np.inf), axis=1)
            lab[ok] = np.where(nmin[ok] > 1, "tie", np.array(cols, dtype=object)[arg[ok]])
            res[label] = pd.Series(lab).value_counts()
        for f in sorted(set(res["sbl"].index) | set(res["ctrl"].index)):
            k, kc = int(res["sbl"].get(f, 0)), int(res["ctrl"].get(f, 0))
            lo, hi = _wilson(k, len(s))
            rows.append(dict(block=name, last_changed=f, n_sbl=k, share_sbl=k / len(s), ci_low=lo, ci_high=hi,
                             share_control=kc / len(ct), above_control=lo > kc / len(ct)))
    t = pd.DataFrame(rows)
    names = list(blocks)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4.5), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, nm in zip(axes, names):
        sub = t[t["block"] == nm].sort_values("share_sbl")
        y = np.arange(len(sub))
        ax.barh(y + 0.2, sub["share_sbl"], 0.4, color=SBL_COLOR,
                xerr=_err(sub["share_sbl"], sub["ci_low"], sub["ci_high"]), capsize=2)
        ax.barh(y - 0.2, sub["share_control"], 0.4, color=CTRL_COLOR)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["last_changed"], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(nm, fontsize=10)
    axes[0].legend(["before SBL", "control times"], fontsize=7, loc="lower right")
    fig.suptitle("Step 7: which channel changed LAST before the time? (share of times; 'tie' = several in the same row)",
                 fontsize=10)
    _save(fig, save_dir, "step7_last_channel.png")
    plt.show() if show else plt.close(fig)
    return t.sort_values(["above_control", "share_sbl"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- step 8
def classify(blocks, sbl_times, n=10, shifts=SHIFTS, guard="30min", ts_col="timestamp", n_folds=5,
             model="logistic", n_boot=500, n_perm=10, seed=0, top=15, save_dir=None, show=True, iv=None,
             require_beam=None, after_sbl="2h"):
    """Out-of-sample: train on some SBLs (with their control times), test on the others.
    Folds are blocks of consecutive SBLs (chronological), so no SBL is in training and test at once."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s = _ns(sbl_times)
    ct, grp = control_times(s, shifts, guard, iv, after_sbl)
    if require_beam:
        keep = beam_present(blocks, ct, require_beam, ts_col)
        ct, grp = ct[keep], grp[keep]
    times = np.r_[s, ct]
    y = np.r_[np.ones(len(s)), np.zeros(len(ct))]
    group = np.r_[np.arange(len(s)), grp]
    parts = []
    a_ns = pd.Timedelta(after_sbl).value if after_sbl else None
    for name, df in blocks.items():
        m, cols = _block_measures(df, times, n, ts_col, s if a_ns is not None else None, a_ns or 0)
        for k, a in m.items():
            parts.append(a.add_prefix(f"{name}:").add_suffix(f"|{k}"))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.notna().mean() > 0.2]
    X = X.loc[:, X.std() > 0]
    names = list(X.columns)
    X = X.to_numpy(float)

    def make():
        if model == "forest":
            return make_pipeline(SimpleImputer(strategy="median"),
                                 RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                        class_weight="balanced", random_state=seed))
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))

    fold_of_sbl = np.minimum((np.arange(len(s)) * n_folds) // len(s), n_folds - 1)
    fold = fold_of_sbl[group]
    score = np.full(len(y), np.nan)
    rng = np.random.default_rng(seed)
    imp = np.zeros(len(names))
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        if y[te].min() == y[te].max():
            continue
        mdl = make().fit(X[tr], y[tr])
        score[te] = mdl.predict_proba(X[te])[:, 1]
        base = roc_auc_score(y[te], score[te])
        for j in range(len(names)):                         # permutation importance on the test fold
            drops = []
            for _ in range(n_perm):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - roc_auc_score(y[te], mdl.predict_proba(Xp)[:, 1]))
            imp[j] += np.mean(drops) / n_folds
    ok = np.isfinite(score)
    auc = roc_auc_score(y[ok], score[ok])
    boots = []                                              # bootstrap over SBLs (each with its controls)
    for _ in range(n_boot):
        pick = rng.integers(0, len(s), len(s))
        sel = np.concatenate([np.where(group == g)[0] for g in pick])
        sel = sel[ok[sel]]
        if y[sel].min() != y[sel].max():
            boots.append(roc_auc_score(y[sel], score[sel]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    importance = (pd.DataFrame({"feature": names, "importance": imp})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.3 * top + 2.5), layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1, 1.4]))
    fpr, tpr, _ = roc_curve(y[ok], score[ok])
    a1.plot(fpr, tpr, color=SBL_COLOR, lw=2, label=f"AUC = {auc:.2f}  [{lo:.2f}, {hi:.2f}]")
    a1.plot([0, 1], [0, 1], "--", color="gray", label="chance (AUC = 0.5)")
    a1.set_xlabel("false positive rate (control times called 'SBL')")
    a1.set_ylabel("true positive rate (SBLs found)")
    a1.set_title(f"out-of-sample, {n_folds} chronological folds, model: {model}", fontsize=10)
    a1.legend(fontsize=9, loc="lower right")
    t = importance.head(top).iloc[::-1]
    a2.barh(t["feature"], t["importance"], color=np.where(t["importance"] > 0, SBL_COLOR, CTRL_COLOR))
    a2.axvline(0, color="black", lw=0.8)
    a2.tick_params(axis="y", labelsize=7)
    a2.set_xlabel("loss in AUC when the feature is shuffled (permutation importance)")
    a2.set_title("which inputs does the model use?", fontsize=10)
    fig.suptitle(f"Step 8: can the last {n} rows tell 'before an SBL' from 'before a control time'?", fontsize=11)
    _save(fig, save_dir, f"step8_classifier_{model}.png")
    plt.show() if show else plt.close(fig)
    print(f"Step 8 ({model}): AUC = {auc:.3f}, 95 % bootstrap over SBLs [{lo:.3f}, {hi:.3f}], "
          f"{len(s)} SBLs, {len(ct)} control times, {len(names)} inputs")
    return dict(auc=auc, auc_ci=(lo, hi), importance=importance, n_sbl=len(s), n_control=len(ct))


# ----------------------------------------------------------------------------------------------- step 9
def robustness(blocks, sbl_times, base_n=10, min_gap="6h", top=15, save_dir=None, show=True, **kw):
    s = pd.Series(pd.to_datetime(pd.Series(sbl_times)).sort_values().to_numpy())
    gap_prev = s.diff().dt.total_seconds()
    half = len(s) // 2
    variants = {f"n={base_n}": (s, base_n), "n=5": (s, 5), "n=20": (s, 20),
                "no full hour": (s[s.dt.minute >= 5], base_n),
                f"prev SBL > {min_gap}": (s[(gap_prev.isna()) | (gap_prev > pd.Timedelta(min_gap).total_seconds())], base_n),
                "1st half": (s.iloc[:half], base_n), "2nd half": (s.iloc[half:], base_n)}
    res = []
    for v, (times, n) in variants.items():
        t = paired_comparison(blocks, times, n=n, **kw)
        res.append(t.assign(variant=v, n_sbl_variant=len(times)))
    t = pd.concat(res, ignore_index=True)
    piv = t.pivot_table(index=["block", "feature", "measure"], columns="variant", values="frac_higher")
    piv = piv[[v for v in variants if v in piv.columns]]
    base = t[t["variant"] == f"n={base_n}"].sort_values("q_bh").head(top)
    keys = list(zip(base["block"], base["feature"], base["measure"]))
    sub = piv.reindex(keys)
    fig, ax = plt.subplots(figsize=(1.1 * sub.shape[1] + 5, 0.35 * len(sub) + 2), layout="constrained")
    im = ax.imshow(sub.to_numpy() - 0.5, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(sub.shape[1]))
    ax.set_xticklabels([f"{c}\n(n={int(t.loc[t['variant'] == c, 'n_sbl_variant'].iat[0])})" for c in sub.columns], fontsize=8)
    ax.set_yticks(range(sub.shape[0]))
    ax.set_yticklabels([f"{b}: {f} ({m})" for b, f, m in keys], fontsize=7)
    fig.colorbar(im, ax=ax, label="share higher than control - 0.5")
    ax.set_title("Step 9: does the result of step 4 survive other choices? (number = share of SBLs higher than control)",
                 fontsize=10)
    _save(fig, save_dir, "step9_robustness.png")
    plt.show() if show else plt.close(fig)
    return piv.reset_index()


# ----------------------------------------------------------------------------------------------- all
def run_all(blocks, sbl_times, beam_block="beam", n=10, save_dir="Abbildung/sbl_vs_control", show=True,
            ring_thr=50.0, iv=None, match=None, adjust=None, require_beam=None, after_sbl="2h"):
    """iv: operation-state interval table (recommended: only BEAM_ON control times).
    match: compare at the same beam current, e.g. {"block": "beam", "feature": "ler_current", "tol": 0.05}.
    adjust: subtract what the beam current explains, e.g. {"block": "beam", "feature": "ler_current"}."""
    kw = dict(iv=iv, match=match, adjust=adjust, require_beam=require_beam, after_sbl=after_sbl)
    out = {}
    print("Step 4: paired comparison ...")
    out["step4"] = paired_comparison(blocks, sbl_times, n=n, verbose=True, **kw)
    plot_paired(out["step4"], save_dir=save_dir, show=show)
    if beam_block in blocks:
        print("Step 5: groups by ring ...")
        groups = ring_groups(blocks[beam_block], sbl_times, thr=ring_thr)
        print("  SBLs per group:", {g: len(v) for g, v in groups.items()})
        out["step5"] = compare_groups(blocks, groups, n=n, save_dir=save_dir, show=show, **kw)
    print("Step 6: freeze ...")
    out["step6"] = plot_freeze(out["step4"], save_dir=save_dir, show=show)
    print("Step 7: last channel ...")
    out["step7"] = last_channel(blocks, sbl_times, save_dir=save_dir, show=show, iv=iv, require_beam=require_beam,
                                 after_sbl=after_sbl)
    print("Step 8: classifier ...")
    out["step8"] = classify(blocks, sbl_times, n=n, save_dir=save_dir, show=show, iv=iv, require_beam=require_beam,
                             after_sbl=after_sbl)
    print("Step 9: robustness ...")
    out["step9"] = robustness(blocks, sbl_times, base_n=n, save_dir=save_dir, show=show, **kw)
    clear = out["step4"][out["step4"]["clear"]]
    print(f"\nStep 4: {len(clear)} clear block/feature/measure combinations (BH q < 0.05 and 95 % interval "
          f"not containing 0.5):")
    print(clear[["block", "feature", "measure", "n_sbl", "mean_score", "ci_low", "ci_high", "median_effect", "q_bh"]]
          .round(3).to_string(index=False) if len(clear) else "  none")
    return out
