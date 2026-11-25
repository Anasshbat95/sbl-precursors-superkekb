"""
snapshot_analysis.py - Precursor search on the 2-hour snapshots (thesis chapters 3-5).

The blocks are snapshots every 2 h (bursts of rows at xx:59:58), so minute-scale windows before an SBL are
empty. This module compares the LAST SNAPSHOT before each SBL with normal snapshots of the same week and the
same beam current (matched percentile q), and tests the result against random pseudo-events.

Usage (notebook):
    from files.snapshot_analysis import (sampling_summary, plot_sampling, clock_check, plot_clock_check,
                                         rank_features, plot_ranking, lag_profile, plot_lag_profile)

    sampling_summary(blocks, sbl_times)                 # table: rows/h, SBLs with data in 5 min ...
    plot_sampling(blocks, sbl_times)                    # fig: 2-h gaps + age of last row before SBL
    clock = clock_check(blocks, sbl_times)              # table: drop share per shift
    plot_clock_check(clock)
    rank = rank_features(blocks, sbl_times)             # main result (takes a few minutes)
    plot_ranking(rank, top=15)
    prof = lag_profile(blocks, sbl_times, [("beam", "her_current", "level"), ("beam", "ler_sigma_y", "level")])
    plot_lag_profile(prof)

Every plot function accepts save="name.png" to write the figure into MasterArbeit/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

H = 3.6e12                                   # ns per hour
FIG_DIR = "MasterArbeit/figures/"
COL = {"beam": "#2a78d6", "background": "#eb6834", "injection": "#1baf7a", "vacuum": "#eda100",
       "position": "#e87ba4"}


def _sbl_ns(sbl_times):
    if isinstance(sbl_times, pd.DataFrame):
        sbl_times = sbl_times.iloc[:, 0]
    return np.sort(pd.to_datetime(pd.Series(sbl_times)).to_numpy("datetime64[ns]"))


def _save(fig, save):
    if save:
        fig.savefig(FIG_DIR + save, dpi=200, bbox_inches="tight")


def snapshots(df, ts_col="timestamp", burst_gap="60s"):
    """One row per burst (rows closer than burst_gap): the last (forward-filled) state of the burst."""
    d = df.sort_values(ts_col).copy()
    ts = pd.to_datetime(d[ts_col])
    d["_burst"] = (ts.diff() > pd.Timedelta(burst_gap)).cumsum()
    g = d.groupby("_burst")
    s = g.last()
    s[ts_col] = g[ts_col].max()
    return s.reset_index(drop=True)


class _Beam:
    """Beam current (max of HER/LER) at any time, from the BEAM snapshots."""
    def __init__(self, beam, cols=("her_current", "ler_current")):
        s = snapshots(beam)
        self.t = s["timestamp"].to_numpy("datetime64[ns]")
        self.c = s[list(cols)].max(axis=1).to_numpy()
        self.cur = s[list(cols)].to_numpy()

    def idx(self, t):
        return np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1)


def _prep_block(df, beam, on_thr=50.0):
    s = snapshots(df)
    feats = [c for c in s.columns if c != "timestamp"]
    t = s["timestamp"].to_numpy("datetime64[ns]")
    i = beam.idx(t)
    on = (beam.c[i] > on_thr) & ((t - beam.t[i]).astype("int64") / H < 2.5)
    X = s[feats].to_numpy(float)
    dX = np.vstack([np.full(len(feats), np.nan), np.diff(X, axis=0)])
    dX[~(np.r_[np.nan, np.diff(t).astype("int64") / H] < 2.5)] = np.nan
    return dict(t=t, on=on, c=beam.c[i], X=X, dX=np.abs(dX), feats=feats)


def _ref_mask(P, sbl):
    """normal = beam on, > 6 h before the next SBL and > 2 h after the previous one"""
    j = np.searchsorted(sbl, P["t"])
    to_next = np.where(j < len(sbl), (sbl[np.minimum(j, len(sbl) - 1)] - P["t"]).astype("int64") / H, np.inf)
    since = np.where(j > 0, (P["t"] - sbl[np.maximum(j - 1, 0)]).astype("int64") / H, np.inf)
    return P["on"] & (to_next > 6) & (since > 2), since


def _snap_pct(P, kind, ref, days=7, cur_tol=0.1, min_ref=10):
    """matched percentile q of every snapshot for every feature (same week, same current)"""
    M = P["X"] if kind == "level" else P["dX"]
    t, c = P["t"], P["c"]
    out = np.full(M.shape, np.nan)
    for i in np.where(P["on"])[0]:
        m = ref & (np.abs((t - t[i]).astype("int64")) / (24 * H) <= days) & (np.abs(c - c[i]) <= cur_tol * max(c[i], 1))
        m[i] = False
        R = M[m]
        if len(R) < min_ref:
            continue
        v = M[i]
        cnt = np.isfinite(R).sum(0)
        pct = ((R < v).sum(0) + 0.5 * (R == v).sum(0)) / np.maximum(cnt, 1)
        pct[(cnt < min_ref) | ~np.isfinite(v)] = np.nan
        out[i] = pct
    return out


def _pick(P, events, since, kth=1):
    """index of the kth snapshot before each event (only if young enough, beam on, not right after an SBL)"""
    t = P["t"]
    i = np.searchsorted(t, events) - kth
    ok = i >= 0
    i = np.where(ok, i, 0)
    age = (events - t[i]).astype("int64") / H
    ok &= (age < 2.5 * kth) & P["on"][i] & (since[i] > 1)
    return i[ok]


def _pseudo_times(beam, sbl, min_dist_h=8, step="10min"):
    grid = np.arange(beam.t[0], beam.t[-1], np.timedelta64(pd.Timedelta(step).value, "ns"))
    gi = beam.idx(grid)
    on = (beam.c[gi] > 50) & ((grid - beam.t[gi]).astype("int64") / H < 2.5)
    j = np.searchsorted(sbl, grid)
    d = np.full(len(grid), np.inf)
    for jj in (j - 1, j):
        ok = (jj >= 0) & (jj < len(sbl))
        d[ok] = np.minimum(d[ok], np.abs((grid[ok] - sbl[jj[ok]]).astype("int64")) / H)
    return grid[on & (d > min_dist_h)]


# ---------------------------------------------------------------------------------------- chapter 3
def sampling_summary(blocks, sbl_times, windows=("1min", "5min", "30min", "1h")):
    """Per block: rows per hour, burst gaps, and the share of SBLs with >= 2 rows in each window before."""
    sbl = _sbl_ns(sbl_times)
    rows = []
    for name, d in blocks.items():
        ts = np.sort(pd.to_datetime(d["timestamp"]).to_numpy("datetime64[ns]"))
        s = snapshots(d)["timestamp"]
        r = dict(block=name, rows=len(ts), bursts=len(s),
                 rows_per_h=round(len(ts) / ((ts[-1] - ts[0]).astype("int64") / H), 1),
                 median_gap_min=round(s.diff().dt.total_seconds().median() / 60, 1))
        last = ts[np.clip(np.searchsorted(ts, sbl) - 1, 0, None)]
        r["median_age_last_row_min"] = round(np.median((sbl - last).astype("int64") / 6e10), 1)
        for w in windows:
            n = np.searchsorted(ts, sbl) - np.searchsorted(ts, sbl - np.timedelta64(pd.Timedelta(w).value, "ns"))
            r[f"sbl_with_data_{w}"] = f"{(n >= 2).mean():.0%}"
        rows.append(r)
    return pd.DataFrame(rows)


def plot_sampling(blocks, sbl_times, block="beam", save=None):
    sbl = _sbl_ns(sbl_times)
    s = snapshots(blocks[block])["timestamp"]
    gaps = s.diff().dt.total_seconds().dropna() / 60
    st = s.to_numpy("datetime64[ns]")
    age = (sbl - st[np.clip(np.searchsorted(st, sbl) - 1, 0, None)]).astype("int64") / 6e10
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(np.clip(gaps, 0, 300), bins=np.arange(0, 305, 5), color=COL.get(block, "C0"), edgecolor="white")
    ax[0].set(xlabel="time between two bursts of rows (min, clipped at 300)", ylabel="number of gaps",
              title=f"(a) {block.upper()}: time between snapshots")
    ax[1].hist(age, bins=np.arange(0, 125, 5), color=COL.get(block, "C0"), edgecolor="white")
    ax[1].axvline(np.median(age), color="k", ls="--", lw=1.2)
    ax[1].set(xlabel=f"age of the last {block.upper()} row before an SBL (min)", ylabel="number of SBLs",
              title=f"(b) median {np.median(age):.0f} min")
    plt.tight_layout(); _save(fig, save); plt.show()


def clock_check(blocks, sbl_times, thr=400, shifts_h=np.arange(-9, 9.5, 0.5), n_random=300, seed=0):
    """Share of SBLs where a ring current drops below half between the snapshot before and after the
    (shifted) event. With the right clock the share peaks at shift 0."""
    beam = _Beam(blocks["beam"])
    sbl = _sbl_ns(sbl_times)

    def rate(times):
        i = np.searchsorted(beam.t, times) - 1
        i = i[(i >= 0) & (i + 1 < len(beam.t))]
        i = i[(beam.t[i + 1] - beam.t[i]).astype("int64") / H < 2.5]
        b, a = beam.cur[i], beam.cur[i + 1]
        on = b.max(1) > thr
        return (on & ((a < 0.5 * b) & (b > thr)).any(1)).sum() / max(on.sum(), 1)

    rng = np.random.default_rng(seed)
    cand = _pseudo_times(beam, sbl)
    rand = [rate(np.sort(rng.choice(cand, len(sbl)))) for _ in range(n_random)]
    out = pd.DataFrame({"shift_h": shifts_h,
                        "drop_share": [rate(sbl + np.timedelta64(int(h * 60), "m")) for h in shifts_h]})
    out.attrs["random_band"] = tuple(np.quantile(rand, [0.025, 0.5, 0.975]))
    return out


def plot_clock_check(clock, save=None):
    lo, md, hi = clock.attrs["random_band"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axhspan(lo, hi, color="#e4e3df", label="random times (95% band)")
    ax.plot(clock.shift_h, clock.drop_share, color=COL["beam"], lw=2, marker="o", ms=4, label="SBL list shifted")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set(xlabel="shift added to the SBL timestamps (h)", ylabel="share of events with current drop")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); _save(fig, save); plt.show()


# ---------------------------------------------------------------------------------------- chapter 5
def unmatched_ranking(blocks, sbl_times):
    """Naive comparison (pre-SBL snapshot vs all normal snapshots, Mann-Whitney AUC). Confounded by the
    beam current and the period of the run - shown only to motivate the matched test."""
    beam, sbl = _Beam(blocks["beam"]), _sbl_ns(sbl_times)
    rows = []
    for name, d in blocks.items():
        P = _prep_block(d, beam)
        ref, since = _ref_mask(P, sbl)
        idx = np.unique(_pick(P, sbl, since))
        for kind in ("level", "change"):
            M = P["X"] if kind == "level" else P["dX"]
            for k, f in enumerate(P["feats"]):
                a, r = M[idx, k], M[ref, k]
                a, r = a[np.isfinite(a)], r[np.isfinite(r)]
                if len(a) < 10 or len(r) < 50 or np.std(r) == 0:
                    continue
                u, p = mannwhitneyu(a, r)
                rows.append(dict(block=name, feature=f, kind=kind, n_sbl=len(a), auc=u / (len(a) * len(r)), p=p))
    res = pd.DataFrame(rows)
    res["p_bonf"] = np.minimum(res.p * len(res), 1)
    return res.assign(strength=(res.auc - 0.5).abs()).sort_values("strength", ascending=False)


def rank_features(blocks, sbl_times, n_random=1000, seed=11, verbose=True):
    """Main result: matched percentile of the last snapshot before each SBL, against n_random sets of
    random pseudo-events processed the same way. Also gives the effect of the 2nd-to-last snapshot (k=2)."""
    beam, sbl = _Beam(blocks["beam"]), _sbl_ns(sbl_times)
    rng = np.random.default_rng(seed)
    cand = _pseudo_times(beam, sbl)
    rows = []
    for name, d in blocks.items():
        if verbose:
            print("block", name)
        P = _prep_block(d, beam)
        ref, since = _ref_mask(P, sbl)
        sets = [_pick(P, np.sort(rng.choice(cand, len(sbl), replace=False)), since) for _ in range(n_random)]
        for kind in ("level", "change"):
            PCT = _snap_pct(P, kind, ref)
            null = np.array([np.nanmean(PCT[s], 0) for s in sets])
            res_k = {}
            for kth in (1, 2):
                idx = _pick(P, sbl, since, kth)
                res_k[kth] = (np.nanmean(PCT[idx], 0), np.isfinite(PCT[idx]).sum(0))
            for k, f in enumerate(P["feats"]):
                nl = null[:, k][np.isfinite(null[:, k])]
                obs, n = res_k[1][0][k], res_k[1][1][k]
                if n < 15 or len(nl) < 100:
                    continue
                mu = np.median(nl)
                rows.append(dict(block=name, feature=f, kind=kind, n=int(n), mean_pct=obs, null_median=mu,
                                 null_lo=np.quantile(nl, .025), null_hi=np.quantile(nl, .975), effect=obs - mu,
                                 p_perm=(1 + np.sum(np.abs(nl - mu) >= abs(obs - mu))) / (1 + len(nl)),
                                 effect_k2=res_k[2][0][k] - mu))
    res = pd.DataFrame(rows)
    res["p_bonf"] = np.minimum(res.p_perm * len(res), 1)
    res = res.assign(abs_effect=res.effect.abs()).sort_values("abs_effect", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"{len(res)} tests | p<0.05: {(res.p_perm < 0.05).sum()} (chance: {0.05 * len(res):.1f}) "
              f"| smallest Bonferroni p: {res.p_bonf.min():.2f}")
    return res


def plot_ranking(rank, top=15, save=None):
    from matplotlib.lines import Line2D
    r = rank.head(top).iloc[::-1]
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(8, 0.33 * top + 0.8))
    ax.hlines(y, r.null_lo, r.null_hi, color="#c9c8c2", lw=6)
    ax.scatter(r.null_median, y, color="#8a8984", s=12, zorder=3)
    ax.scatter(r.mean_pct, y, c=[COL.get(b, "C0") for b in r.block], s=55, zorder=4, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, [f"{f} ({'Δ' if k == 'change' else 'level'})" for f, k in zip(r.feature, r.kind)])
    ax.set_xlabel("mean matched percentile of the last snapshot before the SBL")
    h = [Line2D([], [], marker="o", ls="", color=c, label=n.upper(), markersize=7) for n, c in COL.items()]
    ax.legend(handles=h + [Line2D([], [], color="#c9c8c2", lw=6, label="random events, 95%")],
              frameon=False, loc="lower right", fontsize=8)
    plt.tight_layout(); _save(fig, save); plt.show()


def lag_profile(blocks, sbl_times, features, n_random=300, seed=11):
    """features: list of (block, feature, kind). Mean q for the 1st..4th snapshot before the SBL
    (0-2 h, 2-4 h, 4-6 h, 6-8 h) with bootstrap CI, plus the random-event band."""
    beam, sbl = _Beam(blocks["beam"]), _sbl_ns(sbl_times)
    rng = np.random.default_rng(seed)
    cand = _pseudo_times(beam, sbl)
    rows = []
    for b, f, kind in features:
        P = _prep_block(blocks[b], beam)
        ref, since = _ref_mask(P, sbl)
        PCT = _snap_pct(P, kind, ref)
        k = P["feats"].index(f)
        nl = [np.nanmean(PCT[_pick(P, np.sort(rng.choice(cand, len(sbl), replace=False)), since)][:, k])
              for _ in range(n_random)]
        lo, md, hi = np.quantile(nl, [.025, .5, .975])
        for kth in (1, 2, 3, 4):
            x = PCT[_pick(P, sbl, since, kth)][:, k]
            x = x[np.isfinite(x)]
            bs = np.random.default_rng(2).choice(x, (2000, len(x))).mean(1)
            rows.append(dict(block=b, feature=f, kind=kind, kth=kth, hours=f"{2 * kth - 2}-{2 * kth}", n=len(x),
                             mean_pct=x.mean(), ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975),
                             null_lo=lo, null_median=md, null_hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- reporting activity
RAW_FILES = {"beam": "12_BEAM.csv", "background": "11_BACKGROUND.csv", "injection": "14_INJECTION.csv",
             "vacuum": "15_VACUUM.csv", "position": "13_POSITION.csv"}


def load_raw_blocks(dataset_dir="Dataset"):
    """The blocks BEFORE forward filling: a cell is filled only when the channel reported a new value."""
    from files.data_loading import RENAME_MAPS
    out = {}
    for name, f in RAW_FILES.items():
        d = pd.read_csv(f"{dataset_dir}/{f}").rename(columns=RENAME_MAPS[name])
        d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed")
        out[name] = d.sort_values("timestamp").reset_index(drop=True)
    return out


def reporting_activity(raw_blocks, blocks, sbl_times, window=(-300, -10), n_random=5000, seed=9):
    """Per channel: in how many SBLs does the channel REPORT (raw cell not empty) in `window` seconds
    relative to the SBL, compared with random beam-on times > 8 h from any SBL (permutation p-value).
    The window ends 10 s before the SBL so that the rows written by the loss itself are excluded."""
    beam, sbl = _Beam(blocks["beam"]), _sbl_ns(sbl_times)
    rng = np.random.default_rng(seed)
    cand = _pseudo_times(beam, sbl, step="1min")
    lo, hi = (np.timedelta64(int(w), "s") for w in window)
    rows = []
    for name, d in raw_blocks.items():
        ts = d["timestamp"].to_numpy("datetime64[ns]")
        for col in [c for c in d.columns if c != "timestamp"]:
            t_c = ts[d[col].notna().to_numpy()]

            def hits(ev):
                return (np.searchsorted(t_c, ev + hi) - np.searchsorted(t_c, ev + lo)) > 0
            k = int(hits(sbl).sum())
            null = np.array([hits(np.sort(rng.choice(cand, len(sbl), replace=False))).sum() for _ in range(n_random)])
            rows.append(dict(block=name, feature=col, n_sbl=k, share_sbl=k / len(sbl), random_mean=null.mean(),
                             random_hi=np.quantile(null, .975), ratio=k / max(null.mean(), 1e-9),
                             p_perm=(1 + (null >= k).sum()) / (1 + n_random)))
    res = pd.DataFrame(rows)
    res["p_bonf"] = np.minimum(res.p_perm * len(res), 1)
    return res.sort_values("p_perm").reset_index(drop=True)


def activity_profile(raw_blocks, blocks, sbl_times, features,
                     edges=(-3600, -1800, -900, -600, -300, -120, -60, -10, 0, 30, 120), n_random=1000, seed=9):
    """Share of SBLs in which each (block, feature) reports in each time bin around the SBL, with the
    95% band for random times. features: list of (block, feature)."""
    beam, sbl = _Beam(blocks["beam"]), _sbl_ns(sbl_times)
    rng = np.random.default_rng(seed)
    cand = _pseudo_times(beam, sbl, step="1min")
    sets = [np.sort(rng.choice(cand, len(sbl), replace=False)) for _ in range(n_random)]
    rows = []
    for b, f in features:
        d = raw_blocks[b]
        t_c = d["timestamp"].to_numpy("datetime64[ns]")[d[f].notna().to_numpy()]
        for lo, hi in zip(edges[:-1], edges[1:]):
            l, h = np.timedelta64(int(lo), "s"), np.timedelta64(int(hi), "s")

            def share(ev):
                return np.mean((np.searchsorted(t_c, ev + h) - np.searchsorted(t_c, ev + l)) > 0)
            null = np.array([share(s) for s in sets])
            rows.append(dict(block=b, feature=f, lo=lo, hi=hi, share_sbl=share(sbl),
                             null_lo=np.quantile(null, .025), null_median=np.median(null),
                             null_hi=np.quantile(null, .975)))
    return pd.DataFrame(rows)


def plot_activity_profile(prof, save=None):
    feats = list(dict.fromkeys(prof.feature))
    fig, axs = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 3.4), sharey=True, squeeze=False)
    for ax, f in zip(axs[0], feats):
        p = prof[prof.feature == f].reset_index(drop=True)
        x = np.arange(len(p))
        ax.bar(x, p.null_hi - p.null_lo, bottom=p.null_lo, color="#e4e3df", width=0.8, label="random times, 95%")
        ax.scatter(x, p.null_median, color="#8a8984", s=10, zorder=3)
        ax.scatter(x, p.share_sbl, color=COL.get(p.block[0], "C0"), s=45, zorder=4, edgecolor="white", label="SBLs")
        labels = [f"{lo // 60}…{hi // 60} min" if lo % 60 == 0 and hi % 60 == 0 and hi != 0 else f"{lo}…{hi} s"
                  for lo, hi in zip(p.lo, p.hi)]
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set(title=f, xlabel="time relative to the SBL")
        ax.axvline(x[p.hi == 0][0] + 0.5, color="k", lw=0.8, ls=":")
    axs[0][0].set_ylabel("share of events with a report")
    axs[0][0].legend(frameon=False, fontsize=8, loc="upper right")
    plt.tight_layout(); _save(fig, save); plt.show()


def plot_lag_profile(prof, save=None):
    feats = list(dict.fromkeys(prof.feature))
    fig, axs = plt.subplots(1, len(feats), figsize=(3 * len(feats), 3.2), sharey=True, squeeze=False)
    for ax, f in zip(axs[0], feats):
        p = prof[prof.feature == f].sort_values("kth")
        xs = -(2 * p.kth.to_numpy() - 1)
        ax.axhspan(p.null_lo.iloc[0], p.null_hi.iloc[0], color="#e4e3df")
        ax.axhline(p.null_median.iloc[0], color="#8a8984", lw=0.8)
        ax.errorbar(xs, p.mean_pct, yerr=[p.mean_pct - p.ci_lo, p.ci_hi - p.mean_pct],
                    color=COL.get(p.block.iloc[0], "C0"), marker="o", ms=6, lw=2)
        ax.set_xticks(xs, p.hours)
        ax.set(title=f, xlabel="hours before SBL")
    axs[0][0].set_ylabel("mean matched percentile")
    plt.tight_layout(); _save(fig, save); plt.show()
