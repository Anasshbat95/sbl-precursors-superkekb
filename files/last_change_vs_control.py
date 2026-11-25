"""
last_change_vs_control.py - Is the last change of a block closer to an SBL than to an ordinary time?

The plot "last change before SBL" alone cannot say whether short distances are special: a block that
reports every few seconds always has a recent change. This script computes the same quantity at
CONTROL TIMES and compares.

Control times (same as the other control-time methods of the project):
    every SBL t gets the control times t - 1 h, t - 2 h, ..., t - 6 h.
    Full-hour shifts keep the minute of the hour, so the hourly archive snapshot (all channels are
    written at the full hour) affects SBL and control times in the same way.
    A control time c is dropped if
        - an SBL lies in [c - after_sbl - cap, c)   (c is in the aftermath of an earlier SBL, or its
                                                      look-back reaches into it),
        - an SBL lies in (c, c + before_margin]     (c is itself just before an SBL),
        - require_beam is given and the beam is not on at c (last beam row older than max_age or
          both currents <= thr).
Distances are capped at `cap` (default 2 h) for SBL and control times alike.

Statistic per SBL (paired): score = share of its own control distances that are SMALLER than the SBL
distance (ties count 1/2).  score = 0.5: nothing special; score < 0.5: the last change is CLOSER to the
SBL than to its control times.  Reported: mean score, bootstrap 95 % interval over the SBLs, and a
permutation p-value (which member of each group is "the SBL" is swapped at random). The p-value is
descriptive only (several blocks, no multiple-test correction).

Usage:
    from last_change_vs_control import compare_last_change, plot_last_change_vs_control

    blocks = {"Beam": ds_beam, "Background": ds_background, "Injection": ds_injection,
              "Vacuum": ds_vacuum, "Position": ds_position}
    summary, per_sbl, feat_share = compare_last_change(
        blocks, sbl_times["timestamp"],
        require_beam={"df": ds_beam, "cols": ("her_current", "ler_current"), "thr": 400})
    fig = plot_last_change_vs_control(per_sbl, feat_share, summary, save_as="Abbildung/last_change_vs_control.png")
"""
import numpy as np
import pandas as pd

from last_change_before_sbl import last_feature_change_before_sbl


def _ns(x):
    return pd.to_datetime(pd.Series(x)).to_numpy(dtype="datetime64[ns]").astype("int64")


def make_control_times(sbl_times, shifts_h=(1, 2, 3, 4, 5, 6), after_sbl="2h", cap="2h",
                       before_margin="30min", require_beam=None, ts_col="timestamp"):
    """Table with one row per (SBL, control time) that passed all checks: sbl_index, sbl_time, shift_h, time."""
    sbl = pd.to_datetime(pd.Series(sbl_times)).sort_values().reset_index(drop=True)
    s_ns = _ns(sbl)
    back = pd.Timedelta(after_sbl).value + pd.Timedelta(cap).value
    fwd = pd.Timedelta(before_margin).value

    rows = []
    for i, t in enumerate(sbl):
        for h in shifts_h:
            c = t - pd.Timedelta(hours=h)
            c_ns = c.value
            after_earlier = np.any((s_ns >= c_ns - back) & (s_ns < c_ns))
            before_next = np.any((s_ns > c_ns) & (s_ns <= c_ns + fwd))
            if not (after_earlier or before_next):
                rows.append((i, t, h, c))
    ctrl = pd.DataFrame(rows, columns=["sbl_index", "sbl_time", "shift_h", "time"])

    if require_beam is not None and len(ctrl):
        b = require_beam["df"][[ts_col, *require_beam["cols"]]].dropna().sort_values(ts_col)
        b_ns = _ns(b[ts_col])
        cur = b[list(require_beam["cols"])].max(axis=1).to_numpy()
        max_age = pd.Timedelta(require_beam.get("max_age", "2h")).value
        c_ns = _ns(ctrl["time"])
        pos = np.searchsorted(b_ns, c_ns, side="left") - 1
        ok = (pos >= 0)
        p = np.clip(pos, 0, None)
        ok &= (c_ns - b_ns[p]) <= max_age
        ok &= cur[p] > require_beam["thr"]
        ctrl = ctrl[ok].reset_index(drop=True)
    return ctrl


def _last_change(df, ts_col, feature_cols, times, cap_s):
    """Distance (s, capped) and name of the last changed feature before each time."""
    ch = last_feature_change_before_sbl(df, ts_col, feature_cols, times)
    per_time = ch.drop_duplicates("sbl_time")[["sbl_time", "last_changed_feature", "last_changed_distance_sec"]]
    per_time = per_time.set_index("sbl_time").reindex(pd.to_datetime(pd.Series(times)).to_numpy())
    dist = per_time["last_changed_distance_sec"].to_numpy(dtype=float)
    feat = per_time["last_changed_feature"].to_numpy(dtype=object)
    dist = np.where(np.isnan(dist), cap_s, np.minimum(dist, cap_s))   # no change at all -> at least cap
    return dist, feat


def _scores(x_sbl, groups):
    """Paired score per SBL: share of its control values below the SBL value (ties 1/2)."""
    return np.array([((g < x).sum() + 0.5 * (g == x).sum()) / len(g) for x, g in zip(x_sbl, groups)])


def compare_last_change(blocks, sbl_times, ts_col="timestamp", feature_cols=None, shifts_h=(1, 2, 3, 4, 5, 6),
                        after_sbl="2h", cap="2h", before_margin="30min", require_beam=None, min_controls=2,
                        n_boot=2000, n_perm=2000, seed=0):
    """Compare the last change before SBLs with the last change before their control times, per block.

    blocks: {name: DataFrame}. feature_cols: optional {name: [columns]} (default: all numeric columns).
    Returns (summary, per_sbl, feat_share):
        summary    one row per block: n_sbl, mean_score, ci_low, ci_high, p_perm, median distance SBL / control
        per_sbl    one row per (block, SBL): SBL distance, median control distance, score
        feat_share one row per (block, feature): share of SBLs / of control times where this feature changed last
    """
    rng = np.random.default_rng(seed)
    cap_s = pd.Timedelta(cap).total_seconds()
    sbl = pd.to_datetime(pd.Series(sbl_times)).sort_values().reset_index(drop=True)
    ctrl = make_control_times(sbl, shifts_h, after_sbl, cap, before_margin, require_beam, ts_col)

    summaries, per_sbl_all, shares = [], [], []
    for name, df in blocks.items():
        cols = (feature_cols or {}).get(name) or [c for c in df.select_dtypes("number").columns if c != ts_col]
        d_sbl, f_sbl = _last_change(df, ts_col, cols, sbl, cap_s)
        d_ctl, f_ctl = _last_change(df, ts_col, cols, ctrl["time"], cap_s)
        ctrl_b = ctrl.assign(dist=d_ctl, feat=f_ctl)

        keep, groups = [], []
        for i in range(len(sbl)):
            g = ctrl_b.loc[ctrl_b["sbl_index"] == i, "dist"].to_numpy()
            if len(g) >= min_controls:
                keep.append(i)
                groups.append(g)
        keep = np.array(keep, dtype=int)
        n = len(keep)
        if n == 0:
            summaries.append(dict(block=name, n_sbl=0))
            continue
        x = d_sbl[keep]
        sc = _scores(x, groups)
        mean = sc.mean()
        boot = [sc[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # permutation: in every group pick at random which member plays "the SBL"
        pooled = [np.r_[xi, g] for xi, g in zip(x, groups)]
        perm = np.empty(n_perm)
        for k in range(n_perm):
            s = 0.0
            for pg in pooled:
                j = rng.integers(len(pg))
                xs, gs = pg[j], np.delete(pg, j)
                s += ((gs < xs).sum() + 0.5 * (gs == xs).sum()) / len(gs)
            perm[k] = s / n
        p_perm = (1 + np.sum(np.abs(perm - 0.5) >= abs(mean - 0.5))) / (1 + n_perm)

        summaries.append(dict(block=name, n_sbl=n, n_controls=int(sum(len(g) for g in groups)),
                              mean_score=round(mean, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                              p_perm=round(p_perm, 4),
                              median_dist_sbl_s=float(np.median(x)),
                              median_dist_control_s=float(np.median(np.concatenate(groups))),
                              share_capped_sbl=float(np.mean(x >= cap_s))))
        per_sbl_all.append(pd.DataFrame({"block": name, "sbl_index": keep, "sbl_time": sbl.iloc[keep].to_numpy(),
                                         "dist_sbl_s": x, "median_dist_control_s": [np.median(g) for g in groups],
                                         "n_controls": [len(g) for g in groups], "score": sc}))

        used_ctrl = ctrl_b[ctrl_b["sbl_index"].isin(keep)]
        s_sbl = pd.Series(f_sbl[keep]).value_counts(normalize=True)
        s_ctl = used_ctrl["feat"].value_counts(normalize=True)
        sh = pd.DataFrame({"share_sbl": s_sbl, "share_control": s_ctl}).fillna(0.0)
        sh["ratio"] = sh["share_sbl"] / sh["share_control"].replace(0, np.nan)
        shares.append(sh.rename_axis("feature").reset_index().assign(block=name))

    summary = pd.DataFrame(summaries)
    per_sbl = pd.concat(per_sbl_all, ignore_index=True) if per_sbl_all else pd.DataFrame()
    feat_share = (pd.concat(shares, ignore_index=True)[["block", "feature", "share_sbl", "share_control", "ratio"]]
                  .sort_values(["block", "share_sbl"], ascending=[True, False]) if shares else pd.DataFrame())
    print(f"Control times kept: {len(ctrl)} of {len(sbl) * len(shifts_h)}")
    print(summary.to_string(index=False))
    return summary, per_sbl, feat_share


def plot_last_change_vs_control(per_sbl, feat_share, summary, cap="2h", n_feat=6, save_as=None, dpi=300):
    """Top row: per block, ECDF of the distance of the last change before the SBL vs. before the
    control times' median (log x). Bottom row: share of events where each feature changed last."""
    import matplotlib.pyplot as plt
    blocks = list(summary.loc[summary["n_sbl"] > 0, "block"])
    fig, axes = plt.subplots(2, len(blocks), figsize=(4.2 * len(blocks), 7.5), dpi=dpi,
                             squeeze=False, layout="constrained")
    cap_s = pd.Timedelta(cap).total_seconds()
    for j, b in enumerate(blocks):
        ax = axes[0, j]
        d = per_sbl[per_sbl["block"] == b]
        for col, lab, colr in (("dist_sbl_s", "before SBL", "#d62728"),
                               ("median_dist_control_s", "before control times (median)", "#1f77b4")):
            v = np.sort(np.maximum(d[col].to_numpy(), 1.0))
            ax.step(np.r_[v[0], v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post", color=colr, label=lab)
        ax.axvline(cap_s, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("time since last change [s]")
        ax.set_ylabel("cumulative share of SBLs")
        s = summary[summary["block"] == b].iloc[0]
        ax.set_title(f"{b}\nscore = {s['mean_score']:.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}], n = {int(s['n_sbl'])}",
                     fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")

        ax = axes[1, j]
        f = feat_share[feat_share["block"] == b].head(n_feat)
        y = np.arange(len(f))
        ax.barh(y - 0.2, f["share_sbl"], height=0.4, color="#d62728", label="SBL")
        ax.barh(y + 0.2, f["share_control"], height=0.4, color="#1f77b4", label="control")
        ax.set_yticks(y)
        ax.set_yticklabels(f["feature"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("share of events where this feature changed last")
        ax.grid(axis="x", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Last change before SBL vs. before control times (SBL - 1 h ... - 6 h); score 0.5 = no difference",
                 fontsize=10)
    if save_as:
        fig.savefig(save_as, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig
