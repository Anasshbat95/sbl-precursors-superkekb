"""
sbl_iforest.py - Standalone unsupervised anomaly detection before SBL events
                 with Isolation Forest (multivariate + one model per feature).

No other project file is needed.  Every data block (beam, background, ...) is analysed
on its OWN time axis: the SBL timestamps are attached to each block's timeline, but the
blocks themselves are never merged with each other.

Workflow
    1  load / clean one block, attach the SBL events (nearest row index + distance to the SBL)
    2  cut the window of PRE_SBL_SECONDS before every SBL (+ POST_SBL_SECONDS after it, only
       for display; the post-SBL part is never used for ranking)
    3  Isolation Forest
         * multivariate model on all features  -> overall_anomaly_score
         * one univariate model per feature     -> <feature>_anomaly_score
       All models are fitted ONLY on "normal" rows (farther than NORMAL_MARGIN_SECONDS from
       every SBL), so no pre-SBL window is ever learned as normal.  Scores are inverted
       (higher = more anomalous) and normalised on the normal rows:
             0 = median of normal operation,  1 = 99th percentile of normal operation
    4  plot per SBL: overall score + top features vs. time, red dashed line = SBL event
    5  plot per SBL: ranking of the most anomalous features (bar chart),
       plus one heatmap over all events (event x feature, peak score)
    +  summary: fraction of SBL windows with score > 1, Wilson 95% interval, and the same
       fraction for random windows from normal operation (chance level)

Notebook usage
    from sbl_iforest import analyse_all
    results = analyse_all({"beam": ds_beam, "background": ds_background,
                           "injection": ds_injection, "vacuum": ds_vacuum,
                           "position": ds_position},
                          sbl_times,                        # Series / list / 1-column DataFrame
                          PRE_SBL_SECONDS=120)              # any CONFIG entry can be overridden
    results["background"]["events"]                        # one row per SBL
    results["background"]["summary"]                       # detection vs. chance

Terminal usage
    python sbl_iforest.py --block beam=data/beam.csv --block vacuum=data/vacuum.csv \
                          --sbl data/sbl_times.csv --pre 120
"""
import argparse
import os
import re

import matplotlib
import numpy as np
import pandas as pd

# ============================================================================ configuration
CONFIG = dict(
    TS_COL="timestamp",
    PRE_SBL_SECONDS=120,          # analysis window before each SBL
    POST_SBL_SECONDS=30,          # shown in the plot only, never used for ranking
    NORMAL_MARGIN_SECONDS=900,    # rows farther than this from every SBL train the models
    FFILL_LIMIT=3,                # forward-fill at most this many rows per column
    N_ESTIMATORS=200,
    MAX_TRAIN_ROWS=50000,
    RANDOM_STATE=0,
    TOP_K_FEATURES=10,            # features in each bar chart
    TOP_K_LINES=3,                # feature scores drawn in the time-series plot
    HEATMAP_FEATURES=25,          # columns in the overall heatmap
    VMAX=None,                    # colour cap of the heatmap; None = automatic (99th pct. of the table)
    GAP_BREAK_SECONDS=5,          # plotted lines are interrupted at time gaps longer than this
    N_RANDOM_WINDOWS=2000,        # random normal windows for the chance level
    OUTPUT_DIR="sbl_iforest_output",
    SHOW_PLOTS=False,             # True: also display the figures (in a notebook)
    MAX_EVENT_PLOTS=None,         # None = one figure per SBL; or e.g. 20
    # columns that carry information about the SBL itself must never be model input
    LEAK_PATTERN=r"sbl|time_before|time_after|dist|is_risk|is_normal|phase|label|target",
)

COLOR_OVERALL = "#1f3b73"
COLOR_FEATURES = ["#2a9d8f", "#e9a23b", "#7b5ea7", "#8d99ae", "#5f8f3e"]
COLOR_SBL = "red"


# ============================================================================ 1. loading
def to_naive(x):
    """Timestamps -> tz-naive datetime64[ns] (tz-aware values are converted to UTC)."""
    x = pd.to_datetime(pd.Series(x).reset_index(drop=True))
    if x.dt.tz is not None:
        x = x.dt.tz_convert("UTC").dt.tz_localize(None)
    return x.astype("datetime64[ns]")


def as_sbl_index(sbl_times, ts_col="timestamp"):
    """Series / list / DatetimeIndex / DataFrame (1 column or a ts_col column) -> DatetimeIndex."""
    if isinstance(sbl_times, pd.DataFrame):
        if ts_col in sbl_times.columns:
            sbl_times = sbl_times[ts_col]
        elif sbl_times.shape[1] == 1:
            sbl_times = sbl_times.iloc[:, 0]
        else:
            raise ValueError(f"sbl_times has columns {list(sbl_times.columns)}; pass the timestamp column.")
    arr = np.asarray(sbl_times)
    if arr.ndim == 2 and arr.shape[1] == 1:
        sbl_times = arr[:, 0]
    return pd.DatetimeIndex(to_naive(sbl_times).dropna().sort_values().unique())


def check_same_time_format(df, sbl_times, ts_col):
    def aware(x):
        return pd.to_datetime(pd.Series(np.asarray(x).ravel())).dt.tz is not None
    sbl_vals = sbl_times[ts_col] if isinstance(sbl_times, pd.DataFrame) and ts_col in sbl_times else sbl_times
    if aware(df[ts_col]) != aware(sbl_vals):
        raise ValueError("Block and SBL list use different time formats (one with time zone, one without).")


def clean_block(df, cfg, name="block"):
    """Numeric columns, sorted unique DatetimeIndex, no SBL-derived or constant columns."""
    ts = cfg["TS_COL"]
    if ts not in df.columns:
        raise ValueError(f"[{name}] column '{ts}' not found; columns: {list(df.columns)}")
    pat = re.compile(cfg["LEAK_PATTERN"], re.IGNORECASE)
    leak = [c for c in df.columns if c != ts and pat.search(str(c))]
    d = df.drop(columns=leak + [ts]).select_dtypes("number")
    d.index = pd.DatetimeIndex(to_naive(df[ts]).to_numpy())
    d = d[d.index.notna()].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.ffill(limit=cfg["FFILL_LIMIT"])
    d = d.loc[:, d.notna().sum() > 0]
    d = d.loc[:, d.std() > 0]
    if leak:
        print(f"  [{name}] removed SBL-related columns: {leak}")
    if d.shape[1] == 0:
        raise ValueError(f"[{name}] no usable numeric feature columns.")
    return d


def attach_sbl(d, sbl):
    """Adds the SBL information to the block's own timeline (not as model input):
       dist_to_sbl_s   distance to the nearest SBL in seconds
       sbl_event_id    id of the SBL that this row is closest to
    and returns a table of the events with their onset index in this block."""
    t = d.index.values.astype("int64")
    s = sbl.values.astype("int64")
    i = np.searchsorted(s, t)
    prev = np.clip(i - 1, 0, len(s) - 1)
    nxt = np.clip(i, 0, len(s) - 1)
    use_prev = np.abs(t - s[prev]) <= np.abs(s[nxt] - t)
    nearest = np.where(use_prev, prev, nxt)
    info = pd.DataFrame({"dist_to_sbl_s": np.abs(t - s[nearest]) / 1e9, "sbl_event_id": nearest}, index=d.index)

    onset = np.searchsorted(t, s)                       # first row at or after the SBL
    events = pd.DataFrame({"sbl_time": sbl, "onset_row": onset,
                           "onset_row_time": [d.index[k] if k < len(d) else pd.NaT for k in onset]})
    events.index.name = "event_id"
    return info, events


# ============================================================================ 3. models
def _normalise(score, ref_mask):
    ref = score[ref_mask & np.isfinite(score)]
    med, hi = np.median(ref), np.quantile(ref, 0.99)
    den = hi - med if hi - med > 0 else (ref.std() if ref.std() > 0 else 1.0)
    return (score - med) / den


def fit_scores(d, normal, cfg, name="block"):
    """Multivariate + per-feature Isolation Forest, fitted on normal rows only.
    Returns a DataFrame: overall_anomaly_score + <feature>_anomaly_score (normalised)."""
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(cfg["RANDOM_STATE"])
    out = pd.DataFrame(index=d.index)

    def train_rows(mask):
        idx = np.flatnonzero(mask)
        if len(idx) > cfg["MAX_TRAIN_ROWS"]:
            idx = rng.choice(idx, cfg["MAX_TRAIN_ROWS"], replace=False)
        return idx

    # multivariate: rows where all features are known
    complete = d.notna().all(axis=1).to_numpy()
    tr = train_rows(normal & complete)
    if len(tr) < 50:
        raise ValueError(f"[{name}] only {len(tr)} complete normal rows - too few to fit the model. "
                         "Lower NORMAL_MARGIN_SECONDS or check the data.")
    X = d.to_numpy(float)
    model = IsolationForest(n_estimators=cfg["N_ESTIMATORS"], random_state=cfg["RANDOM_STATE"])
    model.fit(X[tr])
    s = np.full(len(d), np.nan)
    s[complete] = -model.score_samples(X[complete])
    out["overall_anomaly_score"] = _normalise(s, normal & complete)

    # univariate: one model per feature, on that feature's own non-NaN rows
    for c in d.columns:
        x = d[c].to_numpy(float)
        ok = np.isfinite(x)
        tr = train_rows(normal & ok)
        if len(tr) < 50:
            print(f"  [{name}] feature '{c}' skipped: only {len(tr)} normal values")
            continue
        m = IsolationForest(n_estimators=cfg["N_ESTIMATORS"], random_state=cfg["RANDOM_STATE"])
        m.fit(x[tr, None])
        s = np.full(len(d), np.nan)
        s[ok] = -m.score_samples(x[ok, None])
        out[f"{c}_anomaly_score"] = _normalise(s, normal & ok)
    return out


# ============================================================================ 2. windows
def extract_windows(scores, sbl, cfg):
    """dict event_id -> DataFrame of the rows in [SBL - PRE, SBL + POST] with a column
    't_rel_s' (seconds relative to the SBL, negative = before) and 'pre_sbl' (True before)."""
    pre, post = pd.Timedelta(seconds=cfg["PRE_SBL_SECONDS"]), pd.Timedelta(seconds=cfg["POST_SBL_SECONDS"])
    wins = {}
    for k, t in enumerate(sbl):
        w = scores.loc[t - pre: t + post].copy()
        w["t_rel_s"] = (w.index - t).total_seconds()
        w["pre_sbl"] = w["t_rel_s"] < 0                  # the SBL row itself counts as "after"
        wins[k] = w
    return wins


def summarise_events(wins, sbl, feat_cols):
    rows, peaks = [], {}
    for k, w in wins.items():
        p = w[w["pre_sbl"]]
        r = dict(event_id=k, sbl_time=sbl[k], n_rows_pre=len(p))
        if len(p) and p["overall_anomaly_score"].notna().any():
            o = p["overall_anomaly_score"]
            r.update(peak_overall=o.max(), mean_overall=o.mean(),
                     peak_at_s=p.loc[o.idxmax(), "t_rel_s"])
        fp = p[feat_cols].max() if len(p) else pd.Series(np.nan, index=feat_cols)
        peaks[k] = fp
        top = fp.dropna().sort_values(ascending=False).head(3)
        r["top_features"] = ", ".join(f"{c.replace('_anomaly_score', '')} ({v:.1f})" for c, v in top.items())
        rows.append(r)
    events = pd.DataFrame(rows).set_index("event_id")
    peak_table = pd.DataFrame(peaks).T
    peak_table.columns = [c.replace("_anomaly_score", "") for c in peak_table.columns]
    return events, peak_table


# ============================================================================ chance level
def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def chance_summary(scores, normal, events, cfg, threshold=1.0):
    """Fraction of SBL windows whose overall score exceeds `threshold` (1 = q99 of normal),
    compared with random windows of the same length taken from normal operation."""
    s = scores["overall_anomaly_score"]
    ev = events.dropna(subset=["peak_overall"])
    k, n = int((ev["peak_overall"] > threshold).sum()), len(ev)
    lo, hi = _wilson(k, n)

    # Chance level with the SAME number of rows as each SBL window: a window with more rows has
    # more opportunities to exceed the threshold, and in sparse archives the number of rows
    # before an SBL can differ from normal times.  For each event with m rows, draw random runs
    # of m consecutive normal rows (keeps the correlation between neighbouring rows).
    rng = np.random.default_rng(cfg["RANDOM_STATE"])
    sn = s[normal & s.notna().to_numpy()].to_numpy()
    n_draw = max(1, cfg["N_RANDOM_WINDOWS"] // 10)
    per_event = []
    for m in ev["n_rows_pre"].astype(int):
        m = max(1, min(m, len(sn)))
        starts = rng.integers(0, len(sn) - m + 1, n_draw)
        per_event.append(np.mean([sn[j:j + m].max() > threshold for j in starts]))
    chance = float(np.mean(per_event)) if per_event else np.nan
    return pd.DataFrame([dict(
        threshold=threshold, events_with_data=n, events_without_data=len(events) - n,
        events_above=k, rate=round(k / n, 3) if n else np.nan,
        rate_ci95=f"{lo:.2f}-{hi:.2f}" if n else "n/a",
        chance_rate=round(chance, 3), median_rows_per_window=int(ev["n_rows_pre"].median()) if n else 0,
        above_chance=bool(n and lo > chance))])


# ============================================================================ 4 + 5. plots
def _setup_mpl(show):
    # the backend is NOT switched here, so plots keep working in a notebook afterwards;
    # figures that are not shown are closed right after saving
    import matplotlib.pyplot as plt
    return plt


def _break_gaps(w, gap_s):
    """Insert NaN rows at time gaps so that lines are not drawn across missing data."""
    if len(w) < 2:
        return w
    gaps = np.flatnonzero(np.diff(w["t_rel_s"].to_numpy()) > gap_s)
    if not len(gaps):
        return w
    filler = pd.DataFrame(np.nan, index=range(len(gaps)), columns=w.columns)
    filler["t_rel_s"] = w["t_rel_s"].to_numpy()[gaps] + gap_s / 2
    return pd.concat([w.reset_index(drop=True), filler]).sort_values("t_rel_s")


def plot_event(w, k, t, cfg, name, plt):
    """Left: overall + top feature scores vs. time, red dashed line at the SBL.
    Right: ranking of the most anomalous features in the pre-SBL window."""
    feat_cols = [c for c in w.columns if c.endswith("_anomaly_score") and c != "overall_anomaly_score"]
    p = w[w["pre_sbl"]]
    rank = p[feat_cols].max().dropna().sort_values(ascending=False) if len(p) else pd.Series(dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    w = _break_gaps(w, cfg["GAP_BREAK_SECONDS"])
    if w.empty:
        ax1.text(0.5, 0.5, "no data in this window", ha="center", va="center", transform=ax1.transAxes)
    else:
        ax1.plot(w["t_rel_s"], w["overall_anomaly_score"], color=COLOR_OVERALL, lw=2, marker=".", ms=5,
                 label="overall (multivariate)")
        for c, col in zip(rank.index[:cfg["TOP_K_LINES"]], COLOR_FEATURES):
            ax1.plot(w["t_rel_s"], w[c], color=col, lw=1.2, alpha=0.9, marker=".", ms=3,
                     label=c.replace("_anomaly_score", ""))
    ax1.axhline(1.0, color="gray", ls=":", lw=1, label="99th pct. of normal")
    ax1.axvline(0, color=COLOR_SBL, ls="--", lw=2, label="SBL Event")
    ax1.set_xlim(-cfg["PRE_SBL_SECONDS"], cfg["POST_SBL_SECONDS"])
    ax1.set_xlabel("seconds relative to SBL (0 = SBL)")
    ax1.set_ylabel("anomaly score (0 = normal median, 1 = q99)")
    ax1.set_title(f"{name} | SBL #{k}  {t:%Y-%m-%d %H:%M:%S}  ({len(p)} rows before SBL)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    top = rank.head(cfg["TOP_K_FEATURES"])[::-1]
    if len(top):
        ax2.barh([c.replace("_anomaly_score", "") for c in top.index], top.values, color=COLOR_OVERALL, height=0.6)
        ax2.axvline(1.0, color="gray", ls=":", lw=1)
        for y, v in enumerate(top.values):
            ax2.text(v, y, f" {v:.1f}", va="center", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "no data before SBL", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel(f"peak feature score in the {cfg['PRE_SBL_SECONDS']} s before SBL")
    ax2.set_title("most anomalous features")
    ax2.tick_params(axis="y", labelsize=8)
    ax2.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_heatmap(peak_table, events, cfg, name, plt):
    """Rows = SBL events (with data), columns = features, colour = peak score before the SBL."""
    tab = peak_table.dropna(how="all")
    if tab.empty:
        return None
    cols = tab.mean().sort_values(ascending=False).index[:cfg["HEATMAP_FEATURES"]]
    tab = tab[cols]
    vmax = cfg["VMAX"] or max(1.5, float(np.nanquantile(tab.to_numpy(), 0.99)))
    fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 3.5, 0.2 * len(tab) + 2.2))
    im = ax.imshow(tab.clip(lower=0).to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"#{k} {events.loc[k, 'sbl_time']:%Y-%m-%d %H:%M}" for k in tab.index], fontsize=6)
    ax.set_xlabel("feature (sorted by mean peak score)")
    ax.set_ylabel("SBL event")
    ax.set_title(f"{name}: peak per-feature anomaly score in the {cfg['PRE_SBL_SECONDS']} s before each SBL")
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"score (1 = q99 of normal, colour capped at {vmax:.1f})")
    fig.tight_layout()
    return fig


# ============================================================================ main
def analyse_block(df, sbl_times, name="block", **overrides):
    cfg = {**CONFIG, **overrides}
    unknown = sorted(set(overrides) - set(CONFIG))
    if unknown:
        raise ValueError(f"Unknown setting(s) {unknown}; valid: {sorted(CONFIG)}")
    if cfg["NORMAL_MARGIN_SECONDS"] <= cfg["PRE_SBL_SECONDS"]:
        raise ValueError("NORMAL_MARGIN_SECONDS must be larger than PRE_SBL_SECONDS, otherwise the "
                         "pre-SBL windows are learned as normal.")
    plt = _setup_mpl(cfg["SHOW_PLOTS"])

    print(f"\n======== {name} ========")
    check_same_time_format(df, sbl_times, cfg["TS_COL"])
    sbl = as_sbl_index(sbl_times, cfg["TS_COL"])
    d = clean_block(df, cfg, name)                                   # step 1
    info, onsets = attach_sbl(d, sbl)
    normal = (info["dist_to_sbl_s"] > cfg["NORMAL_MARGIN_SECONDS"]).to_numpy()
    print(f"  rows={len(d)}  features={d.shape[1]}  normal rows={int(normal.sum())}  SBL events={len(sbl)}")

    scores = fit_scores(d, normal, cfg, name)                         # step 3
    scores = scores.join(info)
    wins = extract_windows(scores, sbl, cfg)                          # step 2
    feat_cols = [c for c in scores.columns if c.endswith("_anomaly_score") and c != "overall_anomaly_score"]
    events, peak_table = summarise_events(wins, sbl, feat_cols)
    events = events.join(onsets[["onset_row", "onset_row_time"]])
    summary = chance_summary(scores, normal, events, cfg)
    print(f"  SBL windows with data: {summary.loc[0, 'events_with_data']} of {len(sbl)}")
    print(summary.to_string(index=False))

    out_dir = os.path.join(cfg["OUTPUT_DIR"], name)                   # steps 4 + 5
    os.makedirs(out_dir, exist_ok=True)
    n_plot = len(wins) if cfg["MAX_EVENT_PLOTS"] is None else min(cfg["MAX_EVENT_PLOTS"], len(wins))
    for k in list(wins)[:n_plot]:
        fig = plot_event(wins[k], k, sbl[k], cfg, name, plt)
        fig.savefig(os.path.join(out_dir, f"event_{k:03d}.png"), dpi=110)
        plt.show() if cfg["SHOW_PLOTS"] else plt.close(fig)
    fig = plot_heatmap(peak_table, events, cfg, name, plt)
    if fig is not None:
        fig.savefig(os.path.join(out_dir, "heatmap_top_features.png"), dpi=130)
        plt.show() if cfg["SHOW_PLOTS"] else plt.close(fig)
    events.to_csv(os.path.join(out_dir, "events.csv"))
    peak_table.to_csv(os.path.join(out_dir, "peak_feature_scores.csv"))
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"  saved {n_plot} event plots + heatmap + tables to {out_dir}/")
    return dict(scores=scores, windows=wins, events=events, peak_table=peak_table, summary=summary, cfg=cfg)


def analyse_all(blocks, sbl_times, **overrides):
    """Runs analyse_block for every block separately (blocks are never merged)."""
    results = {}
    for name, df in blocks.items():
        try:
            results[name] = analyse_block(df, sbl_times, name, **overrides)
        except ValueError as e:
            print(f"  [{name}] skipped: {e}")
    return results


def _cli():
    ap = argparse.ArgumentParser(description="Isolation Forest anomaly scores before SBL events.")
    ap.add_argument("--block", action="append", required=True, help="name=path.csv (repeatable)")
    ap.add_argument("--sbl", required=True, help="CSV with the SBL timestamps")
    ap.add_argument("--pre", type=float, default=CONFIG["PRE_SBL_SECONDS"], help="seconds before each SBL")
    ap.add_argument("--ts-col", default=CONFIG["TS_COL"])
    a = ap.parse_args()
    blocks = {b.split("=", 1)[0]: pd.read_csv(b.split("=", 1)[1]) for b in a.block}
    sbl = pd.read_csv(a.sbl)
    analyse_all(blocks, sbl, PRE_SBL_SECONDS=a.pre, TS_COL=a.ts_col)


if __name__ == "__main__":
    _cli()


# ============================================================================ aggregate over all events
def aggregate_table(results, stat="median", n_bins=12, per_event="mean"):
    """Event-locked summary over ALL SBL events and ALL blocks.
    Every window is cut into n_bins time bins from -PRE to +POST seconds. Inside a bin each event
    contributes ONE value per feature (its mean - default - or max in that bin; the mean keeps
    normal operation at ~0, the max is biased upwards), then `stat` (median / mean)
    is taken over the events.  Returns (table, n_events): rows = 'block: feature', columns = bin
    centres in seconds relative to the SBL."""
    any_res = next(iter(results.values()))
    pre, post = any_res["cfg"]["PRE_SBL_SECONDS"], any_res["cfg"]["POST_SBL_SECONDS"]
    edges = np.linspace(-pre, post, n_bins + 1)
    # make sure 0 is a bin edge so that no bin mixes "before" and "after" the SBL
    edges = np.unique(np.r_[edges[edges < 0], 0.0, edges[edges > 0]])
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows, counts = {}, {}
    for block, res in results.items():
        cols = ["overall_anomaly_score"] + [c for c in res["scores"].columns
                                            if c.endswith("_anomaly_score") and c != "overall_anomaly_score"]
        per = {c: [[] for _ in centres] for c in cols}
        for w in res["windows"].values():
            if w.empty:
                continue
            b = np.digitize(w["t_rel_s"].to_numpy(), edges) - 1
            for j in np.unique(b[(b >= 0) & (b < len(centres))]):
                sel = w[b == j]
                for c in cols:
                    v = sel[c].dropna()
                    if len(v):
                        per[c][j].append(v.max() if per_event == "max" else v.mean())
        for c in cols:
            label = f"{block}: " + ("OVERALL" if c == "overall_anomaly_score" else c.replace("_anomaly_score", ""))
            f = np.median if stat == "median" else np.mean
            rows[label] = [f(v) if len(v) else np.nan for v in per[c]]
            counts[label] = [len(v) for v in per[c]]
    return (pd.DataFrame(rows, index=np.round(centres, 1)).T,
            pd.DataFrame(counts, index=np.round(centres, 1)).T)


def plot_aggregate(results, stat="median", n_bins=12, min_events=3, top_per_block=None, vmax=None,
                   per_event="mean", save_to=None, show=True):
    """One heatmap for all blocks together: rows = features (grouped by block, OVERALL first),
    columns = time before/after the SBL, colour = median (or mean) score over the events.
    Cells with fewer than `min_events` events are grey.  The numbers on top = events per bin
    (taken from the OVERALL rows).  Red dashed line = SBL."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    tab, n = aggregate_table(results, stat, n_bins, per_event)
    if top_per_block:                                   # keep OVERALL + the top features of each block
        keep = []
        for block in results:
            r = [i for i in tab.index if i.startswith(f"{block}:")]
            over = [i for i in r if i.endswith("OVERALL")]
            rest = tab.loc[[i for i in r if i not in over]]
            pre_cols = [c for c in tab.columns if c < 0]
            best = rest[pre_cols].max(axis=1).sort_values(ascending=False).index[:top_per_block]
            keep += over + list(best)
        tab, n = tab.loc[keep], n.loc[keep]
    shown = tab.where(n >= min_events)
    vals = shown.to_numpy()
    vmax = vmax or max(1.5, float(np.nanquantile(np.abs(vals), 0.98))) if np.isfinite(vals).any() else 1.5

    fig, ax = plt.subplots(figsize=(0.75 * tab.shape[1] + 5, 0.24 * len(tab) + 2.5))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap,
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), interpolation="nearest")
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels(tab.index, fontsize=7)
    for lab in ax.get_yticklabels():
        if lab.get_text().endswith("OVERALL"):
            lab.set_fontweight("bold")
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels([f"{c:.0f}" for c in tab.columns], fontsize=8)
    ax.set_xlabel("seconds relative to SBL (bin centre)")
    # SBL line between the last negative and the first positive bin
    k0 = int(np.sum(np.asarray(tab.columns) < 0))
    ax.axvline(k0 - 0.5, color="red", ls="--", lw=2, label="SBL Event")
    # separators between blocks
    blocks = [i.split(":")[0] for i in tab.index]
    for y in range(1, len(blocks)):
        if blocks[y] != blocks[y - 1]:
            ax.axhline(y - 0.5, color="black", lw=0.8)
    # events per bin, per block, written on the OVERALL row
    for y, lab in enumerate(tab.index):
        if lab.endswith("OVERALL"):
            for x, c in enumerate(tab.columns):
                ax.text(x, y, f"n={n.loc[lab, c]}", ha="center", va="center", fontsize=6, color="black")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.06), fontsize=8, frameon=False)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02,
                 label=f"{stat} score over events\n(0 = normal median, 1 = q99 of normal)")
    ax.set_title(f"{stat} anomaly score over all SBL events (grey = fewer than {min_events} events)")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, tab, n


# ============================================================================ feature vote count
def feature_counts(results, mode="threshold", threshold=1.0, top_k_per_event=3, n_draws=200):
    """For every SBL event, every feature gets +1 if it 'shows' before the SBL:
         mode="threshold": its peak score in the pre-SBL window is above `threshold` (1 = q99 normal)
         mode="topk":      it is among the `top_k_per_event` highest features of its block
    The counts are summed over all events.  The same is done for random windows from normal
    operation with the SAME number of rows as each event (chance level), so you can see whether a
    feature is chosen more often than it would be by luck.
    Returns a DataFrame sorted by count."""
    from math import isnan
    rows = []
    for block, res in results.items():
        cfg = res["cfg"]
        sc = res["scores"]
        fcols = [c for c in sc.columns if c.endswith("_anomaly_score") and c != "overall_anomaly_score"]
        normal = (sc["dist_to_sbl_s"] > cfg["NORMAL_MARGIN_SECONDS"]).to_numpy()
        ref = sc.loc[normal, fcols].to_numpy()
        rng = np.random.default_rng(cfg["RANDOM_STATE"])

        def votes(peaks):
            """peaks: (..., n_features) -> 0/1 votes"""
            if mode == "threshold":
                return (np.nan_to_num(peaks, nan=-np.inf) > threshold).astype(float)
            p = np.nan_to_num(peaks, nan=-np.inf)
            order = np.argsort(-p, axis=-1)[..., :top_k_per_event]
            v = np.zeros_like(p)
            np.put_along_axis(v, order, 1.0, axis=-1)
            return v * np.isfinite(peaks)

        obs = np.zeros(len(fcols))
        exp = np.zeros(len(fcols))
        n_ev = np.zeros(len(fcols))
        for w in res["windows"].values():
            p = w[w["pre_sbl"]]
            if p.empty:
                continue
            peaks = p[fcols].max().to_numpy()
            has = np.isfinite(peaks)
            obs += votes(peaks[None, :])[0]
            n_ev += has
            m = max(1, min(len(p), len(ref)))
            starts = rng.integers(0, len(ref) - m + 1, n_draws)
            rand_peaks = np.stack([np.nanmax(ref[j:j + m], axis=0) if np.isfinite(ref[j:j + m]).any() else
                                   np.full(len(fcols), np.nan) for j in starts])
            rand_peaks[:, ~has] = np.nan                      # compare only features that had data
            exp += np.nanmean(votes(rand_peaks), axis=0)
        for c, o, e, n in zip(fcols, obs, exp, n_ev):
            lo, hi = _wilson(int(o), int(n))
            rows.append(dict(feature=f"{block}: {c.replace('_anomaly_score', '')}", block=block,
                             count=int(o), events_with_data=int(n), expected_by_chance=round(e, 1),
                             excess_over_chance=round(o - e, 1),
                             rate=round(o / n, 3) if n else np.nan,
                             rate_ci95=f"{lo:.2f}-{hi:.2f}" if n else "n/a",
                             chance_rate=round(e / n, 3) if n else np.nan,
                             above_chance=bool(n and not isnan(lo) and lo > e / n)))
    # sorted by how much MORE often than chance a feature was chosen (a plain count would favour
    # blocks with few features in mode="topk", where e.g. 3 features are always the "top 3")
    return pd.DataFrame(rows).sort_values(["excess_over_chance", "count"], ascending=False).reset_index(drop=True)


def plot_feature_counts(results, top_n=10, mode="threshold", threshold=1.0, top_k_per_event=3,
                        save_to=None, show=True):
    """Bar chart of the top_n features chosen most often over all SBL events - ranked by how much
    more often than chance (blue bars = count),
    with the number expected by chance (black tick).  Bars of features whose Wilson interval lies
    above chance are dark, the others light."""
    import matplotlib.pyplot as plt
    tab = feature_counts(results, mode, threshold, top_k_per_event)
    top = tab.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top) + 1.8))
    colors = [COLOR_OVERALL if a else "#9fb3d1" for a in top["above_chance"]]
    y = np.arange(len(top))
    ax.barh(y, top["count"], color=colors, height=0.6)
    ax.scatter(top["expected_by_chance"], y, marker="|", s=300, color="black", zorder=3,
               label="expected by chance")
    for yi, (c, n) in enumerate(zip(top["count"], top["events_with_data"])):
        ax.text(c, yi, f"  {c} / {n} events", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8)
    rule = (f"peak score > {threshold:g} (q99 of normal)" if mode == "threshold"
            else f"among the top {top_k_per_event} of its block")
    ax.set_xlabel(f"number of SBL events where the feature was {rule}")
    ax.set_title(f"Top {len(top)} features over all SBL events\n(dark = clearly above chance, light = not)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, max(top["count"].max(), top["expected_by_chance"].max()) * 1.25 + 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, tab
