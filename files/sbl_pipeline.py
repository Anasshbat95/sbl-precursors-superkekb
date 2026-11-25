"""
sbl_pipeline.py - Complete SBL anomaly-detection pipeline in ONE file (version 2).

    from sbl_pipeline import run, plot_event_locked, feature_heatmaps
    out = run({"beam": ds_beam, "background": ds_background, "injection": ds_injection,
               "vacuum": ds_vacuum, "position": ds_position}, sbl_times)

Steps
    0  remove columns that carry SBL information (distance, label, phase ...)  -> no leakage
       check that blocks and SBL list use the same time format
    1  per block: BEAM_ON filter, cleaning, log10 for wide-range columns (blocks never merged)
    2  backward-looking rolling features per block
    3  models, trained on train-normal rows only (chronological split train | calib | test):
         zscore   max |robust z| over features (NOT clipped -> cannot saturate)
         iforest  Isolation Forest
         lof      Local Outlier Factor (novelty=True)
         mcd      robust Mahalanobis distance (EllipticEnvelope) - assumes ONE mode
         ocsvm, sgd_ocsvm   One-Class SVM (exact / kernel-approximated)
    4  scores (not data) -> common time grid -> normalised (0 = median, 1 = q99 of calibration
       normal) -> fused = max over blocks
    5  validation per EVENT on the test period: detection rate + Wilson 95% interval,
       chance level from the false-alarm rate, lead time, false alarms per hour

Differences to the old pipeline (pipeline_*.py)
    * step 0 is new (leak guard + time check)
    * zscore is no longer clipped at 50 (the old version could never raise an alarm when the
      calibration q99 sat at the clip value)
    * fusion warns when a block's calibration scores are saturated
    * four extra models; operation filter is ON by default
"""
import re

import numpy as np
import pandas as pd

DEFAULTS = dict(
    ts_col="timestamp",
    # step 0
    leak_pattern=r"sbl|time_before|time_after|dist|is_risk|is_normal|phase|label|target|risk",
    # step 1
    use_operation_filter=True,
    beam_block="beam",
    current_cols=("her_current", "ler_current"),
    on_thr=50.0,
    op_rampup="15min",             # first minutes after beam-on are dropped (RAMP_UP); "0min" = keep
    op_max_hold="2h",              # last current is held this long when there are no rows
    op_min_on="5min",              # shorter beam-on blips do not count as BEAM_ON
    ffill_limit=3,
    auto_log=True,
    # step 2
    window="10min",
    min_rows_per_window=5,
    min_periods=3,
    # steps 3 + 4
    methods=("zscore", "iforest", "lof"),
    normal_margin="15min",
    train_frac=0.6,
    calib_frac=0.15,
    n_estimators=200,
    max_train_rows=200000,          # iforest / zscore
    max_train_rows_slow=10000,      # lof / mcd / ocsvm get slow with many rows
    min_train_rows=100,
    random_state=0,
    grid="auto",
    # step 5
    lookback="10min",
    buffer="30s",
    cooldown="10min",
    max_gap="30min",
    thr_quantiles=(0.99, 0.995, 0.999, 0.9999),
    n_boot=500,
)


def _section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# =============================================================================== helpers
def to_naive(x):
    """Timestamps -> tz-naive. tz-aware values are converted to UTC first; naive values stay."""
    x = pd.to_datetime(pd.Series(x).reset_index(drop=True))
    if x.dt.tz is not None:
        x = x.dt.tz_convert("UTC").dt.tz_localize(None)
    return x.astype("datetime64[ns]")


def dist_to_sbl(ts, sbl):
    """Distance (seconds) from each timestamp to the nearest SBL."""
    t = np.asarray(ts, dtype="datetime64[ns]").astype("int64")
    s = np.sort(np.asarray(sbl, dtype="datetime64[ns]").astype("int64"))
    i = np.searchsorted(s, t)
    prev = np.abs(t - s[np.clip(i - 1, 0, len(s) - 1)])
    nxt = np.abs(s[np.clip(i, 0, len(s) - 1)] - t)
    return np.minimum(prev, nxt) / 1e9


def _normal(index, sbl, cfg):
    return dist_to_sbl(index, sbl) > pd.Timedelta(cfg["normal_margin"]).total_seconds()


# =============================================================================== step 0
def drop_sbl_columns(blocks, cfg):
    pat = re.compile(cfg["leak_pattern"], re.IGNORECASE)
    out = {}
    for name, df in blocks.items():
        bad = [c for c in df.columns if c != cfg["ts_col"] and pat.search(str(c))]
        print(f"  {name:11s} removed columns: {bad if bad else '-'}")
        out[name] = df.drop(columns=bad)
    return out


def check_time_format(blocks, sbl_times, cfg):
    def aware(x):
        return pd.to_datetime(pd.Series(x)).dt.tz is not None
    sbl_aware = aware(sbl_times)
    for name, df in blocks.items():
        if aware(df[cfg["ts_col"]]) != sbl_aware:
            raise ValueError(f"Block '{name}' and the SBL list use different time formats (one with time "
                             "zone, one without). Give both the same format before running.")
    print(f"  time format: {'with' if sbl_aware else 'without'} time zone in all blocks and the SBL list - ok")


# =============================================================================== step 1
def operation_intervals(beam, ts_col, cols, on_thr, grid="1min", max_hold="2h", min_on="5min",
                        shutdown_min="12h", rampup="15min"):
    """Interval table (start, end, state) with BEAM_ON / RAMP_UP / BEAM_OFF / SHUTDOWN."""
    def merge(iv):
        iv = iv.sort_values("start").reset_index(drop=True)
        grp = (iv["on"] != iv["on"].shift()).cumsum()
        return iv.groupby(grp).agg(start=("start", "min"), end=("end", "max"),
                                   on=("on", "first")).reset_index(drop=True)

    d = beam[[ts_col, *cols]].dropna()
    cur = pd.Series(d[list(cols)].max(axis=1).to_numpy(), index=pd.DatetimeIndex(d[ts_col])).sort_index()
    cur = cur.groupby(level=0).max()
    g = cur.resample(grid).max().ffill(limit=max(1, int(pd.Timedelta(max_hold) / pd.Timedelta(grid))))
    on = (g > on_thr).to_numpy()
    change = np.r_[True, on[1:] != on[:-1]]
    iv = (pd.DataFrame({"start": g.index, "on": on, "run": np.cumsum(change)})
          .groupby("run").agg(start=("start", "min"), on=("on", "first")))
    iv["end"] = iv["start"].shift(-1)
    iv.loc[iv.index[-1], "end"] = g.index[-1] + pd.Timedelta(grid)
    iv = merge(iv[["start", "end", "on"]].reset_index(drop=True))
    short_on = iv["on"] & ((iv["end"] - iv["start"]) < pd.Timedelta(min_on))
    iv.loc[short_on, "on"] = False
    iv = merge(iv)
    dur = iv["end"] - iv["start"]
    iv["state"] = np.where(iv["on"], "BEAM_ON",
                           np.where(dur >= pd.Timedelta(shutdown_min), "SHUTDOWN", "BEAM_OFF"))
    out = []
    for i, r in iv.iterrows():
        after_off = i > 0 and not iv["on"].iloc[i - 1]
        if pd.Timedelta(rampup) <= pd.Timedelta(0):
            out.append((r["start"], r["end"], r["state"]))
        elif r["state"] == "BEAM_ON" and after_off and (r["end"] - r["start"]) > pd.Timedelta(rampup):
            mid = r["start"] + pd.Timedelta(rampup)
            out += [(r["start"], mid, "RAMP_UP"), (mid, r["end"], "BEAM_ON")]
        elif r["state"] == "BEAM_ON" and after_off:
            out.append((r["start"], r["end"], "RAMP_UP"))
        else:
            out.append((r["start"], r["end"], r["state"]))
    return pd.DataFrame(out, columns=["start", "end", "state"])


def _state_of(times, iv):
    idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
    pos = idx.get_indexer(pd.DatetimeIndex(times))
    return np.where(pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA")


def _clean_block(df, cfg):
    d = df.drop(columns=cfg["ts_col"]).select_dtypes("number").reset_index(drop=True)
    d.index = pd.DatetimeIndex(df[cfg["ts_col"]].to_numpy())
    d = d[d.index.notna()].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.ffill(limit=cfg["ffill_limit"]).dropna()
    return d.loc[:, d.std() > 0]


def _log_columns(d):
    done = []
    for c in d.columns:
        v = d[c]
        if v.min() < 0:
            continue
        pos = v[v > 0]
        if len(pos) < 50:
            continue
        lo, hi = pos.quantile(0.01), pos.quantile(0.99)
        if lo > 0 and hi / lo > 1e3:
            d[c] = np.log10(v + lo * 0.1)
            done.append(c)
    return d, done


def preprocess(blocks, sbl_times, cfg):
    ts = cfg["ts_col"]
    sbl = pd.DatetimeIndex(to_naive(sbl_times).dropna().sort_values())
    conv = {}
    for n, df in blocks.items():
        df = df.copy()
        df[ts] = to_naive(df[ts]).to_numpy()
        conv[n] = df.dropna(subset=[ts])

    iv = None
    if cfg["use_operation_filter"]:
        beam = conv.get(cfg["beam_block"])
        missing = [] if beam is None else [c for c in cfg["current_cols"] if c not in beam.columns]
        if beam is None or missing:
            cols = [] if beam is None else [c for c in beam.columns if c != ts]
            raise ValueError(f"BEAM_ON filter: block '{cfg['beam_block']}' or columns {missing} not found. "
                             f"Columns in that block: {cols}. Set current_cols=(...) to the right names, "
                             "or use_operation_filter=False (not recommended).")
        iv = operation_intervals(beam, ts, list(cfg["current_cols"]), cfg["on_thr"],
                                 max_hold=cfg["op_max_hold"], min_on=cfg["op_min_on"], rampup=cfg["op_rampup"])

    data, rows = {}, []
    for n, df in conv.items():
        n_in = len(df)
        if iv is not None:
            df = df[_state_of(df[ts], iv) == "BEAM_ON"]
        d = _clean_block(df, cfg)
        logged = []
        if cfg["auto_log"]:
            d, logged = _log_columns(d)
        data[n] = d
        spacing = d.index.to_series().diff().dt.total_seconds().median()
        rows.append(dict(block=n, rows_in=n_in, rows_kept=len(d), kept_pct=round(100 * len(d) / max(n_in, 1), 1),
                         columns=d.shape[1], median_spacing_s=round(float(spacing), 2),
                         log10_columns=len(logged)))
    print(pd.DataFrame(rows).to_string(index=False))

    gaps = pd.Series(sbl).diff().dt.total_seconds().dropna()
    ev = dict(n_events=len(sbl), min_gap_min=round(gaps.min() / 60, 1), median_gap_min=round(gaps.median() / 60, 1),
              pairs_closer_than_margin=int((gaps < pd.Timedelta(cfg["normal_margin"]).total_seconds()).sum()))
    if iv is not None:
        st = pd.Series(_state_of(sbl - pd.Timedelta("1min"), iv))
        ev["events_in_beam_on"] = int((st == "BEAM_ON").sum())
        hours = ((iv["end"] - iv["start"]).dt.total_seconds() / 3600).groupby(iv["state"]).sum().round(1)
        print("  hours per state:", hours.to_dict())
        print("  state 1 min before each SBL:", st.value_counts().to_dict())
    print(pd.DataFrame([ev]).to_string(index=False))
    if iv is not None and ev["events_in_beam_on"] < 0.8 * len(sbl):
        print("  WARNING: fewer than 80% of the SBLs fall into BEAM_ON. Check on_thr, current_cols "
              "and whether blocks and SBL list are on the same clock.")
    return data, sbl, iv


# =============================================================================== step 2
def build_features(d, cfg):
    """Backward-looking rolling features: value, mean, std, range, slope + block staleness."""
    spacing = d.index.to_series().diff().dt.total_seconds().median()
    win_s = max(pd.Timedelta(cfg["window"]).total_seconds(), cfg["min_rows_per_window"] * spacing)
    win = pd.Timedelta(seconds=win_s)
    roll = d.rolling(win, min_periods=cfg["min_periods"])
    mx, mn = roll.max(), roll.min()
    t_ns = d.index.values.astype("datetime64[ns]").astype("int64")
    pos = np.searchsorted(t_ns, t_ns - int(win_s * 1e9), side="right") - 1
    vals = d.to_numpy(float)
    slope = np.where((pos >= 0)[:, None], (vals - vals[np.clip(pos, 0, None)]) / win_s, 0.0)
    feats = pd.concat({"val": d, "mean": roll.mean(), "std": roll.std(), "range": mx - mn,
                       "slope": pd.DataFrame(slope, index=d.index, columns=d.columns)}, axis=1)
    feats.columns = [f"{c}__{k}" for k, c in feats.columns]

    chg = d.diff().ne(0)
    chg.iloc[0] = False
    t = d.index.to_series()
    since_start = (t - t.iloc[0]).dt.total_seconds()
    stale = pd.DataFrame({c: (t - t.where(chg[c]).ffill()).dt.total_seconds() for c in d.columns})
    stale = stale.apply(lambda col: col.fillna(since_start))
    feats["_stale_max_log"] = np.log1p(stale.max(axis=1))
    feats["_n_changes_win"] = chg.sum(axis=1).rolling(win, min_periods=1).sum()

    gap_thr = pd.Timedelta(seconds=max(3 * spacing, 30))
    seg = (t.diff() > gap_thr).cumsum()
    age_s = (t - t.groupby(seg).transform("min")).dt.total_seconds()
    feats.loc[(age_s < win_s).to_numpy()] = np.nan
    feats.attrs.update(window_s=win_s, spacing_s=float(spacing))
    return feats.replace([np.inf, -np.inf], np.nan)


# =============================================================================== step 3
def make_split(data, cfg):
    t0 = min(d.index.min() for d in data.values())
    t1 = max(d.index.max() for d in data.values())
    span = t1 - t0
    return dict(train_end=t0 + span * cfg["train_frac"],
                calib_end=t0 + span * (cfg["train_frac"] + cfg["calib_frac"]))


def fit_scorer(method, X_train, random_state=0, n_estimators=200):
    """Fit one model on train-normal rows. Returns f(X) -> score, higher = more anomalous."""
    if method == "iforest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(n_estimators=n_estimators, random_state=random_state)
    elif method == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        m = LocalOutlierFactor(n_neighbors=35, novelty=True)
    elif method == "mcd":
        from sklearn.covariance import EllipticEnvelope
        m = EllipticEnvelope(support_fraction=0.75, random_state=random_state)
    elif method == "ocsvm":
        from sklearn.svm import OneClassSVM
        m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    elif method == "sgd_ocsvm":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(Nystroem(gamma=1.0 / X_train.shape[1], n_components=min(300, len(X_train) - 1),
                                   random_state=random_state),
                          SGDOneClassSVM(nu=0.01, random_state=random_state))
    else:
        raise ValueError(f"Unknown method '{method}'.")
    m.fit(X_train)
    return lambda X: -m.score_samples(X)


def robust_z(F, train):
    Xtr = F[train]
    med = Xtr.median()
    scale = np.maximum((Xtr.quantile(0.75) - Xtr.quantile(0.25)) / 1.349, 0.1 * Xtr.std())
    ok = scale.notna() & (scale > 0)
    return (F.loc[:, ok] - med[ok]) / scale[ok]


def score_block(F, sbl, cfg, split, method):
    F = F.dropna()
    train = _normal(F.index, sbl, cfg) & (F.index < split["train_end"])
    if train.sum() < cfg["min_train_rows"]:
        raise ValueError(f"only {int(train.sum())} normal training rows (need {cfg['min_train_rows']})")
    Z = robust_z(F, train)
    if method == "zscore":
        return Z.abs().max(axis=1).rename(method)                    # no clipping here
    Zc = Z.clip(-50, 50)                                             # clipping only as model input
    fit_rows = Zc[train]
    limit = cfg["max_train_rows"] if method == "iforest" else cfg["max_train_rows_slow"]
    if len(fit_rows) > limit:
        fit_rows = fit_rows.sample(limit, random_state=cfg["random_state"])
    f = fit_scorer(method, fit_rows.to_numpy(), cfg["random_state"], cfg["n_estimators"])
    return pd.Series(f(Zc.to_numpy()), index=Zc.index, name=method)


# =============================================================================== step 4
def fuse_scores(scores, sbl, cfg, split, grid):
    margin = pd.Timedelta(cfg["normal_margin"]).total_seconds()
    cols = {}
    for name, s in scores.items():
        g = s.resample(grid).max()
        normal = dist_to_sbl(g.index, sbl) > margin
        cal = g[normal & (g.index >= split["train_end"]) & (g.index < split["calib_end"])].dropna()
        if len(cal) < 20:
            print(f"  WARNING: block '{name}' has only {len(cal)} calibration points - left out.")
            continue
        med, hi = cal.median(), cal.quantile(0.99)
        if (cal >= cal.max()).mean() > 0.01:
            print(f"  WARNING: block '{name}': more than 1% of the calibration scores sit at the maximum "
                  "value - the score is saturated, thresholds will not work.")
        den = hi - med if hi - med > 0 else (cal.std() if cal.std() > 0 else 1.0)
        cols[name] = (g - med) / den
    if not cols:
        raise ValueError("No block has >= 20 calibration-normal points on the score grid - see the table "
                         "'rows available per period' above. Usually the BEAM_ON filter / ramp-up removed "
                         "the calibration period, or the data are short snippets around the SBLs.")
    return pd.DataFrame(cols)


# =============================================================================== step 5
def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _episodes(times, cooldown):
    return 0 if len(times) == 0 else 1 + int((pd.Series(times).diff() > cooldown).sum())


def _covered_hours(times, cap):
    if len(times) < 2:
        return 0.0
    return float(pd.Series(times).diff().dropna().clip(upper=cap).dt.total_seconds().sum()) / 3600


def evaluate_series(s, sbl, cfg, split):
    idx = s.index
    normal = _normal(idx, sbl, cfg)
    cal = s[normal & (idx >= split["train_end"]) & (idx < split["calib_end"])].dropna()
    test_norm = s[normal & (idx >= split["calib_end"])].dropna()
    if len(cal) < 20 or len(test_norm) < 20:
        raise ValueError(f"too few normal points (calibration={len(cal)}, test={len(test_norm)})")
    lookback, buffer = pd.Timedelta(cfg["lookback"]), pd.Timedelta(cfg["buffer"])
    events = [t for t in sbl if split["calib_end"] <= t <= idx.max()]
    cover_h = _covered_hours(test_norm.index, pd.Timedelta(cfg["max_gap"]))
    window_h = (lookback - buffer).total_seconds() / 3600
    rng = np.random.default_rng(cfg["random_state"])

    summary, per_event = [], []
    for q in cfg["thr_quantiles"]:
        thr = float(cal.quantile(q))
        n_fa = _episodes(test_norm.index[(test_norm > thr).to_numpy()], pd.Timedelta(cfg["cooldown"]))
        fa_h = n_fa / cover_h if cover_h > 0 else np.nan
        det, lead, n_nodata = [], [], 0
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
        dl = lead[np.isfinite(lead)]
        lead_ci = "n/a"
        if len(dl) >= 3:
            meds = [np.median(dl[rng.integers(0, len(dl), len(dl))]) for _ in range(cfg["n_boot"])]
            lead_ci = "{:.0f}-{:.0f}".format(*np.percentile(meds, [2.5, 97.5]))
        chance = 1 - np.exp(-fa_h * window_h) if not np.isnan(fa_h) else np.nan
        summary.append(dict(
            quantile=q, thr=round(thr, 2), events=len(det), no_data=n_nodata, detected=int(det.sum()),
            det_rate=round(det.mean(), 3) if len(det) else np.nan,
            det_ci95=f"{lo:.2f}-{hi:.2f}" if not np.isnan(lo) else "n/a",
            chance_det_rate=round(chance, 3),
            above_chance=bool(lo > chance) if not (np.isnan(lo) or np.isnan(chance)) else False,
            median_lead_s=round(float(np.nanmedian(lead)), 1) if np.isfinite(lead).any() else np.nan,
            lead_ci95=lead_ci, false_alarms=n_fa,
            fa_per_h=round(fa_h, 3) if not np.isnan(fa_h) else np.nan, covered_h=round(cover_h, 1)))
    return pd.DataFrame(summary), pd.DataFrame(per_event)


# =============================================================================== main
def as_sbl_series(sbl_times, ts_col="timestamp"):
    """Accept a Series, list/array, DatetimeIndex or a DataFrame (one column, or a 'timestamp' column)."""
    if isinstance(sbl_times, pd.DataFrame):
        if ts_col in sbl_times.columns:
            sbl_times = sbl_times[ts_col]
        elif sbl_times.shape[1] == 1:
            sbl_times = sbl_times.iloc[:, 0]
        else:
            raise ValueError(f"sbl_times has {sbl_times.shape[1]} columns {list(sbl_times.columns)} - "
                             "pass the one with the SBL timestamps, e.g. sbl_times['timestamp'].")
    arr = np.asarray(sbl_times)
    if arr.ndim == 2 and arr.shape[1] == 1:
        sbl_times = arr[:, 0]
    return pd.Series(sbl_times).reset_index(drop=True)


def run(blocks, sbl_times, overrides=None, cfg=None):
    """overrides (or cfg=...): dict with the settings to change, e.g. {"on_thr": 400}."""
    user = {**(overrides or {}), **(cfg or {})}
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown setting(s) {unknown}. Valid names: {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **user}
    sbl_times = as_sbl_series(sbl_times, cfg["ts_col"])

    _section("Step 0: leak guard + time format")
    blocks = drop_sbl_columns(blocks, cfg)
    check_time_format(blocks, sbl_times, cfg)

    _section("Step 1: preprocessing (BEAM_ON filter, cleaning)")
    data, sbl, iv = preprocess(blocks, sbl_times, cfg)

    _section("Step 2: features")
    feats = {n: build_features(d, cfg) for n, d in data.items()}
    for n, F in feats.items():
        print(f"  {n:11s} {F.shape[1]:3d} features, window {F.attrs['window_s'] / 60:.1f} min, "
              f"row spacing {F.attrs['spacing_s']:.2f} s")

    _section("Steps 3+4: models (train-normal only) + fusion")
    split = make_split(data, cfg)
    print(f"  train until {split['train_end']} | calibration until {split['calib_end']} | test after")
    n_test = int(((sbl >= split["calib_end"])).sum())
    print(f"  SBL events: {len(sbl)} total, {n_test} in the test period")
    grid = cfg["grid"]
    if grid == "auto":
        sp = min(F.attrs["spacing_s"] for F in feats.values())
        grid = f"{max(30, int(np.ceil(sp / 10) * 10))}s"
    print(f"  common score grid: {grid}")

    diag = []
    for n, F in feats.items():
        ok = F.dropna().index
        nm = _normal(ok, sbl, cfg)
        diag.append(dict(block=n, rows=len(F), rows_with_features=len(ok),
                         train_normal=int((nm & (ok < split["train_end"])).sum()),
                         calib_normal=int((nm & (ok >= split["train_end"]) & (ok < split["calib_end"])).sum()),
                         test_rows=int((ok >= split["calib_end"]).sum())))
    print("  rows available per period (normal = farther than normal_margin from every SBL):")
    print(pd.DataFrame(diag).to_string(index=False))

    scores, normed, fused = {}, {}, {}
    for m in cfg["methods"]:
        scores[m] = {}
        for n, F in feats.items():
            try:
                scores[m][n] = score_block(F, sbl, cfg, split, m)
            except ValueError as e:
                print(f"  [{m}] block '{n}' skipped: {e}")
        normed[m] = fuse_scores(scores[m], sbl, cfg, split, grid)
        fused[m] = normed[m].max(axis=1)

    _section("Step 5: validation on the test period")
    summary, per_event = {}, {}
    for m in cfg["methods"]:
        for label, series in [("ALL", fused[m])] + [(n, normed[m][n]) for n in normed[m].columns]:
            key = f"{m} | {label}"
            try:
                summary[key], per_event[key] = evaluate_series(series, sbl, cfg, split)
            except ValueError as e:
                print(f"  {key}: not evaluated - {e}")
    for key, table in summary.items():
        print(f"\n--- {key} ---\n{table.to_string(index=False)}")
    print("\nHow to read: above_chance = True only if the LOWER end of det_ci95 lies above "
          "chance_det_rate. p-values / rates are descriptive (correlated rows, many comparisons).")

    return dict(cfg=cfg, data=data, iv=iv, sbl=sbl, features=feats, split=split, grid=grid,
                scores=scores, normed=normed, fused=fused, summary=summary, per_event=per_event)


# =============================================================================== plots
def plot_event_locked(out, method="iforest", quantile=0.999, lookback="30min", max_events=60):
    """Fused score vs. seconds before SBL for the test events (gray = one event, red = median)."""
    import matplotlib.pyplot as plt
    fused, sbl, cfg, split = out["fused"][method], out["sbl"], out["cfg"], out["split"]
    idx = fused.index
    cal = fused[_normal(idx, sbl, cfg) & (idx >= split["train_end"]) & (idx < split["calib_end"])].dropna()
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
    ax.axhline(thr, color="black", ls="--", label=f"threshold (q={quantile})")
    ax.axvline(0, color="red")
    ax.set_xlabel("seconds relative to SBL (0 = SBL)")
    ax.set_ylabel("fused score (1 = 99th pct. of normal)")
    ax.set_title(f"{method}: fused anomaly score before {len(pooled)} test SBL events")
    ax.legend()
    fig.tight_layout()
    return fig


def per_feature_scores(out, block, method="zscore", max_fit_rows=5000):
    """One score per feature (1D model each), normalised: 0 = median, 1 = q99 of calibration normal."""
    F, sbl, cfg, split = out["features"][block].dropna(), out["sbl"], out["cfg"], out["split"]
    normal = _normal(F.index, sbl, cfg)
    train = normal & (F.index < split["train_end"])
    calib = normal & (F.index >= split["train_end"]) & (F.index < split["calib_end"])
    if train.sum() < cfg["min_train_rows"] or calib.sum() < 20:
        raise ValueError(f"too few normal rows (train={int(train.sum())}, calibration={int(calib.sum())})")
    Z = robust_z(F, train)
    tr_idx = np.flatnonzero(train)
    if len(tr_idx) > max_fit_rows:
        tr_idx = np.random.default_rng(cfg["random_state"]).choice(tr_idx, max_fit_rows, replace=False)
    res = {}
    for c in Z.columns:
        x = Z[c].to_numpy()
        s = np.abs(x) if method == "zscore" else fit_scorer(method, x[tr_idx, None], cfg["random_state"])(x[:, None])
        cal = s[calib]
        m, hi = np.median(cal), np.quantile(cal, 0.99)
        den = hi - m if hi - m > 0 else (cal.std() if cal.std() > 0 else 1.0)
        res[c] = (s - m) / den
    return pd.DataFrame(res, index=F.index)


def _last_points(S, t, n_points, buffer):
    w = S[S.index < t - pd.Timedelta(buffer)].tail(n_points)
    w.index = (w.index - t).total_seconds()
    return w


def feature_heatmaps(out, block, method="zscore", n_points=10, buffer="0s", top_k=20, period="test",
                     max_events=12, vmax=3.0):
    """Returns (fig_per_event, fig_overview, table).
    fig_per_event: one panel per SBL, rows = features, columns = last n rows before the SBL.
    fig_overview:  rows = SBL events, columns = features, colour = max over the last n rows."""
    import matplotlib.pyplot as plt
    S = per_feature_scores(out, block, method)
    sbl, split = out["sbl"], out["split"]
    if period == "test":
        sbl = sbl[sbl >= split["calib_end"]]
    events = [t for t in sbl if S.index.min() < t <= S.index.max()]
    events = [t for t in events if len(_last_points(S, t, n_points, buffer))]
    if not events:
        raise ValueError(f"No SBL with data in block '{block}' for period='{period}'.")
    tab_all = pd.DataFrame({t: _last_points(S, t, n_points, buffer).max() for t in events}).T
    feats = tab_all.max().sort_values(ascending=False).index[:top_k]
    tab = tab_all[feats]

    ev = events[:max_events]
    n_cols = min(4, len(ev))
    n_rows = int(np.ceil(len(ev) / n_cols))
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 0.22 * len(feats) * n_rows + 1.4),
                              squeeze=False, layout="constrained", sharey=True)
    im = None
    for ax, t in zip(axes.ravel(), ev):
        w = _last_points(S, t, n_points, buffer)[feats]
        im = ax.imshow(w.T.clip(lower=0).to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(w)))
        ax.set_xticklabels([f"{x:.0f}" for x in w.index], rotation=90, fontsize=7)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=7)
        ax.set_title(f"SBL {t:%Y-%m-%d %H:%M:%S}", fontsize=8)
    for ax in axes.ravel()[len(ev):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("seconds before SBL", fontsize=8)
    fig1.colorbar(im, ax=axes, shrink=0.6, label=f"feature score (1 = 99th pct. normal, cap {vmax:g})")
    fig1.suptitle(f"{block} ({method}): last {n_points} rows before each SBL")

    fig2, ax = plt.subplots(figsize=(0.28 * len(feats) + 3, 0.22 * len(events) + 2), layout="constrained")
    im = ax.imshow(tab.clip(lower=0).to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([f"{t:%Y-%m-%d %H:%M}" for t in events], fontsize=7)
    ax.set_xlabel("feature")
    ax.set_ylabel("SBL event")
    fig2.colorbar(im, ax=ax, shrink=0.7, label=f"max score, last {n_points} rows\n(1 = 99th pct. normal)")
    ax.set_title(f"{block} ({method}): which feature is unusual before which SBL?")
    return fig1, fig2, tab
