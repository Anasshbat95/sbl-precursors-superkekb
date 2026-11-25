"""
sbl_anomaly.py - Anomaly detection for SBL, trained on normal data only.

Pipeline (the raw data of the 5 blocks is NOT merged):
  1. Per block: a model predicts the next row (LSTM or linear).
     score(t) = max_i |x_hat_i - x_i| / sigma_i      (sigma from normal data)
  2. Smooth the score and put it on a common time grid -> only scores are fused.
  3. Threshold thr = mu + k*sigma from normal data (calibration period).
  4. Per-SBL evaluation: detected? how many seconds ahead? false alarms per hour?

Usage:
    from sbl_anomaly import run
    out = run(
        {"BEAM": BEAM_on, "BACKGROUND": BACKGROUND_on, "INJECTION": INJECTION_on,
         "VACUUM": VACUUM_on, "POSITION": POSITION_on},   # ideally filtered to state == "BEAM_ON"
        SBL_timestamps_Germany["timestamp"],
        {"ts_col": "timestamp", "model": "linear"},        # try "linear" first, then "lstm"
    )
Note: if blocks are pre-filtered, the off periods become time gaps larger than `gap`,
so training windows never span across them.
"""
import numpy as np
import pandas as pd

CFG = dict(
    ts_col="timestamp",
    block_tz=None, sbl_tz=None,        # e.g. "Europe/Berlin" if timestamps are local and tz-naive
    normal_margin="15min",             # normal = farther than this from EVERY SBL
    train_frac=0.6, calib_frac=0.15,   # chronological split (the rest is the test period)
    gap="60s",                         # time gap larger than this starts a new segment
    model="lstm",                      # "lstm" (needs torch) or "linear" (numpy only)
    seq_len=30, hidden=32, epochs=20, batch=256, lr=1e-3, ridge=1.0, seed=0,
    smooth="30s", grid="10s",
    lookback="10min", buffer="30s",    # detection window: [SBL - lookback, SBL - buffer]
    cooldown="5min",                   # alarms closer than this count as one episode
    k_values=(3, 5, 8, 12, 20),
)


# ---------------------------------------------------------------- time helpers
def to_utc_naive(x, tz=None):
    """Timestamps -> UTC without tz info, so all blocks and SBLs are comparable."""
    x = pd.to_datetime(pd.Series(x).reset_index(drop=True))
    if x.dt.tz is None and tz is not None:
        x = x.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    if x.dt.tz is not None:
        x = x.dt.tz_convert("UTC").dt.tz_localize(None)
    return x


def dist_to_sbl(ts, sbl):
    """Distance (seconds) from each timestamp to the nearest SBL."""
    t = np.asarray(ts, dtype="datetime64[ns]").astype("int64")
    s = np.sort(np.asarray(sbl, dtype="datetime64[ns]").astype("int64"))
    i = np.searchsorted(s, t)
    prev = np.abs(t - s[np.clip(i - 1, 0, len(s) - 1)])
    nxt = np.abs(s[np.clip(i, 0, len(s) - 1)] - t)
    return np.minimum(prev, nxt) / 1e9


def prep(df, cfg):
    """Block -> numeric columns only, DatetimeIndex (UTC), sorted, no NaN / constant columns."""
    ts = to_utc_naive(df[cfg["ts_col"]], cfg["block_tz"])
    d = df.drop(columns=cfg["ts_col"]).select_dtypes("number").reset_index(drop=True)
    d.index = pd.DatetimeIndex(ts.to_numpy())
    d = d[d.index.notna()].sort_index()
    d = d.loc[:, d.std() > 0].dropna()
    return d


# ---------------------------------------------------------------- models
def make_windows(X, seg, L):
    """Windows (L rows -> next row), never crossing a time gap."""
    xs, ys, idx = [], [], []
    for s in np.unique(seg):
        ii = np.where(seg == s)[0]
        if len(ii) <= L:
            continue
        w = np.lib.stride_tricks.sliding_window_view(X[ii], (L + 1, X.shape[1]))[:, 0]
        xs.append(w[:, :L]); ys.append(w[:, L]); idx.append(ii[L:])
    if not xs:
        raise ValueError("No segment is long enough - increase 'gap' or decrease 'seq_len'.")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)


def fit_predict_linear(W, Y, tr, ca, cfg):
    """Ridge regression: next row from the last seq_len rows (fast baseline)."""
    A = W.reshape(len(W), -1)
    A1 = np.hstack([A, np.ones((len(A), 1))])
    At = A1[tr]
    coef = np.linalg.solve(At.T @ At + cfg["ridge"] * np.eye(At.shape[1]), At.T @ Y[tr])
    return A1 @ coef


def fit_predict_lstm(W, Y, tr, ca, cfg):
    """LSTM: next row from the last seq_len rows; best model by validation loss (calibration data)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(cfg["seed"])
    F = W.shape[2]

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, cfg["hidden"], batch_first=True)
            self.out = nn.Linear(cfg["hidden"], F)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.out(o[:, -1])

    Wt, Yt = torch.tensor(W, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    tr_i, ca_i = np.where(tr)[0], np.where(ca)[0]
    model, best, best_state = Predictor(), np.inf, None
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = np.random.default_rng(cfg["seed"] + ep).permutation(tr_i)
        for b in range(0, len(perm), cfg["batch"]):
            bi = perm[b:b + cfg["batch"]]
            opt.zero_grad()
            loss_fn(model(Wt[bi]), Yt[bi]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Wt[ca_i]), Yt[ca_i]).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(Wt[i:i + 2048]).numpy() for i in range(0, len(Wt), 2048)])


# ---------------------------------------------------------------- per-block scores
def _masks(d, sbl, cfg, split):
    ts = d.index
    normal = dist_to_sbl(ts, sbl) > pd.Timedelta(cfg["normal_margin"]).total_seconds()
    tr = normal & (ts < split["train_end"])
    ca = normal & (ts >= split["train_end"]) & (ts < split["calib_end"])
    return normal, tr, ca


def score_block(d, sbl, cfg, split):
    ts = d.index
    _, tr, ca = _masks(d, sbl, cfg, split)
    feats = d.to_numpy(float)
    X = (feats - feats[tr].mean(0)) / (feats[tr].std(0) + 1e-9)     # scaled with training-normal data only

    seg = (pd.Series(ts).diff() > pd.Timedelta(cfg["gap"])).cumsum().to_numpy()
    W, Y, idx = make_windows(X, seg, cfg["seq_len"])
    tr_w, ca_w = tr[idx], ca[idx]
    if tr_w.sum() < 100 or ca_w.sum() < 50:
        raise ValueError(f"Too few normal windows (train={tr_w.sum()}, calib={ca_w.sum()}). "
                         "Adjust train_frac/calib_frac or normal_margin.")

    fit = fit_predict_lstm if cfg["model"] == "lstm" else fit_predict_linear
    pred = fit(W, Y, tr_w, ca_w, cfg)
    res = pred - Y
    sig = res[ca_w].std(0) + 1e-9                                   # typical error on normal data
    s = pd.Series((np.abs(res) / sig).max(axis=1), index=ts[idx])   # max over features
    return s.rolling(cfg["smooth"]).mean()


def baseline_block(d, sbl, cfg, split):
    """Reference method: rolling std (max over features), normalised on normal data."""
    _, tr, _ = _masks(d, sbl, cfg, split)
    feats = d.to_numpy(float)
    X = pd.DataFrame((feats - feats[tr].mean(0)) / (feats[tr].std(0) + 1e-9), index=d.index)
    roll = X.rolling(cfg["smooth"]).std()
    return (roll / roll[tr].mean()).max(axis=1)


# ---------------------------------------------------------------- fusion & evaluation
def fuse(scores, sbl, cfg, split):
    """Scores (not raw data!) on a common grid, each block normalised by its normal-data 99% quantile, then max."""
    cols = {}
    for name, s in scores.items():
        g = s.resample(cfg["grid"]).max()
        normal = dist_to_sbl(g.index, sbl) > pd.Timedelta(cfg["normal_margin"]).total_seconds()
        cal = g[normal & (g.index >= split["train_end"]) & (g.index < split["calib_end"])].dropna()
        cols[name] = g / cal.quantile(0.99)
    return pd.DataFrame(cols).max(axis=1, skipna=True)


def evaluate(fused, sbl, cfg, split):
    idx = fused.index
    normal = dist_to_sbl(idx, sbl) > pd.Timedelta(cfg["normal_margin"]).total_seconds()
    cal = fused[normal & (idx >= split["train_end"]) & (idx < split["calib_end"])].dropna()
    test_norm = fused[normal & (idx >= split["calib_end"])].dropna()
    events = sbl[(sbl >= split["calib_end"]) & (sbl <= idx.max())]
    lookback, buffer = pd.Timedelta(cfg["lookback"]), pd.Timedelta(cfg["buffer"])
    hours = len(test_norm) * pd.Timedelta(cfg["grid"]).total_seconds() / 3600

    rows = []
    for k in cfg["k_values"]:
        thr = cal.mean() + k * cal.std()
        # false alarms: alarm episodes during normal times of the test period
        starts = test_norm.index[test_norm > thr]
        n_fa = int((starts.to_series().diff() > pd.Timedelta(cfg["cooldown"])).sum() + (len(starts) > 0))
        # per-event detection: first alarm inside [SBL - lookback, SBL - buffer]
        leads, n_ev = [], 0
        for t in events:
            w = fused[t - lookback: t - buffer].dropna()
            if len(w) == 0:
                continue
            n_ev += 1
            hit = w.index[w > thr]
            if len(hit):
                leads.append((t - hit[0]).total_seconds())
        rows.append(dict(k=k, thr=round(float(thr), 2), events=n_ev, detected=len(leads),
                         det_rate=round(len(leads) / max(n_ev, 1), 3),
                         median_lead_s=float(np.median(leads)) if leads else np.nan,
                         false_alarms_per_h=round(n_fa / hours, 3) if hours else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main entry point
def run(blocks, sbl_times, cfg_overrides=None):
    cfg = {**CFG, **(cfg_overrides or {})}
    sbl = pd.DatetimeIndex(to_utc_naive(sbl_times, cfg["sbl_tz"]).dropna().sort_values())
    data = {n: prep(df, cfg) for n, df in blocks.items()}

    t0, t1 = min(d.index.min() for d in data.values()), max(d.index.max() for d in data.values())
    span = t1 - t0
    split = dict(train_end=t0 + span * cfg["train_frac"],
                 calib_end=t0 + span * (cfg["train_frac"] + cfg["calib_frac"]))
    print(f"Train until {split['train_end']} | calibration until {split['calib_end']} | test after")

    model_scores, base_scores = {}, {}
    for n, d in data.items():
        print(f"[{n}] {d.shape[0]} rows, {d.shape[1]} features -> model '{cfg['model']}'")
        model_scores[n] = score_block(d, sbl, cfg, split)
        base_scores[n] = baseline_block(d, sbl, cfg, split)

    results = {}
    for label, sc in [(cfg["model"].upper(), model_scores), ("BASELINE rolling-std", base_scores)]:
        results[f"{label} | ALL blocks"] = evaluate(fuse(sc, sbl, cfg, split), sbl, cfg, split)
        for n in sc:
            results[f"{label} | only {n}"] = evaluate(fuse({n: sc[n]}, sbl, cfg, split), sbl, cfg, split)
    for name, table in results.items():
        print(f"\n=== {name} ===\n{table.to_string(index=False)}")
    return dict(results=results, model_scores=model_scores, base_scores=base_scores, split=split, cfg=cfg)
