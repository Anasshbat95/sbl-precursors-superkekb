from common import *
from scipy.stats import wilcoxon
blocks, sbl = load()
SN = np.sort(sbl.timestamp.to_numpy("datetime64[ns]"))
H = 3.6e12
def snapshots(d):
    d = d.sort_values("timestamp").copy()
    ts = pd.to_datetime(d["timestamp"])
    d["burst"] = (ts.diff().dt.total_seconds() > 60).cumsum()
    g = d.groupby("burst"); s = g.last(); s["timestamp"] = g["timestamp"].max()
    return s.reset_index(drop=True)
beam_s = snapshots(blocks["beam"])
BT = beam_s.timestamp.to_numpy("datetime64[ns]")
BC = beam_s[["her_current","ler_current"]].max(axis=1).to_numpy()
def at_(t):
    return np.clip(np.searchsorted(BT, t, side="right") - 1, 0, len(BT)-1)
def prep_block(name):
    s = snapshots(blocks[name]); feats = [c for c in s.columns if c not in ("timestamp","burst")]
    t = s.timestamp.to_numpy("datetime64[ns]")
    i = at_(t); on = (BC[i] > 50) & ((t - BT[i]).astype("int64")/H < 2.5); c = BC[i]
    X = s[feats].to_numpy(float)
    dX = np.vstack([np.full(len(feats), np.nan), np.diff(X, axis=0)])
    dX[~(np.r_[np.nan, np.diff(t).astype("int64")/H] < 2.5)] = np.nan
    return dict(t=t, on=on, c=c, X=X, dX=np.abs(dX), feats=feats)
def ref_mask(P, events):
    j = np.searchsorted(events, P["t"])
    to_next = np.where(j < len(events), (events[np.minimum(j, len(events)-1)] - P["t"]).astype("int64")/H, np.inf)
    since = np.where(j > 0, (P["t"] - events[np.maximum(j-1, 0)]).astype("int64")/H, np.inf)
    return P["on"] & (to_next > 6) & (since > 2), since
def matched_pcts(P, k, kind, events, kth=1, min_age=0.0, ref_events=None):
    """percentile of the kth snapshot before each event within matched normal snapshots"""
    M = P["X"] if kind == "level" else P["dX"]
    ref, since = ref_mask(P, SN if ref_events is None else ref_events)
    t, c, out = P["t"], P["c"], []
    for e in events:
        i = np.searchsorted(t, e) - kth
        if i < 0: continue
        age = (e - t[i]).astype("int64")/H
        if not (min_age <= age < 2.5*kth) or not P["on"][i] or since[i] <= 1: continue
        v = M[i, k]
        if not np.isfinite(v): continue
        m = ref & (np.abs((t - t[i]).astype("int64"))/(24*H) <= 7) & (np.abs(c - c[i]) <= 0.1*max(c[i], 1))
        m[i] = False
        r = M[m, k]; r = r[np.isfinite(r)]
        if len(r) < 10: continue
        out.append((np.sum(r < v) + 0.5*np.sum(r == v))/len(r))
    return np.array(out)
def summarize(p):
    if len(p) < 15: return dict(n=len(p), mean_pct=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    rng = np.random.default_rng(1); bs = rng.choice(p, (2000, len(p))).mean(1)
    pv = wilcoxon(p - 0.5).pvalue if np.any(p != 0.5) else 1.0
    return dict(n=len(p), mean_pct=p.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), p=pv)
