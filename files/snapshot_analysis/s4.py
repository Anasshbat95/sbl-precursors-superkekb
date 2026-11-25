from common import *
from scipy.stats import mannwhitneyu
blocks, sbl = load()
sn = np.sort(sbl.timestamp.to_numpy("datetime64[ns]"))

def snapshots(d):
    d = d.sort_values("timestamp").copy()
    ts = pd.to_datetime(d["timestamp"])
    d["burst"] = (ts.diff().dt.total_seconds() > 60).cumsum()
    g = d.groupby("burst")
    s = g.last()                      # last (forward-filled) state of each burst
    s["timestamp"] = g["timestamp"].max()
    return s.reset_index(drop=True)

# beam state per snapshot time from the beam block
beam_s = snapshots(blocks["beam"])
bt = beam_s.timestamp.to_numpy("datetime64[ns]")
bon = (beam_s[["her_current","ler_current"]].max(axis=1) > 50).to_numpy()
def beam_on_at(t):
    i = np.searchsorted(bt, t, side="right") - 1
    ok = (i >= 0)
    out = np.zeros(len(t), bool)
    age = (t[ok] - bt[i[ok]]).astype("int64")/3.6e12
    out[ok] = bon[i[ok]] & (age < 2.5)
    return out

rows = []
for name, d in blocks.items():
    s = snapshots(d)
    feats = [c for c in s.columns if c not in ("timestamp","burst")]
    t = s.timestamp.to_numpy("datetime64[ns]")
    # distance to next SBL / previous SBL (hours)
    j = np.searchsorted(sn, t)
    to_next = np.where(j < len(sn), (sn[np.minimum(j, len(sn)-1)] - t).astype("int64")/3.6e12, np.inf)
    since_prev = np.where(j > 0, (t - sn[np.maximum(j-1, 0)]).astype("int64")/3.6e12, np.inf)
    on = beam_on_at(t)
    X = s[feats].to_numpy(float)
    dX = np.vstack([np.full(len(feats), np.nan), np.diff(X, axis=0)])
    dt_prev = np.r_[np.nan, np.diff(t).astype("int64")/3.6e12]
    dX[~(dt_prev < 2.5)] = np.nan     # only consecutive snapshots
    # pre-SBL snapshot: last snapshot before each SBL, at most 2.5 h before, beam on
    pre_idx = []
    for e in sn:
        i = np.searchsorted(t, e) - 1
        if i >= 0 and (e - t[i]).astype("int64")/3.6e12 < 2.5 and on[i] and since_prev[i] > 1:
            pre_idx.append(i)
    pre_idx = np.unique(pre_idx)
    ref = on & (to_next > 6) & (since_prev > 2)
    for k, f in enumerate(feats):
        for kind, M in (("level", X), ("change", dX)):
            a = M[pre_idx, k]; a = a[np.isfinite(a)]
            r = M[ref, k]; r = r[np.isfinite(r)]
            if kind == "change":
                a, r = np.abs(a), np.abs(r)
            if len(a) < 10 or len(r) < 50 or np.nanstd(r) == 0:
                continue
            u, p = mannwhitneyu(a, r, alternative="two-sided")
            auc = u / (len(a)*len(r))
            rows.append(dict(block=name, feature=f, kind=kind, n_sbl=len(a), n_ref=len(r),
                             median_sbl=np.median(a), median_ref=np.median(r), auc=auc,
                             strength=abs(auc-0.5), p=p))
res = pd.DataFrame(rows)
m = len(res); res["p_bonf"] = np.minimum(res.p*m, 1)
res = res.sort_values("strength", ascending=False)
res.to_csv(OUT+"snapshot_ranking.csv", index=False)
pd.set_option("display.width", 200)
print("tests:", m)
print(res.head(25).round(4).to_string(index=False))
