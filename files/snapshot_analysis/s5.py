from common import *
from scipy.stats import wilcoxon, binomtest
exec(open(os.path.dirname(os.path.abspath(__file__))+"/s4.py").read().split("rows = []")[0])   # reuse snapshots(), beam_on_at()
# current (max of rings) at any time from beam snapshots
bcur = beam_s[["her_current","ler_current"]].max(axis=1).to_numpy()
def cur_at(t):
    i = np.clip(np.searchsorted(bt, t, side="right") - 1, 0, len(bt)-1)
    return bcur[i]
rows, pct_store = [], {}
for name, d in blocks.items():
    s = snapshots(d)
    feats = [c for c in s.columns if c not in ("timestamp","burst")]
    t = s.timestamp.to_numpy("datetime64[ns]")
    j = np.searchsorted(sn, t)
    to_next = np.where(j < len(sn), (sn[np.minimum(j, len(sn)-1)] - t).astype("int64")/3.6e12, np.inf)
    since_prev = np.where(j > 0, (t - sn[np.maximum(j-1, 0)]).astype("int64")/3.6e12, np.inf)
    on = beam_on_at(t); c = cur_at(t)
    X = s[feats].to_numpy(float)
    dX = np.vstack([np.full(len(feats), np.nan), np.diff(X, axis=0)])
    dX[~(np.r_[np.nan, np.diff(t).astype("int64")/3.6e12] < 2.5)] = np.nan
    ref = on & (to_next > 6) & (since_prev > 2)
    pre = []
    for e in sn:
        i = np.searchsorted(t, e) - 1
        if i >= 0 and (e - t[i]).astype("int64")/3.6e12 < 2.5 and on[i] and since_prev[i] > 1:
            pre.append(i)
    pre = np.unique(pre)
    for k, f in enumerate(feats):
        for kind, M in (("level", X), ("change", dX)):
            pcts = []
            for i in pre:
                v = M[i, k]
                if not np.isfinite(v): continue
                m = ref & (np.abs((t - t[i]).astype("int64"))/8.64e13 <= 7) & (np.abs(c - c[i]) <= 0.1*max(c[i],1))
                r = M[m, k]; r = r[np.isfinite(r)]
                if len(r) < 10: continue
                if kind == "change": v, r = abs(v), np.abs(r)
                pcts.append((np.sum(r < v) + 0.5*np.sum(r == v)) / len(r))
            pcts = np.array(pcts)
            if len(pcts) < 15: continue
            # null: percentile uniform -> median 0.5
            p = wilcoxon(pcts - 0.5).pvalue if np.any(pcts != 0.5) else 1.0
            rows.append(dict(block=name, feature=f, kind=kind, n=len(pcts), mean_pct=pcts.mean(),
                             share_above_90=np.mean(pcts > 0.9), share_below_10=np.mean(pcts < 0.1), p=p))
            pct_store[(name, f, kind)] = pcts
res = pd.DataFrame(rows)
res["p_bonf"] = np.minimum(res.p*len(res), 1)
res["strength"] = (res.mean_pct - 0.5).abs()
res = res.sort_values("strength", ascending=False)
res.to_csv(OUT+"matched_ranking.csv", index=False)
pickle.dump(pct_store, open(OUT+"matched_pcts.pkl","wb"))
pd.set_option("display.width", 200)
print("tests:", len(res))
print(res.head(20).round(4).to_string(index=False))
