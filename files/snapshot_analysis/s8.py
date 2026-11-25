from lib import *
rng = np.random.default_rng(11)
NR = 1000
# candidate pseudo-event times: uniform over beam-on periods, > 8 h from any SBL
grid = np.arange(BT[0], BT[-1], np.timedelta64(10, "m"))
gi = at_(grid); gon = (BC[gi] > 50) & ((grid - BT[gi]).astype("int64")/H < 2.5)
dmin = np.full(len(grid), np.inf)
j = np.searchsorted(SN, grid)
for jj in (j-1, j):
    ok = (jj >= 0) & (jj < len(SN))
    dmin[ok] = np.minimum(dmin[ok], np.abs((grid[ok] - SN[jj[ok]]).astype("int64"))/H)
cand_t = grid[gon & (dmin > 8)]
print("candidate pseudo-event times:", len(cand_t))

def snap_pct(P, kind, ref):
    """matched percentile of every snapshot for every feature (NaN if not usable)"""
    M = P["X"] if kind == "level" else P["dX"]; t, c = P["t"], P["c"]
    out = np.full(M.shape, np.nan)
    for i in np.where(P["on"])[0]:
        m = ref & (np.abs((t - t[i]).astype("int64"))/(24*H) <= 7) & (np.abs(c - c[i]) <= 0.1*max(c[i], 1)); m[i] = False
        R = M[m]
        if len(R) < 10: continue
        v = M[i]
        cnt = np.isfinite(R).sum(0)
        pct = ((R < v).sum(0) + 0.5*(R == v).sum(0)) / np.maximum(cnt, 1)
        pct[(cnt < 10) | ~np.isfinite(v)] = np.nan
        out[i] = pct
    return out

def pick(P, events, since, kth=1, min_age=0.0):
    t = P["t"]; i = np.searchsorted(t, events) - kth
    ok = i >= 0; i = np.where(ok, i, 0)
    age = (events - t[i]).astype("int64")/H
    ok &= (age >= min_age) & (age < 2.5*kth) & P["on"][i] & (since[i] > 1)
    return i[ok]

rows, store = [], {}
for b in blocks:
    P = prep_block(b); ref, since = ref_mask(P, SN)
    for kind in ("level", "change"):
        PCT = snap_pct(P, kind, ref)
        null_sets = [pick(P, np.sort(rng.choice(cand_t, len(SN), replace=False)), since) for _ in range(NR)]
        for kth in (1, 2):
            idx = pick(P, SN, since, kth)
            obs = np.nanmean(PCT[idx], 0); nobs = np.isfinite(PCT[idx]).sum(0)
            null = np.array([np.nanmean(PCT[s], 0) for s in null_sets])
            for k, f in enumerate(P["feats"]):
                nl = null[:, k]; nl = nl[np.isfinite(nl)]
                if nobs[k] < 15 or len(nl) < 100: continue
                mu = np.median(nl)
                p = (1 + np.sum(np.abs(nl - mu) >= abs(obs[k] - mu))) / (1 + len(nl))
                rows.append(dict(block=b, feature=f, kind=kind, kth=kth, n=int(nobs[k]), mean_pct=obs[k],
                                 null_median=mu, null_lo=np.quantile(nl, .025), null_hi=np.quantile(nl, .975),
                                 effect=obs[k]-mu, p_perm=p))
                if kth == 1: store[(b, f, kind)] = (PCT[idx][:, k], nl)
res = pd.DataFrame(rows)
r1 = res[res.kth == 1].copy(); m = len(r1)
r1["p_bonf"] = np.minimum(r1.p_perm*m, 1)
r1 = r1.merge(res[res.kth == 2][["block","feature","kind","effect","p_perm"]].rename(columns={"effect":"effect_k2","p_perm":"p_k2"}), on=["block","feature","kind"], how="left")
r1["abs_eff"] = r1.effect.abs(); r1 = r1.sort_values("abs_eff", ascending=False)
r1.to_csv(OUT+"perm_ranking.csv", index=False); pickle.dump(store, open(OUT+"perm_store.pkl","wb"))
pd.set_option("display.width", 220)
print("tests:", m, " min possible p:", 1/(NR+1))
print(r1.head(20).round(3).to_string(index=False))
print("\n# with p_perm < 0.05:", (r1.p_perm < 0.05).sum(), " expected by chance:", round(0.05*m,1))
