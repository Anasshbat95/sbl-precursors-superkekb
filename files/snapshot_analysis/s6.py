from common import *
exec(open(os.path.dirname(os.path.abspath(__file__))+"/s4.py").read().split("rows = []")[0])
cur = beam_s[["her_current","ler_current"]].to_numpy()
# random reference: drop between consecutive snapshots at random times
rng = np.random.default_rng(0)
def drop_rate(times):
    i = np.searchsorted(bt, times) - 1
    ok = (i >= 0) & (i+1 < len(bt))
    i = i[ok]
    gap = (bt[i+1]-bt[i]).astype("int64")/3.6e12
    i = i[gap < 2.5]
    b, a = cur[i], cur[i+1]
    on = b.max(1) > 400
    drop = on & ((a < 0.5*b) & (b > 400)).any(1)
    return drop.sum()/max(on.sum(),1), on.sum()
tr = np.sort(rng.choice(bt, 3000)) + np.timedelta64(30, "m")
print("random times: drop rate %.2f (n=%d)" % drop_rate(tr))
print("shift(h)  drop rate (snapshot before vs after shifted SBL)")
for h in np.arange(-9, 9.5, 0.5):
    r, n = drop_rate(sn + np.timedelta64(int(h*60), "m"))
    print(f"{h:+5.1f}  {r:.2f}  n={n}  " + "#"*int(r*60))
# split before/after DST change 2026-03-29
for lab, msk in (("before 03-29", sn < np.datetime64("2026-03-29")), ("after 03-29", sn >= np.datetime64("2026-03-29"))):
    best = max(((drop_rate(sn[msk] + np.timedelta64(int(h*60), "m"))[0], h) for h in np.arange(-9, 9.5, 0.5)))
    print(lab, "n=", msk.sum(), "best shift", best[1], "rate", round(best[0],2), "| shift0 rate", round(drop_rate(sn[msk])[0],2))
