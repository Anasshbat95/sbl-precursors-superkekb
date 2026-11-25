from common import *
blocks, sbl = load()
b = blocks["beam"].sort_values("timestamp")
ts = pd.to_datetime(b["timestamp"])
gap = ts.diff().dt.total_seconds()
# bursts = rows closer than 60 s
burst = (gap > 60).cumsum()
g = b.assign(burst=burst.values).groupby("burst")
bs = pd.DataFrame({"start": g["timestamp"].min(), "end": g["timestamp"].max(), "n": g.size()})
print("bursts:", len(bs), "rows/burst median", bs.n.median(), "burst length s median", (bs.end-bs.start).dt.total_seconds().median())
gb = bs.start.diff().dt.total_seconds()/60
print("gap between bursts (min) quantiles:", gb.quantile([.1,.25,.5,.75,.9]).round(1).to_dict())
print("burst start minute-of-hour counts (top):", bs.start.dt.minute.value_counts().head(8).to_dict())
print("burst start second counts (top):", bs.start.dt.second.value_counts().head(5).to_dict())
print(bs.head(15))
s = sbl["timestamp"]
sn = s.to_numpy("datetime64[ns]")
bn = np.sort(bs.start.to_numpy("datetime64[ns]"))
print("\nshift(h) | median |dist| SBL->nearest burst (min) | share within 2 min")
for h in range(-10, 11):
    t = sn + np.timedelta64(h*3600, "s")
    i = np.clip(np.searchsorted(bn, t), 1, len(bn)-1)
    d = np.minimum(np.abs(bn[i]-t), np.abs(bn[i-1]-t)).astype("int64")/6e10
    print(f"{h:+3d} {np.median(d):8.1f} {np.mean(d<2):6.0%}")
