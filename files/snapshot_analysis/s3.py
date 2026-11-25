from common import *
blocks, sbl = load()
for n, d in blocks.items():
    ts = pd.to_datetime(d["timestamp"]).sort_values()
    snap = (ts.dt.minute >= 58) & (ts.dt.hour % 2 == 1)
    off = ts[~snap]
    print(f"{n:10s} rows {len(ts):6d} | in 2h-snapshot: {snap.mean():.1%} | other rows: {len(off)}")
    if len(off):
        gaps = off.diff().dt.total_seconds()
        bursts = (gaps > 60).cumsum()
        print("    other rows form", bursts.nunique(), "bursts; first:", off.iloc[:3].dt.strftime('%m-%d %H:%M:%S').tolist())
        # are the extra rows near SBLs?
        sn = np.sort(sbl.timestamp.to_numpy("datetime64[ns]"))
        on = off.to_numpy("datetime64[ns]")
        i = np.clip(np.searchsorted(sn, on), 1, len(sn)-1)
        dmin = np.minimum(np.abs(sn[i]-on), np.abs(sn[i-1]-on)).astype("int64")/6e10
        print("    extra rows within 30 min of an SBL:", f"{(dmin<30).mean():.0%}", " median dist min:", round(np.median(dmin),1))
