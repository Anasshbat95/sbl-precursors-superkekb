from common import *
from operation_state import operation_intervals, summary, label_states
from current_around_sbl import current_around_sbl, summarize_current_events, check_time_offset
blocks, sbl = load()
s = sbl["timestamp"]
sn = np.sort(s.to_numpy("datetime64[ns]"))
print("== rows before SBL per block (share of 128 SBLs with >=2 rows in window)")
for n, d in blocks.items():
    ts = np.sort(pd.to_datetime(d["timestamp"]).to_numpy("datetime64[ns]"))
    row = {}
    for w in ["1min","5min","10min","30min","1h"]:
        wn = np.timedelta64(pd.Timedelta(w).value, "ns")
        cnt = np.searchsorted(ts, sn) - np.searchsorted(ts, sn - wn)
        row[w] = f"{(cnt>=2).mean():.0%}"
    gaps = np.diff(ts).astype("int64")/1e9
    row["rows/h"] = round(len(ts)/((ts[-1]-ts[0]).astype('int64')/3.6e12),1)
    row["p90 gap min"] = round(np.percentile(gaps,90)/60,1)
    print(n, row)
print("\n== operation states thr=50")
iv = operation_intervals(blocks["beam"], on_thr=50, shutdown_min="12h")
print(summary(iv, s))
for thr in (50, 400):
    print(f"\n== current_around_sbl thr={thr}")
    tab = current_around_sbl(blocks["beam"], "timestamp", s, thr=thr)
    summarize_current_events(tab)
    tab.to_csv(OUT+f"current_tab_{thr}.csv", index=False)
print("\n== check_time_offset thr=400")
off = check_time_offset(blocks["beam"], "timestamp", s, thr=400)
off.to_csv(OUT+"offset.csv", index=False)
print("\n== fine offset")
check_time_offset(blocks["beam"], "timestamp", s, thr=400, shifts=[pd.Timedelta(minutes=m) for m in range(-10, 11, 2)])
