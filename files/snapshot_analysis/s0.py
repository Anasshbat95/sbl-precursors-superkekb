from common import *
blocks, sbl = load()
print(type(sbl), sbl.head(2), len(sbl))
for n, d in blocks.items():
    ts = pd.to_datetime(d["timestamp"])
    print(n, d.shape, ts.min(), ts.max(), "median dt s:", ts.sort_values().diff().dt.total_seconds().median(),
          "NaN share:", round(d.isna().mean().mean(), 3))
    print("   ", [c for c in d.columns if c != "timestamp"])
print(blocks["beam"][["her_current","ler_current"]].describe().round(1))
