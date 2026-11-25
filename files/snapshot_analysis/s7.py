from lib import *
top = [("background","her_tune_x","change"),("background","dispersion_correction","change"),
       ("background","mag_current_z_corr","change"),("background","det_luminosity","change"),
       ("background","det_trigger_rate_bw","change"),("background","her_tune_y","change"),("beam","her_current","level")]
PB = {b: prep_block(b) for b in {x[0] for x in top}}
rows = []
# random pseudo-events: beam-on times > 6h from any SBL, same count
rng = np.random.default_rng(7)
bt_on = BT[(BC > 50)]
far = bt_on[np.abs(((bt_on[:, None] - SN[None, :]).astype("int64")/H)).min(1) > 8]
for b, f, kind in top:
    P = PB[b]; k = P["feats"].index(f)
    for kth in (1, 2, 3, 4):
        rows.append(dict(feature=f, kind=kind, test=f"k={kth}", **summarize(matched_pcts(P, k, kind, SN, kth))))
    rows.append(dict(feature=f, kind=kind, test="age>=0.5h", **summarize(matched_pcts(P, k, kind, SN, 1, 0.5))))
    neg = [summarize(matched_pcts(P, k, kind, np.sort(rng.choice(far, 128, replace=False)) + np.timedelta64(rng.integers(0,120),"m"), 1))["mean_pct"] for _ in range(20)]
    rows.append(dict(feature=f, kind=kind, test="random events (20x)", n=np.nan, mean_pct=np.mean(neg), lo=np.quantile(neg,.025), hi=np.quantile(neg,.975), p=np.nan))
res = pd.DataFrame(rows); res.to_csv(OUT+"robustness.csv", index=False)
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))
