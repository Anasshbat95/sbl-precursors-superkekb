from lib import *
exec(open(os.path.dirname(os.path.abspath(__file__))+"/s8.py").read().split("rows, store = [], {}")[0].split("from lib import *")[1])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})
COL = {"beam": "#2a78d6", "background": "#eb6834", "injection": "#1baf7a", "vacuum": "#eda100", "position": "#e87ba4"}
INK, MUTED = "#0b0b0b", "#8a8984"
def save(fig, name):
    fig.savefig(FIG + name, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", name)

# ---- Fig 1: sampling structure
b = blocks["beam"].sort_values("timestamp"); ts = pd.to_datetime(b.timestamp)
starts = ts[ts.diff().dt.total_seconds().fillna(1e9) > 60]
gaps = starts.diff().dt.total_seconds().dropna()/60
age = (SN - BT[np.clip(np.searchsorted(BT, SN)-1, 0, None)]).astype("int64")/6e10
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].hist(np.clip(gaps, 0, 300), bins=np.arange(0, 305, 5), color=COL["beam"], edgecolor="white", linewidth=0.8)
ax[0].set_xlabel("time between two bursts of rows (min, clipped at 300)"); ax[0].set_ylabel("number of gaps")
ax[0].set_title("(a) BEAM block: rows arrive every 2 h", loc="left", fontsize=10)
ax[1].hist(age, bins=np.arange(0, 125, 5), color=COL["beam"], edgecolor="white", linewidth=0.8)
ax[1].axvline(np.median(age), color=INK, lw=1.2, ls="--"); ax[1].text(np.median(age)+2, ax[1].get_ylim()[1]*0.9, f"median {np.median(age):.0f} min", color=INK)
ax[1].set_xlabel("age of the last BEAM row before an SBL (min)"); ax[1].set_ylabel("number of SBLs")
ax[1].set_title("(b) Last information before each SBL", loc="left", fontsize=10)
save(fig, "ch3_sampling_structure.png")

# ---- Fig 2: clock check
cur = beam_s[["her_current","ler_current"]].to_numpy()
def drop_rate(times):
    i = np.searchsorted(BT, times) - 1; ok = (i >= 0) & (i+1 < len(BT)); i = i[ok]
    i = i[((BT[i+1]-BT[i]).astype("int64")/H) < 2.5]; bb, aa = cur[i], cur[i+1]
    on = bb.max(1) > 400; return (on & ((aa < 0.5*bb) & (bb > 400)).any(1)).sum()/max(on.sum(), 1)
shifts = np.arange(-9, 9.5, 0.5); rates = [drop_rate(SN + np.timedelta64(int(h*60), "m")) for h in shifts]
rr = np.random.default_rng(0); rand = [drop_rate(np.sort(rr.choice(cand_t, 128))) for _ in range(300)]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.axhspan(np.quantile(rand, .025), np.quantile(rand, .975), color="#e4e3df", label="random times (95% band)")
ax.plot(shifts, rates, color=COL["beam"], lw=2, marker="o", ms=4, label="SBL list shifted")
ax.axvline(0, color=INK, lw=0.8, ls=":")
ax.set_xlabel("shift added to the SBL timestamps (h)"); ax.set_ylabel("share of events with current drop")
ax.legend(frameon=False, loc="upper left"); save(fig, "ch3_clock_check.png")

# ---- Fig 3 + 4: permutation ranking, lag profile
r1 = pd.read_csv(OUT + "perm_ranking.csv").head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.4))
y = np.arange(len(r1))
ax.hlines(y, r1.null_lo, r1.null_hi, color="#c9c8c2", lw=6, label="random events (95% band)")
ax.scatter(r1.null_median, y, color=MUTED, s=12, zorder=3)
for yy, (_, r) in zip(y, r1.iterrows()):
    ax.scatter(r.mean_pct, yy, color=COL[r.block], s=55, zorder=4, edgecolor="white", linewidth=1.5)
ax.set_yticks(y, [f"{r.feature} ({'Δ' if r.kind=='change' else 'level'})" for _, r in r1.iterrows()])
ax.set_xlabel("mean matched percentile of the last snapshot before the SBL")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=c, label=n.upper(), markersize=7) for n, c in COL.items()]
ax.legend(handles=h + [Line2D([], [], color="#c9c8c2", lw=6, label="random events, 95%")], frameon=False,
          loc="lower right", fontsize=8)
save(fig, "ch5_ranking.png")

cands = [("beam","her_current","level"), ("position","ler_mag_mqc2lp_pos_y","level"),
         ("beam","ler_sigma_y","level"), ("background","dispersion_correction","level")]
prof = []
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for ax, (bl, f, kind) in zip(axs, cands):
    P = prep_block(bl); ref, since = ref_mask(P, SN); PCT = snap_pct(P, kind, ref); k = P["feats"].index(f)
    nulls = [np.nanmean(PCT[pick(P, np.sort(rng.choice(cand_t, 128, replace=False)), since)][:, k]) for _ in range(300)]
    lo, hi, md = np.quantile(nulls, [.025, .975, .5])
    vals = []
    for kth in (1, 2, 3, 4):
        x = PCT[pick(P, SN, since, kth)][:, k]; x = x[np.isfinite(x)]
        bs = np.random.default_rng(2).choice(x, (2000, len(x))).mean(1)
        vals.append((x.mean(), *np.quantile(bs, [.025, .975])))
        prof.append(dict(feature=f, kth=kth, mean=x.mean(), null_lo=lo, null_hi=hi, null_med=md, n=len(x)))
    v = np.array(vals); xs = -np.array([1, 3, 5, 7])
    ax.axhspan(lo, hi, color="#e4e3df"); ax.axhline(md, color=MUTED, lw=0.8)
    ax.errorbar(xs, v[:, 0], yerr=[v[:, 0]-v[:, 1], v[:, 2]-v[:, 0]], color=COL[bl], marker="o", ms=6, lw=2, capsize=0)
    ax.set_title(f, fontsize=9); ax.set_xticks(xs, ["0–2", "2–4", "4–6", "6–8"]); ax.set_xlabel("hours before SBL")
axs[0].set_ylabel("mean matched percentile")
save(fig, "ch5_lag_profile.png")
pd.DataFrame(prof).to_csv(OUT + "lag_profile.csv", index=False); print(pd.DataFrame(prof).round(3).to_string(index=False))
print("rates at 0:", round(rates[list(shifts).index(0)], 2), "random band", np.quantile(rand, [.025, .5, .975]).round(2))
print("gap quantiles", gaps.quantile([.5,.9]).round(1).to_dict(), "age median/p25/p75", np.percentile(age, [25,50,75]).round(0))
