"""
risk_model.py - XGBoost risk model on the 2-hour snapshots: "does an SBL follow in the next 2 h?"

The data are snapshots every 2 h (bursts ending at a full hour; odd hours before, even hours after the switch to summer time), so a warning model for the next
seconds is not possible. Instead, every snapshot is one sample:
    X = state of all five blocks at the snapshot (last row of each block at or before t, never after t)
        + change since the previous snapshot + optional reporting activity of chosen channels in the last hour
    y = 1 if an SBL occurs in (t, t + horizon]
Only snapshots with beam on (HER or LER above ON_THR) are used.

Evaluation never mixes time: leave-one-month-out (every month is test once, the model is trained on the
other months) and, in addition, a chronological split (first 60 % of the time train, rest test).
Every model is compared with a baseline that uses only the beam currents.

Usage (notebook):
    from files.risk_model import build_table, evaluate, plot_curves, plot_importance, plot_risk_time
    tab = build_table(blocks, sbl_times, raw_blocks=raw)          # raw_blocks optional (reporting counts)
    res = evaluate(tab)                                           # dict with summary table + predictions
    res["summary"]
    plot_curves(res, save="risk_roc_pr.png")
    plot_importance(res, save="risk_importance.png")
    plot_risk_time(res, sbl_times, save="risk_time.png")

Every plot function accepts save="name.png" to write the figure into MasterArbeit/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from files.snapshot_analysis import COL, _sbl_ns, _save, snapshots

CONFIG = dict(
    HORIZON_H=2,             # label: SBL within this many hours after the snapshot
    ON_THR=400.0,            # beam on: HER or LER current above this value [mA]
    MAX_AGE_H=2.5,           # a block value older than this at the snapshot counts as missing
    GRID_OFFSET_S=5,         # grid at full hours + 5 s, i.e. just after a burst ending at the full hour
    MAX_SNAP_AGE_MIN=10,     # keep only grid times with a BEAM snapshot at most this old (the regular bursts)
    ACTIVITY=(("background", "ler_oscillator_freq"), ("background", "her_oscillator_freq")),
    ACTIVITY_WINDOW_H=1,     # reporting counts in the last hour before the snapshot
    CHRONO_TRAIN_FRAC=0.6,
    N_BOOT=1000,
    RANDOM_STATE=0,
)
BASELINE = ["beam__her_current", "beam__ler_current"]


# ============================================================================ 1. table
def _grid(blocks, cfg):
    t0 = min(pd.to_datetime(d["timestamp"]).min() for d in blocks.values()).floor("h")
    t1 = max(pd.to_datetime(d["timestamp"]).max() for d in blocks.values()).ceil("h")
    g = pd.date_range(t0, t1, freq="1h") + pd.Timedelta(seconds=cfg["GRID_OFFSET_S"])
    # the bursts end at the odd full hours until the switch to summer time (29.03.2026) and at the even ones
    # after it, so the grid is not fixed to odd/even hours but follows the BEAM snapshots
    tb = np.sort(snapshots(blocks["beam"])["timestamp"].to_numpy("datetime64[ns]"))
    tg = g.to_numpy("datetime64[ns]")
    i = np.searchsorted(tb, tg, side="right") - 1
    age = np.where(i >= 0, (tg - tb[np.clip(i, 0, None)]) / np.timedelta64(1, "m"), np.inf)
    return g[age <= cfg["MAX_SNAP_AGE_MIN"]]


def _asof(block, grid, cfg, name):
    """Last snapshot state of the block at or before every grid time (+ change to the previous grid time)."""
    s = snapshots(block).sort_values("timestamp")
    feats = [c for c in s.columns if c != "timestamp"]
    s = s.rename(columns={c: f"{name}__{c}" for c in feats})
    g = pd.DataFrame({"t": grid})
    m = pd.merge_asof(g, s.rename(columns={"timestamp": "t_row"}), left_on="t", right_on="t_row",
                      direction="backward")
    age_h = (m["t"] - m["t_row"]).dt.total_seconds() / 3600
    cols = [f"{name}__{c}" for c in feats]
    m.loc[age_h > cfg["MAX_AGE_H"], cols] = np.nan
    out = m[cols].copy()
    for c in cols:
        out[f"{c}__diff"] = m[c].diff()
    out[f"{name}__age_h"] = age_h.where(age_h <= cfg["MAX_AGE_H"])
    return out


def build_table(blocks, sbl_times, raw_blocks=None, **overrides):
    """One row per 2-h snapshot with beam on: features, label y and the time t."""
    cfg = {**CONFIG, **overrides}
    sbl = _sbl_ns(sbl_times)
    grid = _grid(blocks, cfg)
    parts = [_asof(d, grid, cfg, name) for name, d in blocks.items()]
    X = pd.concat(parts, axis=1)
    X.insert(0, "t", grid)

    if raw_blocks is not None:                       # reporting activity: rows with a new value in the last hour
        w = np.timedelta64(int(cfg["ACTIVITY_WINDOW_H"] * 3600), "s")
        tg = grid.to_numpy("datetime64[ns]")
        for b, f in cfg["ACTIVITY"]:
            d = raw_blocks[b]
            tc = np.sort(d["timestamp"].to_numpy("datetime64[ns]")[d[f].notna().to_numpy()])
            X[f"{b}__{f}__reports_1h"] = np.searchsorted(tc, tg, side="right") - np.searchsorted(tc, tg - w)

    tg = grid.to_numpy("datetime64[ns]")
    j = np.searchsorted(sbl, tg, side="right")       # first SBL strictly after t
    nxt = np.where(j < len(sbl), (sbl[np.minimum(j, len(sbl) - 1)] - tg) / np.timedelta64(1, "h"), np.inf)
    prev = np.where(j > 0, (tg - sbl[np.maximum(j - 1, 0)]) / np.timedelta64(1, "h"), np.nan)
    X["h_since_last_sbl"] = prev
    X["y"] = (nxt <= cfg["HORIZON_H"]).astype(int)

    cur = X[BASELINE].max(axis=1)
    X = X[cur > cfg["ON_THR"]].reset_index(drop=True)
    X = X.loc[:, X.notna().any()]                    # drop all-empty columns
    print(f"{len(X)} snapshots with beam on, {X['y'].sum()} followed by an SBL within "
          f"{cfg['HORIZON_H']} h ({X['y'].mean():.1%}); {X.shape[1] - 2} features")
    return X


# ============================================================================ 2. models
def _xgb(y_train, cfg):
    from xgboost import XGBClassifier
    pos = max(int(y_train.sum()), 1)
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                         scale_pos_weight=(len(y_train) - pos) / pos, eval_metric="logloss",
                         random_state=cfg["RANDOM_STATE"], n_jobs=4)


def _logreg():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def _models(tab):
    allf = [c for c in tab.columns if c not in ("t", "y")]
    nosbl = [c for c in allf if c != "h_since_last_sbl"]
    return {
        "baseline: currents (logreg)": (_logreg, BASELINE),
        "logreg: all features": (_logreg, nosbl),
        "xgboost: all features": ("xgb", nosbl),
        "xgboost: all + time since last SBL": ("xgb", allf),
    }


def _fit_predict(kind, feats, train, test, cfg):
    m = _xgb(train["y"], cfg) if kind == "xgb" else kind()
    m.fit(train[feats], train["y"])
    return m.predict_proba(test[feats])[:, 1], m


def _boot_ci(y, p, metric, n, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    return np.quantile(vals, [0.025, 0.975])


def evaluate(tab, **overrides):
    """Leave-one-month-out and chronological evaluation of all models.
    Returns dict(summary, pred_lomo, pred_chrono, importance, cfg)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    cfg = {**CONFIG, **overrides}
    month = tab["t"].dt.to_period("M")
    pred_lomo = tab[["t", "y"]].copy()
    cut = tab["t"].quantile(cfg["CHRONO_TRAIN_FRAC"])
    tr_c, te_c = tab[tab["t"] <= cut], tab[tab["t"] > cut]
    pred_chrono = te_c[["t", "y"]].copy()
    rows, importance = [], {}

    for name, (kind, feats) in _models(tab).items():
        p = np.full(len(tab), np.nan)
        for mth in month.unique():
            te = (month == mth).to_numpy()
            if tab.loc[~te, "y"].sum() < 5:
                continue
            p[te], _ = _fit_predict(kind, feats, tab[~te], tab[te], cfg)
        pred_lomo[name] = p
        pc, m = _fit_predict(kind, feats, tr_c, te_c, cfg)
        pred_chrono[name] = pc
        if kind == "xgb":
            importance[name] = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(feats).fillna(0)

        for split, d in (("leave-one-month-out", pred_lomo), ("chronological", pred_chrono)):
            ok = d[name].notna().to_numpy()
            y, s = d["y"].to_numpy()[ok], d[name].to_numpy()[ok]
            auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
            lo, hi = _boot_ci(y, s, roc_auc_score, cfg["N_BOOT"], cfg["RANDOM_STATE"])
            rows.append(dict(split=split, model=name, n=len(y), n_sbl=int(y.sum()), roc_auc=round(auc, 3),
                             roc_auc_ci95=f"{lo:.2f}-{hi:.2f}", pr_auc=round(ap, 3),
                             pr_auc_chance=round(y.mean(), 3)))
    summary = pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)
    return dict(summary=summary, pred_lomo=pred_lomo, pred_chrono=pred_chrono, importance=importance, cfg=cfg)


# ============================================================================ 3. plots
_MCOL = {"baseline: currents (logreg)": "#8a8984", "logreg: all features": "#2a78d6",
         "xgboost: all features": "#eb6834", "xgboost: all + time since last SBL": "#1baf7a"}


def plot_curves(res, split="leave-one-month-out", save=None):
    """ROC (left) and precision-recall (right) curves of all models for one evaluation split."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    d = res["pred_lomo"] if split == "leave-one-month-out" else res["pred_chrono"]
    summ = res["summary"][res["summary"]["split"] == split].set_index("model")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in [c for c in d.columns if c not in ("t", "y")]:
        ok = d[name].notna()
        y, s = d.loc[ok, "y"], d.loc[ok, name]
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        lab = f"{name} (AUC {summ.loc[name, 'roc_auc']:.2f})"
        a1.plot(fpr, tpr, color=_MCOL.get(name), lw=1.8, label=lab)
        a2.plot(rc, pr, color=_MCOL.get(name), lw=1.8, label=f"{name} (AP {summ.loc[name, 'pr_auc']:.2f})")
    a1.plot([0, 1], [0, 1], "k:", lw=0.8)
    base = d["y"].mean()
    a2.axhline(base, color="k", ls=":", lw=0.8, label=f"chance ({base:.2f})")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC ({split})")
    a2.set(xlabel="recall", ylabel="precision", title=f"precision-recall ({split})", ylim=(0, 1))
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend(frameon=False, fontsize=7.5)
    plt.tight_layout(); _save(fig, save); plt.show()


def plot_importance(res, model="xgboost: all features", top=15, save=None):
    """Gain importance of the XGBoost model (chronological training set), coloured by block."""
    imp = res["importance"][model].sort_values(ascending=False).head(top)[::-1]
    imp = imp / imp.sum()
    cols = [COL.get(i.split("__")[0], "#8a8984") for i in imp.index]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh([i.replace("__", ": ", 1).replace("__", " ") for i in imp.index], imp.values, color=cols)
    ax.set(xlabel="share of total gain", title=f"feature importance - {model}")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout(); _save(fig, save); plt.show()


def plot_risk_time(res, sbl_times, model="baseline: currents (logreg)", compare="xgboost: all features",
                   save=None):
    """Predicted risk (leave-one-month-out) of `model` (coloured) and `compare` (grey) against time,
    with the SBLs as red lines."""
    d = res["pred_lomo"]
    sbl = pd.to_datetime(pd.Series(_sbl_ns(sbl_times)))
    fig, ax = plt.subplots(figsize=(12, 3.4))
    if compare:
        ax.plot(d["t"], d[compare], ".", ms=2, color="#8a8984", alpha=0.6, label=compare)
    ax.plot(d["t"], d[model], ".", ms=2.5, color="#1f3b73", label=model)
    for s in sbl:
        ax.axvline(s, color="red", lw=0.5, alpha=0.6)
    ax.plot([], [], color="red", lw=0.8, label="SBL")
    ax.set(ylabel="P(SBL within 2 h)", ylim=(0, 1))
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout(); _save(fig, save); plt.show()


def plot_class_distribution(tab, sbl_times, save=None):
    """Snapshots per month split by class (bars) and share of y = 1 (line), with the number of SBLs."""
    month = tab["t"].dt.to_period("M")
    m = tab.groupby(month)["y"].agg(n="size", y1="sum")
    m["y0"] = m["n"] - m["y1"]
    s = pd.to_datetime(pd.Series(_sbl_ns(sbl_times)))
    m["sbl"] = s.groupby(s.dt.to_period("M")).size().reindex(m.index).fillna(0).astype(int)
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, m["y0"], color="#c9c8c3", label="y = 0 (no SBL within 2 h)")
    ax.bar(x, m["y1"], bottom=m["y0"], color="#eb6834", label="y = 1 (SBL within 2 h)")
    for i, r in enumerate(m.itertuples()):
        ax.text(i, r.n + 4, f"{r.y1}/{r.n}\n({r.sbl} SBLs)", ha="center", va="bottom", fontsize=7.5)
    ax.set(xticks=x, xticklabels=[str(p) for p in m.index], ylabel="snapshots with beam on",
           ylim=(0, m["n"].max() * 1.25))
    a2 = ax.twinx()
    a2.plot(x, 100 * m["y1"] / m["n"], "o-", color="#1f3b73", lw=1.5, label="share y = 1")
    a2.axhline(100 * tab["y"].mean(), color="#1f3b73", ls=":", lw=0.8)
    a2.set(ylabel="share y = 1 [%]", ylim=(0, max(20, 130 * (m["y1"] / m["n"]).max())))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"class distribution: {tab['y'].sum()} of {len(tab)} snapshots with y = 1 "
                 f"({tab['y'].mean():.1%})")
    plt.tight_layout(); _save(fig, save); plt.show()


def plot_merge_example(blocks, sbl_times, start, hours=8, **overrides):
    """How one table row is built: rows of every block (dots), grid times (vertical lines) and, for every
    grid time, the row taken by merge_asof (arrow back to the last row at or before t). SBLs in red."""
    save = overrides.pop("save", None)
    cfg = {**CONFIG, **overrides}
    t0 = pd.Timestamp(start)
    t1 = t0 + pd.Timedelta(hours=hours)
    grid = _grid(blocks, cfg)
    grid = grid[(grid >= t0) & (grid <= t1)]
    sbl = pd.to_datetime(pd.Series(_sbl_ns(sbl_times)))
    sbl = sbl[(sbl >= t0) & (sbl <= t1 + pd.Timedelta(hours=cfg["HORIZON_H"]))]
    names = list(blocks)
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(names) + 1.6))
    for k, name in enumerate(names):
        snap = snapshots(blocks[name])["timestamp"]
        lo = t0 - pd.Timedelta(hours=cfg["MAX_AGE_H"])
        snap = snap[(snap >= lo) & (snap <= t1)]
        ax.scatter(snap, np.full(len(snap), k), s=14, color=COL.get(name, "C0"), zorder=3)
        for g in grid:
            prev = snap[snap <= g]
            if len(prev):
                tr = prev.max()
                ok = (g - tr) <= pd.Timedelta(hours=cfg["MAX_AGE_H"])
                ax.annotate("", xy=(tr, k), xytext=(g, k),
                            arrowprops=dict(arrowstyle="->", color="k" if ok else "#c9c8c3", lw=0.9))
    for g in grid:
        ax.axvline(g, color="#1f3b73", lw=0.8, ls="--")
        y = int(((sbl > g) & (sbl <= g + pd.Timedelta(hours=cfg["HORIZON_H"]))).any())
        ax.text(g, len(names) - 0.4, f"t = {g:%H:%M}\ny = {y}", ha="center", va="bottom", fontsize=8,
                color="#eb6834" if y else "#1f3b73")
    for s in sbl:
        ax.axvline(s, color="red", lw=1.5)
        ax.text(s, -0.8, "SBL", color="red", ha="center", fontsize=8)
    ax.set(yticks=range(len(names)), yticklabels=[n.upper() for n in names], ylim=(-1, len(names) + 0.5),
           xlim=(t0 - pd.Timedelta(hours=cfg["MAX_AGE_H"]), t1 + pd.Timedelta(minutes=20)))
    ax.set_title("dots: snapshot of a block; dashed: table time t; arrow: value taken for t "
                 "(last snapshot at or before t)", fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout(); _save(fig, save); plt.show()
