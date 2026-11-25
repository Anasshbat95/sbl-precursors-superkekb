"""
p_before_sbl.py - The formula

    P = (N_fe / N_sbls) * (1 / N_gesamt)

with the window BEFORE each SBL: [SBL - x, SBL), no buffer.

    N_fe      number of registered value CHANGES of the feature inside the windows of length x before the SBLs
              (all SBLs together)
    N_sbls    number of SBLs (128)
    N_gesamt  total number of changes of the feature in the whole block (default, n_gesamt="total"),
              or its change rate per hour of time WITH DATA (n_gesamt="rate")

Units: N_fe / N_sbls = changes per window, N_gesamt = changes per hour  ->  P is in HOURS.
If the feature changes before the SBLs as often as normally, P equals the window length x (in hours).
So:   P / x = 1  -> as often as normal,   P / x = 3 -> three times as often before the SBLs.
The plot shows P with a dashed line at x ("normal").

Options:
    count     "changes" (default) = rows where the value changed (works on forward-filled data)
              "values"            = non-null values (only for RAW, not forward-filled data)
    n_gesamt  "total" (default) = total number of changes of the feature in the whole block
              "rate"            = changes per hour of time with data (then P is in hours and P = x is "normal")

Usage:
    from p_before_sbl import compute_p_before_sbl, plot_p_before_sbl
    table = compute_p_before_sbl(sbl, blocks, window_seconds=500)
    plot_p_before_sbl(table, save_path="Abbildung/P_before_sbl_500s.png")
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_COLORS = {"BEAM": "steelblue", "BACKGROUND": "darkorange", "INJECTION": "seagreen",
                "VACUUM": "firebrick", "POSITION": "purple"}


def _ts_col(df):
    return "Timestamp" if "Timestamp" in df.columns else "timestamp"


def _sbl_series(sbl_times):
    if isinstance(sbl_times, pd.DataFrame):
        col = "timestamp" if "timestamp" in sbl_times.columns else ("Timestamp" if "Timestamp" in sbl_times.columns
                                                                     else sbl_times.columns[0])
        sbl_times = sbl_times[col]
    return pd.to_datetime(pd.Series(sbl_times)).dropna().sort_values().reset_index(drop=True)


def _hours_with_data(t_ns, max_gap):
    if len(t_ns) < 2:
        return 0.0
    return float(np.minimum(np.diff(t_ns), pd.Timedelta(max_gap).value).sum()) / 3.6e12


def compute_p_before_sbl(sbl_times, blocks: dict, window_seconds=500, count="changes", n_gesamt="total",
                         max_gap="1h"):
    """One row per feature: group, N_fe, N_sbls, changes_total, hours_with_data, N_gesamt, P, P_over_x."""
    if count not in ("values", "changes"):
        raise ValueError("count must be 'values' or 'changes'.")
    if n_gesamt not in ("rate", "total"):
        raise ValueError("n_gesamt must be 'rate' or 'total'.")
    s = _sbl_series(sbl_times).to_numpy(dtype="datetime64[ns]").astype("int64")
    n_sbls = len(s)
    w = int(window_seconds * 1e9)
    x_h = window_seconds / 3600
    rows = []
    for group, df in blocks.items():
        ts_col = _ts_col(df)
        d = df.sort_values(ts_col)
        t = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
        k = np.searchsorted(s, t, side="right")                  # first SBL strictly after the row
        nxt = np.where(k < n_sbls, s[np.clip(k, 0, n_sbls - 1)], np.iinfo(np.int64).max)
        inside = (k < n_sbls) & ((nxt - t) <= w)                 # row in [SBL - x, SBL)
        hours = _hours_with_data(t, max_gap)
        cols = [c for c in d.select_dtypes("number").columns if c not in ("timestamp", "Timestamp")]
        for c in cols:
            v = d[c]
            if count == "changes":
                hit = (v.ne(v.shift()) & v.notna()).to_numpy().copy()
                hit[0] = False
            else:
                hit = v.notna().to_numpy()
            n_fe, total = int(hit[inside].sum()), int(hit.sum())
            ng = total / hours if n_gesamt == "rate" and hours > 0 else float(total)
            p = (n_fe / n_sbls) * (1 / ng) if ng > 0 else np.nan
            rows.append(dict(group=group, feature=c, N_fe=n_fe, N_sbls=n_sbls, changes_total=total,
                             hours_with_data=round(hours, 1), N_gesamt=ng, P=p,
                             P_over_x=p / x_h if n_gesamt == "rate" else np.nan,
                             expected_N_fe=round(ng * x_h * n_sbls, 1) if n_gesamt == "rate" else np.nan))
    table = pd.DataFrame(rows)
    table.index = table["group"] + ":" + table["feature"]
    table.attrs.update(window_seconds=window_seconds, n_gesamt=n_gesamt, count=count)
    return table.sort_values("P", ascending=False)


def plot_p_before_sbl(table, title="P per feature", save_path=None, show=True):
    """Horizontal bars of P, one colour per dataset group. With n_gesamt="rate" a dashed line marks the
    window length x = the value of P for a feature that changes as often as normally."""
    t = table.dropna(subset=["P"]).sort_values("P")
    ws, mode = table.attrs.get("window_seconds"), table.attrs.get("n_gesamt", "rate")
    colors = [GROUP_COLORS.get(str(g).upper(), "gray") for g in t["group"]]
    fig, ax = plt.subplots(figsize=(10, 0.26 * len(t) + 2))
    vals = t["P"] * 60 if mode == "rate" else t["P"]          # hours -> minutes for readability
    ax.barh(t.index, vals, color=colors)
    if mode == "rate" and ws:
        ax.axvline(ws / 60, color="black", ls="--", lw=1.2, label=f"normal (= window {ws / 60:.1f} min)")
        for y, (v, r) in enumerate(zip(vals, t["P_over_x"])):
            ax.text(v, y, f"  x{r:.1f}", va="center", fontsize=6)
        ax.set_xlabel(f"P = (N_fe / N_sbls) * (1 / N_gesamt)   [minutes];   N_sbls = {int(t['N_sbls'].iat[0])},   "
                      "N_gesamt = changes per hour", fontsize=9)
    else:
        ax.set_xlabel(f"P = (N_fe / N_sbls) * (1 / N_gesamt);   N_sbls = {int(t['N_sbls'].iat[0])}")
    ax.set_ylabel("Feature")
    ax.set_title(f"{title}, window {ws} s before SBL" if ws else title, fontsize=12)
    ax.tick_params(axis="y", labelsize=7)
    present = {str(g).upper() for g in t["group"]}
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g in present]
    if mode == "rate" and ws:
        handles.append(plt.Line2D([], [], color="black", ls="--", label=f"normal (= window {ws / 60:.1f} min)"))
    ax.legend(handles=handles, loc="lower right", title="Dataset", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return fig, ax


def plot_p_multi_window(sbl_times, blocks: dict, windows=(60, 120, 500), per_minute=False, sort_by=None,
                        features=None, top=None, title="P per feature for several windows", save_path=None,
                        show=True, **kw):
    """One plot for several window lengths: per feature one bar per window (light = short, dark = long),
    colour = dataset group. Rows sorted by P of `sort_by` (default: longest window).
    per_minute=True divides P by the window length in minutes, so windows of different length are
    comparable (a longer window always collects more changes).
    windows:  any number of window lengths in seconds, e.g. (30, 60, 120, 300, 500, 1800).
    features: only these features, e.g. ["her_tune_x", "background:ler_oscillator_freq"] (with or without
              the "group:" prefix).  top: only the `top` features with the largest P in the sort window.
    Returns (fig, table with one column per window)."""
    windows = list(windows)
    cols, group = {}, {}
    for w in windows:
        t = compute_p_before_sbl(sbl_times, blocks, window_seconds=w, **kw)
        cols[f"{w}s"] = t["P"] / (w / 60) if per_minute else t["P"]
        group.update(t["group"].to_dict())
    tab = pd.DataFrame(cols)
    if features:
        want = set(features)
        keep = [i for i in tab.index if i in want or i.split(":", 1)[1] in want]
        missing = [f for f in features if f not in keep and not any(k.split(":", 1)[1] == f for k in keep)]
        if missing:
            print(f"  not found: {missing}")
        tab = tab.loc[keep]
    key = f"{sort_by or max(windows)}s"
    tab = tab.sort_values(key, ascending=False)
    if top:
        tab = tab.head(top)
    tab = tab.iloc[::-1]
    n_w = len(windows)
    h = 0.8 / n_w
    alphas = np.linspace(0.35, 1.0, n_w)
    y = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(10, (0.14 + 0.08 * n_w) * len(tab) + 2.2))
    for k, c in enumerate(tab.columns):
        colors = [GROUP_COLORS.get(str(group[f]).upper(), "gray") for f in tab.index]
        ax.barh(y + (k - (n_w - 1) / 2) * h, tab[c].fillna(0), h, color=colors, alpha=alphas[k],
                edgecolor="black", lw=0.2)
    ax.set_yticks(y)
    ax.set_yticklabels(tab.index, fontsize=7)
    ax.set_ylim(-0.6, len(tab) - 0.4)
    ax.set_xlabel("P / window length in minutes" if per_minute else "P = (N_fe / N_sbls) * (1 / N_gesamt)")
    ax.set_title(f"{title}  (sorted by {key})", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    present = [g for g in GROUP_COLORS if g in {str(v).upper() for v in group.values()}]
    h1 = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in present]
    h2 = [mpatches.Patch(facecolor="gray", alpha=a, edgecolor="black", lw=0.3, label=f"window {c}")
          for a, c in zip(alphas, tab.columns)][::-1]
    leg1 = ax.legend(handles=h1, loc="lower right", title="Dataset", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="center right", title="per feature, top to bottom", fontsize=8, title_fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return fig, tab.iloc[::-1]
