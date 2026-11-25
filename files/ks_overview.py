"""
ks_overview.py - One number per feature instead of one ECDF plot per feature.

For every feature of every block and every window length W, the row-to-row change |dv| in the window
before the SBLs is compared with a reference by the KS distance D (and the mean ratio). The result is
one table and one heatmap (features x windows), from which the few features worth an ECDF plot are chosen.

Choices (all as discussed for the thesis, section "Distribution comparison"):
    * |dv| is computed on the FULL block, before any filtering, so no |dv| jumps over a removed gap.
    * window:    buffer < tau <= W   (tau = time to the next SBL; buffer="0s" -> 0 <= tau <= W)
    * reference: "normal" = rows farther than normal_margin from every SBL (default), or "all" rows.
    * iv (optional, from operation_state.operation_intervals): the REFERENCE is restricted to BEAM_ON;
      the window rows are kept whatever the state (an SBL needs stored beam).
    * p-values are descriptive only (correlated rows, ties at zero, many tests).

Usage:
    from ks_overview import ks_overview, plot_ks_overview
    blocks = {"beam": ds_beam, "background": ds_background, "injection": ds_injection,
              "vacuum": ds_vacuum, "position": ds_position}
    tab = ks_overview(blocks, sbl_times["timestamp"], windows=["1min", "5min", "30min", "1h"],
                      reference="normal", normal_margin="2h", buffer="30s", iv=iv)
    fig = plot_ks_overview(tab, save_as="Abbildung/ks_overview.png")
    tab[tab["window"] == "30min"].sort_values("D", ascending=False).head(10)   # candidates for ECDF plots
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from files.diff_distribution_before_sbl import _dist_to_nearest_sbl, time_before_next_sbl


def ks_overview(blocks, sbl_times, ts_col="timestamp", windows=("1min", "5min", "30min", "1h"),
                reference="normal", normal_margin="2h", buffer="0s", iv=None, min_n=10, feature_cols=None):
    """Table with one row per (block, feature, window): n_window, n_ref, D, p_value, ratio_mean,
    frac_zero_window, frac_zero_ref, d_frac_zero (= frac_zero_ref - frac_zero_window)."""
    if reference not in ("all", "normal"):
        raise ValueError("reference must be 'all' or 'normal'.")
    if reference == "normal" and pd.Timedelta(normal_margin) < max(pd.Timedelta(w) for w in windows):
        print("WARNING: normal_margin is shorter than the longest window - window and reference rows can overlap.")
    b_s = pd.Timedelta(buffer).total_seconds()
    m_s = pd.Timedelta(normal_margin).total_seconds()
    rows = []
    for name, df in blocks.items():
        d = df.sort_values(ts_col).reset_index(drop=True)
        cols = (feature_cols or {}).get(name) or [c for c in d.select_dtypes("number").columns if c != ts_col]
        tau = time_before_next_sbl(d, ts_col, sbl_times)
        dist = _dist_to_nearest_sbl(d, ts_col, sbl_times)
        ref_mask = np.ones(len(d), bool) if reference == "all" else dist > m_s
        if iv is not None:
            from operation_state import label_states
            ref_mask &= (label_states(d[[ts_col]], iv, ts_col)["state"] == "BEAM_ON").to_numpy()
        for w in windows:
            w_s = pd.Timedelta(w).total_seconds()
            win_mask = (tau <= w_s) & ((tau > b_s) if b_s > 0 else (tau >= 0))   # NaN tau -> False
            for c in cols:
                dv = d[c].diff().abs()
                a = dv[win_mask].dropna().to_numpy()
                r = dv[ref_mask].dropna().to_numpy()
                row = dict(block=name, feature=c, window=w, window_s=w_s, n_window=len(a), n_ref=len(r))
                if len(a) > min_n and len(r) > min_n:
                    D, p = ks_2samp(a, r)
                    fz_a, fz_r = float((a == 0).mean()), float((r == 0).mean())
                    row.update(D=D, p_value=p, ratio_mean=(a.mean() / r.mean()) if r.mean() else np.nan,
                               frac_zero_window=fz_a, frac_zero_ref=fz_r, d_frac_zero=fz_r - fz_a)
                rows.append(row)
    tab = pd.DataFrame(rows)
    for c in ("D", "p_value", "ratio_mean", "frac_zero_window", "frac_zero_ref", "d_frac_zero"):
        if c not in tab:
            tab[c] = np.nan
    return tab


def plot_ks_overview(tab, value="D", save_as=None, dpi=300, title=None):
    """Heatmap: rows = features (grouped by block, sorted by their largest value), columns = windows.
    Blank cells: too few rows in the window. The number in each cell is the value (D by default)."""
    import matplotlib.pyplot as plt
    t = tab.copy()
    t["label"] = t["block"] + ": " + t["feature"]
    piv = t.pivot_table(index="label", columns="window_s", values=value, aggfunc="first")
    blocks_of = t.drop_duplicates("label").set_index("label")["block"]
    order = (piv.max(axis=1).rename("mx").to_frame().join(blocks_of)
             .sort_values(["block", "mx"], ascending=[True, False]).index)
    piv = piv.loc[order]
    labels = {s: w for s, w in t.drop_duplicates("window_s")[["window_s", "window"]].to_numpy()}

    fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 4.5, 0.26 * len(piv) + 1.5), dpi=dpi, layout="constrained")
    vmax = np.nanmax(piv.to_numpy()) if np.isfinite(piv.to_numpy()).any() else 1
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 * vmax else "black")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([labels[s] for s in piv.columns])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    blk = blocks_of.loc[piv.index].to_numpy()
    for i in range(1, len(blk)):
        if blk[i] != blk[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2)
    ax.set_xlabel("window before SBL")
    fig.colorbar(im, ax=ax, shrink=0.6, label=value)
    ax.set_title(title or f"KS distance {value} of |row-to-row change|: window before SBL vs. reference", fontsize=9)
    if save_as:
        fig.savefig(save_as, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig
