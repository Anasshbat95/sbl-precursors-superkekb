"""
ecdf_plot.py - English version of plot_ecdf_window_vs_all (diff_distribution_before_sbl.py) with the
same choices as ks_overview.py, so that the D values in the figure and in the heatmap agree.

Differences to the old function:
    * all labels, legends and titles in English (for the thesis);
    * buffer: rows in the last `buffer` before the SBL are left out of the window (b < tau <= W);
    * iv (from operation_state.operation_intervals): the REFERENCE is restricted to BEAM_ON;
    * |dv| is computed on the full block (sorted by time) before any selection;
    * the default reference is "normal" (rows farther than normal_margin from every SBL).

Usage:
    from ecdf_plot import plot_ecdf_window_vs_ref
    fig, t = plot_ecdf_window_vs_ref(blocks, "timestamp", ["her_oscillator_freq", "ler_vac_pres_global"],
                                     sbl_times["timestamp"], time_before=["5min", "30min"],
                                     normal_margin="2h", buffer="30s", iv=iv, n_cols=2)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from files.diff_distribution_before_sbl import _dist_to_nearest_sbl, _find_feature, add_diff_columns, time_before_next_sbl


def plot_ecdf_window_vs_ref(blocks: dict, ts_col: str, features, sbl_times, time_before="5min",
                            quantity="diff", kind="ecdf", reference="normal", normal_margin="2h",
                            buffer="0s", iv=None, use_abs=True, xscale=None, bins=60, n_cols=3, dpi=300,
                            window_colormap="Dark2", title=None, save_as=None) -> tuple:
    """Distribution of a feature in the window before the SBLs versus a reference, in one panel per feature.

    quantity: "diff" = |v(t_i) - v(t_(i-1))| (the values of the KS test); "value" = v(t_i) itself.
    kind:     "ecdf" = empirical distribution functions; "hist" = density histograms (only for "value").
    reference: "normal" = rows farther than normal_margin from every SBL; "all" = every row.
    buffer:   rows with tau <= buffer (last moments before the SBL) are not in the window.
    iv:       interval table from operation_state; if given, the reference contains only BEAM_ON rows.
    time_before: one window ("5min") or a list (["5min", "30min"]) - one curve per window.
    Returns (figure, table).
    """
    if quantity not in ("diff", "value"):
        raise ValueError("quantity must be 'diff' or 'value'.")
    if kind not in ("ecdf", "hist"):
        raise ValueError("kind must be 'ecdf' or 'hist'.")
    if reference not in ("all", "normal"):
        raise ValueError("reference must be 'all' or 'normal'.")
    if kind == "hist" and quantity == "diff":
        raise ValueError("kind='hist' is only offered for quantity='value'.")
    use_abs = use_abs and quantity == "diff"
    xscale = xscale or ("symlog" if quantity == "diff" else "linear")
    margin_s = pd.Timedelta(normal_margin).total_seconds()
    b_s = pd.Timedelta(buffer).total_seconds()
    windows = [time_before] if isinstance(time_before, str) else list(time_before)
    feats = list(dict.fromkeys(features))
    located = [(f,) + _find_feature(blocks, f) for f in feats]

    n_cols = min(n_cols, len(located))
    n_rows = int(np.ceil(len(located) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.9 * n_rows), dpi=dpi,
                             squeeze=False, layout="constrained")
    axes = axes.ravel()
    cmap = plt.get_cmap(window_colormap)
    if reference == "all":
        ref_label_base = "all rows"
    else:
        ref_label_base = f"normal operation (> {normal_margin} from every SBL" + (", beam on)" if iv is not None else ")")
    rows, cache = [], {}

    for ax, (feat, block, col) in zip(axes, located):
        if block not in cache:
            df = blocks[block].sort_values(ts_col).reset_index(drop=True)
            if quantity == "diff":
                df = add_diff_columns(df, [c for c in df.select_dtypes("number").columns if c != ts_col])
            tau = time_before_next_sbl(df, ts_col, sbl_times)
            dist = _dist_to_nearest_sbl(df, ts_col, sbl_times)
            ref_mask = np.ones(len(df), bool) if reference == "all" else dist > margin_s
            if iv is not None:
                from operation_state import label_states
                ref_mask &= (label_states(df[[ts_col]], iv, ts_col)["state"] == "BEAM_ON").to_numpy()
            cache[block] = (df, tau, ref_mask)
        d, tau, ref_mask = cache[block]
        series = d[f"{col}_diff"] if quantity == "diff" else d[col]
        if use_abs:
            series = series.abs()
        a_sorted = np.sort(series[ref_mask].dropna().to_numpy())
        if len(a_sorted) < 2:
            ax.text(0.5, 0.5, "too few reference values", transform=ax.transAxes, ha="center", color="gray")
            ax.set_title(feat, fontsize=10)
            continue
        ref_label = f"{ref_label_base}, n = {len(a_sorted)}"

        win_sorted = []
        for w in windows:
            w_s = pd.Timedelta(w).total_seconds()
            in_win = (tau <= w_s) & ((tau > b_s) if b_s > 0 else (tau >= 0))
            win_sorted.append(np.sort(series[in_win].dropna().to_numpy()))
        pooled = np.concatenate([a_sorted] + win_sorted)

        if kind == "hist":
            lo, hi = np.percentile(pooled, [0.5, 99.5])
            if not hi > lo:
                lo, hi = float(pooled.min()), float(pooled.max()) + 1e-12
            edges = np.linspace(lo, hi, bins + 1)
            ax.hist(a_sorted, bins=edges, density=True, histtype="stepfilled", color="#377eb8", alpha=0.3)
            ax.hist(a_sorted, bins=edges, density=True, histtype="step", color="#377eb8", lw=1.6, label=ref_label)
            ax.set_xlim(lo, hi)
            ax.set_ylabel("density", fontsize=8)
        else:
            ax.step(np.r_[a_sorted[0], a_sorted], np.r_[0, np.arange(1, len(a_sorted) + 1) / len(a_sorted)],
                    where="post", color="#377eb8", lw=2.0, label=ref_label, zorder=3)
            ax.set_ylabel("cumulative share F(x)", fontsize=8)
            ax.set_ylim(-0.02, 1.02)

        for k, (w, w_sorted) in enumerate(zip(windows, win_sorted)):
            if len(w_sorted) < 2:
                rows.append(dict(feature=feat, block=block, window=w, n_window=len(w_sorted), n_ref=len(a_sorted)))
                ax.text(0.5, 0.5 - 0.08 * k, f"{w}: too few values ({len(w_sorted)})", transform=ax.transAxes,
                        ha="center", color="gray")
                continue
            stat, p = ks_2samp(w_sorted, a_sorted)
            d_crit = 1.36 * np.sqrt((len(w_sorted) + len(a_sorted)) / (len(w_sorted) * len(a_sorted)))
            xs = np.unique(np.concatenate([w_sorted, a_sorted]))
            gap = np.abs(np.searchsorted(w_sorted, xs, side="right") / len(w_sorted)
                         - np.searchsorted(a_sorted, xs, side="right") / len(a_sorted))
            x_at_d = float(xs[int(np.argmax(gap))])
            color = "#e6550d" if len(windows) == 1 else cmap(k)
            w_label = f"last {w} before SBL, n = {len(w_sorted)}" + ("" if len(windows) == 1 else f", D = {stat:.2f}")
            if kind == "hist":
                ax.hist(w_sorted, bins=edges, density=True, histtype="stepfilled", color=color, alpha=0.3)
                ax.hist(w_sorted, bins=edges, density=True, histtype="step", color=color, lw=1.6, label=w_label)
            else:
                ax.step(np.r_[w_sorted[0], w_sorted], np.r_[0, np.arange(1, len(w_sorted) + 1) / len(w_sorted)],
                        where="post", color=color, lw=1.8, zorder=4, label=w_label)
                if len(windows) == 1:
                    lo_y = np.searchsorted(a_sorted, x_at_d, side="right") / len(a_sorted)
                    hi_y = np.searchsorted(w_sorted, x_at_d, side="right") / len(w_sorted)
                    ax.vlines(x_at_d, min(lo_y, hi_y), max(lo_y, hi_y), color="black", lw=1.6, zorder=5)
                    ax.annotate(f"D = {stat:.3f}", (x_at_d, (lo_y + hi_y) / 2), xytext=(6, 0),
                                textcoords="offset points", fontsize=8, va="center", zorder=6)
            rows.append(dict(feature=feat, block=block, column=col, quantity=quantity, reference=reference,
                             window=w, n_window=len(w_sorted), n_ref=len(a_sorted), D=stat, D_crit=d_crit,
                             D_rel=stat / d_crit, p_value=p, x_at_D=x_at_d,
                             frac_zero_window=float((w_sorted == 0).mean()),
                             frac_zero_ref=float((a_sorted == 0).mean()),
                             median_window=float(np.median(w_sorted)), median_ref=float(np.median(a_sorted))))

        if kind == "ecdf":
            pos = pooled[pooled > 0]
            if xscale == "symlog" and len(pos):
                ax.set_xscale("symlog", linthresh=float(pos.min()))
            elif xscale == "log" and len(pos):
                ax.set_xscale("log")
        mine = [r for r in rows if r["feature"] == feat and "D" in r]
        ttl = feat if col == feat else f"{feat}  ({col})"
        if len(windows) == 1 and mine:
            ttl += f"\nD = {mine[0]['D']:.3f},  D_rel = {mine[0]['D_rel']:.2f}"
        ax.set_title(ttl, fontsize=10)
        if quantity == "diff":
            ax.set_xlabel(r"$|\Delta v| = |v(t_i) - v(t_{i-1})|$  (change between two rows)" if use_abs
                          else r"$\Delta v = v(t_i) - v(t_{i-1})$  (change between two rows)", fontsize=8)
        else:
            ax.set_xlabel(f"value of {col}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right" if kind == "ecdf" else "upper right")

    for ax in axes[len(located):]:
        ax.set_visible(False)
    if title is None:
        what = "row-to-row change" if quantity == "diff" else "values"
        how = "Empirical distribution functions" if kind == "ecdf" else "Density histograms"
        ref = "all rows" if reference == "all" else "normal operation"
        title = f"{how} of the {what}: window before SBL vs. {ref}"
    fig.suptitle(title, fontsize=10)
    if save_as:
        fig.savefig(save_as, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, pd.DataFrame(rows)
