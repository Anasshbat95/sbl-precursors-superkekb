"""
discussion_plots.py - Figures for the discussion chapter (limitations of the data).

Usage (notebook):
    from files.discussion_plots import time_since_last_row, plot_time_scales, plot_detectable_rate

    age = time_since_last_row(blocks, sbl_times)          # table: seconds from last row to each SBL
    plot_time_scales(age, save="disc_time_scales.png")    # fig: ECDF of the age per block, log x
    plot_detectable_rate(save="disc_detectable_rate.png") # fig: smallest rate above chance vs. N_SBL

Every plot function accepts save="name.png" to write the figure into MasterArbeit/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from files.snapshot_analysis import COL, _sbl_ns, _save


def time_since_last_row(blocks, sbl_times):
    """Seconds between each SBL and the last row of each block strictly before it (NaN if none)."""
    sbl = _sbl_ns(sbl_times)
    out = {}
    for name, df in blocks.items():
        t = np.sort(pd.to_datetime(df["timestamp"]).to_numpy("datetime64[ns]"))
        i = np.searchsorted(t, sbl) - 1                     # last row with t < t_SBL
        age = (sbl - t[np.clip(i, 0, None)]) / np.timedelta64(1, "s")
        out[name] = np.where(i >= 0, age, np.nan)
    return pd.DataFrame(out, index=pd.DatetimeIndex(sbl, name="sbl_time"))


def plot_time_scales(age, precursor_s=(1e-5, 1e-3), windows_s=(120, 7200), save=None):
    """ECDF of the time since the last row before each SBL, per block, on a log axis.
    precursor_s: range of the precursor time scale reported with fast data in earlier work (shaded);
                 default: 1 turn (10 us) to the ~90 turns (~0.9 ms) of BOR data before an abort [haigh2026].
    windows_s:   observation windows used in this thesis (dashed lines)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvspan(*precursor_s, color="#e4e3df", label="BOR precursors (Haigh 2026)")
    for name in age.columns:
        a = np.sort(age[name].dropna().to_numpy())
        a = np.clip(a, precursor_s[0], None)                # rows written at t_SBL would sit at 0 on a log axis
        y = np.arange(1, len(a) + 1) / len(age)
        ax.step(a, y, where="post", color=COL.get(name, None), lw=1.8,
                ls="--" if name == "position" else "-",        # POSITION shares the BEAM timestamps
                label=f"{name} (median {np.median(a) / 60:.0f} min)")
    for w in windows_s:
        ax.axvline(w, color="k", ls="--", lw=0.8)
        ax.text(w, 1.02, f"{w:g} s", ha="center", va="bottom", fontsize=8)
    ax.set(xscale="log", xlim=(precursor_s[0] / 2, age.max().max() * 1.5), ylim=(0, 1),
           xlabel="time from the last row of the block to the SBL [s]", ylabel="share of SBLs (ECDF)")
    ax.grid(alpha=0.3, which="major")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout(); _save(fig, save); plt.show()


def _wilson_lo(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    return ((p + z * z / (2 * n)) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def detectable_rate(n, p0):
    """Smallest share k/n of SBLs whose Wilson 95% lower bound lies above the chance rate p0."""
    k = np.arange(n + 1)
    ok = _wilson_lo(k, n) > p0
    return k[ok][0] / n if ok.any() else np.nan


def plot_detectable_rate(n_range=range(20, 1001, 5), chance=(0.01, 0.02, 0.05), n_sbl=128, save=None):
    """Smallest detectable share of SBLs (Wilson lower bound > chance) against the number of SBLs,
    for several chance rates; the dotted line marks the number of SBLs of this thesis."""
    n = np.array(list(n_range))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p0, c in zip(chance, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
        r = [detectable_rate(m, p0) for m in n]
        ax.plot(n, np.array(r) * 100, color=c, lw=1.8, label=f"chance rate {p0:.0%}")
        r0 = detectable_rate(n_sbl, p0)
        ax.scatter([n_sbl], [r0 * 100], color=c, zorder=3)
        ax.annotate(f"{r0:.1%}", (n_sbl, r0 * 100), xytext=(6, 4), textcoords="offset points", fontsize=8, color=c)
    ax.axvline(n_sbl, color="k", ls=":", lw=0.8)
    ax.set(xscale="log", xlabel="number of SBLs", ylabel="smallest detectable share of SBLs [%]",
           ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); _save(fig, save); plt.show()
