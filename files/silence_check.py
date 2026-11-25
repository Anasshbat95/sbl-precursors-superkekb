import numpy as np
import pandas as pd


def _times(df, time_col=None):
    """Zeitstempel eines Blocks holen (DatetimeIndex oder Datums-Spalte)."""
    if time_col is not None:
        t = pd.to_datetime(df[time_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        t = pd.Series(df.index)
    else:
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if not cols:
            raise ValueError("Keine Zeitspalte gefunden - time_col angeben.")
        t = pd.to_datetime(df[cols[0]])
    return np.sort(t.values.astype("datetime64[ns]"))


def silence_before_sbl(blocks, sbl, shifts_h=(1, 2, 3, 4, 5, 6),
                       silent_min=30, time_col=None):
    """
    Wie lange war ein ganzer Block STILL (keine einzige Zeile) vor jedem SBL
    und vor seinen eigenen Kontrollzeiten (SBL - 1..6 h)?

    Rueckgabe
    ---------
    per_sbl : DataFrame, eine Zeile pro SBL, Spalten
              <block>_sbl_min  = Minuten seit letzter Zeile vor dem SBL
              <block>_ctrl_min = Median ueber die Kontrollzeiten
              all_silent       = True, wenn ALLE Bloecke > silent_min still waren
    summary : DataFrame pro Block: Anteil SBLs / Kontrollen mit Stille > silent_min
    """
    sbl = pd.to_datetime(pd.Series(sbl)).sort_values().reset_index(drop=True)
    s = sbl.values.astype("datetime64[ns]")
    per_sbl = pd.DataFrame({"sbl_time": sbl})
    summary = []
    for name, df in blocks.items():
        t = _times(df, time_col)

        def gap(q):
            i = np.searchsorted(t, q, side="left") - 1   # letzte Zeile VOR q
            out = np.full(len(q), np.nan)
            ok = i >= 0
            out[ok] = (q[ok] - t[i[ok]]) / np.timedelta64(1, "m")
            return out

        g_sbl = gap(s)
        g_ctrl = np.column_stack([gap(s - np.timedelta64(h, "h")) for h in shifts_h])
        per_sbl[f"{name}_sbl_min"] = g_sbl
        per_sbl[f"{name}_ctrl_min"] = np.nanmedian(g_ctrl, axis=1)
        summary.append({
            "block": name,
            "share_sbl_silent": np.nanmean(g_sbl > silent_min),
            "share_ctrl_silent": np.nanmean(g_ctrl > silent_min),
            "median_sbl_min": np.nanmedian(g_sbl),
            "median_ctrl_min": np.nanmedian(g_ctrl),
        })
    sbl_cols = [c for c in per_sbl.columns if c.endswith("_sbl_min")]
    per_sbl["all_silent"] = (per_sbl[sbl_cols] > silent_min).all(axis=1)
    return per_sbl, pd.DataFrame(summary).set_index("block")
