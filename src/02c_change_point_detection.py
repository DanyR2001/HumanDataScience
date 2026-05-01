#!/usr/bin/env python3
"""
02_change_point_detection_margin.py
=====================================
Change-point detection sul MARGINE (prezzo netto pompa − futures €/L)
benzina e gasolio in corrispondenza di eventi geopolitici rilevanti.

Metodologia — Rasoio di Occam (dal più semplice al più sofisticato)
────────────────────────────────────────────────────────────────────
  L1 · Sliding-window two-sample t-test
       Finestra di HALF_WIN=40 giorni per lato; il punto di rottura τ
       scorre da –SEARCH giorni a +SEARCH giorni rispetto all'evento
       con step=1 (risoluzione giornaliera). Correzione p-value: Bonferroni.
       → restituisce 1 break point (il più forte).

  L2 · CUSUM (Cumulative SUM of deviations)
       Deviazione cumulata dalla media pre-evento sul livello di prezzo.
       Picco/valle = momento di rottura strutturale. Semplice, visivo.
       → 1 break point.

  L3 · Binary Segmentation (ruptures – BinSeg, costo L2)
       Cerca esattamente N break points; usa BIC per scegliere N ottimale
       tra 1 e MAX_BKPS. → potenzialmente MULTIPLI break point.

  L4 · PELT – Pruned Exact Linear Time (ruptures, penalizzazione BIC)
       Soluzione globale esatta con N incognito; O(n log n).
       → potenzialmente MULTIPLI break point.

Finestra ottimale e CLT
────────────────────────
  Prezzi giornalieri: autocorrelazione AR(1) con φ ≈ 0.2–0.3.
  Dimensione effettiva del campione: n_eff ≈ n·(1-φ)/(1+φ).
  Con φ=0.3 → n_eff = n·0.54.
  Per n_eff ≥ 30 (CLT solido) serve n ≥ 56 → uso HALF_WIN=40 (minimo
  pratico, n_eff≈22) con la consapevolezza che il test t è approssimato.
  Se vuoi CLT garantito imposta HALF_WIN=60.

  Sliding step = 1 giorno (risoluzione massima, 2·SEARCH+1 test totali;
  si applica correzione Bonferroni sui p-value).
  Step = 7 riduce il multiple testing ma perde risoluzione infrasettimanale.

Dipendenze extra:
  pip install ruptures
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    warnings.warn("ruptures non installato – L3/L4 disabilitati. "
                  "Installa con: pip install ruptures")

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DAILY_CSV    = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR      = BASE_DIR / "data" / "plots" / "change_point" / "margin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HALF_WIN   = 40
SEARCH     = 40
STEP       = 1
MAX_BKPS   = 5
MIN_SIZE   = 14
ZOOM_WIN   = 25   # giorni prima/dopo τ nel pannello zoom (configurabile)

FUELS = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-12-01"),
        "post_end":  pd.Timestamp("2022-04-24"),
        "color":     "#e74c3c",
        "label":     "Russia-Ucraina\n(24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-04-13"),
        "post_end":  pd.Timestamp("2025-08-13"),
        "color":     "#e67e22",
        "label":     "Iran-Israele\n(13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-12-28"),
        "post_end":  pd.Timestamp("2026-04-30"),
        "color":     "#8e44ad",
        "label":     "Stretto di Hormuz\n(28 feb 2026)",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: n_eff e Welch t-test corretto per autocorrelazione AR(1)
# ══════════════════════════════════════════════════════════════════════════════

def _phi_ar1(series: pd.Series) -> float:
    """Stima il coefficiente AR(1) via correlazione lag-1."""
    if len(series) < 4:
        return 0.0
    s = series.values - series.mean()
    if s.std() < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(s[:-1], s[1:])[0, 1], -0.99, 0.99))


def _n_eff(n: int, phi: float) -> float:
    """Dimensione campionaria effettiva per AR(1): n*(1-φ)/(1+φ)."""
    return max(2.0, n * (1.0 - phi) / (1.0 + phi))


def _welch_neff(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    """
    Welch t-test corretto per autocorrelazione AR(1).
    Stima φ separatamente su pre e post, corregge n prima dei
    gradi di libertà di Welch-Satterthwaite.
    Questo produce p-value meno ottimistici rispetto al Welch naïve
    quando la serie è autocorrelata (φ > 0 riduce n_eff).
    """
    phi_pre  = _phi_ar1(pre)
    phi_post = _phi_ar1(post)
    n1 = _n_eff(len(pre),  phi_pre)
    n2 = _n_eff(len(post), phi_post)

    m1, m2 = float(pre.mean()), float(post.mean())
    v1 = float(pre.var(ddof=1))  / n1   # varianza della media corretta
    v2 = float(post.var(ddof=1)) / n2

    se = np.sqrt(v1 + v2)
    if se < 1e-12:
        return 0.0, 1.0

    t_stat = (m2 - m1) / se
    # Welch-Satterthwaite df
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    df = max(1.0, df)
    p  = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    return float(t_stat), p


# ══════════════════════════════════════════════════════════════════════════════
# L1 – Sliding-window t-test
# ══════════════════════════════════════════════════════════════════════════════
def sliding_ttest(
    series: pd.Series,
    shock: pd.Timestamp,
    half_win: int = HALF_WIN,
    search: int   = SEARCH,
    step: int     = STEP,
) -> pd.DataFrame:
    """
    Per ogni candidato τ in [shock–search, shock+search] (step=step giorni),
    esegue un Welch t-test tra i HALF_WIN giorni prima di τ e i HALF_WIN dopo.
    Restituisce DataFrame ordinato per |t_stat| decrescente.

    Nota sul CLT:
      Con half_win=40 e autocorrelazione AR(1) φ≈0.3,
      n_eff ≈ 40 * 0.54 ≈ 22 < 30  →  CLT approssimativo.
      Imposta half_win=60 per CLT garantito.
    """
    idx   = series.index  # DatetimeIndex
    rows  = []
    n_tests = 0

    candidates = pd.date_range(
        shock - pd.Timedelta(days=search),
        shock + pd.Timedelta(days=search),
        freq=f"{step}D",
    )

    for tau in candidates:
        pre  = series[(idx >= tau - pd.Timedelta(days=half_win)) & (idx < tau)].dropna()
        post = series[(idx >= tau) & (idx < tau + pd.Timedelta(days=half_win))].dropna()
        if len(pre) < 5 or len(post) < 5:
            continue
        t, p = _welch_neff(pre, post)   # corretto per autocorrelazione AR(1)
        delta = post.mean() - pre.mean()
        rows.append({"tau": tau, "t_stat": t, "p_raw": p, "delta_mean": delta,
                     "n_pre": len(pre), "n_post": len(post)})
        n_tests += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Correzione Bonferroni
    df["p_bonf"] = (df["p_raw"] * n_tests).clip(upper=1.0)
    df["abs_t"]  = df["t_stat"].abs()
    return df.sort_values("abs_t", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# L2 – CUSUM
# ══════════════════════════════════════════════════════════════════════════════
def cusum(
    series: pd.Series,
    shock: pd.Timestamp,
    pre_window: int = HALF_WIN,
) -> tuple[pd.Series, pd.Timestamp]:
    """
    CUSUM delle deviazioni dalla media pre-evento.
    Restituisce (serie CUSUM, data del massimo |CUSUM|).
    """
    baseline = series[series.index < shock].tail(pre_window).mean()
    dev      = series - baseline
    cs       = dev.cumsum()
    peak_idx = cs.abs().idxmax()
    return cs, peak_idx


# ══════════════════════════════════════════════════════════════════════════════
# L3 – Binary Segmentation  (ruptures)
# ══════════════════════════════════════════════════════════════════════════════
def binseg_detect(
    series: pd.Series,
    max_bkps: int  = MAX_BKPS,
    min_size: int  = MIN_SIZE,
    model: str     = "rbf",
) -> dict[int, list[pd.Timestamp]]:
    """
    BinSeg su 1..max_bkps break point; sceglie N ottimale con BIC.
    Restituisce {n_bkps: [date break point]} per ogni N testato,
    più la chiave 'best' con N ottimale.
    """
    if not HAS_RUPTURES:
        return {}

    signal = series.values.reshape(-1, 1)
    algo   = rpt.Binseg(model=model, min_size=min_size).fit(signal)

    results: dict = {}
    costs   = []

    for n in range(1, max_bkps + 1):
        try:
            bkps = algo.predict(n_bkps=n)  # indici (1-based, ultimo = len)
        except Exception:
            continue
        dates = [series.index[b - 1] for b in bkps[:-1]]  # escludi sentinel
        results[n] = dates
        # Costo BIC: costo_residuo + n * log(T)
        cost = algo.cost.sum_of_costs(bkps)
        bic  = cost + n * np.log(len(signal))
        costs.append((n, bic))

    if costs:
        best_n = min(costs, key=lambda x: x[1])[0]
        results["best"] = results[best_n]
        results["best_n"] = best_n
        results["bic_curve"] = costs

    return results


# ══════════════════════════════════════════════════════════════════════════════
# L4 – PELT  (ruptures)
# ══════════════════════════════════════════════════════════════════════════════
def pelt_detect(
    series: pd.Series,
    min_size: int = MIN_SIZE,
    model: str    = "rbf",
) -> list[pd.Timestamp]:
    """
    PELT con penalizzazione BIC (pen = log(T)).
    Restituisce lista di date dei change point rilevati.
    """
    if not HAS_RUPTURES:
        return []

    signal = series.values.reshape(-1, 1)
    pen    = np.log(len(signal))  # BIC
    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        bkps = algo.predict(pen=pen)
    except Exception as e:
        warnings.warn(f"PELT fallito: {e}")
        return []

    return [series.index[b - 1] for b in bkps[:-1]]


# ══════════════════════════════════════════════════════════════════════════════
# L5 – Window-based L2 Discrepancy  (Paper BLOCCO 1 – Eq. 1–2)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_cost(y: np.ndarray) -> float:
    """
    Funzione di costo intra-finestra (Paper, Eq. 2):
        c(y_I) = Σ_{t∈I} ||y_t − ȳ||²₂
    Equivale alla varianza pesata per n: misura la dispersione interna
    alla finestra I. Assunzione gaussiana (costo L2 quadratico).
    """
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def window_discrepancy_l2(
    series: pd.Series,
    shock: pd.Timestamp,
    search: int   = SEARCH,
    ma_window: int = 7,
    min_seg: int   = 14,
) -> dict:
    """
    Changepoint detection via discrepanza L2 tra due finestre contigue (Paper, Eq. 1–2).

    Pre-processing obbligatorio (Paper): moving average a 7 giorni.

    Algoritmo:
      Per ogni candidato v ∈ [u+min_seg, w−min_seg]:
        d(y_{uv}, y_{vw}) = c(y_{uw}) − c(y_{uv}) − c(y_{vw})
      Changepoint = argmax_v  d(...)

    La discrepanza d misura quanto v divide la serie in due segmenti
    omogenei al loro interno ma diversi tra loro. Picco di d → break.

    Validazione BIC: il changepoint L5 è attendibile se coincide
    (entro ±7gg) con almeno uno dei break trovati da L3 BinSeg o L4 PELT.

    Returns
    -------
    dict con:
      tau       – data del changepoint rilevato
      d_max     – valore massimo della discrepanza
      d_values  – pd.Series(d, index=candidate_dates)
      ma_series – serie dopo MA a 7 giorni (per plot)
      bic_ok    – bool: il changepoint è in accordo con L3/L4
    """
    # Slice nella finestra di ricerca
    mask = (series.index >= shock - pd.Timedelta(days=search)) & \
           (series.index <= shock + pd.Timedelta(days=search))
    win = series[mask].dropna()

    if len(win) < 2 * min_seg + ma_window:
        return {
            "tau": shock, "d_max": 0.0,
            "d_values": pd.Series(dtype=float),
            "ma_series": pd.Series(dtype=float),
            "bic_ok": False,
        }

    # Pre-processing: MA a 7 giorni (centrante, min_periods=1 per i bordi)
    ma = win.rolling(ma_window, center=True, min_periods=1).mean()
    y  = ma.values
    n  = len(y)

    # c(y_{uw}) — costo dell'intera finestra (costante nel loop)
    c_uw = _l2_cost(y)

    # Scan su tutti i candidati v
    d_vals: list[float] = []
    cand_dates: list[pd.Timestamp] = []

    for v in range(min_seg, n - min_seg):
        c_uv = _l2_cost(y[:v])
        c_vw = _l2_cost(y[v:])
        d    = c_uw - c_uv - c_vw
        d_vals.append(d)
        cand_dates.append(ma.index[v])

    if not cand_dates:
        return {
            "tau": shock, "d_max": 0.0,
            "d_values": pd.Series(dtype=float),
            "ma_series": ma,
            "bic_ok": False,
        }

    d_series = pd.Series(d_vals, index=cand_dates)
    best_tau = d_series.idxmax()

    return {
        "tau":       best_tau,
        "d_max":     float(d_series.max()),
        "d_values":  d_series,
        "ma_series": ma,
        "bic_ok":    False,   # aggiornato in plot_event_fuel dopo L3/L4
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting per evento + carburante
# ══════════════════════════════════════════════════════════════════════════════
def plot_event_fuel(
    event_name: str,
    ev: dict,
    series: pd.Series,
    fuel_label: str,
    fuel_color: str,
    ax_price:  plt.Axes,
    ax_cusum:  plt.Axes,
    ax_tstat:  plt.Axes,
    ax_bic:    plt.Axes,
    ax_l5:     plt.Axes | None = None,
) -> dict:
    """Disegna i 4+1 pannelli per un singolo carburante e restituisce risultati."""

    # ── Slice della finestra ──────────────────────────────────────────────────
    win = series[(series.index >= ev["pre_start"]) &
                 (series.index <= ev["post_end"])].dropna()
    if len(win) < 2 * HALF_WIN:
        ax_price.set_title(f"{fuel_label} – dati insufficienti", fontsize=9)
        return {}

    shock = ev["shock"]
    color = fuel_color

    # ── L1: sliding t-test ────────────────────────────────────────────────────
    ttest_df = sliding_ttest(win, shock)
    best_tau = ttest_df.iloc[0]["tau"]   if not ttest_df.empty else shock
    best_t   = ttest_df.iloc[0]["t_stat"] if not ttest_df.empty else 0
    best_p   = ttest_df.iloc[0]["p_bonf"] if not ttest_df.empty else 1
    delta    = ttest_df.iloc[0]["delta_mean"] if not ttest_df.empty else 0

    # ── L2: CUSUM ─────────────────────────────────────────────────────────────
    cs, cusum_peak = cusum(win, shock)

    # ── L3: BinSeg ───────────────────────────────────────────────────────────
    binseg_res = binseg_detect(win)
    binseg_best = binseg_res.get("best", [])

    # ── L4: PELT ─────────────────────────────────────────────────────────────
    pelt_breaks = pelt_detect(win)

    # ── L5: Window discrepancy L2 (Paper BLOCCO 1 – Eq. 1–2) ─────────────────
    l5_res = window_discrepancy_l2(win, shock, search=SEARCH)
    # Validazione BIC: L5 è attendibile se coincide (±7gg) con L3 o L4
    _all_bic_breaks = binseg_best + pelt_breaks
    if _all_bic_breaks and l5_res["d_values"] is not None and not l5_res["d_values"].empty:
        l5_res["bic_ok"] = any(
            abs((l5_res["tau"] - b).days) <= 7 for b in _all_bic_breaks
        )
    else:
        l5_res["bic_ok"] = False

    # ════ PANNELLO 1: prezzi + break lines  (zoom ZOOM_WIN gg attorno a τ) ═════
    # Centrato su τ (break rilevato), non sullo shock: mostra esattamente
    # la transizione pre→post. Cambia ZOOM_WIN in cima al file per allargare.
    zoom_start = best_tau - pd.Timedelta(days=ZOOM_WIN)
    zoom_end   = best_tau + pd.Timedelta(days=ZOOM_WIN)
    win_zoom   = win[(win.index >= zoom_start) & (win.index <= zoom_end)]

    ax_price.plot(win_zoom.index, win_zoom.values, color=color, lw=0.9, label=fuel_label)
    ax_price.axvline(shock,     color=ev["color"], lw=1.5, ls="--", label="Evento")
    ax_price.axvline(best_tau,  color=color, lw=1.2, ls=":", alpha=0.8,
                     label=f"L1 τ={best_tau.date()}")
    ax_price.axvline(cusum_peak, color="darkorange", lw=1.0, ls="-.",
                     label=f"L2 CUSUM={cusum_peak.date()}")
    for i, d in enumerate(pelt_breaks):
        if zoom_start <= d <= zoom_end:
            ax_price.axvline(d, color="green", lw=0.8, ls=":", alpha=0.7,
                             label=f"L4 PELT" if i == 0 else "")
    for i, d in enumerate(binseg_best):
        if zoom_start <= d <= zoom_end:
            ax_price.axvline(d, color="purple", lw=0.8, ls=":", alpha=0.7,
                             label=f"L3 BinSeg" if i == 0 else "")
    ax_price.set_xlim(zoom_start, zoom_end)
    ax_price.set_ylabel("Margine (€/L)", fontsize=8)
    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
    ax_price.grid(axis="y", alpha=0.25)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ════ PANNELLO 2: CUSUM ═══════════════════════════════════════════════════
    ax_cusum.plot(cs.index, cs.values, color=color, lw=0.9)
    ax_cusum.axhline(0, color="grey", lw=0.7, ls="--")
    ax_cusum.axvline(shock,      color=ev["color"], lw=1.5, ls="--")
    ax_cusum.axvline(cusum_peak, color="darkorange", lw=1.2, ls="-.")
    ax_cusum.set_ylabel("CUSUM margine (€/L)", fontsize=8)
    ax_cusum.fill_between(cs.index, cs.values, 0,
                          where=cs.values > 0, alpha=0.15, color=color)
    ax_cusum.fill_between(cs.index, cs.values, 0,
                          where=cs.values < 0, alpha=0.15, color="green")
    ax_cusum.grid(axis="y", alpha=0.25)
    ax_cusum.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 3: t-stat sliding — solo τ dove margine CRESCE (Δ > 0) ══════
    if not ttest_df.empty:
        rising = ttest_df[ttest_df["delta_mean"] > 0]
        ax_tstat.plot(rising["tau"], rising["t_stat"].abs(),
                      color=color, lw=0.9, label="Δ > 0 (margine cresce)")
        ax_tstat.axvline(shock,    color=ev["color"], lw=1.5, ls="--", label="Evento")
        if not rising.empty and best_tau in rising["tau"].values:
            ax_tstat.axvline(best_tau, color=color, lw=1.2, ls=":", label=f"τ={best_tau.date()}")
        ax_tstat.axhline(stats.t.ppf(0.975, df=HALF_WIN * 2 - 2),
                         color="grey", lw=0.7, ls=":", label="α=0.05")
        ax_tstat.set_ylabel("|t-stat|  (solo Δ>0)", fontsize=8)
        ax_tstat.legend(fontsize=6)
        sig = "★" if best_p < 0.05 else ""
        ax_tstat.set_title(
            f"L1 sliding t-test  |  τ={best_tau.date()}  "
            f"Δ={delta:+.4f} €/L  p_bonf={best_p:.3f}{sig}",
            fontsize=7, pad=2
        )
    ax_tstat.grid(axis="y", alpha=0.25)
    ax_tstat.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 4: BIC curve BinSeg ════════════════════════════════════════
    if binseg_res.get("bic_curve"):
        ns, bics = zip(*binseg_res["bic_curve"])
        ax_bic.plot(ns, bics, marker="o", color=color, lw=1)
        best_n = binseg_res.get("best_n", 1)
        ax_bic.axvline(best_n, color="grey", lw=0.8, ls="--",
                       label=f"N ottimale={best_n}")
        ax_bic.set_xlabel("N break point", fontsize=8)
        ax_bic.set_ylabel("BIC", fontsize=8)
        ax_bic.set_xticks(list(ns))
        ax_bic.legend(fontsize=6)
        ax_bic.set_title(
            f"L3 BinSeg: {len(binseg_best)} break  |  "
            f"L4 PELT: {len(pelt_breaks)} break  →  "
            + (", ".join(str(d.date()) for d in pelt_breaks) if pelt_breaks else "nessuno"),
            fontsize=7, pad=2
        )
    else:
        ax_bic.text(0.5, 0.5, "ruptures non disponibile",
                    ha="center", va="center", transform=ax_bic.transAxes, fontsize=8)
    ax_bic.grid(alpha=0.25)

    # ════ PANNELLO 5: L5 Window Discrepancy (Paper Eq. 1–2) ═══════════════════
    if ax_l5 is not None:
        d_vals = l5_res.get("d_values")
        ma_ser = l5_res.get("ma_series")
        l5_tau = l5_res["tau"]
        bic_ok = l5_res["bic_ok"]

        if d_vals is not None and not d_vals.empty:
            ax_l5.plot(d_vals.index, d_vals.values, color=color, lw=0.9,
                       label="d(y_uv, y_vw) — discrepanza L2")
            ax_l5.axvline(shock,   color=ev["color"], lw=1.5, ls="--", label="Shock")
            ax_l5.axvline(l5_tau,  color=color,       lw=1.8, ls=":",
                          label=f"L5 τ={l5_tau.date()}"
                                f"  {'✓ BIC ok' if bic_ok else '– non confermato BIC'}")
            ax_l5.axhline(0, color="grey", lw=0.6, ls="-")

            # Evidenzia il massimo con una stella
            ax_l5.scatter([l5_tau], [d_vals.max()],
                          color=color, s=60, zorder=5, marker="*")

            ax_l5.set_ylabel("Discrepanza d (€/L)²", fontsize=8)
            ax_l5.legend(fontsize=6)
            bic_tag = "✓ confermato da BIC (L3/L4)" if bic_ok else "⚠ non in accordo con BIC"
            ax_l5.set_title(
                f"L5 Window Discrepancy L2  |  τ={l5_tau.date()}  "
                f"d_max={l5_res['d_max']:.4f}  {bic_tag}",
                fontsize=7, pad=2
            )
        else:
            ax_l5.text(0.5, 0.5, "L5: dati insufficienti",
                       ha="center", va="center", transform=ax_l5.transAxes, fontsize=8)
        ax_l5.grid(alpha=0.25)
        ax_l5.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    return {
        "L1_tau":      best_tau,
        "L1_t":        best_t,
        "L1_p_bonf":   best_p,
        "L1_delta":    delta,
        "L2_cusum":    cusum_peak,
        "L3_binseg":   binseg_best,
        "L3_best_n":   binseg_res.get("best_n"),
        "L4_pelt":     pelt_breaks,
        "L5_tau":      l5_res["tau"],
        "L5_d_max":    l5_res["d_max"],
        "L5_bic_ok":   l5_res["bic_ok"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def load_futures_eurl(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price_usd_ton"] = (df["Price"].str.replace(",", "", regex=False)
                           .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price_usd_ton"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def load_futures_b7h1(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        _IT_MONTHS = {
            "gen": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
            "mag": "May", "giu": "Jun", "lug": "Jul", "ago": "Aug",
            "set": "Sep", "ott": "Oct", "nov": "Nov", "dic": "Dec",
        }
        def _parse_it_date(s: str) -> pd.Timestamp:
            for it, en in _IT_MONTHS.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse_it_date)
    df["price_usd_ton"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = (df.dropna(subset=["date", "price_usd_ton"])
            .sort_values("date").set_index("date"))
    df = df[~df.index.duplicated(keep="first")]
    print(f"  B7H1: {len(df)} righe  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def build_margin(daily: pd.DataFrame,
                 gasoil_eurl: pd.Series,
                 eurobob_eurl: pd.Series | None) -> pd.DataFrame:
    df = daily[["benzina_net", "gasolio_net"]].copy()
    ws_gas = gasoil_eurl.reindex(df.index, method="ffill")
    df["margin_gasolio"] = df["gasolio_net"] - ws_gas
    if eurobob_eurl is not None:
        ws_benz = eurobob_eurl.reindex(df.index, method="ffill")
        df["margin_benzina"] = df["benzina_net"] - ws_benz
    else:
        import numpy as np
        df["margin_benzina"] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Carica dati pompa ─────────────────────────────────────────────────────
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))

    # ── Carica futures e costruisce il margine ────────────────────────────────
    print("Carico futures e EUR/USD...")
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
    )
    gasoil_eurl  = load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    daily = build_margin(daily, gasoil_eurl, eurobob_eurl)

    print(f"\nDati margine: {daily.index.min().date()} → {daily.index.max().date()}")
    benz_avail = daily["margin_benzina"].dropna()
    if not benz_avail.empty:
        print(f"  margin_benzina: {benz_avail.index.min().date()} → {benz_avail.index.max().date()}")
    print(f"  margin_gasolio: {daily['margin_gasolio'].dropna().index.min().date()} → "
          f"{daily['margin_gasolio'].dropna().index.max().date()}")
    print(f"\nConfigurazione: HALF_WIN={HALF_WIN}g  SEARCH=±{SEARCH}g  STEP={STEP}g")
    print(f"Nota CLT: con HALF_WIN={HALF_WIN} e φ≈0.3 → n_eff≈{int(HALF_WIN*0.54)} "
          f"({'OK' if int(HALF_WIN*0.54)>=30 else 'approssimato, considera HALF_WIN≥60'})\n")

    all_results: dict = {}

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        available = daily[(daily.index >= ev["pre_start"]) &
                          (daily.index <= ev["post_end"])]
        if available.empty:
            print(f"⚠  {ev_name}: nessun dato disponibile, salto.")
            continue

        print(f"{'═'*70}")
        print(f"  EVENTO: {ev_name}  (shock={shock.date()})")
        print(f"  Finestra: {ev['pre_start'].date()} → {ev['post_end'].date()}")
        print(f"{'═'*70}")

        fig = plt.figure(figsize=(18, 18))
        fig.suptitle(
            f"Change-point detection sul MARGINE – {ev_name}\n"
            f"Shock: {ev['label'].replace(chr(10), ' ')}",
            fontsize=13, fontweight="bold"
        )

        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.30)

        ev_results: dict = {}

        for col_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            if col_name not in daily.columns:
                print(f"  Colonna {col_name} non trovata, salto.")
                continue

            series = daily[col_name].dropna()

            # Benzina: salta evento se non ci sono dati nel range
            win_check = series[(series.index >= ev["pre_start"]) &
                               (series.index <= ev["post_end"])]
            if len(win_check) < 2 * HALF_WIN:
                print(f"  [{fuel_key}] dati insufficienti nel range "
                      f"(n={len(win_check)}), salto.")
                continue

            ax_price = fig.add_subplot(gs[0, col_idx])
            ax_cusum = fig.add_subplot(gs[1, col_idx])
            ax_tstat = fig.add_subplot(gs[2, col_idx])
            ax_bic   = fig.add_subplot(gs[3, col_idx])
            ax_l5    = fig.add_subplot(gs[4, col_idx])

            ax_price.set_title(
                f"{fuel_key.capitalize()} – {ev['label'].replace(chr(10),' ')}",
                fontsize=9, fontweight="bold", pad=4
            )

            res = plot_event_fuel(
                ev_name, ev, series,
                fuel_label=fuel_key.capitalize(),
                fuel_color=fuel_color,
                ax_price=ax_price,
                ax_cusum=ax_cusum,
                ax_tstat=ax_tstat,
                ax_bic=ax_bic,
                ax_l5=ax_l5,
            )
            ev_results[fuel_key] = res

            if res:
                print(f"\n  [{fuel_key.upper()}]")
                print(f"    L1 τ={res['L1_tau'].date()}  Δ={res['L1_delta']:+.4f} €/L"
                      f"  p_bonf={res['L1_p_bonf']:.3f}"
                      f"  {'★ significativo' if res['L1_p_bonf']<0.05 else '– non sig.'}")
                print(f"    L2 CUSUM peak = {res['L2_cusum'].date()}")
                print(f"    L3 BinSeg ({res['L3_best_n']} break ottimale): "
                      + (", ".join(str(d.date()) for d in res['L3_binseg']) or "nessuno"))
                print(f"    L4 PELT ({len(res['L4_pelt'])} break): "
                      + (", ".join(str(d.date()) for d in res['L4_pelt']) or "nessuno"))
                bic_tag = "✓ BIC ok" if res.get("L5_bic_ok") else "⚠ non confermato BIC"
                print(f"    L5 Window-L2 τ={res['L5_tau'].date()}"
                      f"  d_max={res['L5_d_max']:.4f}  {bic_tag}")

        all_results[ev_name] = ev_results

        out = OUT_DIR / f"cp_margin_{ev_name.replace(' ','_').replace('/','')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}\n")

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("RIEPILOGO CHANGE POINT MARGINE – tutti gli eventi")
    print(f"{'═'*70}")
    print(f"{'Evento':<28} {'Carb.':<10} {'L1 τ':<13} {'Δ €/L':>8} "
          f"{'p_bonf':>8} {'L5 τ (window-L2)':<15} {'L5 BIC'}")
    print("-"*90)
    for ev_name, fuels in all_results.items():
        for fuel, res in fuels.items():
            if not res:
                continue
            pelt_str = ", ".join(str(d.date()) for d in res["L4_pelt"]) or "—"
            l5_str   = str(res.get("L5_tau", shock).date()) if res.get("L5_tau") else "—"
            bic_str  = "✓" if res.get("L5_bic_ok") else "⚠"
            print(f"{ev_name:<28} {fuel:<10} {str(res['L1_tau'].date()):<13} "
                  f"{res['L1_delta']:>+8.4f} {res['L1_p_bonf']:>8.3f}  "
                  f"{l5_str:<15} {bic_str}")


if __name__ == "__main__":
    main()