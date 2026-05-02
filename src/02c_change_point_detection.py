#!/usr/bin/env python3
"""
02c_change_point_detection.py
==============================
Change-point detection sul MARGINE (prezzo netto pompa − futures €/L)
benzina e gasolio in corrispondenza di eventi geopolitici rilevanti.

Output primario
───────────────
  theta_results.csv — una riga per evento × carburante con:
    θ (break canonico via metodo del paper), d_max, theta_confirmed.
  Questo file è la fonte di verità consumata dagli script ITS downstream
  quando operano in modalità detected.

Metodo canonico — θ via Window Discrepancy L2  (Paper BLOCCO 1, Eq. 1–2)
──────────────────────────────────────────────────────────────────────────
  Pre-processing: moving average a 7 giorni sulla serie del margine.

  Per ogni candidato v ∈ [shock−SEARCH, shock+SEARCH]:
    d(y_{uv}, y_{vw}) = c(y_{uw}) − c(y_{uv}) − c(y_{vw})
  dove c(y_I) = Σ_{t∈I} ||y_t − ȳ||² (costo L2 intra-finestra).

  θ = argmax_v  d(y_{uv}, y_{vw})

  La discrepanza d misura quanto v divide la finestra in due segmenti
  omogenei internamente ma diversi tra loro.  Picco di d → break ottimale.

  theta_confirmed = True se θ cade entro ±7 giorni da almeno un break
  rilevato da L3 (BinSeg BIC) o L4 (PELT).  Flag diagnostico, non bloccante.

Metodi di supporto (cross-validazione, non producono θ)
────────────────────────────────────────────────────────
  L1 · Sliding-window Welch t-test con correzione AR(1) e Bonferroni
  L2 · CUSUM delle deviazioni dalla media pre-evento
  L3 · Binary Segmentation (BinSeg, costo RBF, selezione BIC)
  L4 · PELT – Pruned Exact Linear Time (penalizzazione BIC)

  L1–L4 sono visualizzati nel grafico diagnostico come evidenza di
  convergenza, ma non sostituiscono θ.

Dipendenze:
  pip install ruptures statsmodels
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
# θ — Window Discrepancy L2  (Paper BLOCCO 1, Eq. 1–2)
# Metodo canonico per la stima del break point.
# ══════════════════════════════════════════════════════════════════════════════

def _l2_cost(y: np.ndarray) -> float:
    """
    Costo intra-finestra L2 (Paper, Eq. 2):
        c(y_I) = Σ_{t∈I} ||y_t − ȳ||²
    Misura la dispersione interna al segmento I.
    """
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def detect_theta(
    series: pd.Series,
    shock: pd.Timestamp,
    search: int    = SEARCH,
    ma_window: int = 7,
    min_seg: int   = 14,
) -> dict:
    """
    Stima il break canonico θ via massimizzazione della discrepanza L2
    tra due finestre contigue (Paper BLOCCO 1, Eq. 1–2).

    Pre-processing: moving average a `ma_window` giorni (default 7) per
    ridurre il rumore giornaliero prima del calcolo della discrepanza.

    Per ogni candidato v ∈ [u + min_seg, w − min_seg]:
        d(y_{uv}, y_{vw}) = c(y_{uw}) − c(y_{uv}) − c(y_{vw})

    θ = argmax_v  d(...)

    Il flag `theta_confirmed` è impostato dal chiamante dopo aver eseguito
    L3/L4: True se θ cade entro ±7 giorni da almeno un break BIC.
    È un indicatore diagnostico di convergenza, non una condizione bloccante.

    Returns
    -------
    dict:
      theta            – pd.Timestamp: break canonico
      d_max            – float: valore massimo della discrepanza
      d_values         – pd.Series: curva d(v) su tutte le date candidate
      ma_series        – pd.Series: serie dopo smoothing MA
      theta_confirmed  – bool: inizialmente False, aggiornato in plot_event_fuel
    """
    mask = (series.index >= shock - pd.Timedelta(days=search)) & \
           (series.index <= shock + pd.Timedelta(days=search))
    win = series[mask].dropna()

    _fallback = {
        "theta": shock, "d_max": 0.0,
        "d_values": pd.Series(dtype=float),
        "ma_series": pd.Series(dtype=float),
        "theta_confirmed": False,
    }

    if len(win) < 2 * min_seg + ma_window:
        return _fallback

    ma = win.rolling(ma_window, center=True, min_periods=1).mean()
    y  = ma.values
    n  = len(y)
    c_uw = _l2_cost(y)

    d_vals:     list[float]          = []
    cand_dates: list[pd.Timestamp]   = []

    for v in range(min_seg, n - min_seg):
        d = c_uw - _l2_cost(y[:v]) - _l2_cost(y[v:])
        d_vals.append(d)
        cand_dates.append(ma.index[v])

    if not cand_dates:
        return {**_fallback, "ma_series": ma}

    d_series = pd.Series(d_vals, index=cand_dates)
    theta    = d_series.idxmax()

    return {
        "theta":           theta,
        "d_max":           float(d_series.max()),
        "d_values":        d_series,
        "ma_series":       ma,
        "theta_confirmed": False,   # aggiornato in plot_event_fuel
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
    ax_theta:  plt.Axes,   # pannello 1: θ discrepanza L2  (canonico)
    ax_price:  plt.Axes,   # pannello 2: serie margine + tutte le linee break
    ax_cusum:  plt.Axes,   # pannello 3: CUSUM
    ax_tstat:  plt.Axes,   # pannello 4: sliding t-stat (L1, supporto)
    ax_bic:    plt.Axes,   # pannello 5: BIC curve BinSeg (L3, supporto)
) -> dict:
    """
    Disegna i 5 pannelli diagnostici per un singolo carburante.

    Pannello 1 — θ canonico (metodo del paper):
      Curva d(v), picco = θ.  L3/L4 annotati per convergenza.

    Pannelli 2–5 — metodi di supporto L1–L4:
      Contestualizzano θ; non producono un break alternativo.

    Restituisce dict con chiave primaria `theta` e chiavi secondarie L1–L4.
    """

    win = series[(series.index >= ev["pre_start"]) &
                 (series.index <= ev["post_end"])].dropna()
    if len(win) < 2 * HALF_WIN:
        ax_theta.set_title(f"{fuel_label} – dati insufficienti", fontsize=9)
        return {}

    shock = ev["shock"]
    color = fuel_color

    # ── θ: metodo canonico del paper ─────────────────────────────────────────
    theta_res = detect_theta(win, shock, search=SEARCH)

    # ── L1: sliding t-test (supporto) ────────────────────────────────────────
    ttest_df = sliding_ttest(win, shock)
    l1_tau   = ttest_df.iloc[0]["tau"]        if not ttest_df.empty else shock
    l1_t     = ttest_df.iloc[0]["t_stat"]     if not ttest_df.empty else 0.0
    l1_p     = ttest_df.iloc[0]["p_bonf"]     if not ttest_df.empty else 1.0
    l1_delta = ttest_df.iloc[0]["delta_mean"] if not ttest_df.empty else 0.0

    # ── L2: CUSUM (supporto) ─────────────────────────────────────────────────
    cs, cusum_peak = cusum(win, shock)

    # ── L3: BinSeg (supporto + conferma θ) ───────────────────────────────────
    binseg_res  = binseg_detect(win)
    binseg_best = binseg_res.get("best", [])

    # ── L4: PELT (supporto + conferma θ) ─────────────────────────────────────
    pelt_breaks = pelt_detect(win)

    # ── Aggiorna theta_confirmed ──────────────────────────────────────────────
    all_bic_breaks = binseg_best + pelt_breaks
    theta = theta_res["theta"]
    if all_bic_breaks and not theta_res["d_values"].empty:
        theta_res["theta_confirmed"] = any(
            abs((theta - b).days) <= 7 for b in all_bic_breaks
        )

    confirmed = theta_res["theta_confirmed"]
    conf_tag  = "✓ confermato L3/L4" if confirmed else "⚠ non in accordo L3/L4"

    # ════ PANNELLO 1: θ — curva discrepanza L2  (canonico) ════════════════════
    d_vals = theta_res["d_values"]
    ma_ser = theta_res["ma_series"]

    if not d_vals.empty:
        ax_theta.plot(d_vals.index, d_vals.values, color=color, lw=1.1,
                      label="d(y_uv, y_vw)")
        ax_theta.axhline(0, color="grey", lw=0.6, ls="-")
        ax_theta.axvline(shock, color=ev["color"], lw=1.5, ls="--", label="Shock")
        ax_theta.axvline(theta, color=color, lw=2.0, ls="-",
                         label=f"θ = {theta.date()}  ({conf_tag})")
        ax_theta.scatter([theta], [d_vals.max()], color=color, s=80,
                         zorder=5, marker="*")
        # Annota anche i break L3/L4 sulla stessa curva per leggere la convergenza
        for i, b in enumerate(pelt_breaks):
            if b in d_vals.index or d_vals.index.min() <= b <= d_vals.index.max():
                ax_theta.axvline(b, color="green", lw=0.9, ls=":",
                                 alpha=0.7, label="L4 PELT" if i == 0 else "")
        for i, b in enumerate(binseg_best):
            if d_vals.index.min() <= b <= d_vals.index.max():
                ax_theta.axvline(b, color="purple", lw=0.9, ls=":",
                                 alpha=0.7, label="L3 BinSeg" if i == 0 else "")
        ax_theta.set_ylabel("d (€/L)²", fontsize=8)
        ax_theta.legend(fontsize=6, loc="upper left", ncol=2)
        conf_color = "#2ecc71" if confirmed else "#e67e22"
        ax_theta.set_title(
            f"θ = {theta.date()}   d_max = {theta_res['d_max']:.4f}   {conf_tag}",
            fontsize=8, fontweight="bold", pad=3, color=conf_color,
        )
    else:
        ax_theta.text(0.5, 0.5, "θ: dati insufficienti",
                      ha="center", va="center", transform=ax_theta.transAxes, fontsize=8)
    ax_theta.grid(alpha=0.25)
    ax_theta.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 2: serie margine + zoom intorno a θ  (tutte le linee) ═════
    zoom_start = theta - pd.Timedelta(days=ZOOM_WIN)
    zoom_end   = theta + pd.Timedelta(days=ZOOM_WIN)
    win_zoom   = win[(win.index >= zoom_start) & (win.index <= zoom_end)]

    ax_price.plot(win_zoom.index, win_zoom.values, color=color, lw=0.9, label=fuel_label)
    ax_price.axvline(shock,  color=ev["color"], lw=1.5, ls="--", label="Evento")
    ax_price.axvline(theta,  color=color, lw=2.0, ls="-",
                     alpha=0.9, label=f"θ={theta.date()}")
    ax_price.axvline(l1_tau, color="steelblue", lw=1.0, ls=":",
                     alpha=0.7, label=f"L1 τ={l1_tau.date()}")
    ax_price.axvline(cusum_peak, color="darkorange", lw=1.0, ls="-.",
                     alpha=0.7, label=f"L2 CUSUM={cusum_peak.date()}")
    for i, b in enumerate(pelt_breaks):
        if zoom_start <= b <= zoom_end:
            ax_price.axvline(b, color="green", lw=0.8, ls=":", alpha=0.7,
                             label="L4 PELT" if i == 0 else "")
    for i, b in enumerate(binseg_best):
        if zoom_start <= b <= zoom_end:
            ax_price.axvline(b, color="purple", lw=0.8, ls=":", alpha=0.7,
                             label="L3 BinSeg" if i == 0 else "")
    ax_price.set_xlim(zoom_start, zoom_end)
    ax_price.set_ylabel("Margine (€/L)", fontsize=8)
    ax_price.set_title(
        f"{fuel_label} – zoom ±{ZOOM_WIN}gg attorno a θ  (supporto L1–L4)",
        fontsize=8, pad=3,
    )
    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
    ax_price.grid(axis="y", alpha=0.25)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ════ PANNELLO 3: CUSUM (L2 supporto) ════════════════════════════════════
    ax_cusum.plot(cs.index, cs.values, color=color, lw=0.9)
    ax_cusum.axhline(0, color="grey", lw=0.7, ls="--")
    ax_cusum.axvline(shock,      color=ev["color"],  lw=1.5, ls="--")
    ax_cusum.axvline(theta,      color=color,        lw=1.8, ls="-",  alpha=0.8,
                     label=f"θ={theta.date()}")
    ax_cusum.axvline(cusum_peak, color="darkorange",  lw=1.0, ls="-.", alpha=0.7,
                     label=f"L2 CUSUM={cusum_peak.date()}")
    ax_cusum.fill_between(cs.index, cs.values, 0,
                          where=cs.values > 0, alpha=0.15, color=color)
    ax_cusum.fill_between(cs.index, cs.values, 0,
                          where=cs.values < 0, alpha=0.15, color="green")
    ax_cusum.set_ylabel("CUSUM margine (€/L)", fontsize=8)
    ax_cusum.set_title("L2 — CUSUM (supporto)", fontsize=8, pad=3)
    ax_cusum.legend(fontsize=6)
    ax_cusum.grid(axis="y", alpha=0.25)
    ax_cusum.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 4: sliding t-stat L1 (supporto) ═══════════════════════════
    if not ttest_df.empty:
        rising = ttest_df[ttest_df["delta_mean"] > 0]
        ax_tstat.plot(rising["tau"], rising["t_stat"].abs(),
                      color=color, lw=0.9, label="Δ > 0")
        ax_tstat.axvline(shock,  color=ev["color"], lw=1.5, ls="--", label="Evento")
        ax_tstat.axvline(theta,  color=color, lw=1.8, ls="-", alpha=0.8,
                         label=f"θ={theta.date()}")
        if not rising.empty and l1_tau in rising["tau"].values:
            ax_tstat.axvline(l1_tau, color="steelblue", lw=1.0, ls=":",
                             label=f"L1 τ={l1_tau.date()}")
        ax_tstat.axhline(stats.t.ppf(0.975, df=HALF_WIN * 2 - 2),
                         color="grey", lw=0.7, ls=":", label="α=0.05")
        ax_tstat.set_ylabel("|t-stat|  (Δ>0)", fontsize=8)
        sig = "★" if l1_p < 0.05 else ""
        ax_tstat.set_title(
            f"L1 — sliding t-test (supporto)  |  τ={l1_tau.date()}  "
            f"Δ={l1_delta:+.4f} €/L  p_bonf={l1_p:.3f}{sig}",
            fontsize=7, pad=2,
        )
        ax_tstat.legend(fontsize=6)
    ax_tstat.grid(axis="y", alpha=0.25)
    ax_tstat.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 5: BIC curve BinSeg (L3 supporto) ══════════════════════════
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
            f"L3 BinSeg (supporto): {len(binseg_best)} break  |  "
            f"L4 PELT: {len(pelt_breaks)} break  →  "
            + (", ".join(str(d.date()) for d in pelt_breaks) if pelt_breaks else "nessuno"),
            fontsize=7, pad=2,
        )
    else:
        ax_bic.text(0.5, 0.5, "ruptures non disponibile",
                    ha="center", va="center", transform=ax_bic.transAxes, fontsize=8)
    ax_bic.grid(alpha=0.25)

    return {
        # ── Output primario ──────────────────────────────────────────────────
        "theta":             theta,
        "d_max":             theta_res["d_max"],
        "theta_confirmed":   confirmed,
        # ── Metodi di supporto ───────────────────────────────────────────────
        "L1_tau":            l1_tau,
        "L1_t":              l1_t,
        "L1_p_bonf":         l1_p,
        "L1_delta":          l1_delta,
        "L2_cusum":          cusum_peak,
        "L3_binseg":         binseg_best,
        "L3_best_n":         binseg_res.get("best_n"),
        "L4_pelt":           pelt_breaks,
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
    print(f"\nConfigurazione: HALF_WIN={HALF_WIN}g  SEARCH=±{SEARCH}g  STEP={STEP}g\n")

    all_results: dict = {}
    theta_rows:  list = []

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

        # ── Figura: 5 righe × 2 colonne  (riga 0 = θ canonico) ───────────────
        fig = plt.figure(figsize=(18, 20))
        fig.suptitle(
            f"θ (break canonico) + diagnostica di supporto — {ev_name}\n"
            f"Shock: {ev['label'].replace(chr(10), ' ')}",
            fontsize=13, fontweight="bold",
        )
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.58, wspace=0.30)

        ev_results: dict = {}

        for col_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            if col_name not in daily.columns:
                print(f"  Colonna {col_name} non trovata, salto.")
                continue

            series = daily[col_name].dropna()
            win_check = series[(series.index >= ev["pre_start"]) &
                               (series.index <= ev["post_end"])]
            if len(win_check) < 2 * HALF_WIN:
                print(f"  [{fuel_key}] dati insufficienti nel range "
                      f"(n={len(win_check)}), salto.")
                continue

            ax_theta = fig.add_subplot(gs[0, col_idx])   # θ canonico
            ax_price = fig.add_subplot(gs[1, col_idx])   # zoom margine + break
            ax_cusum = fig.add_subplot(gs[2, col_idx])   # CUSUM
            ax_tstat = fig.add_subplot(gs[3, col_idx])   # L1 sliding t-test
            ax_bic   = fig.add_subplot(gs[4, col_idx])   # L3 BIC curve

            # Etichetta colonna in testa
            ax_theta.set_title(
                f"{fuel_key.capitalize()} — θ (Window Discrepancy L2, metodo paper)",
                fontsize=9, fontweight="bold", pad=4,
            )

            res = plot_event_fuel(
                ev_name, ev, series,
                fuel_label=fuel_key.capitalize(),
                fuel_color=fuel_color,
                ax_theta=ax_theta,
                ax_price=ax_price,
                ax_cusum=ax_cusum,
                ax_tstat=ax_tstat,
                ax_bic=ax_bic,
            )
            ev_results[fuel_key] = res

            if res:
                conf_icon = "✓" if res["theta_confirmed"] else "⚠"
                print(f"\n  [{fuel_key.upper()}]")
                print(f"    θ  = {res['theta'].date()}  "
                      f"d_max={res['d_max']:.4f}  "
                      f"{conf_icon} {'confermato L3/L4' if res['theta_confirmed'] else 'non in accordo L3/L4'}")
                print(f"    L1 = {res['L1_tau'].date()}  Δ={res['L1_delta']:+.4f} €/L  "
                      f"p_bonf={res['L1_p_bonf']:.3f}"
                      f"{'  ★' if res['L1_p_bonf'] < 0.05 else ''}")
                print(f"    L2 CUSUM = {res['L2_cusum'].date()}")
                print(f"    L3 BinSeg ({res['L3_best_n']} break): "
                      + (", ".join(str(d.date()) for d in res["L3_binseg"]) or "nessuno"))
                print(f"    L4 PELT ({len(res['L4_pelt'])} break): "
                      + (", ".join(str(d.date()) for d in res["L4_pelt"]) or "nessuno"))

                # ── Accumula righe per theta_results.csv ─────────────────────
                theta_rows.append({
                    "evento":           ev_name,
                    "shock":            shock.date(),
                    "carburante":       fuel_key,
                    "theta":            res["theta"].date(),
                    "d_max":            round(res["d_max"], 6),
                    "theta_confirmed":  res["theta_confirmed"],
                    "L1_tau":           res["L1_tau"].date(),
                    "L1_delta_eurl":    round(res["L1_delta"], 5),
                    "L1_p_bonf":        round(res["L1_p_bonf"], 4),
                    "L2_cusum":         res["L2_cusum"].date(),
                    "L3_binseg":        "; ".join(str(d.date()) for d in res["L3_binseg"]) or "",
                    "L4_pelt":          "; ".join(str(d.date()) for d in res["L4_pelt"]) or "",
                })

        all_results[ev_name] = ev_results

        out = OUT_DIR / f"cp_margin_{ev_name.replace(' ','_').replace('/','')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}\n")

    # ── Export theta_results.csv ──────────────────────────────────────────────
    if theta_rows:
        theta_csv = OUT_DIR / "theta_results.csv"
        pd.DataFrame(theta_rows).to_csv(theta_csv, index=False, encoding="utf-8-sig")
        print(f"  → Esportato: {theta_csv}")

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("RIEPILOGO θ — break canonici (Window Discrepancy L2)")
    print(f"{'═'*70}")
    print(f"{'Evento':<28} {'Carb.':<10} {'shock':<13} {'θ':<13} "
          f"{'d_max':>8} {'confermato'}")
    print("-" * 80)
    for row in theta_rows:
        conf = "✓" if row["theta_confirmed"] else "⚠"
        print(f"{row['evento']:<28} {row['carburante']:<10} "
              f"{str(row['shock']):<13} {str(row['theta']):<13} "
              f"{row['d_max']:>8.4f}  {conf}")


if __name__ == "__main__":
    main()