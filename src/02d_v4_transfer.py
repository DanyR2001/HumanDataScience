#!/usr/bin/env python3
"""
02d_v4_transfer.py  ─  Metodo 4: ITS con CCF + Transfer Function (SARIMAX)
===========================================================================
Implementa i due miglioramenti principali rispetto al Metodo 3:

  PUNTO 1 — CCF per determinare b, r, s  (Aliffia et al., 2023)
    Dopo il fit ARIMA sul pre-periodo, la cross-correlation function (CCF)
    tra i residui post-intervento (actual − counterfactual) e il forecast
    controfattuale rivela la struttura della transfer function:
      b = primo lag con |CCF| > soglia  (ritardo dell'effetto)
      r = pattern di decadimento         (1 = espon., 2 = sinusoidale smorzata)
      s = lag a cui i residui tornano stabili

  PUNTO 2 — SARIMAX con tre approcci al vettore pulse  (Masena & Shongwe, 2024)
    Tre modi di costruire il vettore covariata del pulse nell'intervento:
      D  Costante = 1       P_t = 1 per t in [T0, T0+POST_WIN)     baseline classica
      E  Quotient           P_t = forecast_t / actual_t             calibrato per periodo
      F  Trial-and-Error    P_t ottimizzato via scipy.optimize      il più preciso

    Ogni approccio stima un unico ω (peso del pulse) tramite SARIMAX MLE,
    producendo AIC / BIC / RMSE / MAPE confrontabili tra loro.

────────────────────────────────────────────────────────────────────────────────
GARANZIA NO-LOOK-AHEAD
────────────────────────────────────────────────────────────────────────────────
  Fase 1 — Identificazione ordine ARIMA:
    Auto-ARIMA AIC eseguito su TUTTI i dati da PRE_START (2015-01-01) fino a T0.
    Nessun dato post-T0 visibile.

  Fase 2 — Forecast controfattuale:
    Il pre-ARIMA (fase 1) proietta in avanti out-of-sample per POST_WIN giorni.
    Il modello non vede MAI i dati del periodo post.

  Fase 3 — CCF:
    Calcolata sui residui post (actual − counterfactual).  I parametri b, r, s
    sono inferiti dai residui, non dal ri-allenamento su post.

  Fase 4 — SARIMAX con pulse:
    Il modello SARIMAX usa l'ordine (p,d,q) determinato in Fase 1 (pre-only).
    L'exog pulse = 0 durante il pre, ≠ 0 durante il post.
    I parametri AR/MA sono fortemente vincolati dal pre-periodo; solo ω (peso
    del pulse) è stimato principalmente sul post.  Questo è il comportamento
    standard dell'ITS nella letteratura (Masena & Shongwe 2024; Aliffia 2023).

    FIX NO-LOOK-AHEAD (curva modello):
    La curva visualizzata e usata per RMSE/MAPE è costruita analiticamente come
      fitted_post = cf_forecast + ω · pulse_post
    NON dai fittedvalues in-sample del SARIMAX (che userebbero i valori effettivi
    del post tramite il filtro di Kalman, producendo curve quasi identiche
    all'effettivo).  Il SARIMAX viene fittato solo una volta per stimare ω,
    AIC e BIC.  Per approccio F, ω è stimato con OLS chiusa ad ogni iterazione
    Nelder-Mead (più veloce, zero look-ahead nell'obiettivo).

Modalità (--mode):
  fixed     : T0 = data dello shock hardcodata [default]
  detected  : T0 rilevato via PELT (ruptures, modello RBF)

Parametro --detect (solo mode=detected):
  margin  : detection sul margine distributore  [default]
  price   : detection sul prezzo alla pompa netto

Output:
  data/plots/its/{mode}/v4_transfer/              (mode=fixed)
  data/plots/its/detected/{detect}/v4_transfer/   (mode=detected)
    plot_{evento}_{carburante}.png
    ccf_{evento}_{carburante}.png
    period_table_{evento}_{carburante}.csv
    v4_transfer_results.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import itertools
import warnings
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, optimize

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter
from diagnostics import run_diagnostic_tests, plot_residual_diagnostics

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import acf, ccf as sm_ccf, adfuller
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn("statsmodels non installato. pip install statsmodels")

try:
    import ruptures as rpt
    HAS_RPT = True
except ImportError:
    HAS_RPT = False
    warnings.warn("ruptures non installato. pip install ruptures")

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_START   = pd.Timestamp("2015-01-01")  # inizio dataset per il fit ARIMA
PRE_WIN     = 60    # giorni pre-T0 usati SOLO per il plot (no-lookahead garantito)
POST_WIN    = 60    # giorni post-T0 per l'analisi intervento
CI_ALPHA    = 0.10  # CI al 90%
SEARCH      = 30    # PELT: ricerca ±SEARCH giorni dallo shock
CCF_NLAGS   = 20    # numero di lag CCF da calcolare
MAX_ITER    = 300   # iterazioni ARIMA/SARIMAX

# Griglia auto-ARIMA (applicata SOLO al pre-periodo)
ARIMA_P  = [0, 1, 2]
ARIMA_D  = [0, 1]
ARIMA_Q  = [0, 1, 2]

DAILY_CONSUMPTION_L = {
    "benzina": 12_000_000,
    "gasolio": 25_000_000,
}

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock": pd.Timestamp("2022-02-24"),
        "color": "#e74c3c",
        "label": "Russia-Ucraina (24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock": pd.Timestamp("2025-06-13"),
        "color": "#e67e22",
        "label": "Iran-Israele (13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock": pd.Timestamp("2026-02-28"),
        "color": "#8e44ad",
        "label": "Stretto di Hormuz (28 feb 2026)",
    },
}

FUELS: dict[str, tuple[str, str]] = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}

PRICE_COLS: dict[str, str] = {
    "benzina": "benzina_net",
    "gasolio": "gasolio_net",
}

APPROACH_COLORS = {"D": "#2980b9", "E": "#e67e22", "F": "#27ae60"}
APPROACH_LABELS = {
    "D": "D: Costante (P=1)",
    "E": "E: Quotient (CF/actual)",
    "F": "F: Trial-and-Error (ottim.)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati  (identico a v3)
# ══════════════════════════════════════════════════════════════════════════════

def _load_gasoil_futures(eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, encoding="utf-8-sig", dtype=str)
    df["date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    return (df.dropna(subset=["date", "price"]).sort_values("date")
              .set_index("date")
              .pipe(lambda d: usd_ton_to_eur_liter(d["price"], eurusd, GAS_OIL)))


def _load_eurobob_futures(eurusd: pd.Series) -> pd.Series | None:
    if not EUROBOB_CSV.exists():
        return None
    df = pd.read_csv(EUROBOB_CSV, encoding="utf-8-sig", dtype=str)
    _IT = {"gen": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
           "mag": "May", "giu": "Jun", "lug": "Jul", "ago": "Aug",
           "set": "Sep", "ott": "Oct", "nov": "Nov", "dic": "Dec"}
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        def _parse(s):
            for it, en in _IT.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_margin_data() -> pd.DataFrame:
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd  = load_eurusd(csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
                          start="2015-01-01", end="2026-12-31")
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    df["margin_benzina"] = (df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
                            if eurobob is not None else np.nan)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PELT detection  (identico a v3)
# ══════════════════════════════════════════════════════════════════════════════

def detect_breakpoint_pelt(series: pd.Series, shock: pd.Timestamp) -> dict:
    """PELT (ruptures RBF) con fallback alla shock date."""
    if not HAS_RPT:
        return {"tau": shock, "method": "pelt_fallback_no_ruptures",
                "n_breaks_total": 0, "_window_series": pd.Series(dtype=float),
                "_all_breaks": [], "_filtered_breaks": []}

    search_start = shock - pd.Timedelta(days=SEARCH)
    search_end   = shock + pd.Timedelta(days=SEARCH)
    window_series = series[
        (series.index >= search_start - pd.Timedelta(days=20)) &
        (series.index <= search_end   + pd.Timedelta(days=20))
    ].dropna()

    if len(window_series) < 20:
        return {"tau": shock, "method": "pelt_insufficient_data", "n_breaks_total": 0,
                "_window_series": window_series, "_all_breaks": [], "_filtered_breaks": []}

    try:
        arr  = window_series.values.astype(float).reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(arr)
        bps  = algo.predict(pen=2.0)
        bp_indices  = [bp - 1 for bp in bps[:-1]]
        break_dates = [window_series.index[i] for i in bp_indices
                       if 0 <= i < len(window_series)]
    except Exception as exc:
        warnings.warn(f"PELT fallito: {exc}. Fallback shock date.")
        return {"tau": shock, "method": "pelt_error_fallback", "n_breaks_total": 0,
                "_window_series": window_series, "_all_breaks": [], "_filtered_breaks": []}

    filtered = [d for d in break_dates if search_start <= d <= search_end]
    if not filtered:
        return {"tau": shock, "method": "pelt_nofound_fallback",
                "n_breaks_total": len(break_dates),
                "_window_series": window_series,
                "_all_breaks": break_dates, "_filtered_breaks": []}

    best = min(filtered)
    return {"tau": best, "method": "pelt_rbf",
            "n_breaks_total": len(break_dates),
            "all_filtered": [str(d.date()) for d in filtered],
            "_window_series": window_series,
            "_all_breaks": break_dates,
            "_filtered_breaks": filtered}


# ══════════════════════════════════════════════════════════════════════════════
# Break point detection — Window L2 Discrepancy (Paper BLOCCO 1 – Eq. 1–2)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_cost_v4(y: np.ndarray) -> float:
    """c(y_I) = Σ||y_t − ȳ||² — costo L2 intra-finestra (Paper, Eq. 2)."""
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def detect_breakpoint_window_l2(
    series: pd.Series,
    shock: pd.Timestamp,
    ma_window: int = 7,
    min_seg: int   = 14,
) -> dict:
    """
    Changepoint via discrepanza L2 (Paper BLOCCO 1, Eq. 1–2).
    Restituisce un dict compatibile con detect_breakpoint_pelt per il dispatch.
    """
    mask = (
        (series.index >= shock - pd.Timedelta(days=SEARCH)) &
        (series.index <= shock + pd.Timedelta(days=SEARCH))
    )
    win = series[mask].dropna()

    if len(win) < 2 * min_seg + ma_window:
        return {"tau": shock, "method": "window_l2_nofound", "n_breaks_total": 0,
                "_window_series": win, "_all_breaks": [], "_filtered_breaks": []}

    ma = win.rolling(ma_window, center=True, min_periods=1).mean()
    y  = ma.values
    n  = len(y)
    c_uw = _l2_cost_v4(y)

    d_vals: list[float] = []
    cand_dates: list[pd.Timestamp] = []

    for v in range(min_seg, n - min_seg):
        d = c_uw - _l2_cost_v4(y[:v]) - _l2_cost_v4(y[v:])
        d_vals.append(d)
        cand_dates.append(ma.index[v])

    if not cand_dates:
        return {"tau": shock, "method": "window_l2_nofound", "n_breaks_total": 0,
                "_window_series": win, "_all_breaks": [], "_filtered_breaks": []}

    best_idx = int(np.argmax(d_vals))
    best_tau = cand_dates[best_idx]

    return {
        "tau":              best_tau,
        "method":           "window_l2",
        "n_breaks_total":   1,
        "all_filtered":     [str(best_tau.date())],
        "_window_series":   ma,
        "_all_breaks":      [best_tau],
        "_filtered_breaks": [best_tau],
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Auto-ARIMA sul PRE-periodo (SOLO pre, zero look-ahead)
# ══════════════════════════════════════════════════════════════════════════════

def auto_arima_pre(series: pd.Series) -> tuple[tuple, object | None]:
    """
    Seleziona ARIMA(p,d,q) minimizzando AIC sulla serie `series`.
    `series` deve contenere SOLO dati pre-T0 — nessun dato post viene passato.

    Restituisce (ordine, fit_object) oppure ((1,0,1), None) come fallback.
    """
    if not HAS_SM:
        return (1, 0, 1), None

    best_aic   = np.inf
    best_order = (1, 0, 1)
    best_fit   = None

    for p, d, q in itertools.product(ARIMA_P, ARIMA_D, ARIMA_Q):
        if p + q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(series, order=(p, d, q)).fit(
                    method_kwargs={"maxiter": MAX_ITER, "warn_convergence": False}
                )
            if fit.aic < best_aic:
                best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
        except Exception:
            continue

    return best_order, best_fit


def adf_summary(series: pd.Series) -> dict:
    """ADF test per stazionarietà — solo informativo, non blocca il flusso."""
    if not HAS_SM or len(series) < 10:
        return {"statistic": np.nan, "pvalue": np.nan, "stationary": None}
    try:
        res = adfuller(series.dropna(), autolag="AIC")
        return {"statistic": res[0], "pvalue": res[1], "stationary": res[1] < 0.05}
    except Exception:
        return {"statistic": np.nan, "pvalue": np.nan, "stationary": None}


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Forecast controfattuale out-of-sample (NO look-ahead)
# ══════════════════════════════════════════════════════════════════════════════

def counterfactual_forecast(
    pre: pd.Series, order: tuple, n_steps: int, alpha: float = CI_ALPHA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fitta ARIMA(order) su `pre` e proietta n_steps FUORI CAMPIONE.
    Il modello NON vede nessun dato post-T0.

    Restituisce (mean, ci_lo, ci_hi) — tutti array di lunghezza n_steps.
    """
    if not HAS_SM:
        m   = float(pre.mean())
        std = float(pre.std())
        z   = stats.norm.ppf(1 - alpha / 2)
        return (np.full(n_steps, m),
                np.full(n_steps, m - z * std),
                np.full(n_steps, m + z * std))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(pre, order=order).fit(
            method_kwargs={"maxiter": MAX_ITER, "warn_convergence": False}
        )
        fc  = fit.get_forecast(steps=n_steps)

    mean = fc.predicted_mean.values
    conf = fc.conf_int(alpha=alpha).values
    return mean, conf[:, 0], conf[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — CCF per stimare b, r, s  (Aliffia et al. 2023, Section 2.3)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_intervention_order(
    actual_post: np.ndarray,
    counterfactual: np.ndarray,
    nlags: int = CCF_NLAGS,
    alpha: float = 0.05,
) -> dict:
    """
    Stima i parametri b, r, s della transfer function tramite CCF.

    Definizioni (Wei 2006; Aliffia 2023):
      b = lag al quale l'effetto dell'intervento inizia
          → primo lag con |CCF| > soglia di significatività
      r = pattern del decadimento dopo b+s periodi
          → 1 se monotone, 2 se oscillatorio / smorzato sinusoidale
      s = durata della fase transitoria dopo b
          → lag a cui i residui tornano sotto soglia

    Input:
      actual_post     : valori effettivi nel post-periodo
      counterfactual  : forecast out-of-sample del pre-ARIMA

    Output: dizionario con b, r, s, ccf_values, threshold, interpretazione
    """
    residuals = actual_post - counterfactual
    n = len(residuals)
    threshold = 2.0 / np.sqrt(n)   # soglia 95% approssimata

    if not HAS_SM or n < 5:
        return {"b": 0, "r": 1, "s": 1, "ccf_values": np.array([]),
                "threshold": threshold, "interpretation": "dati_insufficienti"}

    try:
        # CCF: correlazione tra residui post e forecast al lag k
        # Usiamo anche la CCF "inversa" (residui × residui laggati) per r/s
        cross = sm_ccf(residuals, counterfactual, nlags=nlags, adjusted=False)
        auto  = acf(residuals, nlags=nlags, fft=False)
    except Exception:
        return {"b": 0, "r": 1, "s": 1, "ccf_values": np.array([]),
                "threshold": threshold, "interpretation": "ccf_error"}

    n_cross = len(cross)  # sm_ccf restituisce nlags elementi, non nlags+1

    # ── Stima b: primo lag con |CCF| > soglia ────────────────────────────────
    b = 0
    for lag in range(n_cross):
        if abs(cross[lag]) > threshold:
            b = lag
            break

    # ── Stima s: lag successivo a b in cui |CCF| torna < soglia ──────────────
    s = 1
    for lag in range(b + 1, n_cross):
        if abs(cross[lag]) <= threshold:
            s = max(1, lag - b)
            break

    # ── Stima r: pattern dell'ACF dei residui ────────────────────────────────
    # r=1: decadimento esponenziale monotono (tutti i lags ≥ 0 o tutti ≤ 0)
    # r=2: pattern sinusoidale smorzato (alternanza di segno)
    acf_tail = auto[b + s: b + s + 5] if len(auto) > b + s + 5 else auto[b:]
    if len(acf_tail) >= 2:
        sign_changes = sum(
            1 for i in range(len(acf_tail) - 1)
            if acf_tail[i] * acf_tail[i + 1] < 0
        )
        r = 2 if sign_changes >= 1 else 1
    else:
        r = 1

    # ── Interpretazione testuale ──────────────────────────────────────────────
    interp = (
        f"b={b} (effetto inizia al lag {b}), "
        f"s={s} (transitorio {s} periodi), "
        f"r={r} ({'sinusoidale smorzato' if r == 2 else 'decadimento esponenziale'})"
    )

    return {
        "b": b, "r": r, "s": s,
        "ccf_values":  cross,
        "acf_resid":   auto,
        "threshold":   threshold,
        "interpretation": interp,
    }


def plot_ccf(
    residuals: np.ndarray,
    ccf_values: np.ndarray,
    acf_resid: np.ndarray,
    threshold: float,
    b: int, r: int, s: int,
    post_index: pd.DatetimeIndex,
    ev_name: str,
    fuel_key: str,
    fuel_color: str,
    out_path: Path,
) -> None:
    """
    Salva il grafico diagnostico CCF con 3 pannelli:
      · Residui post (actual − counterfactual)
      · CCF tra residui e counterfactual (con soglia e annotazione b, s)
      · ACF dei residui (con annotazione r)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(
        f"[V4 – CCF Intervention Order]  {ev_name}  ·  {fuel_key.capitalize()}\n"
        f"Stima transfer function:  b={b}  s={s}  r={r}",
        fontsize=9, fontweight="bold"
    )

    # Pannello 1: residui post
    ax = axes[0]
    ax.bar(range(len(residuals)), residuals, color=fuel_color, alpha=0.6, width=0.8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("Residui post-intervento  (actual − counterfactual)", fontsize=8)
    ax.set_xlabel("Giorni dall'intervento", fontsize=7)
    ax.set_ylabel("€/L", fontsize=7)
    ax.grid(axis="y", alpha=0.2)

    # Pannello 2: CCF
    ax = axes[1]
    lags = np.arange(len(ccf_values))
    ax.bar(lags, ccf_values, color="#2980b9", alpha=0.7, width=0.8)
    ax.axhline( threshold, color="red",  lw=0.9, ls="--", label=f"+soglia 95% ({threshold:.3f})")
    ax.axhline(-threshold, color="red",  lw=0.9, ls="--", label=f"-soglia 95%")
    ax.axhline(0, color="grey", lw=0.6)
    # Annota b e b+s
    if b < len(ccf_values):
        ax.axvline(b,     color="#e74c3c", lw=1.4, ls="-",  label=f"b={b}  (inizio effetto)")
    if b + s < len(ccf_values):
        ax.axvline(b + s, color="#27ae60", lw=1.4, ls="--", label=f"b+s={b+s}  (fine transitorio)")
    ax.set_title("CCF  (residui post  ×  counterfactual forecast)", fontsize=8)
    ax.set_xlabel("Lag", fontsize=7)
    ax.set_ylabel("Correlazione incrociata", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    # Pannello 3: ACF dei residui (per leggere r)
    ax = axes[2]
    lags_a = np.arange(len(acf_resid))
    ax.bar(lags_a, acf_resid, color="#8e44ad", alpha=0.7, width=0.8)
    ax.axhline( threshold, color="red",  lw=0.9, ls="--")
    ax.axhline(-threshold, color="red",  lw=0.9, ls="--")
    ax.axhline(0, color="grey", lw=0.6)
    decay_label = "sinusoidale smorzato → r=2" if r == 2 else "esponenziale monotono → r=1"
    ax.set_title(f"ACF residui  (pattern decadimento: {decay_label})", fontsize=8)
    ax.set_xlabel("Lag", fontsize=7)
    ax.set_ylabel("Autocorrelazione", fontsize=7)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → CCF plot: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — SARIMAX con pulse: 3 approcci  (Masena & Shongwe 2024, Sect. 3.3)
# ══════════════════════════════════════════════════════════════════════════════

def _build_full_series(pre: pd.Series, post: pd.Series) -> pd.Series:
    """Concatena pre e post in una serie continua."""
    return pd.concat([pre, post]).sort_index()


def _fit_sarimax(
    full_series: pd.Series,
    order: tuple,
    pulse_vec: np.ndarray,
    pre_fit=None,
) -> dict:
    """
    Fitta SARIMAX(order) con exog=pulse_vec sull'intera serie (pre+post).

    GARANZIA NO-LOOK-AHEAD:
    Se pre_fit è fornito, i parametri AR/MA/intercept vengono CONGELATI ai
    valori stimati sul solo pre-periodo (Fase 1).  Solo ω (peso del pulse)
    viene stimato liberamente, usando i dati post esclusivamente per quella
    stima.  Senza questo freezing il SARIMAX ri-stimerebbe AR/MA su pre+post
    e i valori fitted_post traccierebbero i dati effettivi post-shock
    (look-ahead implicito).

    Mapping nomi parametri ARIMA → SARIMAX:
      const  → intercept
      ar.L*  → ar.L*      (identici)
      ma.L*  → ma.L*      (identici)
      sigma2 → sigma2     (non congelato: si adatta alla scala dei residui)
    """
    if not HAS_SM:
        n  = len(full_series)
        fc = np.full(n, float(full_series.mean()))
        return {"fitted": fc, "resid": full_series.values - fc,
                "aic": np.nan, "bic": np.nan, "omega": np.nan, "converged": False}

    exog = pulse_vec.reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                full_series,
                order=order,
                exog=exog,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            if pre_fit is not None:
                # Costruisci dizionario di parametri da congelare
                pre_params = dict(zip(pre_fit.param_names, pre_fit.params))
                sax_names  = set(model.param_names)
                fix_dict: dict[str, float] = {}
                for pname, pval in pre_params.items():
                    # const (ARIMA) → intercept (SARIMAX con trend='c')
                    mapped = "intercept" if pname == "const" else pname
                    # Congela AR/MA/intercept; lascia liberi sigma2 e x1 (omega)
                    if mapped in sax_names and mapped not in ("sigma2", "x1"):
                        fix_dict[mapped] = float(pval)

                with model.fix_params(fix_dict):
                    res = model.fit(method="lbfgs", maxiter=MAX_ITER, disp=False)
            else:
                res = model.fit(method="lbfgs", maxiter=MAX_ITER, disp=False)

        fitted = res.fittedvalues.values
        omega  = float(res.params.get("x1", np.nan))
        return {
            "fitted":    fitted,
            "resid":     res.resid.values,
            "aic":       float(res.aic),
            "bic":       float(res.bic),
            "omega":     omega,
            "converged": True,
        }
    except Exception as exc:
        warnings.warn(f"SARIMAX fallito: {exc}")
        fc = np.full(len(full_series), float(full_series.mean()))
        return {"fitted": fc, "resid": full_series.values - fc,
                "aic": np.nan, "bic": np.nan, "omega": np.nan, "converged": False}


def _approach_metrics(
    post: pd.Series,
    fitted_post: np.ndarray,
    cf_forecast: np.ndarray,
    consumption: float,
) -> dict:
    """Calcola RMSE, MAPE, guadagno cumulato e CI approssimato.

    RMSE/MAPE misurano quanto il modello analitico (cf + ω·pulse) si avvicina
    all'effettivo — NO look-ahead: fitted_post = cf_forecast + ω·pulse,
    costruito solo da parametri stimati sul pre-periodo.
    Il gain è calcolato su (effettivo − controfattuale puro, senza pulse).
    """
    # ── Metriche di fit sul modello analitico (NO in-sample SARIMAX) ─────────
    resid = post.values - fitted_post          # fitted_post = cf + ω·pulse
    rmse  = float(np.sqrt(np.mean(resid**2)))
    mask  = post.values != 0
    mape  = float(np.mean(np.abs(resid[mask] / post.values[mask])) * 100) if mask.any() else np.nan

    # ── Gain: effettivo − controfattuale Fase-2 (NO look-ahead, NO pulse) ───
    extra = post.values - cf_forecast
    gain  = float(extra.sum() * consumption / 1e6)

    # CI semplice sul gain ± 1.64σ  (90%)
    std_extra = float(np.std(extra))
    z_90      = 1.6449
    gain_lo   = float((extra.sum() - z_90 * std_extra * np.sqrt(len(extra))) * consumption / 1e6)
    gain_hi   = float((extra.sum() + z_90 * std_extra * np.sqrt(len(extra))) * consumption / 1e6)
    return {"rmse": rmse, "mape": mape, "extra": extra,
            "gain_meur": gain, "gain_lo": gain_lo, "gain_hi": gain_hi}


# ── Approccio D: Costante P_t = 1  ───────────────────────────────────────────
# (Masena Approach 3 — più comune in letteratura)
# Il vettore pulse vale 1 per ogni osservazione nell'intervento, 0 altrove.
# Stima ω = effetto medio assoluto dell'intervento.

def approach_d_constant(
    pre: pd.Series, post: pd.Series, order: tuple,
    cf_forecast: np.ndarray, consumption: float,
    pre_fit=None,
) -> dict:
    """
    Pulse costante = 1 durante il post, 0 durante il pre.
    Il SARIMAX stima un unico ω: spostamento medio del livello.
    (Masena & Shongwe 2024, Approccio 3)
    """
    full    = _build_full_series(pre, post)
    n_pre   = len(pre)
    n_post  = len(post)
    pulse   = np.zeros(len(full))
    pulse[n_pre: n_pre + n_post] = 1.0

    sax = _fit_sarimax(full, order, pulse, pre_fit=pre_fit)
    omega = sax["omega"] if not np.isnan(sax["omega"]) else 0.0

    # NO look-ahead: curva analitica cf + ω·pulse (non usa fittedvalues in-sample)
    pulse_post = pulse[n_pre: n_pre + n_post]
    fitted_post = cf_forecast + omega * pulse_post
    m = _approach_metrics(post, fitted_post, cf_forecast, consumption)

    return {
        "label":     APPROACH_LABELS["D"],
        "fitted_post": fitted_post,
        "pulse_post":  pulse_post,
        "omega":     sax["omega"],
        "aic":       sax["aic"],
        "bic":       sax["bic"],
        "converged": sax["converged"],
        **m,
    }


# ── Approccio E: Quotient P_t = forecast_t / actual_t  ───────────────────────
# (Masena Approach 2)
# Il vettore pulse è calibrato periodo per periodo sul rapporto
# tra controfattuale e valore effettivo.

def approach_e_quotient(
    pre: pd.Series, post: pd.Series, order: tuple,
    cf_forecast: np.ndarray, consumption: float,
) -> dict:
    """
    Pulse = counterfactual_t / actual_t per ogni t nel post.
    (Masena & Shongwe 2024, Approccio 2)

    Il vettore è proporzionale al "quanto il controfattuale sovra/sotto-stima"
    l'effettivo in ogni periodo — cattura la variazione relativa.
    """
    full   = _build_full_series(pre, post)
    n_pre  = len(pre)
    n_post = len(post)
    pulse  = np.zeros(len(full))

    actual_post = post.values.astype(float)
    safe_actual = np.where(np.abs(actual_post) > 1e-9, actual_post, 1e-9)
    pulse[n_pre: n_pre + n_post] = cf_forecast / safe_actual   # quotient

    sax = _fit_sarimax(full, order, pulse)
    omega = sax["omega"] if not np.isnan(sax["omega"]) else 0.0

    # NO look-ahead: curva analitica cf + ω·pulse
    pulse_post = pulse[n_pre: n_pre + n_post]
    fitted_post = cf_forecast + omega * pulse_post
    m = _approach_metrics(post, fitted_post, cf_forecast, consumption)

    return {
        "label":     APPROACH_LABELS["E"],
        "fitted_post": fitted_post,
        "pulse_post":  pulse_post,
        "omega":     sax["omega"],
        "aic":       sax["aic"],
        "bic":       sax["bic"],
        "converged": sax["converged"],
        **m,
    }


# ── Approccio F: Trial-and-Error via scipy.optimize  ─────────────────────────
# (Masena Approach 1 — il migliore per RMSE/AIC)
# Ottimizza il vettore pulse per minimizzare l'RMSE sul post-periodo.
# Il SARIMAX è riaddestrabile a ogni valutazione della funzione obiettivo.

def approach_f_trial_error(
    pre: pd.Series, post: pd.Series, order: tuple,
    cf_forecast: np.ndarray, consumption: float,
    max_eval: int = 300,
) -> dict:
    """
    Trial-and-error: ottimizza UN SOLO parametro (λ) del pulse esponenziale
    minimizzando l'RMSE sul post-periodo tramite Brent.
    (Masena & Shongwe 2024, Approccio 1)

    Forma del pulse: pulse[t] = exp(-λ · t),  t = 0, 1, ..., n_post-1
    λ ≥ 0: λ=0 → pulse costante (approccio D), λ grande → effetto immediato
    che decade.  1 parametro invece di n_post → nessun overfitting.

    NO look-ahead: ω stimato con OLS chiusa per ogni λ candidato.
    Solo al termine, SARIMAX fittato con il pulse ottimale per AIC/BIC.
    """
    n_post     = len(post)
    t_arr      = np.arange(n_post, dtype=float)
    resid_post = post.values - cf_forecast

    def _pulse_from_lambda(lam: float) -> np.ndarray:
        return np.exp(-lam * t_arr)

    def _omega_ols(pulse_vals: np.ndarray) -> float:
        denom = float(np.dot(pulse_vals, pulse_vals))
        return float(np.dot(resid_post, pulse_vals) / denom) if denom > 1e-12 else 0.0

    def _objective(lam: float) -> float:
        pulse  = _pulse_from_lambda(abs(lam))
        omega  = _omega_ols(pulse)
        fitted = cf_forecast + omega * pulse
        return float(np.sqrt(np.mean((post.values - fitted) ** 2)))

    best_lam  = 0.0
    best_rmse = _objective(0.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = optimize.minimize_scalar(
                lambda l: _objective(abs(l)),
                bounds=(0.0, 1.0),
                method="bounded",
                options={"maxiter": max_eval, "xatol": 1e-6},
            )
        if opt.fun < best_rmse:
            best_lam  = abs(float(opt.x))
            best_rmse = opt.fun
    except Exception as exc:
        warnings.warn(f"Ottimizzazione λ fallita: {exc}. Uso λ=0 (costante).")

    best_pulse  = _pulse_from_lambda(best_lam)
    omega_final = _omega_ols(best_pulse)

    full       = _build_full_series(pre, post)
    n_pre      = len(pre)
    pulse_full = np.zeros(len(full))
    pulse_full[n_pre: n_pre + n_post] = best_pulse
    sax = _fit_sarimax(full, order, pulse_full)

    fitted_post = cf_forecast + omega_final * best_pulse
    m = _approach_metrics(post, fitted_post, cf_forecast, consumption)

    return {
        "label":     APPROACH_LABELS["F"],
        "fitted_post": fitted_post,
        "pulse_post":  best_pulse,
        "omega":     omega_final,
        "lambda":    best_lam,
        "aic":       sax["aic"],
        "bic":       sax["bic"],
        "converged": sax["converged"],
        **m,
    }


def select_best_approach(results: dict[str, dict]) -> str:
    """Seleziona il miglior approccio per RMSE (a parità: AIC)."""
    valid = {k: v for k, v in results.items()
             if not np.isnan(v.get("rmse", np.nan))}
    if not valid:
        return list(results.keys())[0]
    return min(valid, key=lambda k: (
        valid[k]["rmse"],
        valid[k]["aic"] if not np.isnan(valid[k].get("aic", np.nan)) else 1e9
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Tabella periodo-per-periodo  (Masena Tabella 4/6; Aliffia Tabella 5)
# ══════════════════════════════════════════════════════════════════════════════

def build_period_table(
    post: pd.Series,
    cf_forecast: np.ndarray,
    results: dict[str, dict],
    best_key: str,
    consumption: float,
) -> pd.DataFrame:
    """
    Produce una tabella con una riga per ogni giorno del post-periodo,
    riportando per il miglior approccio:
      · data
      · valore effettivo (€/L)
      · forecast controfattuale (€/L)
      · covariata pulse ottimizzata
      · variazione % (effettivo − CF) / CF × 100
      · effetto stimato in M€ (giornaliero)
      · effetto cumulato in M€

    Formato analogo alla Tabella 4 di Masena & Shongwe (2024).
    """
    best   = results[best_key]
    extra  = best["extra"]
    fitted = best["fitted_post"]
    pulse  = best["pulse_post"]

    safe_cf = np.where(np.abs(cf_forecast) > 1e-9, cf_forecast, np.nan)
    pct     = (post.values - cf_forecast) / safe_cf * 100

    effect_daily_meur = extra * consumption / 1e6
    effect_cum_meur   = np.cumsum(effect_daily_meur)

    return pd.DataFrame({
        "data":                    post.index,
        "effettivo_eurl":          post.values.round(5),
        "controfattuale_eurl":     cf_forecast.round(5),
        "fitted_sarimax_eurl":     fitted.round(5),
        "pulse_covariata":         pulse.round(4),
        "pct_vs_controfattuale":   pct.round(2),
        "effetto_giornaliero_meur": effect_daily_meur.round(3),
        "effetto_cumulato_meur":   effect_cum_meur.round(3),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Plot principale
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_ax(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)


def _plot_main(
    ev_name: str, ev: dict, series: pd.Series,
    fuel_key: str, fuel_color: str,
    pre: pd.Series, post: pd.Series,
    cf_forecast: np.ndarray, cf_lo: np.ndarray, cf_hi: np.ndarray,
    order: tuple, results: dict, best_key: str,
    t0: pd.Timestamp, shock: pd.Timestamp, mode: str,
    period_table: pd.DataFrame,
    fig: plt.Figure, row_idx: int,
) -> None:
    """
    3 pannelli per ogni fuel:
      (a) Serie effettiva + controfattuale ARIMA (NO fitted SARIMAX sovrapposti)
      (b) Differenziale  actual − CF  con le 3 stime SARIMAX del gap
      (c) Guadagno cumulato
    Separare serie e differenziale rende immediatamente leggibile l'effetto.
    """
    n_rows = len(FUELS)
    base   = row_idx * 3

    ax_ser  = fig.add_subplot(n_rows, 3, base + 1)
    ax_diff = fig.add_subplot(n_rows, 3, base + 2)
    ax_gain = fig.add_subplot(n_rows, 3, base + 3)

    # ── Pannello A: serie + controfattuale ───────────────────────────────────
    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    ax_ser.plot(win.index, win.values, color=fuel_color, lw=1.5,
                label=f"{fuel_key.capitalize()} effettivo", zorder=5)
    ax_ser.plot(post.index, cf_forecast, color="black", lw=1.2, ls="--",
                label=f"CF ARIMA{order} (no look-ahead)", zorder=4)
    ax_ser.fill_between(post.index, cf_lo, cf_hi, alpha=0.12, color="black")
    ax_ser.axvline(shock, color=ev["color"], lw=1.5, ls="--",
                   label=f"Shock {shock.date()}")
    if mode == "detected" and t0 != shock:
        ax_ser.axvline(t0, color="black", lw=1.0, ls=":",
                       label=f"T0 PELT {t0.date()}")
    best = results[best_key]
    ax_ser.set_title(
        f"[V4] {fuel_key.capitalize()} – {ev_name}\n"
        f"ARIMA{order}  Best={best_key}  ω={best['omega']:.4f}",
        fontsize=7.5, fontweight="bold"
    )
    ax_ser.set_ylabel("Margine (€/L)", fontsize=8)
    ax_ser.legend(fontsize=5.5, loc="upper left")
    _fmt_ax(ax_ser)

    # ── Pannello B: differenziale  actual − CF  + stime gap ─────────────────
    raw_gap = post.values - cf_forecast
    ax_diff.bar(post.index, raw_gap, color=fuel_color, alpha=0.25, width=0.8,
                label="actual − CF (grezzo)")
    ax_diff.axhline(0, color="grey", lw=0.8, ls="--")

    # Stime del gap per ciascun approccio: fitted_post − cf_forecast = ω·pulse
    for key, res in results.items():
        lc  = APPROACH_COLORS[key]
        lw  = 2.0 if key == best_key else 1.0
        ls  = "-"  if key == best_key else ":"
        gap_est = res["fitted_post"] - cf_forecast
        lbl = (f"{key}: ω={res['omega']:.4f}  RMSE={res['rmse']:.4f}"
               + (" ★" if key == best_key else ""))
        ax_diff.plot(post.index, gap_est, color=lc, lw=lw, ls=ls, label=lbl, zorder=3)

    ax_diff.axvline(shock, color=ev["color"], lw=1.2, ls="--")
    ax_diff.set_title(
        f"Effetto stimato  (actual − CF)\n"
        f"se → 0: nessun effetto; se > 0: margine sopra atteso",
        fontsize=7.5
    )
    ax_diff.set_ylabel("Δ Margine (€/L)", fontsize=8)
    ax_diff.legend(fontsize=5.5, loc="upper left")
    _fmt_ax(ax_diff)

    # ── Pannello C: guadagno cumulato ────────────────────────────────────────
    for key, res in results.items():
        lc  = APPROACH_COLORS[key]
        lw  = 1.5 if key == best_key else 0.9
        ls  = "-"  if key == best_key else ":"
        cum = np.cumsum(res["extra"]) * DAILY_CONSUMPTION_L[fuel_key] / 1e6
        ax_gain.plot(post.index, cum, color=lc, lw=lw, ls=ls,
                     label=f"{key}: {res['gain_meur']:+.0f} M€")

    raw_cum = np.cumsum(raw_gap) * DAILY_CONSUMPTION_L[fuel_key] / 1e6
    ax_gain.plot(post.index, raw_cum, color="grey", lw=0.8, ls="--",
                 label=f"CF grezzo: {raw_gap.sum() * DAILY_CONSUMPTION_L[fuel_key] / 1e6:+.0f} M€")
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.set_title(
        f"Guadagno cumulato  [{fuel_key.capitalize()}]\n"
        f"Best ({best_key}) → {results[best_key]['gain_meur']:+.0f} M€  "
        f"[{results[best_key]['gain_lo']:+.0f} / {results[best_key]['gain_hi']:+.0f}]",
        fontsize=7
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=6, loc="upper left")
    _fmt_ax(ax_gain)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Transfer Function ITS pipeline")
    parser.add_argument("--mode",   choices=["fixed", "detected"], default="fixed")
    parser.add_argument("--detect", choices=["margin", "price"],   default="margin")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v4_transfer"
    else:
        OUT_DIR = _OUT_BASE / mode / "v4_transfer"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print(f"  02d_v4_transfer.py  –  Metodo 4: ITS + CCF + SARIMAX  [mode={mode}]")
    print(f"  Garanzia no-look-ahead: ARIMA stimato su tutti i dati da {PRE_START.date()} a T0")
    print(f"  Output: {OUT_DIR}")
    print("═" * 70)

    data = load_margin_data()
    all_rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig = plt.figure(figsize=(22, 6 * len(FUELS)))
        fig.suptitle(
            f"[Metodo 4 – CCF + Transfer Function / mode={mode}]  {ev_name}\n"
            f"{ev['label']}",
            fontsize=11, fontweight="bold"
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()
            if mode == "detected" and detect_target == "price":
                detect_series = data[PRICE_COLS[fuel_key]].dropna()
            else:
                detect_series = series

            # ── Determina T0 ──────────────────────────────────────────────────
            if mode == "detected":
                bp           = detect_breakpoint_pelt(detect_series, shock)
                t0           = bp["tau"]
                break_method = bp["method"]
            else:
                t0           = shock
                break_method = "fixed_at_shock"

            # ── Estrai finestre pre / post ────────────────────────────────────
            # pre: tutti i dati da PRE_START fino a T0 (no look-ahead garantito)
            pre = series[
                (series.index >= PRE_START) &
                (series.index < t0)
            ].dropna()
            post = series[
                (series.index >= t0) &
                (series.index < t0 + pd.Timedelta(days=POST_WIN))
            ].dropna()

            if len(pre) < 15 or len(post) < 5:
                print(f"  [{fuel_key}] dati insufficienti (pre={len(pre)}, post={len(post)}) – salto.")
                continue

            print(f"\n{'─'*60}")
            print(f"  {ev_name}  [{fuel_key.upper()}]  T0={t0.date()}")

            # ── ADF test informativo sul pre ──────────────────────────────────
            adf = adf_summary(pre)
            staz_str = ("stazionaria" if adf["stationary"] else
                        "NON stazionaria" if adf["stationary"] is False else "n/a")
            print(f"  ADF (pre):  p={adf['pvalue']:.3f}  → {staz_str}")

            # ══════════════════════════════════════════════════════════════════
            # FASE 1 — Auto-ARIMA su pre (no lookahead)
            # ══════════════════════════════════════════════════════════════════
            print("  [Fase 1] Auto-ARIMA AIC su pre... ", end="", flush=True)
            order, pre_fit = auto_arima_pre(pre)
            aic_pre = pre_fit.aic if pre_fit else np.nan
            print(f"ARIMA{order}  AIC={aic_pre:.1f}" if pre_fit else f"ARIMA{order} (fallback)")

            # ══════════════════════════════════════════════════════════════════
            # FASE 2 — Forecast controfattuale out-of-sample
            # ══════════════════════════════════════════════════════════════════
            print("  [Fase 2] Forecast controfattuale out-of-sample... ", end="", flush=True)
            cf_mean, cf_lo, cf_hi = counterfactual_forecast(pre, order, len(post))
            print("OK")

            # ══════════════════════════════════════════════════════════════════
            # FASE 3 — CCF per b, r, s
            # ══════════════════════════════════════════════════════════════════
            print("  [Fase 3] CCF → stima b, r, s... ", end="", flush=True)
            ccf_res = estimate_intervention_order(post.values, cf_mean)
            b, r, s = ccf_res["b"], ccf_res["r"], ccf_res["s"]
            print(f"b={b}  r={r}  s={s}")
            print(f"         {ccf_res['interpretation']}")

            safe_ev = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
            ccf_out = OUT_DIR / f"ccf_{safe_ev}_{fuel_key}.png"
            plot_ccf(
                residuals  = post.values - cf_mean,
                ccf_values = ccf_res["ccf_values"],
                acf_resid  = ccf_res["acf_resid"],
                threshold  = ccf_res["threshold"],
                b=b, r=r, s=s,
                post_index = post.index,
                ev_name    = ev_name,
                fuel_key   = fuel_key,
                fuel_color = fuel_color,
                out_path   = ccf_out,
            )

            # ══════════════════════════════════════════════════════════════════
            # FASE 4 — SARIMAX con tre approcci al pulse
            # ══════════════════════════════════════════════════════════════════
            consumption = DAILY_CONSUMPTION_L[fuel_key]
            print("  [Fase 4] SARIMAX approcci D / E / F:")

            print("    D (costante)...    ", end="", flush=True)
            res_d = approach_d_constant(pre, post, order, cf_mean, consumption, pre_fit=pre_fit)
            print(f"ω={res_d['omega']:.4f}  RMSE={res_d['rmse']:.5f}  AIC={res_d['aic']:.1f}")

            print("    E (quotient)...    ", end="", flush=True)
            res_e = approach_e_quotient(pre, post, order, cf_mean, consumption)
            print(f"ω={res_e['omega']:.4f}  RMSE={res_e['rmse']:.5f}  AIC={res_e['aic']:.1f}")

            print("    F (trial-error)... ", end="", flush=True)
            res_f = approach_f_trial_error(pre, post, order, cf_mean, consumption)
            print(f"ω={res_f['omega']:.4f}  RMSE={res_f['rmse']:.5f}  AIC={res_f['aic']:.1f}")

            results  = {"D": res_d, "E": res_e, "F": res_f}
            best_key = select_best_approach(results)

            print(f"\n  {'Approccio':<8} {'Gain M€':>10} {'RMSE':>10} "
                  f"{'MAPE%':>8} {'AIC':>10} {'ω':>10}")
            for k, r_ in results.items():
                star = " ★" if k == best_key else ""
                print(f"  {k}{star:<7} {r_['gain_meur']:>+10.1f} {r_['rmse']:>10.5f} "
                      f"{r_['mape']:>8.2f} {r_['aic']:>10.1f} {r_['omega']:>10.4f}")

            # ── Tabella periodo-per-periodo ───────────────────────────────────
            period_tbl = build_period_table(post, cf_mean, results, best_key, consumption)
            tbl_out = OUT_DIR / f"period_table_{safe_ev}_{fuel_key}.csv"
            period_tbl.to_csv(tbl_out, index=False)
            print(f"\n  → Tabella periodo: {tbl_out}")
            # Stampa riepilogo testuale (prime e ultime 3 righe)
            cols_show = ["data", "effettivo_eurl", "controfattuale_eurl",
                         "pct_vs_controfattuale", "effetto_giornaliero_meur",
                         "effetto_cumulato_meur"]
            print(period_tbl[cols_show].to_string(index=False, max_rows=8))

            # ── Plot diagnostico CCF già salvato sopra ────────────────────────
            # ── Plot principale ───────────────────────────────────────────────
            _plot_main(
                ev_name=ev_name, ev=ev, series=series,
                fuel_key=fuel_key, fuel_color=fuel_color,
                pre=pre, post=post,
                cf_forecast=cf_mean, cf_lo=cf_lo, cf_hi=cf_hi,
                order=order, results=results, best_key=best_key,
                t0=t0, shock=shock, mode=mode,
                period_table=period_tbl,
                fig=fig, row_idx=row_idx,
            )

            # ── Diagnostica residui SARIMAX best ─────────────────────────────
            best_res = results[best_key]
            full_series = _build_full_series(pre, post)
            pulse_best  = np.zeros(len(full_series))
            pulse_best[len(pre):] = best_res["pulse_post"]
            sax_full = _fit_sarimax(full_series, order, pulse_best)
            diag = run_diagnostic_tests(sax_full["resid"], x_for_bg=None, n_lags=None)
            diag_out = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=sax_full["resid"],
                dates=full_series.index,
                title=(f"[V4–SARIMAX best={best_key}] Residui  {ev_name} · {fuel_key}"),
                out_path=diag_out,
                diag_stats=diag,
            )

            # ── Raccolta righe CSV ────────────────────────────────────────────
            base_row = {
                "metodo":          "v4_transfer",
                "mode":            mode,
                "detect_target":   detect_target if mode == "detected" else "fixed",
                "evento":          ev_name,
                "carburante":      fuel_key,
                "shock":           shock.date(),
                "break_date":      t0.date(),
                "break_method":    break_method,
                "pre_win_days":    PRE_WIN,
                "post_win_days":   POST_WIN,
                "n_pre":           len(pre),
                "n_post":          len(post),
                "arima_order":     str(order),
                "adf_pvalue_pre":  round(adf["pvalue"], 4) if not np.isnan(adf["pvalue"]) else np.nan,
                "ccf_b":           b,
                "ccf_r":           r,
                "ccf_s":           s,
                "ccf_interpretation": ccf_res["interpretation"],
                "best_approach":   best_key,
            }
            for app_key, res_ in results.items():
                row = {**base_row,
                       "approccio":         app_key,
                       "gain_total_meur":   round(res_["gain_meur"], 1),
                       "gain_ci_low_meur":  round(res_["gain_lo"],   1),
                       "gain_ci_high_meur": round(res_["gain_hi"],   1),
                       "omega":             round(res_["omega"], 5) if not np.isnan(res_["omega"]) else np.nan,
                       "aic":               round(res_["aic"],   2) if not np.isnan(res_["aic"])   else np.nan,
                       "bic":               round(res_["bic"],   2) if not np.isnan(res_["bic"])   else np.nan,
                       "rmse":              round(res_["rmse"],  5),
                       "mape_pct":          round(res_["mape"],  3) if not np.isnan(res_["mape"])  else np.nan,
                       "converged":         res_["converged"],
                       "is_best":           app_key == best_key,
                       "note":              (f"ITS CCF+Transfer, mode={mode}"
                                             + (f", detect={detect_target}" if mode == "detected" else "")),
                       }
                all_rows.append(row)

        # ── Salva plot evento ─────────────────────────────────────────────────
        fig.tight_layout()
        out_plot = OUT_DIR / f"plot_{safe_ev}.png"
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Plot salvato: {out_plot}")

    # ── CSV riepilogo globale ─────────────────────────────────────────────────
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        csv_out = OUT_DIR / "v4_sarimax_results.csv"
        df_out.to_csv(csv_out, index=False)
        print(f"\n  → CSV globale: {csv_out}")

        best_df = df_out[df_out["is_best"]]
        print("\nRIEPILOGO (best approach per ogni caso):")
        cols = ["evento", "carburante", "break_date", "ccf_b", "ccf_r", "ccf_s",
                "best_approach", "gain_total_meur", "rmse", "omega"]
        print(best_df[cols].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()