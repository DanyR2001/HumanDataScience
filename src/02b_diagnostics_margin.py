#!/usr/bin/env python3
"""
02b_diagnostics_margin.py
=========================
Stessi 5 test diagnostici di 02b_diagnostics.py ma applicati al MARGINE:

    margine = prezzo_netto (ex-tasse, da SISEN) − prezzo_wholesale (futures €/L)

Dati:
  • gasolio  → Gas Oil Futures (2017-01-02 → oggi)      — copertura completa su tutti e 3 gli eventi
  • benzina  → Eurobob B7H1 Futures (2015-01-01 → oggi) — copertura completa su tutti e 3 gli eventi
              Fonte: Eurobob_B7H1_date.csv  (scraping TradingView NYMEX:B7H1!)
              Colonne: data, timestamp (Unix), apertura, massimo, minimo, chiusura, variazione, volume
              Prezzi in USD/MT → convertiti in €/L tramite EUR/USD + heat content Eurobob

Output in data/plots/price/margin/:
  • 00_margine_serie_storica.png   — serie temporale del margine (entrambi i carburanti)
  • 30 PNG diagnostici             — 5 test × 3 eventi × 2 carburanti
  • 6 CSV con risultati numerici
  • README.md
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR     = BASE_DIR / "data" / "plots" / "diagnostics" / "margin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA     = 0.05
HALF_WIN  = 40
LB_LAGS   = [5, 10, 20]
ARCH_LAGS = 10

FUELS = {
    "benzina": ("#E63946", EUROBOB_HC),
    "gasolio": ("#1D3557", GAS_OIL),
}

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-09-01"),
        "post_end":  pd.Timestamp("2022-08-31"),
        "color":     "#e74c3c",
        "label":     "Russia-Ucraina (24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-01-01"),
        "post_end":  pd.Timestamp("2025-10-31"),
        "color":     "#e67e22",
        "label":     "Iran-Israele (13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-08-01"),
        "post_end":  pd.Timestamp("2026-04-30"),
        "color":     "#8e44ad",
        "label":     "Stretto di Hormuz (28 feb 2026)",
    },
}

_OK   = "#2ecc71"
_WARN = "#f39c12"
_FAIL = "#e74c3c"


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def load_futures_eurl(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    """Carica Gas Oil CSV con formato standard Investing.com (Date, Price in USD/MT)."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price_usd_ton"] = (df["Price"].str.replace(",", "", regex=False)
                           .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price_usd_ton"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def load_futures_b7h1(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    """
    Carica Eurobob_B7H1_date.csv prodotto dallo scraper TradingView.

    Colonne attese:
        data       — etichetta testuale (es. "01 gen 2015"), usata solo come fallback
        timestamp  — Unix timestamp (secondi UTC), fonte primaria per la data
        chiusura   — prezzo di chiusura in USD/MT (formato decimale già normalizzato)

    Restituisce una pd.Series indicizzata per data con i prezzi in €/L.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    # ── Data: usa timestamp Unix se disponibile, altrimenti colonna 'data' ──
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        # Fallback: colonna 'data' in formato italiano (es. "01 gen 2015")
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

    # ── Prezzo chiusura (già in formato decimale dallo scraper) ──
    df["price_usd_ton"] = pd.to_numeric(df["chiusura"], errors="coerce")

    df = (df.dropna(subset=["date", "price_usd_ton"])
            .sort_values("date")
            .set_index("date"))

    # Rimuovi duplicati di data (tieni la prima occorrenza = più recente per quel giorno)
    df = df[~df.index.duplicated(keep="first")]

    n = len(df)
    date_range = f"{df.index.min().date()} → {df.index.max().date()}"
    print(f"  B7H1 caricato: {n} righe  ({date_range})")

    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def build_margin(daily: pd.DataFrame,
                 gasoil_eurl: pd.Series,
                 eurobob_eurl: pd.Series | None) -> pd.DataFrame:
    """
    Costruisce DataFrame con due colonne:
      margin_gasolio = gasolio_net − gasoil_futures_eurl
      margin_benzina = benzina_net − eurobob_futures_eurl   (NaN dove Eurobob non disponibile)
    """
    df = daily[["benzina_net", "gasolio_net"]].copy()

    # Gasolio — copertura completa
    ws_gas = gasoil_eurl.reindex(df.index, method="ffill")
    df["margin_gasolio"] = df["gasolio_net"] - ws_gas

    # Benzina — solo dove Eurobob esiste
    if eurobob_eurl is not None:
        ws_benz = eurobob_eurl.reindex(df.index, method="ffill")
        df["margin_benzina"] = df["benzina_net"] - ws_benz
    else:
        df["margin_benzina"] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Test diagnostici (identici a 02b_diagnostics.py)
# ══════════════════════════════════════════════════════════════════════════════

def test_stationarity(series: pd.Series) -> dict:
    adf_stat, adf_p = adfuller(series.dropna(), autolag="AIC")[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(series.dropna(), regression="c", nlags="auto")
    adf_ok  = adf_p  < ALPHA
    kpss_ok = kpss_p > ALPHA
    status  = "OK" if (adf_ok and kpss_ok) else ("⚠ WARN" if (adf_ok or kpss_ok) else "✗ FAIL")
    return {"adf_stat": adf_stat, "adf_p": adf_p,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p, "status": status,
            "detail": f"ADF p={adf_p:.3f} ({'✓' if adf_ok else '✗'}), KPSS p={kpss_p:.3f} ({'✓' if kpss_ok else '✗'})"}


def test_autocorrelation(series: pd.Series) -> dict:
    diff = series.dropna().diff().dropna()
    lb = acorr_ljungbox(diff, lags=LB_LAGS, return_df=True)
    lb_fail = (lb["lb_pvalue"] < ALPHA).any()
    phi = diff.autocorr(lag=1)
    n   = len(series.dropna())
    phi_c = max(min(phi, 0.999), -0.999)
    n_eff = max(1, int(n * (1 - phi_c) / (1 + phi_c)))
    lb_summary = {lag: f"{row['lb_pvalue']:.3f}" for lag, row in lb.iterrows()}
    status = "✗ FAIL" if lb_fail else "OK"
    return {"phi": phi, "n_eff": n_eff, "n": n, "lb": lb_summary, "lb_fail": lb_fail,
            "status": status,
            "detail": f"φ={phi:.3f}  n_eff={n_eff}/{n}  LB p={list(lb_summary.values())}"}


def test_normality(series: pd.Series) -> dict:
    diff = series.dropna().diff().dropna()
    sw_stat, sw_p = stats.shapiro(diff) if len(diff) <= 5000 else (np.nan, np.nan)
    jb_stat, jb_p = stats.jarque_bera(diff)
    skew = float(stats.skew(diff))
    kurt = float(stats.kurtosis(diff))
    sw_ok = (sw_p > ALPHA) if not np.isnan(sw_p) else True
    jb_ok = jb_p > ALPHA
    status = "OK" if (sw_ok and jb_ok) else ("⚠ WARN" if (sw_ok or jb_ok) else "✗ FAIL")
    return {"sw_stat": sw_stat, "sw_p": sw_p, "jb_stat": jb_stat, "jb_p": jb_p,
            "skew": skew, "kurt_excess": kurt, "status": status,
            "detail": f"SW p={sw_p:.3f}  JB p={jb_p:.3f}  skew={skew:.2f}  kurt_exc={kurt:.2f}"}


def test_homoscedasticity(series: pd.Series, shock: pd.Timestamp) -> dict:
    pre  = series[series.index < shock].tail(HALF_WIN).dropna()
    post = series[series.index >= shock].head(HALF_WIN).dropna()
    if len(pre) < 5 or len(post) < 5:
        return {"status": "⚠ WARN", "detail": "campioni insufficienti",
                "lev_stat": np.nan, "lev_p": np.nan, "std_pre": np.nan,
                "std_post": np.nan, "ratio": np.nan}
    lev_stat, lev_p = stats.levene(pre, post)
    std_pre, std_post = float(pre.std()), float(post.std())
    ratio = std_post / std_pre if std_pre > 0 else np.nan
    status = "OK" if lev_p > ALPHA else "✗ FAIL"
    return {"lev_stat": lev_stat, "lev_p": lev_p, "std_pre": std_pre,
            "std_post": std_post, "ratio": ratio, "status": status,
            "detail": f"Levene p={lev_p:.3f}  σ_pre={std_pre:.4f}  σ_post={std_post:.4f}  ratio={ratio:.2f}x"}


def test_arch(series: pd.Series) -> dict:
    diff = series.dropna().diff().dropna()
    try:
        lm_stat, lm_p, f_stat, f_p = het_arch(diff, nlags=ARCH_LAGS)
        status = "OK" if lm_p > ALPHA else "✗ FAIL"
        detail = f"ARCH-LM p={lm_p:.3f}  F-stat={f_stat:.2f}"
    except Exception as e:
        lm_stat = lm_p = f_stat = f_p = np.nan
        status, detail = "⚠ WARN", f"Test fallito: {e}"
    return {"lm_stat": lm_stat, "lm_p": lm_p, "f_stat": f_stat, "f_p": f_p,
            "status": status, "detail": detail}


def run_diagnostics(series: pd.Series, shock: pd.Timestamp, label: str) -> dict:
    win = series.dropna()
    print(f"  [{label.upper()}]  n={len(win)}")
    r = {
        "stationarity":    test_stationarity(win),
        "autocorrelation": test_autocorrelation(win),
        "normality":       test_normality(win),
        "homoscedasticity":test_homoscedasticity(win, shock),
        "arch":            test_arch(win),
    }
    for name, res in r.items():
        icon = {"OK": "✓", "⚠ WARN": "⚠", "✗ FAIL": "✗"}.get(res["status"], "?")
        print(f"    {icon} {name:<18}  {res['status']:<8}  {res['detail']}")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Grafico 0 — serie storica del margine
# ══════════════════════════════════════════════════════════════════════════════

def plot_margin_series(margin: pd.DataFrame) -> plt.Figure:
    """Serie temporale completa del margine per entrambi i carburanti."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Margine industriale+distribuzione+retailer (€/L)\n"
                 "= Prezzo netto ex-tasse  −  Wholesale futures convertiti",
                 fontsize=12, fontweight="bold")

    configs = [
        ("margin_gasolio", "#1D3557", "Gasolio  (netto SISEN − Gas Oil futures €/L)", "Gasolio"),
        ("margin_benzina", "#E63946", "Benzina  (netto SISEN − Eurobob B7H1 futures €/L)", "Benzina"),
    ]

    for ax, (col, color, desc, title) in zip(axes, configs):
        s = margin[col].dropna()
        if s.empty:
            ax.text(0.5, 0.5, "Dati non disponibili", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        else:
            ax.fill_between(s.index, s.values, alpha=0.15, color=color)
            ax.plot(s.index, s.values, color=color, lw=1.0, label=desc)
            ax.axhline(s.mean(), color=color, lw=1, ls=":", alpha=0.8,
                       label=f"media = {s.mean():.3f} €/L")
            ax.axhline(0, color="black", lw=0.6)

            # Evidenzia gli eventi
            for ev_name, ev in EVENTS.items():
                if ev["shock"] >= s.index.min() and ev["shock"] <= s.index.max():
                    ax.axvline(ev["shock"], color=ev["color"], lw=1.5, ls="--", alpha=0.8)
                    ax.text(ev["shock"], ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else s.max(),
                            f" {ev['shock'].strftime('%b %Y')}",
                            color=ev["color"], fontsize=7, va="top")

        ax.set_ylabel("€/L", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Grafici diagnostici (uno per test × evento × carburante)
# ══════════════════════════════════════════════════════════════════════════════

def _badge(ax, status: str) -> None:
    col   = {"OK": _OK, "⚠ WARN": _WARN, "✗ FAIL": _FAIL}.get(status, "#aaa")
    label = {"OK": "✓  OK", "⚠ WARN": "⚠  WARN", "✗ FAIL": "✗  FAIL"}[status]
    ax.text(0.98, 0.97, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=col, edgecolor="none"), zorder=10)


def _base_title(ev: dict, fuel_key: str, test_label: str) -> str:
    return f"{test_label}  —  {ev['label']}  ·  {fuel_key.capitalize()}  [margine]"


def plot_one_stationarity(win, ev, fuel_key, fcolor, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    shock = ev["shock"]
    ax.fill_between(win.index, win.values, alpha=0.15, color=fcolor)
    ax.plot(win.index, win.values, color=fcolor, lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(shock, color=ev["color"], lw=2, ls="--")
    ax.axvspan(win.index.min(), shock, alpha=0.04, color="steelblue")
    ax.axvspan(shock, win.index.max(), alpha=0.04, color="tomato")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_ylabel("Margine (€/L)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    adf_ok  = "✓" if res["adf_p"]  < ALPHA else "✗"
    kpss_ok = "✓" if res["kpss_p"] > ALPHA else "✗"
    ax.text(0.02, 0.05,
            f"ADF   p = {res['adf_p']:.3f}  {adf_ok}   (rifiutare H₀ = stazionario)\n"
            f"KPSS  p = {res['kpss_p']:.3f}  {kpss_ok}   (non rifiutare H₀ = stazionario)",
            transform=ax.transAxes, fontsize=9, va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Stazionarietà — ADF + KPSS"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


def plot_one_autocorrelation(win, ev, fuel_key, fcolor, res):
    from statsmodels.graphics.tsaplots import plot_acf
    diff = win.diff().dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        plot_acf(diff, lags=25, ax=ax, color=fcolor, alpha=0.05, zero=False, title="")
    except Exception:
        ax.text(0.5, 0.5, "n.d.", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Lag (giorni)", fontsize=10)
    ax.set_ylabel("ACF", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lb_vals = list(res["lb"].values())
    ax.text(0.02, 0.05,
            f"φ AR(1)  = {res['phi']:.3f}      n_eff = {res['n_eff']} / {res['n']}\n"
            f"Ljung-Box p (lag 5, 10, 20) = {lb_vals[0]},  {lb_vals[1]},  {lb_vals[2]}",
            transform=ax.transAxes, fontsize=9, va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Autocorrelazione — ACF prime differenze"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


def plot_one_normality(win, ev, fuel_key, fcolor, res):
    diff = win.diff().dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(diff, dist="norm")
    ax.scatter(osm, osr, s=10, color=fcolor, alpha=0.65, rasterized=True)
    xlim = np.array([min(osm), max(osm)])
    ax.plot(xlim, slope * xlim + intercept, color="red", lw=1.5, ls="--", label="Normale teorica")
    ax.set_xlabel("Quantili N(0,1)", fontsize=10)
    ax.set_ylabel("Quantili campionari (Δmargine)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sw_str = f"SW   p = {res['sw_p']:.3f}" if not np.isnan(res.get("sw_p", np.nan)) else "SW   n.d."
    ax.text(0.02, 0.97,
            f"{sw_str}\n"
            f"JB   p = {res['jb_p']:.3f}\n"
            f"skew  = {res['skew']:.2f}   kurt_exc = {res['kurt_excess']:.2f}",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Normalità — QQ plot"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


def plot_one_homoscedasticity(win, ev, fuel_key, fcolor, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    shock = ev["shock"]
    roll_std = win.rolling(14, min_periods=7).std()
    ax.fill_between(roll_std.index, roll_std.values, alpha=0.2, color=fcolor)
    ax.plot(roll_std.index, roll_std.values, color=fcolor, lw=1.5)
    ax.axvline(shock, color=ev["color"], lw=2, ls="--")
    pre_idx  = win.index[win.index < shock]
    post_idx = win.index[win.index >= shock]
    if len(pre_idx) and len(post_idx) and not np.isnan(res["std_pre"]):
        ax.hlines(res["std_pre"],  pre_idx.min(),  pre_idx.max(),
                  colors="steelblue", lw=1.5, ls=":", label=f"σ pre = {res['std_pre']:.4f}")
        ax.hlines(res["std_post"], post_idx.min(), post_idx.max(),
                  colors="tomato",   lw=1.5, ls=":", label=f"σ post = {res['std_post']:.4f}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_ylabel("σ rolling 14gg  (€/L)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.05,
            f"Levene  p = {res['lev_p']:.3f}\n"
            f"ratio σ post/pre = {res.get('ratio', float('nan')):.2f}×",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Omoschedasticità — volatilità rolling + Levene"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


def plot_one_arch(win, ev, fuel_key, fcolor, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    shock = ev["shock"]
    diff2 = win.diff().dropna() ** 2
    ax.fill_between(diff2.index, diff2.values, alpha=0.35, color=fcolor)
    ax.plot(diff2.index, diff2.values, color=fcolor, lw=0.8)
    ax.axvline(shock, color=ev["color"], lw=2, ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_ylabel("(Δmargine)²", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.97,
            f"ARCH-LM  p = {res['lm_p']:.3f}\n"
            f"F-stat     = {res['f_stat']:.2f}",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Effetti ARCH — (Δmargine)²"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Export CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(all_results: dict) -> None:
    rows_summary, rows_stat = [], {k: [] for k in
        ("stationarity","autocorrelation","normality","homoscedasticity","arch")}

    for ev_name, ev in EVENTS.items():
        for fuel_key, res_all in all_results.get(ev_name, {}).items():
            base = {"evento": ev_name, "shock": ev["shock"].date(), "carburante": fuel_key}
            for test_it, test_key in [
                ("Stazionarietà","stationarity"), ("Autocorrelazione","autocorrelation"),
                ("Normalità","normality"), ("Omoschedasticità","homoscedasticity"),
                ("Effetti ARCH","arch"),
            ]:
                res = res_all[test_key]
                rows_summary.append({**base, "test": test_it,
                                     "status": res["status"], "dettaglio": res["detail"]})

            r = res_all["stationarity"]
            rows_stat["stationarity"].append({**base, "status": r["status"],
                "adf_stat": round(r["adf_p"],4), "adf_p": round(r["adf_p"],4),
                "adf_ok": r["adf_p"] < ALPHA,
                "kpss_stat": round(r["kpss_stat"],4), "kpss_p": round(r["kpss_p"],4),
                "kpss_ok": r["kpss_p"] > ALPHA})

            r = res_all["autocorrelation"]
            lb = r["lb"]; lb_keys = list(lb.keys())
            rows_stat["autocorrelation"].append({**base, "status": r["status"],
                "phi_ar1": round(r["phi"],4), "n": r["n"], "n_eff": r["n_eff"],
                **{f"lb_p_lag{k}": lb[k] for k in lb_keys}, "lb_violazione": r["lb_fail"]})

            r = res_all["normality"]
            rows_stat["normality"].append({**base, "status": r["status"],
                "sw_stat": round(r["sw_stat"],4) if not np.isnan(r["sw_stat"]) else "",
                "sw_p":    round(r["sw_p"],4)    if not np.isnan(r["sw_p"])    else "",
                "jb_stat": round(r["jb_stat"],4), "jb_p": round(r["jb_p"],4),
                "skewness": round(r["skew"],4), "kurtosis_exc": round(r["kurt_excess"],4)})

            r = res_all["homoscedasticity"]
            rows_stat["homoscedasticity"].append({**base, "status": r["status"],
                "lev_stat":  round(r["lev_stat"],4)  if not np.isnan(r["lev_stat"])  else "",
                "lev_p":     round(r["lev_p"],4)     if not np.isnan(r["lev_p"])     else "",
                "sigma_pre": round(r["std_pre"],6)   if not np.isnan(r["std_pre"])   else "",
                "sigma_post":round(r["std_post"],6)  if not np.isnan(r["std_post"])  else "",
                "ratio_post_pre": round(r.get("ratio",float("nan")),3)
                                  if not np.isnan(r.get("ratio",float("nan"))) else ""})

            r = res_all["arch"]
            rows_stat["arch"].append({**base, "status": r["status"],
                "lm_stat": round(r["lm_stat"],4) if not np.isnan(r["lm_stat"]) else "",
                "lm_p":    round(r["lm_p"],4)    if not np.isnan(r["lm_p"])    else "",
                "f_stat":  round(r["f_stat"],4)  if not np.isnan(r["f_stat"])  else "",
                "f_p":     round(r["f_p"],4)     if not np.isnan(r["f_p"])     else ""})

    pd.DataFrame(rows_summary).to_csv(OUT_DIR / "risultati_riepilogo.csv",
                                       index=False, encoding="utf-8-sig")
    print("  CSV: risultati_riepilogo.csv")
    for key, fname in [
        ("stationarity","risultati_stazionarieta.csv"),
        ("autocorrelation","risultati_autocorrelazione.csv"),
        ("normality","risultati_normalita.csv"),
        ("homoscedasticity","risultati_omoschedasticita.csv"),
        ("arch","risultati_arch.csv"),
    ]:
        pd.DataFrame(rows_stat[key]).to_csv(OUT_DIR / fname, index=False, encoding="utf-8-sig")
        print(f"  CSV: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Carico dati...")
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))

    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
    )

    gasoil_eurl  = load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    margin = build_margin(daily, gasoil_eurl, eurobob_eurl)

    # Disponibilità benzina
    benz_avail = margin["margin_benzina"].dropna()
    if not benz_avail.empty:
        print(f"  Margine benzina disponibile: {benz_avail.index.min().date()} → {benz_avail.index.max().date()}")
    else:
        print("  Margine benzina: Eurobob non disponibile")
    print(f"  Margine gasolio disponibile: {margin['margin_gasolio'].dropna().index.min().date()} → "
          f"{margin['margin_gasolio'].dropna().index.max().date()}")

    # ── Grafico 0: serie storica del margine ──────────────────────────────────
    print("\nGrafico 0: serie storica margine...")
    fig0 = plot_margin_series(margin)
    out0 = OUT_DIR / "00_margine_serie_storica.png"
    fig0.savefig(out0, dpi=150, bbox_inches="tight")
    plt.close(fig0)
    print(f"  ✓ {out0.name}")

    # ── Diagnostiche per evento × carburante ──────────────────────────────────
    fuel_margin_col = {"gasolio": "margin_gasolio", "benzina": "margin_benzina"}
    fuel_color_map  = {"gasolio": "#1D3557", "benzina": "#E63946"}

    all_results: dict = {}
    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]
        print(f"\n{'═'*72}")
        print(f"  {ev_name}  (shock={shock.date()})")
        print(f"{'═'*72}")
        all_results[ev_name] = {}

        for fuel_key, margin_col in fuel_margin_col.items():
            win = margin[margin_col][
                (margin.index >= ev["pre_start"]) &
                (margin.index <= ev["post_end"])
            ].dropna()
            if len(win) < 20:
                print(f"  [{fuel_key}] dati insufficienti (n={len(win)}) — salto.")
                continue
            all_results[ev_name][fuel_key] = run_diagnostics(win, shock, fuel_key)

    # ── 30 grafici diagnostici ────────────────────────────────────────────────
    TEST_PLOT_FNS = [
        ("01_stazionarieta",    "stationarity",    plot_one_stationarity),
        ("02_autocorrelazione", "autocorrelation", plot_one_autocorrelation),
        ("03_normalita",        "normality",        plot_one_normality),
        ("04_omoschedasticita", "homoscedasticity", plot_one_homoscedasticity),
        ("05_arch",             "arch",             plot_one_arch),
    ]

    print(f"\n{'═'*72}")
    print("Generazione grafici diagnostici...")
    count = 0
    for ev_name, ev in EVENTS.items():
        ev_slug = (ev_name.lower()
                   .replace(" ", "_").replace("(","").replace(")","")
                   .replace("/","").replace("-","_"))
        for fuel_key in ("gasolio", "benzina"):
            if fuel_key not in all_results.get(ev_name, {}):
                continue
            res_all = all_results[ev_name][fuel_key]
            fcolor  = fuel_color_map[fuel_key]
            win = margin[fuel_margin_col[fuel_key]][
                (margin.index >= ev["pre_start"]) &
                (margin.index <= ev["post_end"])
            ].dropna()

            for prefix, test_key, fn in TEST_PLOT_FNS:
                fig = fn(win, ev, fuel_key, fcolor, res_all[test_key])
                out = OUT_DIR / f"{prefix}__{ev_slug}__{fuel_key}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ {out.name}")
                count += 1

    # ── CSV + README ──────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    export_csv(all_results)
    print(f"\nDone. {count} grafici diagnostici + serie storica + CSV in: {OUT_DIR}")


if __name__ == "__main__":
    main()