#!/usr/bin/env python3
"""
02d_v1_naive.py  ─  Metodo 1: OLS Naïve
=========================================
Il metodo più semplice possibile per stimare l'extra-profitto speculativo
sul margine distributori (prezzo pompa netto − futures €/L) in corrispondenza
di eventi geopolitici.

Assunzioni (e limitazioni):
  ─ Baseline     : trend OLS lineare sui PRE_WIN giorni precedenti il break
  ─ Proiezione   : la retta OLS viene estrapolata in avanti (controfattuale)
  ─ Extra profitto : margine_effettivo(t) − baseline_proiettata(t)
  ─ CI           : intervallo di previsione OLS standard (i.i.d.)

Modalità (--mode):
  fixed     : break = data dello shock hardcodata [default]
  detected  : break rilevato via massimizzazione |Δmean| / pooled_std
              su finestra scorrevole (sliding mean naive, la detection
              più semplice possibile — nessuna correzione AR(1) o formale).
              Tra i break nel top quartile dello score, viene scelto
              il PRIMO in ordine temporale (inizio ottimale della rottura).

Parametro --detect (solo quando --mode detected):
  margin  : detection sul margine distributore           [default]
  price   : detection sul prezzo alla pompa netto (€/L)
            → il break viene cercato sul prezzo, ma l'ITS resta sul margine

Output:
  data/plots/its/{mode}/v1_naive/              (se mode=fixed)
  data/plots/its/detected/{detect}/v1_naive/   (se mode=detected)
    plot_{evento}_{carburante}.png
    v1_naive_results.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter
from diagnostics import (
    run_diagnostic_tests,
    plot_residual_diagnostics,
)

try:
    import statsmodels.api as _sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN   = 60    # giorni pre-break per stimare la baseline
POST_WIN  = 60    # giorni post-break per calcolare l'extra profitto
CI_ALPHA  = 0.10  # livello α → intervallo di previsione al 90%

# Detection naive (usata solo in mode=detected)
SEARCH    = 30    # ricerca τ in [shock-SEARCH, shock+SEARCH] giorni
MIN_SEG   = 15    # finestra minima per lato (pre e post) del confronto

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

# Colonne prezzo pompa netto (usate come serie di detection quando --detect price)
PRICE_COLS: dict[str, str] = {
    "benzina": "benzina_net",
    "gasolio": "gasolio_net",
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def _load_gasoil_futures(eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, encoding="utf-8-sig", dtype=str)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price"], eurusd, GAS_OIL)


def _load_eurobob_futures(eurusd: pd.Series) -> pd.Series | None:
    if not EUROBOB_CSV.exists():
        return None
    df = pd.read_csv(EUROBOB_CSV, encoding="utf-8-sig", dtype=str)
    _IT = {"gen":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","mag":"May","giu":"Jun",
           "lug":"Jul","ago":"Aug","set":"Sep","ott":"Oct","nov":"Nov","dic":"Dec"}
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
    df = df.dropna(subset=["date","price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_margin_data() -> pd.DataFrame:
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
    )
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    if eurobob is not None:
        df["margin_benzina"] = df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
    else:
        df["margin_benzina"] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Break point detection — Naive (mode=detected)
# ══════════════════════════════════════════════════════════════════════════════

def detect_breakpoint_naive(series: pd.Series, shock: pd.Timestamp) -> dict:
    """
    Metodo naïve: scansiona ogni giorno τ nell'intervallo
    [shock − SEARCH, shock + SEARCH] e calcola |mean_pre − mean_post| / pooled_std
    su finestre fisse di MIN_SEG giorni per lato.

    È il test più elementare di cambio di livello: non corregge per
    autocorrelazione AR(1) (come fa v2) e non usa modelli parametrici
    (come fa v3 con PELT).

    Selezione del break:
      Tra i candidati nel top quartile dello score (break con segnale genuino),
      viene scelto il PRIMO in ordine temporale (il più a sinistra / più antico).
      Questo cattura l'inizio effettivo della rottura strutturale, non
      necessariamente il picco del segnale.

    Fallback → shock date se nessun candidato supera MIN_SEG osservazioni.
    """
    candidates = pd.date_range(
        shock - pd.Timedelta(days=SEARCH),
        shock + pd.Timedelta(days=SEARCH),
        freq="1D",
    )
    rows = []

    for t in candidates:
        pre  = series[
            (series.index >= t - pd.Timedelta(days=MIN_SEG)) &
            (series.index < t)
        ].dropna()
        post = series[
            (series.index >= t) &
            (series.index < t + pd.Timedelta(days=MIN_SEG))
        ].dropna()

        if len(pre) < MIN_SEG or len(post) < MIN_SEG:
            continue

        diff       = abs(float(pre.mean()) - float(post.mean()))
        pooled_std = np.sqrt((float(pre.std())**2 + float(post.std())**2) / 2)
        score      = diff / pooled_std if pooled_std > 1e-12 else 0.0
        rows.append({"tau": t, "score": score,
                     "dist": abs((t - shock).days)})

    if not rows:
        return {"tau": shock, "score": 0.0, "method": "sliding_mean_naive_nofound",
                "_df": pd.DataFrame(), "_threshold": 0.0}

    df_c = pd.DataFrame(rows)

    # Filtra al top quartile dello score (break con segnale genuino)
    threshold = float(np.percentile(df_c["score"], 75))
    top = df_c[df_c["score"] >= threshold]
    if top.empty:
        top = df_c  # fallback: usa tutti i candidati

    # Scegli il PRIMO in ordine temporale tra i candidati di qualità
    best = top.sort_values("tau").iloc[0]
    return {
        "tau":        best["tau"],
        "score":      round(float(best["score"]), 4),
        "method":     "sliding_mean_naive",
        "_df":        df_c,          # tutti i candidati con score e dist
        "_threshold": threshold,     # soglia 75° percentile
    }


# ══════════════════════════════════════════════════════════════════════════════
# Break point detection — Window L2 Discrepancy (Paper BLOCCO 1 – Eq. 1–2)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_cost(y: np.ndarray) -> float:
    """c(y_I) = Σ||y_t − ȳ||² — costo L2 intra-finestra (Paper, Eq. 2)."""
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def detect_breakpoint_window_l2(
    series: pd.Series,
    shock: pd.Timestamp,
    ma_window: int = 7,
    min_seg: int = 14,
) -> dict:
    """
    Changepoint detection via discrepanza L2 tra due finestre contigue.
    Implementa il metodo del paper (BLOCCO 1, Eq. 1–2):

        d(y_{uv}, y_{vw}) = c(y_{uw}) − c(y_{uv}) − c(y_{vw})
        changepoint = argmax_v  d(...)

    Pre-processing: moving average a 7 giorni (obbligatorio nel paper).

    La discrepanza d misura quanto la divisione in v riduce la varianza
    intra-finestra → il picco di d corrisponde al punto di maggiore
    omogeneità interna a sinistra e a destra.

    Rispetto al metodo naïve (v=media sliding), questo è più robusto
    perché non dipende dalla scelta di una finestra fissa per lato:
    usa l'intera storia disponibile prima e dopo ogni candidato.

    Fallback → shock date se dati insufficienti.
    """
    mask = (
        (series.index >= shock - pd.Timedelta(days=SEARCH)) &
        (series.index <= shock + pd.Timedelta(days=SEARCH))
    )
    win = series[mask].dropna()

    if len(win) < 2 * min_seg + ma_window:
        return {"tau": shock, "score": 0.0, "method": "window_l2_nofound",
                "_df": pd.DataFrame(), "_threshold": 0.0}

    # Pre-processing: MA a 7 giorni (centrante)
    ma = win.rolling(ma_window, center=True, min_periods=1).mean()
    y  = ma.values
    n  = len(y)

    c_uw = _l2_cost(y)       # costo dell'intera finestra (costante)

    d_vals: list[float] = []
    cand_dates: list[pd.Timestamp] = []

    for v in range(min_seg, n - min_seg):
        d = c_uw - _l2_cost(y[:v]) - _l2_cost(y[v:])
        d_vals.append(d)
        cand_dates.append(ma.index[v])

    if not cand_dates:
        return {"tau": shock, "score": 0.0, "method": "window_l2_nofound",
                "_df": pd.DataFrame(), "_threshold": 0.0}

    df_c = pd.DataFrame({"tau": cand_dates, "score": d_vals})
    best = df_c.loc[df_c["score"].idxmax()]

    return {
        "tau":        best["tau"],
        "score":      round(float(best["score"]), 6),
        "method":     "window_l2",
        "_df":        df_c,
        "_threshold": float(df_c["score"].max()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic plot — finestra di selezione del break (mode=detected)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_detection_window(
    detect_series: pd.Series,
    shock: pd.Timestamp,
    df_cands: pd.DataFrame,
    threshold: float,
    selected_tau: pd.Timestamp,
    fuel_key: str,
    fuel_color: str,
    ev_name: str,
    detect_target: str,
    metric_col: str,
    metric_label: str,
    out_path: "Path",
) -> None:
    """
    Salva un grafico diagnostico con due pannelli:
      · Top   : serie nella finestra di ricerca, tutte le τ candidate,
                la data di shock e il break selezionato evidenziato.
      · Bottom: curva dello score/|t| vs τ, soglia 75° percentile,
                zona top-quartile ombreggiata, break scelto con stella.
    """
    if df_cands.empty:
        return

    win_start = shock - pd.Timedelta(days=SEARCH + MIN_SEG + 5)
    win_end   = shock + pd.Timedelta(days=SEARCH + MIN_SEG + 5)
    series_win = detect_series[(detect_series.index >= win_start) &
                               (detect_series.index <= win_end)]

    fig, (ax_s, ax_m) = plt.subplots(2, 1, figsize=(12, 7),
                                     gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle(
        f"[v1 Naïve – detection finestra]  {ev_name}  ·  {fuel_key.capitalize()}\n"
        f"Serie: {'prezzo pompa netto' if detect_target == 'price' else ('MA-7 margine dist. (window-L2)' if detect_target == 'window_l2' else 'margine dist.')}"
        f"  |  Shock: {shock.date()}  |  Break scelto: {selected_tau.date()}"
        f"  ({(selected_tau - shock).days:+d}gg)",
        fontsize=9, fontweight="bold"
    )

    # ── Pannello 1: serie + candidati ────────────────────────────────────────
    ax_s.plot(series_win.index, series_win.values,
              color="grey", lw=0.9, alpha=0.7, label=detect_target)

    is_top = df_cands[metric_col] >= threshold
    for _, row in df_cands[~is_top].iterrows():
        ax_s.axvline(row["tau"], color="#cccccc", lw=0.4, alpha=0.5)
    for _, row in df_cands[is_top].iterrows():
        ax_s.axvline(row["tau"], color=fuel_color, lw=0.6, alpha=0.35)

    ax_s.axvline(shock, color="#e74c3c", lw=1.4, ls="--", label=f"Shock {shock.date()}")
    ax_s.axvline(selected_tau, color=fuel_color, lw=2.2,
                 label=f"Break scelto {selected_tau.date()} ({(selected_tau-shock).days:+d}gg)")

    y_rng = series_win.max() - series_win.min() if not series_win.empty else 1
    y_top = series_win.max() + 0.05 * y_rng if not series_win.empty else 1
    ax_s.annotate(
        f"τ = {selected_tau.date()}\n({(selected_tau-shock).days:+d}gg dallo shock)",
        xy=(selected_tau, y_top),
        xytext=(10, 0), textcoords="offset points",
        fontsize=7.5, color=fuel_color, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=fuel_color, lw=0.8),
    )
    ax_s.set_ylabel("€/L", fontsize=8)
    ax_s.legend(fontsize=7, loc="upper left")
    ax_s.grid(axis="y", alpha=0.20)
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_s.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ── Pannello 2: curva score vs τ ─────────────────────────────────────────
    taus   = pd.to_datetime(df_cands["tau"])
    scores = df_cands[metric_col].values

    ax_m.fill_between(taus, scores, threshold,
                      where=(scores >= threshold),
                      alpha=0.18, color=fuel_color, label="Top quartile (≥ 75°p)")
    ax_m.plot(taus, scores, color="steelblue", lw=1.0, zorder=3)
    ax_m.axhline(threshold, color="darkorange", lw=1.0, ls="--",
                 label=f"Soglia 75° pct = {threshold:.3f}")
    ax_m.axvline(shock, color="#e74c3c", lw=1.2, ls="--", alpha=0.7)
    ax_m.axvline(selected_tau, color=fuel_color, lw=1.8, alpha=0.9)

    sel_score = float(df_cands.loc[df_cands["tau"] == selected_tau, metric_col].iloc[0]
                      if not df_cands.loc[df_cands["tau"] == selected_tau].empty
                      else threshold)
    ax_m.scatter([selected_tau], [sel_score], marker="*", s=180,
                 color=fuel_color, zorder=5, label="Scelto (primo nel top quartile)")

    ax_m.set_ylabel(metric_label, fontsize=8)
    ax_m.set_xlabel("τ candidato", fontsize=8)
    ax_m.legend(fontsize=7, loc="upper left")
    ax_m.grid(axis="y", alpha=0.20)
    ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_m.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_m.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Detection window plot: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Metodo 1 – OLS Naïve
# ══════════════════════════════════════════════════════════════════════════════

def _fit_ols(series: pd.Series, break_date: pd.Timestamp) -> dict | None:
    """
    OLS lineare su PRE_WIN giorni prima di break_date.
    Nessuna correzione per autocorrelazione o eteroschedasticità.
    break_date può essere la shock date (mode=fixed) o il τ rilevato (mode=detected).
    """
    pre = series[
        (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
        (series.index < break_date)
    ].dropna()

    if len(pre) < 10:
        return None

    x = np.array([(d - break_date).days for d in pre.index], dtype=float)
    y = pre.values

    slope, intercept, r, _, _ = stats.linregress(x, y)
    y_hat     = slope * x + intercept
    residuals = y - y_hat
    n         = len(pre)
    mse       = np.sum(residuals**2) / (n - 2)
    sxx       = np.sum((x - x.mean())**2)

    # Design matrix per BG test (costante + t): shape (n, 2)
    X_bg = np.column_stack([np.ones(n), x])

    return dict(slope=slope, intercept=intercept, r2=float(r**2),
                mse=mse, sxx=sxx, x_mean=float(x.mean()), n=n,
                break_date=break_date, pre=pre,
                residuals=residuals, X_bg=X_bg)


def _project_ols(fit: dict, post_index: pd.DatetimeIndex
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Proietta la retta OLS + PI (1-α)% in avanti dal break_date."""
    x = np.array([(d - fit["break_date"]).days for d in post_index], dtype=float)

    baseline = fit["slope"] * x + fit["intercept"]

    se_pred = np.sqrt(
        fit["mse"] * (1 + 1/fit["n"] + (x - fit["x_mean"])**2 / fit["sxx"])
    )
    t_crit  = stats.t.ppf(1 - CI_ALPHA / 2, df=fit["n"] - 2)

    return (
        pd.Series(baseline,                  index=post_index),
        pd.Series(baseline - t_crit*se_pred, index=post_index),
        pd.Series(baseline + t_crit*se_pred, index=post_index),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plot per singolo evento + carburante
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name: str, ev: dict,
    series: pd.Series,
    fuel_key: str, fuel_color: str,
    fit: dict, baseline: pd.Series,
    ci_low: pd.Series, ci_high: pd.Series,
    extra: pd.Series, gain_meur: float,
    ax_main: plt.Axes, ax_gain: plt.Axes,
    break_date: pd.Timestamp, mode: str,
    break_score: float = np.nan,
) -> None:
    shock = ev["shock"]

    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0,
                 label=f"{fuel_key.capitalize()} effettivo")
    ax_main.plot(baseline.index, baseline.values, color="dimgrey", lw=1.3,
                 ls="--", label=f"Baseline OLS (R²={fit['r2']:.2f})")
    ax_main.fill_between(ci_low.index, ci_low.values, ci_high.values,
                         alpha=0.15, color="grey",
                         label=f"PI {int((1-CI_ALPHA)*100)}% (i.i.d.)")
    ax_main.fill_between(extra.index,
                         win.reindex(extra.index), baseline.values,
                         where=(extra >= 0), alpha=0.22, color="green",
                         label="Extra profitto (≥0)")
    ax_main.fill_between(extra.index,
                         win.reindex(extra.index), baseline.values,
                         where=(extra < 0), alpha=0.22, color="red",
                         label="Sotto-baseline (<0)")
    ax_main.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                    label=f"Shock ({shock.date()})")

    if mode == "detected" and break_date != shock:
        ax_main.axvline(break_date, color="black", lw=1.2, ls=":",
                        label=f"τ rilevato ({break_date.date()}, score={break_score:.2f})")

    mode_str = f"Break={break_date.date()} (naive score={break_score:.2f})" \
               if mode == "detected" else f"Break=shock ({shock.date()})"
    ax_main.set_title(
        f"[V1-Naïve / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  Baseline OLS  |  CI i.i.d.",
        fontsize=8, fontweight="bold"
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    cum = (extra * DAILY_CONSUMPTION_L[fuel_key] / 1e6).cumsum()
    ax_gain.plot(cum.index, cum.values, color=fuel_color, lw=1.2)
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum >= 0), alpha=0.25, color="green")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum < 0), alpha=0.25, color="red")
    ax_gain.set_title(
        f"Guadagno extra cumulato → {gain_meur:+.0f} M€  "
        f"({len(extra)}gg post-break)\n"
        f"[{DAILY_CONSUMPTION_L[fuel_key]/1e6:.0f} ML/giorno]",
        fontsize=7
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V1 Naïve OLS – ITS pipeline")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed = usa shock date; detected = sliding mean naive o window_l2")
    parser.add_argument("--detect", choices=["margin", "price", "window_l2"], default="margin",
                        help="(solo mode=detected) serie su cui cercare il break: "
                             "margin = margine distributore [default], "
                             "price  = prezzo pompa netto")
    args, _ = parser.parse_known_args()
    mode         = args.mode
    detect_target = args.detect  # ignorato se mode == "fixed"

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v1_naive"
    else:
        OUT_DIR = _OUT_BASE / mode / "v1_naive"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print(f"  02d_v1_naive.py  –  Metodo 1: OLS Naïve  [mode={mode}]")
    if mode == "fixed":
        print("  Break = shock date hardcodata (nessuna detection)")
    else:
        print(f"  Break = sliding mean naive  (ricerca ±{SEARCH}gg, finestra min {MIN_SEG}gg)")
        print(f"  Detection su: {'MARGINE distributore' if detect_target == 'margin' else 'PREZZO POMPA NETTO'}")
    print(f"  Finestra: PRE={PRE_WIN}gg / POST={POST_WIN}gg dal break point")
    print(f"  Output: {OUT_DIR}")
    print("═"*70)

    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig, axes = plt.subplots(len(FUELS), 2,
                                 figsize=(15, 5 * len(FUELS)),
                                 squeeze=False)
        fig.suptitle(
            f"[Metodo 1 – Naïve OLS / mode={mode}]  {ev_name}\n{ev['label']}",
            fontsize=11, fontweight="bold"
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            # ── Determina serie di detection ──────────────────────────────────
            if mode == "detected" and detect_target == "price":
                detect_series = data[PRICE_COLS[fuel_key]].dropna()
            else:
                detect_series = series  # margine (default) o window_l2 usa sempre il margine

            # ── Determina break date ──────────────────────────────────────────
            if mode == "detected":
                # Scelta del metodo di detection basata su --detect
                if detect_target == "window_l2":
                    # Paper BLOCCO 1: discrepanza L2 con MA a 7 giorni
                    bp = detect_breakpoint_window_l2(series, shock)
                    metric_col   = "score"
                    metric_label = "Discrepanza L2 d(y_uv, y_vw)"
                else:
                    # Metodo naïve originale (margin o price)
                    bp = detect_breakpoint_naive(detect_series, shock)
                    metric_col   = "score"
                    metric_label = "|Δmean| / pooled_std"

                break_date   = bp["tau"]
                break_method = bp["method"]
                break_score  = bp["score"]

                # ── Diagnostic: grafico finestra di selezione ─────────────────
                safe_ev  = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
                det_out  = OUT_DIR / f"detect_{safe_ev}_{fuel_key}.png"
                _plot_detection_window(
                    detect_series=detect_series if detect_target != "window_l2" else series,
                    shock=shock,
                    df_cands=bp["_df"],
                    threshold=bp["_threshold"],
                    selected_tau=break_date,
                    fuel_key=fuel_key,
                    fuel_color=fuel_color,
                    ev_name=ev_name,
                    detect_target=detect_target,
                    metric_col=metric_col,
                    metric_label=metric_label,
                    out_path=det_out,
                )
            else:
                break_date   = shock
                break_method = "fixed_at_shock"
                break_score  = np.nan

            pre_data = series[
                (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
                (series.index < break_date)
            ]
            post_data = series[
                (series.index >= break_date) &
                (series.index < break_date + pd.Timedelta(days=POST_WIN))
            ]

            if len(pre_data) < 10 or len(post_data) < 5:
                print(f"  [{fuel_key}] dati insufficienti – salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "Dati insufficienti",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            fit = _fit_ols(series, break_date)
            if fit is None:
                continue

            baseline, ci_low, ci_high = _project_ols(fit, post_data.index)
            extra     = post_data - baseline
            gain_meur = float(extra.sum() * DAILY_CONSUMPTION_L[fuel_key] / 1e6)
            gain_ci_low  = float((post_data - ci_high).sum()
                                 * DAILY_CONSUMPTION_L[fuel_key] / 1e6)
            gain_ci_high = float((post_data - ci_low).sum()
                                 * DAILY_CONSUMPTION_L[fuel_key] / 1e6)

            _plot_event_fuel(
                ev_name, ev, series, fuel_key, fuel_color,
                fit, baseline, ci_low, ci_high, extra, gain_meur,
                axes[row_idx][0], axes[row_idx][1],
                break_date=break_date, mode=mode, break_score=break_score,
            )

            # ── Diagnostics residui pre-periodo ──────────────────────────────
            pre_resid = fit.get("residuals", np.array([]))
            diag = run_diagnostic_tests(
                pre_resid,
                x_for_bg=fit.get("X_bg"),
                n_lags=None,
            )

            safe_ev = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
            diag_plot_out = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=pre_resid,
                dates=fit["pre"].index,
                title=(f"[V1-Naïve] Diagnostica residui OLS pre-periodo\n"
                       f"{ev_name} · {fuel_key.capitalize()}  "
                       f"(break={break_date.date()})"),
                out_path=diag_plot_out,
                diag_stats=diag,
            )

            print(f"\n  {ev_name}  [{fuel_key.upper()}]")
            print(f"    Break ({break_method}) = {break_date.date()}  (shock={shock.date()})")
            if not np.isnan(break_score):
                print(f"    Score naive           = {break_score:.4f}  "
                      f"(Δ={break_date.date()-shock.date()} dal shock)")
            print(f"    OLS R²                = {fit['r2']:.3f}  "
                  f"slope = {fit['slope']:+.5f} €/L/giorno")
            print(f"    Extra medio           = {extra.mean():+.4f} €/L/giorno")
            print(f"    Guadagno totale       = {gain_meur:+.0f} M€  "
                  f"CI90% [{gain_ci_low:+.0f}, {gain_ci_high:+.0f}] M€")
            sw_ok = not np.isnan(diag.get("sw_p", np.nan))
            lb_ok = not np.isnan(diag.get("lb_p", np.nan))
            if sw_ok:
                print(f"    SW (normalità)        = W={diag['sw_stat']:.3f}  "
                      f"p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠ non normal.'}")
            if lb_ok:
                print(f"    LB({diag['n_lags']}) (autocorr.)   = "
                      f"Q={diag['lb_stat']:.2f}  p={diag['lb_p']:.3f}  "
                      f"{'OK' if diag['lb_p'] > 0.05 else '⚠ autocorr.'}")
            bg_ok = not np.isnan(diag.get("bg_p", np.nan))
            if bg_ok:
                print(f"    BG({diag['n_lags']}) (autocorr.)   = "
                      f"LM={diag['bg_stat']:.2f}  p={diag['bg_p']:.3f}  "
                      f"{'OK' if diag['bg_p'] > 0.05 else '⚠ autocorr.'}")

            rows.append({
                "metodo":            "v1_naive",
                "mode":              mode,
                "detect_target":     detect_target if mode == "detected" else "fixed",
                "evento":            ev_name,
                "carburante":        fuel_key,
                "shock":             shock.date(),
                "break_date":        break_date.date(),
                "break_method":      break_method,
                "break_score":       round(break_score, 4) if not np.isnan(break_score) else np.nan,
                "pre_win_days":      PRE_WIN,
                "post_win_days":     POST_WIN,
                "n_pre":             len(pre_data),
                "n_post":            len(post_data),
                "extra_mean_eurl":   round(float(extra.mean()), 5),
                "extra_sum_eurl":    round(float(extra.sum()), 4),
                "gain_total_meur":   round(gain_meur, 1),
                "gain_ci_low_meur":  round(gain_ci_low, 1),
                "gain_ci_high_meur": round(gain_ci_high, 1),
                "r2_ols":            round(fit["r2"], 4),
                "slope_eurl_day":    round(fit["slope"], 6),
                "ci_type":           f"OLS_iid_{int((1-CI_ALPHA)*100)}pct",
                # ── Diagnostics ──────────────────────────────────────────────
                "sw_stat":           round(diag.get("sw_stat", np.nan), 4),
                "sw_p":              round(diag.get("sw_p", np.nan), 4),
                "lb_stat":           round(diag.get("lb_stat", np.nan), 3),
                "lb_p":              round(diag.get("lb_p", np.nan), 4),
                "bg_stat":           round(diag.get("bg_stat", np.nan), 3),
                "bg_p":              round(diag.get("bg_p", np.nan), 4),
                "diag_n_lags":       diag.get("n_lags", np.nan),
                "note": f"OLS naïve, residui i.i.d., mode={mode}"
                        + (f", detect={detect_target}" if mode == "detected" else ""),
            })

        fig.tight_layout()
        safe = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
        out  = OUT_DIR / f"plot_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v1_naive_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        print("\n" + df[["evento","carburante","break_date","gain_total_meur"]].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()