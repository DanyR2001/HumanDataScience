"""
02_changepoint_detection.py
============================
Per ogni evento di guerra:
  1. Estrai finestra temporale [-60gg, +90gg] dallo shock
  2. Applica changepoint detection (ruptures PELT + Bayesian)
  3. Calcola τ_crude e τ_retail
  4. Calcola D = τ_retail - τ_crude
  5. Test: D << 30 giorni? → rifiuto H₀
  6. Calcola pendenze b1, b2 e doubling time (come nel paper)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ruptures as rpt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Carica dati
# ─────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
print(f"Dataset caricato: {len(merged)} settimane")

# ─────────────────────────────────────────
# Definizione eventi di guerra
# ─────────────────────────────────────────
EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock_date": "2022-02-24",
        "window_start": "2021-11-01",   # 4 mesi prima
        "window_end":   "2022-06-30",   # 4 mesi dopo
        "color":        "#e74c3c",
    },
    "Hormuz (Feb 2026)": {
        "shock_date": "2026-02-28",
        "window_start": "2025-11-01",
        "window_end":   "2026-03-17",   # fine dataset
        "color":        "#8e44ad",
    },
}

H0_THRESHOLD = 30  # giorni — soglia fisica della supply chain


# ─────────────────────────────────────────
# Funzione: regressione piecewise lineare
# ─────────────────────────────────────────
def fit_piecewise(x: np.ndarray, y: np.ndarray, breakpoint_idx: int):
    """
    Fitta due rette prima e dopo il changepoint.
    Restituisce: (b1, b2, a1, r2_before, r2_after)
    """
    idx = breakpoint_idx

    # Prima del changepoint
    x1, y1 = x[:idx], y[:idx]
    # Dopo il changepoint
    x2, y2 = x[idx:], y[idx:]

    def linreg(xv, yv):
        if len(xv) < 2:
            return 0.0, 0.0, 0.0
        slope, intercept, r, *_ = stats.linregress(xv, yv)
        return slope, intercept, r**2

    b1, a1, r2_1 = linreg(x1, y1)
    b2, a2, r2_2 = linreg(x2, y2)
    return b1, b2, a1, r2_1, r2_2


def doubling_time(slope: float) -> float:
    """
    Giorni necessari per raddoppiare i casi.
    DT = ln(2) / slope (in unità di tempo usate nell'asse x)
    Restituisce inf se slope <= 0
    """
    if slope <= 0:
        return np.inf
    return np.log(2) / slope


# ─────────────────────────────────────────
# Funzione principale di analisi per evento
# ─────────────────────────────────────────
def analyze_event(event_name: str, cfg: dict, series_name: str, log_col: str):
    """
    Applica changepoint detection su una finestra temporale.
    serie_name: es. "Brent" o "Benzina"
    log_col: nome colonna log nel dataframe
    """
    df = merged.loc[cfg["window_start"]:cfg["window_end"], log_col].dropna()
    if len(df) < 8:
        print(f"     Dati insufficienti per {series_name} in {event_name}")
        return None

    signal = df.values.reshape(-1, 1)
    x_idx  = np.arange(len(df))

    # Changepoint detection con algoritmo PELT (Pruned Exact Linear Time)
    # Penalità BIC-like per selezionare automaticamente il numero di breakpoint
    algo = rpt.Pelt(model="rbf").fit(signal)
    try:
        breaks = algo.predict(pen=3)          # pen=3 ≈ BIC, restituisce 1 changepoint
        if len(breaks) > 1:
            cp_idx = breaks[0]                # primo changepoint significativo
        else:
            # Fallback: usa Binseg con 1 breakpoint forzato
            algo2 = rpt.Binseg(model="l2").fit(signal)
            cp_idx = algo2.predict(n_bkps=1)[0]
    except Exception:
        algo2 = rpt.Binseg(model="l2").fit(signal)
        cp_idx = algo2.predict(n_bkps=1)[0]

    cp_idx = min(cp_idx, len(df) - 2)        # sicurezza bordo

    cp_date    = df.index[cp_idx]
    shock_date = pd.Timestamp(cfg["shock_date"])
    lag_days   = (cp_date - shock_date).days

    # Fit piecewise regression
    b1, b2, a1, r2_1, r2_2 = fit_piecewise(x_idx, df.values, cp_idx)
    dt1 = doubling_time(b1 * 7)              # converti da weekly a giorni
    dt2 = doubling_time(b2 * 7)

    result = {
        "event":       event_name,
        "series":      series_name,
        "shock_date":  shock_date,
        "cp_date":     cp_date,
        "cp_idx":      cp_idx,
        "lag_days":    lag_days,
        "b1":          b1,
        "b2":          b2,
        "a1":          a1,
        "dt1_giorni":  dt1,
        "dt2_giorni":  dt2,
        "r2_before":   r2_1,
        "r2_after":    r2_2,
        "df":          df,
        "x_idx":       x_idx,
    }
    return result


# ─────────────────────────────────────────
# RUN: analisi per tutti gli eventi
# ─────────────────────────────────────────
results = []
series_map = [
    ("Brent",    "log_brent"),
    ("Benzina",  "log_benzina"),
    ("Diesel",   "log_diesel"),
]

print("\n" + "="*65)
print(f"{'EVENTO':<25} {'SERIE':<10} {'TAU':<14} {'LAG (gg)':<10} {'H₀':<12}")
print("="*65)

for event_name, cfg in EVENTS.items():
    for series_name, log_col in series_map:
        if log_col not in merged.columns:
            continue
        res = analyze_event(event_name, cfg, series_name, log_col)
        if res is None:
            continue
        results.append(res)

        h0_status = "RIFIUTATA ✓" if res["lag_days"] < H0_THRESHOLD else "non rifiutata"
        print(f"{event_name:<25} {series_name:<10} {str(res['cp_date'].date()):<14} "
              f"{res['lag_days']:<10} {h0_status}")

print("="*65)


# ─────────────────────────────────────────
# CALCOLO LAG D = τ_retail - τ_crude
# ─────────────────────────────────────────
print("\n" + "─"*50)
print("LAG D = τ_retail − τ_crude (in giorni)")
print("─"*50)

lag_table = []

for event_name in EVENTS:
    res_by_series = {r["series"]: r for r in results if r["event"] == event_name}

    if "Brent" not in res_by_series:
        continue

    tau_crude = res_by_series["Brent"]["cp_date"]

    for fuel in ["Benzina", "Diesel"]:
        if fuel not in res_by_series:
            continue
        tau_retail = res_by_series[fuel]["cp_date"]
        D = (tau_retail - tau_crude).days

        lag_table.append({
            "Evento":      event_name,
            "Carburante":  fuel,
            "τ_crude":     tau_crude.date(),
            "τ_retail":    tau_retail.date(),
            "D (giorni)":  D,
            "H₀ (≥30gg)":  "RIFIUTATA" if D < H0_THRESHOLD else "non rifiutata",
        })

        print(f"  {event_name} | {fuel}: D = {D:+d} giorni → "
              f"{'  SPECULAZIONE (D < 30gg)' if D < H0_THRESHOLD else 'compatibile con logistica'}")

lag_df = pd.DataFrame(lag_table)
lag_df.to_csv("data/lag_results.csv", index=False)
print(f"\n  Risultati salvati in data/lag_results.csv")


# ─────────────────────────────────────────
# PLOT: regressione piecewise (come fig.1 del paper)
# ─────────────────────────────────────────
n_plots = len(results)
if n_plots > 0:
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        df_plot = res["df"]
        x       = res["x_idx"]
        cp      = res["cp_idx"]
        b1, b2, a1 = res["b1"], res["b2"], res["a1"]

        # Dati grezzi (log)
        ax.scatter(df_plot.index, df_plot.values, s=10, color="black",
                   alpha=0.6, label="log(prezzo)", zorder=3)

        # Retta pre-changepoint (verde)
        x1 = x[:cp]
        y1_fit = a1 + b1 * x1
        ax.plot(df_plot.index[:cp], y1_fit, color="green", lw=2,
                label=f"b₁={b1:.4f}")

        # Retta post-changepoint (rosso)
        a2 = a1 + cp * (b1 - b2)
        x2 = x[cp:]
        y2_fit = a2 + b2 * x2
        ax.plot(df_plot.index[cp:], y2_fit, color="red", lw=2,
                label=f"b₂={b2:.4f}")

        # Changepoint (blu verticale)
        ax.axvline(res["cp_date"], color="blue", lw=2, linestyle="--",
                   label=f"τ = {res['cp_date'].date()}")

        # Shock date (arancione)
        ax.axvline(res["shock_date"], color="orange", lw=1.5,
                   linestyle=":", label=f"Shock = {res['shock_date'].date()}")

        # Titolo con lag
        lag = res["lag_days"]
        color_title = "red" if lag < H0_THRESHOLD else "black"
        ax.set_title(f"{res['event']} | {res['series']}\nD = {lag:+d}gg",
                     fontsize=9, color=color_title)

        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)
        ax.set_ylabel("log(prezzo)", fontsize=8)

    # Nascondi assi vuoti
    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Changepoint detection — Regressione piecewise lineare\n"
        "(verde = pre-shock, rosso = post-shock, blu = changepoint, arancione = data guerra)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("plots/02_changepoints.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n Plot salvato: plots/02_changepoints.png")


# ─────────────────────────────────────────
# TABELLA RIASSUNTIVA (come Table 1 del paper)
# ─────────────────────────────────────────
summary_rows = []
for res in results:
    summary_rows.append({
        "Evento":         res["event"],
        "Serie":          res["series"],
        "τ (changepoint)": res["cp_date"].date(),
        "Shock date":     res["shock_date"].date(),
        "Lag (giorni)":   res["lag_days"],
        "b₁ (pre)":       round(res["b1"], 4),
        "b₂ (post)":      round(res["b2"], 4),
        "DT₁ (giorni)":   round(res["dt1_giorni"], 1) if res["dt1_giorni"] != np.inf else "∞",
        "DT₂ (giorni)":   round(res["dt2_giorni"], 1) if res["dt2_giorni"] != np.inf else "∞",
        "R² pre":         round(res["r2_before"], 3),
        "R² post":        round(res["r2_after"], 3),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("data/table1_changepoints.csv", index=False)
print("\n Table 1:")
print(summary_df.to_string(index=False))
print("\n Script 02 completato.")