"""
02_changepoint_detection.py
============================
Bayesian piecewise linear regression per i tre eventi di guerra.
Plot paper-quality: un grafico per evento per serie.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ruptures as rpt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane\n")

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock_date":   "2022-02-24",
        "window_start": "2021-10-01",
        "window_end":   "2022-07-31",
        "color":        "#e74c3c",
    },
    "Iran-Israele (Giu 2025)": {
        "shock_date":   "2025-06-13",
        "window_start": "2025-02-01",
        "window_end":   "2025-10-31",
        "color":        "#e67e22",
    },
    "Hormuz (Feb 2026)": {
        "shock_date":   "2026-02-28",
        "window_start": "2025-10-01",
        "window_end":   "2026-03-17",
        "color":        "#8e44ad",
    },
}

H0_THRESHOLD = 30
DPI          = 180
FIGSIZE      = (12, 5)

def fit_piecewise(x, y, cp_idx):
    def linreg(xv, yv):
        if len(xv) < 2: return 0., 0., 0.
        s, i, r, *_ = stats.linregress(xv, yv)
        return s, i, r**2
    b1, a1, r2_1 = linreg(x[:cp_idx], y[:cp_idx])
    b2, a2, r2_2 = linreg(x[cp_idx:], y[cp_idx:])
    return b1, b2, a1, r2_1, r2_2

def doubling_time(slope):
    return np.log(2) / slope if slope > 0 else np.inf

def detect_cp(series_values):
    sig = series_values.reshape(-1, 1)
    try:
        breaks = rpt.Pelt(model="rbf").fit(sig).predict(pen=3)
        cp = breaks[0] if len(breaks) > 1 else rpt.Binseg(model="l2").fit(sig).predict(n_bkps=1)[0]
    except Exception:
        cp = rpt.Binseg(model="l2").fit(sig).predict(n_bkps=1)[0]
    return min(cp, len(series_values) - 2)

SERIES = [("Brent", "log_brent"), ("Benzina", "log_benzina"), ("Diesel", "log_diesel")]

results     = []
summary_rows = []

print(f"{'EVENTO':<28} {'SERIE':<10} {'TAU':<14} {'LAG (gg)':<10} {'H0'}")
print("="*70)

for event_name, cfg in EVENTS.items():
    for series_name, log_col in SERIES:
        if log_col not in merged.columns:
            continue
        df = merged.loc[cfg["window_start"]:cfg["window_end"], log_col].dropna()
        if len(df) < 8:
            continue

        cp_idx     = detect_cp(df.values)
        cp_date    = df.index[cp_idx]
        shock_date = pd.Timestamp(cfg["shock_date"])
        lag        = (cp_date - shock_date).days
        x          = np.arange(len(df))
        b1, b2, a1, r2_1, r2_2 = fit_piecewise(x, df.values, cp_idx)

        h0 = "RIFIUTATA" if lag < H0_THRESHOLD else "non rifiutata"
        print(f"{event_name:<28} {series_name:<10} {str(cp_date.date()):<14} {lag:<10} {h0}")

        results.append({**cfg,
            "event": event_name, "series": series_name,
            "cp_date": cp_date, "cp_idx": cp_idx, "lag_days": lag,
            "b1": b1, "b2": b2, "a1": a1,
            "dt1": doubling_time(b1*7), "dt2": doubling_time(b2*7),
            "r2_1": r2_1, "r2_2": r2_2, "df": df, "x": x,
        })
        summary_rows.append({
            "Evento": event_name, "Serie": series_name,
            "tau": cp_date.date(), "Shock": shock_date.date(),
            "Lag (gg)": lag, "b1": round(b1,4), "b2": round(b2,4),
            "DT1 (gg)": round(doubling_time(b1*7),1) if doubling_time(b1*7) != np.inf else "inf",
            "DT2 (gg)": round(doubling_time(b2*7),1) if doubling_time(b2*7) != np.inf else "inf",
            "R2_pre": round(r2_1,3), "R2_post": round(r2_2,3),
        })

# Lag D = tau_retail - tau_crude
print("\nLAG D = tau_retail - tau_crude")
print("─"*50)
lag_rows = []
for event_name in EVENTS:
    by_series = {r["series"]: r for r in results if r["event"] == event_name}
    if "Brent" not in by_series:
        continue
    tau_crude = by_series["Brent"]["cp_date"]
    for fuel in ["Benzina", "Diesel"]:
        if fuel not in by_series:
            continue
        D = (by_series[fuel]["cp_date"] - tau_crude).days
        flag = "SPECULAZIONE" if D < H0_THRESHOLD else "compatibile"
        print(f"  {event_name} | {fuel}: D = {D:+d} gg → {flag}")
        lag_rows.append({"Evento": event_name, "Carburante": fuel,
                         "tau_crude": tau_crude.date(),
                         "tau_retail": by_series[fuel]["cp_date"].date(),
                         "D (gg)": D, "H0": "RIFIUTATA" if D < H0_THRESHOLD else "non rifiutata"})

pd.DataFrame(lag_rows).to_csv("data/lag_results.csv", index=False)
pd.DataFrame(summary_rows).to_csv("data/table1_changepoints.csv", index=False)


# ─────────────────────────────────────────
# PLOT: un grafico per ogni (evento x serie)
# ─────────────────────────────────────────
for res in results:
    df_plot = res["df"]
    x       = res["x"]
    cp      = res["cp_idx"]
    b1, b2, a1 = res["b1"], res["b2"], res["a1"]
    a2      = a1 + cp * (b1 - b2)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Log prezzi grezzi
    ax.scatter(df_plot.index, df_plot.values, s=18, color="black", alpha=0.55,
               zorder=3, label="log(prezzo)")

    # Retta pre-changepoint
    ax.plot(df_plot.index[:cp],
            a1 + b1 * x[:cp],
            color="#27ae60", lw=2.5, label=f"Pre-shock  b₁ = {b1:.4f}")

    # Retta post-changepoint
    ax.plot(df_plot.index[cp:],
            a2 + b2 * x[cp:],
            color="#e74c3c", lw=2.5, label=f"Post-shock b₂ = {b2:.4f}")

    # Changepoint
    ax.axvline(res["cp_date"], color="#2980b9", lw=2.2, linestyle="--",
               label=f"Changepoint τ = {res['cp_date'].date()}")

    # Shock
    ax.axvline(pd.Timestamp(res["shock_date"]), color=res["color"], lw=2.0,
               linestyle=":", label=f"Shock = {res['shock_date']}")

    lag = res["lag_days"]
    dt2 = res["dt2"]
    dt2_str = f"{dt2:.1f}" if dt2 != np.inf else "∞"

    ax.set_title(
        f"{res['event']} — {res['series']}\n"
        f"D = {lag:+d} giorni  |  DT₁ = {str(round(res['dt1'],1)) if res['dt1']!=np.inf else 'inf'}gg  →  "
        f"DT₂ = {dt2_str}gg",
        fontsize=13, fontweight="bold",
        color="#c0392b" if lag < H0_THRESHOLD else "black"
    )
    ax.set_ylabel("log(prezzo)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, fontsize=10)

    safe_event  = res["event"].replace(" ", "_").replace("(","").replace(")","").replace("/","")
    safe_series = res["series"].lower()
    fname = f"plots/02_{safe_event}_{safe_series}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {fname}")

print("\nScript 02 completato.")