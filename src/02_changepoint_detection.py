"""
02_changepoint_detection.py
============================
Bayesian piecewise linear regression per i tre eventi di guerra.
Metodologia:
  - Variabile dipendente: ln(prezzo) con media lineare a tratti
  - Prior su τ:  Uniform(min(x), max(x))
  - Prior su σ:  HalfNormal(sd(y))
  - Prior su b:  StudentT(0, 3*sd(y), nu=3)
  - Prior su a:  StudentT(0, sd(y)/range(x), nu=3)
  - Stima via MCMC (PyMC / NUTS sampler)
  - CI al 95% = credible interval dalla distribuzione posteriore di τ
Plot paper-quality: un grafico per evento per serie.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymc as pm
import pytensor.tensor as pt
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

# MCMC settings — identici allo spirito del paper (mcp usa 3000 iter + 1000 tune)
MCMC_DRAWS  = 2000
MCMC_TUNE   = 1000
MCMC_CHAINS = 2


# ─────────────────────────────────────────
# Funzioni base (mantenute per DT e R²)
# ─────────────────────────────────────────
def fit_piecewise_ols(x, y, cp_idx):
    """OLS sui due segmenti — usato solo per R² e come punto di partenza."""
    def linreg(xv, yv):
        if len(xv) < 2:
            return 0., 0., 0.
        s, i, r, *_ = stats.linregress(xv, yv)
        return s, i, r**2
    b1, a1, r2_1 = linreg(x[:cp_idx], y[:cp_idx])
    b2, _,  r2_2 = linreg(x[cp_idx:], y[cp_idx:])
    return b1, b2, a1, r2_1, r2_2

def doubling_time(slope):
    """DT in giorni dato slope settimanale."""
    return np.log(2) / (slope / 7) if slope > 0 else np.inf


# ─────────────────────────────────────────
# MCMC Bayesiano
# ─────────────────────────────────────────
def bayesian_changepoint(x_vals, y_vals, alpha=0.05):
    """
    Modello piecewise lineare bayesiano con PyMC.

    Modello (identico al paper):
        ln(y) ~ N(mu, sigma²)
        mu = a1 + x*b1            se x < tau
        mu = a2 + x*b2            se x >= tau
        con a2 = a1 + tau*(b1-b2)  (no discontinuità — joined at changepoint)

    Prior (identici al paper):
        tau   ~ Uniform(min(x), max(x))
        sigma ~ HalfNormal(sd(y))
        b1,b2 ~ StudentT(0, 3*sd(y), nu=3)
        a1    ~ StudentT(0, sd(y)/range(x), nu=3)

    Restituisce dict con:
        tau_mean, tau_lo, tau_hi  (indice continuo)
        b1_mean, b1_lo, b1_hi
        b2_mean, b2_lo, b2_hi
        a1_mean
        tau_idx  (indice intero, MAP)
        trace    (oggetto ArviZ per diagnostica)
    """
    n     = len(x_vals)
    sd_y  = float(np.std(y_vals))
    rng_x = float(x_vals[-1] - x_vals[0])

    with pm.Model() as model:
        # ── Prior 
        tau   = pm.Uniform("tau",   lower=x_vals[0], upper=x_vals[-1])
        sigma = pm.HalfNormal("sigma", sigma=sd_y)
        b1    = pm.StudentT("b1",   mu=0, sigma=3 * sd_y,          nu=3)
        b2    = pm.StudentT("b2",   mu=0, sigma=3 * sd_y,          nu=3)
        a1    = pm.StudentT("a1",   mu=0, sigma=sd_y / max(rng_x, 1), nu=3)

        # ── Intercetta post-changepoint: a2 = a1 + tau*(b1-b2)
        #    (le due rette si congiungono in tau — no discontinuità)
        a2 = pm.Deterministic("a2", a1 + tau * (b1 - b2))

        # ── Media lineare a tratti (smooth step via sigmoid per differenziabilità)
        x_pt  = pt.as_tensor_variable(x_vals.astype(float))
        # step  = 1 se x >= tau, 0 se x < tau
        step   = pm.math.sigmoid((x_pt - tau) * 50)   # 50 = pendenza del gradino
        mu     = (a1 + b1 * x_pt) * (1 - step) + (a2 + b2 * x_pt) * step

        # ── Likelihood
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_vals)

        # ── MCMC (NUTS — default PyMC)
        trace = pm.sample(
            draws=MCMC_DRAWS,
            tune=MCMC_TUNE,
            chains=MCMC_CHAINS,
            progressbar=True,
            random_seed=42,
            target_accept=0.9,
            return_inferencedata=True,
        )

    # ── Estrai posteriori
    tau_post = trace.posterior["tau"].values.flatten()
    b1_post  = trace.posterior["b1"].values.flatten()
    b2_post  = trace.posterior["b2"].values.flatten()
    a1_post  = trace.posterior["a1"].values.flatten()

    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    return {
        # changepoint (indice continuo)
        "tau_mean": float(np.mean(tau_post)),
        "tau_lo":   float(np.percentile(tau_post, lo_pct)),
        "tau_hi":   float(np.percentile(tau_post, hi_pct)),
        # indice intero (MAP = mediana arrotondata)
        "tau_idx":  int(np.clip(round(float(np.median(tau_post))), 1, n - 2)),
        # slopes
        "b1_mean":  float(np.mean(b1_post)),
        "b1_lo":    float(np.percentile(b1_post, lo_pct)),
        "b1_hi":    float(np.percentile(b1_post, hi_pct)),
        "b2_mean":  float(np.mean(b2_post)),
        "b2_lo":    float(np.percentile(b2_post, lo_pct)),
        "b2_hi":    float(np.percentile(b2_post, hi_pct)),
        # intercetta
        "a1_mean":  float(np.mean(a1_post)),
        # oggetto trace (per diagnostica)
        "trace":    trace,
    }


# ─────────────────────────────────────────
# RUN PRINCIPALE
# ─────────────────────────────────────────
SERIES = [("Brent", "log_brent"), ("Benzina", "log_benzina"), ("Diesel", "log_diesel")]

results      = []
summary_rows = []

print(f"{'EVENTO':<28} {'SERIE':<10} {'TAU':<14} {'CI 95%':<22} {'LAG (gg)':<10} {'H0'}")
print("=" * 90)

for event_name, cfg in EVENTS.items():
    for series_name, log_col in SERIES:
        if log_col not in merged.columns:
            continue

        df = merged.loc[cfg["window_start"]:cfg["window_end"], log_col].dropna()
        if len(df) < 10:
            continue

        x_vals = np.arange(len(df), dtype=float)
        y_vals = df.values.astype(float)

        print(f"\n  MCMC Bayesiano: {event_name} | {series_name}...")
        ci = bayesian_changepoint(x_vals, y_vals)

        # ── Changepoint: usa mediana posteriore come stima puntuale (come il paper)
        cp_idx  = ci["tau_idx"]
        cp_date = df.index[cp_idx]

        # ── CI in date
        cp_lo_idx  = int(np.clip(round(ci["tau_lo"]),  0, len(df) - 1))
        cp_hi_idx  = int(np.clip(round(ci["tau_hi"]), 0, len(df) - 1))
        cp_lo_date = df.index[cp_lo_idx]
        cp_hi_date = df.index[cp_hi_idx]

        shock = pd.Timestamp(cfg["shock_date"])
        lag   = (cp_date - shock).days

        # ── OLS sui segmenti per R² e DT (usa slope bayesiana come stima principale)
        b1 = ci["b1_mean"]
        b2 = ci["b2_mean"]
        a1 = ci["a1_mean"]
        _, _, _, r2_1, r2_2 = fit_piecewise_ols(x_vals, y_vals, cp_idx)

        h0_status = "RIFIUTATA" if lag < H0_THRESHOLD else "non rifiutata"
        ci_str    = f"[{cp_lo_date.strftime('%d %b %y')} – {cp_hi_date.strftime('%d %b %y')}]"

        print(f"{event_name:<28} {series_name:<10} {str(cp_date.date()):<14} "
              f"{ci_str:<22} {lag:<10} {h0_status}")

        results.append({
            **cfg,
            "event":      event_name,
            "series":     series_name,
            "cp_date":    cp_date,
            "cp_idx":     cp_idx,
            "lag_days":   lag,
            "b1":         b1, "b2": b2, "a1": a1,
            "dt1":        doubling_time(b1), "dt2": doubling_time(b2),
            "r2_1":       r2_1, "r2_2": r2_2,
            "df":         df, "x": x_vals,
            "ci":         ci,
            "cp_lo_date": cp_lo_date,
            "cp_hi_date": cp_hi_date,
        })

        summary_rows.append({
            "Evento":      event_name,
            "Serie":       series_name,
            "tau":         cp_date.date(),
            "CI_95_lo":    cp_lo_date.date(),
            "CI_95_hi":    cp_hi_date.date(),
            "Shock":       shock.date(),
            "Lag (gg)":    lag,
            "b1":          round(b1, 4),
            "b1_CI_lo":    round(ci["b1_lo"], 4),
            "b1_CI_hi":    round(ci["b1_hi"], 4),
            "b2":          round(b2, 4),
            "b2_CI_lo":    round(ci["b2_lo"], 4),
            "b2_CI_hi":    round(ci["b2_hi"], 4),
            "DT1 (gg)":    round(doubling_time(b1), 1) if doubling_time(b1) != np.inf else "inf",
            "DT2 (gg)":    round(doubling_time(b2), 1) if doubling_time(b2) != np.inf else "inf",
            "R2_pre":      round(r2_1, 3),
            "R2_post":     round(r2_2, 3),
            "H0":          h0_status,
        })


# ─────────────────────────────────────────
# LAG D = tau_retail - tau_crude
# ─────────────────────────────────────────
print("\nLAG D = tau_retail - tau_crude")
print("─" * 55)
lag_rows = []
for event_name in EVENTS:
    by_s = {r["series"]: r for r in results if r["event"] == event_name}
    if "Brent" not in by_s:
        continue
    tau_crude = by_s["Brent"]["cp_date"]
    for fuel in ["Benzina", "Diesel"]:
        if fuel not in by_s:
            continue
        D    = (by_s[fuel]["cp_date"] - tau_crude).days
        flag = "SPECULAZIONE" if D < H0_THRESHOLD else "compatibile con logistica"
        print(f"  {event_name} | {fuel}: D = {D:+d} gg → {flag}")
        lag_rows.append({
            "Evento":      event_name,
            "Carburante":  fuel,
            "tau_crude":   tau_crude.date(),
            "tau_retail":  by_s[fuel]["cp_date"].date(),
            "D (gg)":      D,
            "H0":          "RIFIUTATA" if D < H0_THRESHOLD else "non rifiutata",
        })

pd.DataFrame(lag_rows).to_csv("data/lag_results.csv", index=False)
pd.DataFrame(summary_rows).to_csv("data/table1_changepoints.csv", index=False)
ci_method_label = "Bayesian MCMC (PyMC/NUTS) — credible interval 95% posteriore"


# ─────────────────────────────────────────
# PLOT — un grafico per (evento × serie)
# ─────────────────────────────────────────
for res in results:
    df_plot = res["df"]
    x       = res["x"]
    cp      = res["cp_idx"]
    b1, b2, a1 = res["b1"], res["b2"], res["a1"]
    a2      = a1 + cp * (b1 - b2)
    ci      = res["ci"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Dati grezzi
    ax.scatter(df_plot.index, df_plot.values,
               s=18, color="black", alpha=0.55, zorder=3, label="log(prezzo)")

    # Retta pre-shock + banda CI slopes (bayesiana)
    ci_b1_lo = ci["b1_lo"]
    ci_b1_hi = ci["b1_hi"]
    ax.plot(df_plot.index[:cp], a1 + b1 * x[:cp],
            color="#27ae60", lw=2.5,
            label=f"Pre-shock  b₁ = {b1:.4f} [{ci_b1_lo:.4f}, {ci_b1_hi:.4f}]")
    ax.fill_between(df_plot.index[:cp],
                    a1 + ci_b1_lo * x[:cp],
                    a1 + ci_b1_hi * x[:cp],
                    color="#27ae60", alpha=0.12)

    # Retta post-shock + banda CI slopes
    ci_b2_lo = ci["b2_lo"]
    ci_b2_hi = ci["b2_hi"]
    ax.plot(df_plot.index[cp:], a2 + b2 * x[cp:],
            color="#e74c3c", lw=2.5,
            label=f"Post-shock b₂ = {b2:.4f} [{ci_b2_lo:.4f}, {ci_b2_hi:.4f}]")
    ax.fill_between(df_plot.index[cp:],
                    a2 + ci_b2_lo * x[cp:],
                    a2 + ci_b2_hi * x[cp:],
                    color="#e74c3c", alpha=0.12)

    # Changepoint τ (linea + area CI posteriore)
    ax.axvline(res["cp_date"], color="#2980b9", lw=2.2, linestyle="--",
               label=f"τ = {res['cp_date'].date()}")
    ax.axvspan(res["cp_lo_date"], res["cp_hi_date"],
               alpha=0.15, color="#2980b9",
               label=f"CI 95% τ: [{res['cp_lo_date'].strftime('%d %b %y')} – "
                     f"{res['cp_hi_date'].strftime('%d %b %y')}]")

    # Shock date
    ax.axvline(pd.Timestamp(res["shock_date"]), color=res["color"], lw=2.0,
               linestyle=":", label=f"Shock = {res['shock_date']}")

    lag   = res["lag_days"]
    dt1_s = str(round(res["dt1"], 1)) if res["dt1"] != np.inf else "∞"
    dt2_s = str(round(res["dt2"], 1)) if res["dt2"] != np.inf else "∞"

    ax.set_title(
        f"{res['event']} — {res['series']}\n"
        f"D = {lag:+d} giorni  |  DT₁ = {dt1_s} gg  →  DT₂ = {dt2_s} gg",
        fontsize=13, fontweight="bold",
        color="#c0392b" if lag < H0_THRESHOLD else "black",
    )
    ax.set_ylabel("log(prezzo)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, fontsize=10)

    safe_event  = (res["event"]
                   .replace(" ", "_")
                   .replace("(", "").replace(")", "")
                   .replace("/", ""))
    safe_series = res["series"].lower()
    fname = f"plots/02_{safe_event}_{safe_series}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {fname}")

print(f"\nScript 02 completato. Metodo CI: {ci_method_label}")