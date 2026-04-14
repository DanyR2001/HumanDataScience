"""
07_margine_speculation_test.py  (v3 — unità corrette + split su shock date)
============================================================================
TEST DI SPECULAZIONE vs ANTICIPAZIONE RAZIONALE
================================================

CORREZIONI rispetto alla v2:

  [BUG CRITICO] Il margine in v2 era la differenza tra:
    - benzina_4w     → EUR / 1000 litri  (come pubblicato da EU Bulletin)
    - costo Brent    → EUR / litro        (formula mal dimensionata)
  Risultato: 500 - 0.21 ≈ 499.79  → margine ≈ prezzo pompa, privo di senso.

  Correzioni v3:
  1. Rilevamento automatico unità: se benzina_4w > 10 → EUR/1000L → dividi per 1000
     prima di qualsiasi calcolo sul margine.
  2. Split pre/post sul SHOCK DATE (non sul tau MCMC):
     - Il MCMC sul margine non converge quando non c'è un chiaro break strutturale
       (tau scivola al bordo della finestra → t-test p = nan, lag = -136gg)
     - Confrontare pre/post rispetto alla DATA DELLO SHOCK è la domanda economica
       corretta: "il margine è cambiato DOPO lo shock?" (Borenstein et al. 1997)
  3. Rilevamento "no changepoint": se tau cade nel primo o ultimo 15% della finestra
     → flaggato come "margine senza break strutturale" → più informativo di nan.
  4. MCMC sul margine rimane per stimare QUANDO cambia il margine (se cambia),
     ma il test principale di anomalia è sul confronto pre/post shock date.
  5. Rimosso il MCMC sui futures: sostituito con confronto semplice interpretabile.

Metodologia (Borenstein, Cameron & Gilbert 1997 — replacement cost):
  margine_benzina_eur_l = prezzo_pompa_eur_l - costo_rimpiazzo_eur_l
  costo_rimpiazzo       = (brent_futures_eur / 159) × YIELD_GASOLINE
  YIELD_GASOLINE = 0.45  (IEA/ARERA)

  Se crack spread RBOB disponibile → metodo preferito:
  margine_benzina = prezzo_pompa_eur_l - rbob_wholesale_eur_l

Test statistici:
  • Welch t-test (pre vs post shock)
  • KS 2-campioni
  • Bootstrap 1000 campioni (CI 95% su Δmargine)

Classificazione:
  SPECULAZIONE        → Δmargine > 2σ, p < 0.05 (t o KS), CI esclude 0
  COMPRESSIONE MARGINE→ Δmargine < −2σ, p < 0.05
  ANTICIPAZIONE RAZ.  → Δmargine ≤ 2σ o p ≥ 0.05
  INCONCLUSIVO        → segnali contrastanti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ─── Configurazione ──────────────────────────────────────────────────────────
DPI          = 180
MCMC_DRAWS   = 2000
MCMC_TUNE    = 1000
MCMC_CHAINS  = 2
ALPHA        = 0.05

BARREL_LITRES  = 159.0
GAL_LITRES     = 3.78541
YIELD_GASOLINE = 0.45
YIELD_DIESEL   = 0.52

BASELINE_START = "2021-01-01"
BASELINE_END   = "2021-12-31"
EDGE_FRACTION  = 0.15   # tau in prima/ultima 15% della finestra → "no break"

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

SERIES = [("Benzina", "benzina_4w", YIELD_GASOLINE),
          ("Diesel",  "diesel_4w",  YIELD_DIESEL)]

CLAS_COLOR = {
    "SPECULAZIONE":                       "#c0392b",
    "COMPRESSIONE MARGINE":               "#2980b9",
    "ANTICIPAZIONE RAZIONALE / NEUTRO":   "#27ae60",
    "VARIAZIONE STATISTICA (non anomala)":"#e67e22",
    "INCONCLUSIVO":                       "#95a5a6",
}


# ─────────────────────────────────────────
# 1. CARICA DATASET
# ─────────────────────────────────────────
print("Caricamento dataset_merged.csv e table1_changepoints.csv...")
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
table1 = pd.read_csv("data/table1_changepoints.csv")


# ─────────────────────────────────────────
# 2. RILEVAMENTO AUTOMATICO UNITÀ PREZZI POMPA
#    EU Bulletin pubblica in EUR/1000L → benzina_4w > 10 → converti
# ─────────────────────────────────────────
pump_sample = merged["benzina_4w"].dropna().mean()
if pump_sample > 10:
    unit_factor = 1000.0
    unit_label  = f"EUR/1000L (media={pump_sample:.1f}) → convertito in EUR/L"
else:
    unit_factor = 1.0
    unit_label  = f"EUR/L (media={pump_sample:.4f}) → nessuna conversione"

merged["benzina_eur_l_norm"] = merged["benzina_4w"] / unit_factor
merged["diesel_eur_l_norm"]  = merged["diesel_4w"]  / unit_factor
print(f"\nUnità pompa: {unit_label}")
print(f"  Benzina EUR/L post-conversione: {merged['benzina_eur_l_norm'].dropna().mean():.4f}")
print(f"  Diesel  EUR/L post-conversione: {merged['diesel_eur_l_norm'].dropna().mean():.4f}")


# ─────────────────────────────────────────
# 3. EUR/USD
# ─────────────────────────────────────────
print("\nRecupero EUR/USD...")
if "eurusd" in merged.columns and merged["eurusd"].dropna().mean() > 0.5:
    print("  EUR/USD già presente in dataset_merged")
else:
    try:
        eurusd_raw = yf.download("EURUSD=X", start="2021-01-01",
                                  end="2026-03-20", progress=False)
        eurusd_w = (eurusd_raw[["Close"]]
                    .rename(columns={"Close": "eurusd_new"})
                    .resample("W-MON").mean().ffill())
        merged = merged.join(eurusd_w, how="left")
        merged["eurusd"] = merged.get("eurusd", merged["eurusd_new"])
        merged["eurusd"] = merged["eurusd"].ffill().bfill()
        print("  EUR/USD scaricato da yfinance")
    except Exception as e:
        if "eurusd" not in merged.columns:
            merged["eurusd"] = 1.08
        print(f"  Fallback EUR/USD=1.08 ({e})")


# ─────────────────────────────────────────
# 4. BRENT FUTURES EUR/LITRO
# ─────────────────────────────────────────
print("\nBrent Futures (BZ=F)...")
try:
    fut_raw = yf.download("BZ=F", start="2021-01-01",
                           end="2026-03-20", progress=False)
    if fut_raw.empty:
        raise ValueError("download vuoto")

    fut_w = (fut_raw[["Close"]]
             .rename(columns={"Close": "brent_fut_usd"})
             .resample("W-MON").mean().ffill())

    if "brent_fut_usd" not in merged.columns:
        merged = merged.join(fut_w, how="left")
    else:
        merged["brent_fut_usd"] = merged["brent_fut_usd"].fillna(
            fut_w["brent_fut_usd"])

    merged["brent_fut_usd"] = merged["brent_fut_usd"].ffill()
    merged["brent_fut_eur"] = merged["brent_fut_usd"] / merged["eurusd"]
    merged["brent_fut_eur_l"] = merged["brent_fut_eur"] / BARREL_LITRES

    fut_l_mean = merged["brent_fut_eur_l"].dropna().mean()
    print(f"  Futures EUR/L media: {fut_l_mean:.4f} (atteso ~0.40-0.80)")
    if not (0.1 < fut_l_mean < 2.0):
        raise ValueError(f"Valore anomalo: {fut_l_mean:.4f}")
except Exception as e:
    print(f"  Futures non disp. ({e}) → uso Brent spot")
    if "brent_eur" in merged.columns:
        merged["brent_fut_eur_l"] = merged["brent_eur"] / BARREL_LITRES
    else:
        merged["brent_fut_eur_l"] = np.nan


# ─────────────────────────────────────────
# 5. CRACK SPREAD (RBOB + Heating Oil) — metodo preferito
# ─────────────────────────────────────────
print("\nTentativo crack spread RBOB/HO...")
use_crack = False
try:
    rb  = yf.download("RB=F", start="2021-01-01", end="2026-03-20", progress=False)
    ho  = yf.download("HO=F", start="2021-01-01", end="2026-03-20", progress=False)
    if rb.empty or ho.empty:
        raise ValueError("Dati vuoti")

    rb_w = (rb[["Close"]].rename(columns={"Close": "rbob_usd_gal"})
            .resample("W-MON").mean().ffill())
    ho_w = (ho[["Close"]].rename(columns={"Close": "ho_usd_gal"})
            .resample("W-MON").mean().ffill())

    for frame, col in [(rb_w, "rbob_usd_gal"), (ho_w, "ho_usd_gal")]:
        if col not in merged.columns:
            merged = merged.join(frame, how="left")
        merged[col] = merged[col].ffill()

    # USD/gallone → EUR/litro
    merged["rbob_eur_l"] = merged["rbob_usd_gal"] / merged["eurusd"] / GAL_LITRES
    merged["ho_eur_l"]   = merged["ho_usd_gal"]   / merged["eurusd"] / GAL_LITRES

    rb_m = merged["rbob_eur_l"].dropna().mean()
    ho_m = merged["ho_eur_l"].dropna().mean()
    print(f"  RBOB: {rb_m:.4f} EUR/L  |  HO: {ho_m:.4f} EUR/L")
    if 0.2 < rb_m < 2.5 and 0.2 < ho_m < 2.5:
        use_crack = True
        print("  → Crack spread OK, metodo preferito attivo")
    else:
        raise ValueError(f"Anomali: RBOB={rb_m:.3f}, HO={ho_m:.3f}")
except Exception as e:
    print(f"  Crack spread non disp. ({e}) → yield fisso")


# ─────────────────────────────────────────
# 6. MARGINE LORDO (EUR/LITRO, unità omogenee)
# ─────────────────────────────────────────
print("\nCalcolo margine lordo (EUR/L)...")

if use_crack:
    merged["margine_benzina"] = merged["benzina_eur_l_norm"] - merged["rbob_eur_l"]
    merged["margine_diesel"]  = merged["diesel_eur_l_norm"]  - merged["ho_eur_l"]
    metodo = "crack spread (RBOB/HO wholesale, EUR/L)"
else:
    merged["margine_benzina"] = (merged["benzina_eur_l_norm"]
                                 - merged["brent_fut_eur_l"] * YIELD_GASOLINE)
    merged["margine_diesel"]  = (merged["diesel_eur_l_norm"]
                                 - merged["brent_fut_eur_l"] * YIELD_DIESEL)
    metodo = f"yield fisso (YIELD={YIELD_GASOLINE}/{YIELD_DIESEL}) su brent_fut EUR/L"

for fuel, col in [("Benzina", "margine_benzina"), ("Diesel", "margine_diesel")]:
    m = merged[col].dropna().mean()
    flag = "OK" if -0.1 < m < 0.6 else "⚠ ANOMALO — controlla unità"
    print(f"  {fuel}: margine medio = {m:.4f} EUR/L  {flag}")

merged.to_csv("data/dataset_merged_with_futures.csv")
print("  Salvato: data/dataset_merged_with_futures.csv")


# ─────────────────────────────────────────
# 7. SOGLIA BASELINE 2σ (su 2021)
# ─────────────────────────────────────────
print("\nSoglie baseline (2σ su 2021)...")
baseline_thresholds = {}
baseline_means      = {}

for series_name in ["Benzina", "Diesel"]:
    col = f"margine_{series_name.lower()}"
    b   = merged.loc[BASELINE_START:BASELINE_END, col].dropna()
    if len(b) < 8:
        baseline_thresholds[series_name] = 0.030
        baseline_means[series_name]      = float(b.mean()) if len(b) > 0 else np.nan
        print(f"  {series_name}: baseline insufficiente → fallback 0.030 EUR/L")
    else:
        thr = 2.0 * float(b.std())
        baseline_thresholds[series_name] = thr
        baseline_means[series_name]      = float(b.mean())
        print(f"  {series_name}: μ={b.mean():.4f}, σ={b.std():.4f} → soglia 2σ={thr:.4f} EUR/L")


# ─────────────────────────────────────────
# 8. BAYESIAN CHANGEPOINT (StudentT + AR(1))
# ─────────────────────────────────────────
def bayesian_changepoint(x_vals, y_vals, alpha=0.05):
    n, sd_y, rng_x = len(x_vals), float(np.std(y_vals)), float(x_vals[-1]-x_vals[0])
    with pm.Model():
        tau   = pm.Uniform("tau",   lower=x_vals[0], upper=x_vals[-1])
        sigma = pm.HalfNormal("sigma", sigma=sd_y)
        nu    = pm.Exponential("nu", lam=1/30)
        rho   = pm.Uniform("rho",   lower=-1, upper=1)
        b1    = pm.StudentT("b1", mu=0, sigma=3*sd_y, nu=3)
        b2    = pm.StudentT("b2", mu=0, sigma=3*sd_y, nu=3)
        a1    = pm.StudentT("a1", mu=0, sigma=sd_y/max(rng_x,1), nu=3)
        a2    = pm.Deterministic("a2", a1 + tau*(b1-b2))
        x_pt  = pt.as_tensor_variable(x_vals.astype(float))
        step  = pm.math.sigmoid((x_pt - tau) * 50)
        mu    = (a1 + b1*x_pt)*(1-step) + (a2 + b2*x_pt)*step
        eps_init = pm.Normal("eps_init", mu=0, sigma=sigma)
        eta      = pm.Normal("eta", mu=0, sigma=sigma, shape=n-1)
        eps_rest, _ = scan(
            fn=lambda eta_t, eps_prev, rho_v: rho_v*eps_prev + eta_t,
            sequences=[eta], outputs_info=[eps_init], non_sequences=[rho],
        )
        eps = pt.concatenate([[eps_init], eps_rest])
        pm.StudentT("obs", nu=nu, mu=mu+eps, sigma=sigma, observed=y_vals)
        trace = pm.sample(
            draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS,
            progressbar=True, random_seed=42, target_accept=0.9,
            return_inferencedata=True,
        )
    tau_post = trace.posterior["tau"].values.flatten()
    lo, hi   = (alpha/2)*100, (1-alpha/2)*100
    return {
        "tau_mean": float(np.mean(tau_post)),
        "tau_lo":   float(np.percentile(tau_post, lo)),
        "tau_hi":   float(np.percentile(tau_post, hi)),
        "tau_idx":  int(np.clip(round(float(np.median(tau_post))), 1, n-2)),
        "tau_post": tau_post,
        "nu_mean":  float(np.mean(trace.posterior["nu"].values.flatten())),
        "rho_mean": float(np.mean(trace.posterior["rho"].values.flatten())),
    }


def is_edge(idx, n, frac=EDGE_FRACTION):
    return (idx < int(n * frac)) or (idx > int(n * (1 - frac)))


# ─────────────────────────────────────────
# 9. BOOTSTRAP Δmargine CI 95%
# ─────────────────────────────────────────
def bootstrap_delta(pre_vals, post_vals, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    deltas = [rng.choice(post_vals, len(post_vals), replace=True).mean() -
              rng.choice(pre_vals,  len(pre_vals),  replace=True).mean()
              for _ in range(n_boot)]
    arr = np.array(deltas)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ─────────────────────────────────────────
# 10. RUN PRINCIPALE
# ─────────────────────────────────────────
print("\n" + "="*80)
print("TEST SPECULAZIONE vs ANTICIPAZIONE RAZIONALE")
print(f"Metodo: {metodo}")
print(f"Unità: {unit_label}")
print("="*80)

results = []

for event_name, cfg in EVENTS.items():
    shock = pd.Timestamp(cfg["shock_date"])

    for series_name, col_pompa, _ in SERIES:
        margine_col = f"margine_{series_name.lower()}"

        df = merged.loc[cfg["window_start"]:cfg["window_end"]].copy()
        df = df.dropna(subset=[margine_col])
        if len(df) < 10:
            print(f"  SKIP {event_name}|{series_name}: meno di 10 obs")
            continue

        # ── Split pre/post sulla DATA DELLO SHOCK (non su tau MCMC)
        shock_idx = int(np.clip(df.index.searchsorted(shock), 2, len(df)-2))
        pre_m  = df.iloc[:shock_idx][margine_col].dropna()
        post_m = df.iloc[shock_idx:][margine_col].dropna()

        if len(pre_m) < 2 or len(post_m) < 2:
            print(f"  SKIP {event_name}|{series_name}: gruppo pre/post troppo piccolo")
            continue

        # ── Test statistici sul CONFRONTO PRE/POST SHOCK
        t_stat, t_p   = stats.ttest_ind(post_m.values, pre_m.values, equal_var=False)
        ks_stat, ks_p = stats.ks_2samp(pre_m.values, post_m.values)
        delta_mean    = float(post_m.mean() - pre_m.mean())
        boot_mean, boot_lo, boot_hi = bootstrap_delta(pre_m.values, post_m.values)

        # ── MCMC changepoint sul MARGINE (per stimare QUANDO cambia)
        print(f"\n  MCMC margine → {event_name} | {series_name}...")
        x_vals    = np.arange(len(df), dtype=float)
        y_margine = df[margine_col].values.astype(float)
        ci_m      = bayesian_changepoint(x_vals, y_margine)

        cp_idx   = ci_m["tau_idx"]
        cp_date  = df.index[cp_idx]
        no_break = is_edge(cp_idx, len(df))
        lag_vs_shock = (cp_date - shock).days

        # ── Classificazione statistica
        soglia = baseline_thresholds[series_name]
        anomalo = abs(delta_mean) > soglia

        t_sig  = (not np.isnan(t_p)) and (t_p < ALPHA)
        ks_sig = ks_p < ALPHA
        ci_non_zero = (boot_lo > 0) or (boot_hi < 0)
        stat_sig = (t_sig or ks_sig) and ci_non_zero

        if stat_sig and anomalo and delta_mean > 0:
            clas = "SPECULAZIONE"
        elif stat_sig and anomalo and delta_mean < 0:
            clas = "COMPRESSIONE MARGINE"
        elif not anomalo:
            clas = "ANTICIPAZIONE RAZIONALE / NEUTRO"
        elif stat_sig and not anomalo:
            clas = "VARIAZIONE STATISTICA (non anomala)"
        else:
            clas = "INCONCLUSIVO"

        print(f"\n  ══ {event_name} | {series_name} ══")
        print(f"     Δmargine = {delta_mean:+.5f} EUR/L  "
              f"[95%CI: {boot_lo:+.5f}, {boot_hi:+.5f}]")
        print(f"     Soglia 2σ = {soglia:.5f} EUR/L  |  anomalo: {anomalo}")
        print(f"     t-test p = {t_p:.4f}  |  KS p = {ks_p:.4f}")
        print(f"     τ_margine (MCMC) = {cp_date.date()}  "
              f"(lag vs shock = {lag_vs_shock:+d}gg)  "
              f"break strutturale: {not no_break}")
        print(f"     → {clas}")

        results.append({
            "Evento":             event_name,
            "Serie":              series_name,
            "Metodo":             metodo,
            "n_pre":              len(pre_m),
            "n_post":             len(post_m),
            "delta_margine_eur":  round(delta_mean, 5),
            "boot_CI_lo":         round(boot_lo, 5),
            "boot_CI_hi":         round(boot_hi, 5),
            "soglia_2sigma":      round(soglia, 5),
            "delta_anomalo":      anomalo,
            "t_p":                round(float(t_p), 4) if not np.isnan(t_p) else "nan",
            "ks_p":               round(ks_p, 4),
            "tau_margine":        cp_date.date(),
            "lag_tau_vs_shock":   lag_vs_shock,
            "break_strutturale":  not no_break,
            "nu_StudentT":        round(ci_m["nu_mean"], 2),
            "rho_AR1":            round(ci_m["rho_mean"], 3),
            "classificazione":    clas,
        })


# ─────────────────────────────────────────
# 11. SALVA RISULTATI
# ─────────────────────────────────────────
pd.DataFrame(results).to_csv("data/table2_margini_anomaly.csv", index=False)
print(f"\n  Salvato: data/table2_margini_anomaly.csv ({len(results)} righe)")


# ─────────────────────────────────────────
# 12. PLOT SINGOLI (margine + brent futures)
# ─────────────────────────────────────────
print("\nGenerazione plot...")
for res in results:
    event_name  = res["Evento"]
    series_name = res["Serie"]
    cfg         = EVENTS[event_name]
    shock       = pd.Timestamp(cfg["shock_date"])
    df_plot     = merged.loc[cfg["window_start"]:cfg["window_end"]]
    margine_col = f"margine_{series_name.lower()}"
    b_mean      = baseline_means.get(series_name, np.nan)
    soglia      = baseline_thresholds.get(series_name, np.nan)
    clas        = res["classificazione"]
    clas_c      = CLAS_COLOR.get(clas, "#555555")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              height_ratios=[3, 1.5])
    fig.suptitle(
        f"{event_name}  —  Margine lordo {series_name}\n"
        f"{res['Metodo']}\n"
        f"Δ = {res['delta_margine_eur']:+.4f} EUR/L  "
        f"[95%CI: {res['boot_CI_lo']:+.4f}, {res['boot_CI_hi']:+.4f}]  "
        f"KS p={res['ks_p']:.3f}  →  {clas}",
        fontsize=10, fontweight="bold", color=clas_c,
    )

    ax = axes[0]
    m_vals = df_plot[margine_col].dropna()
    ax.plot(m_vals.index, m_vals.values, color="#2c3e50", lw=2.0,
            label="Margine lordo proxy (EUR/L)")

    # Baseline e soglia 2σ
    if not np.isnan(b_mean):
        ax.axhline(b_mean,          color="#7f8c8d", lw=1.2, linestyle=":",
                   label=f"Media baseline 2021: {b_mean:.4f} EUR/L")
        ax.axhline(b_mean + soglia, color="#e67e22", lw=1.3, linestyle="--",
                   label=f"Soglia +2σ: {b_mean+soglia:.4f} EUR/L")
        ax.axhline(b_mean - soglia, color="#e67e22", lw=1.3, linestyle="--",
                   label=f"Soglia −2σ: {b_mean-soglia:.4f} EUR/L")

    # Shading pre/post shock
    if len(df_plot) > 0:
        ax.axvspan(df_plot.index[0], shock, alpha=0.04, color="#27ae60")
        ax.axvspan(shock, df_plot.index[-1], alpha=0.07, color=clas_c)

    ax.axvline(shock, color=cfg["color"], lw=2.5, linestyle=":",
               label=f"Shock {cfg['shock_date']}")

    tau_date = pd.Timestamp(str(res["tau_margine"]))
    if res["break_strutturale"]:
        ax.axvline(tau_date, color="#e74c3c", lw=2.0, linestyle="--",
                   label=f"τ_margine={res['tau_margine']} ({res['lag_tau_vs_shock']:+d}gg)")
    else:
        ax.text(0.985, 0.05,
                "Nessun break strutturale\nnel margine (τ al bordo)",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=9, color="#7f8c8d",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

    ax.text(0.015, 0.97,
            f"Δ = {res['delta_margine_eur']:+.4f} EUR/L\n"
            f"Soglia 2σ = {soglia:.4f} EUR/L\n"
            f"Anomalo: {res['delta_anomalo']}",
            ha="left", va="top", transform=ax.transAxes, fontsize=9,
            color=clas_c if res["delta_anomalo"] else "#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    ax.set_ylabel("Margine lordo (EUR/litro)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # Pannello inferiore: brent futures EUR/L
    ax2 = axes[1]
    if "brent_fut_eur_l" in df_plot.columns:
        fut_vals = df_plot["brent_fut_eur_l"].dropna()
        ax2.plot(fut_vals.index, fut_vals.values, color="#2166ac", lw=1.8,
                 label="Brent futures (EUR/L)")
    ax2.axvline(shock, color=cfg["color"], lw=2.0, linestyle=":",
                label=f"Shock {cfg['shock_date']}")
    ax2.set_ylabel("Brent futures (EUR/L)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    plt.tight_layout(pad=1.5)
    safe_e = (event_name.replace(" ", "_").replace("(", "").replace(")", "")
                        .replace("/", ""))
    fname = f"plots/07_margine_{safe_e}_{series_name.lower()}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {fname}")


# ─────────────────────────────────────────
# 13. PLOT RIASSUNTIVO
# ─────────────────────────────────────────
if results:
    df_r   = pd.DataFrame(results)
    n_rows = len(df_r)
    fig_s, ax_s = plt.subplots(figsize=(12, max(4, n_rows * 1.0)))
    fig_s.suptitle(
        f"Variazione margine lordo post-shock (split = data shock)\n"
        f"{metodo}  |  barre = Δ EUR/L  |  whisker = Bootstrap CI 95%",
        fontsize=11, fontweight="bold",
    )
    labels = [f"{r['Evento'].split('(')[0].strip()} | {r['Serie']}"
              for _, r in df_r.iterrows()]
    deltas = df_r["delta_margine_eur"].values
    ci_lo  = df_r["boot_CI_lo"].values
    ci_hi  = df_r["boot_CI_hi"].values
    colors = [CLAS_COLOR.get(c, "#555555") for c in df_r["classificazione"]]

    bars = ax_s.barh(range(n_rows), deltas, color=colors,
                     alpha=0.78, edgecolor="black", lw=0.7)
    for i in range(n_rows):
        ax_s.errorbar(deltas[i], i,
                      xerr=[[deltas[i]-ci_lo[i]], [ci_hi[i]-deltas[i]]],
                      fmt="none", color="black", capsize=6, lw=1.8)
        ax_s.text(max(ci_hi[i], deltas[i]) + 0.002, i,
                  f"KS p={df_r.iloc[i]['ks_p']:.3f}  {df_r.iloc[i]['classificazione'][:20]}",
                  va="center", fontsize=8)

    ax_s.axvline(0, color="black", lw=0.8)
    ax_s.set_yticks(range(n_rows))
    ax_s.set_yticklabels(labels, fontsize=9)
    ax_s.set_xlabel("Δ margine lordo post-shock (EUR/litro)", fontsize=11)
    ax_s.grid(alpha=0.3, axis="x")
    ax_s.legend(handles=[
        mpatches.Patch(color=c, label=k) for k, c in CLAS_COLOR.items()
    ], fontsize=8, loc="lower right")
    plt.tight_layout(pad=1.5)
    fig_s.savefig("plots/07_summary_delta_margine.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_s)
    print("  Salvato: plots/07_summary_delta_margine.png")


# ─────────────────────────────────────────
# 14. SOMMARIO FINALE
# ─────────────────────────────────────────
print("\n" + "="*80)
print("SOMMARIO SPECULAZIONE vs ANTICIPAZIONE RAZIONALE")
print(f"Metodo: {metodo}")
print(f"Unità: {unit_label}")
print("="*80)

for res in results:
    print(f"\n  {res['Evento']} | {res['Serie']}")
    print(f"    Δmargine   = {res['delta_margine_eur']:+.5f} EUR/L  "
          f"[95%CI: {res['boot_CI_lo']:+.5f}, {res['boot_CI_hi']:+.5f}]")
    print(f"    Soglia 2σ  = {res['soglia_2sigma']:.5f} EUR/L  → anomalo: {res['delta_anomalo']}")
    print(f"    t-test p   = {res['t_p']}  |  KS p = {res['ks_p']}")
    print(f"    τ_margine  = {res['tau_margine']} ({res['lag_tau_vs_shock']:+d}gg vs shock)  "
          f"break strutturale: {res['break_strutturale']}")
    print(f"    → {res['classificazione']}")

print("\n  Output:")
print("    data/table2_margini_anomaly.csv     → Table 2 paper (sezione 4.3)")
print("    plots/07_margine_*.png              → grafici per ogni evento×serie")
print("    plots/07_summary_delta_margine.png  → pannello riassuntivo")
print()
print("  Interpretazione classificazioni:")
print("    SPECULAZIONE         → Δ>2σ, p<0.05, CI esclude 0 (margine sale anomalmente)")
print("    COMPRESSIONE MARGINE → Δ<−2σ, p<0.05 (margine si comprime)")
print("    ANT. RAZ./NEUTRO     → Δ≤2σ: prezzo segue il costo, nessuna anomalia")
print("    VARIAZIONE STAT.     → stat. sig. ma entità nella norma")
print("    INCONCLUSIVO         → segnali contrastanti → integrare con dati ARERA")
print()
print("  NOTA: t-test p=nan → indica gruppo pre o post con <2 obs valide.")
print("  In questo script il test KS viene usato come fallback primario.")
print()
print("Script 07 v3 completato.")