"""
06_statistical_tests.py
========================
Test statistici aggiuntivi per rafforzare il rifiuto di H0.

Tests implementati:
  1. Kolmogorov-Smirnov       — distribuzione prezzi pre vs post shock
  2. ANOVA a un fattore        — varianza tra 3 periodi (pre / shock / post)
  3. Chow Test                 — structural break formale sulla regressione
  4. Cross-Correlation (CCF)   — lag ottimale Brent → pompa
  5. Rolling Correlation       — correlazione mobile nel tempo
  6. Confidence Intervals      — su changepoint e pendenze (bootstrap)
  7. RMSE / MAE                — bonta del fit della regressione piecewise
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Carica dati
# ─────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
merged.dropna(inplace=True)
print(f"Dataset: {len(merged)} settimane | {merged.index[0].date()} → {merged.index[-1].date()}\n")

# Eventi di guerra con finestre temporali
EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":      pd.Timestamp("2022-02-24"),
        "pre_start":  pd.Timestamp("2021-09-01"),
        "post_end":   pd.Timestamp("2022-08-31"),
    },
    "Hormuz (Feb 2026)": {
        "shock":      pd.Timestamp("2026-02-28"),
        "pre_start":  pd.Timestamp("2025-09-01"),
        "post_end":   pd.Timestamp("2026-03-17"),
    },
}

FUELS = {
    "Benzina": "benzina_4w",
    "Diesel":  "diesel_4w",
}

ALPHA = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# 1. KOLMOGOROV-SMIRNOV TEST
#    H0: la distribuzione dei prezzi pre-shock = distribuzione post-shock
#    Rifiuto H0 → i prezzi post-shock appartengono a una distribuzione diversa
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("1. KOLMOGOROV-SMIRNOV TEST")
print("   H0: distribuzione pre-shock = distribuzione post-shock")
print("=" * 65)

ks_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue

        pre  = merged.loc[cfg["pre_start"]:shock, fuel_col].dropna()
        post = merged.loc[shock:cfg["post_end"],   fuel_col].dropna()

        if len(pre) < 4 or len(post) < 4:
            continue

        ks_stat, ks_p = stats.ks_2samp(pre.values, post.values)
        result = "RIFIUTATA" if ks_p < ALPHA else "non rifiutata"

        ks_results.append({
            "Evento":      event_name,
            "Carburante":  fuel_name,
            "n_pre":       len(pre),
            "n_post":      len(post),
            "KS_stat":     round(ks_stat, 4),
            "p_value":     round(ks_p, 6),
            "H0":          result,
        })

        print(f"  {event_name} | {fuel_name}: KS={ks_stat:.4f}, p={ks_p:.6f} → H0 {result}")

pd.DataFrame(ks_results).to_csv("data/ks_results.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANOVA A UN FATTORE
#    Confronta la varianza dei prezzi in 3 periodi:
#      - Periodo A: 6 mesi pre-shock
#      - Periodo B: 0-6 settimane post-shock (shock acuto)
#      - Periodo C: 6 settimane - 6 mesi post-shock (normalizzazione)
#    H0: le medie dei 3 periodi sono uguali
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("2. ANOVA A UN FATTORE (3 periodi: pre / shock acuto / post)")
print("   H0: media prezzi uguale nei 3 periodi")
print("=" * 65)

anova_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue

        period_A = merged.loc[cfg["pre_start"]:shock,                       fuel_col].dropna()
        period_B = merged.loc[shock:shock + pd.Timedelta(weeks=6),          fuel_col].dropna()
        period_C = merged.loc[shock + pd.Timedelta(weeks=6):cfg["post_end"],fuel_col].dropna()

        if len(period_A) < 3 or len(period_B) < 3 or len(period_C) < 3:
            continue

        f_stat, anova_p = stats.f_oneway(period_A, period_B, period_C)
        result = "RIFIUTATA" if anova_p < ALPHA else "non rifiutata"

        anova_results.append({
            "Evento":     event_name,
            "Carburante": fuel_name,
            "F_stat":     round(f_stat, 4),
            "p_value":    round(anova_p, 6),
            "mean_A":     round(period_A.mean(), 4),
            "mean_B":     round(period_B.mean(), 4),
            "mean_C":     round(period_C.mean(), 4),
            "H0":         result,
        })

        print(f"  {event_name} | {fuel_name}: F={f_stat:.4f}, p={anova_p:.6f} → H0 {result}")
        print(f"    medie: pre={period_A.mean():.4f} | shock={period_B.mean():.4f} | post={period_C.mean():.4f}")

pd.DataFrame(anova_results).to_csv("data/anova_results.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHOW TEST
#    Test formale di structural break su una regressione lineare.
#    Divide la serie in due segmenti al punto di break e testa se
#    i coefficienti sono significativamente diversi.
#    H0: nessun structural break (coefficienti uguali prima e dopo)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("3. CHOW TEST (structural break)")
print("   H0: nessuna rottura strutturale nella regressione")
print("=" * 65)

def chow_test(y: np.ndarray, breakpoint: int):
    """
    Chow test for structural break at index `breakpoint`.
    F = [(RSS_r - RSS_u) / k] / [RSS_u / (n - 2k)]
    dove:
      RSS_r = residui regressione sull'intero campione
      RSS_u = RSS_1 + RSS_2 (regressioni separate)
      k = numero di parametri (2: intercetta + slope)
    """
    n = len(y)
    x = np.arange(n)
    k = 2  # intercetta + slope

    # Regressione sull'intero campione (restricted)
    X_full = np.column_stack([np.ones(n), x])
    beta_r, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    rss_r = np.sum((y - X_full @ beta_r) ** 2)

    # Regressioni separate (unrestricted)
    def ols_rss(xv, yv):
        if len(xv) < k + 1:
            return 0.0
        Xv = np.column_stack([np.ones(len(xv)), xv])
        b, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        return np.sum((yv - Xv @ b) ** 2)

    rss1 = ols_rss(x[:breakpoint],  y[:breakpoint])
    rss2 = ols_rss(x[breakpoint:],  y[breakpoint:])
    rss_u = rss1 + rss2

    if rss_u < 1e-12:
        return np.nan, np.nan

    f_stat = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
    p_val  = 1 - stats.f.cdf(f_stat, dfn=k, dfd=n - 2 * k)
    return f_stat, p_val


chow_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue

        series = merged.loc[cfg["pre_start"]:cfg["post_end"], fuel_col].dropna()
        if len(series) < 10:
            continue

        # Trova l'indice corrispondente alla data dello shock
        shock_idx = series.index.searchsorted(shock)
        shock_idx = max(3, min(shock_idx, len(series) - 3))

        f_stat, p_val = chow_test(series.values, shock_idx)

        if np.isnan(f_stat):
            continue

        result = "RIFIUTATA" if p_val < ALPHA else "non rifiutata"
        chow_results.append({
            "Evento":     event_name,
            "Carburante": fuel_name,
            "Break_date": shock.date(),
            "F_stat":     round(f_stat, 4),
            "p_value":    round(p_val, 6),
            "H0":         result,
        })

        print(f"  {event_name} | {fuel_name}: F={f_stat:.4f}, p={p_val:.6f} → H0 {result}")

pd.DataFrame(chow_results).to_csv("data/chow_results.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-CORRELATION FUNCTION (CCF)
#    Calcola la correlazione tra Brent e prezzi pompa per lag da 0 a 12 sett.
#    Il lag al massimo della CCF = lag ottimale di trasmissione.
#    Se lag_ottimale < 4 sett (< 30gg) → H0 rifiutata.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("4. CROSS-CORRELATION FUNCTION (CCF)")
print("   Lag al picco massimo = velocita di trasmissione stimata")
print("=" * 65)

MAX_LAG_CCF = 12  # settimane

ccf_results = {}
d_brent = merged["log_brent"].diff().dropna()

for fuel_name, fuel_col in FUELS.items():
    log_col = "log_benzina" if fuel_name == "Benzina" else "log_diesel"
    if log_col not in merged.columns:
        continue

    d_fuel = merged[log_col].diff().dropna()
    common = d_brent.index.intersection(d_fuel.index)
    x = d_brent.loc[common].values
    y = d_fuel.loc[common].values

    # Calcola cross-correlazione per lag 0..MAX_LAG_CCF
    ccf_vals = []
    for lag in range(0, MAX_LAG_CCF + 1):
        if lag == 0:
            r, _ = stats.pearsonr(x, y)
        else:
            r, _ = stats.pearsonr(x[:-lag], y[lag:])
        ccf_vals.append(r)

    ccf_results[fuel_name] = ccf_vals
    best_lag = np.argmax(np.abs(ccf_vals))
    best_r   = ccf_vals[best_lag]

    print(f"  {fuel_name}: lag ottimale = {best_lag} settimane ({best_lag*7} giorni), "
          f"r = {best_r:.4f} → "
          f"{'H0 RIFIUTATA (lag < 4 sett)' if best_lag < 4 else 'compatibile con logistica'}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROLLING CORRELATION (finestra mobile 12 settimane)
#    Mostra come la correlazione Brent-pompa cambia nel tempo.
#    Durante le guerre la correlazione dovrebbe aumentare bruscamente.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("5. ROLLING CORRELATION (finestra 12 settimane)")
print("=" * 65)

ROLL_WIN = 12

rolling_corr = {}
for fuel_name, fuel_col in FUELS.items():
    if fuel_col not in merged.columns:
        continue
    rc = merged["brent_7d"].rolling(ROLL_WIN).corr(merged[fuel_col])
    rolling_corr[fuel_name] = rc
    print(f"  {fuel_name}: corr media = {rc.mean():.4f} | "
          f"corr durante Ucraina (mar 2022) = "
          f"{rc.loc['2022-03-01':'2022-04-30'].mean():.4f}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. BOOTSTRAP CONFIDENCE INTERVALS SUL LAG D
#    Stima non parametrica dell'incertezza sul lag changepoint.
#    Ricampiona con replacement la serie e ricalcola il changepoint N volte.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("6. BOOTSTRAP CONFIDENCE INTERVALS (95%) sul lag D")
print("=" * 65)

import ruptures as rpt

N_BOOTSTRAP = 500

def detect_changepoint(series_values):
    signal = series_values.reshape(-1, 1)
    try:
        algo = rpt.Binseg(model="l2").fit(signal)
        cp = algo.predict(n_bkps=1)[0]
    except Exception:
        cp = len(series_values) // 2
    return min(cp, len(series_values) - 2)


bootstrap_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]

    # Serie Brent nella finestra
    brent_series = merged.loc[cfg["pre_start"]:cfg["post_end"], "log_brent"].dropna()
    if len(brent_series) < 10:
        continue

    cp_brent_idx  = detect_changepoint(brent_series.values)
    tau_crude_base = brent_series.index[cp_brent_idx]

    for fuel_name, fuel_col in FUELS.items():
        log_col = "log_benzina" if fuel_name == "Benzina" else "log_diesel"
        if log_col not in merged.columns:
            continue

        fuel_series = merged.loc[cfg["pre_start"]:cfg["post_end"], log_col].dropna()
        if len(fuel_series) < 10:
            continue

        np.random.seed(42)
        lag_samples = []

        for _ in range(N_BOOTSTRAP):
            # Ricampiona con replacement (block bootstrap con blocchi da 4 sett.)
            block_size = 4
            n = len(fuel_series)
            n_blocks = n // block_size + 1
            idx = np.concatenate([
                np.arange(i, min(i + block_size, n))
                for i in np.random.choice(range(0, n - block_size + 1), n_blocks)
            ])[:n]
            sample = fuel_series.values[idx]
            cp_idx = detect_changepoint(sample)
            tau_retail_boot = fuel_series.index[min(cp_idx, len(fuel_series) - 1)]
            lag_samples.append((tau_retail_boot - tau_crude_base).days)

        lag_samples = np.array(lag_samples)
        ci_low  = np.percentile(lag_samples, 2.5)
        ci_high = np.percentile(lag_samples, 97.5)
        lag_mean = np.mean(lag_samples)

        bootstrap_results.append({
            "Evento":     event_name,
            "Carburante": fuel_name,
            "Lag_mean":   round(lag_mean, 1),
            "CI_95_low":  round(ci_low, 1),
            "CI_95_high": round(ci_high, 1),
            "H0 (CI < 30gg)": "RIFIUTATA" if ci_high < 30 else "non rifiutata",
        })

        print(f"  {event_name} | {fuel_name}: "
              f"D = {lag_mean:.1f}gg [95% CI: {ci_low:.1f} – {ci_high:.1f}] → "
              f"{'H0 RIFIUTATA' if ci_high < 30 else 'non rifiutata'}")

pd.DataFrame(bootstrap_results).to_csv("data/bootstrap_ci.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. RMSE / MAE — BONTA DEL FIT DELLA REGRESSIONE PIECEWISE
#    Misura quanto bene la regressione piecewise (script 02) fitta i dati.
#    RMSE e MAE bassi = modello a due tratti descrive bene la dinamica reale.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("7. RMSE / MAE — bonta del fit piecewise vs regressione semplice")
print("=" * 65)

def fit_and_score(y, breakpoint_idx=None):
    """
    Fitta regressione semplice e piecewise, restituisce RMSE e MAE per entrambe.
    """
    x = np.arange(len(y))

    # Regressione lineare semplice
    slope, intercept, *_ = stats.linregress(x, y)
    y_hat_simple = intercept + slope * x
    rmse_simple = np.sqrt(np.mean((y - y_hat_simple) ** 2))
    mae_simple  = np.mean(np.abs(y - y_hat_simple))

    if breakpoint_idx is None or breakpoint_idx <= 1 or breakpoint_idx >= len(y) - 1:
        return rmse_simple, mae_simple, np.nan, np.nan

    # Regressione piecewise
    cp = breakpoint_idx
    def ols(xv, yv):
        Xv = np.column_stack([np.ones(len(xv)), xv])
        b, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        return b
    b1 = ols(x[:cp], y[:cp])
    b2 = ols(x[cp:], y[cp:])
    y_hat_pw = np.concatenate([
        b1[0] + b1[1] * x[:cp],
        b2[0] + b2[1] * x[cp:],
    ])
    rmse_pw = np.sqrt(np.mean((y - y_hat_pw) ** 2))
    mae_pw  = np.mean(np.abs(y - y_hat_pw))

    return rmse_simple, mae_simple, rmse_pw, mae_pw


fit_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        log_col = "log_benzina" if fuel_name == "Benzina" else "log_diesel"
        if log_col not in merged.columns:
            continue

        series = merged.loc[cfg["pre_start"]:cfg["post_end"], log_col].dropna()
        if len(series) < 10:
            continue

        shock_idx = series.index.searchsorted(shock)
        shock_idx = max(2, min(shock_idx, len(series) - 2))

        rs, ms, rpw, mpw = fit_and_score(series.values, shock_idx)

        fit_results.append({
            "Evento":        event_name,
            "Carburante":    fuel_name,
            "RMSE_semplice": round(rs, 6),
            "MAE_semplice":  round(ms, 6),
            "RMSE_piecewise": round(rpw, 6) if not np.isnan(rpw) else "N/A",
            "MAE_piecewise":  round(mpw, 6) if not np.isnan(mpw) else "N/A",
            "Miglioramento_%": round((1 - rpw / rs) * 100, 1) if not np.isnan(rpw) and rs > 0 else "N/A",
        })

        if not np.isnan(rpw):
            print(f"  {event_name} | {fuel_name}: "
                  f"RMSE semplice={rs:.5f} | RMSE piecewise={rpw:.5f} "
                  f"(miglioramento: {(1 - rpw/rs)*100:.1f}%)")

pd.DataFrame(fit_results).to_csv("data/fit_quality.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Test statistici aggiuntivi — Speculazione carburanti Italia",
             fontsize=13, fontweight="bold")

war_dates = {
    "Ucraina": pd.Timestamp("2022-02-24"),
    "Hormuz":  pd.Timestamp("2026-02-28"),
}

# --- Plot 1: CCF ---
ax = axes[0, 0]
lags_x = list(range(0, MAX_LAG_CCF + 1))
colors  = ["#e74c3c", "#3498db"]
for (fuel_name, ccf_vals), color in zip(ccf_results.items(), colors):
    ax.plot(lags_x, ccf_vals, marker="o", color=color, lw=2, label=fuel_name)
ax.axvline(4, color="orange", lw=2, linestyle="--", label="Soglia 30gg (4 sett.)")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Lag (settimane)")
ax.set_ylabel("Correlazione")
ax.set_title("Cross-Correlation: Brent → Prezzi Pompa")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(lags_x)

# --- Plot 2: Rolling Correlation ---
ax = axes[0, 1]
colors_rc = ["#e74c3c", "#27ae60"]
for (fuel_name, rc), color in zip(rolling_corr.items(), colors_rc):
    ax.plot(rc.index, rc.values, color=color, lw=1.5, label=fuel_name)
for label, date in war_dates.items():
    if merged.index[0] <= date <= merged.index[-1]:
        ax.axvline(date, color="gray", lw=1.5, linestyle="--")
        ax.text(date, 0.05, label, rotation=90, fontsize=7, color="gray", va="bottom")
ax.set_ylabel("Correlazione (rolling 12 sett.)")
ax.set_title(f"Correlazione mobile Brent-Carburante ({ROLL_WIN} sett.)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# --- Plot 3: KS test — distribuzioni pre vs post ---
ax = axes[1, 0]
event_name_plot = "Ucraina (Feb 2022)"
cfg_plot = EVENTS[event_name_plot]
shock_plot = cfg_plot["shock"]

for (fuel_name, fuel_col), color in zip(FUELS.items(), ["#e74c3c", "#3498db"]):
    if fuel_col not in merged.columns:
        continue
    pre  = merged.loc[cfg_plot["pre_start"]:shock_plot, fuel_col].dropna()
    post = merged.loc[shock_plot:cfg_plot["post_end"],   fuel_col].dropna()
    # ECDF
    pre_sorted  = np.sort(pre.values)
    post_sorted = np.sort(post.values)
    ax.step(pre_sorted,  np.linspace(0, 1, len(pre_sorted)),
            color=color, lw=2, linestyle="--", label=f"{fuel_name} pre")
    ax.step(post_sorted, np.linspace(0, 1, len(post_sorted)),
            color=color, lw=2, label=f"{fuel_name} post")

ax.set_xlabel("Prezzo (EUR/litro)")
ax.set_ylabel("ECDF")
ax.set_title(f"KS Test: ECDF pre vs post\n({event_name_plot})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Plot 4: Bootstrap CI sul lag D ---
ax = axes[1, 1]
if bootstrap_results:
    df_boot = pd.DataFrame(bootstrap_results)
    labels  = [f"{r['Evento'].split('(')[0].strip()}\n{r['Carburante']}"
               for _, r in df_boot.iterrows()]
    y_pos   = np.arange(len(labels))
    means   = df_boot["Lag_mean"].values
    ci_low  = df_boot["CI_95_low"].values
    ci_high = df_boot["CI_95_high"].values

    bar_colors = ["#e74c3c" if h < 30 else "#3498db" for h in ci_high]
    ax.barh(y_pos, means, color=bar_colors, alpha=0.7, edgecolor="black", lw=0.5)
    for i, (l, h) in enumerate(zip(ci_low, ci_high)):
        ax.errorbar(means[i], y_pos[i], xerr=[[means[i]-l], [h-means[i]]],
                    fmt="none", color="black", capsize=5, lw=2)

    ax.axvline(30, color="orange", lw=2, linestyle="--", label="Soglia H0 (30gg)")
    ax.axvline(0,  color="black",  lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Lag D (giorni)")
    ax.set_title("Bootstrap 95% CI sul lag D\n(rosso = H0 rifiutata)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("plots/06_statistical_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot salvato: plots/06_statistical_tests.png")


# ─────────────────────────────────────────────────────────────────────────────
# SOMMARIO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SOMMARIO — Tutti i test statistici")
print("=" * 65)
print(f"  {'Test':<30} {'H0 rifiutata?':<15} {'File output'}")
print(f"  {'-'*60}")
print(f"  {'Kolmogorov-Smirnov':<30} {'vedi tabella':<15} data/ks_results.csv")
print(f"  {'ANOVA (3 periodi)':<30} {'vedi tabella':<15} data/anova_results.csv")
print(f"  {'Chow Test':<30} {'vedi tabella':<15} data/chow_results.csv")
print(f"  {'CCF (lag ottimale)':<30} {'vedi output':<15} —")
print(f"  {'Rolling Correlation':<30} {'visualizzazione':<15} plots/06_...")
print(f"  {'Bootstrap CI (95%)':<30} {'vedi tabella':<15} data/bootstrap_ci.csv")
print(f"  {'RMSE / MAE':<30} {'vedi tabella':<15} data/fit_quality.csv")
print()
print("  Script 02: Bayesian changepoint + piecewise regression")
print("  Script 03: Granger causality (ADF + F-test)")
print("  Script 04: Rockets & Feathers (OLS + t-test asimmetria)")
print("  Script 06: KS, ANOVA, Chow, CCF, Rolling Corr, Bootstrap, RMSE/MAE")
print()
print("Tutti i risultati sono salvati in data/ e plots/")
print("Tutto il codice e i risultati sono riproducibili.")