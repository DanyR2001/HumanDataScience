"""
03_granger_causality.py
========================
Testa se il prezzo del Brent "Granger-causa" i prezzi al dettaglio
con un lag < 30 giorni → evidenza di aggiustamento anticipatorio.

Granger causality: X causa Y nel senso di Granger se
la conoscenza passata di X migliora la previsione di Y
rispetto alla sola conoscenza passata di Y.

H₀ Granger: il Brent NON Granger-causa il prezzo alla pompa con lag k
Se p-value < 0.05 con lag k < 4 settimane → prezzi anticipano la logistica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Carica dati
# ─────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane\n")

MAX_LAG    = 8   # settimane (8 = 56 giorni, oltre la soglia fisica)
ALPHA      = 0.05
H0_LAG_WK  = 4   # 4 settimane ≈ 30 giorni (soglia H₀)


# ─────────────────────────────────────────
# 1. TEST DI STAZIONARIETÀ (ADF)
#    Granger richiede serie stazionarie
# ─────────────────────────────────────────
def adf_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    p = result[1]
    status = "stazionaria ✓" if p < 0.05 else "NON stazionaria ✗ (usa diff)"
    print(f"  ADF {name}: p={p:.4f} → {status}")
    return p < 0.05

print("=" * 55)
print("TEST DI STAZIONARIETÀ (Augmented Dickey-Fuller)")
print("=" * 55)

stationary = {}
for col, name in [("log_brent", "log(Brent)"),
                  ("log_benzina", "log(Benzina)"),
                  ("log_diesel", "log(Diesel)")]:
    if col in merged.columns:
        stationary[col] = adf_test(merged[col], name)

# Se non stazionarie, usa le prime differenze
print("\n  Usando prime differenze (più robuste per time series):")
merged["d_log_brent"]   = merged["log_brent"].diff()
merged["d_log_benzina"] = merged["log_benzina"].diff()
merged["d_log_diesel"]  = merged["log_diesel"].diff()

for col, name in [("d_log_brent",   "Δlog(Brent)"),
                  ("d_log_benzina", "Δlog(Benzina)"),
                  ("d_log_diesel",  "Δlog(Diesel)")]:
    adf_test(merged[col].dropna(), name)

merged.dropna(inplace=True)


# ─────────────────────────────────────────
# 2. GRANGER CAUSALITY TEST
# ─────────────────────────────────────────
def run_granger(cause_col, effect_col, cause_name, effect_name, max_lag):
    """
    Esegui Granger causality test per lag da 1 a max_lag.
    Restituisce DataFrame con p-value per ogni lag.
    """
    data = merged[[effect_col, cause_col]].dropna()
    # Nota: grangercausalitytests vuole [Y, X] dove X → Y

    try:
        gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception as e:
        print(f"  Errore Granger {cause_name}→{effect_name}: {e}")
        return None

    rows = []
    for lag in range(1, max_lag + 1):
        # Usa il test F (più robusto per campioni piccoli)
        f_test = gc_res[lag][0]["ssr_ftest"]
        chi_test = gc_res[lag][0]["ssr_chi2test"]
        rows.append({
            "lag_weeks": lag,
            "lag_days":  lag * 7,
            "F_stat":    round(f_test[0], 4),
            "p_value_F": round(f_test[1], 4),
            "chi2_stat": round(chi_test[0], 4),
            "p_value_chi2": round(chi_test[1], 4),
            "significant": f_test[1] < ALPHA,
        })
    return pd.DataFrame(rows)


print("\n" + "=" * 55)
print("GRANGER CAUSALITY: Brent → Prezzi Pompa Italia")
print(f"(H₀: il Brent non Granger-causa il carburante)")
print("=" * 55)

granger_results = {}

for fuel_col, fuel_name in [("d_log_benzina", "Benzina"),
                              ("d_log_diesel",  "Diesel")]:
    print(f"\n  {fuel_name}:")
    print(f"  {'Lag (sett)':<12} {'Lag (gg)':<10} {'F-stat':<10} {'p-value':<10} {'Sig.'}")
    print(f"  {'-'*50}")

    df_gc = run_granger("d_log_brent", fuel_col, "Brent", fuel_name, MAX_LAG)
    if df_gc is None:
        continue

    granger_results[fuel_name] = df_gc

    for _, row in df_gc.iterrows():
        sig = "*** ✓" if row["p_value_F"] < 0.01 else \
              "**"    if row["p_value_F"] < 0.05 else \
              "*"     if row["p_value_F"] < 0.10 else ""
        h0_flag = " ← H₀ rifiutata (lag < 30gg)" if (row["significant"] and row["lag_days"] < 30) else ""
        print(f"  {int(row['lag_weeks']):<12} {int(row['lag_days']):<10} "
              f"{row['F_stat']:<10} {row['p_value_F']:<10} {sig}{h0_flag}")

    # Trova il lag minimo significativo
    sig_lags = df_gc[df_gc["significant"]]["lag_days"]
    if not sig_lags.empty:
        min_lag = sig_lags.min()
        print(f"\n  → Lag minimo significativo: {min_lag} giorni")
        print(f"  → {'  SPECULAZIONE: lag < 30gg' if min_lag < 30 else 'Compatibile con logistica'}")

    df_gc.to_csv(f"data/granger_{fuel_name.lower()}.csv", index=False)


# ─────────────────────────────────────────
# 3. PLOT P-VALUE PER LAG
# ─────────────────────────────────────────
if granger_results:
    fig, axes = plt.subplots(1, len(granger_results), figsize=(13, 5), sharey=True)
    if len(granger_results) == 1:
        axes = [axes]

    for ax, (fuel_name, df_gc) in zip(axes, granger_results.items()):
        bars = ax.bar(df_gc["lag_days"], df_gc["p_value_F"],
                      color=["#e74c3c" if p < ALPHA else "#3498db"
                             for p in df_gc["p_value_F"]],
                      edgecolor="black", lw=0.5, alpha=0.85)

        # Linea soglia α=0.05
        ax.axhline(ALPHA, color="black", lw=1.5, linestyle="--",
                   label=f"α = {ALPHA}")

        # Linea soglia H₀ (30 giorni)
        ax.axvline(30, color="orange", lw=2, linestyle="--",
                   label="Soglia fisica\n(30 giorni)")

        # Zona rossa = rifiuto H₀
        ax.axvspan(0, 30, alpha=0.08, color="red",
                   label="Zona speculazione\n(lag < 30gg)")

        ax.set_xlabel("Lag (giorni)", fontsize=11)
        ax.set_ylabel("p-value (F-test)" if fuel_name == "Benzina" else "", fontsize=11)
        ax.set_title(f"Granger: Brent → {fuel_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_xticks(df_gc["lag_days"].values)

        # Annota ogni barra con il p-value
        for bar, p in zip(bars, df_gc["p_value_F"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{p:.3f}", ha="center", va="bottom", fontsize=7,
                    color="red" if p < ALPHA else "gray")

    fig.suptitle(
        "Granger Causality Test: Brent → Prezzi Pompa Italia\n"
        "Barre rosse = p < 0.05 (H₀ rifiutata) | Zona arancione = soglia fisica ≥30gg",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("plots/03_granger.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n Plot salvato: plots/03_granger.png")

print("\n Script 03 completato.")