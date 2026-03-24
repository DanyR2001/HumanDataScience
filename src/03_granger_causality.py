"""
03_granger_causality.py — un plot per carburante, paper quality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings("ignore")

merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
MAX_LAG = 8
ALPHA   = 0.05
DPI     = 180

# ADF + differenze
for col in ["log_brent","log_benzina","log_diesel"]:
    if col in merged.columns:
        p = adfuller(merged[col].dropna(), autolag="AIC")[1]
        print(f"  ADF {col}: p={p:.4f} {'OK' if p<0.05 else '-> uso diff'}")

merged["d_log_brent"]   = merged["log_brent"].diff()
merged["d_log_benzina"] = merged["log_benzina"].diff()
merged["d_log_diesel"]  = merged["log_diesel"].diff()
merged.dropna(inplace=True)

print("\nGRANGER: Brent -> Pompa\n" + "="*50)

granger_results = {}
for fuel_col, fuel_name in [("d_log_benzina","Benzina"),("d_log_diesel","Diesel")]:
    data = merged[["d_log_"+fuel_col.split("_")[-1] if "log" not in fuel_col else fuel_col.replace("d_log_","d_log_"), fuel_col]].dropna()
    data = merged[[fuel_col, "d_log_brent"]].dropna()[["d_log_brent", fuel_col]]
    # grangercausalitytests vuole [Y, X]
    data2 = merged[[fuel_col, "d_log_brent"]].dropna()
    try:
        gc = grangercausalitytests(data2, maxlag=MAX_LAG, verbose=False)
    except Exception as e:
        print(f"  Errore {fuel_name}: {e}"); continue
    rows = []
    for lag in range(1, MAX_LAG+1):
        f_stat, p_val = gc[lag][0]["ssr_ftest"][:2]
        rows.append({"lag_weeks": lag, "lag_days": lag*7,
                     "F_stat": round(f_stat,4), "p_value": round(p_val,4),
                     "significant": p_val < ALPHA})
        sig = "***" if p_val<0.01 else "**" if p_val<0.05 else ""
        h0  = " <- H0 RIFIUTATA" if (p_val<ALPHA and lag*7<30) else ""
        print(f"  {fuel_name} lag={lag}sett ({lag*7}gg): F={f_stat:.3f} p={p_val:.4f} {sig}{h0}")
    granger_results[fuel_name] = pd.DataFrame(rows)
    granger_results[fuel_name].to_csv(f"data/granger_{fuel_name.lower()}.csv", index=False)

# Plot: un grafico per carburante
for fuel_name, df_gc in granger_results.items():
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = ["#e74c3c" if p < ALPHA else "#3498db" for p in df_gc["p_value"]]
    bars = ax.bar(df_gc["lag_days"], df_gc["p_value"],
                  color=bar_colors, edgecolor="black", lw=0.7, alpha=0.85, width=5)

    ax.axhline(ALPHA, color="black", lw=1.8, linestyle="--", label=f"α = {ALPHA}")
    ax.axvline(30, color="#e67e22", lw=2.2, linestyle="--", label="Soglia fisica 30 giorni")
    ax.axvspan(0, 30, alpha=0.07, color="#e74c3c", label="Zona speculazione (lag < 30gg)")

    for bar, p in zip(bars, df_gc["p_value"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{p:.3f}", ha="center", va="bottom", fontsize=9,
                color="red" if p < ALPHA else "gray", fontweight="bold" if p < ALPHA else "normal")

    ax.set_xlabel("Lag (giorni)", fontsize=13)
    ax.set_ylabel("p-value (F-test)", fontsize=13)
    ax.set_title(f"Granger Causality: Brent → {fuel_name}\n"
                 f"(barre rosse = H₀ rifiutata p < {ALPHA})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(df_gc["p_value"].max() * 1.3, 0.15))
    ax.set_xticks(df_gc["lag_days"])
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(f"plots/03_granger_{fuel_name.lower()}.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: plots/03_granger_{fuel_name.lower()}.png")

print("\nScript 03 completato.")