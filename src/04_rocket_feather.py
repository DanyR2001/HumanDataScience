"""
04_rocket_feather.py
=====================
Testa l'effetto "Rockets and Feathers" (Bacon, 1991):
  - I prezzi salgono velocemente quando il grezzo sale (razzo)
  - I prezzi scendono lentamente quando il grezzo scende (piuma)

Metodologia:
  1. Separa le variazioni positive e negative del Brent
  2. Stima velocità di pass-through per movimenti up vs down
  3. Test t sulla differenza delle pendenze (b_up vs b_down)
  4. Asimmetria quantitativa: indice R&F
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Carica dati
# ─────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
merged.dropna(inplace=True)
print(f"Dataset: {len(merged)} settimane\n")

FUELS = {
    "Benzina": "benzina_4w",
    "Diesel":  "diesel_4w",
}


# ─────────────────────────────────────────
# 1. CALCOLO VARIAZIONI SETTIMANALI
# ─────────────────────────────────────────
merged["d_brent"]   = merged["brent_7d"].pct_change() * 100      # % change
merged["d_benzina"] = merged["benzina_4w"].pct_change() * 100
merged["d_diesel"]  = merged["diesel_4w"].pct_change() * 100
merged.dropna(inplace=True)

# Separa settimane in cui il Brent sale vs scende
up_mask   = merged["d_brent"] > 0
down_mask = merged["d_brent"] < 0

print(f"Settimane Brent ↑: {up_mask.sum()} | Settimane Brent ↓: {down_mask.sum()}")


# ─────────────────────────────────────────
# 2. REGRESSIONE ASIMMETRICA (OLS)
#    Δpompa = α + β_up * max(ΔBrent, 0) + β_down * min(ΔBrent, 0) + ε
#    β_up > β_down → effetto razzo
# ─────────────────────────────────────────
print("\n" + "="*60)
print("EFFETTO ROCKETS & FEATHERS")
print("="*60)

rf_results = {}

for fuel_name, fuel_col in FUELS.items():
    if f"d_{fuel_col.split('_')[0]}" not in merged.columns:
        # Usa il nome originale
        dcol = f"d_{fuel_name.lower()}"
    else:
        dcol = f"d_{fuel_name.lower()}"

    if dcol not in merged.columns:
        continue

    y = merged[dcol].values

    # Costruisci variabili asimmetriche
    brent_pos = np.maximum(merged["d_brent"].values, 0)   # max(ΔBrent, 0)
    brent_neg = np.minimum(merged["d_brent"].values, 0)   # min(ΔBrent, 0)

    # OLS con termine asimmetrico
    X = np.column_stack([np.ones(len(y)), brent_pos, brent_neg])
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    alpha, b_up, b_down = beta

    # Calcola standard errors manualmente
    y_hat = X @ beta
    e     = y - y_hat
    n, k  = len(y), X.shape[1]
    s2    = np.sum(e**2) / (n - k)
    try:
        var_beta = s2 * np.linalg.inv(X.T @ X)
        se_up    = np.sqrt(var_beta[1, 1])
        se_down  = np.sqrt(var_beta[2, 2])
    except np.linalg.LinAlgError:
        se_up = se_down = np.nan

    # Test t sulla differenza b_up - b_down
    # H₀: b_up = b_down (simmetria)
    # H₁: b_up > b_down (razzo)
    if not np.isnan(se_up) and not np.isnan(se_down):
        se_diff = np.sqrt(se_up**2 + se_down**2)
        t_stat  = (b_up - b_down) / se_diff if se_diff > 0 else np.nan
        p_asym  = stats.t.sf(abs(t_stat), df=n-k) * 2 if not np.isnan(t_stat) else np.nan
    else:
        t_stat = p_asym = np.nan

    # Indice R&F: > 1 = effetto razzo, < 1 = simmetria
    rf_index = abs(b_up) / abs(b_down) if b_down != 0 else np.inf

    rf_results[fuel_name] = {
        "b_up":     b_up,
        "b_down":   b_down,
        "se_up":    se_up,
        "se_down":  se_down,
        "t_stat":   t_stat,
        "p_asym":   p_asym,
        "rf_index": rf_index,
        "brent_pos": brent_pos,
        "brent_neg": brent_neg,
        "y":        y,
    }

    print(f"\n  {fuel_name}:")
    print(f"  β_up   (Brent ↑): {b_up:.4f}  [se={se_up:.4f}]")
    print(f"  β_down (Brent ↓): {b_down:.4f}  [se={se_down:.4f}]")
    print(f"  Indice R&F:       {rf_index:.3f}  {'← RAZZO 🚀' if rf_index > 1 else '← simmetrico'}")
    print(f"  t-stat (asimm.):  {t_stat:.3f}" if not np.isnan(t_stat) else "  t-stat: N/A")
    print(f"  p-value:          {p_asym:.4f}" if not np.isnan(p_asym) else "  p-value: N/A")

    asym_result = "ASIMMETRIA SIGNIFICATIVA 🚀" if (not np.isnan(p_asym) and p_asym < 0.05) else "asimmetria non significativa"
    print(f"  Risultato:        {asym_result}")


# ─────────────────────────────────────────
# 3. ANALISI VISIVA: scatter up vs down
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Effetto 'Rockets & Feathers'\nAsimmetria nel pass-through Brent → Pompa",
             fontsize=13, fontweight="bold")

for col_idx, (fuel_name, res) in enumerate(rf_results.items()):
    # --- Scatter: variazioni Brent vs variazioni pompa ---
    ax = axes[0, col_idx]
    colors = ["#e74c3c" if b > 0 else "#3498db" for b in merged["d_brent"]]
    ax.scatter(merged["d_brent"], res["y"], c=colors, alpha=0.5, s=20)

    # Rette up e down
    x_up_range   = np.linspace(0, merged["d_brent"].max(), 50)
    x_down_range = np.linspace(merged["d_brent"].min(), 0, 50)
    ax.plot(x_up_range,   res["b_up"]   * x_up_range,   color="#e74c3c", lw=2.5,
            label=f"β_up={res['b_up']:.3f} 🚀")
    ax.plot(x_down_range, res["b_down"] * x_down_range, color="#3498db", lw=2.5,
            linestyle="--", label=f"β_down={res['b_down']:.3f} 🪶")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("ΔBrent (%)", fontsize=10)
    ax.set_ylabel(f"Δ{fuel_name} (%)", fontsize=10)
    ax.set_title(f"Scatter: Brent → {fuel_name}\n(R&F index = {res['rf_index']:.2f})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Time series: prezzi normalizzati ---
    ax2 = axes[1, col_idx]
    fuel_col_raw = "benzina_eur_l" if fuel_name == "Benzina" else "diesel_eur_l"

    # Normalizza a 100 al 2021-01-01 (base)
    base_date = merged.index[0]
    base_brent = merged["brent_7d"].iloc[0]
    base_fuel  = merged[FUELS[fuel_name]].iloc[0]

    brent_norm = merged["brent_7d"] / base_brent * 100
    fuel_norm  = merged[FUELS[fuel_name]] / base_fuel * 100

    ax2.plot(merged.index, brent_norm, color="#2c7bb6", lw=1.5, label="Brent (idx=100)")
    ax2.plot(merged.index, fuel_norm,  color="#e74c3c", lw=1.5, label=f"{fuel_name} (idx=100)")

    # Mark eventi
    for event, date in [("Ucraina", "2022-02-24"), ("Hormuz", "2026-02-28")]:
        if pd.Timestamp(date) in merged.index or \
           (merged.index[0] <= pd.Timestamp(date) <= merged.index[-1]):
            ax2.axvline(pd.Timestamp(date), color="orange", lw=1.5,
                       linestyle="--", alpha=0.8)
            ax2.text(pd.Timestamp(date), ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else 50,
                    event, rotation=90, fontsize=7, color="orange", va="bottom")

    ax2.set_ylabel("Indice (base=100)")
    ax2.set_title(f"Prezzi normalizzati: Brent vs {fuel_name}")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("plots/04_rockets_feathers.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n Plot salvato: plots/04_rockets_feathers.png")


# ─────────────────────────────────────────
# 4. TABELLA RIASSUNTIVA R&F
# ─────────────────────────────────────────
rf_table = pd.DataFrame([
    {
        "Carburante":  fuel,
        "β_up (razzo)":  round(res["b_up"], 4),
        "β_down (piuma)": round(res["b_down"], 4),
        "Indice R&F":  round(res["rf_index"], 3),
        "t-stat":      round(res["t_stat"], 3) if not np.isnan(res["t_stat"]) else "N/A",
        "p-value":     round(res["p_asym"], 4) if not np.isnan(res["p_asym"]) else "N/A",
        "Asimmetria":  "Sì ✓" if (not np.isnan(res["p_asym"]) and res["p_asym"] < 0.05) else "No",
    }
    for fuel, res in rf_results.items()
])

rf_table.to_csv("data/rockets_feathers_results.csv", index=False)
print("\n Rockets & Feathers — Risultati:")
print(rf_table.to_string(index=False))
print("\n Script 04 completato.")