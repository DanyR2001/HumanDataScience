"""
04_rocket_feather.py — Rockets & Feathers, plot paper quality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
merged["d_brent"]   = merged["brent_7d"].pct_change() * 100
merged["d_benzina"] = merged["benzina_4w"].pct_change() * 100
merged["d_diesel"]  = merged["diesel_4w"].pct_change() * 100
merged.dropna(inplace=True)

DPI    = 180
FUELS  = {"Benzina": ("d_benzina","benzina_4w","#d6604d"),
          "Diesel":  ("d_diesel", "diesel_4w", "#31a354")}

WAR_EVENTS = {
    "Ucraina": ("2022-02-24", "#e74c3c"),
    "Iran-Israele": ("2025-06-13", "#e67e22"),
    "Hormuz": ("2026-02-28", "#8e44ad"),
}

rf_results = {}

print("ROCKETS & FEATHERS\n" + "="*50)
for fuel_name, (dcol, col4w, fc) in FUELS.items():
    y         = merged[dcol].values
    brent_pos = np.maximum(merged["d_brent"].values, 0)
    brent_neg = np.minimum(merged["d_brent"].values, 0)
    X         = np.column_stack([np.ones(len(y)), brent_pos, brent_neg])
    beta, *_  = np.linalg.lstsq(X, y, rcond=None)
    _, b_up, b_down = beta

    y_hat  = X @ beta
    e      = y - y_hat
    n, k   = len(y), 3
    s2     = np.sum(e**2) / (n-k)
    try:
        var_b  = s2 * np.linalg.inv(X.T @ X)
        se_up, se_down = np.sqrt(var_b[1,1]), np.sqrt(var_b[2,2])
        se_diff = np.sqrt(se_up**2 + se_down**2)
        t_stat  = (b_up - b_down) / se_diff
        p_asym  = stats.t.sf(abs(t_stat), df=n-k) * 2
    except Exception:
        se_up = se_down = t_stat = p_asym = np.nan

    rf_index = abs(b_up)/abs(b_down) if b_down != 0 else np.inf
    rf_results[fuel_name] = {"b_up": b_up, "b_down": b_down,
                              "rf_index": rf_index, "t_stat": t_stat,
                              "p_asym": p_asym, "col": fc, "col4w": col4w}

    print(f"  {fuel_name}: β_up={b_up:.4f}  β_down={b_down:.4f}  "
          f"R&F={rf_index:.3f}  p={p_asym:.4f}")

# ── Plot 1: Scatter beta_up vs beta_down — un plot per carburante
for fuel_name, res in rf_results.items():
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#e74c3c" if b > 0 else "#3498db" for b in merged["d_brent"]]
    ax.scatter(merged["d_brent"], merged[FUELS[fuel_name][0]],
               c=colors, alpha=0.45, s=22)

    x_up   = np.linspace(0, merged["d_brent"].max(), 100)
    x_down = np.linspace(merged["d_brent"].min(), 0, 100)
    ax.plot(x_up,   res["b_up"]   * x_up,   color="#e74c3c", lw=2.8,
            label=f"β_up = {res['b_up']:.4f}  (Brent ↑)")
    ax.plot(x_down, res["b_down"] * x_down, color="#3498db", lw=2.8,
            linestyle="--", label=f"β_down = {res['b_down']:.4f}  (Brent ↓)")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    p_str = f"{res['p_asym']:.4f}" if not np.isnan(res["p_asym"]) else "N/A"
    ax.set_title(f"Rockets & Feathers — {fuel_name}\n"
                 f"R&F index = {res['rf_index']:.3f}   p-value asimmetria = {p_str}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("ΔBrent (%)", fontsize=12)
    ax.set_ylabel(f"Δ{fuel_name} (%)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(f"plots/04_rf_scatter_{fuel_name.lower()}.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: plots/04_rf_scatter_{fuel_name.lower()}.png")

# ── Plot 2: Prezzi normalizzati nel tempo — un plot per carburante
base_date = merged.index[0]
brent_norm_base = merged["brent_7d"].iloc[0]

for fuel_name, res in rf_results.items():
    col4w  = res["col4w"]
    fc     = res["col"]
    fuel_norm_base = merged[col4w].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged.index, merged["brent_7d"] / brent_norm_base * 100,
            color="#2166ac", lw=2.0, label="Brent (base=100)")
    ax.plot(merged.index, merged[col4w] / fuel_norm_base * 100,
            color=fc, lw=2.0, label=f"{fuel_name} (base=100)")

    for label, (date, color) in WAR_EVENTS.items():
        ts = pd.Timestamp(date)
        if merged.index[0] <= ts <= merged.index[-1]:
            ax.axvline(ts, color=color, lw=1.8, linestyle="--", alpha=0.85)
            ax.text(ts + pd.Timedelta(days=5), ax.get_ylim()[1] if ax.get_ylim()[1]!=1 else 150,
                    label, rotation=90, fontsize=9, color=color, va="top")

    ax.set_ylabel("Indice (base = 100)", fontsize=12)
    ax.set_title(f"Prezzi normalizzati: Brent vs {fuel_name} — 2021–2026",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"plots/04_rf_norm_{fuel_name.lower()}.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: plots/04_rf_norm_{fuel_name.lower()}.png")

# Salva risultati
pd.DataFrame([{
    "Carburante": f, "β_up": round(r["b_up"],4), "β_down": round(r["b_down"],4),
    "R&F index": round(r["rf_index"],3),
    "t-stat": round(r["t_stat"],3) if not np.isnan(r["t_stat"]) else "N/A",
    "p-value": round(r["p_asym"],4) if not np.isnan(r["p_asym"]) else "N/A",
} for f,r in rf_results.items()]).to_csv("data/rockets_feathers_results.csv", index=False)

print("\nScript 04 completato.")