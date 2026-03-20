"""
01_data_pipeline.py
====================
Scarica e prepara i dati per l'analisi:
  - Brent crude oil (giornaliero) via yfinance
  - Prezzi carburanti Italia (settimanale) via EU Weekly Oil Bulletin
  - Preprocessing: rolling average, log transform
  - Salva i dati puliti in /data/
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────
# 1. BRENT CRUDE (giornaliero)
# ─────────────────────────────────────────
print(" Scarico Brent crude (yfinance)...")

try:
    brent_raw = yf.download("BZ=F", start="2021-01-01", end="2026-03-20", progress=False)
    if brent_raw.empty:
        raise ValueError("Download vuoto")
    brent = brent_raw[["Close"]].copy()
    brent.columns = ["brent_usd"]
    brent.index = pd.to_datetime(brent.index)
    brent.dropna(inplace=True)
    print("   Brent scaricato da yfinance.")
except Exception as e:
    print(f"     yfinance non disponibile ({e}). Uso dati storici realistici.")
    # Dati Brent storici realistici (basati su valori reali)
    # Pre-Ucraina: ~80$/b | Post-Ucraina spike: ~130$/b | Normalizzazione: ~75-85$/b
    # Post-Hormuz 2026: spike a ~105$/b
    from scipy.ndimage import uniform_filter1d
    dates = pd.date_range("2021-01-04", "2026-03-19", freq="B")  # business days
    n = len(dates)
    prices = np.zeros(n)
    np.random.seed(42)
    for i, d in enumerate(dates):
        if d < pd.Timestamp("2022-02-24"):
            base = 80.0
        elif d < pd.Timestamp("2022-03-10"):
            # Spike immediato: Brent reagisce in 1-3 giorni (mercato futures)
            days_after = (d - pd.Timestamp("2022-02-24")).days
            base = 80.0 + min(days_after * 3.5, 50)  # sale da 80 a 130 in ~15gg
        elif d < pd.Timestamp("2022-07-01"):
            base = 120.0 - (d - pd.Timestamp("2022-03-10")).days * 0.15
        elif d < pd.Timestamp("2023-01-01"):
            base = 95.0
        elif d < pd.Timestamp("2024-01-01"):
            base = 85.0
        elif d < pd.Timestamp("2025-01-01"):
            base = 80.0
        elif d < pd.Timestamp("2026-02-28"):
            base = 73.0
        else:
            # Hormuz shock: spike immediato
            days_after = (d - pd.Timestamp("2026-02-28")).days
            base = 73.0 + min(days_after * 2.0, 35)
        prices[i] = base + np.random.normal(0, 1.5)
    prices = uniform_filter1d(prices, size=3)
    brent = pd.DataFrame({"brent_usd": prices}, index=dates)
    print("     Nota: dati Brent SIMULATI. Sul tuo PC yfinance scarica quelli reali.")

# Rolling average 7 giorni per rimuovere rumore
brent["brent_7d"] = brent["brent_usd"].rolling(7, min_periods=1).mean()

# Log transform (rende la crescita esponenziale lineare)
brent["log_brent"] = np.log(brent["brent_7d"])

print(f"   Brent: {len(brent)} osservazioni | {brent.index[0].date()} → {brent.index[-1].date()}")
brent.to_csv("data/brent_daily.csv")


# ─────────────────────────────────────────
# 2. PREZZI POMPA ITALIA (settimanale)
#    Fonte: EU Weekly Oil Bulletin
#    https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en
#    Colonne di interesse: Italia, benzina senza piombo 95, gasolio
# ─────────────────────────────────────────
print("\n Scarico prezzi pompa Italia (EU Oil Bulletin)...")

# URL diretto al dataset Excel dell'EU Oil Bulletin
EU_URL = (
    "https://energy.ec.europa.eu/system/files/2024-12/"
    "Prices_with_taxes_and_levies_for_motor_fuels_from_2005.xlsx"
)

try:
    resp = requests.get(EU_URL, timeout=30)
    with open("data/eu_oil_bulletin.xlsx", "wb") as f:
        f.write(resp.content)
    print("   File EU Oil Bulletin scaricato.")
    eu_downloaded = True
except Exception as e:
    print(f"     Download EU Bulletin fallito ({e}), uso dati simulati per il demo.")
    eu_downloaded = False


if eu_downloaded:
    try:
        # Il file EU ha più fogli: uno per benzina, uno per diesel
        xl = pd.ExcelFile("data/eu_oil_bulletin.xlsx")
        print(f"   Fogli disponibili: {xl.sheet_names}")

        # Foglio benzina — riga 0 è header con paesi, colonna 0 è data
        df_benzina = pd.read_excel(
            "data/eu_oil_bulletin.xlsx",
            sheet_name=0,
            header=0,
            index_col=0,
        )
        # Cerca colonna Italia (IT o Italy)
        it_cols = [c for c in df_benzina.columns if "IT" in str(c).upper() or "ITALY" in str(c).upper() or "ITAL" in str(c).upper()]
        print(f"   Colonne Italia trovate (benzina): {it_cols}")

        if it_cols:
            benzina_it = df_benzina[it_cols[0]].copy()
            benzina_it.index = pd.to_datetime(benzina_it.index, errors="coerce")
            benzina_it = benzina_it.dropna()
            benzina_it.name = "benzina_eur_l"
        else:
            raise ValueError("Colonna Italia non trovata nel file EU")

        # Foglio diesel
        df_diesel = pd.read_excel(
            "data/eu_oil_bulletin.xlsx",
            sheet_name=1,
            header=0,
            index_col=0,
        )
        it_cols_d = [c for c in df_diesel.columns if "IT" in str(c).upper() or "ITALY" in str(c).upper() or "ITAL" in str(c).upper()]
        if it_cols_d:
            diesel_it = df_diesel[it_cols_d[0]].copy()
            diesel_it.index = pd.to_datetime(diesel_it.index, errors="coerce")
            diesel_it = diesel_it.dropna()
            diesel_it.name = "diesel_eur_l"
        else:
            raise ValueError("Colonna Italia diesel non trovata")

        pompa = pd.concat([benzina_it, diesel_it], axis=1).dropna()
        eu_parsed = True

    except Exception as e:
        print(f"     Parsing EU file fallito ({e}), uso fallback simulato.")
        eu_parsed = False
else:
    eu_parsed = False


if not eu_parsed:
    # ─── FALLBACK: dati sintetici realistici ───────────────────────────
    # Basati su dati storici reali (Staffetta Quotidiana / MASE)
    # Utile per sviluppo e test prima di avere i dati reali
    print("   Generazione dati realistici di fallback...")

    date_range = pd.date_range("2021-01-04", "2026-03-17", freq="W-MON")

    # Scenario realistico: benzina ~1.50 pre-Ucraina, spike a 2.20 post
    np.random.seed(42)
    n = len(date_range)

    def make_fuel_series(base, ukraine_shock, hormuz_shock, noise_std):
        prices = np.ones(n) * base
        for i in range(n):
            d = date_range[i]
            if d >= pd.Timestamp("2022-03-15"):   # ~3 settimane post Ucraina
                prices[i] = base + ukraine_shock
            if d >= pd.Timestamp("2026-03-20"):   # ~3 settimane post Hormuz
                prices[i] = base + ukraine_shock + hormuz_shock
        # Aggiusta gradualmente invece di step function
        from scipy.ndimage import uniform_filter1d
        prices = uniform_filter1d(prices, size=4)
        prices += np.random.normal(0, noise_std, n)
        return prices

    benzina = make_fuel_series(1.55, 0.50, 0.25, 0.015)
    diesel  = make_fuel_series(1.40, 0.55, 0.30, 0.015)

    pompa = pd.DataFrame({
        "benzina_eur_l": benzina,
        "diesel_eur_l":  diesel,
    }, index=date_range)
    print("     Nota: questi sono dati di FALLBACK per test. Sostituisci con dati reali MASE.")


# ─────────────────────────────────────────
# 3. PREPROCESSING PREZZI POMPA
# ─────────────────────────────────────────
pompa = pompa[pompa.index >= "2021-01-01"].copy()
pompa.sort_index(inplace=True)

# Rolling average 4 settimane (analogo al 7gg sul giornaliero)
pompa["benzina_4w"] = pompa["benzina_eur_l"].rolling(4, min_periods=1).mean()
pompa["diesel_4w"]  = pompa["diesel_eur_l"].rolling(4, min_periods=1).mean()

# Log transform
pompa["log_benzina"] = np.log(pompa["benzina_4w"])
pompa["log_diesel"]  = np.log(pompa["diesel_4w"])

print(f"\n   Prezzi pompa: {len(pompa)} osservazioni | {pompa.index[0].date()} → {pompa.index[-1].date()}")
pompa.to_csv("data/prezzi_pompa_italia.csv")


# ─────────────────────────────────────────
# 4. DATASET UNIFICATO (merge settimanale)
# ─────────────────────────────────────────
# Ricampiona il Brent a frequenza settimanale (media della settimana)
brent_weekly = brent[["brent_usd", "brent_7d", "log_brent"]].resample("W-MON").mean()

# Merge
merged = pd.concat([brent_weekly, pompa], axis=1, join="inner")
merged.dropna(inplace=True)
merged.to_csv("data/dataset_merged.csv")
print(f"\n   Dataset unificato: {len(merged)} settimane | {merged.index[0].date()} → {merged.index[-1].date()}")


# ─────────────────────────────────────────
# 5. PLOT OVERVIEW
# ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Dati grezzi — Brent e prezzi pompa Italia", fontsize=14, fontweight="bold")

# Eventi di guerra
war_events = {
    "Invasione Ucraina\n(24 feb 2022)": "2022-02-24",
    "Hormuz closure\n(feb 2026)":        "2026-02-28",
}

for ax in axes:
    for label, date in war_events.items():
        ax.axvline(pd.Timestamp(date), color="red", linestyle="--", alpha=0.7, lw=1.5)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                label, rotation=90, fontsize=7, color="red", va="top")

# Brent
axes[0].plot(merged.index, merged["brent_usd"], color="#2c7bb6", lw=1.5, label="Brent (raw)")
axes[0].plot(merged.index, merged["brent_7d"],  color="#d7191c", lw=1.5, label="Brent (7d avg)")
axes[0].set_ylabel("USD/barile")
axes[0].legend(fontsize=9)
axes[0].set_title("Prezzo Brent Crude")
axes[0].grid(alpha=0.3)

# Benzina
axes[1].plot(merged.index, merged["benzina_eur_l"], color="#fdae61", lw=1.5, label="Benzina (raw)")
axes[1].plot(merged.index, merged["benzina_4w"],    color="#d7191c", lw=1.5, label="Benzina (4w avg)")
axes[1].set_ylabel("€/litro")
axes[1].legend(fontsize=9)
axes[1].set_title("Prezzo Benzina Italia (€/litro, tasse incluse)")
axes[1].grid(alpha=0.3)

# Diesel
axes[2].plot(merged.index, merged["diesel_eur_l"], color="#abdda4", lw=1.5, label="Diesel (raw)")
axes[2].plot(merged.index, merged["diesel_4w"],    color="#1a9641", lw=1.5, label="Diesel (4w avg)")
axes[2].set_ylabel("€/litro")
axes[2].legend(fontsize=9)
axes[2].set_title("Prezzo Diesel Italia (€/litro, tasse incluse)")
axes[2].grid(alpha=0.3)

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("plots/01_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n Plot salvato: plots/01_overview.png")
print(" Script 01 completato. Dati salvati in /data/")