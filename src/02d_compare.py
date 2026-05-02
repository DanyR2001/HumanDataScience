#!/usr/bin/env python3
"""
02d_compare.py  ─  Confronto dei 3 Metodi ITS
===============================================
Legge i CSV di output prodotti da v1, v2, v3 e crea:

  1. Tabella comparativa guadagni extra (M€) per evento × carburante × metodo
  2. Barplot gruppi: un gruppo per evento, barre per metodo (benzina + gasolio)
  3. Scatter plot: v1 vs v2, v1 vs v3, v2 vs v3 → misura di accordo
  4. Heatmap: accordo tra metodi per ogni combinazione evento+carburante

Modalità (--mode):
  fixed     : legge da data/plots/its/fixed/{metodo}/         [default]
  detected  : legge da data/plots/its/detected/{detect}/{metodo}/

Parametro --detect (solo quando --mode detected):
  margin  : legge la variante detection-su-margine  [default]
  price   : legge la variante detection-su-prezzo

Output:
  data/plots/its/{mode}/compare/              (se mode=fixed)
  data/plots/its/detected/{detect}/compare/   (se mode=detected)
    compare_table.csv
    compare_barplot.png
    compare_scatter.png
    compare_heatmap.png
"""

from __future__ import annotations
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
_OUT_BASE = BASE_DIR / "data" / "plots" / "its"

COLORS = {
    "v1_naive":        "#2c7fb8",
    "v2_intermediate": "#31a354",
    "v3_arimax":       "#e6550d",
    "v4_sarimax":      "#0de2e6",
    "v5_causalimpact": "#984ea3",
    "v6_glm_gamma":    "#ff7f00",
}

LABELS = {
    "v1_naive":        "V1 – Naïve OLS",
    "v2_intermediate": "V2 – OLS HAC",
    "v3_arimax":       "V3 – ARIMAX/ITS",
    "v4_sarimax":      "V4 – SARIMAX",
    "v5_causalimpact": "V5 – BSTS CausalImpact",
    "v6_glm_gamma":    "V6 – GLM Gamma",
}

FUEL_PATTERNS = {"benzina": "/", "gasolio": ""}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento e normalizzazione risultati
# ══════════════════════════════════════════════════════════════════════════════

def load_results(mode: str, detect_target: str = "margin") -> pd.DataFrame:
    if mode == "detected":
        its_dir = _OUT_BASE / "detected" / detect_target
    else:
        its_dir = _OUT_BASE / mode
    csv_paths = {
        "v1_naive":        its_dir / "v1_naive"        / "v1_naive_results.csv",
        "v2_intermediate": its_dir / "v2_intermediate" / "v2_intermediate_results.csv",
        "v3_arimax":       its_dir / "v3_arimax"       / "v3_arimax_results.csv",
        "v4_sarimax":      its_dir / "v4_transfer"     / "v4_sarimax_results.csv",
        "v5_causalimpact": its_dir / "v5_causalimpact" / "v5_causalimpact_results.csv",
        "v6_glm_gamma":    its_dir / "v6_glm_gamma"    / "v6_glm_gamma_results.csv",
    }

    frames = []
    for method, path in csv_paths.items():
        if not path.exists():
            print(f"  ⚠ {method}: file non trovato ({path}) – salto.")
            continue

        df = pd.read_csv(path)

        if method in ("v3_arimax", "v4_sarimax") and "is_best" in df.columns:
            df = df[df["is_best"].astype(bool)].copy()

        df["metodo"] = method

        # Normalizza colonne method-specific → nomi comuni:
        # v2 usa gain_ols_meur invece di gain_total_meur
        if "gain_total_meur" not in df.columns and "gain_ols_meur" in df.columns:
            df = df.rename(columns={"gain_ols_meur": "gain_total_meur"})
        # v5 usa abs_effect_avg_eurl invece di extra_mean_eurl
        if "extra_mean_eurl" not in df.columns and "abs_effect_avg_eurl" in df.columns:
            df = df.rename(columns={"abs_effect_avg_eurl": "extra_mean_eurl"})

        cols_keep = ["metodo", "evento", "carburante",
                     "gain_total_meur", "gain_ci_low_meur", "gain_ci_high_meur",
                     "extra_mean_eurl"]
        missing = [c for c in cols_keep if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        frames.append(df[cols_keep])

    if not frames:
        print(f"  ✗ Nessun CSV trovato in {its_dir}. Eseguire prima v1, v2, v3 con --mode {mode}"
              + (f" --detect {detect_target}" if mode == "detected" else "") + ".")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Caricati {len(frames)} metodi: {[f['metodo'].iloc[0] for f in frames]}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tabella comparativa
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["evento", "carburante"],
        columns="metodo",
        values="gain_total_meur",
        aggfunc="first",
    )
    available = [m for m in ["v1_naive","v2_intermediate","v3_arimax","v4_sarimax","v5_causalimpact","v6_glm_gamma"] if m in pivot.columns]
    if len(available) > 1:
        pivot["range_meur"] = pivot[available].max(axis=1) - pivot[available].min(axis=1)
        pivot["mean_meur"]  = pivot[available].mean(axis=1)
        pivot["sign_agree"] = (pivot[available].gt(0).all(axis=1) |
                               pivot[available].lt(0).all(axis=1)).map({True:"✓", False:"✗"})
    pivot = pivot.rename(columns=LABELS)
    return pivot.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Barplot comparativo
# ══════════════════════════════════════════════════════════════════════════════

def plot_barplot(df: pd.DataFrame, out_path: Path, mode: str) -> None:
    events  = df["evento"].unique()
    methods = df["metodo"].unique()
    fuels   = df["carburante"].unique()
    n_ev    = len(events)
    n_bars  = len(methods) * len(fuels)
    width   = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(10, n_ev * 4), 6))
    fig.suptitle(
        f"Confronto Guadagni Extra Speculativi – 3 Metodi ITS  [mode={mode}]\n"
        "(basati su stime controfattuali, ±CI 90%)",
        fontsize=11, fontweight="bold"
    )

    x = np.arange(n_ev)
    bar_idx = 0

    for method in methods:
        for fuel in fuels:
            sub = df[(df["metodo"] == method) & (df["carburante"] == fuel)]
            vals, ci_lo_err, ci_hi_err = [], [], []
            for ev in events:
                row = sub[sub["evento"] == ev]
                if row.empty:
                    vals.append(0); ci_lo_err.append(0); ci_hi_err.append(0)
                else:
                    g   = row["gain_total_meur"].values[0]
                    clo = row["gain_ci_low_meur"].values[0]
                    chi = row["gain_ci_high_meur"].values[0]
                    vals.append(g)
                    ci_lo_err.append(max(0, g - clo))
                    ci_hi_err.append(max(0, chi - g))

            offset = (bar_idx - n_bars / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=f"{LABELS.get(method, method)} – {fuel}",
                color=COLORS.get(method, "grey"),
                hatch=FUEL_PATTERNS.get(fuel, ""),
                alpha=0.85, edgecolor="white",
            )
            ax.errorbar(x + offset, vals, yerr=[ci_lo_err, ci_hi_err],
                        fmt="none", color="black", capsize=3, lw=0.8)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (1 if v >= 0 else -4),
                            f"{v:+.0f}", ha="center", va="bottom",
                            fontsize=6, rotation=0)
            bar_idx += 1

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([ev.replace("(","(\n") for ev in events], fontsize=9)
    ax.set_ylabel("Guadagno extra cumulato (M€)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=2,
              title="Metodo – Carburante", title_fontsize=7)
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Barplot: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(pivot_df: pd.DataFrame, out_path: Path, mode: str) -> None:
    available = [m for m in ["v1_naive","v2_intermediate","v3_arimax","v4_sarimax","v5_causalimpact","v6_glm_gamma"]
                 if m in pivot_df.columns]
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i+1:]]

    if not pairs:
        print("  ⚠ Scatter: meno di 2 metodi, salto.")
        return

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Accordo tra Metodi – Guadagni Extra (M€)  [mode={mode}]", fontsize=11)

    markers = {"benzina": "o", "gasolio": "s"}
    fcolors = {"benzina": "#E63946", "gasolio": "#1D3557"}
    fuels   = pivot_df["carburante"].unique() if "carburante" in pivot_df.columns else []

    for ax, (m1, m2) in zip(axes, pairs):
        for fuel in fuels:
            sub  = pivot_df[pivot_df["carburante"] == fuel]
            xs   = sub[m1].values
            ys   = sub[m2].values
            mask = ~(np.isnan(xs) | np.isnan(ys))
            if not mask.any():
                continue
            ax.scatter(xs[mask], ys[mask],
                       marker=markers.get(fuel, "o"),
                       color=fcolors.get(fuel, "grey"),
                       s=60, alpha=0.8, label=fuel.capitalize(), zorder=5)
            if "evento" in pivot_df.columns:
                for x_, y_, ev in zip(xs[mask], ys[mask], sub["evento"].values[mask]):
                    ax.annotate(ev.split("(")[0].strip()[:10], (x_, y_),
                                fontsize=5, xytext=(3, 3), textcoords="offset points")

        all_v = np.concatenate([pivot_df[m1].dropna().values, pivot_df[m2].dropna().values])
        if len(all_v):
            lo, hi = all_v.min() - 5, all_v.max() + 5
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y=x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel(LABELS.get(m1, m1) + " (M€)", fontsize=8)
        ax.set_ylabel(LABELS.get(m2, m2) + " (M€)", fontsize=8)
        ax.set_title(f"{LABELS.get(m1,m1)}\nvs\n{LABELS.get(m2,m2)}", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.20)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.axvline(0, color="grey", lw=0.6, ls=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Scatter: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df: pd.DataFrame, out_path: Path, mode: str) -> None:
    available = [m for m in ["v1_naive","v2_intermediate","v3_arimax","v4_sarimax","v5_causalimpact","v6_glm_gamma"]
                 if m in df["metodo"].unique()]

    if len(available) < 2:
        print("  ⚠ Heatmap: meno di 2 metodi, salto.")
        return

    pivot = df[df["metodo"].isin(available)].pivot_table(
        index=["evento", "carburante"], columns="metodo",
        values="gain_total_meur", aggfunc="first",
        dropna=False,
    )

    if pivot.empty or pivot.values.size == 0:
        print("  ⚠ Heatmap: pivot vuoto, salto.")
        return

    data_np = pivot.values.astype(float)
    if np.all(np.isnan(data_np)):
        print("  ⚠ Heatmap: tutti i valori NaN, salto.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(available)*2), max(4, len(pivot)*0.8)))
    fig.suptitle(f"Guadagni Extra (M€) – Confronto Metodi  [mode={mode}]",
                 fontsize=11, fontweight="bold")

    vmax    = np.nanmax(np.abs(data_np)) + 1e-6
    norm    = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im      = ax.imshow(data_np, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)
    plt.colorbar(im, ax=ax, label="Guadagno (M€)", shrink=0.8)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([LABELS.get(c, c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{ev}\n{fuel}" for ev, fuel in pivot.index], fontsize=7)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data_np[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(v) > vmax*0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Heatmap: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Accordo tra segni
# ══════════════════════════════════════════════════════════════════════════════

def print_sign_agreement(pivot_df: pd.DataFrame) -> None:
    available = [m for m in ["v1_naive","v2_intermediate","v3_arimax","v4_sarimax","v5_causalimpact","v6_glm_gamma"]
                 if m in pivot_df.columns]
    if len(available) < 2:
        return

    print("\n  ACCORDO TRA METODI (segno del guadagno):")
    print(f"  {'Evento':<28} {'Carb.':<10}", end="")
    for m in available:
        print(f"  {LABELS.get(m,m)[:14]:>14}", end="")
    print("  Accordo segno")
    print("  " + "─"*90)

    for _, row in pivot_df.iterrows():
        ev   = str(row.get("evento",""))[:27]
        fuel = str(row.get("carburante",""))[:9]
        vals = [row.get(m, np.nan) for m in available]
        signs = [np.sign(v) for v in vals if not (isinstance(v, float) and np.isnan(v))]
        agree = "✓" if len(set(signs)) == 1 else "✗"
        print(f"  {ev:<28} {fuel:<10}", end="")
        for v in vals:
            s = f"{v:+.0f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a"
            print(f"  {s:>14}", end="")
        print(f"  {agree}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Confronto 3 metodi ITS")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed o detected: deve corrispondere ai file già prodotti")
    parser.add_argument("--detect", choices=["margin", "price"], default="margin",
                        help="(solo mode=detected) variante da confrontare: "
                             "margin [default] o price")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "compare"
    else:
        OUT_DIR = _OUT_BASE / mode / "compare"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print(f"  02d_compare.py  –  Confronto 3 Metodi ITS  [mode={mode}]")
    if mode == "detected":
        print(f"  Variante detection: {detect_target}")
        print(f"  Legge da: {_OUT_BASE / 'detected' / detect_target}")
    else:
        print(f"  Legge da: {_OUT_BASE / mode}")
    print(f"  Output:   {OUT_DIR}")
    print("═"*70)

    df = load_results(mode, detect_target)
    if df.empty:
        return

    n_methods_with_data = df.groupby("metodo")["gain_total_meur"].apply(
        lambda s: s.notna().any()
    ).sum()
    if n_methods_with_data == 0:
        print("  ✗ gain_total_meur assente in tutti i metodi caricati. "
              "Verificare i nomi colonne nei CSV.")
        return

    mode_label = f"detected/{detect_target}" if mode == "detected" else mode

    # ── 1. Tabella ────────────────────────────────────────────────────────────
    table = make_comparison_table(df)
    csv_out = OUT_DIR / "compare_table.csv"
    table.to_csv(csv_out, index=False)
    print(f"\n  → Tabella: {csv_out}")
    print("\n" + table.to_string(index=False))

    # ── Accordo segni ─────────────────────────────────────────────────────────
    pivot_raw = df.pivot_table(
        index=["evento","carburante"], columns="metodo",
        values="gain_total_meur", aggfunc="first",
    ).reset_index()
    print_sign_agreement(pivot_raw)

    # ── 2–4. Plot ─────────────────────────────────────────────────────────────
    plot_barplot(df, OUT_DIR / "compare_barplot.png", mode_label)
    plot_scatter(pivot_raw, OUT_DIR / "compare_scatter.png", mode_label)
    plot_heatmap(df, OUT_DIR / "compare_heatmap.png", mode_label)

    # ── Statistiche ───────────────────────────────────────────────────────────
    available = [m for m in ["v1_naive","v2_intermediate","v3_arimax","v4_sarimax","v5_causalimpact","v6_glm_gamma"]
                 if m in pivot_raw.columns]
    if len(available) >= 2:
        gains  = pivot_raw[available].values.astype(float)
        ranges = np.nanmax(gains, axis=1) - np.nanmin(gains, axis=1)
        means  = np.nanmean(gains, axis=1)
        cv     = np.abs(ranges / np.where(means != 0, means, np.nan))
        print(f"\n  STATISTICHE ACCORDO INTER-METODO:")
        print(f"    Range medio (M€):  {np.nanmean(ranges):+.1f}")
        print(f"    Range max  (M€):   {np.nanmax(ranges):+.1f}")
        print(f"    CV medio:          {np.nanmean(cv)*100:.1f}%")


if __name__ == "__main__":
    main()