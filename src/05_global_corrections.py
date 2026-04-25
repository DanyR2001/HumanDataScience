"""
05_global_corrections.py
=========================
Correzione per test multipli su tutti i p-value confirmatory del paper.

PERCHE' QUESTO PASSAGGIO E' NECESSARIO
----------------------------------------
Gli script 03 e 04 hanno prodotto:
  - 16 test confirmatory su H0 (Welch t, Mann-Whitney, block permutation, HAC)
    per 2 eventi x 2 carburanti x 4 metodi  ->  data/confirmatory_pvalues.csv
  - N test DiD (specificita' italiana)
    per 2 eventi x 2 paesi x 2 carburanti   ->  data/auxiliary_pvalues.csv

Applicare alfa = 0.05 separatamente gonfia il tasso di falsi positivi:
con 20 test indipendenti ci aspettiamo ~1 rigetto spurio per caso.
La Benjamini-Hochberg correction (FDR <= 5%) controlla l'atteso di falsi
positivi sull'intera famiglia di test confirmatory.

FAMIGLIA DI TEST
-----------------
Sono confirmatory solo i test che testano direttamente H0:
"il margine lordo non aumenta anomalmente rispetto al baseline 2019"
Granger, R&F, KS, ANOVA, Chow sono esplorativi (velocita', asimmetria
strutturale) e non entrano nella famiglia per non confondere domande diverse.

NOTA SULLA DIPENDENZA TRA TEST
--------------------------------
I test Welch t, MW, block perm, HAC sulla stessa coppia evento x carburante
condividono i dati. Con correlazione positiva, BH e' conservativa:
il FDR reale e' <= 5% (Benjamini & Yekutieli 2001).

Input:
  data/confirmatory_pvalues.csv   (script 03)
  data/auxiliary_pvalues.csv      (script 04: DiD)
  data/table2_margin_anomaly.csv  (aggiornato con BH locale in script 03)

Output:
  data/global_bh_corrections.csv
  data/table2_margin_anomaly.csv  (aggiornato con BH_global_reject)
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

ALPHA = 0.05


def bh_correction(p_values: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg (1995) con monotonicity enforcement."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order  = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    p_adj  = np.minimum(1.0, p * n / ranked)
    p_adj_m = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_out   = np.empty(n)
    p_out[order] = p_adj_m
    return p_out <= alpha, p_out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raccolta p-value confirmatory
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Raccolta p-value confirmatory")
print("=" * 65)

all_rows = []

for path, label in [
    ("data/confirmatory_pvalues.csv", "Script 03 (margine)"),
    ("data/auxiliary_pvalues.csv",    "Script 04 (DiD)"),
]:
    if os.path.exists(path):
        df_src = pd.read_csv(path)
        for _, row in df_src.iterrows():
            all_rows.append({
                "fonte":       row["fonte"],
                "tipo":        "confirmatory",
                "descrizione": row["descrizione"],
                "p_value":     float(row["p_value"]),
            })
        print(f"  {label}: {len(df_src)} test caricati")
    else:
        print(f"  {label}: {path} non trovato — skip")

if not all_rows:
    raise SystemExit("Nessun p-value trovato. Eseguire prima 03 e 04.")

df_all = pd.DataFrame(all_rows)
df_all["p_value"] = pd.to_numeric(df_all["p_value"], errors="coerce")
df_all = df_all.dropna(subset=["p_value"])
n_total = len(df_all)
print(f"\n  Totale test confirmatory: {n_total}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BH globale
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Benjamini-Hochberg FDR correction (alfa = 5%)")
print("=" * 65)

reject, p_adj = bh_correction(df_all["p_value"].values, alpha=ALPHA)
df_all["BH_global_reject"]    = reject
df_all["p_value_BH_adjusted"] = p_adj

n_reject = int(reject.sum())
print(f"\n  Famiglia: {n_total} test  |  Rigettati a FDR 5%: {n_reject}")

print("\n  Test con H0 rigettata (ordinati per p nominale):")
df_rej = df_all[df_all["BH_global_reject"]].sort_values("p_value")
if df_rej.empty:
    print("    Nessuno.")
else:
    for _, row in df_rej.iterrows():
        print(f"    p={row['p_value']:.4f} adj={row['p_value_BH_adjusted']:.4f}"
              f" | {row['fonte'].split('_')[0]:12} | {row['descrizione'][:45]}")

print("\n  Elenco completo:")
for _, row in df_all.sort_values("p_value").iterrows():
    flag = "RIGETTATA" if row["BH_global_reject"] else "         "
    print(f"  {flag} | p={row['p_value']:.4f} adj={row['p_value_BH_adjusted']:.4f}"
          f" | {row['fonte'].split('_')[0]:12} | {row['descrizione'][:40]}")

df_all.to_csv("data/global_bh_corrections.csv", index=False)
print(f"\n  Salvato: data/global_bh_corrections.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggiorna table2 con BH globale
# ─────────────────────────────────────────────────────────────────────────────
t2_path = "data/table2_margin_anomaly.csv"
if os.path.exists(t2_path):
    df_t2    = pd.read_csv(t2_path)
    df_welch = df_all[df_all["fonte"].str.startswith("Welch_t")].reset_index(drop=True)

    # Hormuz è escluso dalla BH (dati preliminari) → Welch_t ha 4 righe,
    # table2 ne ha 6 (3 eventi × 2 carburanti).
    # Uniamo solo sulle righe non-preliminari, usando l'ordine posizionale
    # (entrambe le fonti seguono lo stesso ordinamento evento × carburante).
    prel_col = "preliminare" if "preliminare" in df_t2.columns else None
    if prel_col:
        nonprel_idx = df_t2[~df_t2[prel_col].astype(bool)].index
    else:
        nonprel_idx = df_t2.index   # fallback: aggiorna tutto

    if len(df_welch) == len(nonprel_idx):
        df_t2.loc[nonprel_idx, "BH_global_reject"]       = df_welch["BH_global_reject"].values
        df_t2.loc[nonprel_idx, "t_p_BH_global_adjusted"] = df_welch["p_value_BH_adjusted"].values
        # Hormuz: segna esplicitamente come non incluso nella BH globale
        prel_rows = df_t2[df_t2[prel_col].astype(bool)].index if prel_col else []
        df_t2.loc[prel_rows, "BH_global_reject"]       = pd.NA
        df_t2.loc[prel_rows, "t_p_BH_global_adjusted"] = pd.NA
        df_t2.to_csv(t2_path, index=False)
        print(f"  Aggiornato: {t2_path}")
        print(f"    Non-prelim: {len(nonprel_idx)} righe aggiornate con BH globale")
        print(f"    Preliminari (Hormuz): {len(df_t2) - len(nonprel_idx)} righe → BH_global_reject = NA")
    else:
        print(f"  Mismatch table2 non-prelim ({len(nonprel_idx)}) vs Welch_t ({len(df_welch)}) — non aggiornato")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sommario
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SOMMARIO — risultati finali dopo correzione per test multipli")
print("=" * 65)

for fonte_prefix in ["Welch_t", "MannWhitney", "BlockPerm", "HAC", "DiD"]:
    sub = df_all[df_all["fonte"].str.startswith(fonte_prefix)]
    if sub.empty:
        continue
    n_rej_s = int(sub["BH_global_reject"].sum())
    print(f"  {fonte_prefix:15}: {n_rej_s:2d} / {len(sub):2d} rigettati (FDR 5%)")

print(f"""
  Nota: i test Welch t, MW, block perm e HAC sulla stessa coppia
  evento x carburante sono correlati positivamente. BH con correlazione
  positiva e' conservativa: il FDR reale e' <= 5%.

  Le classificazioni descrivono pattern statistici osservati, non cause
  economiche: "margine anomalo positivo" e' consistente con comportamento
  speculativo ma anche con effetti FIFO/LIFO, risk premium razionale,
  cost-push non catturato dalla proxy ARA/ICE.
""")

print("Script 05 completato.")