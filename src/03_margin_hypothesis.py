"""
03_margin_hypothesis.py  (v2: H0 riformulata, changepoint margine, grafici aggiornati)
========================

IPOTESI NULLA E ALTERNATIVA (revisione)
-----------------------------------------
La H0 precedente confrontava la finestra pre-shock con quella post-shock
(two-sample), testando se ci fosse un salto locale intorno all'evento.
Il limite è che se il margine era già elevato nella finestra pre-shock
(coerente con D = −46/−73 giorni dei changepoint sui prezzi) il delta
locale risulta piccolo anche se il margine post-shock è ben sopra il 2019.

  H0 (nuova):  Il margine lordo medio nel periodo post-shock è
               statisticamente indistinguibile dal livello medio 2019.
               μ_post = μ_baseline_2019

  H1 (nuova):  Il margine lordo medio nel periodo post-shock è
               significativamente superiore al livello medio 2019.
               μ_post > μ_baseline_2019   (one-sided, upper tail)

  Test primario:  one-sample Welch t (ttest_1samp vs μ_2019, one-sided)
  Test secondari: Mann-Whitney, Block perm, HAC — mantengono il confronto
                  pre-window vs post-window per misurare il salto locale.

ANALISI PRE-SHOCK
------------------
Poiché i changepoint sui log-prezzi anticipano lo shock di 46–73 giorni,
testiamo anche la finestra pre-shock vs baseline 2019:
  - δ_pre = mean(pre_window) − μ_2019
  - pre_anomalo = δ_pre > 2σ_2019
  Se TRUE, il margine era già elevato prima dello shock → coerente con
  anticipazione di mercato, non con speculazione post-evento.

CHANGEPOINT SUL MARGINE
------------------------
Per ogni evento × carburante calcoliamo anche τ_margin = data in cui il
margine ha subito la sua rottura strutturale più netta (argmax del t di
Welch su tutte le possibili partizioni). Viene confrontato con τ_price
(proveniente da script 02, tabella 1) per vedere se il margine si è
rotto prima o dopo il prezzo.

NOTA HORMUZ: i dati post-shock Hormuz (feb 2026) coprono solo ~7-8 settimane
al momento dell'analisi. I risultati sono preliminari e da non citare come
conclusivi nella relazione; vengono inclusi per completezza della pipeline.

Input:
  data/dataset_merged_with_futures.csv
  data/regression_diagnostics.csv
  data/table1_changepoints.csv          <- prodotto da 02_changepoint.py

Output:
  data/table2_margin_anomaly.csv
  data/confirmatory_pvalues.csv
  data/baseline_sensitivity.csv
  plots/03_margins.png                  <- ora include τ_price e τ_margin
  plots/03_delta_summary.png

NOTA SULLA DOPPIA IMPLEMENTAZIONE PERM/HAC
--------------------------------------------
Block permutation e HAC vengono eseguiti con due split distinti:

  [PRINCIPALE]  split su τ_price (changepoint sul log-prezzo, da script 02).
    - Esogeno al margine → nessuna circolarità.
    - Pre window termina prima del segnale anticipatorio di prezzo.
    - Entra nella famiglia BH correction (confirmatory_pvalues.csv).

  [ROBUSTNESS]  split su τ_margin (changepoint strutturale del margine stesso).
    - Endogeno: massimizza il segnale catturato nella finestra post.
    - Utile quando τ_margin ≠ τ_price (IRAN-ISRAELE: τ_margin precede τ_price
      di ~7gg; UCRAINA: τ_margin segue τ_price di ~70gg).
    - NON entra nella BH; riportato come check metodologico nel CSV.

Convergenza perm_split_convergence / hac_split_convergence:
  True  → entrambi gli split concordano sul rigetto/non-rigetto. Risultato robusto.
  False → divergenza: documentare nella discussione del paper.

Il Welch 1-sample (test primario BH) non è influenzato dalla scelta dello
split perché confronta solo la finestra post vs μ_2019, senza usare un pre.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy import stats
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

ALPHA        = 0.05
N_PERM       = 10_000
SEED         = 42
BLOCK_SIZE   = 4
HAC_MAXLAGS  = 4
DPI          = 180

BASELINE_START = "2019-01-01"
BASELINE_END   = "2019-12-31"

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-09-01"),
        "post_end":  pd.Timestamp("2022-08-31"),
        "preliminare": False,
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-01-01"),
        "post_end":  pd.Timestamp("2025-10-31"),
        "preliminare": False,
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-10-01"),
        "post_end":  pd.Timestamp("2026-04-27"),   # dati fino a oggi
        "preliminare": True,   # <8 settimane post-shock
    },
}

MARGIN_COLS = {
    "Benzina": "margine_benz_crack",
    "Diesel":  "margine_dies_crack",
}

WAR_DATES = {
    "Ucraina": (pd.Timestamp("2022-02-24"), "#e74c3c"),
    "Iran":    (pd.Timestamp("2025-06-13"), "#e67e22"),
    "Hormuz":  (pd.Timestamp("2026-02-28"), "#8e44ad"),
}

CLAS_COLOR = {
    "Margine anomalo positivo":     "#c0392b",
    "Compressione margine":         "#2980b9",
    "Neutro / trasmissione attesa": "#27ae60",
    "Variazione statistica":        "#e67e22",
    "Inconclusivo":                 "#95a5a6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def bh_correction(p_values, alpha=0.05):
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order   = np.argsort(p)
    ranked  = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    p_adj   = np.minimum(1.0, p * n / ranked)
    p_adj_m = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_adj_out = np.empty(n)
    p_adj_out[order] = p_adj_m
    return p_adj_out <= alpha, p_adj_out


def block_permutation(combined, n_post, rng, block_size=BLOCK_SIZE):
    n = len(combined)
    n_blocks = int(np.ceil(n / block_size))
    blocks   = [combined[i*block_size : min((i+1)*block_size, n)]
                for i in range(n_blocks)]
    rng.shuffle(blocks)
    perm = np.concatenate(blocks)
    return float(np.median(perm[-n_post:]) - np.median(perm[:-n_post]))


def perm_test(pre, post, n_perm=N_PERM, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    obs  = float(np.median(post) - np.median(pre))
    comb = np.concatenate([pre, post])
    n_po = len(post)
    nulls = [block_permutation(comb, n_po, rng) for _ in range(n_perm)]
    p = float(np.mean(np.array(nulls) >= obs))
    return obs, p


def mann_whitney_full(pre, post):
    U_one, p_one = mannwhitneyu(post, pre, alternative="greater")
    U_two, p_two = mannwhitneyu(post, pre, alternative="two-sided")
    U_max = len(pre) * len(post)
    hl  = float(np.median(np.array([p - q for p in post for q in pre])))
    pre_s = np.sort(pre)
    more  = sum(np.searchsorted(pre_s, x, side="left")       for x in post)
    less  = sum(len(pre_s) - np.searchsorted(pre_s, x, "right") for x in post)
    cd    = (more - less) / (len(post) * len(pre))
    mag   = ("trascurabile" if abs(cd) < 0.147 else
             "piccolo"       if abs(cd) < 0.330 else
             "medio"         if abs(cd) < 0.474 else "grande")
    return {
        "U_stat": round(U_one, 1), "U_max": U_max,
        "AUC":    round(float(U_one/U_max), 3),
        "p_one":  round(p_one, 4), "p_two": round(p_two, 4),
        "hodges_lehmann": round(hl, 5),
        "cliffs_delta": round(cd, 3), "magnitude": mag,
        "mw_H0": "RIFIUTATA" if p_one < ALPHA else "non rifiutata",
    }


def hac_test(pre, post):
    y   = np.concatenate([pre, post])
    d   = np.concatenate([np.zeros(len(pre)), np.ones(len(post))])
    X   = sm.add_constant(d)
    try:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
        return {
            "delta_hac": round(float(res.params[1]), 5),
            "hac_p":     round(float(res.pvalues[1]), 4),
            "hac_H0":    "RIFIUTATA" if res.pvalues[1] < ALPHA else "non rifiutata",
        }
    except Exception:
        return {"delta_hac": np.nan, "hac_p": np.nan, "hac_H0": "errore"}


def bootstrap_ci(pre, post, n_boot=2000, seed=SEED):
    rng    = np.random.default_rng(seed)
    deltas = [rng.choice(post,len(post),replace=True).mean() -
              rng.choice(pre, len(pre), replace=True).mean()
              for _ in range(n_boot)]
    arr = np.array(deltas)
    return float(arr.mean()), float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))


def _stars(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."


def run_tests_at_split(margin_series: pd.Series,
                       split_date: pd.Timestamp,
                       mu_2019: float,
                       soglia: float,
                       rng_perm,
                       pre_override: np.ndarray = None,
                       skip_perm_hac: bool = False) -> dict:
    """
    Esegue la batteria completa (Welch 1-sample, MW, block perm, HAC)
    con la finestra pre/post definita da split_date.

    Parametri aggiuntivi:
      pre_override   — array numpy da usare come 'pre' per MW, perm, HAC al
                       posto della finestra ricavata da split_date. Tipico uso:
                       passare la distribuzione 2019 come baseline esogena.
      skip_perm_hac  — se True, salta block permutation e HAC (usato per
                       split "pre_2019" dove perm/HAC non sono temporalmente
                       validi: 2019 e post-evento non sono adiacenti).
    """
    idx = int(np.clip(margin_series.index.searchsorted(split_date),
                      2, len(margin_series) - 2))
    pre_from_split = margin_series.iloc[:idx].values
    post           = margin_series.iloc[idx:].values
    pre_for_tests  = pre_override if pre_override is not None else pre_from_split

    if len(pre_for_tests) < 4 or len(post) < 4:
        return {"ok": False, "n_pre": len(pre_for_tests), "n_post": len(post)}

    # H0: μ_post = μ_2019   H1: μ_post > μ_2019  (one-sample, one-sided upper)
    t_s, t_p = stats.ttest_1samp(post, popmean=mu_2019, alternative="greater")
    delta_vs_bl = float(post.mean() - mu_2019)
    # delta_local usa sempre pre_from_split (confronto locale intorno all'evento)
    delta_local  = float(post.mean() - pre_from_split.mean())

    # Pre vs baseline
    _, t_p_pre = stats.ttest_1samp(pre_from_split, popmean=mu_2019, alternative="greater")
    delta_pre_bl = float(pre_from_split.mean() - mu_2019)

    mw = mann_whitney_full(pre_for_tests, post)

    if skip_perm_hac:
        obs_perm, p_perm = np.nan, np.nan
        hac = {"delta_hac": np.nan, "hac_p": np.nan}
    else:
        obs_perm, p_perm = perm_test(pre_for_tests, post, rng=rng_perm)
        hac              = hac_test(pre_for_tests, post)

    return {
        "ok":           True,
        "n_pre":        len(pre_for_tests),
        "n_post":       len(post),
        "split_date":   split_date.strftime("%Y-%m-%d"),
        "delta_vs_bl":  round(delta_vs_bl, 5),
        "delta_local":  round(delta_local, 5),
        "delta_pre_bl": round(delta_pre_bl, 5),
        "pre_anomalo":  delta_pre_bl > soglia,
        "anomalo":      delta_vs_bl > soglia,
        "t_stat":       round(float(t_s), 4),
        "t_p":          round(float(t_p), 4),
        "t_p_pre":      round(float(t_p_pre), 4),
        "mw_p_one":     mw["p_one"],
        "mw_cliff":     mw["cliffs_delta"],
        "mw_hl":        mw["hodges_lehmann"],
        "perm_p":       round(p_perm, 4) if not np.isnan(p_perm) else np.nan,
        "perm_delta":   round(obs_perm, 5) if not np.isnan(obs_perm) else np.nan,
        "hac_p":        hac["hac_p"],
        "hac_delta":    hac["delta_hac"],
    }

def margin_changepoint_date(series_vals: np.ndarray,
                            series_idx: pd.DatetimeIndex,
                            min_frac: float = 0.25) -> tuple:
    """
    Stima τ_margin = data del changepoint strutturale sul margine.
    Metodo: argmax del |t di Welch| su tutte le possibili partizioni
    (Bai-Perron single-break, brute force). Restituisce (date, t_stat).
    La ricerca esclude il primo e l'ultimo min_frac della serie per evitare
    changepoint ai bordi con campioni troppo piccoli.
    """
    n = len(series_vals)
    if n < 8:
        return None, np.nan
    best_t, best_k = 0.0, n // 2
    min_n = max(4, int(n * min_frac))
    for k in range(min_n, n - min_n):
        pre_k, post_k = series_vals[:k], series_vals[k:]
        if len(pre_k) < 4 or len(post_k) < 4:
            continue
        t_k, _ = stats.ttest_ind(pre_k, post_k, equal_var=False)
        if abs(t_k) > abs(best_t):
            best_t, best_k = t_k, k
    return series_idx[best_k], best_t


# ─────────────────────────────────────────────────────────────────────────────
# Leggi diagnostici da script 02
# ─────────────────────────────────────────────────────────────────────────────
diag_path = "data/regression_diagnostics.csv"
if os.path.exists(diag_path):
    df_diag = pd.read_csv(diag_path)
    print("Diagnostici OLS letti da script 02:\n")
    for _, r in df_diag.iterrows():
        dw_flag = "AUTOCORR" if float(r["DW"]) < 1.5 else "ok"
        sw_flag = "NON-NORM" if (r["SW_p"] is not None and
                                  not np.isnan(float(r["SW_p"])) and
                                  float(r["SW_p"]) < ALPHA) else "ok"
        bp_flag = "ETEROSC." if (r["BP_p"] is not None and
                                  not np.isnan(float(r["BP_p"])) and
                                  float(r["BP_p"]) < ALPHA) else "ok"
        print(f"  {r['Evento'][:30]:30} | {r['Serie']:7}: "
              f"DW={r['DW']:.2f} [{dw_flag}]  "
              f"SW_p={r['SW_p']} [{sw_flag}]  "
              f"BP_p={r['BP_p']} [{bp_flag}]")
    print()
else:
    print("regression_diagnostics.csv non trovato — eseguire prima 02_changepoint.py\n")

# ─────────────────────────────────────────────────────────────────────────────
# Carica τ_price da script 02 (changepoint sui log-prezzi)
# Mappa: (ev_name, fuel_name) -> Timestamp
# ─────────────────────────────────────────────────────────────────────────────
tau_price_map = {}
t1_path = "data/table1_changepoints.csv"
if os.path.exists(t1_path):
    df_t1 = pd.read_csv(t1_path)
    # Le serie "Benzina" e "Diesel" in table1 corrispondono ai margini
    for _, r in df_t1.iterrows():
        if r["Serie"] in ("Benzina", "Diesel"):
            tau_price_map[(r["Evento"], r["Serie"])] = pd.Timestamp(r["tau"])
    print(f"τ_price caricati da table1_changepoints.csv: {len(tau_price_map)} entry")
else:
    print("table1_changepoints.csv non trovato — τ_price non disponibile")


# ─────────────────────────────────────────────────────────────────────────────
# Carica dataset
# ─────────────────────────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged_with_futures.csv",
                     index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane | "
      f"{merged.index[0].date()} – {merged.index[-1].date()}\n")

# Safety check unità (se ancora in EUR/1000L normalizza qui)
for raw_col, eur_l_col in [("benzina_4w","benzina_eur_l"),("diesel_4w","diesel_eur_l")]:
    if eur_l_col in merged.columns:
        med = merged[eur_l_col].dropna().median()
        if med > 10:
            merged[eur_l_col] = merged[eur_l_col] / 1000.0
            if raw_col in merged.columns:
                merged[raw_col] = merged[eur_l_col]
            print(f"  [safety] Normalizzato {eur_l_col} /1000")
    elif raw_col in merged.columns:
        med = merged[raw_col].dropna().median()
        merged[eur_l_col] = merged[raw_col] / (1000.0 if med > 10 else 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Soglie baseline (2sigma su 2019)
# ─────────────────────────────────────────────────────────────────────────────
baseline   = merged.loc[BASELINE_START:BASELINE_END]
thresholds = {}
baseline_mu = {}   # μ_2019 per fuel — usato nella H0 one-sample
for fuel, col in MARGIN_COLS.items():
    if col in baseline.columns:
        vals = baseline[col].dropna()
        thresholds[fuel]  = float(2 * vals.std()) if len(vals) >= 4 else 0.030
        baseline_mu[fuel] = float(vals.mean())    if len(vals) >= 4 else 0.0
        print(f"Baseline 2019 | {fuel}: mu={baseline_mu[fuel]:.5f}  "
              f"sigma={vals.std():.5f}  soglia 2sigma={thresholds[fuel]:.5f} EUR/L")

# Sensitivity baseline
sens_rows = []
for bl_label, bl_start, bl_end in [
    ("2019_full", "2019-01-01","2019-12-31"),
    ("2021_full", "2021-01-01","2021-12-31"),
]:
    for col in MARGIN_COLS.values():
        if col not in merged.columns: continue
        b = merged.loc[bl_start:bl_end, col].dropna()
        if len(b) < 4: continue
        sens_rows.append({
            "baseline": bl_label, "serie": col, "n_weeks": len(b),
            "mean": round(float(b.mean()),5), "std": round(float(b.std()),5),
            "soglia_2sigma": round(float(2*b.std()),5),
        })
if sens_rows:
    pd.DataFrame(sens_rows).to_csv("data/baseline_sensitivity.csv", index=False)
    print("\nSensitivity baseline (2019 vs 2021):")
    for r in sens_rows:
        print(f"  {r['baseline']:10} | {r['serie']:22}: 2sigma={r['soglia_2sigma']:.5f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Test H0 per ogni evento x carburante
# ─────────────────────────────────────────────────────────────────────────────
results      = []
conf_pvalues = []
rng_perm     = np.random.default_rng(SEED)

for ev_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    prelim = cfg.get("preliminare", False)
    prelim_note = " [PRELIMINARE: dati post-shock limitati]" if prelim else ""

    print("="*65)
    print(f"{ev_name}{prelim_note}")
    print("="*65)

    for fuel, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns:
            print(f"  {fuel}: colonna {margin_col} non trovata — skip")
            continue

        df_ev  = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        shi    = int(np.clip(df_ev.index.searchsorted(shock), 2, len(df_ev)-2))
        pre_m  = df_ev.iloc[:shi][margin_col].dropna().values
        post_m = df_ev.iloc[shi:][margin_col].dropna().values

        if len(pre_m) < 4 or len(post_m) < 4:
            print(f"  {fuel}: campioni troppo piccoli (pre={len(pre_m)}, post={len(post_m)}) — skip")
            continue

        mu_2019  = baseline_mu.get(fuel, 0.0)
        soglia   = thresholds.get(fuel, 0.030)

        # ── H0 (nuova): one-sample t-test post vs μ_2019, one-sided upper ──
        # H0: μ_post = μ_2019   H1: μ_post > μ_2019
        t_stat, t_p = stats.ttest_1samp(post_m, popmean=mu_2019, alternative="greater")
        delta_vs_baseline = float(post_m.mean() - mu_2019)   # δ principale
        delta_mean        = float(post_m.mean() - pre_m.mean())  # salto locale (ausiliario)
        boot_m, b_lo, b_hi = bootstrap_ci(pre_m, post_m)

        # δ finestra pre-shock vs baseline (era già elevato prima dello shock?)
        t_pre, t_p_pre = stats.ttest_1samp(pre_m, popmean=mu_2019, alternative="greater")
        delta_pre_vs_baseline = float(pre_m.mean() - mu_2019)
        pre_anomalo = delta_pre_vs_baseline > soglia

        # ── Anomalia flagging ora su δ vs baseline 2019 ──────────────────
        anomalo = delta_vs_baseline > soglia   # positiva, supera 2σ_2019

        # ── τ_margin: changepoint strutturale sul margine ─────────────────
        df_margin_full = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        tau_margin_date, tau_margin_t = margin_changepoint_date(
            df_margin_full[margin_col].values,
            df_margin_full.index
        )
        tau_price_date = tau_price_map.get((ev_name, fuel), None)

        # ── τ_lag = τ_margin − τ_price ────────────────────────────────────
        # Interpretazione:
        #   lag < −7gg  → ANTICIPATORIO: il margine si rompe PRIMA del wholesale
        #                  Pattern incompatibile con semplice cost pass-through;
        #                  coerente con pricing anticipatorio / speculativo.
        #   −7 ≤ lag ≤ 14 → SINCRONO: rottura contestuale
        #                  I distributori espandono il margine NON appena
        #                  il costo wholesale si muove, ma insieme ad esso.
        #   lag > 14gg  → REATTIVO: il margine si adatta dopo il prezzo
        #                  Coerente con trasmissione graduale dei costi.
        if tau_margin_date is not None and tau_price_date is not None:
            lag_tau_days = int((tau_margin_date - tau_price_date).days)
            if lag_tau_days < -7:
                tau_lag_interp = (
                    f"ANTICIPATORIO (lag={lag_tau_days:+d}gg): τ_margin precede τ_price — "
                    "margine si espande PRIMA dello shock wholesale"
                )
            elif lag_tau_days <= 14:
                tau_lag_interp = (
                    f"SINCRONO (lag={lag_tau_days:+d}gg): τ_margin e τ_price coincidono — "
                    "espansione margine contestuale al movimento di prezzo"
                )
            else:
                tau_lag_interp = (
                    f"REATTIVO (lag={lag_tau_days:+d}gg): τ_margin segue τ_price — "
                    "coerente con trasmissione graduale dei costi (cost pass-through)"
                )
        else:
            lag_tau_days = None
            tau_lag_interp = "N/A (un τ non disponibile)"

        # ── Margine integrato in eccesso sopra μ_2019 (proxy windfall) ───
        # Calcolato sull'intero window evento (pre_start → post_end).
        # Unità: EUR/L × settimane.  Per convertire in €M moltiplicare per
        # volume_settimana_L / 1e6 (volume proxy: MISE 2022).
        _VOLUME_L_WEEK = {"Benzina": 182_000_000, "Diesel": 596_000_000}
        excess_series      = df_margin_full[margin_col] - mu_2019
        integrated_pos     = float(excess_series.clip(lower=0).sum())   # Σ settimane sopra baseline
        integrated_neg     = float((-excess_series).clip(lower=0).sum())# Σ settimane sotto baseline
        integrated_net     = float(excess_series.sum())                  # netto (pos−neg)
        vol_week           = _VOLUME_L_WEEK.get(fuel, 0)
        windfall_meur_pos  = integrated_pos  * vol_week / 1e6
        windfall_meur_net  = integrated_net  * vol_week / 1e6
        n_weeks_above      = int((excess_series > 0).sum())
        n_weeks_below      = int((excess_series < 0).sum())

        # ── Determinazione split per test locali ──────────────────────────
        # Il problema dello split shock_hard per i test locali (MW, perm, HAC):
        # quando τ_price precede shock_hard di 46-73 giorni (caso anticipatorio),
        # la finestra "pre" include già il periodo in cui il margine stava salendo.
        # Questo contamina il gruppo di controllo e sottostima il δ locale.
        #
        # Strategia per test differenti:
        #   MW  → usa la distribuzione 2019 come pre (baseline esogena, nessuna
        #          assunzione di adiacenza temporale). Allinea MW direttamente
        #          con l'H0 del Welch 1-sample. Conservativo e pulito.
        #   perm, HAC → richiedono adiacenza temporale. Due implementazioni:
        #
        #     [PRINCIPALE]   split su τ_price — esogeno al margine (stimato sui
        #          prezzi, nessuna circolarità). Pre window termina prima del
        #          segnale anticipatorio. Fallback su shock_hard se τ_price non
        #          disponibile o posteriore allo shock.
        #
        #     [ROBUSTNESS]   split su τ_margin — usa il changepoint del margine
        #          stesso come confine. Endogeno ma massimizza il segnale nella
        #          finestra post; utile quando τ_margin ≠ τ_price.
        #          TENSIONE IRAN-ISRAELE: τ_margin (21 apr) precede τ_price
        #          (28 apr) di 7gg. Con τ_price come split, i 7gg di rottura
        #          del margine precedenti finiscono nel "post" → effetto trascura-
        #          bile in pratica. Il robustness check con τ_margin isola
        #          esattamente la pre-window libera da questa ambiguità.
        #          UCRAINA: τ_margin = +70gg rispetto a τ_price → la pre window
        #          con τ_price è genuinamente pulita; il robustness dà risultati
        #          attesi più conservativi (pre window più breve).
        #
        #     Entrambi i risultati vengono salvati nel CSV. Il principale (τ_price)
        #     entra nella BH correction; il robustness è riportato come check.
        #
        # NOTA: il Welch 1-sample (test primario BH) rimane invariato — usa
        # shock_hard come split perché confronta solo la finestra post vs μ_2019,
        # indipendentemente da dove si trova il confine pre/post.

        # 2019 baseline come pre per MW
        bl_pre_m = merged.loc[BASELINE_START:BASELINE_END, margin_col].dropna().values

        # ── Split PRINCIPALE per perm/HAC: τ_price ────────────────────────
        if (tau_price_date is not None
                and cfg["pre_start"] < tau_price_date < shock
                and tau_price_date < cfg["post_end"]):
            shi_lc = int(np.clip(df_ev.index.searchsorted(tau_price_date), 2, len(df_ev)-2))
            pre_lc  = df_ev.iloc[:shi_lc][margin_col].dropna().values
            post_lc = df_ev.iloc[shi_lc:][margin_col].dropna().values
            split_lc_lbl = f"tau_price ({tau_price_date.date()})"
        else:
            pre_lc, post_lc = pre_m, post_m
            split_lc_lbl = "shock_hard (fallback τ_price N/A)"

        # Fallback campioni troppo piccoli
        if len(pre_lc) < 4 or len(post_lc) < 4:
            pre_lc, post_lc = pre_m, post_m
            split_lc_lbl = "shock_hard (fallback piccolo campione)"

        # ── Split ROBUSTNESS per perm/HAC: τ_margin ───────────────────────
        # τ_margin è già calcolato sopra (margin_changepoint_date).
        # È endogeno al margine ma cattura esattamente la rottura strutturale;
        # usato solo come check, non entra nella BH correction.
        if (tau_margin_date is not None
                and cfg["pre_start"] < tau_margin_date < cfg["post_end"]):
            shi_tm = int(np.clip(df_ev.index.searchsorted(tau_margin_date), 2, len(df_ev)-2))
            pre_tm  = df_ev.iloc[:shi_tm][margin_col].dropna().values
            post_tm = df_ev.iloc[shi_tm:][margin_col].dropna().values
            split_tm_lbl = f"tau_margin ({tau_margin_date.date()})"
        else:
            pre_tm, post_tm = pre_m, post_m
            split_tm_lbl = "shock_hard (fallback τ_margin N/A)"

        if len(pre_tm) < 4 or len(post_tm) < 4:
            pre_tm, post_tm = pre_m, post_m
            split_tm_lbl = "shock_hard (fallback piccolo campione τ_margin)"

        # ── 2. Mann-Whitney (post vs distribuzione 2019 — baseline esogena) ──
        # Testa se la distribuzione post-shock stochasticamente domina il 2019.
        # Allineato con H0 Welch: entrambi i test ora confrontano il post con il 2019.
        mw = mann_whitney_full(bl_pre_m, post_m)

        # Versione locale shock_hard (mantenuta per confronto nel CSV)
        mw_local = mann_whitney_full(pre_m, post_m)

        # ── 3. Block permutation ──────────────────────────────────────────
        # [3a] Principale: split τ_price (entra nella BH correction)
        obs_perm, p_perm = perm_test(pre_lc, post_lc, rng=rng_perm)
        # [3b] Robustness: split τ_margin (check metodologico, fuori dalla BH)
        obs_perm_tm, p_perm_tm = perm_test(pre_tm, post_tm, rng=rng_perm)

        # ── 4. HAC ───────────────────────────────────────────────────────
        # [4a] Principale: split τ_price (entra nella BH correction)
        hac = hac_test(pre_lc, post_lc)
        # [4b] Robustness: split τ_margin (check metodologico, fuori dalla BH)
        hac_tm = hac_test(pre_tm, post_tm)

        stat_sig = t_p < ALPHA
        if stat_sig and anomalo and delta_mean > 0:
            clas = "Margine anomalo positivo"
        elif stat_sig and anomalo and delta_mean < 0:
            clas = "Compressione margine"
        elif not anomalo:
            clas = "Neutro / trasmissione attesa"
        elif stat_sig and not anomalo:
            clas = "Variazione statistica"
        else:
            clas = "Inconclusivo"

        divergence = ""
        if (t_p < ALPHA) != (mw["p_one"] < ALPHA):
            divergence = (
                "DIVERGENZA t vs MW: t reagisce a media/outlier; "
                "MW al rango mediano. Vedi HAC e block perm per arbitraggio."
            )

        print(f"\n  {fuel}")
        print(f"    n pre={len(pre_m)}  n post={len(post_m)}"
              + (" [DATI LIMITATI]" if prelim else ""))
        print(f"    μ_2019={mu_2019:+.5f}  soglia 2σ={soglia:.5f}")
        print(f"    δ_post_vs_baseline = {delta_vs_baseline:+.5f} EUR/L  "
              f"(anomalo: {anomalo})")
        print(f"    δ_pre_vs_baseline  = {delta_pre_vs_baseline:+.5f} EUR/L  "
              f"(pre già elevato: {pre_anomalo})")
        print(f"    δ_locale (post−pre) = {delta_mean:+.5f} EUR/L  "
              f"[boot CI: {b_lo:+.5f}, {b_hi:+.5f}]")
        print(f"    Welch t (1-sample vs μ_2019, one-sided): "
              f"t={t_stat:.3f}  p={t_p:.4f} {_stars(t_p)}")
        print(f"    Welch t pre vs baseline: t={t_pre:.3f}  p={t_p_pre:.4f} {_stars(t_p_pre)}")
        print(f"    Mann-Whitney (post vs 2019 baseline, n_2019={len(bl_pre_m)}): "
              f"AUC={mw['AUC']:.3f}  "
              f"p(one)={mw['p_one']:.4f} {_stars(mw['p_one'])}  "
              f"HL={mw['hodges_lehmann']:+.5f}  Cliff={mw['cliffs_delta']:+.3f} "
              f"[{mw['magnitude']}]")
        print(f"    MW_local (post vs pre shock_hard, per confronto): "
              f"p(one)={mw_local['p_one']:.4f} {_stars(mw_local['p_one'])}")
        print(f"    Block perm [PRINCIPALE — split={split_lc_lbl}]: "
              f"delta_med={obs_perm:+.5f}  p={p_perm:.4f} {_stars(p_perm)}")
        p_perm_tm_str = f"{p_perm_tm:.4f} {_stars(p_perm_tm)}" if not np.isnan(p_perm_tm) else "N/A"
        print(f"    Block perm [ROBUSTNESS — split={split_tm_lbl}]: "
              f"delta_med={obs_perm_tm:+.5f}  p={p_perm_tm_str}")
        print(f"    HAC        [PRINCIPALE — split={split_lc_lbl}]: "
              f"delta={hac['delta_hac']:+.5f}  "
              f"p={hac['hac_p']} {_stars(hac['hac_p']) if not np.isnan(hac['hac_p']) else ''}")
        hac_tm_p_str = (f"{hac_tm['hac_p']} {_stars(hac_tm['hac_p'])}"
                        if not np.isnan(hac_tm["hac_p"]) else "N/A")
        print(f"    HAC        [ROBUSTNESS — split={split_tm_lbl}]: "
              f"delta={hac_tm['delta_hac']:+.5f}  p={hac_tm_p_str}")
        tp_str  = tau_price_date.strftime("%Y-%m-%d") if tau_price_date else "N/A"
        tm_str  = tau_margin_date.strftime("%Y-%m-%d") if tau_margin_date is not None else "N/A"
        print(f"    τ_price  = {tp_str}  |  τ_margin = {tm_str}  "
              f"(t={tau_margin_t:+.2f})")
        print(f"    τ_lag: {tau_lag_interp}")
        print(f"    Windfall window {cfg['pre_start'].date()}→{cfg['post_end'].date()}: "
              f"settimane sopra baseline={n_weeks_above}/{len(df_margin_full)}  "
              f"integrato_pos={integrated_pos:.3f} EUR/L·sett "
              f"(≈ {windfall_meur_pos:.0f} M€ lordi, volume proxy MISE 2022)")
        print(f"    Windfall netto (pos−neg) = {integrated_net:+.3f} EUR/L·sett "
              f"(≈ {windfall_meur_net:+.0f} M€)")
        print(f"    Classificazione: {clas}")
        if prelim:
            print(f"    NOTA: risultato PRELIMINARE — solo {len(post_m)} settimane post-shock")
        if divergence:
            print(f"    {divergence}")

        row = {
            "Evento":                  ev_name,
            "Carburante":              fuel,
            "preliminare":             prelim,
            "n_pre":                   len(pre_m),
            "n_post":                  len(post_m),
            # ── H0 nuova: post vs baseline 2019 ──────────────────────────
            "mu_baseline_2019":        round(mu_2019, 5),
            "soglia_2sigma":           round(soglia, 5),
            "delta_vs_baseline":       round(delta_vs_baseline, 5),
            "delta_anomalo_vs_bl":     anomalo,
            # ── Pre-shock vs baseline ─────────────────────────────────────
            "delta_pre_vs_baseline":   round(delta_pre_vs_baseline, 5),
            "pre_anomalo":             pre_anomalo,
            "t_pre_p":                 round(float(t_p_pre), 4),
            # ── Salto locale post−pre (ausiliario) ────────────────────────
            "delta_mean_eur":          round(delta_mean, 5),
            "boot_CI_lo":              round(b_lo, 5),
            "boot_CI_hi":              round(b_hi, 5),
            # ── τ changepoint ─────────────────────────────────────────────
            "tau_price":               tp_str,
            "tau_margin":              tm_str,
            "tau_margin_t_stat":       round(float(tau_margin_t), 3),
            "lag_tau_days":            lag_tau_days,
            "tau_lag_interpretation":  tau_lag_interp,
            # ── Windfall proxy ────────────────────────────────────────────
            "n_weeks_event_window":    len(df_margin_full),
            "n_weeks_above_baseline":  n_weeks_above,
            "n_weeks_below_baseline":  n_weeks_below,
            "integrated_excess_EurL_weeks": round(integrated_pos, 4),
            "integrated_net_EurL_weeks":    round(integrated_net, 4),
            "windfall_gross_meur":     round(windfall_meur_pos, 1),
            "windfall_net_meur":       round(windfall_meur_net, 1),
            # ── Test primario (one-sample vs μ_2019) ─────────────────────
            "t_stat":                  round(float(t_stat), 4),
            "t_p":                     round(float(t_p), 4),
            "t_H0":                    "RIFIUTATA" if stat_sig else "non rifiutata",
            # ── MW: ora vs 2019 baseline (primario) ──────────────────────
            **{f"mw_{k}": v for k, v in mw.items()},
            "mw_n_pre_2019":           len(bl_pre_m),
            # ── MW_local: vs shock_hard pre (confronto) ──────────────────
            "mw_local_p_one":          mw_local["p_one"],
            "mw_local_cliff":          mw_local["cliffs_delta"],
            # ── Perm/HAC [PRINCIPALE]: split τ_price — entra nella BH ──────
            "split_main_type":         split_lc_lbl,
            "n_pre_main":              len(pre_lc),
            "n_post_main":             len(post_lc),
            "perm_delta_med":          round(obs_perm, 5) if not np.isnan(obs_perm) else np.nan,
            "perm_p":                  round(p_perm, 4) if not np.isnan(p_perm) else np.nan,
            "perm_H0":                 ("RIFIUTATA" if (not np.isnan(p_perm) and p_perm < ALPHA)
                                        else "non rifiutata"),
            **{f"hac_{k}": v for k, v in hac.items()},
            # ── Perm/HAC [ROBUSTNESS]: split τ_margin — fuori dalla BH ─────
            # Controlla che le conclusioni del principale reggano quando il
            # confine pre/post coincide esattamente con la rottura del margine.
            # Convergenza: stesso segno di rigetto → risultato robusto alla
            # scelta dello split. Divergenza: documentare nella tabella.
            "split_robustness_type":   split_tm_lbl,
            "n_pre_robustness":        len(pre_tm),
            "n_post_robustness":       len(post_tm),
            "perm_rob_delta_med":      round(obs_perm_tm, 5) if not np.isnan(obs_perm_tm) else np.nan,
            "perm_rob_p":              round(p_perm_tm, 4) if not np.isnan(p_perm_tm) else np.nan,
            "perm_rob_H0":             ("RIFIUTATA" if (not np.isnan(p_perm_tm) and p_perm_tm < ALPHA)
                                        else "non rifiutata"),
            **{f"hac_rob_{k}": v for k, v in hac_tm.items()},
            # ── Convergenza principale vs robustness ──────────────────────
            # True = entrambi gli split concordano sul rigetto / non-rigetto di H0
            "perm_split_convergence":  (
                (not np.isnan(p_perm) and not np.isnan(p_perm_tm)) and
                ((p_perm < ALPHA) == (p_perm_tm < ALPHA))
            ),
            "hac_split_convergence":   (
                (not np.isnan(hac["hac_p"]) and not np.isnan(hac_tm["hac_p"])) and
                ((hac["hac_p"] < ALPHA) == (hac_tm["hac_p"] < ALPHA))
            ),
            "classificazione":         clas,
            "divergenza_t_mw":         divergence,
        }
        results.append(row)

        # Solo eventi non preliminari entrano nella BH correction.
        # Perm e HAC: entra il PRINCIPALE (split τ_price), non il robustness.
        # Logica: τ_price è esogeno al margine → nessuna circolarità; è lo
        # split metodologicamente difendibile per il test primario.
        if not prelim:
            perm_conv = (
                (not np.isnan(p_perm) and not np.isnan(p_perm_tm)) and
                ((p_perm < ALPHA) == (p_perm_tm < ALPHA))
            )
            hac_conv = (
                (not np.isnan(hac["hac_p"]) and not np.isnan(hac_tm["hac_p"])) and
                ((hac["hac_p"] < ALPHA) == (hac_tm["hac_p"] < ALPHA))
            )
            for fonte, p_val, note in [
                (f"Welch_t_{ev_name}_{fuel}",    float(t_p),            ""),
                (f"MannWhitney_{ev_name}_{fuel}", float(mw["p_one"]),    "vs 2019"),
                (f"BlockPerm_{ev_name}_{fuel}",   float(p_perm)
                 if not np.isnan(p_perm) else None,
                 f"split={split_lc_lbl}; rob_p={p_perm_tm:.4f} conv={'✓' if perm_conv else '✗'}"),
                (f"HAC_{ev_name}_{fuel}",         float(hac["hac_p"])
                 if not np.isnan(hac["hac_p"]) else None,
                 f"split={split_lc_lbl}; rob_p={hac_tm['hac_p']} conv={'✓' if hac_conv else '✗'}"),
            ]:
                if p_val is not None and not np.isnan(p_val):
                    conf_pvalues.append({
                        "fonte": fonte, "tipo": "confirmatory",
                        "descrizione": f"{ev_name} | {fuel}",
                        "p_value": p_val,
                        "note_split": note,
                    })
        else:
            print(f"    [skip BH] Hormuz escluso dalla BH correction (dati preliminari)")


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI MULTI-SPLIT
# Per ogni evento × carburante rieseguiamo la batteria di test usando tre
# punti di split alternativi:
#   shock_hard  — data geopolitica hardcodata (analisi primaria sopra)
#   tau_price   — changepoint sul log-prezzo (da script 02 / table1)
#   tau_margin  — changepoint strutturale sul margine (calcolato qui sopra)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ANALISI MULTI-SPLIT (shock_hard / tau_price / tau_margin)")
print("="*65)

multi_split_results = []
rng_ms = np.random.default_rng(SEED + 1)   # seed separato per riproducibilità

for ev_name, cfg in EVENTS.items():
    prelim = cfg.get("preliminare", False)
    prelim_note = " [PREL.]" if prelim else ""

    for fuel, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns:
            continue

        # Serie di margine per questo evento
        df_ev_full = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        if len(df_ev_full) < 8:
            continue

        margin_series = df_ev_full[margin_col]
        mu_2019  = baseline_mu.get(fuel, 0.0)
        soglia   = thresholds.get(fuel, 0.030)

        # Recupera tau_price e tau_margin dai risultati già calcolati
        prim_row = next((r for r in results
                         if r["Evento"] == ev_name and r["Carburante"] == fuel), None)
        tau_price_date  = tau_price_map.get((ev_name, fuel))
        tau_margin_date = (pd.Timestamp(prim_row["tau_margin"])
                           if prim_row and prim_row["tau_margin"] not in ("N/A", None, "")
                           else None)

        # baseline 2019 per MW nel multi-split
        bl_pre_ms = merged.loc[BASELINE_START:BASELINE_END, margin_col].dropna().values

        splits = {
            "shock_hard":  cfg["shock"],
            "tau_price":   tau_price_date,
            "tau_margin":  tau_margin_date,
            "pre_2019":    cfg["shock"],   # usa shock_hard per definire post; pre = 2019 baseline
        }

        print(f"\n  {ev_name}{prelim_note} — {fuel}")
        print(f"  {'Split':12}  {'split_date':12}  {'δ_bl':>8}  "
              f"{'t_p':>7}  {'mw_p':>7}  {'perm_p':>7}  {'hac_p':>7}  {'anomalo'}")
        print("  " + "-"*78)

        for split_label, split_date in splits.items():
            if split_date is None:
                print(f"  {split_label:12}  {'N/A':12}  — split non disponibile")
                continue

            # pre_2019: usa 2019 come pre per MW, salta perm/HAC (non temporalmente adiacenti)
            if split_label == "pre_2019":
                res = run_tests_at_split(margin_series, split_date,
                                         mu_2019, soglia, rng_ms,
                                         pre_override=bl_pre_ms,
                                         skip_perm_hac=True)
                split_date_str = f"{str(split_date.date())} (post); pre=2019"
            else:
                res = run_tests_at_split(margin_series, split_date,
                                         mu_2019, soglia, rng_ms)
                split_date_str = str(split_date.date())

            if not res["ok"]:
                print(f"  {split_label:12}  {split_date_str:25}  "
                      f"campione troppo piccolo (pre={res['n_pre']}, post={res['n_post']})")
                continue

            # Consensus: conteggio flessibile (perm/HAC NaN con pre_2019)
            perm_rej = (not np.isnan(res["perm_p"])) and res["perm_p"] < ALPHA
            hac_rej  = (not np.isnan(res["hac_p"])) and res["hac_p"] < ALPHA
            n_rej = sum([
                res["t_p"]      < ALPHA,
                res["mw_p_one"] < ALPHA,
                perm_rej,
                hac_rej,
            ])
            # Denominatore: per pre_2019 solo 2 test validi (Welch + MW)
            n_valid = 2 if split_label == "pre_2019" else 4
            consensus = "✓" if n_rej >= max(1, n_valid // 2) else "–"

            perm_str = f"{res['perm_p']:7.4f}" if not np.isnan(res["perm_p"]) else "    N/A"
            hac_str  = f"{res['hac_p']:7.4f}"  if not np.isnan(res["hac_p"])  else "    N/A"

            print(f"  {split_label:12}  {split_date_str:25}  "
                  f"{res['delta_vs_bl']:+8.4f}  "
                  f"{res['t_p']:7.4f}  "
                  f"{res['mw_p_one']:7.4f}  "
                  f"{perm_str}  {hac_str}  "
                  f"{'SÌ' if res['anomalo'] else 'no':4} {_stars(res['t_p'])} "
                  f"[{n_rej}/{n_valid} {consensus}]")

            multi_split_results.append({
                "Evento":        ev_name,
                "Carburante":    fuel,
                "preliminare":   prelim,
                "split_type":    split_label,
                **{k: v for k, v in res.items() if k != "ok"},
                "n_tests_valid": n_valid,
                "n_tests_reject": n_rej,
                "consensus_2of4": n_rej >= max(1, n_valid // 2),
            })

df_ms = pd.DataFrame(multi_split_results)
if not df_ms.empty:
    df_ms.to_csv("data/table2_multi_split.csv", index=False)
    print(f"\nSalvato: data/table2_multi_split.csv ({len(df_ms)} righe)")


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI ANNUALE — "alla fine dell'anno hanno guadagnato uguale?"
# Per ogni anno solare 2019–2026 confrontiamo la distribuzione del margine
# lordo con il 2019 baseline usando Mann-Whitney (non parametrico) e
# calcoliamo il margine integrato annuale in eccesso (proxy windfall annuo).
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ANALISI ANNUALE DEI MARGINI (confronto vs baseline 2019)")
print("="*65)

annual_rows = []
_VOLUME_L_WEEK = {"Benzina": 182_000_000, "Diesel": 596_000_000}

for fuel, margin_col in MARGIN_COLS.items():
    if margin_col not in merged.columns:
        continue

    mu_2019 = baseline_mu.get(fuel, 0.0)
    soglia  = thresholds.get(fuel, 0.030)
    vol_week = _VOLUME_L_WEEK.get(fuel, 0)

    # Baseline 2019: distribuzione di riferimento
    bl_vals = merged.loc[BASELINE_START:BASELINE_END, margin_col].dropna().values

    print(f"\n  {fuel}  |  μ_2019 = {mu_2019:.5f} EUR/L  soglia 2σ = {soglia:.5f}")
    print(f"  {'Anno':6}  {'n':>4}  {'media':>8}  {'mediana':>8}  "
          f"{'δ_vs_2019':>10}  {'MW_p':>8}  {'anomalo':>8}  "
          f"{'sett>bl':>7}  {'windfall_netto_meur':>20}")
    print("  " + "-"*95)

    years = sorted(merged.index.year.unique())
    for yr in years:
        yr_data = merged.loc[str(yr), margin_col].dropna()
        if len(yr_data) < 4:
            continue
        yr_vals = yr_data.values

        # Mann-Whitney vs baseline 2019 (non parametrico)
        _, mw_p = mannwhitneyu(yr_vals, bl_vals, alternative="greater")

        mean_yr   = float(yr_vals.mean())
        med_yr    = float(np.median(yr_vals))
        delta_yr  = mean_yr - mu_2019
        anomalo_yr = delta_yr > soglia

        # Windfall integrato annuo
        excess_yr    = yr_data - mu_2019
        integrated_n = float(excess_yr.sum())
        windfall_n   = integrated_n * vol_week / 1e6
        n_above      = int((excess_yr > 0).sum())

        print(f"  {yr:<6}  {len(yr_vals):>4}  {mean_yr:>8.5f}  {med_yr:>8.5f}  "
              f"{delta_yr:>+10.5f}  {mw_p:>8.4f}  "
              f"{'SÌ' if anomalo_yr else 'no':>8}  "
              f"{n_above:>4}/{len(yr_vals):<3}  "
              f"{windfall_n:>+20.1f}")

        annual_rows.append({
            "anno":             yr,
            "carburante":       fuel,
            "n_settimane":      len(yr_vals),
            "media_eur_l":      round(mean_yr, 5),
            "mediana_eur_l":    round(med_yr, 5),
            "delta_vs_2019":    round(delta_yr, 5),
            "anomalo_2sigma":   anomalo_yr,
            "mw_p_vs_2019":     round(float(mw_p), 4),
            "n_settimane_sopra_baseline": n_above,
            "integrated_excess_EurL_weeks": round(float(excess_yr.clip(lower=0).sum()), 4),
            "integrated_net_EurL_weeks":    round(integrated_n, 4),
            "windfall_net_meur":            round(windfall_n, 1),
            "volume_proxy_source":          "MISE 2022 (~9.5 Mld L/anno benz, ~31 Mld L/anno diesel)",
        })

df_annual = pd.DataFrame(annual_rows)
if not df_annual.empty:
    df_annual.to_csv("data/annual_margin_analysis.csv", index=False)
    print(f"\n  Salvato: data/annual_margin_analysis.csv")

    # Nota interpretativa
    print("""
  Nota: "windfall_net_meur" è la somma algebrica di tutte le settimane
  dell'anno (positive e negative) moltiplicata per il volume proxy.
  Un valore elevato positivo significa che la media annua di margine
  è rimasta sopra il 2019 anche pesando i periodi di compressione.
  Questo risponde a "hanno guadagnato uguale alla fine dell'anno?":
  se windfall_net > 0 per un anno di crisi, la risposta è NO —
  hanno guadagnato PIÙ del normale anche su base annua.
  Nota: volume proxy approssimativo; per stime precise usare dati MISE/ENAC.
""")



df_res = pd.DataFrame(results)

if not df_res.empty:
    # BH solo sui non-preliminari
    df_nonprel = df_res[~df_res["preliminare"]]
    if not df_nonprel.empty:
        bh_rej, bh_adj = bh_correction(df_nonprel["t_p"].values, alpha=ALPHA)
        df_res.loc[df_nonprel.index, "BH_reject_local"] = bh_rej
        df_res.loc[df_nonprel.index, "t_p_BH_adj"]     = bh_adj
        df_res["BH_reject_local"] = df_res["BH_reject_local"].fillna(False)
        df_res["t_p_BH_adj"]      = df_res["t_p_BH_adj"].fillna(np.nan)

    def _reclassify(row):
        if pd.isna(row.get("BH_reject_local")) or not row["BH_reject_local"]:
            return "Neutro / trasmissione attesa" if not row["delta_anomalo_vs_bl"] \
                   else "Variazione statistica"
        return row["classificazione"]
    df_res["classificazione_BH"] = df_res.apply(_reclassify, axis=1)

    df_res.to_csv("data/table2_margin_anomaly.csv", index=False)
    print(f"\nSalvato: data/table2_margin_anomaly.csv ({len(df_res)} righe)")
    n_rej = int(df_res["BH_reject_local"].sum())
    print(f"BH locale (su Welch t non-preliminari): "
          f"{n_rej}/{len(df_nonprel)} test rigettati a FDR 5%")

pd.DataFrame(conf_pvalues).to_csv("data/confirmatory_pvalues.csv", index=False)
print(f"Salvato: data/confirmatory_pvalues.csv ({len(conf_pvalues)} test)")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: margini nel tempo con banda baseline + τ_price e τ_margin
# ─────────────────────────────────────────────────────────────────────────────
# Costruiamo lookup: (ev_name, fuel) -> (tau_price, tau_margin)
tau_lookup = {}
for r in results:
    tp = pd.Timestamp(r["tau_price"])  if r["tau_price"]  != "N/A" else None
    tm = pd.Timestamp(r["tau_margin"]) if r["tau_margin"] != "N/A" else None
    tau_lookup[(r["Evento"], r["Carburante"])] = (tp, tm)

# Colori degli eventi per le linee di shock (già definito in WAR_DATES)
EV_COLOR = {ev: clr for ev, (_, clr) in WAR_DATES.items()}
EV_NAME_SHORT = {
    "Ucraina (Feb 2022)":    "Ucraina",
    "Iran-Israele (Giu 2025)": "Iran",
    "Hormuz (Feb 2026)":     "Hormuz",
}

def _war_lines(ax, y_top):
    for label, (dt, color) in WAR_DATES.items():
        if merged.index[0] <= dt <= merged.index[-1]:
            ax.axvline(dt, color=color, lw=1.8, ls="--", alpha=0.85)
            ax.text(dt + pd.Timedelta(days=5), y_top*0.96,
                    label, rotation=90, fontsize=8, color=color, va="top")

fig_m, axes_m = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
baseline_data  = merged.loc[BASELINE_START:BASELINE_END]

# τ linestyle per tipo
LS_PRICE  = "-."   # τ_price (log-prezzo, da script 02)
LS_MARGIN = ":"    # τ_margin (margine, calcolato qui)
LW_TAU    = 1.6

for ax, (fuel, col), color in zip(
    axes_m,
    [("Benzina","margine_benz_crack"),("Diesel","margine_dies_crack")],
    ["#e67e22","#8e44ad"]
):
    if col not in merged.columns: continue
    s  = merged[col].dropna()
    bl = baseline_data[col].dropna()
    ax.plot(s.index, s.values, color=color, lw=1.8, label=fuel)
    if len(bl) >= 4:
        ax.axhspan(bl.mean()-2*bl.std(), bl.mean()+2*bl.std(),
                   alpha=0.12, color="#888", label="Baseline ±2σ (2019)")
        ax.axhline(bl.mean(), color="#888", lw=1.0, ls="--")
        ax.text(merged.index[5], bl.mean(),
                f"μ₂₀₁₉: {bl.mean():.3f} EUR/L", fontsize=8, color="#555")

    _war_lines(ax, s.max())

    # ── τ_price e τ_margin per ogni evento ──────────────────────────────
    for ev_name_full, cfg in EVENTS.items():
        ev_short = EV_NAME_SHORT.get(ev_name_full, ev_name_full[:6])
        ev_color = [c for lbl, (_, c) in WAR_DATES.items()
                    if ev_name_full.startswith(lbl)][0] if any(
                    ev_name_full.startswith(lbl) for lbl in WAR_DATES) else "#555"
        # recupera colore dall'evento
        for lbl, (_, c) in WAR_DATES.items():
            if ev_name_full.startswith(lbl):
                ev_color = c
                break

        tp, tm = tau_lookup.get((ev_name_full, fuel), (None, None))

        if tp is not None and merged.index[0] <= tp <= merged.index[-1]:
            ax.axvline(tp, color=ev_color, lw=LW_TAU, ls=LS_PRICE, alpha=0.80)
            ax.text(tp - pd.Timedelta(days=12), s.quantile(0.92),
                    f"τ_p\n{ev_short}", rotation=90, fontsize=7,
                    color=ev_color, va="top", alpha=0.85)

        if tm is not None and merged.index[0] <= tm <= merged.index[-1]:
            ax.axvline(tm, color=ev_color, lw=LW_TAU, ls=LS_MARGIN, alpha=0.80)
            ax.text(tm + pd.Timedelta(days=4), s.quantile(0.85),
                    f"τ_m\n{ev_short}", rotation=90, fontsize=7,
                    color=ev_color, va="top", alpha=0.85)

    ax.set_ylabel("Margine lordo (EUR/litro)", fontsize=10)
    ax.set_title(f"Margine crack spread — {fuel}", fontsize=11, fontweight="bold")

    # Legenda manuale compatta
    from matplotlib.lines import Line2D
    legend_handles = [
        mpatches.Patch(color=color, label=fuel, alpha=0.8),
        mpatches.Patch(color="#888", label="Baseline ±2σ (2019)", alpha=0.4),
        Line2D([0],[0], color="#555", lw=LW_TAU, ls=LS_PRICE,  label="τ_price (da script 02)"),
        Line2D([0],[0], color="#555", lw=LW_TAU, ls=LS_MARGIN, label="τ_margin (strutturale)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

axes_m[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes_m[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=40, fontsize=9)
fig_m.suptitle(
    "Margine lordo crack spread — H₀: μ_post = μ₂₀₁₉\n"
    "τ_price (-.): changepoint log-prezzi (script 02)  |  "
    "τ_margin (:): changepoint strutturale sul margine",
    fontsize=10, fontweight="bold"
)
plt.tight_layout(rect=[0,0,1,0.96])
fig_m.savefig("plots/03_margins.png", dpi=DPI, bbox_inches="tight")
plt.close(fig_m)
print("Salvato: plots/03_margins.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: delta summary (solo eventi non-preliminari per il grafico principale)
# ─────────────────────────────────────────────────────────────────────────────
if not df_res.empty:
    clas_col = "classificazione_BH" if "classificazione_BH" in df_res.columns \
               else "classificazione"

    # Plot tutti gli eventi (incluso Hormuz con marcatura visiva)
    fig_s, ax_s = plt.subplots(figsize=(14, max(5, len(df_res)*0.85)))
    labels  = [f"{r['Evento'].split('(')[0].strip()}\n{r['Carburante']}"
               + (" ⚠ preliminare" if r['preliminare'] else "")
               for _, r in df_res.iterrows()]
    deltas  = df_res["delta_mean_eur"].values
    ci_lo   = df_res["boot_CI_lo"].values
    ci_hi   = df_res["boot_CI_hi"].values
    colors  = [CLAS_COLOR.get(c,"#555") for c in df_res[clas_col]]

    ax_s.barh(range(len(df_res)), deltas, color=colors,
              alpha=0.78, edgecolor="black", lw=0.7)
    for i in range(len(df_res)):
        ax_s.errorbar(deltas[i], i,
                      xerr=[[deltas[i]-ci_lo[i]],[ci_hi[i]-deltas[i]]],
                      fmt="none", color="black", capsize=5, lw=1.8)
        lbl = df_res.iloc[i][clas_col][:30]
        if df_res.iloc[i]["preliminare"]:
            lbl += " [prelim.]"
        ax_s.text(max(ci_hi[i], deltas[i])+0.003, i, lbl, va="center", fontsize=8)
    ax_s.axvline(0, color="black", lw=0.8)
    ax_s.set_yticks(range(len(df_res))); ax_s.set_yticklabels(labels, fontsize=9)
    ax_s.set_xlabel("Delta margine lordo post-shock (EUR/litro)", fontsize=11)
    ax_s.set_title("Variazione margine lordo — classificazione con BH FDR 5%\n"
                   "(⚠ Hormuz = dati preliminari, <8 settimane post-shock)",
                   fontsize=12, fontweight="bold")
    ax_s.legend(handles=[mpatches.Patch(color=c,label=k)
                         for k,c in CLAS_COLOR.items()],
                fontsize=8, loc="lower right")
    ax_s.grid(alpha=0.3, axis="x")
    plt.tight_layout(pad=1.5)
    fig_s.savefig("plots/03_delta_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_s)
    print("Salvato: plots/03_delta_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: confronto tre split per ogni evento × carburante
# ─────────────────────────────────────────────────────────────────────────────
if not df_ms.empty:
    SPLIT_COLORS = {
        "shock_hard":  "#2c3e50",   # grigio scuro
        "tau_price":   "#2980b9",   # blu
        "tau_margin":  "#e67e22",   # arancione
    }
    SPLIT_LABELS = {
        "shock_hard": "Data shock\n(hardcoded)",
        "tau_price":  "τ_price\n(log-prezzo)",
        "tau_margin": "τ_margin\n(margine)",
    }

    # Raggruppa per evento × carburante
    groups = df_ms.groupby(["Evento", "Carburante"])
    n_groups = len(groups)
    fig_sp, axes_sp = plt.subplots(
        n_groups, 1,
        figsize=(14, max(5, n_groups * 2.8)),
        squeeze=False
    )

    for ax_idx, ((ev, fuel), grp) in enumerate(groups):
        ax = axes_sp[ax_idx, 0]
        prelim = grp["preliminare"].iloc[0]

        splits_present = grp["split_type"].unique()
        x_pos = np.arange(len(splits_present))
        bar_w = 0.55

        for xi, spl in enumerate(splits_present):
            row = grp[grp["split_type"] == spl].iloc[0]
            color = SPLIT_COLORS.get(spl, "#888")
            delta = row["delta_vs_bl"]
            ax.bar(xi, delta, width=bar_w, color=color, alpha=0.80,
                   edgecolor="black", lw=0.8,
                   label=SPLIT_LABELS.get(spl, spl))

            # Asterischi sopra le barre
            stars = _stars(row["t_p"])
            ax.text(xi, delta + (0.002 if delta >= 0 else -0.004),
                    stars, ha="center", va="bottom" if delta >= 0 else "top",
                    fontsize=10, fontweight="bold")

            # Box info sotto barra: n_pre/post e consensus
            info = f"pre={row['n_pre']} post={row['n_post']}\n{int(row['n_tests_reject'])}/4 test"
            ax.text(xi, min(delta, 0) - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.04 if ax.get_ylim()[1] != ax.get_ylim()[0] else -0.004,
                    info, ha="center", va="top", fontsize=7, color="#555")

        ax.axhline(0, color="black", lw=0.8)
        # banda ±2σ baseline
        soglia_f = thresholds.get(fuel, 0.030)
        ax.axhspan(-soglia_f, soglia_f, alpha=0.08, color="#888",
                   label="±2σ baseline 2019")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [SPLIT_LABELS.get(s, s) for s in splits_present],
            fontsize=9
        )
        ax.set_ylabel("δ vs μ₂₀₁₉ (EUR/L)", fontsize=9)
        title = f"{ev.split('(')[0].strip()} — {fuel}"
        if prelim:
            title += "  ⚠ PRELIMINARE"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7, loc="upper right")

    fig_sp.suptitle(
        "Confronto pre/post su tre split point — H₀: μ_post = μ₂₀₁₉\n"
        "★ = p<0.05  ★★ = p<0.01  ★★★ = p<0.001  (Welch 1-sample, one-sided)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_sp.savefig("plots/03_split_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_sp)
    print("Salvato: plots/03_split_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Analisi annuale — media margine per anno + windfall netto
# ─────────────────────────────────────────────────────────────────────────────
if not df_annual.empty:
    fuels_plot = df_annual["carburante"].unique()
    fig_ann, axes_ann = plt.subplots(
        len(fuels_plot), 2,
        figsize=(15, 4 * len(fuels_plot)),
        squeeze=False
    )
    FUEL_COLOR = {"Benzina": "#e67e22", "Diesel": "#8e44ad"}

    for fi, fuel_p in enumerate(fuels_plot):
        sub = df_annual[df_annual["carburante"] == fuel_p].sort_values("anno")
        mu_bl = baseline_mu.get(fuel_p, 0.0)
        soglia_p = thresholds.get(fuel_p, 0.030)
        fc = FUEL_COLOR.get(fuel_p, "#555")

        # Subplot sinistra: media annua vs baseline
        ax_l = axes_ann[fi, 0]
        years_a = sub["anno"].values
        means_a = sub["media_eur_l"].values
        colors_bar = [
            "#c0392b" if d > soglia_p else "#27ae60" if d < -soglia_p else "#888"
            for d in sub["delta_vs_2019"].values
        ]
        ax_l.bar(years_a, means_a, color=colors_bar, alpha=0.80,
                 edgecolor="black", lw=0.7, width=0.65)
        ax_l.axhline(mu_bl, color="#2c3e50", lw=2.0, ls="--",
                     label=f"μ₂₀₁₉ = {mu_bl:.4f}")
        ax_l.axhspan(mu_bl - soglia_p, mu_bl + soglia_p,
                     alpha=0.08, color="#888", label="±2σ baseline 2019")

        # Asterischi MW
        for yr_i, (yr_v, mw_pv, mean_v) in enumerate(
                zip(sub["anno"], sub["mw_p_vs_2019"], sub["media_eur_l"])):
            stars = _stars(float(mw_pv))
            if stars != "n.s.":
                ax_l.text(yr_v, mean_v + 0.002, stars,
                          ha="center", fontsize=9, fontweight="bold", color="#c0392b")

        ax_l.set_ylabel("Margine lordo medio (EUR/L)", fontsize=9)
        ax_l.set_title(f"{fuel_p} — Media annua vs μ₂₀₁₉ (★ = MW p<0.05)", fontsize=10)
        ax_l.legend(fontsize=8)
        ax_l.grid(alpha=0.25, axis="y")
        ax_l.set_xticks(years_a)
        ax_l.tick_params(axis="x", labelsize=9)

        # Subplot destra: windfall netto annuo in M€
        ax_r = axes_ann[fi, 1]
        wf_vals = sub["windfall_net_meur"].values
        wf_colors = ["#c0392b" if v > 0 else "#2980b9" for v in wf_vals]
        ax_r.bar(years_a, wf_vals, color=wf_colors, alpha=0.80,
                 edgecolor="black", lw=0.7, width=0.65)
        ax_r.axhline(0, color="black", lw=0.8)
        ax_r.set_ylabel("Windfall netto annuo (M€, proxy)", fontsize=9)
        ax_r.set_title(
            f"{fuel_p} — Extramargine annuo vs 2019 in M€\n"
            "(rosso=sopra baseline; blu=sotto; volume proxy MISE 2022)",
            fontsize=9
        )
        ax_r.grid(alpha=0.25, axis="y")
        ax_r.set_xticks(years_a)
        ax_r.tick_params(axis="x", labelsize=9)

        # Etichette valori
        for yr_v, wv in zip(years_a, wf_vals):
            ax_r.text(yr_v, wv + (15 if wv >= 0 else -20), f"{wv:+.0f}",
                      ha="center", fontsize=8, fontweight="bold",
                      color="#c0392b" if wv > 0 else "#2980b9")

    fig_ann.suptitle(
        "Analisi annuale margini — H₀: 'alla fine dell'anno hanno guadagnato uguale?'\n"
        "Rosso = anno sopra soglia 2σ baseline 2019  |  Windfall = Σ (margin_t − μ₂₀₁₉) × volume",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_ann.savefig("plots/03_annual_margins.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_ann)
    print("Salvato: plots/03_annual_margins.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: τ_lag heatmap — relazione tra τ_price e τ_margin per ogni evento
# ─────────────────────────────────────────────────────────────────────────────
tau_lag_rows = [
    r for r in results
    if r.get("lag_tau_days") is not None and r["tau_margin"] != "N/A"
]
if tau_lag_rows:
    fig_tl, ax_tl = plt.subplots(figsize=(10, max(3, len(tau_lag_rows)*0.7)))

    labels_tl = [f"{r['Evento'].split('(')[0].strip()} {r['Carburante']}"
                 + (" ⚠" if r["preliminare"] else "")
                 for r in tau_lag_rows]
    lags       = [r["lag_tau_days"] for r in tau_lag_rows]
    bar_colors = [
        "#c0392b" if lg < -7 else "#e67e22" if lg <= 14 else "#27ae60"
        for lg in lags
    ]

    ax_tl.barh(range(len(lags)), lags, color=bar_colors, alpha=0.82,
               edgecolor="black", lw=0.7, height=0.55)
    ax_tl.axvline(-7, color="#c0392b", lw=1.2, ls=":", alpha=0.7,
                  label="Soglia ANTICIPATORIO (< −7gg)")
    ax_tl.axvline(14, color="#27ae60", lw=1.2, ls=":", alpha=0.7,
                  label="Soglia REATTIVO (> +14gg)")
    ax_tl.axvline(0, color="black", lw=0.8)
    ax_tl.axvspan(-200, -7, alpha=0.04, color="#c0392b")   # zona anticipatoria
    ax_tl.axvspan(14, 200, alpha=0.04, color="#27ae60")    # zona reattiva

    for i, (lg, row) in enumerate(zip(lags, tau_lag_rows)):
        interp_short = (
            "ANTICIP." if lg < -7 else
            "SINCRONO" if lg <= 14 else "REATTIVO"
        )
        ax_tl.text(lg + (3 if lg >= 0 else -3), i,
                   f"{lg:+d}gg  {interp_short}",
                   va="center", ha="left" if lg >= 0 else "right",
                   fontsize=8, color="black")

    ax_tl.set_yticks(range(len(lags)))
    ax_tl.set_yticklabels(labels_tl, fontsize=9)
    ax_tl.set_xlabel("lag_tau = τ_margin − τ_price (giorni)", fontsize=10)
    ax_tl.set_title(
        "Relazione temporale τ_margin vs τ_price\n"
        "Rosso = margine si rompe PRIMA del wholesale (ANTICIPATORIO — segnale speculativo)\n"
        "Arancio = sincrono  |  Verde = margine segue il prezzo (REATTIVO — cost pass-through)",
        fontsize=9, fontweight="bold"
    )
    ax_tl.legend(fontsize=8, loc="lower right")
    ax_tl.grid(alpha=0.25, axis="x")
    plt.tight_layout()
    fig_tl.savefig("plots/03_tau_lag.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_tl)
    print("Salvato: plots/03_tau_lag.png")


print("\nScript 03 completato.")
print("  H0 (nuova): μ_post = μ_2019  — one-sample t-test, one-sided upper.")
print("  H1 (nuova): μ_post > μ_2019")
print("  MW migliorato: confronto post vs distribuzione 2019 (n~52 settimane).")
print("    → allineato con H0 Welch; nessuna assunzione di adiacenza temporale.")
print("    → mw_local (vs shock_hard pre) mantenuto nel CSV per confronto.")
print("  Block perm / HAC: DOPPIA IMPLEMENTAZIONE:")
print("    [PRINCIPALE]  split τ_price — esogeno al margine, pre window pulita.")
print("      → entra nella BH correction come test confirmatory.")
print("    [ROBUSTNESS]  split τ_margin — endogeno, isola la rottura del margine.")
print("      → fuori dalla BH; check metodologico riportato nel CSV.")
print("      → convergenza τ_price/τ_margin documentata in perm_split_convergence")
print("         e hac_split_convergence: True = risultato robusto allo split.")
print("  TENSIONE IRAN-ISRAELE: τ_margin precede τ_price di ~7gg.")
print("    → Con split τ_price i 7gg di rottura del margine finiscono nel 'post'.")
print("    → Effetto trascurabile in pratica; il robustness check lo verifica.")
print("  Multi-split: shock_hard / tau_price / tau_margin / pre_2019 (4 split).")
print("    → pre_2019: MW+Welch vs 2019; perm/HAC skippati (non adiacenti).")
print("  Hormuz incluso come preliminare (escluso dalla BH correction).")
print("  data/confirmatory_pvalues.csv -> input per 05_global_corrections.py")