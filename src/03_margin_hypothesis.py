"""
03_margin_hypothesis.py  (fix: Hormuz aggiunto, unità EUR/L corrette)
========================
Il margine lordo dei distributori italiani e' aumentato anomalmente
rispetto al baseline 2019 dopo gli shock energetici?

NOTA HORMUZ: i dati post-shock Hormuz (feb 2026) coprono solo ~7-8 settimane
al momento dell'analisi. I risultati sono preliminari e da non citare come
conclusivi nella relazione; vengono inclusi per completezza della pipeline.

Input:
  data/dataset_merged_with_futures.csv
  data/regression_diagnostics.csv

Output:
  data/table2_margin_anomaly.csv
  data/confirmatory_pvalues.csv
  data/baseline_sensitivity.csv
  plots/03_margins.png
  plots/03_delta_summary.png
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
for fuel, col in MARGIN_COLS.items():
    if col in baseline.columns:
        vals = baseline[col].dropna()
        thresholds[fuel] = float(2 * vals.std()) if len(vals) >= 4 else 0.030
        print(f"Baseline 2019 | {fuel}: mu={vals.mean():.5f}  "
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

        # 1. Welch t-test
        t_stat, t_p = stats.ttest_ind(post_m, pre_m, equal_var=False)
        delta_mean  = float(post_m.mean() - pre_m.mean())
        boot_m, b_lo, b_hi = bootstrap_ci(pre_m, post_m)

        # 2. Mann-Whitney
        mw = mann_whitney_full(pre_m, post_m)

        # 3. Block permutation
        obs_perm, p_perm = perm_test(pre_m, post_m, rng=rng_perm)

        # 4. HAC
        hac = hac_test(pre_m, post_m)

        soglia  = thresholds.get(fuel, 0.030)
        anomalo = abs(delta_mean) > soglia

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
        print(f"    delta_mean = {delta_mean:+.5f} EUR/L  "
              f"[boot CI: {b_lo:+.5f}, {b_hi:+.5f}]")
        print(f"    soglia 2sigma = {soglia:.5f}  |  delta anomalo: {anomalo}")
        print(f"    Welch t: t={t_stat:.3f}  p={t_p:.4f} {_stars(t_p)}")
        print(f"    Mann-Whitney: AUC={mw['AUC']:.3f}  "
              f"p(one)={mw['p_one']:.4f} {_stars(mw['p_one'])}  "
              f"HL={mw['hodges_lehmann']:+.5f}  Cliff={mw['cliffs_delta']:+.3f} "
              f"[{mw['magnitude']}]")
        print(f"    Block perm: delta_med={obs_perm:+.5f}  "
              f"p={p_perm:.4f} {_stars(p_perm)}")
        print(f"    HAC: delta={hac['delta_hac']:+.5f}  "
              f"p={hac['hac_p']} {_stars(hac['hac_p']) if not np.isnan(hac['hac_p']) else ''}")
        print(f"    Classificazione: {clas}")
        if prelim:
            print(f"    NOTA: risultato PRELIMINARE — solo {len(post_m)} settimane post-shock")
        if divergence:
            print(f"    {divergence}")

        row = {
            "Evento":             ev_name,
            "Carburante":         fuel,
            "preliminare":        prelim,
            "n_pre":              len(pre_m),
            "n_post":             len(post_m),
            "delta_mean_eur":     round(delta_mean, 5),
            "boot_CI_lo":         round(b_lo, 5),
            "boot_CI_hi":         round(b_hi, 5),
            "soglia_2sigma":      round(soglia, 5),
            "delta_anomalo":      anomalo,
            "t_stat":             round(float(t_stat), 4),
            "t_p":                round(float(t_p), 4),
            "t_H0":               "RIFIUTATA" if stat_sig else "non rifiutata",
            **{f"mw_{k}": v for k,v in mw.items()},
            "perm_delta_med":     round(obs_perm, 5),
            "perm_p":             round(p_perm, 4),
            "perm_H0":            "RIFIUTATA" if p_perm < ALPHA else "non rifiutata",
            **{f"hac_{k}": v for k,v in hac.items()},
            "classificazione":    clas,
            "divergenza_t_mw":    divergence,
        }
        results.append(row)

        # Solo eventi non preliminari entrano nella BH correction
        if not prelim:
            for fonte, p_val in [
                (f"Welch_t_{ev_name}_{fuel}",    float(t_p)),
                (f"MannWhitney_{ev_name}_{fuel}", float(mw["p_one"])),
                (f"BlockPerm_{ev_name}_{fuel}",   float(p_perm)),
                (f"HAC_{ev_name}_{fuel}",         float(hac["hac_p"])
                 if not np.isnan(hac["hac_p"]) else None),
            ]:
                if p_val is not None and not np.isnan(p_val):
                    conf_pvalues.append({
                        "fonte": fonte, "tipo": "confirmatory",
                        "descrizione": f"{ev_name} | {fuel}",
                        "p_value": p_val,
                    })
        else:
            print(f"    [skip BH] Hormuz escluso dalla BH correction (dati preliminari)")


# ─────────────────────────────────────────────────────────────────────────────
# BH correction locale (sui test primari Welch t)
# ─────────────────────────────────────────────────────────────────────────────
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
            return "Neutro / trasmissione attesa" if not row["delta_anomalo"] \
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
# Plot 1: margini nel tempo con banda baseline
# ─────────────────────────────────────────────────────────────────────────────
def _war_lines(ax, y_top):
    for label, (dt, color) in WAR_DATES.items():
        if merged.index[0] <= dt <= merged.index[-1]:
            ax.axvline(dt, color=color, lw=1.8, ls="--", alpha=0.85)
            ax.text(dt + pd.Timedelta(days=5), y_top*0.96,
                    label, rotation=90, fontsize=8, color=color, va="top")

fig_m, axes_m = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
baseline_data  = merged.loc[BASELINE_START:BASELINE_END]

for ax, (fuel, col), color in zip(
    axes_m,
    [("Benzina","margine_benz_crack"),("Diesel","margine_dies_crack")],
    ["#e67e22","#8e44ad"]
):
    if col not in merged.columns: continue
    s = merged[col].dropna()
    ax.plot(s.index, s.values, color=color, lw=1.8, label=fuel)
    bl = baseline_data[col].dropna()
    if len(bl) >= 4:
        ax.axhspan(bl.mean()-2*bl.std(), bl.mean()+2*bl.std(),
                   alpha=0.12, color="#888", label="Baseline ±2σ (2019)")
        ax.axhline(bl.mean(), color="#888", lw=1.0, ls="--")
    _war_lines(ax, s.max())
    ax.set_ylabel("Margine lordo (EUR/litro)", fontsize=10)
    ax.set_title(f"Margine crack spread — {fuel}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # Annotazione range atteso
    ax.text(merged.index[5], bl.mean() if len(bl) >= 4 else s.mean(),
            f"media 2019: {bl.mean():.3f} EUR/L" if len(bl) >= 4 else "",
            fontsize=8, color="#555")

axes_m[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes_m[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=40, fontsize=9)
plt.tight_layout()
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


print("\nScript 03 completato.")
print("  H0 testata con Welch t + Mann-Whitney + Block perm + HAC.")
print("  Hormuz incluso come preliminare (escluso dalla BH correction).")
print("  data/confirmatory_pvalues.csv -> input per 05_global_corrections.py")