# Fuel Price Speculation in Italy: Statistical Analysis of Three Geopolitical Shocks

This project tests whether the gross distribution margin on Italian retail fuels (gasoline,
diesel, pre-tax) increases anomalously beyond the 2σ baseline following geopolitical shocks.
All results are fully reproducible from the scripts and public data sources listed below.

---

## Null Hypothesis

**H₀**: The gross wholesale margin on Italian retail fuels (crack spread = retail pre-tax
price − European wholesale futures price) does not increase by more than 2σ relative to the
2019 baseline during the three energy crises analyzed.

**H₁**: The margin increases anomalously (> 2σ), consistent with opportunistic pricing in
the distribution chain.

The 2σ threshold is computed from the full-year 2019 baseline (52 weeks):

| Benchmark | Mean (EUR/litre) | σ | Threshold 2σ |
|---|---|---|---|
| Benzina (Eurobob crack) | 0.1683 | 0.0189 | **0.0377** |
| Diesel (Gas Oil crack) | 0.1488 | 0.0183 | **0.0367** |

Sensitivity analysis with Full-2021 as alternative baseline produces thresholds of 0.0485
(benzina) and 0.0522 (diesel); qualitative conclusions are unchanged. 2020 is excluded from
the baseline due to COVID-19 (WTI negative, demand collapse ~25%, structurally non-stationary).

**Three events analyzed:**

| Event | Shock date |
|---|---|
| Russian invasion of Ukraine | 24 February 2022 |
| Israel–Iran War (Twelve-Day War) | 13 June 2025 |
| Strait of Hormuz closure | 28 February 2026 |

---

## Dataset

| Variable | Source | Ticker / File | Frequency | Units |
|---|---|---|---|---|
| Brent crude oil price | Yahoo Finance via `yfinance` | `BZ=F` | Daily → weekly (W-MON) | USD/barrel → EUR/barrel |
| EUR/USD exchange rate | Yahoo Finance via `yfinance` | `EURUSD=X` | Daily → weekly mean | — |
| Italian retail fuel prices (pre-tax) | EU Weekly Oil Bulletin, EC | Sheet *Prices wo taxes*, cols `IT_price_wo_tax_euro95`, `IT_price_wo_tax_diesel` | Weekly (Monday) | EUR/litre |
| Eurobob gasoline futures | TradingView — manual CSV export | `Eurobob Futures Historical Data.csv` | Daily → weekly | USD/tonne → EUR/litre |
| Gas Oil ICE London futures | Investing.com — manual CSV export | `London Gas Oil Futures Historical Data.csv` | Daily → weekly | USD/tonne → EUR/litre |

**Coverage**: 380 weekly observations, 2019-01-07 → 2026-04-13.

**Manual CSV exports** — the two futures files are not available via API and were obtained
through manual browser export:

- `Eurobob Futures Historical Data.csv`: exported from TradingView, ticker `NYMEX:B7H1!`
  (Eurobob Gasoline ARA continuous front-month).
  URL: https://it.tradingview.com/chart/v8Lm6UVY/?symbol=NYMEX%3AB7H1%21
  Procedure: open the link → select the full date range → export via
  "Export chart data" (CSV). Column `Price` = daily settlement in USD/tonne.

- `London Gas Oil Futures Historical Data.csv`: exported from Investing.com, instrument
  ID 1184928 (ICE Low Sulphur Gasoil Futures, London).
  URL: https://www.investing.com/commodities/london-gas-oil-historical-data?cid=1184928
  Procedure: open the link → set date range to 2019-01-01 → present → click
  "Download Data" (free account required). Column `Price` = daily settlement in USD/tonne.

Both files are included in the repository. To extend the analysis beyond 2026-04-13,
repeat the manual export from the same URLs and replace the CSV files before re-running.

**Unit conversions** (reproducible):
- Brent: `brent_eur = brent_usd / eurusd`
- Eurobob → EUR/litre: `eurobob_usd_tonne / eurusd / (1000 / 0.74)` (density 0.74 kg/L, ≈ 1351 L/t)
- Gas Oil → EUR/litre: `gasoil_usd_tonne / eurusd / (1000 / 0.84)` (density 0.84 kg/L, ≈ 1190 L/t)

**Missing data**: 9 weeks with partial or absent retail prices (holiday periods):
`2019-04-22, 2019-12-23, 2019-12-30, 2020-04-13, 2020-12-28, 2021-01-04, 2021-04-05,
2021-12-27, 2022-04-18`. Handled by linear time interpolation; no synthetic data is generated.
The full list is saved to `data/missing_weeks.json` and `data/missing_weeks.csv`.

**Smoothing**: Brent receives a 7-day daily rolling average before weekly resampling, to
reduce intra-week noise from futures trading. Retail prices are used at their native weekly
frequency. No additional smoothing is applied to either weekly series, to avoid introducing
artificial transmission lags.

---

## Methods and Scripts

Scripts must be run in order. The full pipeline is launched via `run_all.py`.

```bash
python run_all.py
```

Or individually:

```bash
python 01_data_pipeline.py
python 02_core_analysis.py
python 03_statistical_tests.py
python 04_global_corrections.py
```

### 01_data_pipeline.py — Data collection and merging

Downloads Brent, EUR/USD, and EU Oil Bulletin data. Loads Eurobob and Gas Oil futures from
local CSV files (Investing.com export). Resamples all series to weekly frequency (W-MON),
merges on the retail price index, applies forward-fill with `limit=4` for isolated missing
Brent weeks. Saves `data/dataset_merged.csv` (prices only) and
`data/dataset_merged_with_futures.csv` (prices + crack spreads). Produces three overview
plots with war event markers.

### 02_core_analysis.py — Primary H₀ test + Bayesian changepoint detection

**Section B — Bayesian changepoint on log-prices (Table 1)**

Fits a piecewise linear regression to each event × series combination (9 total: 3 events ×
{Brent, benzina, diesel}) using MCMC via PyMC (NUTS sampler, 4 chains). Dependent variable:
log-transformed weekly price. The two regression lines meet at τ (continuity constraint:
`a2 = a1 + τ·(b1 − b2)`). Piecewise mean approximated via sigmoid with slope 50.

Prior specification:
- `τ_raw ~ Beta(2, 2)` rescaled to `[x_min, x_max]` (avoids boundary effects of Uniform)
- `σ ~ HalfNormal(sd(y))`
- `ν ~ Gamma(2, 0.1)` → E[ν] = 20 (Student-T likelihood with estimated degrees of freedom)
- `b1, b2 ~ Normal(0, 3·sd(y))`
- `a1 ~ Normal(mean(y), sd(y))`

AR(1) is not modelled explicitly (removed after testing: creates sequential geometry
incompatible with NUTS → systematic max_treedepth, Rhat > 1.01). Autocorrelation is
reported as a diagnostic (Durbin-Watson in script 03) but not modelled in the changepoint step.

MCMC settings per scenario (configurable in `MCMC_CONFIG` dict):

| Scenario | draws | tune | target_accept | init |
|---|---|---|---|---|
| Default | 2000 | 2000 | 0.95 | `adapt_diag` |
| Ucraina / Brent | 3000 | 6000 | 0.99 | `adapt_full` |
| Ucraina / Diesel margin | 2000 | 4000 | 0.98 | `adapt_full` |
| Iran / Diesel margin | 3000 | 5000 | 0.99 | `adapt_full` |

Convergence criteria: Rhat ≤ 1.01 (acceptable up to 1.05 with warning), ESS ≥ 100 per chain.
All 9 Table 1 scenarios converge: Rhat_max ≤ 1.010, ESS_min ≥ 355.

Lag D (days) = τ̂ − shock_date. Negative D means the changepoint precedes the shock.

**Section C — Crack spread construction**

Gross margin = `pump_price_eur_l − futures_wholesale_eur_l` for each week.
Two series: `margine_benz_crack` (Eurobob) and `margine_dies_crack` (Gas Oil ICE).

**Section D — Primary anomaly test on margins (Table 2) + BH local correction**

For each event × fuel combination (excluding Hormuz: ≤ 4 post-shock weeks), computes:
- Welch t-test (unequal variances) on pre- vs post-shock crack spread: **primary test**
- KS two-sample test: diagnostic only (not in decision rule)
- Bootstrap 95% CI on Δ (2000 iterations): diagnostic only

Bayesian changepoint is also fit to the margin series (same model, scenario-specific MCMC
config), to identify when the margin structurally shifted.

Classification uses the Welch t-test p-value as the sole gate (for BH consistency):

| Classification | Condition |
|---|---|
| MARGINE ANOMALO POSITIVO | t_p < α AND Δ > 2σ AND Δ > 0 |
| COMPRESSIONE MARGINE | t_p < α AND Δ > 2σ AND Δ < 0 |
| NEUTRO / TRASMISSIONE ATTESA | |Δ| ≤ 2σ |
| VARIAZIONE STATISTICA | t_p < α AND |Δ| ≤ 2σ |
| INCONCLUSIVO | t_p ≥ α AND |Δ| > 2σ |

Benjamini-Hochberg FDR correction at 5% is applied to the 4 t-test p-values (BH local).

### 03_statistical_tests.py — Auxiliary tests

All tests in this script are **exploratory evidence** and do not enter the primary H₀
decision rule.

**§1 Granger causality (Brent → retail prices)**
First-differenced log-prices (ADF-verified non-stationary in levels). Lags 1–8 weeks.
F-test variant. Year 2020 excluded (COVID-19 structural non-stationarity).

**§2 Rockets & Feathers (OLS+HAC)**
Separate OLS pass-through coefficients for positive (β_up) and negative (β_down) Brent
changes. SE adjusted via Newey-West HAC (4 lags). R&F index = |β_up| / |β_down|.
Note: GLSAR AR(1) failed due to a compatibility issue with the installed NumPy version;
OLS+HAC is the fallback. With DW ≈ 0.15–0.29 and ρ_AR1 ≈ 0.85–0.92 in all event windows,
OLS SE are inflated; HAC correction partially addresses this but the t-test on asymmetry
remains imprecise.

**§3 KS, ANOVA, Chow**
KS: full distribution comparison pre vs post shock.
ANOVA: three-period comparison (pre / shock+6w / post).
Chow: structural break test at the official shock date.

**§4 CCF and rolling correlation**
Cross-correlation function (lags 0–12 weeks) between Brent and retail price changes.
Rolling Pearson correlation (12-week window).

**§5 Bootstrap 95% CI on lag D**
2000 bootstrap samples; changepoint estimated per sample as the index maximising the
absolute difference between pre- and post-split means.

**§6 Regression type selection**
Reports Breusch-Pagan, Ljung-Box, Durbin-Watson, ρ_AR1, and SE inflation (OLS vs HAC)
for each event × fuel window. All windows show DW ≈ 0.15–0.29 and ρ_AR1 ≈ 0.85–0.92,
confirming near-unit-root autocorrelation and recommending Bayesian StudentT models over OLS.

**§7 Welch t-test on simplified margin proxy**
Proxy: `M̃_t = pump_price_eur_l − brent_eur / 159`. This auxiliary measure does not
separate refining costs from distribution margins. Results are reported for comparison
but are methodologically weaker than the crack spread test.

**§8 Difference-in-Differences (DiD)**
Countries: Germany and Sweden (from EU Oil Bulletin, same file).
Model: `Margin_{c,t} = α + β₁·Italy_c + β₂·Post_t + δ·(Italy_c × Post_t) + ε`,
estimated with OLS HC3 robust SE.
Parallel trends pre-test (PTA): OLS on `Italy × t` interaction in the pre-shock window;
p_PTA ≥ 0.05 = assumption not rejected.
δ = DiD estimator: margin change in Italy relative to control country after shock.

### 04_global_corrections.py — Global Benjamini-Hochberg correction

Collects all p-values from scripts 02 and 03. Applies BH correction globally:
- **Confirmatory** (16 tests): 4 Welch t-tests (Table 2) + 12 DiD δ estimates
- **Exploratory** (32 tests): Granger (16), R&F (2), KS (6), ANOVA (4), Chow (6)

Exploratory tests are not corrected (reported as nominal p-values).
Saves `data/global_bh_corrections.csv` and updates `data/table2_margin_anomaly.csv` with
columns `BH_global_reject` and `t_p_BH_adjusted`.

---

## Results

### Table 1 — Bayesian changepoints on log-prices

| Event | Series | τ̂ (median) | 95% CI | Lag D (days) | H₀\|D\|<30 |
|---|---|---|---|---|---|
| Ukraine (Feb 2022) | Brent EUR | 13 Dec 2021 | 15 Nov – 20 Jun | −73 | rejected |
| Ukraine (Feb 2022) | Benzina | 03 Jan 2022 | 27 Dec – 17 Jan | −52 | rejected |
| Ukraine (Feb 2022) | Diesel | 03 Jan 2022 | 06 Dec – 17 Jan | −52 | rejected |
| Iran-Israel (Jun 2025) | Brent EUR | 28 Apr 2025 | 14 Apr – 12 May | −46 | rejected |
| Iran-Israel (Jun 2025) | Benzina | 28 Apr 2025 | 14 Apr – 12 May | −46 | rejected |
| Iran-Israel (Jun 2025) | Diesel | 05 May 2025 | 21 Apr – 19 May | −39 | rejected |
| Hormuz (Feb 2026) | Brent EUR | 23 Feb 2026 | 09 Feb – 02 Mar | −5 | not rejected |
| Hormuz (Feb 2026) | Benzina | 02 Mar 2026 | 23 Feb – 02 Mar | +2 | not rejected |
| Hormuz (Feb 2026) | Diesel | 23 Feb 2026 | 23 Feb – 02 Mar | −5 | not rejected |

The 30-day threshold in Table 1 tests **proximity** of the structural break to the shock
date, not margin expansion. All Ukraine and Iran-Israel changepoints are anticipatory
(negative lag), consistent with futures markets pricing in geopolitical risk before the
official event. Hormuz changepoints are near-synchronous, consistent with an unexpected event.

MCMC diagnostics: Rhat_max ≤ 1.010 for all scenarios; ESS_min ≥ 355 (Brent Ukraine, worst
case due to complex geometry requiring `adapt_full` and extended tuning). ν (Student-T
degrees of freedom) ranges from 1.46 (Benzina Ukraine, heavy tails) to 22.82 (Diesel Iran).

### Table 2 — Primary H₀ test: crack spread anomaly

(Hormuz excluded: ≤ 4 post-shock weekly observations as of 2026-04-13.)

| Event | Fuel | Method | Δ crack (EUR/l) | Boot. 95% CI | Welch p | BH local | BH global (p_adj) | Classification |
|---|---|---|---|---|---|---|---|---|
| Ukraine (Feb 2022) | Benzina | Eurobob ARA | +0.039 | [−0.008; +0.084] | 0.108 | — | 0.297 | Variazione stat. |
| Ukraine (Feb 2022) | Diesel | Gas Oil ICE | +0.048 | [−0.001; +0.095] | 0.065 | — | 0.297 | Variazione stat. |
| Iran-Israel (Jun 2025) | Benzina | Eurobob ARA | −0.009 | [−0.023; +0.004] | 0.192 | — | 0.297 | Neutro |
| Iran-Israel (Jun 2025) | Diesel | Gas Oil ICE | **−0.028** | [−0.045; −0.012] | **0.004** | **✓** | **0.028** | Neutro (↓) |

H₀ is rejected in **1 of 4 cases** after BH FDR 5% correction. The single rejection
(Iran-Israel diesel, p_adj = 0.028) indicates a **decrease** in the crack spread, not an
increase. Margins on Eurobob/Gas Oil do not expand anomalously during any of the events.

Note: The bootstrap CI crosses zero for Ukraine benzina and diesel, and for Iran-Israel
benzina, confirming non-significance. The KS test is significant for Ukraine (KS = 1.000,
p < 0.001) and Iran-Israel diesel (p = 0.002) but is a diagnostic only and does not alter
the BH-corrected decision.

### Auxiliary results

**Granger causality** (327 weeks, 2020 excluded):
Brent Granger-causes benzina and diesel at all lags 1–8 weeks (F = 54.83 at lag 1 for
benzina, F = 40.58 for diesel; all p < 0.0001). Significance is sustained at lag 8 (56
days, F ≥ 6.3). This confirms rapid and persistent transmission from crude oil to retail
prices, but does not distinguish efficient markets from anticipatory pricing.

**Rockets & Feathers** (OLS+HAC, Newey-West 4 lags):
β_up = 0.398, β_down = 0.265 for benzina (R&F index = 1.50, p = 0.527, n.s.).
β_up = 0.537, β_down = 0.234 for diesel (R&F index = 2.30, p = 0.280, n.s.).
The asymmetry is economically suggestive but statistically non-significant, likely due to
OLS SE inflation from near-unit-root autocorrelation (ρ_AR1 ≈ 0.85–0.92 in all windows).

**KS, ANOVA, Chow** (prices, not margins):
KS_stat = 1.000 for Ukraine (both fuels, p < 0.000001): pre- and post-shock price
distributions are completely non-overlapping. ANOVA F = 142.8 (benzina) and 217.5 (diesel)
for Ukraine, p < 0.000001. Chow F = 14.8–224.0 across all events and fuels, p < 0.001.
These tests confirm structural price breaks at the shock dates but do not isolate margin
changes from Brent pass-through.

**Welch t-test on simplified margin proxy** (M̃ = pump − Brent/159):
Ukraine: Δ = +0.150 EUR/l benzina (d = 2.61, p < 0.001), Δ = +0.237 EUR/l diesel (d = 3.77,
p < 0.001). Iran-Israel: Δ = +0.017 EUR/l benzina (d = 0.99, p = 0.002), Δ = +0.015 EUR/l
diesel (d = 0.72, p = 0.024). Hormuz: not significant (p > 0.09). The large Ukraine effects
are partially artefactual: the Brent/159 proxy does not deduct refining costs; during Ukraine
the Eurobob crack rose substantially more than the simple Brent-to-litre conversion suggests.

**DiD** (Italy vs Germany and Sweden):
No significant Italy-specific margin increase detected for Ukraine or Iran-Israel (all δ ∈
[−0.034; +0.030] EUR/l, p > 0.15). For Hormuz vs Germany benzina: δ = −0.093 EUR/l, p = 0.002
(BH global rejected), but the parallel trends assumption is violated (PTA p < 0.001), making
causal interpretation unreliable. Sweden comparisons are consistently non-significant with
PTA satisfied (p > 0.07 in all cases).

**Global BH correction** (16 confirmatory tests):
2 of 16 tests rejected at FDR 5%:
1. Iran-Israel diesel crack spread Welch t: p = 0.004 → p_adj = 0.028 (margin *decrease*)
2. Hormuz benzina DiD vs Germany: p = 0.002 → p_adj = 0.028 (PTA violated, unreliable)

---

## Methodological Notes

### Why crack spread rather than Brent/159

The Brent-to-litre proxy (`pump − Brent/159`) conflates refining costs with distribution
margins. During supply shocks, Eurobob and Gas Oil futures rise faster than Brent because
refining capacity constraints amplify wholesale spreads. Using Eurobob/Gas Oil as the
wholesale benchmark isolates the distribution margin from the refining margin. Analyses
based solely on Brent/159 will systematically overestimate distribution margin anomalies
during supply-side shocks.

### Why 2019 as baseline (not 2021)

2020: COVID-19, WTI futures negative in April, demand collapse ~25% — structurally
non-stationary, not representative of normal market conditions. 2021 H1: post-COVID
recovery, prices still rebounding, margins structurally compressed — introduces survivorship
bias. 2019: stable Brent (60–70 USD/bbl), no structural shocks, 52 full weeks available.
Sensitivity analysis with Full-2021 baseline is reported in `data/baseline_sensitivity.csv`.

### Why AR(1) is excluded from the Bayesian changepoint model

Including an explicit AR(1) term in the PyMC model creates sequentially dependent latent
variables (ε_t depends on ε_{t−1}), producing funnel geometry that causes systematic
max_treedepth hits and Rhat > 1.01 with NUTS. Autocorrelation is instead handled through
the heavy-tailed Student-T likelihood (ν estimated from data). DW and ρ_AR1 are reported as
diagnostics in `data/regression_selection.csv` but are not modelled in the changepoint step.
See Betancourt (2017) §5 for the geometric argument.

### Hormuz as a preliminary event

As of 2026-04-13, only 6–7 post-shock weekly observations are available for retail prices
(shock: 2026-02-28; last data point: 2026-04-13). Table 2 requires a minimum of 5 post-shock
weeks; Hormuz is excluded with this explanation logged in the script output. Changepoint
estimates (Table 1) and Granger/KS/Chow tests are computed on the available data and should
be treated as preliminary. Results will stabilize as additional post-shock weeks accumulate.

### Multiple testing

48 p-values are produced by the full pipeline. The BH correction is applied in two layers:
- **BH local** (script 02): controls FDR on the 4 primary Welch t-tests (Table 2 margin tests)
- **BH global** (script 04): controls FDR on the 16 confirmatory tests (4 Welch + 12 DiD)

Using AND(t-test, KS) as a composite gate was explicitly rejected: it would create an
uncontrolled composite test with nominal α not governed by the BH correction on t_p alone
(Benjamini & Hochberg 1995; Holm 1979). KS results are reported as diagnostics.

---

## Output Files

| File | Description |
|---|---|
| `data/dataset_merged.csv` | 380 weeks: Brent EUR + retail prices (benzina, diesel) |
| `data/dataset_merged_with_futures.csv` | As above + Eurobob, Gas Oil, crack spreads |
| `data/missing_weeks.csv` / `.json` | 9 weeks with missing retail data (linearly interpolated) |
| `data/table1_changepoints.csv` | Table 1: τ̂, 95% CI, lag D, OLS slopes, doubling times, MCMC diagnostics |
| `data/table2_margin_anomaly.csv` | Table 2: crack spread test, BH local + global results |
| `data/baseline_sensitivity.csv` | 2σ thresholds under 2019 and Full-2021 baselines |
| `data/global_bh_corrections.csv` | All 48 p-values with BH global correction (confirmatory) |
| `data/granger_benzina.csv` | Granger F-stats and p-values, lags 1–8, benzina |
| `data/granger_diesel.csv` | Granger F-stats and p-values, lags 1–8, diesel |
| `data/rockets_feathers_results.csv` | β_up, β_down, SE_HAC, R&F index, Wald p |
| `data/ks_results.csv` | KS statistic and p-value per event × fuel |
| `data/anova_results.csv` | ANOVA F, p, per-period means per event × fuel |
| `data/chow_results.csv` | Chow F and p per event × fuel |
| `data/bootstrap_ci.csv` | Bootstrap 95% CI on lag D per event × fuel |
| `data/regression_selection.csv` | BP, Ljung-Box, DW, ρ_AR1, SE inflation per event × fuel |
| `data/ttest_margin.csv` | Welch t-test on simplified proxy M̃ = pump − Brent/159 |
| `data/did_results.csv` | DiD δ, SE_HC3, CI, PTA p-value, 12 combinations |
| `data/brent_weekly_eur.csv` | Brent weekly series in EUR/barrel |
| `data/prezzi_pompa_italia.csv` | Italian retail weekly prices pre-tax, EUR/litre |
| `data/regression_diagnostics.csv` | BP, SW, DW per event × series (from script 02) |
| `plots/01a_brent.png` | Brent EUR/barrel 2019–2026 with event markers |
| `plots/01b_benzina.png` | Retail benzina EUR/litre with event markers |
| `plots/01c_diesel.png` | Retail diesel EUR/litre with event markers |
| `plots/02_{event}_{series}.png` | Bayesian piecewise regression + posterior KDE of τ (9 plots) |
| `plots/02_{event}_{series}_diag.png` | OLS regression diagnostics (residuals, QQ, ACF) (9 plots) |
| `plots/03_granger_combined.png` | Granger p-values by lag, benzina + diesel side by side |
| `plots/04_rf_combined.png` | R&F scatter plots, benzina + diesel |
| `plots/06_statistical_tests.png` | CCF, rolling correlation, KS ECDF, ANOVA F, Chow p, bootstrap CI |
| `plots/07_margins_margine_benz_crack.png` | Benzina crack spread 2019–2026 with baseline ±2σ |
| `plots/07_margins_margine_dies_crack.png` | Diesel crack spread 2019–2026 with baseline ±2σ |
| `plots/07_delta_summary.png` | Δ crack spread per event × fuel with bootstrap CI |
| `plots/08_regression_selection.png` | (produced if run standalone) |
| `plots/09_ttest_did.png` | Welch t-test forest plot + DiD δ forest plot |

---

## Requirements and Setup

Python 3.9 or higher.

```bash
git clone https://github.com/your-username/fuel-price-speculation-italy.git
cd fuel-price-speculation-italy
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
deactivate
```

`requirements.txt`:

```
yfinance
pymc>=5.0
pytensor
arviz
statsmodels
pandas
numpy
matplotlib
scipy
requests
openpyxl
```

**Expected runtime**: approximately 10–15 minutes on a modern laptop (dominated by MCMC
sampling: ~4 minutes for Brent Ukraine with 6000 tune steps + 3000 draws).

---

## Reproducibility Checklist

To reproduce all results exactly:

1. Run `python run_all.py` from the repository root.
2. The EU Oil Bulletin file (`data/eu_oil_bulletin_history.xlsx`) must be present. Script 01
   downloads it automatically; if the download fails (URL changes), download manually from
   [energy.ec.europa.eu](https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en)
   and save to `data/eu_oil_bulletin_history.xlsx`.
3. Eurobob and Gas Oil CSV files (`data/Eurobob Futures Historical Data.csv`,
   `data/London Gas Oil Futures Historical Data.csv`) must be present. These were obtained
   via manual browser export (see **Manual CSV exports** above) and are included in the
   repository. To extend the analysis beyond 2026-04-13, re-export from the same URLs:
   TradingView `NYMEX:B7H1!` for Eurobob, Investing.com cid=1184928 for Gas Oil ICE.
4. MCMC results have stochastic variation across runs (different random seeds). Stored results
   in `data/table1_changepoints.csv` and `data/table2_margin_anomaly.csv` reflect the run
   described above. Re-running will produce numerically close but not bit-identical values.
   All qualitative conclusions are robust to this variation (Rhat ≤ 1.010 in all scenarios).
5. Brent and EUR/USD prices from Yahoo Finance may differ slightly if downloaded at a
   different time, as futures series are occasionally revised. The `data/brent_weekly_eur.csv`
   file in the repository reflects the download dated 2026-04-20.

---

## References

- Bacon RW. *Rockets and Feathers: The Asymmetric Speed of Adjustment of UK Retail Gasoline
  Prices to Cost Changes.* Energy Economics, 1991.
- Benjamini Y, Hochberg Y. *Controlling the False Discovery Rate: A Practical and Powerful
  Approach to Multiple Testing.* JRSS-B, 1995.
- Betancourt M. *A Conceptual Introduction to Hamiltonian Monte Carlo.* arXiv:1701.02434, 2017.
- Borenstein S, Cameron AC, Gilbert R. *Do Gasoline Prices Respond Asymmetrically to Crude
  Oil Price Changes?* Quarterly Journal of Economics, 1997.
- Casini A, Perron P. *Structural Breaks in Time Series.* Oxford Research Encyclopedia, 2021.
- Gelman A et al. *Bayesian Data Analysis*, 3rd ed. CRC Press, 2013.
- Meyer J, von Cramon-Taubadel S. *Asymmetric Price Transmission: A Survey.*
  Journal of Agricultural Economics, 2004.
- Newey WK, West KD. *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica, 1987.
- Angrist JD, Pischke J-S. *Mostly Harmless Econometrics.* Princeton UP, 2009.

---

## License

MIT License. All data used is publicly available from the sources listed above.