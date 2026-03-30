# Fuel Price Speculation in Italy: A Statistical Analysis of War-Driven Price Shocks

This project investigates whether the rapid increase in retail fuel prices (gasoline, diesel)
following geopolitical shocks is compatible with the physical supply chain timeline, or
whether it constitutes anticipatory pricing behavior — commonly referred to as speculation.

Three war events are analyzed:

- Russian invasion of Ukraine (24 February 2022)
- Israel-Iran War, also known as the Twelve-Day War (13 June 2025)
- Strait of Hormuz closure (28 February 2026)

---

## Null Hypothesis

H0: The lag between a crude oil price shock (Brent) and the corresponding change in Italian
retail fuel prices is >= 30 days, consistent with the physical supply chain
(maritime transport + refining + storage + distribution).

Rejecting H0 implies that retail prices respond faster than the logistics chain allows,
which constitutes evidence of speculative or anticipatory pricing.

The 30-day threshold is derived from the Italian import profile. Over 90% of crude oil
arrives by sea (IEA, 2022). Tanker transit times range from 7-21 days from the Persian Gulf
and 5-10 days from North Africa. Adding refinery processing (1-3 days), quality control,
mandatory strategic storage, and final distribution, a conservative lower bound of 30 days
is justified and consistent with the supply chain literature (Borenstein et al., 1997;
Meyer & von Cramon-Taubadel, 2004).

---

## Methods

The project runs six sequential analysis scripts.

### 1. Data Pipeline (01_data_pipeline.py)

Downloads Brent crude oil prices at daily frequency from Yahoo Finance (ticker BZ=F) and
Italian retail fuel prices at weekly frequency from the EU Weekly Oil Bulletin published by
the European Commission.

**Currency**: all prices are expressed in EUR. Brent (originally quoted in USD/barrel) is
converted to EUR/barrel using the weekly EUR/USD exchange rate downloaded from Yahoo Finance
(ticker EURUSD=X). Italian retail prices from the EU Oil Bulletin are natively in EUR/litre
and require no conversion.

**Smoothing**: Brent is smoothed with a 7-day rolling average before weekly resampling, to
reduce intra-week noise from daily futures trading. Retail prices are used at their native
weekly frequency without additional smoothing. This symmetric treatment avoids the artificial
~2-week lag that a 4-week rolling average on retail prices would introduce into transmission
estimates — a bias that would work against rejecting H0.

Applies natural log transformation to linearize exponential growth. Saves three separate
high-resolution overview plots (Brent, benzina, diesel) with war event markers.

### 2. Changepoint Detection (02_changepoint_detection.py)

Fits a Bayesian piecewise linear regression to each price series (Brent, benzina, diesel)
within a temporal window around each war event using Markov Chain Monte Carlo (MCMC) via
PyMC (NUTS sampler). The dependent variable is the log-transformed weekly price; the
independent variable is a numeric time index (weeks since the start of the window). The
model estimates a changepoint τ and two regression slopes b1 (pre-shock) and b2 (post-shock).

**Model specification**: the two regression lines are constrained to meet at the changepoint
(no level discontinuity), enforced via the deterministic relation a2 = a1 + τ·(b1 − b2).
The piecewise mean is implemented using a sigmoid approximation of the Heaviside step
function for gradient-based sampling compatibility:

```
μ(x) = (a1 + b1·x)·(1 − σ(50·(x−τ))) + (a2 + b2·x)·σ(50·(x−τ))
```

**Prior distributions** (following the referenced paper):
- τ ~ Uniform(min(x), max(x))
- σ ~ HalfNormal(sd(y))
- b1, b2 ~ StudentT(μ=0, σ=3·sd(y), ν=3)
- a1 ~ StudentT(μ=0, σ=sd(y)/range(x), ν=3)

Heavy-tailed Student-T priors on slopes and intercept are weakly informative and robust to
outliers in the price series. The HalfNormal prior on σ regularizes residual variance toward
the empirical scale of the data.

**MCMC settings**: 2000 posterior draws, 1000 tuning steps, 2 chains, target_accept=0.9,
random_seed=42. Convergence is monitored via the ArviZ trace object returned by each run.

**Credible intervals**: uncertainty on τ is quantified as a 95% Bayesian credible interval
derived from the marginal posterior distribution of τ — i.e., the shortest interval that
contains 95% of the posterior probability mass. This is reported alongside the posterior
mean as the point estimate (median used as the integer index). Credible intervals on b1 and
b2 are likewise extracted from their marginal posteriors and displayed as shaded bands on
each plot. A CI on τ entirely preceding the shock date constitutes strong evidence of
anticipatory pricing.

OLS is still used on the two identified segments solely for computing R-squared fit quality;
all slope and intercept estimates reported in Table 1 and plots come from the MCMC posteriors.

For each event and each fuel type, the script computes:
- tau_crude: changepoint date in the Brent series, with 95% posterior credible interval
- tau_retail: changepoint date in the retail series, with 95% posterior credible interval
- D = tau_retail - tau_crude: the observed transmission lag in days
- DT1 and DT2: price doubling times before and after the changepoint (in days)
- R-squared for each regression segment (OLS auxiliary fit)

H0 is rejected if D < 30 days. For Hormuz, results are flagged as preliminary given the
short post-shock observation window (< 4 weeks of data available as of the analysis date).
Produces one high-resolution plot per combination of event and series (9 plots total); each
plot shows raw log-price data, posterior mean regression lines with slope credible bands,
the changepoint τ with its posterior CI, and the official shock date.

### 3. Granger Causality (03_granger_causality.py)

Tests whether Brent prices Granger-cause retail prices at lags from 1 to 8 weeks (7 to 56
days). Uses first-differenced log prices to ensure stationarity, verified by the Augmented
Dickey-Fuller test. Applies the F-test variant of the Granger test as it is more robust for
small samples. A significant result at lag < 4 weeks (< 30 days) constitutes direct evidence
against H0: it means that past values of Brent improve the prediction of future retail prices
within a time window that the physical supply chain cannot explain.

**Output figures**: produces three plots.
- `plots/03_granger_benzina.png` and `plots/03_granger_diesel.png`: individual panel per fuel.
- `plots/03_granger_combined.png`: a single paper-quality figure with benzina and diesel
  side by side (shared layout, independent y-axes), intended for direct inclusion in the paper.

**Visual encoding**: bar color encodes statistical significance on a four-level scale —
dark red (#8b1a1a) for p < 0.001, medium red (#c0392b) for p < 0.01, orange-red (#e74c3c)
for p < 0.05, and grey-blue (#95a5a6) for non-significant results. Each bar is annotated
with the numeric p-value and asterisk notation (*, **, ***) above the bar. The α = 0.05
threshold is drawn as a dashed horizontal line with an integrated label; the 30-day physical
threshold is drawn as a dashed vertical line with a shaded region marking the speculative
zone (lag < 30 days). A console summary table prints all significant lags below 30 days
with F-statistics for quick inspection.

### 4. Rockets and Feathers (04_rocket_feather.py)

Tests the asymmetric price transmission hypothesis introduced by Bacon (1991): whether retail
prices rise faster when crude oil increases than they fall when crude oil decreases. Estimates
separate OLS pass-through coefficients for positive Brent changes (beta_up) and negative
Brent changes (beta_down). Tests the null of symmetry using a t-test on the difference
beta_up - beta_down. Computes the Rockets and Feathers index (R&F = |beta_up| / |beta_down|):
values greater than 1 indicate upward asymmetry. Produces scatter plots of Brent changes vs
retail changes with the two regression lines, and normalized time series for visual inspection.

### 5. Additional Statistical Tests (06_statistical_tests.py)

Seven complementary tests that corroborate H0 rejection from independent methodological angles:

**Kolmogorov-Smirnov**: a two-sample KS test on the empirical cumulative distribution of
prices before and after each shock. Tests whether the full distribution changes, not just
the mean. A KS statistic close to 1 means the two distributions are almost entirely
non-overlapping.

**One-way ANOVA**: compares price means across three periods — pre-shock (6 months before),
acute shock (first 6 weeks after), and post-shock normalization. A significant F-statistic
confirms that between-period variance dominates within-period variance.

**Chow Test**: a formal structural break test on the linear regression at the shock date.
Tests whether slope and intercept change significantly at the break point. Unlike changepoint
detection, the Chow test does not estimate where the break is — it tests whether a break
exists at a pre-specified date.

**Cross-Correlation Function (CCF)**: estimates the correlation between Brent and retail
price changes for lags from 0 to 12 weeks. The lag at which the correlation is maximized is
the empirical speed of price transmission. A peak lag below 4 weeks directly rejects H0.

**Rolling Correlation**: computes the Pearson correlation between Brent and retail prices
over a 12-week moving window. Near-zero correlation in non-war periods that spikes sharply
during conflicts reveals that the speculative transmission channel activates selectively
under geopolitical stress.

**Bootstrap Confidence Intervals on lag D**: a non-parametric block bootstrap (500 iterations,
block size 4 weeks) on the lag D = tau_retail - tau_crude, to quantify estimation uncertainty
without distributional assumptions. A 95% CI entirely below 30 days constitutes strong
evidence against H0 regardless of point estimate uncertainty. This complements the CI on the
individual changepoint dates reported in script 02.

**RMSE and MAE**: compares fit quality of the piecewise regression against a simple linear
model on the same data. A meaningful RMSE improvement validates the changepoint as a genuine
structural feature of the data rather than a statistical artifact.

---

## Results

All results are obtained from the pipeline run with real Brent data (Yahoo Finance, ticker
BZ=F) and real Italian retail fuel prices (EU Weekly Oil Bulletin, sheet "Prices wo taxes",
columns IT_price_wo_tax_euro95 and IT_price_wo_tax_diesel). The dataset covers 272 weekly
observations from 2021-01-11 to 2026-03-23.

### Changepoint Detection

**Ukraine (24 Feb 2022)**

The Brent changepoint falls on 3 January 2022, 52 days before the official invasion,
with a 95% credible interval of [11 Oct 2021 – 25 Jul 2022]. Both retail series show their
changepoints substantially earlier than the official shock date: benzina on 25 October 2021
(CI [11 Oct 2021 – 17 Jan 2022]) and diesel on 11 October 2021 (CI [4 Oct 2021 – 27 Jun 2022]).
The transmission lag D is –70 days for benzina and –84 days for diesel — both retail series
shifted before the crude oil changepoint, and all three shifts preceded the official invasion
by weeks to months. H0 is rejected for both fuels. The negative D values mean retail prices
began their structural upward shift ahead of the crude market itself, a pattern consistent
with anticipatory downstream pricing driven by early intelligence about supply risk.

**Iran-Israel War (13 Jun 2025)**

The Brent changepoint falls on 10 March 2025, 95 days before the official start of the
conflict, with a 95% credible interval of [10 Feb 2025 – 12 May 2025]. Diesel shows its
changepoint on the same date as Brent (10 March 2025, D = 0 days, H0 rejected). Benzina
shifts later, on 28 April 2025, with CI [14 Apr 2025 – 12 May 2025] (D = +49 days), which
falls above the 30-day threshold and is classified as compatible with logistics. The Brent
lag of –95 days remains the most striking result for this event: crude oil futures anticipated
the conflict by more than three months, consistent with escalating diplomatic and military
tensions in the region beginning in early 2025.

**Strait of Hormuz Closure (28 Feb 2026)**

The Brent changepoint falls on 1 December 2025, 89 days before the official closure,
with a 95% credible interval of [13 Oct 2025 – 2 Mar 2026]. Diesel shifts on 15 December 2025
(CI [13 Oct 2025 – 2 Mar 2026], D = +14 days, H0 rejected). Benzina shifts on 2 March 2026
(CI [16 Feb 2026 – 2 Mar 2026], D = +91 days), classified as compatible with logistics. The
CI lower bound for the Brent series (13 October 2025) precedes the official shock by more
than 4 months, suggesting that the anticipatory pricing signal in crude futures was visible
well before the event. Results for Hormuz are flagged as preliminary given the short
post-shock observation window (< 4 weeks available as of the analysis date).

**Summary of D values (tau_retail − tau_crude):**

| Event | tau_Brent | Benzina τ | Benzina D | Diesel τ | Diesel D | Verdict |
|---|---|---|---|---|---|---|
| Ukraine (Feb 2022) | 3 Jan 2022 | 25 Oct 2021 | −70 days | 11 Oct 2021 | −84 days | SPECULATION |
| Iran-Israel (Jun 2025) | 10 Mar 2025 | 28 Apr 2025 | +49 days | 10 Mar 2025 | 0 days | mixed |
| Hormuz (Feb 2026) | 1 Dec 2025 | 2 Mar 2026 | +91 days | 15 Dec 2025 | +14 days | partial |

### Granger Causality

The ADF test confirms non-stationarity of all three log-price series (p = 0.145 for Brent,
p = 0.141 for benzina, p = 0.275 for diesel); first differences are used throughout.

Brent Granger-causes benzina at all lags from 1 to 8 weeks (F ranging from 6.42 to 39.85,
all p < 0.0001). The F-statistic is maximized at lag 1 week (7 days): F = 39.85. Brent
Granger-causes diesel at the same lags with comparable strength (F from 5.41 to 34.91, all
p < 0.0001). The minimum significant lag — 7 days — is more than four times below the 30-day
physical threshold. H0 is rejected for both fuel types at all tested lags below 30 days
(1 to 4 weeks).

The uninterrupted significance from lag 1 through lag 8 for both fuels confirms that the
Brent-retail causal relationship is persistent and not a transient artifact of a single lag.

### Rockets and Feathers

| Fuel | β_up | β_down | R&F index | p-value |
|---|---|---|---|---|
| Benzina | 0.8184 | 0.2275 | 3.598 | < 0.0001 |
| Diesel | 1.1046 | 0.1304 | 8.470 | < 0.0001 |

Both asymmetries are highly significant. The R&F index for diesel (8.47) indicates that the
upward pass-through of Brent price increases is approximately 8.5 times larger than the
downward pass-through of Brent price decreases. For benzina, the asymmetry factor is 3.6.
These values are substantially larger than those reported in Borenstein et al. (1997) for
the US market (R&F ≈ 1.5–2.0), suggesting a more pronounced asymmetry in the Italian retail
context over this period.

### Additional Statistical Tests

**Kolmogorov-Smirnov**: KS = 1.000 for both benzina and diesel (Ukraine event), p < 0.000001.
A KS statistic of exactly 1 means the pre- and post-shock price distributions are completely
non-overlapping — not a single post-shock weekly price falls within the range of pre-shock
prices. This is one of the strongest possible results a two-sample KS test can produce.

**ANOVA**: F = 142.78 for benzina and F = 217.45 for diesel (Ukraine), p < 0.000001 for both.
Mean benzina rises from 697.1 (pre-shock) to 968.0 (acute shock) to 1066.0 (post-shock
normalization). Mean diesel rises from 694.8 to 1036.3 to 1155.4. The three-regime structure
is unambiguous and statistically indistinguishable from a step function.

**Chow Test**: F = 14.76 for benzina and F = 24.92 for diesel (Ukraine), p < 0.00001 for
both. For Hormuz: F = 80.62 for benzina and F = 73.23 for diesel, p < 0.000001. Structural
breaks are formally confirmed as discrete, not gradual, at the shock dates for all events
where sufficient post-shock data is available.

**CCF**: the optimal transmission lag is 0 weeks (0 days) for both benzina (r = 0.680) and
diesel (r = 0.689). A peak correlation at zero lag means that Brent and retail price changes
co-move within the same week, which is physically impossible if a 30-day supply chain
intervenes. H0 is rejected.

**Rolling Correlation**: the average Brent-retail correlation over the full sample is 0.759
for benzina and 0.730 for diesel. During the Ukraine war (March 2022), it spikes to 0.976
for benzina and 0.963 for diesel — an increase of roughly 30 percentage points. This pattern
confirms that the speculative transmission channel activates selectively under geopolitical
stress.

**Bootstrap CI on lag D**: for Ukraine, the 95% CI on D is [–140, +175] days — wide and
inconclusive for this event as a standalone test. For Hormuz, the CI is [–140, 0] days: the
upper bound is exactly 0, meaning that even the most conservative bootstrap resampling
confirms that retail prices moved no later than the official shock date. H0 is rejected for
Hormuz by this test.

**RMSE improvement**: the piecewise model reduces RMSE by 26.0% for benzina and 34.1% for
diesel (Ukraine), and by 60.8% for benzina and 56.9% for diesel (Hormuz). Improvements of
this magnitude confirm that the two-regime model captures a genuine structural feature of
the data and is not overfitting to noise.

---

## Requirements

Python 3.9 or higher is required. Dependencies are listed in requirements.txt:

```
yfinance
pymc
pytensor
arviz
statsmodels
pandas
numpy
matplotlib
seaborn
scipy
requests
```

---

## Setup and Usage

**1. Clone the repository**

```bash
git clone https://github.com/your-username/fuel-price-speculation-italy.git
cd fuel-price-speculation-italy
```

**2. Create and activate a virtual environment**

Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the full pipeline**

```bash
python 05_run_all.py
```

Or run individual scripts in order:

```bash
python 01_data_pipeline.py
python 02_changepoint_detection.py
python 03_granger_causality.py
python 04_rocket_feather.py
python 06_statistical_tests.py
```

**5. Deactivate the virtual environment when done**

```bash
deactivate
```

All outputs are written to data/ and plots/, created automatically on first run.

---

## Data Sources

| Variable | Source | Frequency | Currency |
|---|---|---|---|
| Brent crude oil price | Yahoo Finance (ticker: BZ=F) via yfinance | Daily | USD → EUR (via EURUSD=X) |
| EUR/USD exchange rate | Yahoo Finance (ticker: EURUSD=X) via yfinance | Daily (weekly avg) | — |
| Italian retail fuel prices | EU Weekly Oil Bulletin (European Commission) | Weekly | EUR/litre (native) |

Retail prices used are the pre-tax series (sheet "Prices wo taxes"), specifically columns
IT_price_wo_tax_euro95 (benzina) and IT_price_wo_tax_diesel. Using pre-tax prices isolates
the commodity component from the tax component, which is fixed by law and cannot be
speculative by construction.

---

## Output Files

| File | Description |
|---|---|
| plots/01a_brent.png | Brent crude time series with war event markers |
| plots/01b_benzina.png | Italian gasoline retail price with war event markers |
| plots/01c_diesel.png | Italian diesel retail price with war event markers |
| plots/02_Ucraina_Feb_2022_brent.png | Piecewise regression — Ukraine, Brent |
| plots/02_Ucraina_Feb_2022_benzina.png | Piecewise regression — Ukraine, benzina |
| plots/02_Ucraina_Feb_2022_diesel.png | Piecewise regression — Ukraine, diesel |
| plots/02_Iran-Israele_Giu_2025_brent.png | Piecewise regression — Iran-Israel, Brent |
| plots/02_Iran-Israele_Giu_2025_benzina.png | Piecewise regression — Iran-Israel, benzina |
| plots/02_Iran-Israele_Giu_2025_diesel.png | Piecewise regression — Iran-Israel, diesel |
| plots/02_Hormuz_Feb_2026_brent.png | Piecewise regression — Hormuz, Brent |
| plots/02_Hormuz_Feb_2026_benzina.png | Piecewise regression — Hormuz, benzina |
| plots/02_Hormuz_Feb_2026_diesel.png | Piecewise regression — Hormuz, diesel |
| plots/03_granger_benzina.png | Granger p-values by lag, gasoline |
| plots/03_granger_diesel.png | Granger p-values by lag, diesel |
| plots/03_granger_combined.png | Granger p-values for benzina and diesel side by side (paper figure) |
| plots/04_rf_scatter_benzina.png | Rockets and Feathers scatter, gasoline |
| plots/04_rf_scatter_diesel.png | Rockets and Feathers scatter, diesel |
| plots/04_rf_norm_benzina.png | Normalized price index, gasoline vs Brent |
| plots/04_rf_norm_diesel.png | Normalized price index, diesel vs Brent |
| plots/06_statistical_tests.png | KS distributions, CCF, rolling correlation, bootstrap CI |
| data/table1_changepoints.csv | Changepoint dates τ, 95% posterior credible intervals, slopes, doubling times (Table 1) |
| data/lag_results.csv | D = tau_retail - tau_crude for each event and fuel type |
| data/granger_benzina.csv | Granger test results by lag for gasoline |
| data/granger_diesel.csv | Granger test results by lag for diesel |
| data/rockets_feathers_results.csv | Asymmetry test: beta_up, beta_down, R&F index |
| data/ks_results.csv | Kolmogorov-Smirnov test results |
| data/anova_results.csv | ANOVA results across three price regimes |
| data/chow_results.csv | Chow structural break test results |
| data/bootstrap_ci.csv | Bootstrap 95% CI on lag D (complementary to CI on τ in script 02) |
| data/fit_quality.csv | RMSE and MAE for piecewise vs simple linear regression |

---

## Methodological Notes

### Currency consistency

All price series are expressed in EUR throughout the analysis. Brent crude (quoted in USD/barrel
on futures markets) is converted using the contemporaneous weekly EUR/USD mid-rate. This ensures
that apparent price movements are not artifacts of USD/EUR fluctuations.

### Smoothing symmetry

Previous versions applied a 4-week rolling average to retail prices but not to Brent (which
received only a 7-day daily rolling average before resampling). This asymmetry introduced an
artificial lag of approximately 2 weeks into transmission estimates, biasing results against
rejecting H0. The current version applies no additional smoothing to either weekly series after
resampling.

### Bayesian credible intervals on the changepoint date

The primary uncertainty measure in this analysis is the 95% Bayesian credible interval on
the changepoint date τ, derived from the marginal posterior distribution of τ estimated via
MCMC (PyMC/NUTS) in script 02. Unlike a frequentist confidence interval, the credible
interval has a direct probabilistic interpretation: there is a 95% posterior probability
that the true structural break falls within the reported interval. The block structure of
price autocorrelation is handled implicitly through the model's likelihood and prior
specification rather than through resampling.

The CI answers the question: *on what date did the structural shift in prices occur?* A CI
whose upper bound precedes the official shock date constitutes strong evidence of anticipatory
pricing independently of the exact point estimate. Credible intervals on slopes b1 and b2
are similarly extracted from their marginal posteriors and visualized as shaded bands on
each regression plot.

The complementary CI on the transmission lag D (script 06, block bootstrap) answers a
different question — *how much time elapsed between the crude oil shift and the retail
shift?* — and is subject to wider uncertainty because it compounds the estimation error of
two independent changepoints.

### Hormuz as a preliminary event

The Hormuz closure (28 February 2026) falls close to the end of the data collection window.
At the time of analysis, fewer than 4 post-shock weekly observations are available for retail
prices. Changepoint and slope estimates for this event carry wide credible intervals and
should be interpreted as preliminary. The Brent credible interval spans [13 Oct 2025 – 2 Mar 2026]
with a point estimate of 1 December 2025, confirming that the crude market shifted well before
the official event. For retail fuels, the diesel changepoint (15 Dec 2025, D = +14 days, CI
entirely below 30 days) is robust to the short post-shock window and supports H0 rejection.
The benzina changepoint (2 Mar 2026, D = +91 days) falls after the shock and is consistent
with logistics, but should be treated with caution given data scarcity. All post-shock slope
estimates for this event should be considered indicative pending additional data.

---

## References

- Bacon RW. "Rockets and Feathers: The Asymmetric Speed of Adjustment of UK Retail Gasoline
  Prices to Cost Changes." Energy Economics, 1991.

- Borenstein S, Cameron AC, Gilbert R. "Do Gasoline Prices Respond Asymmetrically to Crude
  Oil Price Changes?" Quarterly Journal of Economics, 1997.

- Meyer J, von Cramon-Taubadel S. "Asymmetric Price Transmission: A Survey."
  Journal of Agricultural Economics, 2004.

- International Energy Agency. "Italy Oil Supply and Demand." IEA, 2022.

---

## License

MIT License. All data used is publicly available from the sources listed above.
All results are fully reproducible using the code and instructions in this repository.