# Fuel Price Speculation in Italy: A Statistical Analysis of War-Driven Price Shocks

This project investigates whether the rapid increase in retail fuel prices (gasoline, diesel)
following geopolitical shocks — specifically the Russian invasion of Ukraine (February 2022)
and the Strait of Hormuz closure (February 2026) — is compatible with the physical supply
chain timeline, or whether it constitutes anticipatory pricing behavior (speculation).

---

## Null Hypothesis

H0: The lag between a crude oil price shock (Brent) and the corresponding change in Italian
retail fuel prices is >= 30 days, consistent with the physical supply chain
(maritime transport + refining + storage + distribution).

Rejecting H0 implies that retail prices respond faster than the logistics chain allows,
which is evidence of speculative or anticipatory pricing.

The 30-day threshold is derived from the Italian import profile: over 90% of crude oil
arrives by sea (IEA, 2022), with tanker transit times of 7-21 days from the Persian Gulf
and 5-10 days from North Africa, plus refining and distribution time.

---

## Methods

The project uses six sequential analysis steps:

1. **Data pipeline** — Downloads Brent crude prices (daily, via yfinance) and Italian retail
   fuel prices (weekly, via EU Weekly Oil Bulletin). Applies 7-day and 4-week rolling
   averages and log transformation.

2. **Changepoint detection** — Fits a Bayesian piecewise linear regression (ruptures, PELT
   algorithm) to both the Brent and retail price series for each shock event. Estimates the
   changepoint date tau for each series and computes D = tau_retail - tau_crude. If D < 30
   days, H0 is rejected. Reports slopes b1 (pre-shock) and b2 (post-shock), and the price
   doubling time for each segment.

3. **Granger causality** — Tests whether Brent prices Granger-cause retail prices at lags
   from 1 to 8 weeks. A significant result at lag < 4 weeks (< 30 days) constitutes
   additional evidence against H0. Uses first-differenced log prices to ensure stationarity
   (verified by Augmented Dickey-Fuller test).

4. **Rockets and Feathers** — Tests the asymmetric price transmission hypothesis (Bacon,
   1991): whether prices rise faster when crude increases than they fall when crude decreases.
   Estimates separate pass-through coefficients for positive (beta_up) and negative
   (beta_down) crude oil changes via OLS. Tests the null of symmetry using a t-test on
   the difference beta_up - beta_down.

5. **Additional statistical tests** — Seven complementary tests that corroborate H0 rejection
   from independent angles:
   - Kolmogorov-Smirnov: tests whether the full price distribution changes after the shock,
     not just the mean. A KS statistic close to 1 means the pre- and post-shock distributions
     share almost no overlap.
   - One-way ANOVA: compares price means across three periods (pre-shock, acute shock,
     post-shock normalization). A large F-statistic confirms that between-period variance
     dominates within-period variance, i.e. the three regimes are statistically distinct.
   - Chow Test: formal structural break test on the regression. Tests whether the slope and
     intercept of the price series change significantly at the shock date. A significant result
     means the break is not a gradual trend but a discrete snap in the data-generating process.
   - Cross-Correlation Function (CCF): estimates the optimal transmission lag between Brent
     and retail prices across lags 0-12 weeks. The lag at maximum correlation is the empirical
     speed of price transmission. A peak lag below 4 weeks (30 days) directly rejects H0.
   - Rolling Correlation: computes the Brent-retail correlation over a 12-week moving window.
     Near-zero correlation in normal periods that spikes sharply during war events reveals that
     the speculative transmission channel activates only under geopolitical stress.
   - Bootstrap Confidence Intervals: non-parametric block bootstrap (500 iterations) on the
     lag D to quantify estimation uncertainty without distributional assumptions. A 95% CI
     entirely below 30 days constitutes strong evidence against H0.
   - RMSE / MAE: compares fit quality of the piecewise regression against a simple linear
     model. A meaningful improvement validates the changepoint as a genuine structural feature
     of the data rather than a statistical artifact.

---

## Key Results

The following results are obtained with real data from yfinance and the EU Weekly Oil Bulletin.
With the fallback simulated data the direction of all results is preserved.

- Changepoint detection: D = 0 days for Ukraine (benzina and diesel change slope
  simultaneously with Brent). H0 rejected.
- Granger causality: Brent Granger-causes retail prices at lag 1 week (7 days), p < 0.001.
  Minimum significant lag is well below the 30-day threshold. H0 rejected.
- Rockets and Feathers: asymmetry is statistically significant for both gasoline
  (p = 0.015) and diesel (p = 0.023). Prices rise faster than they fall.
- Kolmogorov-Smirnov: KS = 0.926, p < 0.000001. Pre- and post-shock distributions are
  almost entirely non-overlapping. H0 rejected.
- ANOVA: F = 834, p < 0.000001. The three price regimes are statistically distinct. H0
  rejected.
- Chow Test: F = 17.5, p < 0.00001 for Ukraine. Structural break is formally confirmed at
  the shock date. H0 rejected.
- CCF: optimal transmission lag is 3 weeks (21 days), below the 30-day physical threshold.
  H0 rejected.
- Rolling correlation: near zero (-0.04) in normal periods, spikes to 0.68 during the
  Ukraine war. The speculative channel opens only during geopolitical shocks.
- Bootstrap CI (Hormuz): 95% CI falls entirely below zero, meaning prices began moving
  before the official shock date. Consistent with anticipatory pricing.
- RMSE improvement: piecewise model fits 24-26% better than a linear model for Ukraine,
  confirming the changepoint is a genuine structural feature.

---

## Requirements

Python 3.9 or higher is required. Dependencies are listed in `requirements.txt`:

```
yfinance
ruptures
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

All outputs (CSV tables and PNG plots) are written to `data/` and `plots/` respectively,
which are created automatically on first run.

---

## Data Sources

| Variable | Source | Frequency |
|---|---|---|
| Brent crude oil price | Yahoo Finance (ticker: BZ=F) via yfinance | Daily |
| Italian retail fuel prices | EU Weekly Oil Bulletin (European Commission) | Weekly |
| Alternative IT retail prices | MASE/MIMIT Osservatorio Prezzi Carburanti | Weekly |

To replace the EU Oil Bulletin data with official Italian government data, download the
weekly CSV from https://carburanti.mise.gov.it/ and update the file path in
`01_data_pipeline.py`.

---

## Output Files

| File | Description |
|---|---|
| `plots/01_overview.png` | Time series of Brent and retail prices with war event markers |
| `plots/02_changepoints.png` | Piecewise regression plots for each event and series |
| `plots/03_granger.png` | Granger causality p-values by lag, with 30-day threshold |
| `plots/04_rockets_feathers.png` | Asymmetric pass-through scatter and normalized price series |
| `plots/06_statistical_tests.png` | KS distributions, CCF, rolling correlation, bootstrap CI |
| `data/table1_changepoints.csv` | Changepoint estimates, slopes, and doubling times (Table 1) |
| `data/lag_results.csv` | D = tau_retail - tau_crude for each event and fuel type |
| `data/granger_benzina.csv` | Granger test results by lag for gasoline |
| `data/granger_diesel.csv` | Granger test results by lag for diesel |
| `data/rockets_feathers_results.csv` | Asymmetry test results |
| `data/ks_results.csv` | Kolmogorov-Smirnov test results |
| `data/anova_results.csv` | ANOVA results across three price regimes |
| `data/chow_results.csv` | Chow structural break test results |
| `data/bootstrap_ci.csv` | Bootstrap 95% confidence intervals on lag D |
| `data/fit_quality.csv` | RMSE and MAE for piecewise vs simple linear regression |

---

## References

- Bacon RW. "Rockets and Feathers: The Asymmetric Speed of Adjustment of UK Retail Gasoline
  Prices to Cost Changes." Energy Economics, 1991. [asymmetry framework]

- Borenstein S, Cameron AC, Gilbert R. "Do Gasoline Prices Respond Asymmetrically to Crude
  Oil Price Changes?" Quarterly Journal of Economics, 1997. [theoretical basis for H0]

- Meyer J, von Cramon-Taubadel S. "Asymmetric Price Transmission: A Survey."
  Journal of Agricultural Economics, 2004. [literature review]

---

## License

MIT License. All data used is publicly available from the sources listed above.
All results are fully reproducible using the code and instructions in this repository.