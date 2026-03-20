# Fuel Price Speculation in Italy: A Statistical Analysis of War-Driven Price Shocks

This project investigates whether the rapid increase in retail fuel prices (gasoline, diesel)
following geopolitical shocks — specifically the Russian invasion of Ukraine (February 2022)
and the Strait of Hormuz closure (February 2026) — is compatible with the physical supply
chain timeline, or whether it constitutes anticipatory pricing behavior (speculation).

The methodology is inspired by Casini & Roccetti (BMJ Open, 2021), who used Bayesian
piecewise linear regression to detect changepoints in COVID-19 case growth rates relative
to school reopening dates in Italy. The same logic is applied here: if the changepoint in
retail prices occurs within days of the shock to crude oil, it cannot be explained by the
physical supply chain alone.

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

The project uses four sequential analysis steps:

1. **Data pipeline** — Downloads Brent crude prices (daily, via yfinance) and Italian retail
   fuel prices (weekly, via EU Weekly Oil Bulletin). Applies 7-day and 4-week rolling
   averages and log transformation, following the preprocessing in Casini & Roccetti (2021).

2. **Changepoint detection** — Fits a Bayesian piecewise linear regression (ruptures, PELT
   algorithm) to both the Brent and retail price series for each shock event. Estimates the
   changepoint date tau for each series and computes D = tau_retail - tau_crude. If D < 30
   days, H0 is rejected. Reports slopes b1 (pre-shock) and b2 (post-shock), and the price
   doubling time for each segment, analogous to Table 1 in Casini & Roccetti (2021).

3. **Granger causality** — Tests whether Brent prices Granger-cause retail prices at lags
   from 1 to 8 weeks. A significant result at lag < 4 weeks (< 30 days) constitutes
   additional evidence against H0. Uses first-differenced log prices to ensure stationarity
   (verified by Augmented Dickey-Fuller test).

4. **Rockets and Feathers** — Tests the asymmetric price transmission hypothesis (Bacon,
   1991): whether prices rise faster when crude increases than they fall when crude decreases.
   Estimates separate pass-through coefficients for positive (beta_up) and negative
   (beta_down) crude oil changes via OLS. Tests the null of symmetry using a t-test on
   the difference beta_up - beta_down.

---

## Requirements

```
pip install yfinance ruptures statsmodels pandas numpy matplotlib seaborn scipy requests
```

Python 3.9 or higher is required.

---

## Usage

Run the full pipeline with:

```bash
python 05_run_all.py
```

Or run individual scripts in order:

```bash
python 01_data_pipeline.py
python 02_changepoint_detection.py
python 03_granger_causality.py
python 04_rocket_feather.py
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
| `data/table1_changepoints.csv` | Changepoint estimates, slopes, and doubling times (Table 1) |
| `data/lag_results.csv` | D = tau_retail - tau_crude for each event and fuel type |
| `data/granger_benzina.csv` | Granger test results by lag for gasoline |
| `data/granger_diesel.csv` | Granger test results by lag for diesel |
| `data/rockets_feathers_results.csv` | Asymmetry test results |

---

## References

- Casini L, Roccetti M. "Reopening Italy's schools in September 2020: a Bayesian estimation
  of the change in the growth rate of new SARS-CoV-2 cases." BMJ Open, 2021. [methodology]

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
