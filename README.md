# Predictive Air Quality Modeling: ARIMA vs. LSTM with Apriori Pattern Mining

A hybrid time-series forecasting and pattern-mining project on hourly urban air pollution data. Built to compare a classical statistical model against a deep learning model for pollutant forecasting, and to surface actionable seasonal pollution patterns via association rule mining.

## Overview

Urban air quality monitoring is typically **reactive** — hazards are flagged only after they occur. This project explores a **predictive** approach using the [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality) (9,357 hourly multisensor readings, March 2004–February 2005, Italy).

**Goals:**
1. Forecast hourly CO concentration using both a linear (ARIMA) and non-linear (LSTM) model
2. Quantify temporal dependencies between pollutants
3. Extract actionable association rules describing high-pollution conditions
4. Compare ARIMA and LSTM performance directly

## Methods

| Stage | Approach |
|---|---|
| **Preprocessing** | Sentinel value (`-200`) handling → NaN → imputation; IQR-based outlier capping; `MinMaxScaler` rescaling; 24-hour lag feature engineering |
| **ARIMA** | Augmented Dickey-Fuller stationarity test; grid search over `(p, q)` by AIC; best model: **ARIMA(3,0,3)** |
| **LSTM** | 2-layer LSTM with dropout (0.2), 24-hour lookback window, early stopping |
| **Apriori** | Pollutant levels discretized into Low/Medium/High; frequent itemsets and association rules mined with support/confidence thresholds |

## Results

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| ARIMA(3,0,3) | 1.3255 | 1.0877 | 138.39% | -0.0338 |
| **LSTM** | **0.5913** | **0.4175** | **37.69%** | **0.7935** |

The LSTM model substantially outperformed ARIMA on every metric. ARIMA's negative R² indicates it performed worse than a naive mean predictor — a reflection of how volatile and non-linear the hourly CO signal is, which a purely linear model with fixed autoregressive terms struggles to capture.

**Key association rules discovered:**
- `Season_Autumn, CO_High → NOx_High` (confidence 0.98, lift 2.88) — suggests shared seasonal traffic/heating emission sources
- `NOx_High, C6H6_High → CO_High` (confidence 0.91) — indicates simultaneous emission-source spikes

*Note: because CO, NOx, NO2, and C6H6 are strongly correlated (r = 0.85–0.98) in this dataset, some association rules likely reflect this underlying multicollinearity rather than independent causal relationships — worth keeping in mind when interpreting "actionable" rules.*

## Repository Contents

- `air_quality_analysis.ipynb` — full analysis notebook (data loading → preprocessing → ARIMA → LSTM → Apriori → comparison dashboard)
- `DM_Report.pdf` — written project report

## Tech Stack

`pandas` · `numpy` · `matplotlib` / `seaborn` · `statsmodels` (ARIMA) · `tensorflow` / `keras` (LSTM) · `mlxtend` (Apriori) · `scikit-learn`

## Future Work

- Incorporate exogenous meteorological/traffic data (wind speed, traffic volume) into the LSTM
- Explore online/incremental learning for real-time adaptive forecasting
- Validate association rules against multicollinearity to isolate genuinely independent co-occurrence patterns

---
*Course project — Data Mining, University of Management and Technology*
