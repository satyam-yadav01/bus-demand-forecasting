# Bus Passenger Demand Forecasting

An end-to-end machine learning project that forecasts passenger demand for scheduled bus departures using historical demand patterns, route information and temporal features. The system supports capacity-planning decisions by identifying departures where predicted passenger demand may exceed planned bus capacity.

## 1. Problem Statement

Public transportation operators need to estimate passenger demand in advance to plan bus capacity and scheduling efficiently.

Insufficient capacity can result in overcrowding and passengers being unable to board, while excessive capacity can lead to inefficient resource utilization.

This project develops a machine learning pipeline to forecast passenger demand for individual bus departures **seven days before the scheduled departure**.

The predicted demand is compared with planned bus capacity to identify potential capacity shortages.

## 2. Project Objectives

- Generate and analyze a synthetic bus ridership dataset.
- Perform data cleaning, exploratory data analysis and feature engineering.
- Prevent temporal data leakage during historical feature construction.
- Develop a historical baseline and machine learning models.
- Evaluate models using chronological train, validation and test splits.
- Analyze model performance across routes and peak/off-peak periods.
- Build a FastAPI-based prediction service.
- Identify potential capacity shortages using predicted demand.

## 3. Dataset

The project uses a synthetically generated dataset because real-world operational ridership data was not available.

| Attribute | Description |
|---|---|
| Total records | 36,500 |
| Number of routes | 5 |
| Date range | 1 January 2024 – 30 December 2025 |
| Scheduled departures | 10 per day |
| Target variable | `passenger_demand` |
| Planned capacity | 40 or 60 passengers |

### Dataset generation

Passenger demand is generated using a Negative Binomial distribution with route-specific demand patterns, temporal variations, weekend effects, seasonal multipliers and occasional event-related demand spikes.

A fixed random seed ensures reproducibility.

The dataset includes:
- Route identifiers.
- Scheduled departure timestamps.
- Planned bus capacity.
- Passenger demand.
- Trip completion timestamps.
- Actual passenger counts.

**Target definition:** `passenger_demand` represents unconstrained passenger demand, whereas `actual_passenger_count` represents passengers carried after applying the capacity limit.

`actual_passenger_count = min(passenger_demand, planned_capacity)`

The model predicts unconstrained demand rather than passengers actually carried.

## 4. Exploratory Data Analysis

EDA was performed to understand demand distributions and identify patterns across routes and time periods.

The analysis includes:
- Passenger demand distribution.
- Route-wise demand comparison.
- Hourly demand patterns.
- Weekend and seasonal variations.
- Peak versus off-peak demand.

Generated visualizations are available in `reports/eda_plots/`.

High-demand observations were not automatically removed as outliers because some represent meaningful event-related demand spikes.

## 5. Feature Engineering

The prediction unit is one scheduled bus departure.

The forecast horizon is seven days before departure.

For each departure:

`prediction_cutoff = departure_datetime - 7 days`

Historical observations are eligible only when:

`completion_datetime < prediction_cutoff`

The strict inequality ensures that observations completed exactly at the cutoff or afterward are excluded.

### Features used

**Temporal features**
- `departure_hour`
- `day_of_week`
- `month`
- `is_weekend`
- `season`
- `peak_period`

**Operational features**
- `route_id`
- `planned_capacity`

**Historical demand features**
- `hist_route_time_avg`
- `route_time_history_count`
- `hist_route_avg`
- `route_history_count`
- `hist_global_time_avg`

Historical features use only observations eligible at the corresponding prediction cutoff.

Features that would reveal future outcomes, such as trip completion time, actual passenger count and trip duration, are not used as model inputs.

## 6. Data Splitting Strategy

A chronological split was used instead of random splitting to simulate real-world forecasting.

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 25,550 | 70% |
| Validation | 5,450 | 15% |
| Testing | 5,500 | 15% |
| **Total** | **36,500** | **100%** |

All departures on the same calendar date are kept in the same partition.

The training dataset is used for fitting models and preprocessing. Validation is used for model comparison, while the final test set is reserved for the final performance evaluation.

## 7. Models Developed

Three approaches were evaluated.

### Historical Baseline

A historical-average baseline uses the following fallback hierarchy:

1. Route and departure-hour historical average, when sufficient history is available.
2. Route-level historical average.
3. Global departure-hour historical average.
4. No prediction when no eligible history exists.

### Ridge Regression

A regularized linear regression model used as a machine learning benchmark.

Preprocessing includes:
- Median imputation for numerical features.
- Standard scaling for numerical features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features.

### XGBoost Regressor

A gradient-boosted decision tree model used to capture nonlinear relationships between route, temporal and historical demand features.

The trained preprocessing and model pipeline is saved together to ensure consistent inference.

## 8. Model Evaluation

The following metrics are used:

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **WAPE:** Weighted Absolute Percentage Error.

### Validation performance

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Historical Baseline | 9.629 | 12.569 | 28.34% |
| Ridge Regression | 8.569 | 11.228 | 25.22% |
| XGBoost | 8.283 | 10.938 | 24.38% |

XGBoost achieved the best overall validation performance and also outperformed Ridge across the evaluated routes and peak/off-peak groups.

### Final test performance

The selected XGBoost pipeline was evaluated on the untouched chronological test dataset.

| Metric | Test Result |
|---|---:|
| MAE | 8.568 |
| RMSE | 11.453 |
| WAPE | 24.23% |

The final test evaluation contains 5,500 records.

### Test performance by route

| Route | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| R01 | 7.731 | 9.977 | 24.46% |
| R02 | 9.691 | 13.251 | 23.29% |
| R03 | 6.468 | 8.246 | 26.49% |
| R04 | 10.944 | 14.514 | 24.10% |
| R05 | 8.004 | 10.104 | 23.71% |

### Peak and off-peak test performance

| Period | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Off-Peak | 7.409 | 9.810 | 24.43% |
| Peak | 9.727 | 12.889 | 24.08% |

### Feature importance

XGBoost feature importance indicates that historical route-hour demand, peak-period classification and route identity are among the most influential model features.

The complete feature importance results are saved in `reports/xgboost_feature_importance.csv`.

Feature importance describes model behavior and should not be interpreted as causal relationships.

## 9. Capacity Risk Analysis

Predicted demand is compared with planned capacity to identify potential shortages.

A capacity shortage is flagged when:

`predicted_demand > planned_capacity`

The final test capacity-risk results are:

| Metric | Result |
|---|---:|
| Actual capacity shortages | 1,026 |
| Predicted capacity shortages | 842 |
| True positives | 530 |
| False positives | 312 |
| False negatives | 496 |
| True negatives | 4,162 |
| Precision | 62.9% |
| Recall | 51.7% |

The model identifies some capacity shortages but misses a substantial proportion of actual shortage cases.

Therefore, the capacity-risk output should be treated as a decision-support signal requiring human review, not an automatic bus-allocation instruction.

## 10. FastAPI Inference Service

A FastAPI application provides an interface for generating predictions using the saved XGBoost pipeline.

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Application and model health |
| POST | `/predict` | Predict passenger demand |

### Run the API

Activate the virtual environment and execute:

```bash
python -m uvicorn src.api.app:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

### Example prediction request

```json
{
  "route_id": "R01",
  "departure_datetime": "2025-10-15T08:00:00",
  "planned_capacity": 40,
  "season": "Monsoon",
  "peak_period": "Peak",
  "hist_route_time_avg": 42.5,
  "route_time_history_count": 4,
  "hist_route_avg": 38.2,
  "route_history_count": 20,
  "hist_global_time_avg": 35.7
}
```

The API returns:
- Predicted passenger demand.
- Planned capacity.
- Capacity status.
- Estimated capacity shortfall.
- Estimated spare capacity.
- An operational warning.

Historical features must be supplied by the caller using eligible historical observations. The current API accepts these features as request fields; automatic historical feature-store integration is not implemented.

## 11. Project Structure

```text
bus-demand-forecasting/
│
├── artifacts/
│   ├── feature_store/
│   └── models/
│       ├── ridge_pipeline.joblib
│       └── xgboost_pipeline.joblib
│
├── data/
│   ├── raw/
│   │   └── bus_demand.csv
│   └── processed/
│       ├── bus_demand_features.csv
│       └── splits/
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
│
├── reports/
│   ├── eda_plots/
│   ├── model_comparison_validation.csv
│   ├── xgboost_feature_importance.csv
│   ├── final_test_metrics.csv
│   ├── final_test_route_metrics.csv
│   ├── final_test_period_metrics.csv
│   ├── final_test_capacity_metrics.csv
│   └── evaluation_report.md
│
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── generate.py
│   │   └── split_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── test.py
│   └── eda.py
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_splits.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 12. Installation and Execution

### Requirements

- Python 3.11 or later, subject to dependency compatibility.
- pip
- Virtual environment support.

### Setup

Clone or download the project repository and navigate to its root directory.

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run automated tests

```bash
python -m pytest tests/ -v
```

The current project test suite contains 16 passing tests covering:
- API endpoints and request validation.
- Historical feature eligibility and leakage prevention.
- Chronological splitting and date separation.

### Run model inference API

```bash
python -m uvicorn src.api.app:app --reload
```

## 13. Limitations and Responsible Use

1. **Synthetic data:** The dataset is generated rather than collected from real transportation operations. Model performance does not establish real-world forecasting accuracy.

2. **Unusual demand events:** Sudden events and unexpected disruptions can create demand spikes that are difficult to predict without reliable advance information.

3. **Capacity shortage detection:** The final test recall is 51.7%, meaning many actual capacity shortages are not detected.

4. **Historical feature availability:** The current API requires historical demand features to be supplied by the caller. A production implementation should integrate a validated historical feature store.

5. **New routes:** The model has limited ability to generalize to routes not represented in its training dataset.

6. **Human oversight:** Predictions should support planning decisions. Operational staff should review forecasts alongside local conditions, service disruptions and other available information.

7. **Data privacy:** Real deployment should follow appropriate data minimization, access control and privacy practices.

## 14. Future Improvements

- Integrate real-world passenger-count and ticketing data.
- Build automatic historical feature retrieval.
- Improve event and disruption-aware forecasting.
- Explore probabilistic prediction and prediction intervals.
- Improve capacity-shortage recall while considering operational costs.
- Add monitoring for data drift and model performance.
- Develop a dashboard for route-level demand and capacity planning.

## 15. Conclusion

This project demonstrates an end-to-end passenger-demand forecasting workflow, from synthetic data generation and leakage-aware feature engineering to model evaluation and API-based inference.

XGBoost achieved a test WAPE of 24.23% on the chronological holdout dataset. The API successfully loads the trained model and produces demand predictions and capacity-risk estimates.

The project provides a reproducible prototype for exploring data-driven bus capacity planning, while recognizing the limitations of synthetic data and the need for human oversight.

---

**Project:** Bus Passenger Demand Forecasting  
**Model:** XGBoost Regressor  
**Prediction Horizon:** 7 days before scheduled departure  
**API Framework:** FastAPI  
**Test Records:** 5,500  
**Automated Tests:** 16 passed