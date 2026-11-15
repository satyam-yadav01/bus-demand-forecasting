
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "xgboost_pipeline.joblib"
)


# ---------------------------------------------------------
# 2. FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="Bus Passenger Demand Forecasting API",
    description=(
        "Predict passenger demand for scheduled bus departures "
        "using a trained XGBoost model."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# 3. LOAD MODEL
# ---------------------------------------------------------

model = None


@app.on_event("startup")
def load_model():

    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded successfully.")


# ---------------------------------------------------------
# 4. REQUEST SCHEMA
# ---------------------------------------------------------

class PredictionRequest(BaseModel):

    route_id: str = Field(
        ...,
        description="Route identifier, e.g. R01",
    )

    departure_datetime: datetime = Field(
        ...,
        description="Scheduled departure date and time",
    )

    planned_capacity: int = Field(
        ...,
        gt=0,
        description="Planned passenger capacity",
    )

    season: str = Field(
        ...,
        description="Season category used during model training",
    )

    peak_period: str = Field(
        ...,
        description="Peak or Off-Peak",
    )

    hist_route_time_avg: float | None = Field(
        None,
        ge=0,
        description="Historical average demand for the same route and hour",
    )

    route_time_history_count: int = Field(
        0,
        ge=0,
        description="Number of eligible route-hour historical observations",
    )

    hist_route_avg: float | None = Field(
        None,
        ge=0,
        description="Historical average demand for the route",
    )

    route_history_count: int = Field(
        0,
        ge=0,
        description="Number of eligible route historical observations",
    )

    hist_global_time_avg: float | None = Field(
        None,
        ge=0,
        description="Historical average demand across routes for the same hour",
    )


# ---------------------------------------------------------
# 5. HEALTH CHECK
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Bus Passenger Demand Forecasting API",
        "status": "running",
        "model": "XGBoost",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ---------------------------------------------------------
# 6. PREDICTION ENDPOINT
# ---------------------------------------------------------

@app.post("/predict")
def predict_demand(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    try:

        departure = request.departure_datetime

        # Derive calendar features from scheduled departure.
        features = {
            "departure_hour": departure.hour,
            "day_of_week": departure.weekday(),
            "month": departure.month,
            "is_weekend": int(departure.weekday() >= 5),
            "planned_capacity": request.planned_capacity,

            "hist_route_time_avg": request.hist_route_time_avg,
            "route_time_history_count": request.route_time_history_count,

            "hist_route_avg": request.hist_route_avg,
            "route_history_count": request.route_history_count,

            "hist_global_time_avg": request.hist_global_time_avg,

            "route_id": request.route_id,
            "season": request.season,
            "peak_period": request.peak_period,
        }

        # Keep the same feature order as model training.
        feature_columns = [
            "departure_hour",
            "day_of_week",
            "month",
            "is_weekend",
            "planned_capacity",
            "hist_route_time_avg",
            "route_time_history_count",
            "hist_route_avg",
            "route_history_count",
            "hist_global_time_avg",
            "route_id",
            "season",
            "peak_period",
        ]

        input_df = pd.DataFrame(
            [features],
            columns=feature_columns,
        )

        # Generate prediction.
        prediction = float(model.predict(input_df)[0])

        # Demand cannot be negative.
        prediction = max(0.0, prediction)

        # Capacity-risk assessment.
        shortage = prediction > request.planned_capacity

        capacity_difference = (
            prediction - request.planned_capacity
        )

        if shortage:
            capacity_status = "Insufficient capacity"
            estimated_shortfall = capacity_difference
            estimated_spare_capacity = 0.0
        else:
            capacity_status = "Capacity appears sufficient"
            estimated_shortfall = 0.0
            estimated_spare_capacity = -capacity_difference

        return {
            "route_id": request.route_id,
            "departure_datetime": departure.isoformat(),
            "predicted_passenger_demand": round(prediction, 2),
            "planned_capacity": request.planned_capacity,
            "capacity_status": capacity_status,
            "estimated_shortfall": round(
                estimated_shortfall, 2
            ),
            "estimated_spare_capacity": round(
                estimated_spare_capacity, 2
            ),
            "warning": (
                "Prediction is an estimate. Review operational "
                "conditions before making capacity decisions."
            ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
