
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.evaluate import calculate_metrics


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "xgboost_pipeline.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. MODEL FEATURES
# ---------------------------------------------------------

NUMERIC_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "route_id",
    "season",
    "peak_period",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "passenger_demand"


# ---------------------------------------------------------
# 3. LOAD TEST DATA AND SAVED MODEL
# ---------------------------------------------------------

def load_test_data():

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    test_df = pd.read_csv(TEST_DATA_PATH)

    return test_df


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ---------------------------------------------------------
# 4. EVALUATE FINAL TEST PERFORMANCE
# ---------------------------------------------------------

def evaluate_test_set():

    print("\nLoading untouched test dataset...")

    test_df = load_test_data()

    print("Loading saved XGBoost model...")

    model = load_model()

    print(f"\nTest records: {len(test_df)}")

    # Check required columns.
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Prepare features and target.
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Generate predictions using the saved pipeline.
    predictions = model.predict(X_test)

    # Passenger demand cannot be negative.
    predictions = np.maximum(predictions, 0)

    # -----------------------------------------------------
    # OVERALL TEST METRICS
    # -----------------------------------------------------

    overall_metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("\n" + "=" * 65)
    print("FINAL XGBOOST TEST PERFORMANCE")
    print("=" * 65)

    print(f"MAE:  {overall_metrics['MAE']:.3f}")
    print(f"RMSE: {overall_metrics['RMSE']:.3f}")
    print(f"WAPE: {overall_metrics['WAPE']:.3f}%")

    # -----------------------------------------------------
    # SAVE TEST PREDICTIONS
    # -----------------------------------------------------

    prediction_df = test_df[
        [
            "trip_id",
            "route_id",
            "departure_datetime",
            "peak_period",
            "planned_capacity",
            TARGET,
        ]
    ].copy()

    prediction_df["predicted_demand"] = predictions

    prediction_df["prediction_error"] = (
        prediction_df["predicted_demand"]
        - prediction_df[TARGET]
    )

    prediction_df["absolute_error"] = (
        prediction_df["prediction_error"].abs()
    )

    prediction_df["actual_capacity_shortage"] = (
        prediction_df[TARGET]
        > prediction_df["planned_capacity"]
    )

    prediction_df["predicted_capacity_shortage"] = (
        prediction_df["predicted_demand"]
        > prediction_df["planned_capacity"]
    )

    prediction_path = (
        REPORT_DIR / "xgboost_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(f"\nSaved test predictions: {prediction_path}")

    # -----------------------------------------------------
    # SAVE OVERALL METRICS
    # -----------------------------------------------------

    overall_results = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Dataset": "Test",
                "Records": len(test_df),
                "MAE": overall_metrics["MAE"],
                "RMSE": overall_metrics["RMSE"],
                "WAPE": overall_metrics["WAPE"],
            }
        ]
    )

    overall_path = REPORT_DIR / "final_test_metrics.csv"

    overall_results.to_csv(
        overall_path,
        index=False,
    )

    # -----------------------------------------------------
    # ROUTE-WISE TEST PERFORMANCE
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("ROUTE-WISE TEST PERFORMANCE")
    print("=" * 65)

    route_results = []

    for route_id, group_df in prediction_df.groupby("route_id"):

        metrics = calculate_metrics(
            group_df[TARGET],
            group_df["predicted_demand"],
        )

        route_results.append(
            {
                "Route": route_id,
                "Records": len(group_df),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "WAPE": metrics["WAPE"],
            }
        )

    route_results_df = pd.DataFrame(route_results)

    print(route_results_df.to_string(index=False))

    route_path = REPORT_DIR / "final_test_route_metrics.csv"

    route_results_df.to_csv(
        route_path,
        index=False,
    )

    # -----------------------------------------------------
    # PEAK / OFF-PEAK TEST PERFORMANCE
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("PEAK / OFF-PEAK TEST PERFORMANCE")
    print("=" * 65)

    period_results = []

    for period, group_df in prediction_df.groupby("peak_period"):

        metrics = calculate_metrics(
            group_df[TARGET],
            group_df["predicted_demand"],
        )

        period_results.append(
            {
                "Period": period,
                "Records": len(group_df),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "WAPE": metrics["WAPE"],
            }
        )

    period_results_df = pd.DataFrame(period_results)

    print(period_results_df.to_string(index=False))

    period_path = REPORT_DIR / "final_test_period_metrics.csv"

    period_results_df.to_csv(
        period_path,
        index=False,
    )

    # -----------------------------------------------------
    # CAPACITY SHORTAGE DETECTION
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST CAPACITY RISK EVALUATION")
    print("=" * 65)

    actual_shortage = prediction_df["actual_capacity_shortage"]
    predicted_shortage = prediction_df["predicted_capacity_shortage"]

    tp = int((actual_shortage & predicted_shortage).sum())
    fp = int((~actual_shortage & predicted_shortage).sum())
    fn = int((actual_shortage & ~predicted_shortage).sum())
    tn = int((~actual_shortage & ~predicted_shortage).sum())

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    print(f"Actual shortages: {int(actual_shortage.sum())}")
    print(f"Predicted shortages: {int(predicted_shortage.sum())}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    capacity_results = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Actual_Shortages": int(actual_shortage.sum()),
                "Predicted_Shortages": int(predicted_shortage.sum()),
                "True_Positive": tp,
                "False_Positive": fp,
                "False_Negative": fn,
                "True_Negative": tn,
                "Precision": precision,
                "Recall": recall,
            }
        ]
    )

    capacity_path = REPORT_DIR / "final_test_capacity_metrics.csv"

    capacity_results.to_csv(
        capacity_path,
        index=False,
    )

    # -----------------------------------------------------
    # COMPLETION SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 65)

    print("\nGenerated reports:")

    print("- final_test_metrics.csv")
    print("- final_test_route_metrics.csv")
    print("- final_test_period_metrics.csv")
    print("- final_test_capacity_metrics.csv")
    print("- xgboost_test_predictions.csv")

    print("\nThe test set has now been evaluated.")


# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_test_set()
