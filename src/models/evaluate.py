
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------
# 2. CALCULATE REGRESSION METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    total_actual = np.sum(np.abs(y_true))

    if total_actual == 0:
        wape = np.nan
    else:
        wape = (
            np.sum(np.abs(y_true - y_pred))
            / total_actual
        ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    }


# ---------------------------------------------------------
# 3. LOAD MODEL PREDICTIONS
# ---------------------------------------------------------

def load_validation_predictions():

    ridge_path = REPORT_DIR / "ridge_validation_predictions.csv"
    xgb_path = REPORT_DIR / "xgboost_validation_predictions.csv"

    ridge_df = pd.read_csv(ridge_path)
    xgb_df = pd.read_csv(xgb_path)

    # Verify that both models were evaluated on the same trips.
    if not ridge_df["trip_id"].equals(xgb_df["trip_id"]):
        raise ValueError(
            "Ridge and XGBoost predictions do not contain "
            "the same trips in the same order."
        )

    # Verify that actual demand values match.
    if not np.allclose(
        ridge_df["passenger_demand"],
        xgb_df["passenger_demand"],
    ):
        raise ValueError(
            "Actual passenger demand differs between model files."
        )

    return {
        "Ridge": ridge_df,
        "XGBoost": xgb_df,
    }


# ---------------------------------------------------------
# 4. EVALUATE ONE GROUP
# ---------------------------------------------------------

def evaluate_group(df, model_name, group_name, group_value):

    metrics = calculate_metrics(
        df["passenger_demand"],
        df["predicted_demand"],
    )

    return {
        "Model": model_name,
        "Group_Type": group_name,
        "Group": group_value,
        "Records": len(df),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "WAPE": metrics["WAPE"],
    }


# ---------------------------------------------------------
# 5. DETAILED VALIDATION EVALUATION
# ---------------------------------------------------------

def evaluate_models_by_group():

    print("\nLoading validation predictions...")

    model_predictions = load_validation_predictions()

    results = []

    for model_name, df in model_predictions.items():

        print("\n" + "=" * 65)
        print(f"DETAILED VALIDATION EVALUATION: {model_name}")
        print("=" * 65)

        # -------------------------------------------------
        # Overall performance
        # -------------------------------------------------

        overall_result = evaluate_group(
            df,
            model_name,
            "Overall",
            "All Trips",
        )

        results.append(overall_result)

        print("\nOverall performance:")
        print(
            f"MAE:  {overall_result['MAE']:.3f}\n"
            f"RMSE: {overall_result['RMSE']:.3f}\n"
            f"WAPE: {overall_result['WAPE']:.3f}%"
        )

        # -------------------------------------------------
        # Route-wise performance
        # -------------------------------------------------

        print("\nRoute-wise performance:")

        route_results = []

        for route_id, group_df in df.groupby("route_id"):

            result = evaluate_group(
                group_df,
                model_name,
                "Route",
                route_id,
            )

            results.append(result)
            route_results.append(result)

        route_table = pd.DataFrame(route_results)

        print(
            route_table[
                ["Group", "Records", "MAE", "RMSE", "WAPE"]
            ].to_string(index=False)
        )

        # -------------------------------------------------
        # Peak / Off-Peak performance
        # -------------------------------------------------

        print("\nPeak / Off-Peak performance:")

        period_results = []

        for period, group_df in df.groupby("peak_period"):

            result = evaluate_group(
                group_df,
                model_name,
                "Peak_Period",
                period,
            )

            results.append(result)
            period_results.append(result)

        period_table = pd.DataFrame(period_results)

        print(
            period_table[
                ["Group", "Records", "MAE", "RMSE", "WAPE"]
            ].to_string(index=False)
        )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "validation_detailed_metrics.csv"

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("DETAILED VALIDATION EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Saved results: {output_path}")

    # -----------------------------------------------------
    # Compare models by route
    # -----------------------------------------------------

    route_comparison = results_df[
        results_df["Group_Type"] == "Route"
    ].pivot(
        index="Group",
        columns="Model",
        values="WAPE",
    )

    print("\nRoute-wise WAPE comparison (%):")
    print(route_comparison.round(3).to_string())

    # -----------------------------------------------------
    # Compare models by peak period
    # -----------------------------------------------------

    period_comparison = results_df[
        results_df["Group_Type"] == "Peak_Period"
    ].pivot(
        index="Group",
        columns="Model",
        values="WAPE",
    )

    print("\nPeak / Off-Peak WAPE comparison (%):")
    print(period_comparison.round(3).to_string())


# ---------------------------------------------------------
# 8. CAPACITY RISK ANALYSIS
# ---------------------------------------------------------

def analyze_capacity_risk():

    model_predictions = load_validation_predictions()

    validation_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "splits"
        / "validation.csv"
    )

    validation_data = pd.read_csv(validation_path)

    # Retrieve planned capacity for each validation trip.
    capacity_data = validation_data[
        ["trip_id", "planned_capacity"]
    ].copy()

    all_results = []

    for model_name, predictions_df in model_predictions.items():

        df = predictions_df.merge(
            capacity_data,
            on="trip_id",
            how="left",
            validate="one_to_one",
        )

        if df["planned_capacity"].isna().any():
            raise ValueError(
                "Some trips are missing planned capacity."
            )

        # Actual and predicted capacity shortage flags.
        df["actual_capacity_shortage"] = (
            df["passenger_demand"] > df["planned_capacity"]
        )

        df["predicted_capacity_shortage"] = (
            df["predicted_demand"] > df["planned_capacity"]
        )

        actual_shortage = df["actual_capacity_shortage"]
        predicted_shortage = df["predicted_capacity_shortage"]

        # Confusion matrix for capacity shortage detection.
        true_positive = (actual_shortage & predicted_shortage).sum()
        false_positive = (~actual_shortage & predicted_shortage).sum()
        false_negative = (actual_shortage & ~predicted_shortage).sum()
        true_negative = (~actual_shortage & ~predicted_shortage).sum()

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else np.nan
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else np.nan
        )

        actual_shortage_rate = actual_shortage.mean() * 100
        predicted_shortage_rate = predicted_shortage.mean() * 100

        result = {
            "Model": model_name,
            "Actual_Shortages": int(actual_shortage.sum()),
            "Predicted_Shortages": int(predicted_shortage.sum()),
            "True_Positive": int(true_positive),
            "False_Positive": int(false_positive),
            "False_Negative": int(false_negative),
            "True_Negative": int(true_negative),
            "Precision": precision,
            "Recall": recall,
            "Actual_Shortage_Rate": actual_shortage_rate,
            "Predicted_Shortage_Rate": predicted_shortage_rate,
        }

        all_results.append(result)

        print("\n" + "=" * 65)
        print(f"CAPACITY RISK ANALYSIS: {model_name}")
        print("=" * 65)

        print(f"Total validation trips: {len(df)}")
        print(f"Actual capacity shortages: {int(actual_shortage.sum())}")
        print(f"Predicted capacity shortages: {int(predicted_shortage.sum())}")

        print("\nConfusion matrix:")
        print(f"True positives:  {true_positive}")
        print(f"False positives: {false_positive}")
        print(f"False negatives: {false_negative}")
        print(f"True negatives:  {true_negative}")

        print("\nCapacity shortage detection:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")

        print(f"Actual shortage rate:    {actual_shortage_rate:.2f}%")
        print(f"Predicted shortage rate: {predicted_shortage_rate:.2f}%")

        # Save trip-level capacity analysis.
        output_path = (
            REPORT_DIR
            / f"{model_name.lower()}_capacity_risk.csv"
        )

        df.to_csv(output_path, index=False)

        print(f"\nSaved: {output_path}")

    # Save model comparison.
    results_df = pd.DataFrame(all_results)

    summary_path = REPORT_DIR / "capacity_risk_summary.csv"

    results_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("CAPACITY RISK MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df[
            [
                "Model",
                "Actual_Shortages",
                "Predicted_Shortages",
                "Precision",
                "Recall",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved summary: {summary_path}")




# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_models_by_group()


# ---------------------------------------------------------
# 7. VALIDATION ERROR ANALYSIS
# ---------------------------------------------------------

def analyze_prediction_errors():

    model_predictions = load_validation_predictions()

    all_error_results = []

    for model_name, df in model_predictions.items():

        error_df = df.copy()

        # Positive error means overprediction.
        # Negative error means underprediction.
        error_df["prediction_error"] = (
            error_df["predicted_demand"]
            - error_df["passenger_demand"]
        )

        error_df["absolute_error"] = (
            error_df["prediction_error"].abs()
        )

        error_df["error_type"] = np.where(
            error_df["prediction_error"] > 0,
            "Overprediction",
            np.where(
                error_df["prediction_error"] < 0,
                "Underprediction",
                "Exact",
            ),
        )

        over_count = (
            error_df["error_type"] == "Overprediction"
        ).sum()

        under_count = (
            error_df["error_type"] == "Underprediction"
        ).sum()

        exact_count = (
            error_df["error_type"] == "Exact"
        ).sum()

        total = len(error_df)

        result = {
            "Model": model_name,
            "Mean_Error": error_df["prediction_error"].mean(),
            "Overprediction_Count": over_count,
            "Underprediction_Count": under_count,
            "Exact_Count": exact_count,
            "Overprediction_Rate": over_count / total * 100,
            "Underprediction_Rate": under_count / total * 100,
        }

        all_error_results.append(result)

        print("\n" + "=" * 65)
        print(f"ERROR ANALYSIS: {model_name}")
        print("=" * 65)

        print(
            f"Mean prediction error: "
            f"{result['Mean_Error']:.3f}"
        )

        print(
            f"Overprediction: {over_count} "
            f"({result['Overprediction_Rate']:.2f}%)"
        )

        print(
            f"Underprediction: {under_count} "
            f"({result['Underprediction_Rate']:.2f}%)"
        )

        print(
            f"Exact predictions: {exact_count}"
        )

        # Largest prediction errors.
        print("\n10 largest absolute prediction errors:")

        worst_predictions = error_df.nlargest(
            10,
            "absolute_error",
        )

        print(
            worst_predictions[
                [
                    "trip_id",
                    "route_id",
                    "departure_datetime",
                    "peak_period",
                    "passenger_demand",
                    "predicted_demand",
                    "prediction_error",
                    "absolute_error",
                ]
            ].to_string(index=False)
        )

        # Save individual prediction errors.
        error_path = (
            REPORT_DIR
            / f"{model_name.lower()}_validation_error_analysis.csv"
        )

        error_df.to_csv(error_path, index=False)

        print(f"\nSaved error details: {error_path}")

    # Save model-level error summary.
    summary_df = pd.DataFrame(all_error_results)

    summary_path = REPORT_DIR / "validation_error_summary.csv"

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("VALIDATION ERROR ANALYSIS SUMMARY")
    print("=" * 65)

    print(summary_df.to_string(index=False))

    print(f"\nSaved summary: {summary_path}")


