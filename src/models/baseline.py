
import numpy as np
import pandas as pd


def predict_historical_baseline(
    df,
    min_route_time_history=3,
):
    """
    Generate predictions using historical averages.

    Fallback priority:
    1. Route + departure-hour average (minimum 3 records)
    2. Route average
    3. Global departure-hour average
    4. No prediction if no eligible history exists

    Returns a copy of the input DataFrame with:
        baseline_prediction
        baseline_source
    """

    required_columns = [
        "hist_route_time_avg",
        "route_time_history_count",
        "hist_route_avg",
        "route_history_count",
        "hist_global_time_avg",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    predictions = []
    sources = []

    for row in result.itertuples(index=False):

        prediction = np.nan
        source = "no_history"

        # 1. Route + departure-hour average.
        if (
            row.route_time_history_count >= min_route_time_history
            and pd.notna(row.hist_route_time_avg)
        ):
            prediction = float(row.hist_route_time_avg)
            source = "route_time_average"

        # 2. Route average.
        elif (
            row.route_history_count >= 1
            and pd.notna(row.hist_route_avg)
        ):
            prediction = float(row.hist_route_avg)
            source = "route_average"

        # 3. Global departure-hour average.
        elif pd.notna(row.hist_global_time_avg):
            prediction = float(row.hist_global_time_avg)
            source = "global_time_average"

        predictions.append(prediction)
        sources.append(source)

    result["baseline_prediction"] = predictions
    result["baseline_source"] = sources

    return result


if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]

    validation_path = (
        project_root
        / "data"
        / "processed"
        / "splits"
        / "validation.csv"
    )

    output_path = (
        project_root
        / "reports"
        / "baseline_validation_predictions.csv"
    )

    df = pd.read_csv(validation_path)

    result = predict_historical_baseline(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(output_path, index=False)

    print("\nHISTORICAL BASELINE GENERATED")
    print("-" * 45)

    print("Total records:", len(result))

    print("\nPrediction source distribution:")
    print(result["baseline_source"].value_counts())

    print(
        "\nPredictions available:",
        result["baseline_prediction"].notna().sum(),
    )

    print(
        "\nPredictions unavailable:",
        result["baseline_prediction"].isna().sum(),
    )

    print("\nSample predictions:")
    print(
        result[
            [
                "route_id",
                "departure_datetime",
                "passenger_demand",
                "baseline_prediction",
                "baseline_source",
            ]
        ].head(10)
    )

    print("\nSaved to:", output_path)
