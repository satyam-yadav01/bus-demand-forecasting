
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROUTE_TIME_WINDOW = 4
ROUTE_WINDOW = 20
GLOBAL_TIME_WINDOW = 20

FORECAST_HORIZON_DAYS = 7


# --------------------------------------------------
# Helper: season
# --------------------------------------------------

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4]:
        return "Spring"
    elif month in [5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    return "Autumn"


# --------------------------------------------------
# Helper: peak period
# --------------------------------------------------

def get_peak_period(hour):
    if hour in [7, 8, 9, 17, 18]:
        return "Peak"
    return "Off-Peak"


# --------------------------------------------------
# Historical lookup preparation
# --------------------------------------------------


def prepare_history_lookup(df, group_columns):
    """
    Prepare sorted historical records for fast timestamp lookup.

    All timestamps are converted to nanoseconds to ensure
    consistent comparisons with pandas Timestamp.value.
    """

    lookup = {}

    for key, group in df.groupby(
        group_columns,
        sort=False,
        observed=True,
    ):

        # Normalize scalar and multi-column group keys.
        if not isinstance(key, tuple):
            key = (key,)

        group = group.sort_values("completion_datetime")

        # Explicitly normalize datetime precision to nanoseconds.
        timestamps = (
            group["completion_datetime"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )

        demands = (
            group["passenger_demand"]
            .to_numpy(dtype=float)
        )

        lookup[key] = (timestamps, demands)

    return lookup


# --------------------------------------------------
# Historical mean calculation
# --------------------------------------------------


def get_historical_mean(
    lookup,
    key,
    cutoff,
    window_size,
):
    """
    Calculate the mean of the last N eligible observations.

    Strict condition:
        completion_datetime < cutoff
    """

    timestamps, demands = lookup.get(
        key,
        (np.array([], dtype=np.int64), np.array([], dtype=float)),
    )

    cutoff_ns = pd.Timestamp(cutoff).value

    eligible_end = np.searchsorted(
        timestamps,
        cutoff_ns,
        side="left",
    )

    start = max(0, eligible_end - window_size)

    eligible_demands = demands[start:eligible_end]

    count = len(eligible_demands)

    if count == 0:
        return np.nan, 0

    return float(eligible_demands.mean()), count



# --------------------------------------------------
# Main feature builder
# --------------------------------------------------

def build_historical_features(df):
    """
    Build leakage-safe features for each scheduled departure.

    The target is passenger_demand.

    All historical demand features use only trips that
    completed strictly before the prediction cutoff.
    """

    required_columns = [
        "trip_id",
        "route_id",
        "departure_datetime",
        "completion_datetime",
        "planned_capacity",
        "passenger_demand",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy().reset_index(drop=True)

    data["departure_datetime"] = pd.to_datetime(
        data["departure_datetime"]
    )

    data["completion_datetime"] = pd.to_datetime(
        data["completion_datetime"]
    )

    if data["trip_id"].duplicated().any():
        raise ValueError("Duplicate trip IDs detected.")

    if (data["completion_datetime"] < data["departure_datetime"]).any():
        raise ValueError(
            "A trip cannot complete before it departs."
        )

    if data["passenger_demand"].isna().any():
        raise ValueError("Passenger demand contains missing values.")

    if (data["passenger_demand"] < 0).any():
        raise ValueError("Passenger demand cannot be negative.")

    # --------------------------------------------------
    # Derive known-in-advance calendar features
    # --------------------------------------------------

    data["departure_hour"] = (
        data["departure_datetime"].dt.hour
    )

    data["day_of_week"] = (
        data["departure_datetime"].dt.day_of_week
    )

    data["month"] = (
        data["departure_datetime"].dt.month
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["season"] = (
        data["month"].apply(get_season)
    )

    data["peak_period"] = (
        data["departure_hour"].apply(get_peak_period)
    )

    # --------------------------------------------------
    # Calculate prediction cutoff
    # --------------------------------------------------

    data["prediction_cutoff"] = (
        data["departure_datetime"]
        - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
    )

    # --------------------------------------------------
    # Prepare historical lookups
    # --------------------------------------------------

    route_time_lookup = prepare_history_lookup(
        data,
        ["route_id", "departure_hour"],
    )

    route_lookup = prepare_history_lookup(
        data,
        ["route_id"],
    )

    global_time_lookup = prepare_history_lookup(
        data,
        ["departure_hour"],
    )

    # --------------------------------------------------
    # Generate historical features
    # --------------------------------------------------

    route_time_means = []
    route_time_counts = []

    route_means = []
    route_counts = []

    global_time_means = []

    for row in data.itertuples(index=False):

        cutoff = row.prediction_cutoff

        # Same route and same departure hour.
        route_time_mean, route_time_count = (
            get_historical_mean(
                route_time_lookup,
                (row.route_id, row.departure_hour),
                cutoff,
                ROUTE_TIME_WINDOW,
            )
        )

        # Same route, all departure hours.
        route_mean, route_count = get_historical_mean(
            route_lookup,
            (row.route_id,),
            cutoff,
            ROUTE_WINDOW,
        )

        # All routes, same departure hour.
        global_time_mean, _ = get_historical_mean(
            global_time_lookup,
            (row.departure_hour,),
            cutoff,
            GLOBAL_TIME_WINDOW,
        )

        route_time_means.append(route_time_mean)
        route_time_counts.append(route_time_count)

        route_means.append(route_mean)
        route_counts.append(route_count)

        global_time_means.append(global_time_mean)

    data["hist_route_time_avg"] = route_time_means
    data["route_time_history_count"] = route_time_counts

    data["hist_route_avg"] = route_means
    data["route_history_count"] = route_counts

    data["hist_global_time_avg"] = global_time_means

    # --------------------------------------------------
    # Validate output
    # --------------------------------------------------

    assert len(data) == len(df)

    assert (
        data["hist_route_time_avg"].isna()
        == (data["route_time_history_count"] == 0)
    ).all()

    return data



# --------------------------------------------------
# Execute feature generation
# --------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    INPUT_PATH = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "bus_demand.csv"
    )

    OUTPUT_PATH = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "bus_demand_features.csv"
    )

    df = pd.read_csv(INPUT_PATH)

    df["departure_datetime"] = pd.to_datetime(
        df["departure_datetime"]
    )

    df["completion_datetime"] = pd.to_datetime(
        df["completion_datetime"]
    )

    feature_df = build_historical_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFEATURE GENERATION COMPLETED")
    print("-" * 45)

    print("Total records:", len(feature_df))

    print("\nHistorical feature availability:")
    print(
        feature_df[
            [
                "hist_route_time_avg",
                "hist_route_avg",
                "hist_global_time_avg",
            ]
        ].notna().sum()
    )

    print("\nFirst 10 records:")
    print(
        feature_df[
            [
                "trip_id",
                "prediction_cutoff",
                "hist_route_time_avg",
                "hist_route_avg",
                "hist_global_time_avg",
            ]
        ].head(10)
    )

    print("\nSaved to:", OUTPUT_PATH)
