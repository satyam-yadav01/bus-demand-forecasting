
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_SEED = 42
START_DATE = "2024-01-01"
NUM_DAYS = 730

DEPARTURE_HOURS = [6, 7, 8, 9, 12, 14, 16, 17, 18, 20]

# Each route has a different average demand and weekend pattern.
ROUTES = {
    "R01": {"base_demand": 28, "weekend_factor": 0.90, "duration": (45, 75)},
    "R02": {"base_demand": 34, "weekend_factor": 1.10, "duration": (55, 90)},
    "R03": {"base_demand": 22, "weekend_factor": 0.85, "duration": (40, 70)},
    "R04": {"base_demand": 38, "weekend_factor": 1.05, "duration": (65, 110)},
    "R05": {"base_demand": 30, "weekend_factor": 0.95, "duration": (50, 85)},
}

# Time-of-day demand multipliers.
TIME_FACTORS = {
    6: 1.20,
    7: 1.35,
    8: 1.40,
    9: 1.15,
    12: 0.80,
    14: 0.85,
    16: 1.10,
    17: 1.35,
    18: 1.40,
    20: 1.00,
}

# Seasonal demand multipliers.
SEASON_FACTORS = {
    "Winter": 1.05,
    "Spring": 0.98,
    "Summer": 1.12,
    "Monsoon": 0.88,
    "Autumn": 1.00,
}

# Negative Binomial dispersion parameter.
# Smaller values produce greater demand variability.
DISPERSION = 18


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def get_season(month):
    """Return the season corresponding to a calendar month."""

    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4]:
        return "Spring"
    elif month in [5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    else:
        return "Autumn"


def get_peak_period(hour):
    """Classify a departure as peak or off-peak."""

    if hour in [7, 8, 9, 17, 18]:
        return "Peak"
    return "Off-Peak"


def generate_event_multipliers(rng, num_days, num_routes):
    """
    Generate occasional route-specific demand shocks.

    These represent unobserved special events or disruptions.
    They are deliberately not exposed as model input features.
    """

    multipliers = np.ones((num_days, num_routes))

    event_mask = rng.random((num_days, num_routes)) < 0.025

    for day_idx, route_idx in zip(*np.where(event_mask)):

        if rng.random() < 0.70:
            # Positive demand spike.
            multipliers[day_idx, route_idx] = rng.uniform(1.5, 2.0)
        else:
            # Negative demand shock.
            multipliers[day_idx, route_idx] = rng.uniform(0.45, 0.75)

    return multipliers


# --------------------------------------------------
# 3. Main dataset generator
# --------------------------------------------------

def generate_dataset(
    start_date=START_DATE,
    num_days=NUM_DAYS,
    random_seed=RANDOM_SEED,
    output_path=None,
):
    """
    Generate synthetic bus passenger-demand data.

    Prediction target:
        passenger_demand

    Actual passengers carried:
        min(passenger_demand, planned_capacity)
    """

    rng = np.random.default_rng(random_seed)

    dates = pd.date_range(
        start=start_date,
        periods=num_days,
        freq="D",
    )

    route_ids = list(ROUTES.keys())

    event_multipliers = generate_event_multipliers(
        rng,
        num_days,
        len(route_ids),
    )

    records = []
    trip_number = 1

    # Iterate through every date.
    for day_idx, date in enumerate(dates):

        is_weekend = int(date.weekday() >= 5)
        season = get_season(date.month)

        # Iterate through all routes.
        for route_idx, (route_id, route_info) in enumerate(ROUTES.items()):

            base_demand = route_info["base_demand"]
            weekend_factor = route_info["weekend_factor"]
            min_duration, max_duration = route_info["duration"]

            event_factor = event_multipliers[day_idx, route_idx]

            # Weekend effect.
            weekend_multiplier = (
                weekend_factor if is_weekend else 1.0
            )

            # Iterate through scheduled departure times.
            for hour in DEPARTURE_HOURS:

                departure_datetime = (
                    date + pd.Timedelta(hours=hour)
                )

                # Time and seasonal effects.
                time_multiplier = TIME_FACTORS[hour]
                season_multiplier = SEASON_FACTORS[season]

                # Expected passenger demand.
                mean_demand = (
                    base_demand
                    * time_multiplier
                    * weekend_multiplier
                    * season_multiplier
                    * event_factor
                )

                # Negative Binomial distribution.
                probability = (
                    DISPERSION / (DISPERSION + mean_demand)
                )

                passenger_demand = int(
                    rng.negative_binomial(
                        n=DISPERSION,
                        p=probability,
                    )
                )

                # Capacity is assigned independently of demand.
                planned_capacity = int(
                    rng.choice([40, 60])
                )

                # Actual passengers cannot exceed capacity.
                actual_passenger_count = min(
                    passenger_demand,
                    planned_capacity,
                )

                # Simulate trip duration.
                trip_duration_minutes = int(
                    rng.integers(
                        min_duration,
                        max_duration + 1,
                    )
                )

                completion_datetime = (
                    departure_datetime
                    + pd.Timedelta(
                        minutes=trip_duration_minutes
                    )
                )

                records.append({
                    "trip_id": f"TRIP{trip_number:06d}",
                    "route_id": route_id,
                    "departure_datetime": departure_datetime,
                    "trip_duration_minutes": trip_duration_minutes,
                    "completion_datetime": completion_datetime,
                    "planned_capacity": planned_capacity,
                    "passenger_demand": passenger_demand,
                    "actual_passenger_count": actual_passenger_count,
                    "is_weekend": is_weekend,
                    "season": season,
                    "peak_period": get_peak_period(hour),
                })

                trip_number += 1

    # Convert generated records into a DataFrame.
    df = pd.DataFrame(records)

    # --------------------------------------------------
    # 4. Data quality checks
    # --------------------------------------------------

    assert df["trip_id"].is_unique
    assert (df["passenger_demand"] >= 0).all()
    assert (df["actual_passenger_count"] >= 0).all()

    assert (
        df["actual_passenger_count"]
        <= df["planned_capacity"]
    ).all()

    assert (
        df["actual_passenger_count"]
        <= df["passenger_demand"]
    ).all()

    assert df["departure_datetime"].notna().all()
    assert df["completion_datetime"].notna().all()

    # --------------------------------------------------
    # 5. Save dataset
    # --------------------------------------------------

    if output_path is None:
        project_root = Path(__file__).resolve().parents[2]

        output_path = (
            project_root
            / "data"
            / "raw"
            / "bus_demand.csv"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_path, index=False)

    # --------------------------------------------------
    # 6. Display summary
    # --------------------------------------------------

    print("\nDATASET GENERATED SUCCESSFULLY")
    print("-" * 45)

    print(f"Total records: {len(df)}")
    print(f"Unique routes: {df['route_id'].nunique()}")
    print(f"Date range: {df['departure_datetime'].min()} to "
          f"{df['departure_datetime'].max()}")

    print(f"\nSaved to: {output_path}")

    print("\nPassenger demand statistics:")
    print(df["passenger_demand"].describe())

    print("\nCapacity distribution:")
    print(df["planned_capacity"].value_counts())

    print("\nRoute-wise average demand:")
    print(
        df.groupby("route_id")["passenger_demand"]
        .mean()
        .round(2)
    )

    print("\nPeak/off-peak average demand:")
    print(
        df.groupby("peak_period")["passenger_demand"]
        .mean()
        .round(2)
    )

    return df


# --------------------------------------------------
# 7. Execute generator
# --------------------------------------------------

if __name__ == "__main__":
    generate_dataset()
