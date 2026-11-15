
import numpy as np
import pandas as pd

from src.features.build_features import (
    get_historical_mean,
    prepare_history_lookup,
    build_historical_features,
)


def test_exact_cutoff_observation_is_excluded():
    """
    A historical trip completing exactly at cutoff
    must not be included.
    """

    history = pd.DataFrame({
        "route_id": ["R01", "R01", "R01"],
        "departure_hour": [8, 8, 8],
        "completion_datetime": pd.to_datetime([
            "2024-01-01 07:59:00",
            "2024-01-01 08:00:00",
            "2024-01-01 08:01:00",
        ]),
        "passenger_demand": [20, 40, 60],
    })

    lookup = prepare_history_lookup(
        history,
        ["route_id", "departure_hour"],
    )

    mean, count = get_historical_mean(
        lookup,
        ("R01", 8),
        pd.Timestamp("2024-01-01 08:00:00"),
        10,
    )

    assert count == 1
    assert mean == 20


def test_future_observations_are_excluded():

    history = pd.DataFrame({
        "route_id": ["R01", "R01"],
        "completion_datetime": pd.to_datetime([
            "2024-01-01 09:00:00",
            "2024-01-01 11:00:00",
        ]),
        "passenger_demand": [25, 100],
    })

    lookup = prepare_history_lookup(
        history,
        ["route_id"],
    )

    mean, count = get_historical_mean(
        lookup,
        ("R01",),
        pd.Timestamp("2024-01-01 10:00:00"),
        10,
    )

    assert count == 1
    assert mean == 25


def test_last_n_eligible_observations_are_used():

    history = pd.DataFrame({
        "route_id": ["R01"] * 5,
        "completion_datetime": pd.to_datetime([
            "2024-01-01 08:00:00",
            "2024-01-02 08:00:00",
            "2024-01-03 08:00:00",
            "2024-01-04 08:00:00",
            "2024-01-05 08:00:00",
        ]),
        "passenger_demand": [10, 20, 30, 40, 50],
    })

    lookup = prepare_history_lookup(
        history,
        ["route_id"],
    )

    mean, count = get_historical_mean(
        lookup,
        ("R01",),
        pd.Timestamp("2024-01-06 08:00:00"),
        3,
    )

    assert count == 3
    assert mean == 40


def test_no_history_returns_nan():

    lookup = {
        ("R01",): (
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )
    }

    mean, count = get_historical_mean(
        lookup,
        ("R01",),
        pd.Timestamp("2024-01-01 08:00:00"),
        10,
    )

    assert np.isnan(mean)
    assert count == 0


def test_prediction_cutoff_is_exactly_seven_days_earlier():

    df = pd.DataFrame({
        "trip_id": ["T001"],
        "route_id": ["R01"],
        "departure_datetime": pd.to_datetime([
            "2024-02-15 08:00:00"
        ]),
        "completion_datetime": pd.to_datetime([
            "2024-02-15 09:00:00"
        ]),
        "planned_capacity": [40],
        "passenger_demand": [35],
    })

    result = build_historical_features(df)

    expected_cutoff = pd.Timestamp("2024-02-08 08:00:00")

    assert result.loc[0, "prediction_cutoff"] == expected_cutoff

    assert pd.isna(result.loc[0, "hist_route_time_avg"])


def test_route_history_uses_scalar_group_key_correctly():

    history = pd.DataFrame({
        "route_id": ["R01", "R01"],
        "completion_datetime": pd.to_datetime([
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
        ]),
        "passenger_demand": [20, 40],
    })

    lookup = prepare_history_lookup(
        history,
        ["route_id"],
    )

    mean, count = get_historical_mean(
        lookup,
        ("R01",),
        pd.Timestamp("2024-01-01 09:00:00"),
        10,
    )

    assert count == 2
    assert mean == 30
