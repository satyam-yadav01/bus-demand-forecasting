
import pandas as pd
import pytest

from src.data.split_data import chronological_split


def create_sample_data(num_days=20):

    dates = pd.date_range(
        "2024-01-01",
        periods=num_days,
        freq="D",
    )

    records = []

    for date in dates:

        # Multiple trips per day.
        for hour in [8, 12, 18]:

            records.append({
                "departure_datetime": (
                    date + pd.Timedelta(hours=hour)
                ),
                "passenger_demand": 30,
            })

    return pd.DataFrame(records)


def test_chronological_order():

    df = create_sample_data()

    train, validation, test = chronological_split(df)

    assert (
        train["departure_datetime"].max()
        < validation["departure_datetime"].min()
    )

    assert (
        validation["departure_datetime"].max()
        < test["departure_datetime"].min()
    )


def test_no_date_overlap():

    df = create_sample_data()

    train, validation, test = chronological_split(df)

    train_dates = set(
        train["departure_datetime"].dt.date
    )

    validation_dates = set(
        validation["departure_datetime"].dt.date
    )

    test_dates = set(
        test["departure_datetime"].dt.date
    )

    assert train_dates.isdisjoint(validation_dates)
    assert validation_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)


def test_all_records_are_preserved():

    df = create_sample_data()

    train, validation, test = chronological_split(df)

    assert len(train) + len(validation) + len(test) == len(df)


def test_all_trips_on_same_date_stay_together():

    df = create_sample_data()

    train, validation, test = chronological_split(df)

    for partition in [train, validation, test]:

        counts = (
            partition.groupby(
                partition["departure_datetime"].dt.date
            ).size()
        )

        assert (counts == 3).all()


def test_invalid_split_ratios():

    df = create_sample_data()

    with pytest.raises(ValueError):
        chronological_split(
            df,
            train_ratio=0.8,
            validation_ratio=0.3,
        )


def test_empty_dataset():

    df = pd.DataFrame({
        "departure_datetime": pd.to_datetime([]),
        "passenger_demand": [],
    })

    with pytest.raises(ValueError):
        chronological_split(df)
