
from pathlib import Path

import pandas as pd


def chronological_split(
    df,
    train_ratio=0.70,
    validation_ratio=0.15,
):
    """
    Split records chronologically using unique departure dates.

    All records belonging to the same date remain in
    the same partition.
    """

    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Train and validation ratios must sum to less than 1."
        )

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    data = df.copy()

    data["departure_datetime"] = pd.to_datetime(
        data["departure_datetime"]
    )

    data["_departure_date"] = (
        data["departure_datetime"].dt.normalize()
    )

    unique_dates = sorted(
        data["_departure_date"].unique()
    )

    total_dates = len(unique_dates)

    if total_dates < 3:
        raise ValueError(
            "At least three unique dates are required."
        )

    train_end = round(total_dates * train_ratio)

    validation_end = round(
        total_dates * (train_ratio + validation_ratio)
    )

 
    # Assign calendar dates to each partition.
    train_dates = unique_dates[:train_end]

    validation_dates = unique_dates[
        train_end:validation_end
    ]

    test_dates = unique_dates[validation_end:]

    # Select records belonging to each partition.
    train_df = data[
        data["_departure_date"].isin(train_dates)
    ].copy()

    validation_df = data[
        data["_departure_date"].isin(validation_dates)
    ].copy()

    test_df = data[
        data["_departure_date"].isin(test_dates)
    ].copy()

    # Remove temporary date column and reset indices.
    train_df = train_df.drop(
        columns=["_departure_date"]
    ).reset_index(drop=True)

    validation_df = validation_df.drop(
        columns=["_departure_date"]
    ).reset_index(drop=True)

    test_df = test_df.drop(
        columns=["_departure_date"]
    ).reset_index(drop=True)


    # Validation checks.
 
    assert len(train_df) + len(validation_df) + len(test_df) == len(df)

    assert train_df["departure_datetime"].max() < (
        validation_df["departure_datetime"].min()
    )

    assert validation_df["departure_datetime"].max() < (
        test_df["departure_datetime"].min()
    )

    return train_df, validation_df, test_df


def print_split_summary(train_df, validation_df, test_df):

    partitions = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df,
    }

    print("\nCHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 60)

    for name, partition in partitions.items():

        print(f"\n{name}")

        print("Records:", len(partition))

        print(
            "Unique dates:",
            partition["departure_datetime"].dt.normalize().nunique()
        )

        print(
            "Date range:",
            partition["departure_datetime"].min(),
            "to",
            partition["departure_datetime"].max(),
        )

        print(
            "Average demand:",
            round(partition["passenger_demand"].mean(), 2),
        )


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    INPUT_PATH = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "bus_demand_features.csv"
    )

    OUTPUT_DIR = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "splits"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_PATH)

    df["departure_datetime"] = pd.to_datetime(
        df["departure_datetime"]
    )

    train_df, validation_df, test_df = chronological_split(df)

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print_split_summary(
        train_df,
        validation_df,
        test_df,
    )

    print("\nSplit files saved successfully.")
