
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bus_demand.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["departure_datetime"] = pd.to_datetime(
    df["departure_datetime"]
)

df["completion_datetime"] = pd.to_datetime(
    df["completion_datetime"]
)

sns.set_theme(style="whitegrid")


def save_plot(filename):
    """Save the current plot and close the figure."""

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()
    plt.close()


# --------------------------------------------------
# 2. Dataset overview
# --------------------------------------------------

print("\nDATASET OVERVIEW")
print("=" * 50)

print("Shape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print(
    "\nDuplicate trip IDs:",
    df["trip_id"].duplicated().sum()
)

print("\nUnique routes:")
print(df["route_id"].unique())

print("\nDate range:")
print(
    df["departure_datetime"].min(),
    "to",
    df["departure_datetime"].max()
)


# --------------------------------------------------
# 3. Numerical summary
# --------------------------------------------------

print("\nNUMERICAL SUMMARY")
print("=" * 50)

print(df.describe())


# --------------------------------------------------
# 4. Passenger demand distribution
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="passenger_demand",
    bins=40,
    kde=True
)

plt.title("Distribution of Passenger Demand")
plt.xlabel("Passenger Demand")
plt.ylabel("Frequency")

save_plot("01_demand_distribution.png")


# --------------------------------------------------
# 5. Route-wise demand
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="route_id",
    y="passenger_demand"
)

plt.title("Passenger Demand Distribution by Route")

save_plot("02_route_demand.png")


# --------------------------------------------------
# 6. Hourly demand
# --------------------------------------------------

df["departure_hour"] = (
    df["departure_datetime"].dt.hour
)

hourly_demand = (
    df.groupby("departure_hour")["passenger_demand"]
    .mean()
)

plt.figure(figsize=(10, 5))

sns.lineplot(
    x=hourly_demand.index,
    y=hourly_demand.values,
    marker="o"
)

plt.title("Average Passenger Demand by Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Average Passenger Demand")

save_plot("03_hourly_demand.png")


# --------------------------------------------------
# 7. Weekend and seasonal demand
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(
    data=df,
    x="is_weekend",
    y="passenger_demand",
    ax=axes[0]
)

axes[0].set_title("Weekend vs Weekday Demand")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Weekday", "Weekend"])

sns.barplot(
    data=df,
    x="season",
    y="passenger_demand",
    ax=axes[1]
)

axes[1].set_title("Average Demand by Season")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_weekend_seasonal.png", dpi=150)
plt.show()
plt.close()


# --------------------------------------------------
# 8. Capacity analysis
# --------------------------------------------------

df["capacity_utilization"] = (
    df["actual_passenger_count"]
    / df["planned_capacity"]
) * 100

df["demand_exceeds_capacity"] = (
    df["passenger_demand"]
    > df["planned_capacity"]
)

print("\nCAPACITY ANALYSIS")
print("=" * 50)

print(
    "Trips where demand exceeded capacity:",
    df["demand_exceeds_capacity"].sum()
)

print(
    "Percentage of trips with unmet demand:",
    round(
        df["demand_exceeds_capacity"].mean() * 100,
        2
    ),
    "%"
)

print(
    "Average capacity utilization:",
    round(df["capacity_utilization"].mean(), 2),
    "%"
)


# --------------------------------------------------
# 9. Peak vs off-peak demand
# --------------------------------------------------

peak_demand = (
    df.groupby("peak_period")["passenger_demand"]
    .mean()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="peak_period",
    y="passenger_demand"
)

plt.title("Peak vs Off-Peak Average Demand")

save_plot("05_peak_offpeak.png")


# --------------------------------------------------
# 10. Route-wise summary
# --------------------------------------------------

print("\nROUTE-WISE DEMAND SUMMARY")
print("=" * 50)

print(
    df.groupby("route_id")["passenger_demand"]
    .agg(["mean", "median", "std", "min", "max"])
    .round(2)
)
print("\nWEEKEND EFFECT BY ROUTE")
print("=" * 50)

weekend_analysis = (
    df.groupby(["route_id", "is_weekend"])["passenger_demand"]
    .mean()
    .unstack()
    .rename(columns={0: "Weekday", 1: "Weekend"})
)

weekend_analysis["Difference"] = (
    weekend_analysis["Weekend"]
    - weekend_analysis["Weekday"]
)

print(weekend_analysis.round(2))
