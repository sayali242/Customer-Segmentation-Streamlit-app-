"""
Bike Rental EDA — Exploratory Data Analysis for bike-sharing datasets.

Place hour.csv and day.csv in the `data/` folder (see README for dataset source).
Run from project root:  python bike_rental_eda.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use repo root: script can be run as `python bike_rental_eda.py` from project root
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

HOUR_CSV = DATA_DIR / "hour.csv"
DAY_CSV = DATA_DIR / "day.csv"


def _ensure_data():
    """Check that required CSV files exist; exit with instructions if not."""
    missing = []
    if not HOUR_CSV.exists():
        missing.append("hour.csv")
    if not DAY_CSV.exists():
        missing.append("day.csv")
    if missing:
        print("Missing data files:", ", ".join(missing))
        print(f"Expected location: {DATA_DIR}")
        print(
            "Download the UCI Bike Sharing dataset and place hour.csv and day.csv in the 'data/' folder."
        )
        print("Dataset: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
        sys.exit(1)


# =============================================================================
# 1. Load the datasets
# =============================================================================
_ensure_data()

hour = pd.read_csv(HOUR_CSV)
day = pd.read_csv(DAY_CSV)

print("Hour data shape:", hour.shape)
print("Day data shape:", day.shape)
print(hour.head())

# =============================================================================
# 2. Data overview and quality check
# =============================================================================
print(day.head())
print(hour.info())
print(hour.isnull().sum())
print(day.info())
print(day.isnull().sum())

# =============================================================================
# 3. Data cleaning & preprocessing (remove duplicates)
# =============================================================================
hour = hour.drop_duplicates()
day = day.drop_duplicates()
print(hour.columns)
print(day.columns)

# =============================================================================
# 4. Merging the datasets
# =============================================================================
hour["dteday"] = pd.to_datetime(hour["dteday"])
day["dteday"] = pd.to_datetime(day["dteday"])

hour_daily = (
    hour.groupby("dteday")
    .agg(
        {
            "cnt": "sum",
            "temp": "mean",
            "atemp": "mean",
            "hum": "mean",
            "windspeed": "mean",
        }
    )
    .reset_index()
)
hour_daily = hour_daily.rename(
    columns={
        "cnt": "hourly_total_cnt",
        "temp": "hourly_avg_temp",
        "atemp": "hourly_avg_atemp",
        "hum": "hourly_avg_hum",
        "windspeed": "hourly_avg_windspeed",
    }
)

merged = pd.merge(day, hour_daily, on="dteday", how="left")
print(merged.head())

# =============================================================================
# 5. Exploratory Data Analysis - Peak days and hours
# =============================================================================
plt.figure(figsize=(10, 6))
sns.lineplot(x="dteday", y="cnt", data=day)
plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="hr", y="cnt", data=hour)
plt.title("Bike Rentals by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Rentals")
plt.show()

# =============================================================================
# 6. EDA - Rentals by user type
# =============================================================================
plt.figure(figsize=(8, 6))
day[["casual", "registered"]].sum().plot(kind="bar")
plt.title("Total Rentals by User Type")
plt.ylabel("Total Rentals")
plt.show()

# =============================================================================
# 7. EDA - Seasonal trends
# =============================================================================
plt.figure(figsize=(10, 6))
sns.boxplot(x="season", y="cnt", data=day)
plt.title("Bike Rentals by Season")
plt.xlabel("Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)")
plt.ylabel("Total Rentals")
plt.show()

# =============================================================================
# 8. EDA - Weather conditions
# =============================================================================
plt.figure(figsize=(8, 6))
sns.boxplot(x="weathersit", y="cnt", data=day)
plt.title("Bike Rentals by Weather Situation")
plt.xlabel(
    "Weather Situation (1:Clear, 2:Mist, 3:Light Snow/Rain, 4:Heavy Rain)"
)
plt.ylabel("Total Rentals")
plt.show()

# =============================================================================
# 9. EDA - Outliers in daily rentals
# =============================================================================
plt.figure(figsize=(10, 4))
sns.boxplot(x=day["cnt"])
plt.title("Outliers in Total Daily Rentals")
plt.show()

# =============================================================================
# 10. EDA - Working day vs non-working day
# =============================================================================
plt.figure(figsize=(8, 6))
sns.boxplot(x="workingday", y="cnt", data=day)
plt.title("Bike Rentals: Working Day vs. Non-Working Day")
plt.xlabel("Working Day (1=Yes, 0=No)")
plt.ylabel("Total Rentals")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="holiday", y="cnt", data=day)
plt.title("Bike Rentals: Holiday vs. Non-Holiday")
plt.xlabel("Holiday (1=Yes, 0=No)")
plt.ylabel("Total Rentals")
plt.show()

# =============================================================================
# 11. EDA - Yearly trend
# =============================================================================
day["year"] = pd.DatetimeIndex(day["dteday"]).year
yearly_trend = day.groupby("year")["cnt"].sum().reset_index()

plt.figure(figsize=(8, 6))
sns.lineplot(x="year", y="cnt", data=yearly_trend, marker="o")
plt.title("Total Bike Rentals by Year")
plt.xlabel("Year")
plt.ylabel("Total Rentals")
plt.show()
