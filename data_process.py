'''
BWI Flight Data Preprocessing Script
-----------------------------------
Purpose:
1. Batch-read BTS flight CSV files.
2. Filter records to destination = BWI.
3. Remove canceled flights.
4. Fix/normalize time fields (handle cross-midnight cases).
5. Recompute arrival delays (in minutes).
6. Aggregate into a continuous 5-minute time series.
7. Export the result to a CSV file.
'''

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import os

# ===========================
# 1. Set file paths.
# ===========================
data_path = "./data/test/"
output_file = "merged_test_set.csv"
files = sorted(glob.glob(data_path + "T_ONTIME_REPORTING_*.csv"))

print(f"Totally {len(files)} monthly files：")
for f in files:
    print(" -", os.path.basename(f))

# ===========================
# 2. Parse-time function.
# ===========================
def parse_time(row_time):
    """Transfer 4 digts to HH:MM form"""
    if pd.isna(row_time):
        return pd.NaT
    try:
        t = f"{int(row_time):04d}"
        return pd.to_datetime(f"{t[:2]}:{t[2:]}", format="%H:%M", errors="coerce").time()
    except:
        return pd.NaT

# ===========================
# 3. Read files seperately and process.
# ===========================
df_list = []

for file in tqdm(files, desc="Processing monthly files"):
    df = pd.read_csv(file, low_memory=False)

    # DESTNATION = "BWI"
    df = df[(df["DEST"] == "BWI") & (df["CANCELLED"] == 0)]

    # Transfer each date form.
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df["CRS_ARR_TIME_dt"] = df.apply(
        lambda r: pd.to_datetime(f"{r['FL_DATE'].date()} {parse_time(r['CRS_ARR_TIME'])}", errors="coerce"), axis=1
    )
    df["ARR_TIME_dt"] = df.apply(
        lambda r: pd.to_datetime(f"{r['FL_DATE'].date()} {parse_time(r['ARR_TIME'])}", errors="coerce"), axis=1
    )

    # Justify midnight.
    df.loc[df["ARR_TIME_dt"] < df["CRS_ARR_TIME_dt"], "ARR_TIME_dt"] += pd.Timedelta(days=1)

    # Recalculate delays (min).
    df["ArrDelayMinutes"] = (df["ARR_TIME_dt"] - df["CRS_ARR_TIME_dt"]).dt.total_seconds() / 60

    # Delete abnormal values.
    df = df[df["ArrDelayMinutes"].between(-200, 1000)]
    df = df.dropna(subset=["CRS_ARR_TIME_dt", "ArrDelayMinutes"])

    df_list.append(df[["CRS_ARR_TIME_dt", "ArrDelayMinutes", "WHEELS_ON", "TAXI_IN"]])

# ===========================
# 4. Merge all months data.
# ===========================
all_df = pd.concat(df_list, ignore_index=True)
all_df = all_df.sort_values("CRS_ARR_TIME_dt")

# ===========================
# 5. Aggregate in 5-minute intervals.
# ===========================
df_resampled = (
    all_df.set_index("CRS_ARR_TIME_dt")
          .resample("5min")
          .agg({
              "ArrDelayMinutes": "mean",      # Average arrival delay (minutes)
              "WHEELS_ON": "count",           # Number of arriving flights
              "TAXI_IN": "mean"               # Average taxi-in time (minutes)
          })
          .rename(columns={"WHEELS_ON": "NumArrivals", "TAXI_IN": "AvgTaxiIn"})
          .fillna(0)
)

# ===========================
# 6. Add time-related features.
# ===========================
df_resampled["Hour"] = df_resampled.index.hour
df_resampled["DayOfWeek"] = df_resampled.index.dayofweek
df_resampled["Month"] = df_resampled.index.month

# Cyclical encoding (sin/cos) for time-of-day
df_resampled["sin_hour"] = np.sin(2 * np.pi * df_resampled["Hour"] / 24)
df_resampled["cos_hour"] = np.cos(2 * np.pi * df_resampled["Hour"] / 24)

# ===========================
# 7. Add U.S. Federal Holiday feature
# ===========================
from pandas.tseries.holiday import USFederalHolidayCalendar

# Generate a list of holidays within the data time range
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df_resampled.index.min(), end=df_resampled.index.max())

# Mark whether each timestamp falls on a holiday
df_resampled["Holiday"] = df_resampled.index.normalize().isin(holidays).astype(int)

# Compute days to nearest holiday
df_resampled["DaysToHoliday"] = df_resampled.index.to_series().apply(
    lambda t: np.abs((holidays - t.normalize()).days).min()
)

# ===========================
# 8. Print output.
# ===========================
df_resampled.to_csv(output_file)
print(f"\nProcess finished, save as:{output_file}")
print(f"output shape: {df_resampled.shape}")
print(df_resampled.head(10))