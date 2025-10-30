from pathlib import Path
import pandas as pd
import numpy as np
import glob, os
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar

# =====================================
# 1. File paths
# =====================================
data_path   = "../data/test/"
out_dir     = Path("../data/test")               # Output directory
out_dir.mkdir(parents=True, exist_ok=True)

out_flights = out_dir / "flights_test_raw.csv"   # Flight-level raw dataset (keep carrier, tail, date keys)
out_merged  = out_dir / "merged_test_set.csv"    # 5-min aggregated dataset (for NODE model)

files = sorted(glob.glob(os.path.join(data_path, "T_ONTIME_REPORTING_*.csv")))
print(f"Totally {len(files)} monthly files:")
for f in files:
    print(" -", os.path.basename(f))

# =====================================
# 2. Time parsing helper
# =====================================
def parse_time(row_time):
    """Convert 4-digit numeric time to HH:MM"""
    if pd.isna(row_time):
        return pd.NaT
    try:
        t = f"{int(row_time):04d}"
        return pd.to_datetime(f"{t[:2]}:{t[2:]}", format="%H:%M", errors="coerce").time()
    except:
        return pd.NaT

# =====================================
# 3. Read and preprocess monthly files
# =====================================
df_list = []
for file in tqdm(files, desc="Processing monthly files"):
    df = pd.read_csv(file, low_memory=False)

    # Keep only flights arriving at BWI and not canceled
    df = df[(df["DEST"] == "BWI") & (df["CANCELLED"] == 0)]

    # Keep essential columns
    keep_cols = [
        "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM",
        "CRS_ARR_TIME", "ARR_TIME", "WHEELS_ON", "TAXI_IN"
    ]
    df = df[keep_cols].copy()

    # Convert dates and times
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df["CRS_ARR_TIME_dt"] = df.apply(
        lambda r: pd.to_datetime(f"{r['FL_DATE'].date()} {parse_time(r['CRS_ARR_TIME'])}", errors="coerce"), axis=1
    )  # Scheduled arrival time
    df["ARR_TIME_dt"] = df.apply(
        lambda r: pd.to_datetime(f"{r['FL_DATE'].date()} {parse_time(r['ARR_TIME'])}", errors="coerce"), axis=1
    )  # Actual arrival time

    # Adjust for cross-midnight flights
    df.loc[df["ARR_TIME_dt"] < df["CRS_ARR_TIME_dt"], "ARR_TIME_dt"] += pd.Timedelta(days=1)

    # Compute arrival delay (minutes)
    df["ArrDelayMinutes"] = (df["ARR_TIME_dt"] - df["CRS_ARR_TIME_dt"]).dt.total_seconds() / 60

    # Filter out abnormal or missing delay values
    df = df[df["ArrDelayMinutes"].between(-200, 1000)]
    df = df.dropna(subset=["CRS_ARR_TIME_dt", "ArrDelayMinutes"])

    # Keep full flight-level information
    df_list.append(df)

# =====================================
# 4. Combine all months (flight-level table)
# =====================================
flights = pd.concat(df_list, ignore_index=True).sort_values("CRS_ARR_TIME_dt")

# Add year and month for later merging with reference tables (A/B/C/D)
flights["YEAR"]  = flights["FL_DATE"].dt.year
flights["MONTH"] = flights["FL_DATE"].dt.month

# Save flight-level table (for merging tail → aircraft → seats later)
flights.to_csv(out_flights, index=False)
print(f"Saved flight-level base table: {out_flights}  shape={flights.shape}")

# =====================================
# 5. Aggregate to 5-min intervals
# =====================================
# Use 'size' to count flights instead of relying on WHEELS_ON (more robust if missing)
df_resampled = (
    flights.set_index("CRS_ARR_TIME_dt")
           .resample("5min")
           .agg(
               ArrDelayMinutes=("ArrDelayMinutes", "mean"),
               NumArrivals=("ArrDelayMinutes", "size"),
               AvgTaxiIn=("TAXI_IN", "mean")
           )
)

# Fill count column only; avoid overwriting mean columns with 0 (can bias the data)
df_resampled["NumArrivals"] = df_resampled["NumArrivals"].fillna(0).astype(int)
# Optionally, interpolate mean columns if you prefer continuous values:
# df_resampled["ArrDelayMinutes"] = df_resampled["ArrDelayMinutes"].interpolate(limit_direction="both")
# df_resampled["AvgTaxiIn"] = df_resampled["AvgTaxiIn"].interpolate(limit_direction="both")

# =====================================
# 6. Add time-based features
# =====================================
df_resampled["Hour"] = df_resampled.index.hour
df_resampled["DayOfWeek"] = df_resampled.index.dayofweek
df_resampled["Month"] = df_resampled.index.month

# Cyclical encoding for hour of day
df_resampled["sin_hour"] = np.sin(2 * np.pi * df_resampled["Hour"] / 24)
df_resampled["cos_hour"] = np.cos(2 * np.pi * df_resampled["Hour"] / 24)

# =====================================
# 7. Add US Federal Holiday features
# =====================================
cal = USFederalHolidayCalendar()
if len(df_resampled) > 0:
    holidays = cal.holidays(start=df_resampled.index.min(), end=df_resampled.index.max())
    df_resampled["Holiday"] = df_resampled.index.normalize().isin(holidays).astype(int)
    df_resampled["DaysToHoliday"] = df_resampled.index.to_series().apply(
        lambda t: np.abs((holidays - t.normalize()).days).min()
    )
else:
    df_resampled["Holiday"] = []
    df_resampled["DaysToHoliday"] = []

# =====================================
# 8. Save output files
# =====================================
df_resampled.to_csv(out_merged)
print(f"Saved 5-min aggregated dataset: {out_merged}  shape={df_resampled.shape}")
print(df_resampled.head(10))
