import pandas as pd
import numpy as np
import requests
from io import StringIO
import re

# ===========================================
# 1. Load flight data
# ===========================================
flights_df = pd.read_csv("../data/ref/data_set.csv")

print("Step 1: flights_df rows =", len(flights_df))

# Clean DEST before anything else
flights_df["DEST"] = (
    flights_df["DEST"]
    .astype(str)
    .str.upper()
    .str.replace(r"\s+", "", regex=True)
)


# Remove invalid DEST rows
flights_df["DEST"] = flights_df["DEST"].replace({"": np.nan, "NA": np.nan})
flights_df = flights_df[flights_df["DEST"].notna()]

print("Step 2: flights_df after DEST clean =", len(flights_df))

# Filter BWI flights
bwi_df = flights_df[flights_df["DEST"] == "BWI"].copy()
bwi_df = bwi_df[bwi_df["ARR_TIME_dt"].notna()].copy()

print("Step 3: bwi_df rows =", len(bwi_df))


# Convert time columns
for col in ["ARR_TIME_dt", "CRS_ARR_TIME_dt", "WHEELS_ON_dt"]:
    bwi_df[col] = pd.to_datetime(bwi_df[col], errors="coerce")

print("Step 4: ARR_TIME_dt null count =", bwi_df["ARR_TIME_dt"].isna().sum())
print("Step 4: bwi_df rows after dropping NaN =", len(bwi_df))

# ===========================================
# 2. Download ASOS (with robust parsing)
# ===========================================
def download_asos(airport, year):
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={airport}&data=all"
        f"&year1={year}&month1=1&day1=1"
        f"&year2={year}&month2=12&day2=31"
        f"&tz=UTC&format=comma&latlon=no"
    )

    print(f"[INFO] Downloading ASOS for {airport} {year} ...")
    r = requests.get(url)
    text = r.text.strip()

    if len(text) < 100:
        print(f"[WARN] No ASOS data for {airport} {year}")
        return None

    if "<html" in text.lower():
        print(f"[WARN] HTML error for {airport} {year}")
        return None

    try:
        df = pd.read_csv(StringIO(text), comment="#", low_memory=False)
    except Exception as e:
        print(f"[WARN] Failed to parse ASOS for {airport} {year}: {e}")
        print("First 200 chars:", text[:200])
        return None

    if "valid" not in df.columns:
        print(f"[WARN] Missing 'valid' field in ASOS for {airport} {year}")
        return None

    df["weather_time"] = pd.to_datetime(df["valid"])
    df["station"] = airport
    return df


# ===========================================
# 3. Download BWI weather for years in dataset
# ===========================================
years = bwi_df["YEAR"].unique().tolist()
weather_list = []

for yr in years:
    df = download_asos("BWI", yr)
    if df is not None:
        weather_list.append(df)

weather_df = pd.concat(weather_list, ignore_index=True)
weather_df = weather_df.sort_values("weather_time")
weather_df["DEST"] = weather_df["station"]

print("Step 5: weather_list length =", len(weather_list))

# ===========================================
# 4. Extract & Clean Weather Fields
# ===========================================

# Rename ASOS raw columns to readable names
weather_df.rename(columns={
    "tmpf": "temp_F",              # temperature (F)
    "drct": "wind_dir",            # wind direction (degrees)
    "sknt": "wind_speed_knots",    # wind speed in knots
    "gust": "wind_gust_knots",     # wind gust in knots
    "vsby": "visibility_mi",       # visibility in miles
    "p01i": "precip_in",           # precipitation in inches (last hour)
    "metar": "raw_metar"           # raw METAR text
}, inplace=True)


# -------------------------------------------------------
# 4.1 Ensure numeric values for speed/visibility/precip
# -------------------------------------------------------
numeric_cols = ["wind_speed_knots", "wind_gust_knots",
                "visibility_mi", "precip_in"]

for col in numeric_cols:
    weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")


# -------------------------------------------------------
# 4.2 Unit Conversion
# -------------------------------------------------------
# knots → meters/second
weather_df["wind_speed"] = weather_df["wind_speed_knots"] * 0.514444
weather_df["wind_gust"] = weather_df["wind_gust_knots"] * 0.514444

# miles → meters
weather_df["visibility"] = weather_df["visibility_mi"] * 1609.34


# -------------------------------------------------------
# 4.3 Weather Phenomenon Flags (based on METAR)
# -------------------------------------------------------
def flag(df, code):
    """Create binary flag for METAR conditions, e.g., 'RA', 'SN', 'FG'."""
    return df["raw_metar"].fillna("").str.contains(code).astype(int)

weather_df["rain_flag"] = flag(weather_df, "RA")
weather_df["snow_flag"] = flag(weather_df, "SN")
weather_df["thunder_flag"] = flag(weather_df, "TS")
weather_df["fog_flag"] = flag(weather_df, "FG")
weather_df["ice_flag"] = flag(weather_df, "IC")   # icing conditions


# -------------------------------------------------------
# 4.4 Low Cloud Ceiling (OVC/BKN)
# -------------------------------------------------------
import re

def extract_ceiling(metar):
    if pd.isna(metar):
        return np.nan
    match = re.search(r"(OVC|BKN)(\d{3})", metar)
    if match:
        return int(match.group(2)) * 100   # hundreds of ft → feet
    return np.nan

weather_df["ceiling_ft"] = weather_df["raw_metar"].apply(extract_ceiling)
weather_df["low_ceiling_flag"] = (
    weather_df["ceiling_ft"].fillna(99999) < 1000
).astype(int)

print("Step 6: weather_df rows =", len(weather_df))
print("Step 6: bwi_df rows before merge =", len(bwi_df))

# ===========================================
# 5. Merge weather with BWI flights
# ===========================================
bwi_df = bwi_df.sort_values("ARR_TIME_dt")
weather_df = weather_df.sort_values("weather_time")

# # Check DEST in both tables
# missing_dest_left = bwi_df[bwi_df["DEST"].isna()]
# missing_dest_right = weather_df[weather_df["DEST"].isna()]

# print("\nMissing DEST (left bwi_df):", len(missing_dest_left))
# print(missing_dest_left.head())

# print("\nMissing DEST (right weather_df):", len(missing_dest_right))
# print(missing_dest_right.head())

# # Check ARR_TIME_dt
# missing_time_left = bwi_df[bwi_df["ARR_TIME_dt"].isna()]
# print("\nMissing ARR_TIME_dt in bwi_df:", len(missing_time_left))
# print(missing_time_left.head())

# # Check weather_time
# missing_time_right = weather_df[weather_df["weather_time"].isna()]
# print("\nMissing weather_time in weather_df:", len(missing_time_right))
# print(missing_time_right.head())

merged = pd.merge_asof(
    bwi_df,
    weather_df,
    left_on="ARR_TIME_dt",
    right_on="weather_time",
    by="DEST",
    direction="nearest",
    tolerance=pd.Timedelta("45min")
)


# ===========================================
# 6. Save
# ===========================================
merged.to_csv("../data/ref/bwi_flights_with_weather.csv", index=False)
print("[DONE] Saved to bwi_flights_with_weather.csv")
