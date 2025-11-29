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

flights_df.dropna(subset=["DEST"], inplace=True)

print("Step 2: flights_df after DEST clean =", len(flights_df))

# Filter BWI flights
bwi_df = flights_df[flights_df["DEST"] == "BWI"].copy()

# Convert time columns
for col in ["ARR_TIME_dt", "CRS_ARR_TIME_dt", "WHEELS_ON_dt"]:
    bwi_df[col] = pd.to_datetime(bwi_df[col], errors="coerce")

# Remove invalid times
bwi_df = bwi_df[bwi_df["ARR_TIME_dt"].notna()].copy()

print("Step 3: bwi_df rows =", len(bwi_df))
print("Step 4: ARR_TIME_dt null count =", bwi_df["ARR_TIME_dt"].isna().sum())

# ===========================================
# 2. Download ASOS with robust parsing
# ===========================================
def download_asos(airport, year):
    """Download & parse ASOS from Iowa Mesonet."""
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={airport}&data=tmpf,drct,sknt,gust,vsby,p01i,metar"
        f"&year1={year}&month1=1&day1=1"
        f"&year2={year}&month2=12&day2=31"
        f"&tz=UTC&format=comma&latlon=no"
    )

    print(f"[INFO] Downloading ASOS for {airport} {year} ...")
    r = requests.get(url)
    text = r.text.strip()

    # Reject HTML
    if "<html" in text.lower() or len(text) < 50:
        print(f"[WARN] Failed ASOS {airport} {year}")
        return None

    # Read CSV, convert "M" to NaN
    try:
        df = pd.read_csv(
            StringIO(text), 
            comment="#",
            na_values=["M", ""],   # <==== Key fix
            keep_default_na=True
        )
    except Exception as e:
        print("[WARN] CSV parse error:", e)
        return None

    if "valid" not in df.columns:
        print(f"[WARN] No 'valid' field for {airport} {year}")
        return None

    df["weather_time"] = pd.to_datetime(df["valid"], errors="coerce")
    df["DEST"] = airport
    return df


# ===========================================
# 3. Download weather for BWI
# ===========================================
years = sorted(bwi_df["YEAR"].unique().tolist())
weather_list = []

for yr in years:
    df = download_asos("BWI", yr)
    if df is not None:
        weather_list.append(df)

print("Step 5: weather_list length =", len(weather_list))

weather_df = pd.concat(weather_list, ignore_index=True)
weather_df.sort_values("weather_time", inplace=True)

# ===========================================
# 4. Clean & Extract Weather Fields
# ===========================================
weather_df.rename(columns={
    "tmpf": "temp_F",
    "drct": "wind_dir",
    "sknt": "wind_speed_knots",
    "gust": "wind_gust_knots",
    "vsby": "visibility_mi",
    "p01i": "precip_in",
    "metar": "raw_metar"
}, inplace=True)

# Convert numeric
weather_numeric = [
    "temp_F", "wind_dir", "wind_speed_knots", "wind_gust_knots",
    "visibility_mi", "precip_in"
]

for col in weather_numeric:
    weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

# Derived units
weather_df["wind_speed"] = weather_df["wind_speed_knots"] * 0.514444
weather_df["wind_gust"] = weather_df["wind_gust_knots"] * 0.514444
weather_df["visibility"] = weather_df["visibility_mi"] * 1609.34

# Flags
def flag(df, code):
    return df["raw_metar"].fillna("").str.contains(code).astype(int)

weather_df["rain_flag"] = flag(weather_df, "RA")
weather_df["snow_flag"] = flag(weather_df, "SN")
weather_df["thunder_flag"] = flag(weather_df, "TS")
weather_df["fog_flag"] = flag(weather_df, "FG")
weather_df["ice_flag"] = flag(weather_df, "IC")

# Ceiling
def extract_ceiling(metar):
    if pd.isna(metar):
        return np.nan
    m = re.search(r"(OVC|BKN)(\d{3})", str(metar))
    return int(m.group(2)) * 100 if m else np.nan

weather_df["ceiling_ft"] = weather_df["raw_metar"].apply(extract_ceiling)
weather_df["low_ceiling_flag"] = (weather_df["ceiling_ft"] < 1000).astype(int)

print("Step 6: weather_df rows =", len(weather_df))
print("Step 6: bwi_df rows before merge =", len(bwi_df))

# ===========================================
# 5. ASOF Merge (final)
# ===========================================
bwi_df.sort_values("ARR_TIME_dt", inplace=True)
weather_df.sort_values("weather_time", inplace=True)

merged = pd.merge_asof(
    bwi_df,
    weather_df,
    left_on="ARR_TIME_dt",
    right_on="weather_time",
    by="DEST",
    direction="nearest",
    tolerance=pd.Timedelta("75min")   # <=== 45min → 75min 更稳
)

# ===========================================
# 6. Save
# ===========================================
merged.to_csv("../data/ref/bwi_flights_with_weather.csv", index=False)
print("[DONE] Saved to bwi_flights_with_weather.csv")
