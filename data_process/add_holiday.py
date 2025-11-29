import pandas as pd
import holidays

df = pd.read_csv("../data/ref/bwi_flights_with_weather.csv")

# 1. parse FL_DATE
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

# 2. build holiday calendar
years = df["FL_DATE"].dt.year.unique().tolist()
us_holidays = holidays.US(years=years)

# 3. holiday flag (NO astype needed)
df["holiday_flag"] = df["FL_DATE"].dt.date.isin(us_holidays).astype(int)

# 4. holiday name
df["holiday_name"] = df["FL_DATE"].dt.date.map(us_holidays).fillna("None")

# 5. save
df.to_csv("../data/ref/bwi_flights_with_weather_holiday.csv", index=False)

print("[DONE] Holiday columns added!")
