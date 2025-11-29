import pandas as pd

df = pd.read_csv("../data/ref/bwi_flights_with_weather.csv").copy()

weather_cols = [
    "temp_F", "wind_dir", "wind_speed_knots", "wind_gust_knots",
    "visibility_mi", "precip_in",
    "wind_speed", "wind_gust", "visibility",
    "rain_flag", "snow_flag", "thunder_flag", "fog_flag", "ice_flag",
    "ceiling_ft", "low_ceiling_flag"
]

weather_cols = [c for c in weather_cols if c in df.columns]

# 计算统计范围
summary = df[weather_cols].agg(
    ["min", "max", "mean", "median", "std"]
).T

# 缺失率
summary["missing_ratio"] = df[weather_cols].isna().mean()

print("\n===== Weather Feature Value Ranges =====\n")
print(summary)
