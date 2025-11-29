import pandas as pd

# 假设 merged 就是你已经加入天气后的数据
df = pd.read_csv("../data/ref/bwi_flights_with_weather.csv").copy()

# 天气相关列（从我刚才给的完整列表中筛掉 raw_metar、station 等文字列）
weather_cols = [
    "temp_F", "wind_dir", "wind_speed_knots", "wind_gust_knots",
    "visibility_mi", "precip_in",
    "wind_speed", "wind_gust", "visibility",
    "rain_flag", "snow_flag", "thunder_flag", "fog_flag", "ice_flag",
    "ceiling_ft", "low_ceiling_flag"
]

# 保留存在的列并且是数值型
weather_cols = [c for c in weather_cols if c in df.columns]

# 计算相关性
corr = df[weather_cols + ["ARR_DELAY"]].corr()["ARR_DELAY"].drop("ARR_DELAY")

# Top-10 （按绝对值排序）
top10 = corr.abs().sort_values(ascending=False).head(10)

print("\n===== TOP 10 Weather Features Most Correlated with ARR_DELAY =====\n")
print(top10)
print("\n(Showing absolute correlations; you can check signs via corr[c])")
