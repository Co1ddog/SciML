import numpy as np
import pandas as pd

# 输入文件（根据你路径）
in_path = "/Users/colddog/Desktop/课程/SciML/SciML/data/ref/bwi_flights_with_weather_holiday.csv"
out_path = "/Users/colddog/Desktop/课程/SciML/SciML/data/ref/bwi_flights_with_weather_holiday_synth.csv"

# -------------------------------------------------------
# 1. 读取 CSV
# -------------------------------------------------------
df = pd.read_csv(in_path)
print(f"Loaded dataframe with {len(df)} rows")

# -------------------------------------------------------
# 2. 选择用于构造天气的延误列
# -------------------------------------------------------
delay_col = None
if "ARR_DELAY_NEW" in df.columns:
    delay_col = "ARR_DELAY_NEW"
elif "ARR_DELAY" in df.columns:
    delay_col = "ARR_DELAY"
else:
    raise ValueError("❌ CSV 中找不到 ARR_DELAY_NEW 或 ARR_DELAY，请检查列名")

print(f"Using delay column: {delay_col}")

# 延误值
delay = df[delay_col].fillna(0).values
delay = np.maximum(0, delay)  # 不允许负数
max_delay = delay.max() if delay.max() > 0 else 1
delay_norm = delay / max_delay

# -------------------------------------------------------
# 3. 生成 synthetic weather columns
#    (延误越大 → 天气越差)
# -------------------------------------------------------
np.random.seed(42)

syn_vis       = 10 - 8*delay_norm + np.random.normal(0,0.2,len(delay))
syn_cloud     = 2 + 8*delay_norm + np.random.normal(0,0.3,len(delay))
syn_wind      = 3 + 12*delay_norm + np.random.normal(0,0.5,len(delay))
syn_precip    = 0.5*delay_norm + np.random.normal(0,0.1,len(delay))
syn_temp_dev  = 5*(delay_norm**2) + np.random.normal(0,0.2,len(delay))

# Clip 确保范围真实
syn_vis    = np.clip(syn_vis, 0.1, 10)
syn_cloud  = np.clip(syn_cloud, 0, 10)
syn_wind   = np.clip(syn_wind, 0, 40)
syn_precip = np.clip(syn_precip, 0, 5)

# -------------------------------------------------------
# 4. 加入 DataFrame
# -------------------------------------------------------
df["syn_vis"] = syn_vis
df["syn_cloud"] = syn_cloud
df["syn_wind"] = syn_wind
df["syn_precip"] = syn_precip
df["syn_temp_dev"] = syn_temp_dev

print("Added synthetic weather columns:")
print(["syn_vis", "syn_cloud", "syn_wind", "syn_precip", "syn_temp_dev"])

# -------------------------------------------------------
# 5. 保存文件
# -------------------------------------------------------
df.to_csv(out_path, index=False)
print(f"✅ Saved new CSV to:\n{out_path}")
