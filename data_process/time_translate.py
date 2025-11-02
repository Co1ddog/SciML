import pandas as pd
import numpy as np

def hhmm_to_hour_min(val):
    """
    将 703/703.0/'703'/'0703' → (hour=7, minute=3)
    NaN/空返回 (None, None)
    """
    if pd.isna(val):
        return None, None
    s = str(int(float(val))).zfill(4)  # 703.0 -> '0703'
    return int(s[:-2]), int(s[-2:])

def combine_date_hhmm(date_series, hhmm_series, tz=None):
    """
    将日期列 + HHMM 列 合成为 pandas datetime（本地时）
    """
    dt = pd.to_datetime(date_series, errors="coerce", infer_datetime_format=True)
    h, m = zip(*hhmm_series.map(hhmm_to_hour_min))
    out = dt + pd.to_timedelta(h, unit="h") + pd.to_timedelta(m, unit="m")
    out = pd.to_datetime(out, errors="coerce")
    if tz:
        out = out.dt.tz_localize(tz, nonexistent='NaT', ambiguous='NaT')
    return out

def add_cyc(x, period):
    ang = 2*np.pi*(x % period)/period
    return np.sin(ang), np.cos(ang)

def build_time_features(df, use_gate_open_buffer=True, buffer_min=1):
    """
    生成：
    - ARR_TIME_dt, CRS_ARR_TIME_dt, WHEELS_ON_dt
    - DOOR_OPEN_TS ≈ WHEELS_ON_dt + TAXI_IN (+1min 缓冲可选)
    - HOUR/DOW/MONTH/MIN_OF_DAY 的 sin/cos 周期特征
    """
    # 1) 各关键时刻
    df["ARR_TIME_dt"]      = combine_date_hhmm(df["FL_DATE"], df["ARR_TIME"])
    df["CRS_ARR_TIME_dt"]  = combine_date_hhmm(df["FL_DATE"], df["CRS_ARR_TIME"])
    df["WHEELS_ON_dt"]     = combine_date_hhmm(df["FL_DATE"], df["WHEELS_ON"])

    # 2) 估计“开门时间”（建议用于 deplaning 起点）
    door_open = df["WHEELS_ON_dt"] + pd.to_timedelta(df["TAXI_IN"].fillna(0), unit="m")
    if use_gate_open_buffer:
        door_open = door_open + pd.to_timedelta(buffer_min, unit="m")
    df["DOOR_OPEN_TS"] = door_open

    # 3) 提取时间组成（以 DOOR_OPEN_TS 为准；若缺失则回退 ARR_TIME_dt）
    ts = df["DOOR_OPEN_TS"].where(df["DOOR_OPEN_TS"].notna(), df["ARR_TIME_dt"])
    df["ARR_HOUR"] = ts.dt.hour
    df["DAY_OF_WEEK_ENC"] = pd.to_datetime(df["FL_DATE"]).dt.weekday   # 0=Mon..6=Sun
    df["MONTH_ENC"] = pd.to_datetime(df["FL_DATE"]).dt.month           # 1..12
    df["MIN_OF_DAY"] = ts.dt.hour*60 + ts.dt.minute                     # 0..1439

    # 4) 周期编码
    df["HOUR_SIN"],  df["HOUR_COS"]  = add_cyc(df["ARR_HOUR"], 24)
    df["DOW_SIN"],   df["DOW_COS"]   = add_cyc(df["DAY_OF_WEEK_ENC"], 7)
    df["MONTH_SIN"], df["MONTH_COS"] = add_cyc(df["MONTH_ENC"]-1, 12)  # 0..11
    df["MOD_SIN"],   df["MOD_COS"]   = add_cyc(df["MIN_OF_DAY"], 1440)

    return df

# === 批量处理 ===
datasets = ["train", "test", "validation"]
for name in datasets:
    path_in  = f"../data/ref/T_ONTIME_REPORTING_{name}_ALL_enriched.csv"
    path_out = f"../data/ref/{name}_set.csv"

    df = pd.read_csv(path_in)
    df = build_time_features(df, use_gate_open_buffer=True, buffer_min=1)

    df.to_csv(path_out, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{path_out}")
