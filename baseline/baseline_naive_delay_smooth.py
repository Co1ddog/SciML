# ==============================================================
# baseline_naive_delay_smooth.py (UPDATED WITH TRAIN/VAL/TEST SPLIT)
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# 1) Load CSV
# ==============================================================
CSV_PATH = "../data/ref/bwi_flights_data.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')
print("[INFO] loaded rows:", len(df))

TARGET = "ARR_DELAY_NEW"
if TARGET not in df.columns:
    raise RuntimeError(f"[ERROR] target {TARGET} not found.")

# ==============================================================
# 2) 5-min Aggregation (Same as NODE)
# ==============================================================
df["__bin5"] = df["CRS_ARR_TIME_dt"].dt.floor("5min")

g_raw = df.groupby("__bin5")[TARGET].mean()

full_idx = pd.date_range(
    start=g_raw.index.min(),
    end=g_raw.index.max(),
    freq="5min"
)

g_raw = g_raw.reindex(full_idx).fillna(0.0)

# ==============================================================
# 3) Target smoothing
# ==============================================================
y_raw = np.maximum(g_raw.to_numpy(np.float32), 0.0)
y_log = np.log1p(y_raw)

y_log_smooth = (
    pd.Series(y_log, index=full_idx)
      .rolling(12, min_periods=1)
      .mean()
      .to_numpy(np.float32)
)

y_delay = y_log_smooth

# ==============================================================
# 4) Time truncation (same as NODE)
# ==============================================================
MAX_STEPS = 5000
if len(y_delay) > MAX_STEPS:
    start = len(y_delay) - MAX_STEPS
    print(f"[INFO] truncate to last {MAX_STEPS} steps")
    y_delay = y_delay[start:]
    full_idx = full_idx[start:]

T_total = len(y_delay)
print("[INFO] final steps:", T_total)

true_log = y_delay
true_min = np.expm1(true_log)

# ==============================================================
# 5) Train / Validation / Test Split (Unified with NODE/LSTM/MLP)
# ==============================================================
train_end = int(0.70 * T_total)
val_end   = int(0.85 * T_total)

train_slice = slice(0, train_end)
valid_slice = slice(train_end, val_end)
test_slice  = slice(val_end, T_total)

print("[INFO] data split:")
print("  Train      :", train_end)
print("  Validation :", val_end - train_end)
print("  Test       :", T_total - val_end)

# ==============================================================
# 6) Naive Forecast (persistence)
# ==============================================================
def naive_forecast(y):
    pred = np.zeros_like(y)
    pred[0] = y[0]
    pred[1:] = y[:-1]
    return pred

pred_log = naive_forecast(true_log)
pred_min = np.expm1(pred_log)

# ==============================================================
# 7) Evaluate only on TEST SET (for fairness)
# ==============================================================
true_test_log = true_log[test_slice]
pred_test_log = pred_log[test_slice]

true_test_min = np.expm1(true_test_log)
pred_test_min = np.expm1(pred_test_log)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def r2(true, pred):
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    return float(1 - ss_res / ss_tot)

print("\n========== Naive Baseline Metrics (TEST SET) ==========")
print("RMSE:", rmse(pred_test_min, true_test_min))
print("MAE :", mae(pred_test_min, true_test_min))
print("R²  :", r2(true_test_min, pred_test_min))
print("=======================================================\n")

# ==============================================================
# 8) Plot full sequence (for visualization)
# ==============================================================
plt.figure(figsize=(14,6))
plt.plot(true_min, label="Smoothed Delay (truth)", alpha=0.9)
plt.plot(pred_min, label="Predicted Smoothed Delay (Naive)", alpha=0.9)
plt.legend()
plt.grid()
plt.title("Naive Prediction of Smoothed Arrival Delay")
plt.xlabel("5-min steps")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("baseline_naive_delay_smooth_results.png", dpi=200)
plt.show()

print("[INFO] saved baseline_naive_delay_smooth_results.png")
