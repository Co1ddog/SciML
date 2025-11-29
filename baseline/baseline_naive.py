import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ 1) Load CSV ============
CSV_PATH = "../data/ref/data_set.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')
print(f"[INFO] Loaded rows: {len(df):,}")

# ============ 2) Identify columns ============
delay_col = 'ARR_DELAY_NEW' if 'ARR_DELAY_NEW' in df.columns else (
            'ARR_DELAY' if 'ARR_DELAY' in df.columns else (
            'ArrDelayMinutes' if 'ArrDelayMinutes' in df.columns else None))

taxi_col = 'TAXI_IN' if 'TAXI_IN' in df.columns else (
           'AvgTaxiIn' if 'AvgTaxiIn' in df.columns else None)

if delay_col is None: raise RuntimeError("Delay column not found!")
if taxi_col  is None: raise RuntimeError("Taxi-in column not found!")

has_num_arrivals = 'NumArrivals' in df.columns

# ============ 3) 5-min aggregation ============
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')
g_targets = df.groupby('__bin5').agg({
    delay_col: 'mean',
    taxi_col: 'mean'
})

if has_num_arrivals:
    g_targets['__count'] = df.groupby('__bin5')['NumArrivals'].sum()
else:
    g_targets['__count'] = df.groupby('__bin5').size()

# Reindex to continuous grid
full_idx = pd.date_range(
    start=g_targets.index.min(),
    end=g_targets.index.max(),
    freq='5min'
)
g_targets = g_targets.reindex(full_idx)

# Fill missing
g_targets[delay_col] = g_targets[delay_col].ffill().fillna(0)
g_targets[taxi_col]  = g_targets[taxi_col].ffill().fillna(0)
g_targets['__count'] = g_targets['__count'].fillna(0)

# Convert to numpy
delay_true = g_targets[delay_col].to_numpy(np.float32)
taxi_true  = g_targets[taxi_col].to_numpy(np.float32)
count_true = g_targets['__count'].to_numpy(np.float32)

T = len(delay_true)
print("[INFO] 5-min steps:", T)

# ============ 4) Naive Prediction: pred[t] = true[t-1] ============
def naive_forecast(y):
    y = np.asarray(y)
    pred = np.zeros_like(y)
    pred[0] = y[0]
    pred[1:] = y[:-1]
    return pred

delay_pred = naive_forecast(delay_true)
taxi_pred  = naive_forecast(taxi_true)
count_pred = naive_forecast(count_true)

# ============ 5) Evaluation ============
def mse(a, b):
    return float(np.mean((a - b)**2))

print("\n========== Naive Baseline Performance ==========")
print("Delay MSE:", mse(delay_pred, delay_true))
print("Taxi-In MSE:", mse(taxi_pred, taxi_true))
print("Count MSE:", mse(count_pred, count_true))
print("================================================\n")

# ============ 6) Visualization ============
time_axis = np.arange(T)

plt.figure(figsize=(14, 8))

plt.subplot(3,1,1)
plt.plot(delay_true, label="True Delay", color="black", alpha=0.6)
plt.plot(delay_pred, label="Pred Delay", color="red", alpha=0.8)
plt.title("Arrival Delay (Naive Forecast)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3,1,2)
plt.plot(taxi_true, label="True TaxiIn", color="black", alpha=0.6)
plt.plot(taxi_pred, label="Pred TaxiIn", color="blue", alpha=0.8)
plt.title("Taxi-In Time (Naive Forecast)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3,1,3)
plt.plot(count_true, label="True Count", color="black", alpha=0.6)
plt.plot(count_pred, label="Pred Count", color="green", alpha=0.8)
plt.title("Number of Arrivals (Naive Forecast)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_naive_results.png", dpi=200)
plt.show()

print("[INFO] Plot saved to baseline_naive_results.png")
