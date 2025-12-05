# ==============================================================
# baseline_lstm_delay_smooth.py
# LSTM baseline for Smoothed Arrival Delay (log1p + 1h rolling mean)
# ==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ==============================================================
# 1) Load CSV
# ==============================================================
CSV_PATH = "../data/ref/bwi_flights_data.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values("CRS_ARR_TIME_dt")
print("[INFO] loaded rows:", len(df))

# ==============================================================
# 2) 5-min aggregation (same as NODE)
# ==============================================================
df["__bin5"] = df["CRS_ARR_TIME_dt"].dt.floor("5min")

TARGET = "ARR_DELAY_NEW"
if TARGET not in df.columns:
    raise RuntimeError(f"[ERROR] target {TARGET} not found.")

g_raw = df.groupby("__bin5")[TARGET].mean()
g_count = df.groupby("__bin5").size()

u_cols_time = [
    "HOUR_SIN","HOUR_COS",
    "DOW_SIN","DOW_COS",
    "MONTH_SIN","MONTH_COS",
    "MIN_OF_DAY","ARR_HOUR"
]
u_cols_weather = [
    "visibility","low_ceiling_flag","wind_speed",
    "rain_flag","snow_flag","thunder_flag","fog_flag","ice_flag",
    "syn_vis","syn_cloud","syn_wind","syn_precip","syn_temp_dev"
]
u_cols_holiday = ["holiday_flag"]

u_all = [c for c in (u_cols_time + u_cols_weather + u_cols_holiday) if c in df.columns]
g_u = df.groupby("__bin5")[u_all].mean()

full_idx = pd.date_range(
    min(g_raw.index.min(), g_u.index.min()),
    max(g_raw.index.max(), g_u.index.max()),
    freq="5min"
)

g_raw = g_raw.reindex(full_idx).fillna(0.0)
g_u = g_u.reindex(full_idx)
g_count = g_count.reindex(full_idx).fillna(0.0)

# ==============================================================
# 3) Smoothing: clip + log + rolling
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
# 4) Prepare input U
# ==============================================================
U = g_u.to_numpy(np.float32)
U = np.nan_to_num(U, nan=0.0)

arr_1h = (
    pd.Series(g_count, index=full_idx)
    .rolling(12, min_periods=1)
    .sum()
    .to_numpy(np.float32)
)
U = np.concatenate([U, arr_1h[:, None]], axis=1)
u_cols = u_all + ["arrivals_1h"]

# ==============================================================
# 5) Time truncation
# ==============================================================
MAX_STEPS = 5000
if len(U) > MAX_STEPS:
    start = len(U) - MAX_STEPS
    print(f"[INFO] truncate to last {MAX_STEPS} steps")
    U = U[start:]
    y_delay = y_delay[start:]
    full_idx = full_idx[start:]

T_total = len(U)
print("[INFO] final steps:", T_total)

# ==============================================================
# 6) Normalization
# ==============================================================
def zscore(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-8
    return (x - mu) / sd, mu, sd

y_delay_n, mu_delay, sd_delay = zscore(y_delay)

U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U = (U - U_mu) / U_sd
U = np.clip(np.nan_to_num(U), -5.0, 5.0)

# ==============================================================
# 7) Split: 70% train / 15% val / 15% test
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

U_t = torch.tensor(U, dtype=torch.float32, device=device)
y_t = torch.tensor(y_delay_n, dtype=torch.float32, device=device)

T = len(U_t)
train_end = int(0.70 * T)
val_end   = int(0.85 * T)

train_slice = slice(0, train_end)
valid_slice = slice(train_end, val_end)
test_slice  = slice(val_end, T)

print("[INFO] data split:")
print("  Train:", train_end)
print("  Val  :", val_end - train_end)
print("  Test :", T - val_end)

# ==============================================================
# 8) LSTM Model
# ==============================================================
class LSTMDelay(nn.Module):
    def __init__(self, u_dim, hdim=64):
        super().__init__()
        self.lstm = nn.LSTM(u_dim, hdim, batch_first=True)
        self.out = nn.Linear(hdim, 1)

    def forward(self, U_seq):
        h, _ = self.lstm(U_seq)
        return self.out(h).squeeze(-1)

model = LSTMDelay(U_t.shape[1]).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# ==============================================================
# 9) Training (TBPTT)
# ==============================================================
epochs = 20
window = 128

for ep in range(1, epochs+1):
    model.train()
    total = 0
    batches = 0
    start = train_slice.start

    while start < train_slice.stop:
        end = min(start + window, train_slice.stop)
        if end - start < 2:
            break

        Ub = U_t[start:end].unsqueeze(0)
        yb = y_t[start:end]

        pred = model(Ub).squeeze(0)
        loss = mse(pred, yb)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item()
        batches += 1
        start = end

    # Validation
    model.eval()
    with torch.no_grad():
        pv = model(U_t[valid_slice].unsqueeze(0)).squeeze(0)
        val = mse(pv, y_t[valid_slice]).item()

    print(f"[LSTM] Epoch {ep:02d} | train {total/batches:.4f} | val {val:.4f}")

# ==============================================================
# 10) Predict: Full sequence + Test-only evaluation
# ==============================================================
model.eval()
with torch.no_grad():
    pred_n_full = model(U_t.unsqueeze(0)).squeeze(0).cpu().numpy()
    pred_n_test = model(U_t[test_slice].unsqueeze(0)).squeeze(0).cpu().numpy()

# ---- Denormalize ----
pred_log_full = pred_n_full * sd_delay + mu_delay
pred_log_test = pred_n_test * sd_delay + mu_delay

true_log_full = y_delay
true_log_test = y_delay[test_slice]

pred_min_full = np.expm1(pred_log_full)
pred_min_test = np.expm1(pred_log_test)
true_min_full = np.expm1(true_log_full)
true_min_test = np.expm1(true_log_test)

# ==============================================================
# 11) Evaluation Metrics
# ==============================================================
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def r2(true, pred):
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    return float(1 - ss_res / ss_tot)

print("\n========== LSTM Baseline Metrics (TEST SET) ==========")
print("RMSE:", rmse(pred_min_test, true_min_test))
print("MAE :", mae(pred_min_test, true_min_test))
print("R²  :", r2(true_min_test, pred_min_test))
print("======================================================\n")

# ==============================================================
# 12) Plot
# ==============================================================
plt.figure(figsize=(14,6))
plt.plot(true_min_full, label="Smoothed Delay (truth)", alpha=0.9)
plt.plot(pred_min_full,  label="Predicted Smoothed Delay (LSTM)", alpha=0.9)
plt.legend()
plt.grid()
plt.title("LSTM Prediction of Smoothed Arrival Delay")
plt.xlabel("5-min steps")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("baseline_lstm_delay_smooth_results.png", dpi=200)
plt.show()

print("[INFO] saved baseline_lstm_delay_smooth_results.png")
