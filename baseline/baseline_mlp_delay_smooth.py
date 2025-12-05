# ==============================================================
# baseline_mlp_delay_smooth.py
# MLP baseline for Smoothed Arrival Delay (log1p + 1h rolling mean)
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
df = pd.read_csv(CSV_PATH, parse_dates=["CRS_ARR_TIME_dt"], low_memory=False)
df = df.sort_values("CRS_ARR_TIME_dt")
print("[INFO] loaded rows:", len(df))

# ==============================================================
# 2) 5-min aggregation (same as NODE)
# ==============================================================
df["__bin5"] = df["CRS_ARR_TIME_dt"].dt.floor("5min")

TARGET = "ARR_DELAY_NEW"
if TARGET not in df.columns:
    raise RuntimeError(f"[ERROR] target {TARGET} not found.")

# mean delay per 5-min bin
g_raw = df.groupby("__bin5")[TARGET].mean()

# arrival count per bin
g_count = df.groupby("__bin5").size()

# feature columns (same family as NODE / LSTM smooth)
u_cols_time = [
    "HOUR_SIN", "HOUR_COS",
    "DOW_SIN", "DOW_COS",
    "MONTH_SIN", "MONTH_COS",
    "MIN_OF_DAY", "ARR_HOUR"
]
u_cols_weather = [
    "visibility", "low_ceiling_flag", "wind_speed",
    "rain_flag", "snow_flag", "thunder_flag", "fog_flag", "ice_flag",
    "syn_vis", "syn_cloud", "syn_wind", "syn_precip", "syn_temp_dev"
]
u_cols_holiday = ["holiday_flag"]

u_all = [c for c in (u_cols_time + u_cols_weather + u_cols_holiday) if c in df.columns]

g_u = df.groupby("__bin5")[u_all].mean()

# 完整时间轴
full_idx = pd.date_range(
    min(g_raw.index.min(), g_u.index.min()),
    max(g_raw.index.max(), g_u.index.max()),
    freq="5min"
)

g_raw = g_raw.reindex(full_idx).fillna(0.0)
g_u = g_u.reindex(full_idx)
g_count = g_count.reindex(full_idx).fillna(0.0)

# ==============================================================
# 3) Target smoothing: clip + log1p + 1h rolling mean
# ==============================================================
y_raw = np.maximum(g_raw.to_numpy(np.float32), 0.0)
y_log = np.log1p(y_raw)

y_log_smooth = (
    pd.Series(y_log, index=full_idx)
      .rolling(12, min_periods=1)
      .mean()
      .to_numpy(np.float32)
)

# final target (NODE / LSTM / MLP 共用)
y_delay = y_log_smooth

# ==============================================================
# 4) Prepare input U
# ==============================================================
U = g_u.to_numpy(np.float32)
U = np.nan_to_num(U, nan=0.0)

# add arrivals_1h
arr_1h = (
    pd.Series(g_count, index=full_idx)
      .rolling(12, min_periods=1)
      .sum()
      .to_numpy(np.float32)
)
U = np.concatenate([U, arr_1h[:, None]], axis=1)
u_cols = u_all + ["arrivals_1h"]

# ==============================================================
# 5) Time truncation (same as NODE / LSTM)
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
# 6) Normalization (same as NODE / LSTM)
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
# 7) Tensors & Split (70% train / 15% val / 15% test)
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
print("  Train      :", train_end)
print("  Validation :", val_end - train_end)
print("  Test       :", T - val_end)
print(f"[INFO] U-dim = {U_t.shape[1]}, steps = {T}")

# ==============================================================
# 8) MLP model
# ==============================================================
class MLPDelay(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

mlp = MLPDelay(input_dim=U_t.shape[1]).to(device)
optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
mse = nn.MSELoss()

# ==============================================================
# 9) Training (supervised regression)
# ==============================================================
epochs = 20
for ep in range(1, epochs + 1):
    mlp.train()
    pred = mlp(U_t[train_slice])
    loss = mse(pred, y_t[train_slice])

    optim.zero_grad()
    loss.backward()
    optim.step()

    # validation
    mlp.eval()
    with torch.no_grad():
        pv = mlp(U_t[valid_slice])
        val = mse(pv, y_t[valid_slice]).item()

    print(f"[MLP] Epoch {ep:02d} | TrainLoss={loss.item():.4f} | ValLoss={val:.4f}")

print("\n[INFO] MLP training finished.\n")

# ==============================================================
# 10) Prediction: full sequence + test-only evaluation
# ==============================================================
mlp.eval()
with torch.no_grad():
    pred_n_full = mlp(U_t).cpu().numpy()
    pred_n_test = mlp(U_t[test_slice]).cpu().numpy()

# inverse transform
pred_log_full = pred_n_full * sd_delay + mu_delay
pred_log_test = pred_n_test * sd_delay + mu_delay

true_log_full = y_delay
true_log_test = y_delay[test_slice]

pred_min_full = np.expm1(pred_log_full)
pred_min_test = np.expm1(pred_log_test)
true_min_full = np.expm1(true_log_full)
true_min_test = np.expm1(true_log_test)

# ==============================================================
# 11) Metrics (same format as NODE/LSTM)
# ==============================================================
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def r2(true, pred):
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    return float(1 - ss_res / ss_tot)

print("========== MLP Baseline Metrics (TEST SET) ==========")
print("RMSE:", rmse(pred_min_test, true_min_test))
print("MAE :", mae(pred_min_test, true_min_test))
print("R²  :", r2(true_min_test, pred_min_test))
print("=====================================================\n")

# ==============================================================
# 12) Plot
# ==============================================================
plt.figure(figsize=(14, 6))
plt.plot(true_min_full, label="Smoothed Delay (truth)", alpha=0.9)
plt.plot(pred_min_full, label="Predicted Smoothed Delay (MLP)", alpha=0.9)
plt.legend()
plt.grid()
plt.title("MLP Prediction of Smoothed Arrival Delay")
plt.xlabel("5-min steps")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("baseline_mlp_delay_smooth_results.png", dpi=200)
plt.show()

print("[INFO] saved baseline_mlp_delay_smooth_results.png")
