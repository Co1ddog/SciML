# ========================================================
# baseline_lstm.py
# Fully standalone LSTM baseline for:
#   delay, taxi-in, count (arrivals per 5min)
# ========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ======================================================
# 1) Load CSV
# ======================================================
CSV_PATH = "../data/ref/data_set.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')
print(f"[INFO] Loaded rows: {len(df):,}")

# ======================================================
# 2) Identify target columns
# ======================================================
delay_col = 'ARR_DELAY_NEW' if 'ARR_DELAY_NEW' in df.columns else (
            'ARR_DELAY' if 'ARR_DELAY' in df.columns else (
            'ArrDelayMinutes' if 'ArrDelayMinutes' in df.columns else None))

taxi_col = 'TAXI_IN' if 'TAXI_IN' in df.columns else (
           'AvgTaxiIn' if 'AvgTaxiIn' in df.columns else None)

if delay_col is None:
    raise RuntimeError("Delay column not found!")
if taxi_col is None:
    raise RuntimeError("Taxi-in column not found!")

has_num_arrivals = 'NumArrivals' in df.columns

# ======================================================
# 3) 5-min aggregation (same logic as NODE)
# ======================================================
df["__bin5"] = df["CRS_ARR_TIME_dt"].dt.floor("5min")

g_targets = df.groupby("__bin5").agg({
    delay_col: "mean",
    taxi_col: "mean",
})

if has_num_arrivals:
    g_targets["__count"] = df.groupby("__bin5")["NumArrivals"].sum()
else:
    g_targets["__count"] = df.groupby("__bin5").size()

# exogenous inputs U
u_primary  = ["sin_hour","cos_hour","Hour","DayOfWeek","Month","Holiday","DaysToHoliday"]
u_fallback = ["HOUR_SIN","HOUR_COS","DOW_SIN","DOW_COS","MONTH_SIN","MONTH_COS",
              "MOD_SIN","MOD_COS","DAY_OF_WEEK_ENC","MONTH_ENC","MIN_OF_DAY","ARR_HOUR"]

u_cols = [c for c in u_primary if c in df.columns] or [c for c in u_fallback if c in df.columns]
if not u_cols:
    raise RuntimeError("No usable exogenous input columns for U.")

g_u = df.groupby("__bin5")[u_cols].mean()

# Continuous grid
full_idx = pd.date_range(
    start=min(g_targets.index.min(), g_u.index.min()),
    end=max(g_targets.index.max(), g_u.index.max()),
    freq="5min"
)
g_targets = g_targets.reindex(full_idx)
g_u = g_u.reindex(full_idx)

# Fill target missing
g_targets[delay_col] = g_targets[delay_col].ffill().fillna(0.0)
g_targets[taxi_col]  = g_targets[taxi_col].ffill().fillna(0.0)
g_targets["__count"] = g_targets["__count"].fillna(0.0)

# to numpy
delay_true = g_targets[delay_col].to_numpy(np.float32)
taxi_true  = g_targets[taxi_col].to_numpy(np.float32)
count_true = g_targets["__count"].to_numpy(np.float32)

# U to numpy
U = g_u.to_numpy(np.float32)

# standardize U
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U = (U - U_mu) / U_sd
U = np.nan_to_num(U, nan=0.0)
U = np.clip(U, -5.0, 5.0)

# add rolling 1h arrivals to U
arr_1h = pd.Series(g_targets["__count"], index=full_idx).rolling(12, min_periods=1).sum().to_numpy(np.float32)
U = np.concatenate([U, arr_1h[:, None]], axis=1)

# ======================================================
# 4) normalize targets (same flavor as NODE/MLP)
# ======================================================
def zscore(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-8
    return (x - mu) / sd, mu, sd

delay_n, delay_mu, delay_sd = zscore(delay_true)
taxi_n,  taxi_mu,  taxi_sd  = zscore(taxi_true)
count_lp = np.log1p(np.clip(count_true, 0, None)).astype(np.float32)

# ======================================================
# 5) Tensors & train/valid split
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

U_t = torch.tensor(U, dtype=torch.float32, device=device)
y_d = torch.tensor(delay_n, dtype=torch.float32, device=device)
y_t = torch.tensor(taxi_n,  dtype=torch.float32, device=device)
y_c = torch.tensor(count_lp, dtype=torch.float32, device=device)

T = U_t.shape[0]
split = int(0.8 * T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

print(f"[INFO] Total steps: {T}, Train: {split}, Valid: {T - split}")
print(f"[INFO] U-dim: {U_t.shape[1]}")

# ======================================================
# 6) LSTM model
# ======================================================
class LSTMBaseline(nn.Module):
    def __init__(self, u_dim, hdim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=u_dim,
            hidden_size=hdim,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(hdim, 3)  # delay, taxi, count

    def forward(self, U_seq):
        # U_seq: [B, T, u_dim]
        out_seq, _ = self.lstm(U_seq)   # [B, T, hdim]
        pred = self.out(out_seq)        # [B, T, 3]
        return pred

model = LSTMBaseline(u_dim=U_t.shape[1], hdim=64, num_layers=1).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# ======================================================
# 7) Training with simple TBPTT-style windows
# ======================================================
epochs = 20
window = 128   # sequence length

for ep in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    count_batches = 0

    # iterate over train_slice in chunks
    start = train_slice.start
    while start < train_slice.stop:
        end = min(start + window, train_slice.stop)
        if end - start <= 1:
            break

        U_blk = U_t[start:end].unsqueeze(0)   # [1, L, u_dim]
        yd_blk = y_d[start:end]
        yt_blk = y_t[start:end]
        yc_blk = y_c[start:end]

        pred_blk = model(U_blk).squeeze(0)    # [L, 3]

        loss = (mse(pred_blk[:, 0], yd_blk) +
                mse(pred_blk[:, 1], yt_blk) +
                mse(pred_blk[:, 2], yc_blk))

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        count_batches += 1
        start = end

    avg_train = total_loss / max(count_batches, 1)

    # validation
    model.eval()
    with torch.no_grad():
        Uv = U_t[valid_slice].unsqueeze(0)  # [1, T_valid, u_dim]
        pv = model(Uv).squeeze(0)           # [T_valid, 3]

        val_loss = (mse(pv[:,0], y_d[valid_slice]) +
                    mse(pv[:,1], y_t[valid_slice]) +
                    mse(pv[:,2], y_c[valid_slice])).item()

    print(f"[LSTM] Epoch {ep:02d}, TrainLoss={avg_train:.4f}, ValLoss={val_loss:.4f}")

print("\n[INFO] LSTM training finished.\n")

# ======================================================
# 8) Predict full series and inverse-transform
# ======================================================
model.eval()
with torch.no_grad():
    U_full = U_t.unsqueeze(0)          # [1, T, u_dim]
    pred_full = model(U_full).squeeze(0).cpu().numpy()  # [T,3]

delay_pred = pred_full[:, 0] * delay_sd + delay_mu
taxi_pred  = pred_full[:, 1] * taxi_sd  + taxi_mu
count_pred = np.expm1(pred_full[:, 2])
count_pred = np.clip(count_pred, 0, None)

# ======================================================
# 9) Compute MSE (original scale)
# ======================================================
def mse_np(a, b):
    return float(np.mean((a - b) ** 2))

print("========== LSTM Baseline Performance ==========")
print("Delay MSE:", mse_np(delay_pred, delay_true))
print("Taxi-In MSE:", mse_np(taxi_pred, taxi_true))
print("Count MSE:", mse_np(count_pred, count_true))
print("================================================\n")

# ======================================================
# 10) Plot results
# ======================================================
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(delay_true, label="True Delay", color="black", alpha=0.6)
plt.plot(delay_pred, label="LSTM Pred Delay", color="red", alpha=0.8)
plt.legend(); plt.grid(True, alpha=0.3); plt.title("Delay Prediction (LSTM)")

plt.subplot(3, 1, 2)
plt.plot(taxi_true, label="True TaxiIn", color="black", alpha=0.6)
plt.plot(taxi_pred, label="LSTM Pred TaxiIn", color="blue", alpha=0.8)
plt.legend(); plt.grid(True, alpha=0.3); plt.title("Taxi-In Prediction (LSTM)")

plt.subplot(3, 1, 3)
plt.plot(count_true, label="True Count", color="black", alpha=0.6)
plt.plot(count_pred, label="LSTM Pred Count", color="green", alpha=0.8)
plt.legend(); plt.grid(True, alpha=0.3); plt.title("Arrival Count Prediction (LSTM)")

plt.tight_layout()
plt.savefig("baseline_lstm_results.png", dpi=200)
plt.show()

print("[INFO] Plot saved as baseline_lstm_results.png")
