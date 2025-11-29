import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

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

if delay_col is None: raise RuntimeError("Delay column not found!")
if taxi_col  is None: raise RuntimeError("Taxi-in column not found!")
has_num_arrivals = 'NumArrivals' in df.columns

# ======================================================
# 3) 5-min aggregation (same as NODE)
# ======================================================
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

g_targets = df.groupby('__bin5').agg({
    delay_col: 'mean',
    taxi_col: 'mean'
})

# number of arrivals
if has_num_arrivals:
    g_targets['__count'] = df.groupby('__bin5')['NumArrivals'].sum()
else:
    g_targets['__count'] = df.groupby('__bin5').size()

# Exogenous input U (same logic as your NODE)
u_primary  = ['sin_hour','cos_hour','Hour','DayOfWeek','Month','Holiday','DaysToHoliday']
u_fallback = ['HOUR_SIN','HOUR_COS','DOW_SIN','DOW_COS','MONTH_SIN','MONTH_COS',
              'MOD_SIN','MOD_COS','DAY_OF_WEEK_ENC','MONTH_ENC','MIN_OF_DAY','ARR_HOUR']

u_cols = [c for c in u_primary if c in df.columns] or [c for c in u_fallback if c in df.columns]
if not u_cols:
    raise RuntimeError("No exogenous U columns found.")

g_u = df.groupby('__bin5')[u_cols].mean()

# Continuous time grid
full_idx = pd.date_range(
    start=min(g_targets.index.min(), g_u.index.min()),
    end=max(g_targets.index.max(), g_u.index.max()),
    freq='5min'
)
g_targets = g_targets.reindex(full_idx)
g_u = g_u.reindex(full_idx)

# Fill missing
g_targets[delay_col] = g_targets[delay_col].ffill().fillna(0)
g_targets[taxi_col]  = g_targets[taxi_col].ffill().fillna(0)
g_targets['__count'] = g_targets['__count'].fillna(0)

# Numpy arrays
delay_true = g_targets[delay_col].to_numpy(np.float32)
taxi_true  = g_targets[taxi_col].to_numpy(np.float32)
count_true = g_targets['__count'].to_numpy(np.float32)

# Standardize U
U = g_u.to_numpy(np.float32)
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U = (U - U_mu) / U_sd
U = np.nan_to_num(U, nan=0.0)
U = np.clip(U, -5.0, 5.0)

# Add arrivals–1h rolling feature (same as NODE)
arr_1h = pd.Series(g_targets['__count'], index=full_idx).rolling(12, min_periods=1).sum().to_numpy(np.float32)
U = np.concatenate([U, arr_1h[:, None]], axis=1)

# ======================================================
# 4) Log-transform count + z-score delay & taxi
# ======================================================
def zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x)+1e-8
    return (x-mu)/sd, mu, sd

delay_n, delay_mu, delay_sd = zscore(delay_true)
taxi_n,  taxi_mu,  taxi_sd  = zscore(taxi_true)
count_lp = np.log1p(np.clip(count_true,0,None)).astype(np.float32)

# ######################################################
# Convert to tensors
# ######################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
U_t = torch.tensor(U, dtype=torch.float32, device=device)
y_d  = torch.tensor(delay_n, dtype=torch.float32, device=device)
y_t  = torch.tensor(taxi_n,  dtype=torch.float32, device=device)
y_c  = torch.tensor(count_lp, dtype=torch.float32, device=device)

T = len(U_t)
split = int(0.8 * T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

print(f"[INFO] U-dim = {U_t.shape[1]}, steps = {T}")

# ======================================================
# 5) MLP Model
# ======================================================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,3)  # delay, taxi, count
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP(input_dim=U_t.shape[1]).to(device)
optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
mse = nn.MSELoss()

# ======================================================
# 6) Training
# ======================================================
epochs = 20
for ep in range(1, epochs+1):
    mlp.train()
    optim.zero_grad()

    pred = mlp(U_t[train_slice])    # [T_train, 3]

    loss = (mse(pred[:,0], y_d[train_slice]) +
            mse(pred[:,1], y_t[train_slice]) +
            mse(pred[:,2], y_c[train_slice]))

    loss.backward()
    optim.step()

    # validation
    mlp.eval()
    with torch.no_grad():
        pv = mlp(U_t[valid_slice])
        val = (mse(pv[:,0], y_d[valid_slice]) +
               mse(pv[:,1], y_t[valid_slice]) +
               mse(pv[:,2], y_c[valid_slice])).item()

    print(f"[MLP] Epoch {ep:02d}, TrainLoss={loss.item():.4f}, Val={val:.4f}")

print("\n[INFO] Training finished.\n")

# ======================================================
# 7) Predict entire series + transform back
# ======================================================
with torch.no_grad():
    pred_all = mlp(U_t).cpu().numpy()

delay_pred = pred_all[:,0] * delay_sd + delay_mu
taxi_pred  = pred_all[:,1] * taxi_sd  + taxi_mu
count_pred = np.expm1(pred_all[:,2])
count_pred = np.clip(count_pred, 0, None)

# ======================================================
# 8) Compute MSE
# ======================================================
def mse_np(a,b): return float(np.mean((a-b)**2))

print("========== MLP Baseline Performance ==========")
print("Delay MSE:", mse_np(delay_pred, delay_true))
print("Taxi-In MSE:", mse_np(taxi_pred, taxi_true))
print("Count MSE:", mse_np(count_pred, count_true))
print("================================================\n")

# ======================================================
# 9) Plotting results
# ======================================================
plt.figure(figsize=(14,8))

plt.subplot(3,1,1)
plt.plot(delay_true, label="True Delay", color="black", alpha=0.6)
plt.plot(delay_pred, label="MLP Pred Delay", color="red")
plt.legend(); plt.grid(True); plt.title("Delay Prediction")

plt.subplot(3,1,2)
plt.plot(taxi_true, label="True Taxi", color="black", alpha=0.6)
plt.plot(taxi_pred, label="MLP Pred Taxi", color="blue")
plt.legend(); plt.grid(True); plt.title("Taxi-In")

plt.subplot(3,1,3)
plt.plot(count_true, label="True Count", color="black", alpha=0.6)
plt.plot(count_pred, label="MLP Pred Count", color="green")
plt.legend(); plt.grid(True); plt.title("Arrival Count")

plt.tight_layout()
plt.savefig("baseline_mlp_results.png", dpi=200)
plt.show()

print("[INFO] Results saved to baseline_mlp_results.png")
