"""
evaluate_node_model.py
-----------------------------------------
Evaluate a trained Controlled Neural ODE model
(saved as controlled_node_best.pt) on the full dataset.
Computes RMSE, MAE, and R² for Delay / TaxiIn / NumArrivals.

"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================================================
# 1. Environment & device
# ======================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[device] using {device}")

# ======================================================
# 2. Load data
# ======================================================
CSV_PATH = "./data/ref/train_set.csv"
print(f"[INFO] loading dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')

delay_col = 'ARR_DELAY_NEW' if 'ARR_DELAY_NEW' in df.columns else (
            'ARR_DELAY' if 'ARR_DELAY' in df.columns else (
            'ArrDelayMinutes' if 'ArrDelayMinutes' in df.columns else None))
taxi_col  = 'TAXI_IN' if 'TAXI_IN' in df.columns else (
            'AvgTaxiIn' if 'AvgTaxiIn' in df.columns else None)
if delay_col is None or taxi_col is None:
    raise RuntimeError("Delay or TaxiIn column not found in dataset.")

df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')
g_targets = df.groupby('__bin5').agg({
    delay_col: 'mean',
    taxi_col:  'mean'
})
g_targets['__count'] = df.groupby('__bin5').size()

# ======================================================
# 3. Load checkpoint to infer model dimensions
# ======================================================
ckpt = torch.load("controlled_node_best.pt", map_location=device)
state = ckpt["model"]
hdim_ckpt = state["h0"].shape[1]
in_expected = state["f.net.0.weight"].shape[1]
u_dim_expected = in_expected - hdim_ckpt
print(f"[ckpt] hdim={hdim_ckpt}, expected u_dim={u_dim_expected}")

# ======================================================
# 4. Select and prepare exogenous inputs
# ======================================================
u_primary  = ['sin_hour','cos_hour','Hour','DayOfWeek','Month','Holiday','DaysToHoliday']
u_fallback = ['HOUR_SIN','HOUR_COS','DOW_SIN','DOW_COS','MONTH_SIN','MONTH_COS',
              'MOD_SIN','MOD_COS','DAY_OF_WEEK_ENC','MONTH_ENC','MIN_OF_DAY','ARR_HOUR']

def build_U(df, cols, target_dim):
    g_u = df.groupby('__bin5')[cols].mean() if cols else pd.DataFrame(index=df['__bin5'].unique())
    full_idx = pd.date_range(g_targets.index.min(), g_targets.index.max(), freq='5min')
    g_u = g_u.reindex(full_idx)
    U = g_u.to_numpy(dtype=np.float32) if len(cols)>0 else np.zeros((len(full_idx), 0), np.float32)

    # standardize
    if U.shape[1] > 0:
        mu, sd = np.nanmean(U,0), np.nanstd(U,0)+1e-8
        U = (U - mu)/sd
    U = np.nan_to_num(U)

    # pad or truncate
    if U.shape[1] < target_dim:
        pad = np.zeros((U.shape[0], target_dim - U.shape[1]), np.float32)
        U = np.concatenate([U, pad], axis=1)
        print(f"[warn] padded {pad.shape[1]} zeros to match u_dim={target_dim}")
    elif U.shape[1] > target_dim:
        U = U[:, :target_dim]
        print(f"[warn] truncated features to u_dim={target_dim}")
    return U, full_idx

u_cols = [c for c in u_primary if c in df.columns]
if not u_cols:
    u_cols = [c for c in u_fallback if c in df.columns]
U, full_idx = build_U(df, u_cols, u_dim_expected)

# Align targets to full index
g_targets = g_targets.reindex(full_idx).ffill()

# ======================================================
# 5. Prepare tensors and normalization
# ======================================================
y_delay = g_targets[delay_col].to_numpy(np.float32)
y_taxi  = g_targets[taxi_col].to_numpy(np.float32)
y_count = g_targets['__count'].to_numpy(np.float32)

def zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x)+1e-8
    return (x - mu)/sd, mu, sd

y_delay, mu_delay, sd_delay = zscore(y_delay)
y_taxi, mu_taxi, sd_taxi = zscore(y_taxi)

U_t = torch.tensor(U, dtype=torch.float32, device=device)
dt = torch.tensor(5/60, dtype=torch.float32)

# ======================================================
# 6. Define model (must match checkpoint)
# ======================================================
class Dynamics(nn.Module):
    def __init__(self, hdim, u_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hdim + u_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, hdim),
        )
    def forward(self, h, u):
        return self.net(torch.cat([h, u], dim=-1))

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.delay_head = nn.Linear(hdim, 1)
        self.taxi_head  = nn.Linear(hdim, 1)
        self.count_head = nn.Linear(hdim, 1)
    def forward(self, h):
        return (self.delay_head(h).squeeze(-1),
                self.taxi_head(h).squeeze(-1),
                self.count_head(h).squeeze(-1))

class ControlledNODE(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt = dt
        self.f = Dynamics(hdim, u_dim)
        self.out = Readout(hdim)
        self.h0 = nn.Parameter(torch.zeros(1, hdim))
    def step(self, h, u):
        return h + self.f(h, u) * self.dt
    def forward(self, U, h0=None):
        T = U.size(0)
        h = self.h0 if h0 is None else h0
        d_list, t_list, l_list = [], [], []
        for t in range(T):
            d_hat, t_hat, l_hat = self.out(h)
            d_list.append(d_hat); t_list.append(t_hat); l_list.append(l_hat)
            h = self.step(h, U[t])
        return (torch.stack(d_list).squeeze(-1),
                torch.stack(t_list).squeeze(-1),
                torch.stack(l_list).squeeze(-1),
                h)

# ======================================================
# 7. Load checkpoint
# ======================================================
model = ControlledNODE(hdim=hdim_ckpt, u_dim=u_dim_expected, dt=dt.to(device)).to(device)
model.load_state_dict(state, strict=True)
model.eval()
print("[INFO] Model loaded successfully.")

# ======================================================
# 8. Inference with progress bar
# ======================================================
print("[INFO] Running inference ...")
start_time = time.time()

with torch.no_grad():
    T = U_t.size(0)
    h = model.h0
    d_list, t_list, l_list = [], [], []
    for i in tqdm(range(T), desc="Evaluating", ncols=100):
        d_hat, t_hat, l_hat = model.out(h)
        d_list.append(d_hat)
        t_list.append(t_hat)
        l_list.append(l_hat)
        u = U_t[i].unsqueeze(0)
        h = model.step(h, u)

    d_hat_all = torch.stack(d_list).squeeze(-1)
    t_hat_all = torch.stack(t_list).squeeze(-1)
    loglam_all = torch.stack(l_list).squeeze(-1)

elapsed = time.time() - start_time
print(f"[INFO] Inference completed in {elapsed/60:.2f} min ({T/elapsed:.1f} steps/s)")

# ======================================================
# 9. Denormalize & evaluate metrics
# ======================================================
delay_pred = d_hat_all.cpu().numpy() * sd_delay + mu_delay
taxi_pred  = t_hat_all.cpu().numpy() * sd_taxi  + mu_taxi
count_pred = np.exp(loglam_all.cpu().numpy())

delay_true = y_delay * sd_delay + mu_delay
taxi_true  = y_taxi  * sd_taxi  + mu_taxi
count_true = y_count

def compute_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:12s} | RMSE={rmse:.3f} | MAE={mae:.3f} | R²={r2:.3f}")
    return rmse, mae, r2

delay_m = compute_metrics(delay_true, delay_pred, "Delay")
taxi_m  = compute_metrics(taxi_true, taxi_pred, "Taxi-In")
count_m = compute_metrics(count_true, count_pred, "NumArrivals")

metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²"],
    "Delay": [delay_m[0], delay_m[1], delay_m[2]],
    "Taxi-In": [taxi_m[0], taxi_m[1], taxi_m[2]],
    "NumArrivals": [count_m[0], count_m[1], count_m[2]]
})
metrics_df.to_csv("node_evaluation_metrics.csv", index=False)
print("\n[INFO] Metrics saved → node_evaluation_metrics.csv")

