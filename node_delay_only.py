# ==============================================================
# node_delay_smooth.py
# Neural ODE for Smoothed Arrival Delay (log + rolling mean)
# ==============================================================
import platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ==============================================================
# 0) Device
# ==============================================================
print("[env] torch:", torch.__version__)
print("[env] cuda:", torch.cuda.is_available(),
      "| mps:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[device] Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[device] CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("[device] CPU")

USE_AMP = (device.type == "cuda")
scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ==============================================================
# 1) Load CSV
# ==============================================================
CSV_PATH = "./data/ref/bwi_flights_with_weather_holiday_synth.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')
print("[INFO] loaded rows:", len(df))

TARGET = "ARR_DELAY_NEW"
if TARGET not in df.columns:
    raise RuntimeError(f"Target {TARGET} not found.")

# ==============================================================
# 2) 5-min Aggregation
# ==============================================================
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

# 原始目标（raw delay）
g_raw = df.groupby('__bin5')[TARGET].mean()

# Flight count
g_count = df.groupby('__bin5').size()

# Feature columns
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

u_all = [c for c in (u_cols_time + u_cols_weather + u_cols_holiday)
         if c in df.columns]

g_u = df.groupby('__bin5')[u_all].mean()

# Build full time index
full_idx = pd.date_range(
    min(g_raw.index.min(), g_u.index.min()),
    max(g_raw.index.max(), g_u.index.max()),
    freq='5min'
)

g_raw = g_raw.reindex(full_idx).fillna(0.0)
g_count = g_count.reindex(full_idx).fillna(0.0)
g_u = g_u.reindex(full_idx)

# ==============================================================
# 2.1) Preprocess target: clip + log1p + rolling mean
# ==============================================================
y_raw = g_raw.to_numpy(np.float32)
y_raw = np.maximum(y_raw, 0.0)        # remove negative early arrivals

# log1p compression
y_log = np.log1p(y_raw)

# rolling mean (1 hour = 12 bins)
y_log_smooth = (
    pd.Series(y_log, index=full_idx)
      .rolling(12, min_periods=1)
      .mean()
      .to_numpy(np.float32)
)

# Final target
y_delay = y_log_smooth

# ==============================================================
# Input U
# ==============================================================
U = g_u.to_numpy(np.float32)
U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)

# Add arrivals_1h
arr_1h = (
    pd.Series(g_count, index=full_idx)
      .rolling(12, min_periods=1)
      .sum()
      .to_numpy(np.float32)
)
U = np.concatenate([U, arr_1h[:,None]], axis=1)
u_all_plus = u_all + ["arrivals_1h"]

# ==============================================================
# 2.2 Time truncation
# ==============================================================
MAX_STEPS = 5000
if len(U) > MAX_STEPS:
    start = len(U) - MAX_STEPS
    print(f"[INFO] truncate: {start} → end")
    U = U[start:]
    y_delay = y_delay[start:]
    full_idx = full_idx[start:]

T_total = len(U)
print("[INFO] final steps:", T_total)

# ==============================================================
# 3) Normalize
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
# 4) Model: Controlled NODE
# ==============================================================
DYN_GAIN = 0.02
STATE_DAMP = 0.10
H_CLIP = 5.0

def xavier_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class Dynamics(nn.Module):
    def __init__(self, hdim, u_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hdim + u_dim, 128),
            nn.SiLU(),
            nn.Linear(128,128),
            nn.SiLU(),
            nn.Linear(128,hdim)
        )
        self.apply(xavier_init_)
        self.damp = STATE_DAMP

    def forward(self, h, u):
        if u.dim()==1:
            u = u.unsqueeze(0)
        drift = self.net(torch.cat([h,u], dim=-1))
        return DYN_GAIN * drift - self.damp * h

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.head = nn.Linear(hdim,1)
        self.apply(xavier_init_)

    def forward(self, h):
        return self.head(h).squeeze(-1)

class ControlledNODE(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt = dt
        self.f = Dynamics(hdim, u_dim)
        self.out = Readout(hdim)
        self.h0 = nn.Parameter(torch.zeros(1,hdim))

    @torch.no_grad()
    def _ensure(self,x):
        return x if x.dim()==2 else x.unsqueeze(0)

    def rk4_step(self, h, u):
        u = self._ensure(u)
        dt = self.dt
        k1 = self.f(h,u)
        k2 = self.f(h + 0.5*dt*k1, u)
        k3 = self.f(h + 0.5*dt*k2, u)
        k4 = self.f(h + dt*k3, u)
        h_next = h + (dt/6)*(k1+2*k2+2*k3+k4)
        h_next = torch.tanh(h_next)
        return torch.clamp(h_next, -H_CLIP, H_CLIP)

    def forward(self, U, h0=None):
        h = self.h0 if h0 is None else h0
        outs=[]
        for t in range(U.size(0)):
            outs.append(self.out(h))
            h = self.rk4_step(h, U[t])
        return torch.stack(outs), h

# ==============================================================
# 5) Prepare tensors
# ==============================================================
U_t = torch.tensor(U, dtype=torch.float32, device=device)
y_t = torch.tensor(y_delay_n, dtype=torch.float32, device=device)

T = U_t.shape[0]
split = int(0.8*T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

model = ControlledNODE(hdim=32, u_dim=U_t.shape[1],
                       dt=torch.tensor(5/60, dtype=torch.float32, device=device)).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
mse = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=40)

# ==============================================================
# 6) Warmup
# ==============================================================
warm_L = min(64,T)
with autocast(device_type=device.type, enabled=USE_AMP):
    d_hat,_ = model(U_t[:warm_L])
    warm_loss = mse(d_hat.squeeze(-1), y_t[:warm_L])

optim.zero_grad()
if USE_AMP:
    scaler.scale(warm_loss).backward()
    scaler.step(optim); scaler.update()
else:
    warm_loss.backward(); optim.step()

print("[INFO] warmup done")

# ==============================================================
# 7) Training (TBPTT)
# ==============================================================
epochs = 40
window = 256
best_val = float("inf")

outer = tqdm(range(epochs), desc="Epochs")
for ep in outer:
    model.train()
    h=None
    total=0
    for s in range(0, T, window):
        e = min(s+window, T)
        U_blk = U_t[s:e]
        y_blk = y_t[s:e]

        with autocast(device_type=device.type, enabled=USE_AMP):
            d_hat, h = model(U_blk, h)
            loss = mse(d_hat.squeeze(-1), y_blk)

        optim.zero_grad()
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
        else:
            loss.backward(); optim.step()

        h = h.detach()
        total += loss.item()

    # Validation
    model.eval()
    with torch.no_grad(), autocast(device_type=device.type, enabled=USE_AMP):
        pred,_ = model(U_t[valid_slice])
        val = mse(pred.squeeze(-1), y_t[valid_slice]).item()

    outer.set_postfix(train=f"{total:.3f}", val=f"{val:.3f}")

    scheduler.step()
    if val < best_val:
        best_val = val
        torch.save({
            "model": model.state_dict(),
            "mu_delay": mu_delay,
            "sd_delay": sd_delay,
            "U_mu": U_mu,
            "U_sd": U_sd,
            "u_cols": u_all_plus
        }, "node_delay_smooth_best.pt")

print("Training done. best val:", best_val)

# ==============================================================
# 8) Visualization
# ==============================================================
ckpt = torch.load("node_delay_smooth_best.pt", map_location=device, weights_only=False)

model.load_state_dict(ckpt["model"])
model.eval()

with torch.no_grad():
    pred_n,_ = model(U_t)

pred_log = pred_n.cpu().numpy() * sd_delay + mu_delay
true_log = y_delay

pred_min = np.expm1(pred_log)
true_min = np.expm1(true_log)

# Raw delay for comparison
raw_delay = g_raw.to_numpy()[-T_total:]

plt.figure(figsize=(14,6))
# plt.plot(raw_delay, label="Raw Delay (noisy)", alpha=0.3)
plt.plot(true_min, label="Smoothed Delay (truth)", alpha=0.9)
plt.plot(pred_min, label="Predicted Smoothed Delay", alpha=0.9)
plt.legend()
plt.grid()
plt.title("Neural ODE Prediction of Smoothed Arrival Delay")
plt.xlabel("5-min steps")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("node_delay_smooth_results.png", dpi=200)
plt.show()

print("[INFO] saved node_delay_smooth_results.png")
