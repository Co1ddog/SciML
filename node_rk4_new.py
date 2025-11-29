# node_weather_delay.py  -- Controlled Neural ODE for Arrival Delay
import platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ============ 0) Environment & Device ============
print("[env] torch:", torch.__version__)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[device] MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[device] CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("[device] CPU")

USE_AMP = (device.type == "cuda")
scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ============ 1) Load CSV ============
CSV_PATH = "./data/ref/bwi_flights_with_weather_holiday.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')
print(f"[INFO] loaded rows: {len(df):,}")

# Delay column
delay_col = 'ARR_DELAY_NEW' if 'ARR_DELAY_NEW' in df.columns else 'ARR_DELAY'
assert delay_col in df.columns, "Delay column not found."

# ============ 2) 5-min aggregation ============
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

g_delay = df.groupby('__bin5')[delay_col].mean()

# ===== Your selected inputs =====
u_cols = [
    "visibility",
    "low_ceiling_flag",
    "wind_speed",
    "rain_flag",
    "snow_flag",
    "thunder_flag",
    "fog_flag",
    "ice_flag",
    "holiday_flag"
]

available_cols = [c for c in u_cols if c in df.columns]
print("[INFO] Using U features:", available_cols)

g_u = df.groupby('__bin5')[available_cols].mean()

full_idx = pd.date_range(
    df['__bin5'].min(),
    df['__bin5'].max(),
    freq='5min'
)

g_delay = g_delay.reindex(full_idx).ffill()
g_u = g_u.reindex(full_idx).ffill()

# numpy
y_delay = g_delay.to_numpy(dtype=np.float32)
U = g_u.to_numpy(dtype=np.float32)

# U normalization
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U = (U - U_mu) / U_sd
U = np.nan_to_num(U, nan=0.0)
U = np.clip(U, -5, 5)

# Standardize delay
mu_delay = np.nanmean(y_delay)
sd_delay = np.nanstd(y_delay) + 1e-8
y_delay_n = (y_delay - mu_delay) / sd_delay
y_delay_n = np.nan_to_num(y_delay_n)

# convert tensors
U_t = torch.tensor(U, dtype=torch.float32, device=device)
y_delay_t = torch.tensor(y_delay_n, dtype=torch.float32, device=device)

# ============ 3) NODE Dynamics ============
dt_minutes = 5.0
dt = torch.tensor(dt_minutes/60.0, dtype=torch.float32, device=device)

H_CLIP = 5.0
DYN_GAIN = 0.03
STATE_DAMP = 0.12

def xavier_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class Dynamics(nn.Module):
    def __init__(self, hdim, u_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hdim + u_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, hdim),
        )
        self.apply(xavier_)
        self.damp = STATE_DAMP

    def forward(self, h, u):
        u = u.unsqueeze(0) if u.dim() == 1 else u
        x = torch.cat([h, u], dim=-1)
        drift = self.net(x)
        return DYN_GAIN * drift - self.damp * h

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.delay_head = nn.Linear(hdim, 1)
        self.apply(xavier_)

    def forward(self, h):
        return self.delay_head(h).squeeze(-1)

class ControlledNODE(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt = dt
        self.f = Dynamics(hdim, u_dim)
        self.out = Readout(hdim)
        self.h0 = nn.Parameter(torch.zeros(1, hdim))

    def rk4_step(self, h, u):
        dt = self.dt
        k1 = self.f(h, u)
        k2 = self.f(h + 0.5*dt*k1, u)
        k3 = self.f(h + 0.5*dt*k2, u)
        k4 = self.f(h + dt*k3, u)
        h_next = h + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        h_next = torch.tanh(h_next)
        h_next = torch.clamp(h_next, -H_CLIP, H_CLIP)
        return h_next

    def forward(self, U, h0=None):
        h = self.h0 if h0 is None else h0
        preds = []
        for t in range(U.size(0)):
            d = self.out(h)
            preds.append(d)
            h = self.rk4_step(h, U[t])
        return torch.stack(preds).squeeze(-1), h

# ============ 4) Training ============
hdim = 32
model = ControlledNODE(hdim, U_t.shape[1], dt).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
mse = nn.MSELoss()

T = len(U_t)
split = int(0.8*T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

epochs = 60
window = 256

best_val = float("inf")

for ep in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    h = None

    for start in range(0, T, window):
        end = min(start + window, T)

        U_blk = U_t[start:end]
        y_blk = y_delay_t[start:end]

        opt.zero_grad()

        d_hat, h = model(U_blk, h0=h)

        d_hat = torch.nan_to_num(d_hat, 0.0)
        loss = mse(d_hat, y_blk)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        h = h.detach()
        total_loss += loss.item()

    # validation
    model.eval()
    with torch.no_grad():
        d_val, _ = model(U_t[valid_slice])
        val_loss = mse(d_val, y_delay_t[valid_slice]).item()

    print(f"Epoch {ep:03d} | Train {total_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "weather_delay_NODE.pt")

print("Training done. Best val =", best_val)

# ============ 5) Visualization ============
model.load_state_dict(torch.load("weather_delay_NODE.pt", map_location=device))
model.eval()

with torch.no_grad():
    d_pred_n, _ = model(U_t)
    d_pred_n = torch.nan_to_num(d_pred_n, 0.0).cpu().numpy()

delay_pred = d_pred_n * sd_delay + mu_delay
delay_true = y_delay

plt.figure(figsize=(14,6))
plt.plot(delay_true, label="True Delay", color="black", alpha=0.6)
plt.plot(delay_pred, label="Pred Delay", color="red", alpha=0.8)
plt.title("Arrival Delay Prediction (Weather+Holiday NODE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("weather_delay_NODE_results.png", dpi=200)
plt.show()
