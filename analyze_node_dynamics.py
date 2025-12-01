# ==============================================================
# analyze_node_dynamics.py
# Analyze latent dynamics of trained Neural ODE
# ==============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.serialization

# --------------------------------------------------------------
# Load checkpoint + data preprocessing
# --------------------------------------------------------------

CSV_PATH = "./data/ref/bwi_flights_with_weather_holiday_synth.csv"
CKPT_PATH = "node_delay_smooth_best.pt"

device = torch.device("cpu")

print("[device]", device)

# ------------------- Load CSV -----------------------
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
df = df.sort_values('CRS_ARR_TIME_dt')

TARGET = "ARR_DELAY_NEW"
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

g_raw = df.groupby('__bin5')[TARGET].mean()
g_count = df.groupby('__bin5').size()

# Time/weather/holiday features
u_cols_time = [
    "HOUR_SIN","HOUR_COS","DOW_SIN","DOW_COS",
    "MONTH_SIN","MONTH_COS","MIN_OF_DAY","ARR_HOUR"
]
u_cols_weather = [
    "visibility","low_ceiling_flag","wind_speed",
    "rain_flag","snow_flag","thunder_flag","fog_flag","ice_flag",
    "syn_vis","syn_cloud","syn_wind","syn_precip","syn_temp_dev"
]
u_cols_holiday = ["holiday_flag"]

u_all = [c for c in (u_cols_time + u_cols_weather + u_cols_holiday) if c in df.columns]
g_u = df.groupby('__bin5')[u_all].mean()

full_idx = pd.date_range(
    min(g_raw.index.min(), g_u.index.min()),
    max(g_raw.index.max(), g_u.index.max()),
    freq='5min'
)

g_raw = g_raw.reindex(full_idx).fillna(0.0)
g_count = g_count.reindex(full_idx).fillna(0.0)
g_u = g_u.reindex(full_idx)

# ------------------- Target preprocessing ---------------------
y_raw = np.maximum(g_raw.to_numpy(np.float32), 0.0)
y_log = np.log1p(y_raw)
y_delay = (
    pd.Series(y_log, index=full_idx)
      .rolling(12, min_periods=1).mean()
      .to_numpy(np.float32)
)

# ------------------- Build input U ----------------------------
U = g_u.to_numpy(np.float32)
U = np.nan_to_num(U)

arr_1h = (
    pd.Series(g_count, index=full_idx)
      .rolling(12, min_periods=1).sum()
      .to_numpy(np.float32)
)
U = np.concatenate([U, arr_1h[:, None]], axis=1)

# ------------------- Normalize -------------------------------
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U_norm = (U - U_mu) / U_sd
U_norm = np.clip(U_norm, -5, 5)

def zscore(x):
    mu = np.mean(x)
    sd = np.std(x) + 1e-8
    return (x - mu) / sd, mu, sd

y_delay_n, mu_delay, sd_delay = zscore(y_delay)

U_t = torch.tensor(U_norm, device=device, dtype=torch.float32)
y_t = torch.tensor(y_delay_n, device=device, dtype=torch.float32)

T = len(U_t)

# --------------------------------------------------------------
# Rebuild model (same architecture as training)
# --------------------------------------------------------------

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
            nn.Linear(hdim + u_dim, 128), nn.SiLU(),
            nn.Linear(128,128), nn.SiLU(),
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
        h_next = h + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        h_next = torch.tanh(h_next)
        return torch.clamp(h_next, -H_CLIP, H_CLIP)

    def forward(self, U, return_h=False, return_f=False):
        h = self.h0
        outs = []
        h_seq = []
        f_seq = []

        for t in range(U.size(0)):
            h_seq.append(h)
            if return_f:
                f_seq.append(self.f(h, self._ensure(U[t])))
            outs.append(self.out(h))
            h = self.rk4_step(h, U[t])

        outs = torch.stack(outs)
        h_seq = torch.stack(h_seq)

        if return_f:
            f_seq = torch.stack(f_seq)

        if return_h and return_f:
            return outs, h, h_seq, f_seq
        elif return_h:
            return outs, h, h_seq
        return outs, h

# --------------------------------------------------------------
# Load checkpoint
# --------------------------------------------------------------
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model = ControlledNODE(hdim=32, u_dim=U_t.shape[1],
                       dt=torch.tensor(5/60, device=device)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print("[INFO] Model loaded.")

# --------------------------------------------------------------
# Forward pass to extract dynamics
# --------------------------------------------------------------

with torch.no_grad():
    pred_n, _, h_seq, f_seq = model(U_t, return_h=True, return_f=True)

# Transform predictions back to minute scale
pred_log = pred_n.cpu().numpy() * sd_delay + mu_delay
true_log = y_delay
pred_min = np.expm1(pred_log)
true_min = np.expm1(true_log)

# # --------------------------------------------------------------
# # Visualization: Prediction vs Truth
# # --------------------------------------------------------------

# plt.figure(figsize=(14,5))
# plt.plot(true_min, label="Truth", alpha=0.8)
# plt.plot(pred_min, label="Predicted", alpha=0.8)
# plt.legend()
# plt.title("Neural ODE Prediction (Smoothed Delay)")
# plt.grid(alpha=0.3)
# plt.show()

# # --------------------------------------------------------------
# # PCA of hidden state trajectory
# # --------------------------------------------------------------

h_np = h_seq.cpu().numpy().reshape(len(h_seq), -1)
pca = PCA(n_components=2)
h_pca = pca.fit_transform(h_np)

# plt.plot(h_pca[:, 0])
# plt.title("First Principal Component of Hidden State")
# plt.grid(alpha=0.3)
# plt.show()

# # --------------------------------------------------------------
# # Norm of hidden derivative: ||h_{t+1}-h_t||
# # --------------------------------------------------------------

dh = np.linalg.norm(np.diff(h_np, axis=0), axis=1)
# plt.plot(dh)
# plt.title("Hidden State Velocity ||h_{t+1} - h_t||")
# plt.grid(alpha=0.3)
# plt.show()

# # --------------------------------------------------------------
# # Norm of vector field: ||f(h,u)||
# # --------------------------------------------------------------

f_np = f_seq.cpu().numpy()
f_norm = np.linalg.norm(f_np, axis=1)

# plt.plot(f_norm)
# plt.title("Vector Field Magnitude ||f(h,u)||")
# plt.grid(alpha=0.3)
# plt.show()

# print("[INFO] All analyses completed.")

# ==============================================================
# Print latent-dynamics indicators for operational insights
# ==============================================================

print("\n============== Latent Dynamics Analysis ==============\n")

# 1) Global hidden-state smoothness
mean_dh = float(np.mean(dh))
p95_dh  = float(np.percentile(dh, 95))

print(f"Global mean state velocity (||h[t+1]-h[t]||): {mean_dh:.6f}")
print(f"Global 95th percentile state velocity      : {p95_dh:.6f}")

# 2) Vector-field strength (system sensitivity to inputs)
mean_f = float(np.mean(f_norm))
p95_f  = float(np.percentile(f_norm, 95))

print(f"\nGlobal mean vector field magnitude (||f(h,u)||): {mean_f:.6f}")
print(f"Global 95th percentile vector field        : {p95_f:.6f}")

# --------------------------------------------------------------
# Segment-wise analysis: peak/off-peak, severe/normal, holiday
# --------------------------------------------------------------

hours = np.array([ts.hour for ts in full_idx])

peak_mask     = (hours >= 14) & (hours <= 20)
offpeak_mask  = ~peak_mask

def segment_stats(name, mask):
    seg_dh = dh[mask[:-1]]            # dh is T-1 long
    seg_f  = f_norm[mask]

    print(f"\n--- {name} ---")
    print(f"Samples                     : {mask.sum()}")
    print(f"Mean state velocity         : {np.mean(seg_dh):.6f}")
    print(f"95th pct state velocity     : {np.percentile(seg_dh,95):.6f}")
    print(f"Mean vector-field magnitude : {np.mean(seg_f):.6f}")
    print(f"95th pct vector-field mag   : {np.percentile(seg_f,95):.6f}")

# Weather segmentation
weather_idx = U_norm[:, u_all.index("syn_precip")]  # >=85% = severe
th = np.percentile(weather_idx, 85)

severe_mask = (weather_idx >= th)
normal_mask = ~severe_mask

# Holiday segmentation
holiday_mask = U_norm[:, u_all.index("holiday_flag")] > 0.5
regular_mask = ~holiday_mask

# Print all segment stats
segment_stats("Peak Hours (14–20)", peak_mask)
segment_stats("Off-Peak Hours", offpeak_mask)
segment_stats("Severe Weather", severe_mask)
segment_stats("Normal Weather", normal_mask)
segment_stats("Holiday", holiday_mask)
segment_stats("Regular Days", regular_mask)

print("\n============== End of Latent-Dynamics Analysis ==============\n")
