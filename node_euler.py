# node_new.py
import platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ============ 0) Environment & Device ============
print("[env] torch:", torch.__version__, "| python:", sys.version.split()[0])
print("[env] cuda_available:", torch.cuda.is_available(),
      "| mps_built:", torch.backends.mps.is_built(),
      "| mps_available:", torch.backends.mps.is_available())
print("[env] machine:", platform.machine())

if torch.backends.mps.is_available():
    device = torch.device("mps"); print("[device] Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("[device] CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu"); print("[device] CPU")

# Enable AMP only on CUDA; keep FP32 on MPS/CPU for stability
USE_AMP = (device.type == "cuda")
if device.type == "cuda":
    torch.set_float32_matmul_precision("medium")
scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ============ 1) Load CSV (keep original column names) ============
CSV_PATH = "./data/ref/train_set.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
print(f"[INFO] loaded rows: {len(df):,}")
df = df.sort_values('CRS_ARR_TIME_dt')

# ============ 2) 5-min Aggregation → Regular Grid ============
# Target columns auto-detection: delay / taxi-in / arrival count
delay_col = 'ARR_DELAY_NEW' if 'ARR_DELAY_NEW' in df.columns else (
            'ARR_DELAY' if 'ARR_DELAY' in df.columns else (
            'ArrDelayMinutes' if 'ArrDelayMinutes' in df.columns else None))
taxi_col  = 'TAXI_IN' if 'TAXI_IN' in df.columns else (
            'AvgTaxiIn' if 'AvgTaxiIn' in df.columns else None)

if delay_col is None:
    raise RuntimeError("Delay column not found (expected: ARR_DELAY_NEW/ARR_DELAY/ArrDelayMinutes).")
if taxi_col is None:
    raise RuntimeError("Taxi-in column not found (expected: TAXI_IN/AvgTaxiIn).")

# Arrival counting strategy:
# - If 'NumArrivals' exists: sum over the 5-min bucket
# - Otherwise: use the record count per bucket
has_num_arrivals = 'NumArrivals' in df.columns

# 5-minute bucket key
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

# Aggregate targets
g_targets = df.groupby('__bin5').agg({
    delay_col: 'mean',
    taxi_col:  'mean',
})
if has_num_arrivals:
    g_targets['__count'] = df.groupby('__bin5')['NumArrivals'].sum()
else:
    g_targets['__count'] = df.groupby('__bin5').size()

# Exogenous inputs U:
# Prefer your lowercase features; if missing, fall back to uppercase trigonometric bases
u_primary = ['sin_hour','cos_hour','Hour','DayOfWeek','Month','Holiday','DaysToHoliday']
u_fallback = ['HOUR_SIN','HOUR_COS','DOW_SIN','DOW_COS','MONTH_SIN','MONTH_COS',
              'MOD_SIN','MOD_COS','DAY_OF_WEEK_ENC','MONTH_ENC','MIN_OF_DAY','ARR_HOUR']
u_cols = [c for c in u_primary if c in df.columns]
if not u_cols:
    u_cols = [c for c in u_fallback if c in df.columns]
if not u_cols:
    raise RuntimeError("No exogenous input columns found (expected sin_hour/cos_hour/... or HOUR_SIN/HOUR_COS/...).")

g_u = df.groupby('__bin5')[u_cols].mean()

# Align to a complete 5-min regular index and fill
full_idx = pd.date_range(
    min(g_targets.index.min(), g_u.index.min()),
    max(g_targets.index.max(), g_u.index.max()),
    freq='5min'
)
g_targets = g_targets.reindex(full_idx)
g_u       = g_u.reindex(full_idx)

# Forward-fill numeric targets; set missing counts to 0
g_targets[delay_col] = g_targets[delay_col].ffill()
g_targets[taxi_col]  = g_targets[taxi_col].ffill()
g_targets['__count'] = g_targets['__count'].fillna(0.0)

# To numpy
y_delay = g_targets[delay_col].to_numpy(dtype=np.float32)
y_taxi  = g_targets[taxi_col ].to_numpy(dtype=np.float32)
y_count = g_targets['__count'].to_numpy(dtype=np.float32)
U       = g_u.to_numpy(dtype=np.float32)

# --- sanitize targets ---
y_delay = np.nan_to_num(y_delay, nan=0.0, posinf=0.0, neginf=0.0)
y_taxi  = np.nan_to_num(y_taxi,  nan=0.0, posinf=0.0, neginf=0.0)
y_count = np.nan_to_num(y_count, nan=0.0, posinf=0.0, neginf=0.0)

# --- standardize U column-wise (z-score) ---
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U = (U - U_mu) / U_sd
U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)


# Normalize only regression targets (z-score); keep counts as non-negative
def zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x) + 1e-8
    return (x - mu)/sd, mu, sd

y_delay, mu_delay, sd_delay = zscore(y_delay)
y_taxi,  mu_taxi,  sd_taxi  = zscore(y_taxi)
y_count = np.clip(y_count, 0, None)

dt_minutes = 5.0
dt = torch.tensor(dt_minutes/60.0, dtype=torch.float32)
T_total = U.shape[0]
print(f"[INFO] 5-min steps: {T_total:,} | U-dim: {U.shape[1]} | U-cols: {u_cols}")

# ============ 3) Model ============
class Dynamics(nn.Module):
    def __init__(self, hdim, u_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hdim + u_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, hdim),
        )
    def forward(self, h, u):
        if u.dim() == 1: u = u.unsqueeze(0)
        x = torch.cat([h, u], dim=-1)
        return self.net(x)

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.delay_head = nn.Linear(hdim, 1)
        self.taxi_head  = nn.Linear(hdim, 1)
        self.count_head = nn.Linear(hdim, 1)  # log λ
    def forward(self, h):
        d = self.delay_head(h).squeeze(-1)
        t = self.taxi_head(h).squeeze(-1)
        l = self.count_head(h).squeeze(-1)
        return d, t, l

class ControlledNODE(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt   = dt
        self.f    = Dynamics(hdim, u_dim)
        self.out  = Readout(hdim)
        self.h0   = nn.Parameter(torch.zeros(1, hdim))
    def step(self, h, u):
        dh = self.f(h, u) * self.dt
        return h + dh
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

# ============ 4) Tensors to Device ============
print(f"[info] tensors → {device}")
U_t = torch.tensor(U,        dtype=torch.float32, device=device)
y_delay_t = torch.tensor(y_delay, dtype=torch.float32, device=device)
y_taxi_t  = torch.tensor(y_taxi,  dtype=torch.float32, device=device)
y_count_t = torch.tensor(y_count, dtype=torch.float32, device=device)

hdim = 16
model = ControlledNODE(hdim=hdim, u_dim=U_t.shape[1], dt=dt.to(device)).to(device)

optim = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
mse = nn.MSELoss()

def poisson_nll(log_lambda, y):
    log_lambda = torch.clamp(log_lambda, -10.0, 10.0)
    return F.poisson_nll_loss(log_lambda, y, log_input=True, full=False, reduction='mean')

# ============ 5) Split & Warmup ============
T = U_t.shape[0]
split = int(0.8 * T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

model.train()
warm_L = min(64, T)
with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
              enabled=(device.type=="cuda")):
    d_hat, t_hat, loglam, _ = model(U_t[:warm_L])
    warm_loss = mse(d_hat, y_delay_t[:warm_L]) + mse(t_hat, y_taxi_t[:warm_L]) + poisson_nll(loglam, y_count_t[:warm_L])
optim.zero_grad(set_to_none=True)
if USE_AMP:
    scaler.scale(warm_loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim); scaler.update()
else:
    warm_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
print("[INFO] warmup done")

# ============ 6) Training ============
epochs = 100
window = 256
best_val = float("inf")

outer = tqdm(range(1, epochs+1), desc="Epochs", position=0, ncols=100)
for ep in outer:
    model.train()
    total_loss = 0.0
    h = None
    inner = tqdm(range(0, T, window), desc=f"[Train] e{ep}", leave=False, position=1, ncols=100)
    for start in inner:
        end = min(start + window, T)
        U_blk  = U_t[start:end]
        yd_blk = y_delay_t[start:end]
        yt_blk = y_taxi_t[start:end]
        yc_blk = y_count_t[start:end]

        with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                      enabled=(device.type=="cuda")):
            d_hat, t_hat, loglam, h = model(U_blk, h0=h)
            loss = mse(d_hat, yd_blk) + mse(t_hat, yt_blk) + poisson_nll(loglam, yc_blk)

        optim.zero_grad(set_to_none=True)
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ---- NaN/Inf guard ----
            if not torch.isfinite(loss):
                print("[WARN] NaN/Inf loss detected. Abort epoch.")
                raise SystemExit
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(loss):
                print("[WARN] NaN/Inf loss detected. Abort epoch.")
                raise SystemExit
            optim.step()

        if h is not None:
            h = h.detach()
        total_loss += loss.item()
        inner.set_postfix(loss=f"{loss.item():.4f}")

    # Validation
    model.eval()
    with torch.no_grad(), autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                                   enabled=(device.type=="cuda")):
        d_hat_v, t_hat_v, loglam_v, _ = model(U_t[valid_slice])
        val_loss = (mse(d_hat_v, y_delay_t[valid_slice]) +
                    mse(t_hat_v,  y_taxi_t[valid_slice]) +
                    poisson_nll(loglam_v, y_count_t[valid_slice])).item()

    outer.set_postfix(train=f"{total_loss:.4f}", val=f"{val_loss:.4f}",
                      lr=f"{optim.param_groups[0]['lr']:.1e}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model": model.state_dict()}, "controlled_node_best.pt")

print("Training done. Best val loss:", best_val)

# ============ 7) Visualization ============
print("[INFO] plotting results ...")

# Load best checkpoint
ckpt = torch.load("controlled_node_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Inference on the full horizon
with torch.no_grad():
    d_hat_all, t_hat_all, loglam_all, _ = model(U_t)
    # De-normalize to original units
    delay_pred = d_hat_all.cpu().numpy() * sd_delay + mu_delay
    taxi_pred  = t_hat_all.cpu().numpy() * sd_taxi + mu_taxi
    count_pred = np.exp(loglam_all.cpu().numpy())  # E[count] from Poisson λ

delay_true = y_delay * sd_delay + mu_delay
taxi_true  = y_taxi  * sd_taxi  + mu_taxi
count_true = y_count
time_axis = np.arange(T_total)

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time_axis, delay_true, label='True Delay', color='black', alpha=0.6)
plt.plot(time_axis, delay_pred, label='Pred Delay', color='red', alpha=0.7)
plt.title("Arrival Delay (minutes)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(time_axis, taxi_true, label='True TaxiIn', color='black', alpha=0.6)
plt.plot(time_axis, taxi_pred, label='Pred TaxiIn', color='blue', alpha=0.7)
plt.title("Average Taxi-In Time (minutes)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(time_axis, count_true, label='True NumArrivals', color='black', alpha=0.6)
plt.plot(time_axis, count_pred, label='Pred NumArrivals', color='green', alpha=0.7)
plt.title("Number of Arrivals per 5min")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("controlled_node_results.png", dpi=200)
plt.show()
print("[INFO] figure saved as controlled_node_results.png")
