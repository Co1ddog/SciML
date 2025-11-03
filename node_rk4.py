# node_new_super_stable.py  -- Controlled Neural ODE (all-MSE), super-stable for MPS
import platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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

USE_AMP = (device.type == "cuda")
if device.type == "cuda":
    torch.set_float32_matmul_precision("medium")
scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ============ 1) Load CSV ============
CSV_PATH = "./data/ref/train_set.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
print(f"[INFO] loaded rows: {len(df):,}")
df = df.sort_values('CRS_ARR_TIME_dt')

# ============ 2) 5-min aggregation → regular grid ============
delay_col = 'ARR_DELAY_NEW' if 'ARR_DELAY_NEW' in df.columns else (
            'ARR_DELAY' if 'ARR_DELAY' in df.columns else (
            'ArrDelayMinutes' if 'ArrDelayMinutes' in df.columns else None))
taxi_col  = 'TAXI_IN' if 'TAXI_IN' in df.columns else (
            'AvgTaxiIn' if 'AvgTaxiIn' in df.columns else None)
if delay_col is None: raise RuntimeError("Delay column not found.")
if taxi_col  is None: raise RuntimeError("Taxi-in column not found.")

has_num_arrivals = 'NumArrivals' in df.columns
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

g_targets = df.groupby('__bin5').agg({delay_col: 'mean', taxi_col: 'mean'})
if has_num_arrivals:
    g_targets['__count'] = df.groupby('__bin5')['NumArrivals'].sum()
else:
    g_targets['__count'] = df.groupby('__bin5').size()

# U
u_primary  = ['sin_hour','cos_hour','Hour','DayOfWeek','Month','Holiday','DaysToHoliday']
u_fallback = ['HOUR_SIN','HOUR_COS','DOW_SIN','DOW_COS','MONTH_SIN','MONTH_COS',
              'MOD_SIN','MOD_COS','DAY_OF_WEEK_ENC','MONTH_ENC','MIN_OF_DAY','ARR_HOUR']
u_cols = [c for c in u_primary if c in df.columns] or [c for c in u_fallback if c in df.columns]
if not u_cols: raise RuntimeError("No usable exogenous input columns.")
g_u = df.groupby('__bin5')[u_cols].mean()

# 完整索引
full_idx = pd.date_range(min(g_targets.index.min(), g_u.index.min()),
                         max(g_targets.index.max(), g_u.index.max()), freq='5min')
g_targets = g_targets.reindex(full_idx)
g_u       = g_u.reindex(full_idx)

# 填充
g_targets[delay_col] = g_targets[delay_col].ffill()
g_targets[taxi_col]  = g_targets[taxi_col].ffill()
g_targets['__count'] = g_targets['__count'].fillna(0.0)

# numpy
y_delay = g_targets[delay_col].to_numpy(dtype=np.float32)
y_taxi  = g_targets[taxi_col ].to_numpy(dtype=np.float32)
y_count = g_targets['__count'].to_numpy(dtype=np.float32)
U       = g_u.to_numpy(dtype=np.float32)

# 清理
y_delay = np.nan_to_num(y_delay, nan=0.0, posinf=0.0, neginf=0.0)
y_taxi  = np.nan_to_num(y_taxi,  nan=0.0, posinf=0.0, neginf=0.0)
y_count = np.nan_to_num(y_count, nan=0.0, posinf=0.0, neginf=0.0)

# 标准化 U
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U = (U - U_mu) / U_sd
U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
U = np.clip(U, -5.0, 5.0)  # 输入限幅，防外点

# 最近 1h 到达量
arr_1h = pd.Series(g_targets['__count'], index=full_idx).rolling(12, min_periods=1).sum().to_numpy(np.float32)
U = np.concatenate([U, arr_1h[:, None]], axis=1)

# 目标归一化
def zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x) + 1e-8
    return (x - mu)/sd, mu, sd

y_delay_n, mu_delay, sd_delay = zscore(y_delay)
y_taxi_n,  mu_taxi,  sd_taxi  = zscore(y_taxi)
y_count_lp = np.log1p(np.clip(y_count, 0, None)).astype(np.float32)

dt_minutes = 5.0
dt = torch.tensor(dt_minutes/60.0, dtype=torch.float32)
T_total = U.shape[0]
print(f"[INFO] 5-min steps: {T_total:,} | U-dim: {U.shape[1]} | U-cols: {u_cols}+['arrivals_1h']")

# ============ 3) Model（加阻尼 + 有界化） ============
DYN_GAIN   = 0.02   # 动力学缩放（越小越稳）
STATE_DAMP = 0.10   # 线性阻尼系数（越大越稳，别太大以免欠拟合）
H_CLIP     = 5.0    # 隐状态最大范数（逐元素裁剪）

def xavier_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

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
        self.apply(xavier_init_)
        self.damp = STATE_DAMP

    def forward(self, h, u):
        if u.dim() == 1: u = u.unsqueeze(0)
        x = torch.cat([h, u], dim=-1)
        drift = self.net(x)
        # 合成：缩放后的非线性漂移 - 线性阻尼 * h
        return DYN_GAIN * drift - self.damp * h

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.delay_head = nn.Linear(hdim, 1)
        self.taxi_head  = nn.Linear(hdim, 1)
        self.count_head = nn.Linear(hdim, 1)
        self.apply(xavier_init_)
    def forward(self, h):
        d = self.delay_head(h).squeeze(-1)
        t = self.taxi_head(h).squeeze(-1)
        c = self.count_head(h).squeeze(-1)
        return d, t, c

class ControlledNODE(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt   = dt
        self.f    = Dynamics(hdim, u_dim)
        self.out  = Readout(hdim)
        self.h0   = nn.Parameter(torch.zeros(1, hdim))

    @torch.no_grad()
    def _ensure_2d(self, x):
        return x if x.dim() == 2 else x.unsqueeze(0)

    def rk4_step(self, h, u):
        u  = self._ensure_2d(u)
        dt = self.dt

        k1 = self.f(h,                   u)
        k2 = self.f(h + 0.5 * dt * k1,  u)
        k3 = self.f(h + 0.5 * dt * k2,  u)
        k4 = self.f(h + dt * k3,        u)

        h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 数值安全：就地消毒 + 有界投影
        h_next = torch.nan_to_num(h_next, nan=0.0, posinf=0.0, neginf=0.0)
        h_next = torch.tanh(h_next)               # 投影到 (-1,1)
        h_next = torch.clamp(h_next, -H_CLIP, H_CLIP)  # 再裁一层保险
        return h_next

    def forward(self, U, h0=None):
        T = U.size(0)
        h = self.h0 if h0 is None else h0
        d_list, t_list, c_list = [], [], []
        for t in range(T):
            d_hat, t_hat, c_hat = self.out(h)
            d_list.append(d_hat); t_list.append(t_hat); c_list.append(c_hat)
            h = self.rk4_step(h, U[t])
        return (torch.stack(d_list).squeeze(-1),
                torch.stack(t_list).squeeze(-1),
                torch.stack(c_list).squeeze(-1),
                h)

# ============ 4) Tensors ============
print(f"[info] tensors → {device}")
U_t           = torch.tensor(U,             dtype=torch.float32, device=device)
y_delay_t     = torch.tensor(y_delay_n,     dtype=torch.float32, device=device)
y_taxi_t      = torch.tensor(y_taxi_n,      dtype=torch.float32, device=device)
y_count_lp_t  = torch.tensor(y_count_lp,    dtype=torch.float32, device=device)

hdim = 32
model = ControlledNODE(hdim=hdim, u_dim=U_t.shape[1], dt=dt.to(device)).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)  # 更稳：更小 lr + 更强 wd
mse   = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)

def _sanitize_(t, name):
    if not torch.isfinite(t).all():
        print(f"[FIX] {name} had non-finite values. Forcing nan_to_num + clamp.")
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(t, -10.0, 10.0)

U_t           = _sanitize_(U_t,           "U_t")
y_delay_t     = _sanitize_(y_delay_t,     "y_delay_t")
y_taxi_t      = _sanitize_(y_taxi_t,      "y_taxi_t")
y_count_lp_t  = _sanitize_(y_count_lp_t,  "y_count_lp_t")

assert torch.isfinite(U_t).all() and torch.isfinite(y_delay_t).all() \
       and torch.isfinite(y_taxi_t).all() and torch.isfinite(y_count_lp_t).all(), \
       "Non-finite values remain after sanitize."

# ============ 5) Split & Warmup ============
T = U_t.shape[0]
split = int(0.8 * T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

model.train()
warm_L = min(64, T)
with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
              enabled=(device.type=="cuda")):
    d_hat, t_hat, c_hat, _ = model(U_t[:warm_L])
    for z in (d_hat, t_hat, c_hat):
        z.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    warm_loss = mse(d_hat, y_delay_t[:warm_L]) + mse(t_hat, y_taxi_t[:warm_L]) + mse(c_hat, y_count_lp_t[:warm_L])

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

# ============ 6) Training (TBPTT, NaN-safe) ============
epochs = 60
window = 256
best_val = float("inf")

outer = tqdm(range(1, epochs+1), desc="Epochs", position=0, ncols=100)
for ep in outer:
    model.train()
    total_loss = 0.0
    skipped = 0
    h = None
    inner = tqdm(range(0, T, window), desc=f"[Train] e{ep}", leave=False, position=1, ncols=100)
    for start in inner:
        end = min(start + window, T)
        U_blk  = U_t[start:end]
        yd_blk = y_delay_t[start:end]
        yt_blk = y_taxi_t[start:end]
        yc_blk = y_count_lp_t[start:end]

        with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                      enabled=(device.type=="cuda")):
            d_hat, t_hat, c_hat, h = model(U_blk, h0=h)

            # 输出消毒 + 限幅
            for z in (d_hat, t_hat, c_hat):
                z.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                z.clamp_(-10.0, 10.0)

            # 隐状态保底检查
            if not torch.isfinite(h).all():
                skipped += 1
                h = torch.zeros_like(h)
                continue

            loss = mse(d_hat, yd_blk) + mse(t_hat, yt_blk) + mse(c_hat, yc_blk)

        if not torch.isfinite(loss):
            skipped += 1
            h = torch.zeros_like(h)
            continue

        optim.zero_grad(set_to_none=True)
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        # 断开跨窗口计算图
        if h is not None:
            h = h.detach()

        total_loss += float(loss.detach().cpu())

    # Validation
    model.eval()
    h = None
    with torch.no_grad(), autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                                   enabled=(device.type=="cuda")):
        d_hat_v, t_hat_v, c_hat_v, _ = model(U_t[valid_slice], h0=h)
        for z in (d_hat_v, t_hat_v, c_hat_v):
            z.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        val_loss = (mse(d_hat_v, y_delay_t[valid_slice]) +
                    mse(t_hat_v,  y_taxi_t[valid_slice]) +
                    mse(c_hat_v,  y_count_lp_t[valid_slice])).item()

    outer.set_postfix(train=f"{total_loss:.4f}", val=f"{val_loss:.4f}",
                      lr=f"{optim.param_groups[0]['lr']:.1e}", skip=skipped)
    scheduler.step()

    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model": model.state_dict()}, "controlled_node_best_rk4.pt")

print("Training done. Best val loss:", best_val)

# ============ 7) Visualization ============
print("[INFO] plotting results ...")
ckpt = torch.load("controlled_node_best_rk4.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

with torch.no_grad():
    d_hat_all, t_hat_all, c_hat_all, _ = model(U_t)
    for z in (d_hat_all, t_hat_all, c_hat_all):
        z.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    delay_pred_n  = d_hat_all.cpu().numpy()
    taxi_pred_n   = t_hat_all.cpu().numpy()
    count_pred_lp = c_hat_all.cpu().numpy()

delay_pred = delay_pred_n * sd_delay + mu_delay
taxi_pred  = taxi_pred_n  * sd_taxi  + mu_taxi
count_pred = np.expm1(count_pred_lp)
count_pred = np.clip(count_pred, 0.0, None)

delay_true = y_delay
taxi_true  = y_taxi
count_true = y_count
time_axis  = np.arange(T_total)

plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(time_axis, delay_true, label='True Delay', color='black', alpha=0.6)
plt.plot(time_axis, delay_pred,  label='Pred Delay',  color='red',   alpha=0.8)
plt.title("Arrival Delay (minutes)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(time_axis, taxi_true, label='True TaxiIn', color='black', alpha=0.6)
plt.plot(time_axis, taxi_pred, label='Pred TaxiIn', color='blue',  alpha=0.8)
plt.title("Average Taxi-In Time (minutes)")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(time_axis, count_true, label='True NumArrivals', color='black', alpha=0.6)
plt.plot(time_axis, count_pred, label='Pred NumArrivals', color='green', alpha=0.8)
plt.title("Number of Arrivals per 5min")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("controlled_node_results_rk4.png", dpi=200)
plt.show()
print("[INFO] figure saved as controlled_node_results_rk4.png")
