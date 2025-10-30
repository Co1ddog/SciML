import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ============ 1) 读数据 ============
# 假设保存为 arrivals.csv，含你给的列
df = pd.read_csv('./data/train/merged_train_set.csv', parse_dates=['CRS_ARR_TIME_dt'])
df = df.sort_values('CRS_ARR_TIME_dt').reset_index(drop=True)

# 生成等间隔时间网格（分钟），并对缺失分钟做前向填充（确保是规则网格）
df = df.set_index('CRS_ARR_TIME_dt').asfreq('5min')  # 例如每5分钟一个采样；若是逐分钟可改为 '1T'
df[['ArrDelayMinutes','NumArrivals','AvgTaxiIn']] = df[['ArrDelayMinutes','NumArrivals','AvgTaxiIn']].ffill()
df[['Hour','DayOfWeek','Month','sin_hour','cos_hour','Holiday','DaysToHoliday']] = \
    df[['Hour','DayOfWeek','Month','sin_hour','cos_hour','Holiday','DaysToHoliday']].ffill()
df = df.reset_index().rename(columns={'index':'CRS_ARR_TIME_dt'})

# 归一化（简单起见，只对回归目标与部分输入做 z-score；计数用对数强度不需要标准化）
def zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x) + 1e-8
    return (x - mu)/sd, mu, sd

y_delay, mu_delay, sd_delay = zscore(df['ArrDelayMinutes'].values.astype(np.float32))
y_taxi,  mu_taxi,  sd_taxi  = zscore(df['AvgTaxiIn'].values.astype(np.float32))

y_count = df['NumArrivals'].values.astype(np.float32)  # 保持原始计数

# 外生输入 u(t)：建议包含周期与节假日、距节假日
u_cols = ['sin_hour','cos_hour','Hour','DayOfWeek','Month','Holiday','DaysToHoliday']
U = df[u_cols].values.astype(np.float32)

# 时间步长（分钟为单位；与 asfreq 保持一致）
dt_minutes = 5.0
dt = torch.tensor(dt_minutes/60.0, dtype=torch.float32)  # 转成小时为单位（随意，只要一致）

T = len(df)

# ============ 2) 定义受控 Neural ODE 模型 ============
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
        # dh/dt = f(h, u)
        x = torch.cat([h, u], dim=-1)
        return self.net(x)

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.delay_head = nn.Linear(hdim, 1)     # 回归
        self.taxi_head  = nn.Linear(hdim, 1)     # 回归
        self.count_head = nn.Linear(hdim, 1)     # Poisson 强度的 log λ
    def forward(self, h):
        delay_hat = self.delay_head(h)           # 标准化空间
        taxi_hat  = self.taxi_head(h)            # 标准化空间
        loglam    = self.count_head(h)           # 直接输出 log λ
        return delay_hat.squeeze(-1), taxi_hat.squeeze(-1), loglam.squeeze(-1)

class ControlledNODE(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt = dt
        self.f = Dynamics(hdim, u_dim)
        self.out = Readout(hdim)
        self.h0 = nn.Parameter(torch.zeros(1, hdim))  # 带 batch 维

    def step(self, h, u):
        if u.dim() == 1:
            u = u.unsqueeze(0)        # (1, u_dim)
        dh = self.f(h, u) * self.dt    # h: (1, hdim)
        return h + dh                  # (1, hdim)

    def forward(self, U, h0=None):
        T = U.size(0)
        h = self.h0 if h0 is None else h0  # (1, hdim)

        delay_preds, taxi_preds, loglam_preds = [], [], []

        for t in range(T):
            d_hat, tx_hat, ll_hat = self.out(h)   # (1,1) 或 (1,)
            delay_preds.append(d_hat)
            taxi_preds.append(tx_hat)
            loglam_preds.append(ll_hat)
            h = self.step(h, U[t])                # U[t]: (u_dim,)

        delay_preds = torch.stack(delay_preds).squeeze(-1).squeeze(-1)  # (T,)
        taxi_preds  = torch.stack(taxi_preds ).squeeze(-1).squeeze(-1)  # (T,)
        loglam_preds= torch.stack(loglam_preds).squeeze(-1).squeeze(-1) # (T,)

        return delay_preds, taxi_preds, loglam_preds, h


# ============ 3) 准备张量 & 训练 ============
device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)

print("Using device:", device)

U_t = torch.tensor(U, dtype=torch.float32, device=device)
y_delay_t = torch.tensor(y_delay, dtype=torch.float32, device=device)
y_taxi_t  = torch.tensor(y_taxi,  dtype=torch.float32, device=device)
y_count_t = torch.tensor(y_count, dtype=torch.float32, device=device)

hdim = 16
model = ControlledNODE(hdim=hdim, u_dim=U_t.shape[1], dt=dt.to(device)).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss(reduction='mean')

def poisson_nll(log_lambda, y):
    # NLL = exp(logλ) - y*logλ + log(y!)；常数项可忽略
    return (torch.exp(log_lambda) - y * log_lambda).mean()

# 简单切分训练/验证（按时间）
split = int(0.8 * T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

# 预热（MPS 第一次会编译内核，先用小窗口走一遍）
model.train()
warm_L = min(64, U_t.shape[0])
d_hat, t_hat, loglam, _ = model(U_t[:warm_L])
warm_loss = (
    mse(d_hat, y_delay_t[:warm_L]) +
    mse(t_hat,  y_taxi_t[:warm_L]) +
    poisson_nll(loglam, y_count_t[:warm_L])
)
optim.zero_grad(set_to_none=True)
warm_loss.backward()
optim.step()

# ===== TBPTT 训练 =====
epochs = 100
window = 256   # 可调：128/256/512
T = U_t.shape[0]

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    h = None  # 跨窗口传递隐藏态

    pbar = tqdm(range(0, T, window), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for start in pbar:
        end = min(start + window, T)
        U_blk = U_t[start:end]
        yd_blk = y_delay_t[start:end]
        yt_blk = y_taxi_t[start:end]
        yc_blk = y_count_t[start:end]

        d_hat, t_hat, loglam, h = model(U_blk, h0=h)  # 让 forward 支持 h0
        loss = mse(d_hat, yd_blk) + mse(t_hat, yt_blk) + poisson_nll(loglam, yc_blk)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # 阻断跨窗口的反向传播
        if h is not None:
            h = h.detach()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1}/{epochs} | Train {total_loss:.4f}")

# ============ 4) 反归一化与示例预测 ============
model.eval()
with torch.no_grad():
    d_hat_all, t_hat_all, loglam_all, _ = model(U_t)
    delay_pred = d_hat_all.cpu().numpy()*sd_delay + mu_delay
    taxi_pred  = t_hat_all.cpu().numpy()*sd_taxi  + mu_taxi
    count_pred = np.exp(loglam_all.cpu().numpy())  # 期望值 λ(t)

# 把预测拼回 DataFrame 便于画图/评估
df['Delay_Pred']  = delay_pred
df['Taxi_Pred']   = taxi_pred
df['Arrivals_E']  = count_pred  # 期望到达架次（可与实际 NumArrivals 对比）
print(df.tail())
