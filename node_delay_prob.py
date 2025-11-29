# ============================================================
# node_delay_prob.py —— Neural ODE 分类版（预测延误概率）
# - 目标: ARR_DEL15，按 5min 聚合为窗口延误概率
# - 输出: 预测概率 vs 真实 0/1；特征重要性；高/低风险窗口特征
# ============================================================

import platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ================== 0) Device & AMP ==================
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

# ================== 1) Load CSV ==================
CSV_PATH = "./data/ref/bwi_flights_with_weather.csv"
print("[INFO] loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['CRS_ARR_TIME_dt'], low_memory=False)
print(f"[INFO] loaded rows: {len(df):,}")

df = df.sort_values('CRS_ARR_TIME_dt')

# ---------- 分类目标：ARR_DEL15 ----------
TARGET = "ARR_DEL15"
if TARGET not in df.columns:
    raise RuntimeError("ARR_DEL15 not found. Make sure your dataset contains it.")

# ================== 2) 5-min aggregation → probability target ==================
df['__bin5'] = df['CRS_ARR_TIME_dt'].dt.floor('5min')

# 每 5min 的延误比例（平均 0/1 就是概率）
g_targets = df.groupby('__bin5').agg({TARGET: 'mean'})
g_targets['__count'] = df.groupby('__bin5').size()

# U 输入列（时间+天气+假日）
u_cols = [
    "visibility", "low_ceiling_flag", "wind_speed",
    "rain_flag", "snow_flag", "thunder_flag", "fog_flag", "ice_flag",
    "HOUR_SIN", "HOUR_COS", "DOW_SIN", "DOW_COS",
    "MONTH_SIN", "MONTH_COS",
    "holiday_flag"
]
u_cols = [c for c in u_cols if c in df.columns]
print("[INFO] U columns (without arrivals_1h):", u_cols)

g_u = df.groupby('__bin5')[u_cols].mean()

# 完整时间轴
full_idx = pd.date_range(
    min(g_targets.index.min(), g_u.index.min()),
    max(g_targets.index.max(), g_u.index.max()),
    freq='5min'
)

g_targets = g_targets.reindex(full_idx)
g_u       = g_u.reindex(full_idx)

# 补全
g_targets[TARGET] = g_targets[TARGET].fillna(0.0)
g_targets['__count'] = g_targets['__count'].fillna(0)

y_prob = g_targets[TARGET].to_numpy(np.float32)   # 每窗口平均延误比例
U      = g_u.to_numpy(np.float32)                 # exogenous features

y_prob = np.nan_to_num(y_prob, nan=0.0)
U      = np.nan_to_num(U,       nan=0.0)

# 最近 1h 到达量 (12 个 5min 窗口的和)
arr_1h = (
    pd.Series(g_targets['__count'], index=full_idx)
      .rolling(12, min_periods=1)
      .sum()
      .to_numpy(np.float32)
)
U = np.concatenate([U, arr_1h[:, None]], axis=1)
u_cols_plus = u_cols + ["arrivals_1h"]
print("[INFO] U total dim:", U.shape[1], "| feature names:", u_cols_plus)

# ========== 限制时间序列长度，加速训练 ==========
MAX_STEPS = 6000
T_full = len(U)

if T_full > MAX_STEPS:
    start = T_full - MAX_STEPS
    print(f"[INFO] trim: keep last {MAX_STEPS}/{T_full}")
    U = U[start:]
    y_prob = y_prob[start:]
    full_idx = full_idx[start:]

T_total = len(U)
print(f"[INFO] final time steps: {T_total}")

# ========== Normalize U ==========
U_mu = np.nanmean(U, axis=0)
U_sd = np.nanstd(U, axis=0) + 1e-8
U_norm = (U - U_mu) / U_sd
U_norm = np.clip(U_norm, -3, 3)

dt_minutes = 5.0
dt = torch.tensor(dt_minutes/60.0, dtype=torch.float32)

# ================== 3) Model (Classification) ==================
DYN_GAIN   = 0.01     # 更小，防爆炸
STATE_DAMP = 0.30     # 更强阻尼
H_CLIP     = 3.0      # 限制隐状态

def xavier_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

class Dynamics(nn.Module):
    def __init__(self, hdim, u_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hdim + u_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, hdim),
        )
        self.apply(xavier_init_)
        self.damp = STATE_DAMP

    def forward(self, h, u):
        if u.dim() == 1:
            u = u.unsqueeze(0)
        x = torch.cat([h, u], dim=-1)
        drift = self.net(x)
        return DYN_GAIN * drift - self.damp * h

class Readout(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.fc = nn.Linear(hdim, 1)
        self.apply(xavier_init_)

    def forward(self, h):
        logits = self.fc(h)
        logits = torch.clamp(logits, -10, 10)  # 防爆炸
        return logits.squeeze(-1)

class NODE_Classifier(nn.Module):
    def __init__(self, hdim, u_dim, dt):
        super().__init__()
        self.hdim = hdim
        self.dt = dt
        self.f  = Dynamics(hdim, u_dim)
        self.out = Readout(hdim)
        self.h0 = nn.Parameter(torch.zeros(1, hdim))

    def rk4_step(self, h, u):
        u = u.unsqueeze(0)
        dt = self.dt
        k1 = self.f(h, u)
        k2 = self.f(h + 0.5*dt*k1, u)
        k3 = self.f(h + 0.5*dt*k2, u)
        k4 = self.f(h + dt*k3,     u)
        h_next = h + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        h_next = torch.tanh(h_next)
        return torch.clamp(h_next, -H_CLIP, H_CLIP)

    def forward(self, U, h0=None):
        T = U.size(0)
        h = self.h0 if h0 is None else h0
        out_list = []
        for t in range(T):
            logits = self.out(h)
            out_list.append(logits)
            h = self.rk4_step(h, U[t])
        return torch.stack(out_list), h

# ================== 4) Tensor Setup ==================
print("[info] tensors →", device)
U_t     = torch.tensor(U_norm, dtype=torch.float32, device=device)
y_t     = torch.tensor(y_prob, dtype=torch.float32, device=device)

T = len(U_t)
split = int(0.8 * T)
train_slice = slice(0, split)
valid_slice = slice(split, T)

# 类别不平衡处理：正例大约 2~5%
pos_rate = float(y_t.mean().cpu())
pos_weight = torch.tensor([(1 - pos_rate) / (pos_rate + 1e-6)], device=device)
print(f"[INFO] positive rate ~ {pos_rate:.4f}, pos_weight={pos_weight.item():.1f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

hdim = 32
model = NODE_Classifier(hdim=hdim, u_dim=U_t.shape[1], dt=dt.to(device)).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=40)

# ================== 5) Warmup ==================
model.train()
warm_L = min(64, T)
with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
              enabled=(device.type=="cuda")):
    logits, _ = model(U_t[:warm_L])
    logits = logits.squeeze(-1)
    warm_loss = criterion(logits, y_t[:warm_L])

optim.zero_grad()
warm_loss.backward()
optim.step()
print("[INFO] warmup done")

# ================== 6) Training ==================
epochs = 40
window = 256
best_val = 1e9

outer = tqdm(range(1, epochs+1), ncols=100, desc="Epochs")
for ep in outer:
    model.train()
    total_loss = 0.0
    h = None

    for start in range(0, T, window):
        end = min(start + window, T)
        U_blk = U_t[start:end]
        y_blk = y_t[start:end]

        logits, h = model(U_blk, h0=h)
        logits = logits.squeeze(-1)

        loss = criterion(logits, y_blk)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        h = h.detach()
        total_loss += loss.detach().item()

    # validation
    model.eval()
    with torch.no_grad():
        logits_v, _ = model(U_t[valid_slice])
        logits_v = logits_v.squeeze(-1)
        val_loss = criterion(logits_v, y_t[valid_slice]).item()

    outer.set_postfix(train=f"{total_loss:.4f}", val=f"{val_loss:.4f}")

    scheduler.step()

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "model": model.state_dict(),
            "U_mu": U_mu.tolist(),
            "U_sd": U_sd.tolist(),
            "u_cols_plus": u_cols_plus,
        }, "node_delay_prob_best.pt")

print("[INFO] training done. best val =", best_val)

# ================== 7) Visualization (0/1 window delay + predicted prob) ==================
print("\n[INFO] Visualization: Plotting predicted probability vs true 0/1 delay...")

ckpt = torch.load(
    "node_delay_prob_best.pt",
    map_location=device,
    weights_only=False  # 显式关闭 weights_only
)
model.load_state_dict(ckpt["model"])
model.eval()

# forward pass
with torch.no_grad():
    logits_all, _ = model(U_t)
    logits_all = logits_all.squeeze(-1)
    prob_all = torch.sigmoid(logits_all).cpu().numpy()   # 预测概率（0~1）

# ground truth: y_prob 是窗口平均延误比例，y_bin 是是否有延误（0/1）
y_bin = (y_prob > 0).astype(np.float32)
time_axis = np.arange(len(prob_all))

# plt.figure(figsize=(15, 7))
# plt.plot(time_axis, prob_all, label="Predicted Delay Probability (0~1)", color="red", alpha=0.7)
# plt.step(time_axis, y_bin, label="True Delay (0/1 per 5min)", color="black", linewidth=1.2)
# plt.title("Neural ODE — Delay Probability vs True Delay (5-minute Windows)")
# plt.xlabel("Time (5-min steps)")
# plt.ylabel("Probability")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("node_delay_prob_results_bin.png", dpi=200)
# plt.show()
# print("[INFO] figure saved as node_delay_prob_results_bin.png")

# ---------- Visualization: Show prob vs true delay rate ----------
plt.figure(figsize=(15, 7))

# 预测概率（红线）
plt.plot(prob_all, label="Predicted Delay Probability", color="red", alpha=0.8)

# 真实延误比例（蓝线）
plt.plot(y_prob, label="True Delay Rate (0~1 avg in 5min)", color="blue", alpha=0.5)

plt.title("Neural ODE — Predicted Probability vs True Delay Rate (5-minute windows)")
plt.xlabel("Time (5-min steps)")
plt.ylabel("Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("node_delay_prob_results_rate.png", dpi=200)
plt.show()

print("[INFO] figure saved as node_delay_prob_results_rate.png")


# ================== 8) Feature Importance via Gradient (Saliency) ==================
print("\n[INFO] Computing feature importance (gradient-based)...")

model.eval()
# 重新构造一个需要梯度的 U_t
U_t_grad = torch.tensor(U_norm, dtype=torch.float32, device=device, requires_grad=True)

logits_v, _ = model(U_t_grad[valid_slice])
logits_v = logits_v.squeeze(-1)
probs_v = torch.sigmoid(logits_v)

# 取验证集概率最高的那个窗口
idx_rel = torch.argmax(probs_v).item()          # 验证集内的索引
idx_abs = split + idx_rel                       # 在全序列中的绝对索引

loss_sample = logits_v[idx_rel]  # 对这个 logit 求梯度
loss_sample.backward()

grads = U_t_grad.grad[valid_slice][idx_rel].detach().cpu().numpy()
importance = np.abs(grads)

feat_names = u_cols_plus
imp_sorted = sorted(zip(feat_names, importance), key=lambda x: -x[1])

print("\n===== Feature Importance (Gradient-based, at highest-prob window) =====")
for name, score in imp_sorted:
    print(f"{name:20s} : {score:.6f}")

pd.DataFrame(imp_sorted, columns=["feature", "importance"]) \
  .to_csv("feature_importance_grad.csv", index=False)
print("[INFO] Feature importance saved to feature_importance_grad.csv")

# ================== 9) 高 / 低概率窗口的特征值导出 ==================
print("\n[INFO] Exporting high-probability and low-probability windows with features...")

# 反标准化 U，回到原始尺度（含 arrivals_1h）
U_denorm = U_norm * U_sd + U_mu  # shape: [T, dim]
df_windows = pd.DataFrame(U_denorm, columns=feat_names, index=full_idx)
df_windows["pred_prob"] = prob_all
df_windows["true_rate"] = y_prob
df_windows["true_bin"]  = y_bin

# Top-K 高概率 / 低概率窗口
K = 20
high_idx = np.argsort(prob_all)[-K:][::-1]
low_idx  = np.argsort(prob_all)[:K]

high_df = df_windows.iloc[high_idx].copy()
low_df  = df_windows.iloc[low_idx].copy()

high_df.sort_values("pred_prob", ascending=False, inplace=True)
low_df.sort_values("pred_prob", ascending=True, inplace=True)

high_df.to_csv("high_prob_windows.csv")
low_df.to_csv("low_prob_windows.csv")

print("[INFO] Saved top-K high-probability windows to high_prob_windows.csv")
print("[INFO] Saved top-K low-probability windows to low_prob_windows.csv")
print("[INFO] Example high-prob rows:")
print(high_df.head(5))
print("\n[INFO] Example low-prob rows:")
print(low_df.head(5))
