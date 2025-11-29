# =============================================================
#  Plot: feature vs predicted probability
#  (No y_bin, no y_prob, no 0/1 curves)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ---------- Load checkpoint ----------
ckpt = torch.load("node_delay_prob_best.pt", map_location="cpu")

U_mu = ckpt["U_mu"]
U_sd = ckpt["U_sd"]

print("[INFO] ckpt keys:", ckpt.keys())

# ---------- Load CSV ----------
df = pd.read_csv("./data/ref/bwi_flights_with_weather.csv",
                 parse_dates=["CRS_ARR_TIME_dt"])
df = df.sort_values("CRS_ARR_TIME_dt")
df["__bin5"] = df["CRS_ARR_TIME_dt"].dt.floor("5min")

# ---------- Feature columns ----------
u_cols = [
    "visibility", "low_ceiling_flag", "wind_speed",
    "rain_flag", "snow_flag", "thunder_flag", "fog_flag", "ice_flag",
    "HOUR_SIN", "HOUR_COS", "DOW_SIN", "DOW_COS",
    "MONTH_SIN", "MONTH_COS",
    "holiday_flag"
]
u_cols = [c for c in u_cols if c in df.columns]

g_u = df.groupby("__bin5")[u_cols].mean()
g_cnt = df.groupby("__bin5").size()

arr_1h = g_cnt.rolling(12, min_periods=1).sum()
g_u["arrivals_1h"] = arr_1h
g_u = g_u.fillna(0.0)

feature_names = list(g_u.columns)

# ---------- Normalize ----------
U_np = g_u.to_numpy(np.float32)
U_np = (U_np - U_mu) / U_sd
U_np = np.clip(U_np, -3, 3)

U_t = torch.tensor(U_np, dtype=torch.float32)

# ---------- Load model ----------
from node_delay_prob import NODE_Classifier, dt

hdim = 32
model = NODE_Classifier(hdim=hdim,
                        u_dim=U_t.shape[1],
                        dt=dt)
model.load_state_dict(ckpt["model"])
model.eval()

# ---------- Predict ----------
with torch.no_grad():
    logits, _ = model(U_t)
    probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

print("[INFO] predicted prob shape:", probs.shape)

# =============================================================
#      PLOT feature vs probability (only scatter plots)
# =============================================================

for fname in feature_names:
    plt.figure(figsize=(7,5))
    plt.scatter(g_u[fname], probs, s=4, alpha=0.2)
    plt.xlabel(fname)
    plt.ylabel("Predicted Delay Probability")
    plt.title(f"Feature vs Predicted Delay Probability: {fname}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"feature_vs_prob_{fname}.png", dpi=200)
    plt.close()

print("[INFO] Finished: saved feature_vs_prob_*.png")
