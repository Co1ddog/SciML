import pandas as pd
from pathlib import Path

# === 文件路径 ===
TAILS_CSV   = "tail_registry_template.csv"
MASTER_TXT  = "MASTER.txt"      # 逗号分隔
ACFTREF_TXT = "ACFTREF.txt"     # 逗号分隔
OUT_CSV     = "tails_with_models_cleaned.csv"

# === 1) 读取尾号清单 ===
tails = pd.read_csv(TAILS_CSV, dtype=str)

# 你CSV里尾号那一列的名字（改成你实际的）
col_tail = "tail_number"  # ← 请改成你的列名！

# 如果列名不确定，可自动猜测
if col_tail not in tails.columns:
    lower = {c.lower(): c for c in tails.columns}
    for k in ("tail", "tail_number", "registration", "reg", "n-number", "n_number"):
        if k in lower:
            col_tail = lower[k]
            break
    else:
        col_tail = tails.columns[0]

# 清洗尾号，统一格式
tails["TAIL_NUM"] = tails[col_tail].astype(str).str.strip().str.upper()

# FAA 格式：去掉前缀 N 和连接符（因为 MASTER 里不带 N）
def to_master_key(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip().upper().replace("-", "").replace(" ", "")
    return s[1:] if s.startswith("N") else s

tails["N_NUMBER_KEY"] = tails["TAIL_NUM"].map(to_master_key)

# === 2) 读取 FAA 主表和机型参考表 ===
master = pd.read_csv(MASTER_TXT, delimiter=",", dtype=str)
ref    = pd.read_csv(ACFTREF_TXT, delimiter=",", dtype=str)

# 去空格
for c in ("N-NUMBER", "MFR MDL CODE", "YEAR MFR"):
    if c in master.columns:
        master[c] = master[c].astype(str).str.strip()

if "CODE" in ref.columns:
    ref["CODE"] = ref["CODE"].astype(str).str.strip()
for c in ("MFR", "MODEL"):
    if c in ref.columns:
        ref[c] = ref[c].astype(str).str.strip()

# === 3) 合并 ===
m1 = tails.merge(master, left_on="N_NUMBER_KEY", right_on="N-NUMBER", how="left")
m2 = m1.merge(ref, left_on="MFR MDL CODE", right_on="CODE", how="left")

# === 4) 判断哪些匹配成功 ===
m2["MATCHED"] = m2["MODEL"].notna() & (m2["MODEL"].astype(str).str.len() > 0)

# === 5) 提取需要的四列 ===
cols = ["TAIL_NUM", "MFR", "MODEL", "YEAR MFR"]
cols = [c for c in cols if c in m2.columns]
out = m2[cols + ["MATCHED"]].copy()

# === 6) 先排匹配到的，再排未匹配的 ===
out_sorted = out.sort_values(by="MATCHED", ascending=False, ignore_index=True).drop(columns=["MATCHED"])

# === 7) 导出 ===
out_sorted.to_csv(OUT_CSV, index=False)
print(f"✅ 已生成 {Path(OUT_CSV).resolve()}")
print(f"共 {len(out_sorted)} 条，其中匹配成功 {out['MATCHED'].sum()} 条。")
