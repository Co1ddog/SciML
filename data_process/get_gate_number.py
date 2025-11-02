import pandas as pd

# 每个 concourse 的 gate 数
concourse_gate_counts = {"A": 11, "B": 14, "C": 14, "D": 22, "E": 5}

def sum_gate_counts(concourse_str):
    if pd.isna(concourse_str) or not str(concourse_str).strip():
        return None
    # 支持 "A, B, C, E" 或 "A,B,C,E"
    concourses = [c.strip() for c in str(concourse_str).split(",")]
    total = 0
    for c in concourses:
        if c in concourse_gate_counts:
            total += concourse_gate_counts[c]
    return total if total > 0 else None

def per_concourse_cols(concourse_str):
    # 返回 A/B/C/D/E 的 gate 数（不存在则为0）
    vals = {k: 0 for k in concourse_gate_counts.keys()}
    if pd.isna(concourse_str) or not str(concourse_str).strip():
        return vals
    concourses = [c.strip() for c in str(concourse_str).split(",")]
    for c in concourses:
        if c in concourse_gate_counts:
            vals[c] = concourse_gate_counts[c]
    return vals

datasets = ["train", "test", "validation"]

for name in datasets:
    path = f"../data/ref/flights_{name}_with_airline_concourse.csv"  # 上一步已生成的带 CONCOURSE 的文件
    df = pd.read_csv(path)

    # 总 gate 数（针对该行的 CONCOURSE，多个 concourse 求和）
    df["GATE_COUNT"] = df["CONCOURSE"].apply(sum_gate_counts)

    # 拆分出 A/B/C/D/E 各自的 gate 数列
    per_cols = df["CONCOURSE"].apply(per_concourse_cols).apply(pd.Series)
    per_cols = per_cols.rename(columns={
        "A": "A_GATES", "B": "B_GATES", "C": "C_GATES", "D": "D_GATES", "E": "E_GATES"
    })
    df = pd.concat([df, per_cols], axis=1)

    # 保存
    out_path = f"../data/ref/flights_{name}_with_airline_concourse_and_gates.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{out_path}")
