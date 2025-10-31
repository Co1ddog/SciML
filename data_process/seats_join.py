import pandas as pd

# 1️⃣ 读取两份文件
flights = pd.read_csv("tails_with_standardized_models_v2.csv")
seats_ref = pd.read_csv("unique_model_std_list.csv")

# 2️⃣ 清理字段（防止大小写/空格不匹配）
flights["MODEL_STD"] = flights["MODEL_STD"].astype(str).str.strip().str.upper()
seats_ref["MODEL_STD"] = seats_ref["MODEL_STD"].astype(str).str.strip().str.upper()

# 3️⃣ 合并：左连接，保持所有航班行
merged = pd.merge(
    flights,
    seats_ref[["MODEL_STD", "MAX_SEATS"]],
    on="MODEL_STD",
    how="left"
)

# 4️⃣ 导出新文件
merged.to_csv("../data/ref/tails_num_2_seats_num.csv", index=False)

print(f"合并完成，共 {len(merged)} 条航班记录。")
print(f"新文件：tails_with_max_seats.csv")
print(f"已为 {merged['MAX_SEATS'].notna().sum()} 条记录补上 MAX_SEATS。")
