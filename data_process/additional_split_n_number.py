import pandas as pd

# 读取文件
df = pd.read_csv("tail_registry_template.csv", dtype=str)

# 确保列名一致，比如 tail number 在 "N-NUMBER" 或 "tail_number"
# 你可以先 print(df.columns) 看看实际列名
col = "TAIL_NUM"

# 区分 N 开头与非 N 开头
df_n = df[df[col].str.startswith("N", na=False)]
df_non_n = df[~df[col].str.startswith("N", na=False)]

# 分别保存
df_n.to_csv("tail_registry_N_only.csv", index=False)
df_non_n.to_csv("tail_registry_non_N.csv", index=False)

print(f"N开头: {len(df_n)} 条, 非N开头: {len(df_non_n)} 条")
