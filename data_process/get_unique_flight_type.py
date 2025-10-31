import pandas as pd

# 读取文件
df = pd.read_csv("tails_with_standardized_models_v2.csv")

# 提取唯一机型（忽略空值和 nan）
unique_models = (
    df["MODEL_STD"]
    .dropna()
    .astype(str)
    .str.strip()
    .replace(["nan", "None", ""], pd.NA)
    .dropna()
    .unique()
)

# 排序
unique_models = sorted(unique_models)

# 保存结果
pd.DataFrame({"MODEL_STD": unique_models}).to_csv("unique_model_std_list.csv", index=False)

print(f"共 {len(unique_models)} 个唯一机型。结果已保存为 unique_model_std_list.csv。")
