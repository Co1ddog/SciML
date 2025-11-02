import pandas as pd

# 读取CSV文件
files = ['train', 'test', 'validation']
all_airlines = set()

for file in files:
    df = pd.read_csv(f"../data/ref/flights_{file}_raw_with_model_and_seats.csv")
    # 获取某一列的唯一值
    # unique_values = df["OP_UNIQUE_CARRIER"].unique()
    # print(f"{file} dataset has airline:" )
    # print(unique_values)
    all_airlines.update(df["OP_UNIQUE_CARRIER"].dropna().unique())

print("All unique airlines across datasets:")
print(sorted(all_airlines))
