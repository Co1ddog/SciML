import pandas as pd
from pathlib import Path

# === 定位路径 ===
base_dir = Path(__file__).resolve().parent  # dataprocess/
data_dir = base_dir.parent / "data"         # ../data/
ref_dir = data_dir / "ref"                  # ../data/ref/

# === 文件路径 ===
ref_path = ref_dir / "tails_num_2_seats_num.csv"
datasets = [
    data_dir / "train" / "flights_train_raw.csv",
    data_dir / "test" / "flights_test_raw.csv",
    data_dir / "validation" / "flights_validation_raw.csv"
]

# === 读取参考表 ===
ref = pd.read_csv(ref_path)
ref["TAIL_NUM"] = ref["TAIL_NUM"].astype(str).str.strip().str.upper()
ref["TAIL_NUM_CLEAN"] = ref["TAIL_NUM"].str.replace("^N", "", regex=True)

# 确认字段存在
if not {"MODEL_STD", "MAX_SEATS"}.issubset(ref.columns):
    raise ValueError("❌ tails_num_2_seats_num.csv 必须包含 MODEL_STD 和 MAX_SEATS 列")

# === 输出目录（确保存在） ===
out_dir = ref_dir
out_dir.mkdir(parents=True, exist_ok=True)

# 美国2025年7月数据，飞机平均载客率为86%
passenger_load_factor = 0.86

# === 遍历三个数据集 ===
for file in datasets:
    df = pd.read_csv(file)
    
    # 清洗尾号
    df["TAIL_NUM"] = df["TAIL_NUM"].astype(str).str.strip().str.upper()
    df["TAIL_NUM_CLEAN"] = df["TAIL_NUM"].str.replace("^N", "", regex=True)
    
    # 合并
    merged = pd.merge(
        df,
        ref[["TAIL_NUM_CLEAN", "MODEL_STD", "MAX_SEATS"]],
        on="TAIL_NUM_CLEAN",
        how="left"
    )
    
    # 添加 expected passengers (based on 86% load factor)
    merged["EXPECTED_PASSENGERS"] = (merged["MAX_SEATS"] * passenger_load_factor).round(0)

    # 删除临时列
    merged.drop(columns=["TAIL_NUM_CLEAN"], inplace=True)
    
    # 输出文件名（带前缀）
    out_path = out_dir / f"{Path(file).stem}_with_model_and_seats.csv"
    merged.to_csv(out_path, index=False)
    
    print(f"✅ {file.name} → {out_path.name}")
    print(f"   匹配成功 {merged['MAX_SEATS'].notna().sum()} 条记录")

print(f"🎯 所有结果已保存到: {out_dir.resolve()}")
