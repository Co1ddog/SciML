from pathlib import Path
import pandas as pd

# 路径
BASE = Path(__file__).resolve().parent
REF_DIR = BASE.parent / "data" / "ref"

FILES = [
    "T_ONTIME_REPORTING_train_ALL_enriched.csv",
    "T_ONTIME_REPORTING_test_ALL_enriched.csv",
    "T_ONTIME_REPORTING_validation_ALL_enriched.csv",
]

def count_unmatched(file_path: Path):
    df = pd.read_csv(file_path, low_memory=False)
    total = len(df)
    if "MODEL_STD" not in df.columns:
        print(f"[WARN] {file_path.name} 中没有 MODEL_STD 列")
        return None
    unmatched = df["MODEL_STD"].isna().sum()
    matched = total - unmatched
    return total, matched, unmatched, unmatched / total * 100

def main():
    print("\n=== 未匹配航班统计（以 MODEL_STD 为空计） ===")
    for fname in FILES:
        f = REF_DIR / fname
        if not f.exists():
            print(f"[WARN] 文件不存在：{fname}")
            continue
        result = count_unmatched(f)
        if result:
            total, matched, unmatched, pct = result
            print(f"{fname}: 总计 {total:,}，匹配 {matched:,}，未匹配 {unmatched:,}（{pct:.2f}%）")

if __name__ == "__main__":
    main()
