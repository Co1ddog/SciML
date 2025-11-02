# concat_ontime_splits.py
from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parent          # .../data_process
DATA_DIR = BASE.parent / "data"                 # .../data
REF_DIR = DATA_DIR / "ref"                      # .../data/ref
SPLITS = ["train", "test", "validation"]        # 子目录：.../data/<split>/
PATTERN = re.compile(r"^T_ON.*\.csv$", re.IGNORECASE)  # 匹配 T_ONTIME_REPORTING_*.csv 等

def list_monthlies(split_dir: Path):
    return sorted([p for p in split_dir.glob("*.csv") if PATTERN.match(p.name)])

def union_concat(paths):
    if not paths: 
        return pd.DataFrame()
    # 列并集对齐
    all_cols = []
    seen = set()
    for p in paths:
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception as e:
            print(f"[WARN] 跳过（读表头失败）{p.name}: {e}")
            continue
        for c in cols:
            if c not in seen:
                seen.add(c); all_cols.append(c)

    parts = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"[WARN] 跳过（读数据失败）{p.name}: {e}")
            continue
        for c in all_cols:
            if c not in df.columns:
                df[c] = pd.NA
        parts.append(df[all_cols])
        print(f"[OK] {p.name}: {len(df)} rows")
    if not parts: 
        return pd.DataFrame(columns=all_cols)
    return pd.concat(parts, ignore_index=True)

def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"[WARN] 目录不存在，跳过：{split_dir}")
            continue
        csvs = list_monthlies(split_dir)
        if not csvs:
            print(f"[WARN] {split} 下无匹配 T_ON* 文件，跳过。")
            continue
        print(f"\n=== 合并 {split}（{len(csvs)} 个文件） ===")
        df = union_concat(csvs)
        out = REF_DIR / f"T_ONTIME_REPORTING_{split}_ALL.csv"
        df.to_csv(out, index=False)
        print(f"[DONE] {split}: {len(df)} rows -> {out}")

if __name__ == "__main__":
    main()
