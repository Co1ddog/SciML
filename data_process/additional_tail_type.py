from pathlib import Path
import pandas as pd

ROOT = Path("../data")

# 只用“航班级”表；5min聚合的 merged_*_set 没有 TAIL_NUM
paths = [
    ROOT/"train"/"flights_train_raw.csv",
    ROOT/"validation"/"flights_validation_raw.csv",
    ROOT/"test"/"flights_test_raw.csv",
]

dfs = [pd.read_csv(p, usecols=["TAIL_NUM"]) for p in paths if p.exists()]
tails = pd.concat(dfs, ignore_index=True)

# 清洗：去空、统一大写、去异常标记
def clean_tail(x):
    if pd.isna(x): return None
    x = str(x).strip().upper()
    if x in {"", "NA", "NAN", "NULL", "UNKNOWN"}: return None
    if "BLOCK" in x or set(x) == {"*"}: return None
    return x

tails = tails["TAIL_NUM"].map(clean_tail).dropna().drop_duplicates().to_frame(name="TAIL_NUM")

# 输出到 ref 目录，作为 A 表模板起点
ref_dir = ROOT.parent / "ref"
ref_dir.mkdir(parents=True, exist_ok=True)
(tails.assign(ICAO="", Seats="", SOURCE=""))
tails.to_csv(ref_dir/"tail_registry_template.csv", index=False)

print(f"Unique tails: {len(tails)} -> {ref_dir/'tail_registry_template.csv'}")
