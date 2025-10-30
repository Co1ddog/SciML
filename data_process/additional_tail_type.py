import pandas as pd
import glob, os, re

data_path = "./data/raw_monthly/"
files = sorted(glob.glob(os.path.join(data_path, "T_ONTIME_REPORTING_*.csv")))

tails = []
for f in files:
    df = pd.read_csv(f, usecols=["TAIL_NUM"], low_memory=False)
    tails.append(df["TAIL_NUM"])

tails = pd.concat(tails, ignore_index=True)

# 清洗：统一大写、去空白、去异常
def clean_tail(x: str):
    if pd.isna(x): return None
    x = str(x).strip().upper()
    if x in ("", "NA", "NAN", "NULL", "UNKNOWN"): return None
    # BTS 里偶尔会有“Blocked”或“*****”
    if "BLOCK" in x or set(x) == {"*"}: return None
    return x

tails = tails.map(clean_tail).dropna().drop_duplicates().to_frame(name="TAIL_NUM")

print("Unique tails:", len(tails))
tails.head()
