from pathlib import Path
import pandas as pd

# ---------- 路径 ----------
BASE      = Path(__file__).resolve().parent          # .../data_process
DATA_DIR  = BASE.parent / "data"                     # .../data
REF_DIR   = DATA_DIR / "ref"                         # .../data/ref

# 三份“月份合并后的”航班汇总
ALL_FILES = {
    "train": REF_DIR / "T_ONTIME_REPORTING_train_ALL.csv",
    "test": REF_DIR / "T_ONTIME_REPORTING_test_ALL.csv",
    "validation": REF_DIR / "T_ONTIME_REPORTING_validation_ALL.csv",
}

# 三份“飞机参数”明细（放在 data/ref 下；如果你放在别处，把下面路径改一下）
DETAIL_FILES = [
    REF_DIR / "flights_train_add_more_flight_details.csv",
    REF_DIR / "flights_test_add_more_flight_details.csv",
    REF_DIR / "flights_validation_add_more_flight_details.csv",
]

# 需要映射的字段
MAP_COLS = [
    "MODEL_STD","MAX_SEATS","EXPECTED_PASSENGERS","AIRLINE_NAME","CONCOURSE",
    "GATE_COUNT","A_GATES","B_GATES","C_GATES","D_GATES","E_GATES",
    "NUM_DOORS_USED","SEAT_LAYOUT","AVG_ROW_NUM"
]

def normalize_tail(s):
    """仅去掉前导 N（不改其它字符），用于两边对齐比较"""
    if pd.isna(s):
        return None
    t = str(s).strip()
    # 去掉最左侧的 'N' 或 'n'
    if t[:1].upper() == "N":
        t = t[1:]
    return t

def load_and_build_master():
    parts = []
    for p in DETAIL_FILES:
        if not p.exists():
            print(f"[WARN] 未找到明细：{p}")
            continue
        df = pd.read_csv(p, low_memory=False)
        if "TAIL_NUM" not in df.columns:
            print(f"[WARN] 明细缺少 TAIL_NUM，跳过：{p.name}")
            continue
        # 只保留主键和需要的列（存在的才保留）
        keep = ["TAIL_NUM"] + [c for c in MAP_COLS if c in df.columns]
        df = df[keep].copy()
        df["TAIL_KEY"] = df["TAIL_NUM"].apply(normalize_tail)
        parts.append(df)

    if not parts:
        raise FileNotFoundError("没有可用的飞机参数明细文件。")

    big = pd.concat(parts, ignore_index=True)

    # 同一 TAIL_KEY 可能多条：按列取“首个非空”合并
    def first_valid(series):
        for v in series:
            if pd.notna(v):
                return v
        return pd.NA

    agg_dict = {c: first_valid for c in set(big.columns) - {"TAIL_NUM", "TAIL_KEY"}}
    master = big.sort_values(by=["TAIL_KEY"]).groupby("TAIL_KEY", as_index=False).agg(agg_dict)

    # 为了可追溯，保留一个代表性的原始 TAIL_NUM（可选）
    rep_tail = big.groupby("TAIL_KEY", as_index=False)["TAIL_NUM"].first()
    master = rep_tail.merge(master, on="TAIL_KEY", how="right")

    # 规范列顺序
    ordered = ["TAIL_KEY","TAIL_NUM"] + [c for c in MAP_COLS if c in master.columns]
    master = master[ordered]
    print(f"[INFO] 参数主表行数：{len(master)}（唯一 TAIL）")
    return master

def enrich_all(master):
    for split, p in ALL_FILES.items():
        if not p.exists():
            print(f"[WARN] 未找到汇总文件，跳过：{p}")
            continue
        df = pd.read_csv(p, low_memory=False)
        if "TAIL_NUM" not in df.columns:
            print(f"[WARN] {p.name} 缺少 TAIL_NUM，跳过。")
            continue

        df["TAIL_KEY"] = df["TAIL_NUM"].apply(normalize_tail)

        # 仅并入存在于 master 的字段，避免 KeyError
        join_cols = [c for c in MAP_COLS if c in master.columns]
        add_cols  = ["TAIL_KEY"] + join_cols

        out = df.merge(master[add_cols], on="TAIL_KEY", how="left", suffixes=("", "_FROM_MASTER"))

        # 清理临时键
        out.drop(columns=["TAIL_KEY"], inplace=True)

        # 输出文件名（不覆盖原文件）
        out_path = p.with_name(p.stem + "_enriched.csv")
        out.to_csv(out_path, index=False)
        print(f"[DONE] {split}: {len(out)} rows -> {out_path}")

def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    master = load_and_build_master()
    # 可选：把汇总后的“飞机参数主表”也存一份
    master.to_csv(REF_DIR / "aircraft_params_master_by_tail.csv", index=False)
    print(f"[INFO] 已写出参数主表：{(REF_DIR / 'aircraft_params_master_by_tail.csv')}")
    enrich_all(master)

if __name__ == "__main__":
    main()
