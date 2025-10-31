# save as normalize_models.py
import re
import sys
import pandas as pd
from pathlib import Path

IN_FILE  = "tails_with_models_cleaned.csv"
OUT_FILE = "tails_with_standardized_models_v2.csv"

# —— 机型标准化规则 —— #
def normalize_model(model: str, mfr: str = "") -> str:
    """
    把带客户码/后缀的型号（如 737-924、737-924ER、757-251、A320-214）统一成标准型号
    例：
      737-7H4 -> 737-700
      737-924 -> 737-900
      737-924ER -> 737-900ER
      757-251 -> 757-200
      A320-214 -> A320
    """
    if not isinstance(model, str):
        return ""
    s = model.strip().upper().replace("–", "-")  # 归一化破折号
    if s == "":
        return ""

    # —— Boeing 737 NG —— #
    if re.match(r"^737-7[A-Z0-9]{2,3}$", s): return "737-700"
    if re.match(r"^737-8[A-Z0-9]{2,3}$", s): return "737-800"
    if re.match(r"^737-9[A-Z0-9]{2,3}ER$", s): return "737-900ER"
    if re.match(r"^737-9[A-Z0-9]{2,3}$", s): return "737-900"

    # === Boeing 737 MAX 系列 ===
    if re.match(r"^(737-7[A-Z0-9]{0,3}|737\s*MAX\s*7)$", s): return "737 MAX 7"
    if re.match(r"^(737-8[A-Z0-9]{0,3}|737\s*MAX\s*8)$", s): return "737 MAX 8"
    if re.match(r"^(737-9[A-Z0-9]{0,3}|737\s*MAX\s*9)$", s): return "737 MAX 9"
    if re.match(r"^(737-10[A-Z0-9]{0,3}|737\s*MAX\s*10)$", s): return "737 MAX 10"

    # —— Boeing 757 —— #
    if re.match(r"^757-2[A-Z0-9]{2,3}$", s): return "757-200"
    if re.match(r"^757-3[A-Z0-9]{2,3}$", s): return "757-300"

    # —— Boeing 767 —— #
    if re.match(r"^767-2[A-Z0-9]{2,3}$", s): return "767-200"
    if re.match(r"^767-3[A-Z0-9]{2,3}$", s): return "767-300"
    if re.match(r"^767-4[A-Z0-9]{2,3}$", s): return "767-400"

    # —— Boeing 777 —— #
    if re.match(r"^777-2[A-Z0-9]{2,3}(ER|LR)?$", s):
        return "777-200" + ("ER" if s.endswith("ER") else "LR" if s.endswith("LR") else "")
    if re.match(r"^777-3[A-Z0-9]{2,3}(ER)?$", s):
        return "777-300" + ("ER" if s.endswith("ER") else "")

    # —— Boeing 787 —— #
    if re.match(r"^787-8[A-Z0-9]{0,3}$", s):  return "787-8"
    if re.match(r"^787-9[A-Z0-9]{0,3}$", s):  return "787-9"
    if re.match(r"^787-10[A-Z0-9]{0,3}$", s): return "787-10"

    # —— Boeing 747 / 727 / 717 / 707（常见）—— #
    if re.match(r"^747-1[A-Z0-9]{2,3}$", s): return "747-100"
    if re.match(r"^747-2[A-Z0-9]{2,3}$", s): return "747-200"
    if re.match(r"^747-3[A-Z0-9]{2,3}$", s): return "747-300"
    if re.match(r"^747-4[A-Z0-9]{2,3}$", s): return "747-400"
    if re.match(r"^747-8[A-Z0-9]{1,3}$", s): return "747-8"
    if re.match(r"^727-1[A-Z0-9]{2,3}$", s): return "727-100"
    if re.match(r"^727-2[A-Z0-9]{2,3}$", s): return "727-200"
    if re.match(r"^717-2[A-Z0-9]{2,3}$", s): return "717-200"
    if re.match(r"^707-3[A-Z0-9]{2,3}$", s): return "707-320"

    # === Airbus A320 家族 ===
    if re.match(r"^A318[-A-Z0-9]*$", s): return "A318"
    if re.match(r"^A319-1[A-Z0-9]*$", s): return "A319-100"
    if re.match(r"^A320-1[A-Z0-9]*$", s): return "A320-100"
    if re.match(r"^A320-2[A-Z0-9]*$", s): return "A320-200"
    if re.match(r"^A321-1[A-Z0-9]*$", s): return "A321-100"
    if re.match(r"^A321-2[A-Z0-9]*$", s): return "A321-200"

    # === Airbus A330 / A340 ===
    if re.match(r"^A330-2[A-Z0-9]*$", s): return "A330-200"
    if re.match(r"^A330-3[A-Z0-9]*$", s): return "A330-300"
    if re.match(r"^A340-2[A-Z0-9]*$", s): return "A340-200"
    if re.match(r"^A340-3[A-Z0-9]*$", s): return "A340-300"
    if re.match(r"^A340-5[A-Z0-9]*$", s): return "A340-500"
    if re.match(r"^A340-6[A-Z0-9]*$", s): return "A340-600"

    # === Airbus A350 / A380 ===
    if re.match(r"^A350[-A-Z0-9]*$", s): return "A350"
    if re.match(r"^A380[-A-Z0-9]*$", s): return "A380"

    # === Airbus A220 (BD-500) ===
    if re.match(r"^BD-500-1A10$", s): return "A220-100"
    if re.match(r"^BD-500-1A11$", s): return "A220-300"

    # Bombardier CRJ family
    if re.match(r"^CL-600-2B19$", s): return "CRJ200"
    if re.match(r"^CL-600-2C10$", s): return "CRJ700"
    if re.match(r"^CL-600-2C11$", s): return "CRJ550"
    if re.match(r"^CL-600-2D15$", s): return "CRJ705"
    if re.match(r"^CL-600-2D24$", s): return "CRJ900"
    if re.match(r"^CL-600-2E25$", s): return "CRJ1000"

    # === Embraer ERJ / E-Jet 系列 ===
    # ERJ 145 家族
    if re.match(r"^EMB-145", s): return "ERJ145"
    if re.match(r"^EMB-135", s): return "ERJ135"
    if re.match(r"^EMB-140", s): return "ERJ140"

    # E170/E175/E190/E195 家族
    if re.match(r"^ERJ ?170-1\d{2}", s): return "E170"
    if re.match(r"^ERJ ?170-2\d{2}", s): return "E175"
    if re.match(r"^ERJ ?190-1\d{2}", s): return "E190"
    if re.match(r"^ERJ ?190-2\d{2}", s): return "E195"

    # —— 已经是标准样式就保留 —— #
    if re.match(r"^(737|747|757|767|777|787)-\d{2,3}(ER|LR|F)?$", s): return s
    if re.match(r"^A(318|319|320|321|330|340|350|380)$", s): return s

    # 默认返回原值（大写）
    return s

def main(in_file=IN_FILE, out_file=OUT_FILE):
    df = pd.read_csv(in_file, dtype=str)

    cols = {c.lower(): c for c in df.columns}
    tail_col  = "TAIL_NUM"
    mfr_col   = "MFR"
    model_col = "MODEL"
    year_col  = "YEAR MFR"

    # 清洗
    for c in (mfr_col, model_col, year_col, tail_col):
        df[c] = df[c].astype(str).str.strip()

    # 生成 MODEL_STD
    df["MODEL_STD"] = [normalize_model(m, df.at[i, mfr_col]) for i, m in enumerate(df[model_col])]

    # 只保留需要的 5 列并重排：MODEL_STD 非空/发生变化的行排前面，未匹配排后面
    out = df[[tail_col, mfr_col, model_col, "MODEL_STD", year_col]].copy()
    out.rename(columns={
        tail_col: "TAIL_NUM",
        mfr_col: "MFR",
        model_col: "MODEL",
        year_col: "YEAR MFR",
    }, inplace=True)

    has_std = out["MODEL_STD"].fillna("").str.strip() != ""
    changed = (out["MODEL_STD"].str.upper() != out["MODEL"].str.upper()) & has_std
    # 优先级：1) changed=True  2) has_std=True  3) else
    out["_sort_key"] = (~changed).astype(int) * 2 + (~has_std).astype(int)  # 0/1/2
    out = out.sort_values(by=["_sort_key", "MFR", "MODEL_STD", "MODEL", "TAIL_NUM"], ascending=[True, True, True, True, True])
    out = out.drop(columns=["_sort_key"])

    out.to_csv(out_file, index=False)
    p = Path(out_file).resolve()
    print(f"✅ 标准化完成：{p}")
    print(f"行数：{len(out)}")
    # 小预览
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    # 支持命令行参数：python normalize_models.py input.csv output.csv
    if len(sys.argv) >= 2:
        IN_FILE = sys.argv[1]
    if len(sys.argv) >= 3:
        OUT_FILE = sys.argv[2]
    main(IN_FILE, OUT_FILE)
