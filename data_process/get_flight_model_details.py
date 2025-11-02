import pandas as pd

model_features = {
    # —— Boeing 737 NG —— #
    "737-700":   {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 28},
    "737-800":   {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 31},
    "737-900":   {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 34},
    "737-900ER": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 34},

    # === Boeing 737 MAX ===
    "737 MAX 7":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 28},
    "737 MAX 8":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 31},
    "737 MAX 9":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 34},
    "737 MAX 10": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 36},

    # —— Boeing 757 —— #
    "757-200": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 36},
    "757-300": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 44},

    # —— Boeing 767 —— #
    "767-200": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 34},
    "767-300": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 38},
    "767-400": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 42},

    # —— Boeing 777 —— #
    "777-200":   {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 40},
    "777-200ER": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 40},
    "777-200LR": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 40},
    "777-300":   {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 44},
    "777-300ER": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 44},

    # —— Boeing 787 —— #
    "787-8":  {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 35},
    "787-9":  {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 40},
    "787-10": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 44},

    # —— Boeing 747 / 727 / 717 / 707 —— #
    "747-100": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 50},
    "747-200": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 50},
    "747-300": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 52},
    "747-400": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 52},
    "747-8":   {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 55},

    "727-100": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 25},
    "727-200": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 30},
    "717-200": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 26},
    "707-320": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 30},

    # === Airbus A320 family ===
    "A318":     {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 25},
    "A319-100": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 27},
    "A320-100": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 29},
    "A320-200": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 30},
    "A321-100": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 34},
    "A321-200": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 36},

    # === Airbus A330 / A340 ===
    "A330-200": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 38},
    "A330-300": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 42},
    "A340-200": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 42},
    "A340-300": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 44},
    "A340-500": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 48},
    "A340-600": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 52},

    # === Airbus A350 / A380 ===
    "A350": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 42},
    "A380": {"NUM_DOORS_USED": 2, "SEAT_LAYOUT": "Wide", "AVG_ROW_NUM": 50},

    # === Airbus A220 (BD-500) ===
    "A220-100": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 23},
    "A220-300": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 26},

    # === Bombardier CRJ ===
    "CRJ200":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 13},
    "CRJ550":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 14},
    "CRJ705":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 19},
    "CRJ700":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 18},
    "CRJ900":  {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 21},
    "CRJ1000": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 22},

    # === Embraer ERJ / E-Jets ===
    "ERJ135": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 11},
    "ERJ140": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 12},
    "ERJ145": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 13},

    "E170": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 20},
    "E175": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 21},
    "E190": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 25},
    "E195": {"NUM_DOORS_USED": 1, "SEAT_LAYOUT": "Narrow", "AVG_ROW_NUM": 27},
}

def map_model_feature(df, model_features):
    df["NUM_DOORS_USED"] = df["MODEL_STD"].map(lambda m: model_features.get(m, {}).get("NUM_DOORS_USED", None))
    df["SEAT_LAYOUT"] = df["MODEL_STD"].map(lambda m: model_features.get(m, {}).get("SEAT_LAYOUT", None))
    df["AVG_ROW_NUM"] = df["MODEL_STD"].map(lambda m: model_features.get(m, {}).get("AVG_ROW_NUM", None))
    return df

# 应用
datasets = ["train", "test", "validation"]
for name in datasets:
    path = f"../data/ref/flights_{name}_with_airline_concourse_and_gates.csv"  # 上一步已生成的带 CONCOURSE 的文件
    df = pd.read_csv(path)
    df = map_model_feature(df, model_features)

    out_path = f"../data/ref/flights_{name}_add_more_flight_details.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{out_path}")
