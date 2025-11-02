import pandas as pd

files = ['train', 'test', 'validation']
all_airlines = set()

# 读取CSV文件
for file in files:
    df = pd.read_csv(f"../data/ref/flights_{file}_raw_with_model_and_seats.csv")
    output_path = f"../data/ref/flights_{file}_with_airline_concourse.csv"

    # 获取某一列的唯一值
    unique_values = df["OP_UNIQUE_CARRIER"].unique()

    bwi_airline_concourse_map = {
        "WN": {"airline": "Southwest Airlines", "concourse": "A, B, C, E"},
        "NK": {"airline": "Spirit Airlines", "concourse": "D"},
        "SY": {"airline": "Sun Country Airlines", "concourse": "D"},
        "UA": {"airline": "United Airlines", "concourse": "D"},
        "AS": {"airline": "Alaska Airlines", "concourse": "D"},
        "AA": {"airline": "American Airlines", "concourse": "D"},
        "DL": {"airline": "Delta Air Lines", "concourse": "D"},
        "F9": {"airline": "Frontier Airlines", "concourse": "D"},

        # 以下为BWI数据集中常见但图中未列出的，其航站楼根据FAA/BWI资料推测：
        "OH": {"airline": "PSA Airlines (American Eagle)", "concourse": "D"},
        "YX": {"airline": "Republic Airways", "concourse": "D"},
        "OO": {"airline": "SkyWest Airlines", "concourse": "D"},
        "MQ": {"airline": "Envoy Air (American Eagle)", "concourse": "D"},
        "9E": {"airline": "Endeavor Air (Delta Connection)", "concourse": "D"},
        "G4": {"airline": "Allegiant Air", "concourse": "E"},
        "YV": {"airline": "Mesa Airlines", "concourse": "D"},
        "B6": {"airline": "JetBlue Airways", "concourse": "E"},
    }

    df["AIRLINE_NAME"] = df["OP_UNIQUE_CARRIER"].map(
        {k: v["airline"] for k, v in bwi_airline_concourse_map.items()}
    )

    df["CONCOURSE"] = df["OP_UNIQUE_CARRIER"].map(
        {k: v["concourse"] for k, v in bwi_airline_concourse_map.items()}
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已处理并保存: {output_path}")