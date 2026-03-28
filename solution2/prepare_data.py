import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw data to single all-in-one table."
    )

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise ValueError("input_dir not found")
    
    return args


def convert_train_labels(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["target"] = data["target"].astype(np.int8)
    return data


def convert_events(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # event_dttm
    data["event_dttm"] = pd.to_datetime(data["event_dttm"], format="%Y-%m-%d %H:%M:%S")
    
    # event_type_nm
    assert(data["event_type_nm"].min() >= 0)
    assert(data["event_type_nm"].max() < 256)
    data["event_type_nm"] = data["event_type_nm"].astype(np.uint8)
    
    # event_desc
    assert(data["event_desc"].min() >= 0)
    assert(data["event_desc"].max() < 256)
    data["event_desc"] = data["event_desc"].astype(np.uint8)

    # channel_indicator_type
    assert(data["channel_indicator_type"].min() >= 0)
    assert(data["channel_indicator_type"].max() < 256)
    data["channel_indicator_type"] = data["channel_indicator_type"].astype(np.uint8)
    
    # channel_indicator_sub_type
    assert(data["channel_indicator_sub_type"].min() >= 0)
    assert(data["channel_indicator_sub_type"].max() < 256)
    data["channel_indicator_sub_type"] = data["channel_indicator_sub_type"].astype(np.uint8)

    # currency_iso_cd
    assert(data["currency_iso_cd"].min() >= 0)
    assert(data["currency_iso_cd"].max() < 256)
    assert((data["currency_iso_cd"] - data["currency_iso_cd"] // 1).sum() == 0)
    data["currency_iso_cd"] = (data["currency_iso_cd"].fillna(-1.0) + 1).astype(np.uint8)
    
    # mcc_code
    mcc_code = data["mcc_code"].fillna("-1").astype(int) + 1
    assert(mcc_code.min() >= 0)
    assert(mcc_code.max() < 256)
    data["mcc_code"] = mcc_code.astype(np.uint8)
    
    # pos_cd
    assert(data["pos_cd"].min() >= 0)
    assert(data["pos_cd"].max() < 256)
    assert((data["pos_cd"] - data["pos_cd"] // 1).sum() == 0)
    data["pos_cd"] = (data["pos_cd"].fillna(-1.0) + 1).astype(np.uint8)
    
    # browser_language
    def browser_language_to_int(s: str | None) -> int:
        if s is None:
            return 0
        if s == "not available":
            return 1
        return 2

    data["browser_language"] = data["browser_language"].apply(browser_language_to_int).astype(np.uint8)

    # timezone
    assert(data["timezone"].min() >= 0)
    assert(data["timezone"].max() < 60000)
    assert((data["timezone"] - data["timezone"] // 1).sum() == 0)
    data["timezone"] = (data["timezone"].fillna(-1.0) + 1).astype(np.uint16)
    
    # operating_system_type
    assert(data["operating_system_type"].fillna(0.0).min() >= 0)
    assert(data["operating_system_type"].fillna(0.0).max() < 256)
    assert((data["operating_system_type"] - data["operating_system_type"] // 1).sum() == 0)
    data["operating_system_type"] = (data["operating_system_type"].fillna(-1.0) + 1).astype(np.uint8)
    
    # battery
    def battery_to_float(s: str | None) -> float:
        if s is None:
            return np.nan
        if s == "NaN%":
            return -2.0
        if s == "not available":
            return -1.0
        if s.startswith(":"):
            return float(s[1:].partition("%")[0])
        assert s.endswith("%")
        assert s[0].isdigit()
        return float(s.removesuffix("%"))

    data["battery"] = data["battery"].apply(battery_to_float).astype(np.float32)

    # device_system_version
    def version_to_int(s: str | None) -> int:
        if s is None:
            return 0
        x = s.split(".")
        assert len(x) <= 3
        a = int(x[0])
        b = int(x[1]) if len(x) > 1 else 0
        c = int(x[2]) if len(x) > 2 else 0
        assert a > 0 and a < 100
        assert b >= 0 and b < 100
        assert c >= 0 and c < 100
        return a * 10000 + b * 100 + c

    device_system_version_parts = data["device_system_version"].apply(lambda x: 0 if x is None else x.count(".") + 1).astype(np.uint8)
    device_system_version = data["device_system_version"].apply(version_to_int).astype(np.uint32)
    
    data.drop(columns=["device_system_version"], inplace=True)
    data["device_system_version"] = device_system_version
    data["device_system_version_parts"] = device_system_version_parts

    # screen_size
    s1 = data["screen_size"].apply(lambda x: int(x.split("x")[0]) if x is not None else 0)
    assert(s1.min() >= 0)
    assert(s1.max() < 10000)
    data["screen_size_1"] = s1.astype(np.uint16)
    
    s2 = data["screen_size"].apply(lambda x: int(x.split("x")[1]) if x is not None else 0)
    assert(s2.min() >= 0)
    assert(s2.max() < 10000)
    data["screen_size_2"] = s2.astype(np.uint16)

    data.drop(columns=["screen_size"], inplace=True)

    # developer_tools
    developer_tools = data["developer_tools"].fillna("-1").astype(int) + 1
    assert(developer_tools.min() >= 0)
    assert(developer_tools.max() < 256)
    data["developer_tools"] = developer_tools.astype(np.uint8)
    
    # phone_voip_call_state
    assert(data["phone_voip_call_state"].fillna(0.0).min() >= 0)
    assert(data["phone_voip_call_state"].fillna(0.0).max() < 256)
    assert((data["phone_voip_call_state"] - data["phone_voip_call_state"] // 1).sum() == 0)
    data["phone_voip_call_state"] = (data["phone_voip_call_state"].fillna(-1.0) + 1).astype(np.uint8)
    
    # web_rdp_connection
    assert(data["web_rdp_connection"].fillna(0.0).min() >= 0)
    assert(data["web_rdp_connection"].fillna(0.0).max() < 256)
    assert((data["web_rdp_connection"] - data["web_rdp_connection"] // 1).sum() == 0)
    data["web_rdp_connection"] = (data["web_rdp_connection"].fillna(-1.0) + 1).astype(np.uint8)
    
    # compromised
    compromised = data["compromised"].fillna("-1").astype(int) + 1
    assert(compromised.min() >= 0)
    assert(compromised.max() < 256)
    data["compromised"] = compromised.astype(np.uint8)

    return data


def prepare_data(input_dir: Path, output_path: Path):
    print("Load data...")

    pretrain_data = pd.concat([
        pd.read_parquet(input_dir / "pretrain_part_1.parquet"),
        pd.read_parquet(input_dir / "pretrain_part_2.parquet"),
        pd.read_parquet(input_dir / "pretrain_part_3.parquet"),
    ], ignore_index=True)

    train_data = pd.concat([
        pd.read_parquet(input_dir / "train_part_1.parquet"),
        pd.read_parquet(input_dir / "train_part_2.parquet"),
        pd.read_parquet(input_dir / "train_part_3.parquet"),
    ], ignore_index=True)

    pretest_data = pd.read_parquet(input_dir / "pretest.parquet")
    test_data = pd.read_parquet(input_dir / "test.parquet")
    train_labels_data = pd.read_parquet(input_dir / "train_labels.parquet")

    print("Convert data...")

    pretest_data = convert_events(pretest_data)
    test_data = convert_events(test_data)
    pretrain_data = convert_events(pretrain_data)
    train_data = convert_events(train_data)

    train_labels_data =convert_train_labels(train_labels_data)

    print("Combine data...")

    pretrain_data["original_index"] = pretrain_data.index.astype(np.uint32)
    train_data["original_index"] = train_data.index.astype(np.uint32)
    pretest_data["original_index"] = pretest_data.index.astype(np.uint32)
    test_data["original_index"] = test_data.index.astype(np.uint32)

    pretrain_data["dataset"] = 0
    train_data["dataset"] = 1
    pretest_data["dataset"] = 2
    test_data["dataset"] = 3

    pretrain_data["target"] = 0
    train_data["target"] = 1
    pretest_data["target"] = 0
    test_data["target"] = 0

    events0 = set(train_labels_data[train_labels_data["target"] == 0]["event_id"])
    events1 = set(train_labels_data[train_labels_data["target"] == 1]["event_id"])

    train_data.loc[train_data["event_id"].isin(events0), "target"] = 2
    train_data.loc[train_data["event_id"].isin(events1), "target"] = 3

    all_data = pd.concat([pretrain_data, train_data, pretest_data, test_data], ignore_index=True, axis=0)
    all_data["dataset"] = all_data["dataset"].astype(np.uint8)
    all_data["target"] = all_data["target"].astype(np.uint8)

    all_data = all_data.sort_values(["customer_id", "event_dttm", "dataset"], kind="stable", ignore_index=True)

    print("Drop duplicates...")

    all_data["duplicated_event"] = all_data["event_id"].duplicated(keep="first")
    all_data = all_data.drop_duplicates("event_id", keep="last", ignore_index=True)

    print("Save data...")
    all_data.to_parquet(output_path, compression="zstd")


def main():
    args = parse_args()
    prepare_data(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()
