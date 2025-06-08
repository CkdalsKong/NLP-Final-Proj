import os
import json
import pandas as pd

def parse_filename_metadata(filename):
    # 예: "가족_10_남성.json"
    base = os.path.basename(filename).replace(".json", "")
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]  # context, 연령대, 성별
    return "Unknown", "Unknown", "Unknown"

from collections import defaultdict

def load_slang_data_to_dataframe(folder_path):
    record_map = defaultdict(list)

    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue

        context, age_group, gender = parse_filename_metadata(fname)
        file_path = os.path.join(folder_path, fname)

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for entry in data:
                    slang_example = entry.get("slang_sentence", "")
                    for pair in entry.get("slang_meaning_pairs", []):
                        key = (
                            context,
                            age_group,
                            gender,
                            pair.get("slang"),
                            pair.get("meaning")
                        )
                        record_map[key].append(slang_example)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

    # 변환된 딕셔너리 → DataFrame
    rows = []
    for key, examples in record_map.items():
        rows.append({
            "context": key[0],
            "연령대": key[1],
            "성별": key[2],
            "slang": key[3],
            "slang mean": key[4],
            "slang example": examples  # 또는 "; ".join(examples)
        })

    return pd.DataFrame(rows)

slang_folder = "./preprocess_slang/"  # 경로만 실제로 교체
df_slang = load_slang_data_to_dataframe(slang_folder)
df_slang.to_csv(os.path.join(slang_folder, "combined_slang_dataset.csv"), index=False, encoding="utf-8-sig")