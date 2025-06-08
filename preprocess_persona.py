import os
import json
import pandas as pd
import re
from collections import defaultdict

def extract_context_from_filename(filename):
    return filename.split(".")[0]

def load_persona_json_to_dataframe(json_path):
    context = extract_context_from_filename(os.path.basename(json_path))
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for persona_id, attributes in data.items():
        record = {
            'persona_num': persona_id,
            'context': context,
            '나이대': None,
            '성별': None,
            '직업군': None,
            '취향': None,
            '취미/관심사': None,
            '환경': None,
            '성격/가치관': None,
            '현황': None
        }
        for item in attributes:
            if "성별" in item:
                record['성별'] = item.split(":", 1)[-1].strip()
            elif "연령대" in item or "나이" in item:
                age_value = item.split(":", 1)[-1].strip()
                if any(age_value.startswith(str(age)) for age in [10, 20, 30, 40, 50, 60]):
                    record['나이대'] = age_value
            elif "직업군" in item:
                record['직업군'] = item.split(":", 1)[-1].strip()
            elif "취향" in item:
                record['취향'] = item.split(":", 1)[-1].strip()
            elif "취미" in item or "관심사" in item:
                record['취미/관심사'] = item.split(":", 1)[-1].strip()
            elif "환경" in item:
                record['환경'] = item.split(":", 1)[-1].strip()
            elif "성격" in item or "가치관" in item:
                record['성격/가치관'] = item.split(":", 1)[-1].strip()
            elif "현황" in item:
                record['현황'] = item.split(":", 1)[-1].strip()
        


        records.append(record)

    return pd.DataFrame(records)

def load_all_personas_and_attach_utterances(folder_path):
    all_records = []
    utterance_map = defaultdict(list)

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        # 1. 페르소나 처리
        if filename.endswith("personas.json"):
            df_persona = load_persona_json_to_dataframe(full_path)
            all_records.append(df_persona)

        # 2. 대화 파일 처리
        elif filename.endswith("dialogues.json"):
            with open(full_path, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
                for d in dialogues:
                    speaker1 = d.get("speaker1")
                    speaker2 = d.get("speaker2")
                    for line in d.get("dialog", []):
                        if line.startswith(f"{speaker1}:"):
                            utterance_map[speaker1].append(line.split(":", 1)[1].strip())
                        elif line.startswith(f"{speaker2}:"):
                            utterance_map[speaker2].append(line.split(":", 1)[1].strip())

    # 3. 전체 persona 통합
    df_total = pd.concat(all_records, ignore_index=True)

    # 4. 발화 매핑
    df_total["발화"] = df_total["persona_num"].map(lambda pid: utterance_map.get(pid, []))

    return df_total

# 실행
folder = "./persona_dataset"
df_persona = load_all_personas_and_attach_utterances(folder)
df_persona = df_persona.dropna(subset=['나이대', '성별'])

# 나이대 데이터 분석
print("\n=== 나이대 데이터 분석 ===")
print("\n1. 전체 나이대 분포:")
print(df_persona['나이대'].value_counts())
print("\n2. 나이대별 데이터 수:")
print(df_persona.groupby('나이대').size())
print("\n3. 나이대가 None인 데이터 수:", df_persona['나이대'].isna().sum())

df_persona.to_csv(os.path.join(folder, "combined_personas_with_utterances.csv"), index=False, encoding="utf-8-sig")
