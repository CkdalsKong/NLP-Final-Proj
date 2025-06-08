import pandas as pd
import random
import json

def generate_persona_prompt_ko(conversation: str, personas: list, persona_num: int) -> str:
    assert len(personas) == persona_num, "정확히 5개의 페르소나가 필요합니다."
    options = ['A', 'B', 'C', 'D', 'E']
    persona_lines = [f"{opt}. {desc}" for opt, desc in zip(options, personas)]
    prompt = f"""
### 지시:
아래의 후보 페르소나 중 가장 적합한 페르소나를 선택하세요. 보기 중 해당하는 글자 하나만 출력하세요 (예: "A").
### 후보 페르소나 (가장 가능성 높은 하나를 선택하세요):
{"\n".join(persona_lines)}
        """
    return prompt

def generate_metric2_prompt_ko(context: str) -> str:
    return f"""

### 지시:    
아래는 두 화자의 대화입니다. 대화의 흐름을 고려하여, 마지막 화자의 답변(발화)를 작성하세요.

### 대화:
{context}

이에 대한 다음 화자의 답변을 생성하세요.
### 답변: """

def add_speaker_labels(utterances, speaker1="persona1", speaker2="persona2"):
    labeled = []
    for i, utt in enumerate(utterances):
        speaker = speaker1 if i % 2 == 0 else speaker2
        labeled.append(f"{speaker}: {utt}")
    return labeled

# CSV 파일 로드
df = pd.read_csv("combined_personas_with_utterances.csv")  # ← 여기에 파일 경로 입력

# 필요한 필드 정리
def extract_persona_text(row):
    attrs = [row['나이대'], row['성별'], row['직업군'], row['취향'], row['취미/관심사'], row['환경'], row['성격/가치관'], row['현황']]
    return " / ".join([str(x) for x in attrs if pd.notna(x) and x != ""])

df['persona_text'] = df.apply(extract_persona_text, axis=1)

def generate_metric1_prompts_from_df(df: pd.DataFrame, output_file: str = "metric1_prompts.jsonl"):
    prompts = []
    for idx, row in df.iterrows():
        try:
            gt_persona = row['persona_text']
            persona_num = row['persona_num']
            if not gt_persona or pd.isna(row['발화']):
                continue

            # 다른 샘플 페르소나 4개 무작위 추출
            others = df[df.index != idx]['persona_text'].dropna().tolist()
            sampled = random.sample(others, 4)
            all_personas = sampled + [gt_persona]
            random.shuffle(all_personas)

            # 정답 위치 저장 (선택지에서 어디 있는지)
            answer_index = all_personas.index(gt_persona)
            answer_letter = ['A','B','C','D','E'][answer_index]

            # 대화 내용 구성 (리스트 중 일부 샘플링 또는 전체 join)
            conversation_text = "\n".join(eval(row['발화'])[:5])  # 앞 5개만 사용

            # 프롬프트 생성
            prompt_text = generate_persona_prompt_ko(conversation_text, all_personas, 5)

            # 저장 형태 (JSONL-friendly)
            prompts.append({
                "persona_num": persona_num,
                "prompt": prompt_text,
                "answer": answer_letter,
                "ground_truth_persona": gt_persona
            })

        except Exception as e:
            print(f"Error at row {idx}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{len(prompts)}개의 metric1 프롬프트를 생성하였습니다.")

def generate_metric2_prompts_from_df(df: pd.DataFrame, output_file: str = "metric2_prompts.jsonl", context_turns: int = 4):
    prompts = []
    for idx, row in df.iterrows():
        try:
            utterances = eval(row['발화'])
            if len(utterances) <= context_turns:
                continue  # 대화 길이가 너무 짧으면 스킵

            labeled_utterances = add_speaker_labels(utterances[:context_turns])
            context = "\n".join(labeled_utterances)
            answer = utterances[context_turns]

            prompt_text = generate_metric2_prompt_ko(context)

            prompts.append({
                "persona_num": row.get('persona_num', None),
                "prompt": prompt_text,
                "answer": answer,
                "ground_truth_persona": row.get('persona_text', None)
            })

        except Exception as e:
            print(f"Error at row {idx}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{len(prompts)}개의 metric2 프롬프트를 생성하였습니다.")

def generate_combined_metric_prompts(df: pd.DataFrame, output_file: str = "combined_metric_prompts.jsonl", context_turns: int = 4):
    prompts = []
    for idx, row in df.iterrows():
        try:
            gt_persona = row['persona_text']
            persona_num = row['persona_num']
            if not gt_persona or pd.isna(row['발화']):
                continue

            # metric1 프롬프트
            others = df[df.index != idx]['persona_text'].dropna().tolist()
            sampled = random.sample(others, 4) if len(others) >= 4 else others
            all_personas = sampled + [gt_persona]
            if len(all_personas) < 5:
                continue
            random.shuffle(all_personas)
            answer_index = all_personas.index(gt_persona)
            answer_letter = ['A','B','C','D','E'][answer_index]
            conversation_text = "\n".join(eval(row['발화'])[:5])
            prompt1 = generate_persona_prompt_ko(conversation_text, all_personas, 5)

            # metric2 프롬프트
            utterances = eval(row['발화'])
            if len(utterances) <= context_turns:
                continue
            labeled_utterances = add_speaker_labels(utterances[:context_turns])
            context = "\n".join(labeled_utterances)
            answer2 = utterances[context_turns]
            prompt2 = generate_metric2_prompt_ko(context)

            prompts.append({
                "persona_num": persona_num,
                "metric1_prompt": prompt1,
                "metric1_GT": answer_letter,
                "metric2_prompt": prompt2,
                "metric2_GS": answer2,
                "ground_truth_persona": gt_persona
            })
        except Exception as e:
            print(f"Error at row {idx}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(prompts)}개의 combined 프롬프트를 생성하였습니다.")

generate_combined_metric_prompts(df)