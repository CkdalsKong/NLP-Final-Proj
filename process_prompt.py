import pandas as pd
import random
import ast

# CSV 파일 불러오기
df_persona = pd.read_csv('persona_dataset/combined_personas_with_utterances.csv')
df_slang = pd.read_csv('slang_dataset/slang_sentence/combined_slang_dataset.csv')

# 연령대 컬럼을 문자열로 변환
df_slang["연령대"] = df_slang["연령대"].astype(str)

# === 데이터 전처리 ===
df_persona["발화"] = df_persona["발화"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_slang["slang example"] = df_slang["slang example"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_slang.columns = df_slang.columns.str.strip()

# context -> slang category 매핑
persona_to_slang = {
    'TL_가족': ['가족', '일상', '주거'],
    'TL_식음료': ['가족', '일상', '음식'],
    'TL_자연,환경': ['가족', '일상', '주거'],
    'TL_주거,생활': ['가족', '일상', '주거'],
    'TL_건강,의학': ['일상', '교육', '음식'],
    'TL_경제,금융,산업': ['일상', '경제', '직장'],
    'TL_과학,기술,IT': ['일상', '직장', '교육'],
    'TL_미용,외모': ['일상', 'SNS', '연애결혼'],
    'TL_반려동물': ['일상', '주거'],
    'TL_쇼핑,상품': ['일상', 'SNS', '연애결혼'],
    'TL_스포츠': ['일상', '여가', '군대'],
    'TL_아티스트,공연': ['일상', 'SNS', '여가'],
    'TL_연애,결혼': ['일상', 'SNS', '연애결혼'],
    'TL_예술,문학': ['일상', '교육', 'SNS'],
    'TL_학교,학업': ['일상', '직장', '교육'],
    'TL_시사,사회,인문': ['경제', '교육', '군대'],
    'TL_일,직장': ['경제', '직장', '교육'],
    'TL_미디어,콘텐츠': ['SNS', '게임', '여가'],
    'TL_여가,오락': ['SNS', '게임', '여가'],
    'TL_여행,레저': ['여가', '연애결혼', '음식']
}

# === 유틸 함수들 ===

def format_persona(p):
    base = f"{p['나이대']}, {p['성별']}"
    traits = [p[k] for k in ['직업군', '취향', '취미/관심사', '환경', '성격/가치관', '현황'] if pd.notna(p[k])]
    return base + (", " + " ".join(traits) if traits else "")

def generate_slang_block(persona_row, context_key):
    slang_contexts = persona_to_slang.get(context_key, [])
    filtered = df_slang[
        df_slang["context"].isin(slang_contexts) &
        df_slang["slang mean"].notna() &
        df_slang["slang example"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ]

    if isinstance(persona_row["나이대"], str):
        filtered = filtered[filtered["연령대"].str.contains(persona_row["나이대"][:2])]
    if isinstance(persona_row["성별"], str):
        filtered = filtered[filtered["성별"] == persona_row["성별"]]    

    if len(filtered) >= 5:
        sample = filtered.sample(5)
    else:
        needed = 5 - len(filtered)
        others = df_slang[
            df_slang["slang mean"].notna() &
            df_slang["slang example"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ].sample(needed)
        sample = pd.concat([filtered, others])

    slang_block = "###Slang###\n"
    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        examples = row["slang example"]
        example = random.choice(examples) if isinstance(examples, list) and examples else "예시 없음"
        slang_block += f"Slang{idx}: {row['slang']}, 뜻: {row['slang mean']}, 예시: {example}\n"
    return slang_block

# === 페르소나 샘플링 ===

for _ in range(10):  # 최대 10번 시도
    p1 = df_persona.sample(1).iloc[0]
    context = p1["context"]
    candidates = df_persona[(df_persona["context"] == context) & (df_persona["persona_num"] != p1["persona_num"])]
    if not candidates.empty:
        p2 = candidates.sample(1).iloc[0]
        break

# 발화 및 예시 추출
utterance1 = random.choice(p2["발화"]) if p2["발화"] else "..."
utterance2 = random.choice(p1["발화"]) if p1["발화"] else "..."
persona_ex1 = random.choice(p1["발화"]) if p1["발화"] else "..."
persona_ex2 = random.choice(p2["발화"]) if p2["발화"] else "..."

# 슬랭 블록
slang_block1 = generate_slang_block(p1, context)
slang_block2 = generate_slang_block(p2, context)

# === 템플릿 ===

template = """
### Persona ###
Persona: {persona}

### Task ###
- You are currently having a conversation with a person who has the traits: "{partner_traits}".
- The topic of the conversation is "{context}".
- Based on the current conversation ([Input]), write your next response.
- If you don't understand the meaning of a [slang], refer to [slang mean].
- If the usage context of the [slang] is unclear, refer to [slang example].
- You usually speak in the style shown in [persona example].
- The conversation should be in **Korean**.

{slang_block}

### Persona Example ###
{persona_example}

### Output ###
[Note: If your response includes slang, you must use **at least one** slang expression from the [slang] list.]
- Response with slang:  
- Response without slang:  
"""

# === 프롬프트 생성 ===

prompt1 = template.format(
    persona=format_persona(p1),
    partner_traits=f"{p2['나이대']}, {p2['성별']}",
    context=context[3:],  # TL_ 제거
    slang_block=slang_block1,
    persona_example=persona_ex1,
    utterance=utterance1
)

prompt2 = template.format(
    persona=format_persona(p2),
    partner_traits=f"{p1['나이대']}, {p1['성별']}",
    context=context[3:],
    slang_block=slang_block2,
    persona_example=persona_ex2,
    utterance=utterance2
)

# === 출력 ===
print("Prompt 1:\n", prompt1)
print("\nPrompt 2:\n", prompt2)

# prompt를 txt 파일로 저장
with open('prompt1.txt', 'w', encoding='utf-8') as f:
    f.write(prompt1)

with open('prompt2.txt', 'w', encoding='utf-8') as f:
    f.write(prompt2)
