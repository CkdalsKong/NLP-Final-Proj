import openai
import time
import requests
import pandas as pd
import random
import ast
import uuid
import os
import json
from pathlib import Path

# 저장용 디렉토리 생성
session_id = str(uuid.uuid4())[:8]  # 간단한 세션 ID
base_dir = Path("/data/nlp/output_final")
base_dir.mkdir(parents=True, exist_ok=True)  # /data/nlp/output 디렉토리 생성
output_dir = base_dir / session_id
output_dir.mkdir(parents=True, exist_ok=True)

# 생성할 대화 개수 설정
NUM_DIALOGUES = 1  # 원하는 대화 개수로 설정

client = openai.OpenAI(api_key="")

# CSV 파일 불러오기
df_persona = pd.read_csv('persona_dataset/combined_personas_with_utterances.csv')
df_slang = pd.read_csv('preprocess_slang/combined_slang_dataset.csv')

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

start_input = {
    'TL_가족': "요즘 가족들이랑 시간 보낼 기회가 잘 없네요. 주말에 뭐 하면 좋을까요?",
    'TL_식음료': "최근에 먹은 음식 중에 진짜 맛있었던 거 있어요? 추천 해주세요!",
    'TL_자연,환경': "요즘 날씨 너무 좋던데, 혹시 자연 즐길 수 있는 좋은 장소 아세요?",
    'TL_주거,생활': "집안일 중에 제일 귀찮은 건 어떤건가요?",
    'TL_건강,의학': "건강 챙기려고 요즘 뭐 하고 계세요? 혹시 추천하는 건강 식단 있나요?",
    'TL_경제,금융,산업': "최근에 물가 오른 거 체감하시나요? 절약 방법 같은 거 있으면 공유해요!",
    'TL_과학,기술,IT': "최근에 나온 기술 중에 흥미로운 거 있었나요?",
    'TL_미용,외모': "피부 관리나 헤어스타일 요즘 어떻게 관리하세요?",
    'TL_반려동물': "반려동물 키우시는 분들, 요즘 어떤 일상 보내고 계세요?",
    'TL_쇼핑,상품': "최근에 산 것 중에서 진짜 만족스러웠던 상품 있어요?",
    'TL_스포츠': "혹시 요즘 주목할 운동 있나요?",
    'TL_아티스트,공연': "최근에 추천해주실 공연이나 콘서트 있나요?",
    'TL_연애,결혼': "요즘 연애는 어떠신가요? 결혼 계획은 있으세요?",
    'TL_예술,문학': "요즘 최근 즐긴 문화 생활 중 인상 깊었던 거 있나요?",
    'TL_학교,학업': "요즘 공부는 어떻게들 하고 계세요?",
    'TL_시사,사회,인문': "최근 뉴스 중에 인상 깊었던 이야기 있으세요?",
    'TL_일,직장': "요즘 회사 분위기 어떤가요?",
    'TL_미디어,콘텐츠': "요즘 빠져있는 드라마나 유튜브 콘텐츠 있어요?",
    'TL_여가,오락': "주말에 시간 날 때 뭐 하면서 쉬세요? 게임, 영화, 뭐든 좋아요!",
    'TL_여행,레저': "올여름 여행 계획 있으세요? 국내든 해외든 요즘 어디가 좋은가요?"
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
    slang_list = []

    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        examples = row["slang example"]
        example = random.choice(examples) if isinstance(examples, list) and examples else "예시 없음"
        slang_block += f"Slang{idx}: {row['slang']}, 뜻: {row['slang mean']}, 예시: {example}\n"
        slang_list.append({
            "slang": row["slang"],
            "mean": row["slang mean"],
            "example": example
        })
    return slang_block, slang_list

def extract_persona_dict(p, speaker_name):
    info = {
        "speaker": speaker_name,
        "나이대": p.get("나이대", ""),
        "성별": p.get("성별", "")
    }
    # 선택적으로 존재하는 항목들
    optional_fields = ["직업군", "취향", "취미/관심사", "환경", "성격/가치관", "현황"]
    for field in optional_fields:
        if pd.notna(p.get(field, None)) and str(p.get(field)).strip() != "":
            info[field] = p[field]
    return info

# === 템플릿 ===

template = """
###Persona###
Persona: {persona}

###Task###
- 현재 당신은 "{partner_traits}"의 사람과 대화 중 입니다.
- 대화의 주제는 "{context}"입니다.
- 현재 대화([Input])에 대해서 다음에 당신의 답변을 작성하세요.
- [slang]의 뜻을 잘 모르겠다면 [slang mean]을 참고하세요.
- [slang]의 사용 맥락이 어렵다면 [slang example]을 참고하세요.
- 당신은 주로 [persona example]처럼 이야기 합니다.
- 대화는 한국어로 합니다.

{slang_block}

###Persona Exapmle###
{persona_example}

###Output###
[Note: 이때, slang이 포함된 답변과 slang을 사용하지 않는 답변은 반드시 동일한 내용을 담아야 합니다. 단지 표현 방식만 다를 뿐, 전달하고자 하는 메시지는 같아야 합니다. slang이 포함된 답변은 [slang]에 포함된 최소 1개 이상의 slang 단어를 사용하세요. slang의 번호를 완성된 문장에 작성하지 마세요. []기호를 사용하지 마세요.]
- Slang 포함 답변: 
- Slang 미포함 답변: 
"""

# 전체 대화 데이터를 저장할 리스트
all_dialogues = []

for dialogue_idx in range(NUM_DIALOGUES):
    print(f"\n=== 대화 {dialogue_idx + 1} 시작 ===")
    
    # 대화 이력 초기화
    history = []
    
    # 페르소나 선택
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
    
    # 첫 발화
    _, slang_list = generate_slang_block(p1, context)
    utterance1 = start_input.get(context, "대화를 시작할 수 있는 문장을 찾지 못했어요.")
    user_input = utterance1
    print(f"Context: {context}")
    print(f"첫 발화: {user_input}")
    
    dialogue_data = []  # 현재 대화의 turn들을 저장할 리스트
    
    for turn in range(10):
        is_p1_turn = (turn % 2 == 0)
        speaker = "Persona1" if is_p1_turn else "Persona2"
        
        # 대화 기록 누적 (중복 제거)
        if turn == 0:
            full_input = user_input
        else:
            # 이전 대화 기록을 모두 포함
            history_text = "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in history])
            full_input = f"{history_text}"
        
        print(f"\n[대화 {dialogue_idx + 1} - Turn {turn+1} - history]")
        print(full_input)
        # 매 턴마다 slang block 새로 생성
        slang_block1, slang_list1 = generate_slang_block(p1, context)
        slang_block2, slang_list2 = generate_slang_block(p2, context)
        
        # 매 턴마다 prompt 새로 생성
        prompt1 = template.format(
            persona=format_persona(p1),
            partner_traits=f"{p2['나이대']}, {p2['성별']}",
            context=context[3:],
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
        
        system_prompt = prompt1 if is_p1_turn else prompt2
        print(f"\n[대화 {dialogue_idx + 1} - Turn {turn+1} - {speaker}]")
        
        # GPT 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"###Input###\n{full_input}"}
            ]
        )
        content = response.choices[0].message.content
        print(content)
        
        # 저장용 JSON 생성
        turn_data = {
            "turn": turn + 1,
            "speaker": speaker,
            "prompt": system_prompt,
            "input": full_input,
            "response": content,
            "slang_list": slang_list1 if is_p1_turn else slang_list2
        }
        dialogue_data.append(turn_data)
        
        # 다음 발화 추출
        try:
            next_utterance = content.split("Slang 미포함 답변:")[-1].strip()
        except Exception:
            next_utterance = content.strip()
        history.append({"speaker": speaker, "text": next_utterance})
        user_input = next_utterance
        time.sleep(1)
    
    # 현재 대화의 페르소나 정보 저장
    persona_info = {
        "Persona1": extract_persona_dict(p1, "Persona1"),
        "Persona2": extract_persona_dict(p2, "Persona2")
    }
    
    # 현재 대화 데이터 저장
    dialogue_info = {
        "dialogue_id": dialogue_idx + 1,
        "context": context,
        "persona_info": persona_info,
        "turns": dialogue_data
    }
    all_dialogues.append(dialogue_info)
    
    # 각 대화별로 파일 저장
    dialogue_dir = output_dir / f"dialogue_{dialogue_idx + 1}"
    dialogue_dir.mkdir(exist_ok=True)
    
    with open(dialogue_dir / "dialogue.json", "w", encoding="utf-8") as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
    
    with open(dialogue_dir / "persona.json", "w", encoding="utf-8") as f:
        json.dump(persona_info, f, ensure_ascii=False, indent=2)

# 모든 대화 데이터를 하나의 파일로 저장
with open(output_dir / "all_dialogues.json", "w", encoding="utf-8") as f:
    json.dump(all_dialogues, f, ensure_ascii=False, indent=2)