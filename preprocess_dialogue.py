import json
import os
from pathlib import Path
import pandas as pd

def process_dialogue_file(dialogue_path):
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)
    
    slang_dialogues = []
    non_slang_dialogues = []
    
    for turn in dialogue_data:
        # 응답에서 슬랭 포함/미포함 답변 분리
        response = turn['response']
        if 'Slang 포함 답변:' in response and 'Slang 미포함 답변:' in response:
            slang_part = response.split('Slang 미포함 답변:')[0].replace('Slang 포함 답변:', '').strip()
            non_slang_part = response.split('Slang 미포함 답변:')[1].strip()
            
            # 슬랭 사용 대화 저장
            slang_turn = turn.copy()
            slang_turn['response'] = slang_part
            slang_dialogues.append(slang_turn)
            
            # 슬랭 미사용 대화 저장
            non_slang_turn = turn.copy()
            non_slang_turn['response'] = non_slang_part
            non_slang_dialogues.append(non_slang_turn)
    
    return slang_dialogues, non_slang_dialogues

def process_all_dialogues(input_dir, output_dir):
    # 입력/출력 디렉토리 생성
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    slang_output = output_path / 'slang_dialogues'
    non_slang_output = output_path / 'non_slang_dialogues'
    
    slang_output.mkdir(parents=True, exist_ok=True)
    non_slang_output.mkdir(parents=True, exist_ok=True)
    
    # 모든 대화 폴더 처리
    for dialogue_dir in input_path.glob('dialogue_*'):
        dialogue_num = dialogue_dir.name.split('_')[1]
        
        # dialogue.json 파일 처리
        dialogue_file = dialogue_dir / 'dialogue.json'
        if dialogue_file.exists():
            slang_dialogues, non_slang_dialogues = process_dialogue_file(dialogue_file)
            
            # 페르소나 정보 복사
            persona_file = dialogue_dir / 'persona.json'
            if persona_file.exists():
                with open(persona_file, 'r', encoding='utf-8') as f:
                    persona_data = json.load(f)
                
                # 슬랭 사용 대화 저장
                slang_data = {
                    'persona_info': persona_data,
                    'dialogues': slang_dialogues
                }
                with open(slang_output / f'dialogue_{dialogue_num}.json', 'w', encoding='utf-8') as f:
                    json.dump(slang_data, f, ensure_ascii=False, indent=2)
                
                # 슬랭 미사용 대화 저장
                non_slang_data = {
                    'persona_info': persona_data,
                    'dialogues': non_slang_dialogues
                }
                with open(non_slang_output / f'dialogue_{dialogue_num}.json', 'w', encoding='utf-8') as f:
                    json.dump(non_slang_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_dir = "/data/nlp/output_final/aa4a9c5f"  # 입력 디렉토리
    output_dir = "/data/nlp/output_final"  # 출력 디렉토리
    
    process_all_dialogues(input_dir, output_dir)
    print("전처리가 완료되었습니다.") 