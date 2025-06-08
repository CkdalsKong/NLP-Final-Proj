import json
import pandas as pd
from pathlib import Path

def load_persona_data(csv_path):
    """CSV 파일에서 페르소나 데이터 로드"""
    df = pd.read_csv(csv_path)
    # 페르소나 정보를 문자열로 만들어서 매칭할 수 있도록 준비
    persona_mapping = {}
    for _, row in df.iterrows():
        # CSV의 페르소나 정보를 대화 파일 형식으로 변환
        persona_info = {
            '나이대': row['나이대'],
            '성별': row['성별']
        }
        # 선택적 정보 추가 (빈 값이 아닌 경우에만)
        for field in ['직업군', '취향', '취미/관심사', '환경', '성격/가치관', '현황']:
            if field in row and pd.notna(row[field]) and row[field] != '':
                persona_info[field] = row[field]
        
        # 페르소나 정보를 문자열로 변환하여 키로 사용
        persona_key = json.dumps(persona_info, sort_keys=True, ensure_ascii=False)
        persona_mapping[persona_key] = row['persona_num']
    
    return persona_mapping

def update_dialogue_files(dialogue_dir, persona_mapping, output_base_dir):
    """대화 파일들의 persona_info에 persona_num 추가"""
    dialogue_dir = Path(dialogue_dir)
    # 입력 디렉토리 이름 가져오기 (slang_dialogues 또는 non_slang_dialogues)
    dialogue_type = dialogue_dir.name
    # 출력 디렉토리 생성
    output_dir = output_base_dir / dialogue_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    updated_files = 0
    persona1_updated = 0
    persona2_updated = 0
    
    for dialogue_file in dialogue_dir.glob('dialogue_*.json'):
        total_files += 1
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_updated = False
        # 각 화자의 persona_info 확인
        for speaker in ['Persona1', 'Persona2']:
            if speaker in data['persona_info']:
                speaker_info = data['persona_info'][speaker].copy()  # 원본 보존을 위해 복사
                # speaker와 persona_num 필드 제거 (CSV와 비교할 때는 필요없음)
                if 'speaker' in speaker_info:
                    del speaker_info['speaker']
                if 'persona_num' in speaker_info:
                    del speaker_info['persona_num']
                
                # 페르소나 정보를 문자열로 변환
                persona_key = json.dumps(speaker_info, sort_keys=True, ensure_ascii=False)
                # 매칭되는 persona_num 찾기
                if persona_key in persona_mapping:
                    data['persona_info'][speaker]['persona_num'] = persona_mapping[persona_key]
                    file_updated = True
                    if speaker == 'Persona1':
                        persona1_updated += 1
                    else:
                        persona2_updated += 1
        
        if file_updated:
            updated_files += 1
            # 업데이트된 정보 저장
            output_file = output_dir / dialogue_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    return total_files, updated_files, persona1_updated, persona2_updated

def main():
    # 파일 경로 설정
    csv_path = Path("/home/ubuntu/changmin/final_processed_dataset/persona_dataset/combined_personas_with_utterances.csv")
    slang_dir = Path("/data/nlp/output_final/processed_dialogues/slang_dialogues")
    non_slang_dir = Path("/data/nlp/output_final/processed_dialogues/non_slang_dialogues")
    output_base_dir = Path("/data/nlp/output_final/reprocessed_dialogues")
    
    # 페르소나 데이터 로드
    print("페르소나 데이터 로드 중...")
    persona_mapping = load_persona_data(csv_path)
    print(f"로드된 페르소나 매핑 수: {len(persona_mapping)}")
    
    # slang 대화 파일 업데이트
    print("\nslang 대화 파일 업데이트 중...")
    slang_total, slang_updated, slang_p1, slang_p2 = update_dialogue_files(slang_dir, persona_mapping, output_base_dir)
    print(f"총 파일 수: {slang_total}")
    print(f"업데이트된 파일 수: {slang_updated}")
    print(f"Persona1 업데이트 수: {slang_p1}")
    print(f"Persona2 업데이트 수: {slang_p2}")
    
    # non_slang 대화 파일 업데이트
    print("\nnon_slang 대화 파일 업데이트 중...")
    non_slang_total, non_slang_updated, non_slang_p1, non_slang_p2 = update_dialogue_files(non_slang_dir, persona_mapping, output_base_dir)
    print(f"총 파일 수: {non_slang_total}")
    print(f"업데이트된 파일 수: {non_slang_updated}")
    print(f"Persona1 업데이트 수: {non_slang_p1}")
    print(f"Persona2 업데이트 수: {non_slang_p2}")
    
    print("\n=== 전체 통계 ===")
    print(f"총 파일 수: {slang_total + non_slang_total}")
    print(f"총 업데이트된 파일 수: {slang_updated + non_slang_updated}")
    print(f"총 Persona1 업데이트 수: {slang_p1 + non_slang_p1}")
    print(f"총 Persona2 업데이트 수: {slang_p2 + non_slang_p2}")

if __name__ == "__main__":
    main() 