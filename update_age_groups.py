import json
from pathlib import Path

def update_age_groups(dialogue_dir):
    """대화 파일들의 나이대 정보를 업데이트"""
    dialogue_dir = Path(dialogue_dir)
    total_files = 0
    updated_files = 0
    
    for dialogue_file in dialogue_dir.glob('dialogue_*.json'):
        total_files += 1
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_updated = False
        # 각 화자의 persona_info 확인
        for speaker in ['Persona1', 'Persona2']:
            if speaker in data['persona_info']:
                speaker_info = data['persona_info'][speaker]
                if '나이대' in speaker_info and speaker_info['나이대'] == '10대 이하':
                    speaker_info['나이대'] = '10대'
                    file_updated = True
        
        if file_updated:
            updated_files += 1
            # 업데이트된 정보 저장
            with open(dialogue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    return total_files, updated_files

def main():
    # 파일 경로 설정
    base_dir = Path("/data/nlp/output_final/reprocessed_dialogues")
    slang_dir = base_dir / "slang_dialogues"
    non_slang_dir = base_dir / "non_slang_dialogues"
    
    # slang 대화 파일 업데이트
    print("\nslang 대화 파일 업데이트 중...")
    slang_total, slang_updated = update_age_groups(slang_dir)
    print(f"총 파일 수: {slang_total}")
    print(f"업데이트된 파일 수: {slang_updated}")
    
    # non_slang 대화 파일 업데이트
    print("\nnon_slang 대화 파일 업데이트 중...")
    non_slang_total, non_slang_updated = update_age_groups(non_slang_dir)
    print(f"총 파일 수: {non_slang_total}")
    print(f"업데이트된 파일 수: {non_slang_updated}")
    
    print("\n=== 전체 통계 ===")
    print(f"총 파일 수: {slang_total + non_slang_total}")
    print(f"총 업데이트된 파일 수: {slang_updated + non_slang_updated}")

if __name__ == "__main__":
    main() 