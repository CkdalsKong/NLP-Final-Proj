import json
from pathlib import Path
from typing import Dict, Any

def process_dialogue_file(dialogue_data: Dict[str, Any], dialogue_type: str) -> Dict[str, Any]:
    """대화 데이터에서 response의 -와 공백을 제거"""
    processed_dialogues = []
    for dialogue in dialogue_data["dialogues"]:
        processed_dialogue = dialogue.copy()
        # response에서 -와 공백 제거
        if isinstance(processed_dialogue["response"], str):
            if dialogue_type == "slang":
                # slang 대화의 경우 - 문장\n\n- 형식에서 문장만 추출
                response = processed_dialogue["response"].strip()
                if response.startswith("-") and "\n\n-" in response:
                    response = response[1:].split("\n\n-")[0].strip()
                processed_dialogue["response"] = response
            else:
                # non_slang 대화의 경우 기존 방식대로 처리
                processed_dialogue["response"] = processed_dialogue["response"].strip("- ")
        processed_dialogues.append(processed_dialogue)
    
    return {
        "persona_info": dialogue_data["persona_info"],
        "dialogues": processed_dialogues
    }

def main():
    # 입력/출력 디렉토리 설정
    base_dir = Path("/data/nlp/output_final")
    output_base_dir = Path("/data/nlp/output_final/processed_dialogues")
    
    # slang과 non_slang 각각 처리
    for dialogue_type in ["slang"]:
        input_dir = base_dir / f"{dialogue_type}_dialogues"
        output_dir = output_base_dir / f"{dialogue_type}_dialogues"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 대화 파일 처리
        for dialogue_file in input_dir.glob('dialogue_*.json'):
            dialogue_num = int(dialogue_file.stem.split('_')[1])
                
            # 대화 데이터 로드 및 처리
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogue_data = json.load(f)
            
            processed_data = process_dialogue_file(dialogue_data, dialogue_type)
            
            # 처리된 데이터 저장
            output_file = output_dir / f"dialogue_{dialogue_num}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            print(f"Processed {dialogue_type} dialogue_{dialogue_num}")

if __name__ == "__main__":
    main() 