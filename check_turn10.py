import json
from pathlib import Path
import re

def check_turn10_predictions():
    """턴 10의 예측을 확인"""
    predictions_dir = Path("/data/nlp/output_final/persona_predictions")
    
    # m2 예측 파일 처리
    for file_path in predictions_dir.glob("*_m2_predictions.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        method = data['method']
        dialogue_type = data['dialogue_type']
        predictions = data['predictions']
        
        # 각 화자별 예측 확인
        for speaker in ['Persona1', 'Persona2']:
            if speaker not in predictions:
                print(f"\n파일: {file_path.name}")
                print(f"Method: {method}, Type: {dialogue_type}")
                print(f"{speaker}: 예측 없음")
                continue
                
            if '10' not in predictions[speaker]:
                print(f"\n파일: {file_path.name}")
                print(f"Method: {method}, Type: {dialogue_type}")
                print(f"{speaker}: 턴 10 예측 없음")
                continue
                
            turn10_data = predictions[speaker]['10']
            prediction = turn10_data.get('prediction', '').strip()
            
            if not prediction:
                print(f"\n파일: {file_path.name}")
                print(f"Method: {method}, Type: {dialogue_type}")
                print(f"{speaker} - 턴 10: 빈 예측")

if __name__ == "__main__":
    check_turn10_predictions() 