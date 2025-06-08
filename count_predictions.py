import json
from pathlib import Path
from collections import defaultdict

def count_predictions():
    # persona_predictions 폴더 경로
    predictions_dir = Path("/data/nlp/output_final/persona_predictions3")
    
    # 메트릭, 메소드, 타입별 파일 개수를 저장할 딕셔너리
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # 모든 JSON 파일 검사
    for file_path in predictions_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 파일명에서 정보 추출
            metric = data.get('metric', 'unknown')
            method = data.get('method', 'unknown')
            dialogue_type = data.get('dialogue_type', 'unknown')
            
            # 카운트 증가
            counts[metric][method][dialogue_type] += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # 결과 출력
    print("\n=== 예측 파일 개수 통계 ===")
    for metric in sorted(counts.keys()):
        print(f"\n[{metric}]")
        for method in sorted(counts[metric].keys()):
            print(f"  {method}:")
            for dialogue_type in sorted(counts[metric][method].keys()):
                count = counts[metric][method][dialogue_type]
                print(f"    - {dialogue_type}: {count}개")
    
    # 전체 파일 개수
    total_files = sum(sum(sum(counts[m][me][t] for t in counts[m][me]) for me in counts[m]) for m in counts)
    print(f"\n총 파일 개수: {total_files}개")

if __name__ == "__main__":
    count_predictions() 