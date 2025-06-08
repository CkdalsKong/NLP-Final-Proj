import json
from pathlib import Path
from collections import defaultdict
import re
import argparse
from typing import Dict

def extract_choice(prediction: str) -> str:
    """예측값에서 선택지 추출 (A, B, C, D, E 중 하나)"""
    # 입력이 None이거나 빈 문자열인 경우
    if not prediction:
        return ''
        
    # 문자열 앞뒤 공백 제거
    prediction = prediction.strip()
    
    # 마침표 제거
    prediction = prediction.replace('.', '')
    
    # 첫 글자가 A-E 중 하나인 경우
    if prediction and prediction[0] in 'ABCDE':
        return prediction[0]
    
    # 정규식으로 선택지 추출 (예: "D)", "D:", "D " 등)
    match = re.search(r'[A-E][\)\:\s]', prediction)
    if match:
        return match.group()[0]  # 첫 글자만 반환 (A, B, C, D, E)
    
    # 마지막 시도: 문자열에서 A-E 찾기
    for char in prediction:
        if char in 'ABCDE':
            return char
            
    return ''

def load_ground_truth():
    """combined_metric_prompts.jsonl에서 ground truth 로드"""
    ground_truth = {}
    print("\n=== Ground Truth 로딩 시작 ===")
    try:
        with open("combined_metric_prompts.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                persona_num = data.get('persona_num')
                if persona_num:
                    ground_truth[persona_num] = {
                        'm1': data.get('metric1_GT'),
                        'm2': data.get('metric2_GS')  # metric2_GS 사용
                    }
        print(f"로드된 Ground Truth 개수: {len(ground_truth)}")
        print("첫 5개 Ground Truth 샘플:")
        for i, (persona_num, data) in enumerate(list(ground_truth.items())[:5]):
            print(f"  {persona_num}: {data}")
    except Exception as e:
        print(f"Ground Truth 로딩 중 에러 발생: {e}")
    return ground_truth

def calculate_m1_accuracy(predictions_dir: str, ground_truth: Dict, method: str = 'all') -> Dict:
    """m1 메트릭의 정확도 계산"""
    # 선택된 메소드만 결과 딕셔너리에 포함
    results = {
        method: {'slang': {'correct': 0, 'total': 0}, 'non_slang': {'correct': 0, 'total': 0}}
    } if method != 'all' else {
        'a1': {'slang': {'correct': 0, 'total': 0}, 'non_slang': {'correct': 0, 'total': 0}},
        'a2': {'slang': {'correct': 0, 'total': 0}, 'non_slang': {'correct': 0, 'total': 0}},
        'a3': {'slang': {'correct': 0, 'total': 0}, 'non_slang': {'correct': 0, 'total': 0}},
        'a4': {'slang': {'correct': 0, 'total': 0}, 'non_slang': {'correct': 0, 'total': 0}}
    }
    
    # m1 예측 파일 처리
    pattern = f"*_m1_predictions.json" if method == 'all' else f"*_{method}_*_m1_predictions.json"
    for file_path in Path(predictions_dir).glob(pattern):
        print(f"\n파일 처리 중: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        method = data['method']
        dialogue_type = data['dialogue_type']
        persona_nums = data['persona_nums']
        predictions = data['predictions']
        
        print(f"Method: {method}, Type: {dialogue_type}")
        print(f"Persona nums: {persona_nums}")
        
        # 각 화자별 예측 처리
        for speaker, persona_num in persona_nums.items():
            if speaker not in predictions:
                print(f"Warning: {speaker}에 대한 예측이 없습니다.")
                continue
                
            prediction_data = predictions[speaker]
            
            # a3 메소드의 경우 final_prediction 사용
            if method == 'a3':
                prediction = prediction_data.get('final_prediction', '')
            else:
                prediction = prediction_data.get('prediction', '')
                
            extracted_choice = extract_choice(prediction)
            
            # ground truth 찾기
            gt = ground_truth.get(persona_num, {}).get('m1')
            
            if gt is None:
                print(f"Warning: {persona_num}에 대한 ground truth를 찾을 수 없습니다.")
                continue
                
            print(f"{speaker} ({persona_num}):")
            print(f"  예측: {extracted_choice} (원본: {prediction})")
            print(f"  정답: {gt}")
            
            results[method][dialogue_type]['total'] += 1
            if extracted_choice == gt:
                results[method][dialogue_type]['correct'] += 1
                print("  결과: 정답")
            else:
                print("  결과: 오답")
    
    return results

def calculate_m2_accuracy(predictions_dir: str, ground_truth: Dict, method: str = 'all') -> Dict:
    """m2 메트릭의 정확도 계산"""
    # 선택된 메소드만 결과 딕셔너리에 포함
    results = {
        method: {
            'slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]},
            'non_slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}
        }
    } if method != 'all' else {
        'a1': {'slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}, 'non_slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}},
        'a2': {'slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}, 'non_slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}},
        'a3': {'slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}, 'non_slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}},
        'a4': {'slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}, 'non_slang': {str(i): {'correct': 0, 'total': 0} for i in [2, 4, 6, 8, 10]}}
    }
    
    # m2 예측 파일 처리
    pattern = f"*_m2_predictions.json" if method == 'all' else f"*_{method}_*_m2_predictions.json"
    for file_path in Path(predictions_dir).glob(pattern):
        print(f"\n파일 처리 중: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        method = data['method']
        dialogue_type = data['dialogue_type']
        persona_nums = data['persona_nums']
        predictions = data['predictions']
        
        print(f"Method: {method}, Type: {dialogue_type}")
        print(f"Persona nums: {persona_nums}")
        
        # 각 화자별 예측 처리
        for speaker, persona_num in persona_nums.items():
            if speaker not in predictions:
                print(f"Warning: {speaker}에 대한 예측이 없습니다.")
                continue
                
            # 각 턴별 예측 처리
            for turn_num, turn_data in predictions[speaker].items():
                prediction = turn_data.get('prediction', '')
                extracted_choice = extract_choice(prediction)
                
                # ground truth 찾기 (m2의 경우에도 metric1_GT 사용)
                gt = ground_truth.get(persona_num, {}).get('m1')  # m1의 ground truth 사용
                if gt is None:
                    print(f"Warning: {persona_num}에 대한 ground truth를 찾을 수 없습니다.")
                    continue
                    
                print(f"{speaker} ({persona_num}) - 턴 {turn_num}:")
                print(f"  예측: {extracted_choice} (원본: {prediction})")
                print(f"  정답: {gt}")
                
                results[method][dialogue_type][turn_num]['total'] += 1
                if extracted_choice == gt:
                    results[method][dialogue_type][turn_num]['correct'] += 1
                    print("  결과: 정답")
                else:
                    print("  결과: 오답")
    
    return results

def print_results(m1_results, m2_results, metric_type=None):
    """결과 출력"""
    if metric_type is None or metric_type == 'm1':
        print("\n=== m1 메트릭 정확도 ===")
        for method in sorted(m1_results.keys()):
            print(f"\n[{method}]")
            total_correct = 0
            total_samples = 0
            for dialogue_type in sorted(m1_results[method].keys()):
                stats = m1_results[method][dialogue_type]
                accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"  {dialogue_type}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
                total_correct += stats['correct']
                total_samples += stats['total']
            avg_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
            print(f"  평균: {avg_accuracy:.2f}% ({total_correct}/{total_samples})")
    
    if metric_type is None or metric_type == 'm2':
        print("\n=== m2 메트릭 정확도 ===")
        for method in sorted(m2_results.keys()):
            print(f"\n[{method}]")
            total_correct = 0
            total_samples = 0
            for dialogue_type in sorted(m2_results[method].keys()):
                print(f"  {dialogue_type}:")
                type_correct = 0
                type_samples = 0
                for turn_num in sorted(m2_results[method][dialogue_type].keys(), key=int):
                    stats = m2_results[method][dialogue_type][turn_num]
                    accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"    턴 {turn_num}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
                    type_correct += stats['correct']
                    type_samples += stats['total']
                type_avg = (type_correct / type_samples * 100) if type_samples > 0 else 0
                print(f"    평균: {type_avg:.2f}% ({type_correct}/{type_samples})")
                total_correct += type_correct
                total_samples += type_samples
            avg_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
            print(f"  전체 평균: {avg_accuracy:.2f}% ({total_correct}/{total_samples})")

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='대화 예측 정확도 계산')
    parser.add_argument('--metric', choices=['m1', 'm2', 'both'], default='both',
                      help='계산할 메트릭 선택 (m1, m2, both)')
    parser.add_argument('--method', choices=['a1', 'a2', 'a3', 'a4', 'all'], default='all',
                      help='계산할 메소드 선택 (a1, a2, a3, a4, all)')
    parser.add_argument('--type', choices=['slang', 'non_slang', 'both'], default='both',
                      help='계산할 대화 유형 선택 (slang, non_slang, both)')
    args = parser.parse_args()
    
    # persona_predictions 폴더 경로
    predictions_dir = Path("/data/nlp/output_final/persona_predictions2")
    print(f"\n=== 시작 ===")
    print(f"예측 파일 디렉토리: {predictions_dir}")
    print(f"선택된 메트릭: {args.metric}")
    print(f"선택된 메소드: {args.method}")
    print(f"선택된 대화 유형: {args.type}")
    
    # Ground truth 로드
    ground_truth = load_ground_truth()
    
    # 선택된 메트릭에 따라 정확도 계산
    m1_results = {}
    m2_results = {}
    
    if args.metric in ['m1', 'both']:
        m1_results = calculate_m1_accuracy(predictions_dir, ground_truth, args.method)
        # 선택된 대화 유형에 따라 결과 필터링
        if args.type != 'both':
            for method in m1_results:
                m1_results[method] = {k: v for k, v in m1_results[method].items() if k == args.type}
    
    if args.metric in ['m2', 'both']:
        m2_results = calculate_m2_accuracy(predictions_dir, ground_truth, args.method)
        # 선택된 대화 유형에 따라 결과 필터링
        if args.type != 'both':
            for method in m2_results:
                m2_results[method] = {k: v for k, v in m2_results[method].items() if k == args.type}
    
    # 결과 출력
    print_results(m1_results, m2_results, args.metric)

if __name__ == "__main__":
    main()