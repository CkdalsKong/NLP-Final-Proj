import json
import time
import requests
import argparse
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class PersonaPredictor:
    def __init__(self, vllm_server_url: str, model_name: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.VLLM_SERVER_URL = vllm_server_url
        self.MODEL_NAME = model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def load_metric_prompt(self, metric: str, persona_num: str) -> str:
        """메트릭 프롬프트 파일 로드"""
        if metric == 'm1':
            with open("combined_metric_prompts.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('persona_num') == persona_num:
                        return data.get('metric1_prompt', '')
        else:  # metric == 'm2'
            # combined_metric_prompts.jsonl에서 persona_num에 해당하는 metric2_prompt 가져오기
            with open("combined_metric_prompts.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('persona_num') == persona_num:
                        return data.get('metric2_prompt', '')

    def generate_style_summary(self, dialogue_data: Dict, speaker: str) -> str:
        """대화의 스타일을 요약"""
        dialogues = dialogue_data["dialogues"]
        
        # 대화 컨텍스트 생성
        dialogue_context = ""
        speaker_turns = [turn for turn in dialogues if turn['speaker'] == speaker]
        for turn in speaker_turns:
            dialogue_context += f"{turn['speaker']}: {turn['response']}\n"
        
        # 스타일 요약을 위한 프롬프트 생성
        system_prompt = """당신은 대화를 통해 사람의 성격과 특징을 읽어내는 심리학자입니다.
        대화에서 드러나는 화자의 말투, 관심사, 가치관, 생활방식 등을 통해 그 사람의 모습을 그려보세요."""
        
        messages = [{
            "role": "user",
            "content": f"""
다음 대화를 통해 화자의 모습을 그려보세요.
대화에서 드러나는 화자의 말투, 관심사, 가치관, 생활방식 등을 통해 
그 사람이 어떤 사람인지, 어떤 삶을 살고 있는지, 어떤 생각을 하는지 
자연스럽게 묘사해주세요.

대화:
{dialogue_context}"""
        }]
        
        # vLLM을 통한 요약 생성
        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.VLLM_SERVER_URL}/chat/completions"
        
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        
        payload = {
            "model": self.MODEL_NAME,
            "messages": formatted_messages,
            "temperature": 0.0,
            "max_tokens": 512,
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
            "dtype": "float32"
        }
        
        for attempt in range(5):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if response.status_code != 200:
                    print(f"Error response: {response.text}")
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt+1}/5] Request failed: {e}")
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("Failed to get response from vLLM server after 5 attempts")

    def find_similar_styles(self, style_summary: str, dialogue_type: str, top_k: int = 3) -> List[Dict]:
        """유사한 스타일의 대화 찾기"""
        # 스타일 임베딩 생성
        query_embedding = self.embedding_model.encode(style_summary)
        
        # FAISS 인덱스 로드
        index_path = Path(f"/data/nlp/output_final/processed_dialogues/style_embeddings/{dialogue_type}_style_index.faiss")
        metadata_path = Path(f"/data/nlp/output_final/processed_dialogues/style_embeddings/{dialogue_type}_style_metadata.json")
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("스타일 인덱스나 메타데이터 파일을 찾을 수 없습니다.")
        
        # FAISS 인덱스와 메타데이터 로드
        index = faiss.read_index(str(index_path))
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 유사도 검색
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        
        # 결과 반환
        similar_styles = []
        for idx in indices[0]:
            similar_styles.append(metadata[idx])
        
        return similar_styles

    def get_dialogue_context_a1(self, turns: List[Dict]) -> str:
        """4턴씩 끊어서 대화 컨텍스트 생성"""
        dialogue_text = ""
        for turn in turns:
            dialogue_text += f"{turn['speaker']}: {turn['response']}\n"
        return dialogue_text

    def get_dialogue_context_a2(self, turns: List[Dict]) -> str:
        """전체 대화 컨텍스트 생성"""
        dialogue_text = ""
        for turn in turns:
            dialogue_text += f"{turn['speaker']}: {turn['response']}\n"
        return dialogue_text

    def get_dialogue_context_a3(self, turns: List[Dict]) -> str:
        """대화의 주제와 맥락을 포함한 컨텍스트 생성"""
        dialogue_text = ""
        for turn in turns:
            dialogue_text += f"{turn['speaker']}: {turn['response']}\n"
        return dialogue_text

    def get_dialogue_context_a4(self, turns: List[Dict], speaker: str, dialogue_type: str) -> tuple[str, List[Dict]]:
        """대화의 감정과 어조를 포함한 컨텍스트 생성 및 유사 스타일 검색"""
        # 현재 대화의 스타일 요약 생성
        dialogue_data = {"dialogues": turns}
        style_summary = self.generate_style_summary(dialogue_data, speaker)
        
        # 유사한 스타일의 대화 찾기
        similar_styles = self.find_similar_styles(style_summary, dialogue_type)
        
        # 대화 컨텍스트 생성 (해당 화자의 대화만)
        dialogue_context = ""
        speaker_turns = [turn for turn in turns if turn['speaker'] == speaker]
        for turn in speaker_turns:
            dialogue_context += f"{turn['speaker']}: {turn['response']}\n"
        
        # 유사 스타일 예시 생성
        similar_examples = "대화 스타일과 유사한 페르소나 예시들:\n\n"
        for i, style in enumerate(similar_styles, 1):
            persona = style['persona_info']
            similar_examples += f"예시 {i}:\n"
            similar_examples += f"- {persona.get('나이대', '알 수 없음')} {persona.get('성별', '알 수 없음')}\n"
            if '취향' in persona:
                similar_examples += f"- {persona['취향']}\n"
            if '취미/관심사' in persona:
                similar_examples += f"- {persona['취미/관심사']}\n"
            if '환경' in persona:
                similar_examples += f"- {persona['환경']}\n"
            if '성격/가치관' in persona:
                similar_examples += f"- {persona['성격/가치관']}\n"
            if '직업군' in persona:
                similar_examples += f"- {persona['직업군']}\n"
            if '현황' in persona:
                similar_examples += f"- {persona['현황']}\n"
            similar_examples += "\n"
        
        return dialogue_context, similar_examples

    def create_prompt(self, dialogue_context: str, metric_prompt: str, metric: str, method: str, similar_examples: str = None) -> tuple[str, List[Dict]]:
        """메트릭 프롬프트를 사용하여 최종 프롬프트 생성"""
        if metric == 'm1':
            # metric1 (선택형)의 경우
            system_prompt = """{당신은 대화를 분석하여 화자의 페르소나를 추측하는 전문가입니다.
            또한 처음 보는 단어가 있다면 그 뜻을 유추하고 아래 과제를 수행하세요.
            주어진 후보 페르소나 중에서 가장 적합한 하나를 선택해주세요.
            선택한 페르소나의 글자만 출력하세요 (예: "A").}"""
            
            if method == "a1":
                content = f"""다음은 전체 대화입니다. 이 대화를 통해 각 화자의 페르소나를 추측해주세요.

대화:
{dialogue_context}

{metric_prompt}"""
            elif method == "a2":
                content = f"""다음은 한 화자의 대화입니다. 이 대화를 통해 화자의 페르소나를 추측해주세요.

대화:
{dialogue_context}

{metric_prompt}"""
            elif method == "a3":
                content = f"""다음은 대화의 일부 구간입니다. 이 구간을 통해 화자의 페르소나를 추측해주세요.

대화:
{dialogue_context}

{metric_prompt}"""
            elif method == "a4":
                content = f"""다음은 한 화자의 대화입니다. 이 대화를 통해 화자의 페르소나를 추측해주세요.

대화:
{dialogue_context}"""

                if similar_examples:
                    content += f"\n\n참고할 유사한 스타일의 대화 예시들:\n{similar_examples}"
                
                content += f"\n{metric_prompt}"
        else:
            # metric2 (발화 생성형)의 경우
            system_prompt = """당신은 대화를 분석하여 화자의 페르소나를 추측하는 전문가입니다.
            주어진 대화의 맥락을 고려하여 마지막 화자의 답변을 생성해주세요."""
            
            if method == "a1":
                content = f"""다음은 전체 대화입니다. 이 대화를 통해 각 화자의 페르소나를 추측하고, 마지막 화자의 답변을 생성해주세요.

대화:
{dialogue_context}

{metric_prompt}"""
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        return system_prompt, messages

    def generate_message_vllm(self, messages: List[Dict], system_prompt: str, max_tokens: int = 512) -> str:
        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.VLLM_SERVER_URL}/chat/completions"
        
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        
        payload = {
            "model": self.MODEL_NAME,
            "messages": formatted_messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
            "dtype": "float32"
        }
        
        for attempt in range(5):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if response.status_code != 200:
                    print(f"Error response: {response.text}")
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt+1}/5] Request failed: {e}")
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("Failed to get response from vLLM server after 5 attempts")

    def process_dialogue(self, dialogue_data: Dict, method: str, metric: str, dialogue_type: str) -> Dict[str, Any]:
        """대화 데이터를 처리하여 페르소나 추측"""
        dialogues = dialogue_data["dialogues"]
        print(f"\n=== {method} 방법으로 대화 처리 시작 ===")
        
        # 각 화자의 persona_num 저장
        persona_nums = {}
        for speaker in ['Persona1', 'Persona2']:
            persona_num = dialogue_data["persona_info"][speaker]["persona_num"]
            persona_nums[speaker] = persona_num
        predictions = {}
        
        if metric == 'm3':
            # m3 메트릭의 경우 speaker_num에 해당하는 m2 프롬프트만 사용
            for speaker in ['Persona1', 'Persona2']:
                print(f"\n{speaker} 처리 중...")
                # 각 화자별로 m2 프롬프트 로드
                persona_num = dialogue_data["persona_info"][speaker]["persona_num"]
                metric_prompt = self.load_metric_prompt('m2', persona_num)
                
                # 프롬프트 생성 (대화 컨텍스트 없이)
                system_prompt = """당신은 대화를 생성하는 전문가입니다.
                주어진 프롬프트에 따라 적절한 대화를 생성해주세요."""
                
                messages = [{
                    "role": "user",
                    "content": metric_prompt
                }]
                
                prediction = self.generate_message_vllm(messages, system_prompt)
                predictions[speaker] = {
                    "prediction": prediction,
                    "prompt": {
                        "system_prompt": system_prompt,
                        "user_prompt": messages[0]["content"]
                    }
                }
                
        elif metric == 'm2':
            # m2 메트릭의 경우 특정 턴에서만 예측
            target_turns = [2, 4, 6, 8, 10]
            turn_predictions = {}
            
            for turn_num in target_turns:
                if turn_num <= len(dialogues):
                    print(f"\n[1/2] 턴 {turn_num} 처리 중...")
                    # 해당 턴까지의 대화 컨텍스트 생성
                    dialogue_context = ""
                    for turn in dialogues[:turn_num]:
                        dialogue_context += f"{turn['speaker']}: {turn['response']}\n"
                    
                    # 각 화자별로 처리
                    for speaker in ['Persona1', 'Persona2']:
                        print(f"[2/2] {speaker}의 페르소나 추측 중...")
                        # 각 화자별로 메트릭 프롬프트 로드 (m1 프롬프트 사용)
                        persona_num = dialogue_data["persona_info"][speaker]["persona_num"]
                        metric_prompt = self.load_metric_prompt('m1', persona_num)
                        
                        system_prompt, messages = self.create_prompt(dialogue_context, metric_prompt, 'm1', method)
                        prediction = self.generate_message_vllm(messages, system_prompt)
                        
                        if speaker not in turn_predictions:
                            turn_predictions[speaker] = {}
                        
                        turn_predictions[speaker][turn_num] = {
                            "prediction": prediction,
                            "prompt": {
                                "system_prompt": system_prompt,
                                "user_prompt": messages[0]["content"]
                            }
                        }
            
            predictions = turn_predictions
            
        elif method == "a4":
            print("\n[1/3] 대화 스타일 분석 중...")
            # 각 화자별로 처리
            for speaker in ['Persona1', 'Persona2']:
                print(f"  - {speaker}의 스타일 분석 중...")
                # 각 화자별로 메트릭 프롬프트 로드
                persona_num = dialogue_data["persona_info"][speaker]["persona_num"]
                metric_prompt = self.load_metric_prompt('m1', persona_num)
                
                dialogue_context, similar_examples = self.get_dialogue_context_a4(dialogues, speaker, dialogue_type)
                
                print("[2/3] 페르소나 추측 중...")
                system_prompt, messages = self.create_prompt(dialogue_context, metric_prompt, metric, method, similar_examples)
                prediction = self.generate_message_vllm(messages, system_prompt)
                
                predictions[speaker] = {
                    "turns": [turn["turn"] for turn in dialogues if turn["speaker"] == speaker],
                    "prediction": prediction,
                    "prompt": {
                        "system_prompt": system_prompt,
                        "user_prompt": messages[0]["content"]
                    }
                }
            
        else:
            # 기존 메소드들 처리
            if method in ["a1", "a2"]:
                print("\n[1/2] 전체 대화 컨텍스트 생성 중...")
                dialogue_context = self.get_dialogue_context_a1(dialogues) if method == "a1" else self.get_dialogue_context_a2(dialogues)
                
                # 각 화자별로 처리
                for speaker in ['Persona1', 'Persona2']:
                    print(f"[2/2] {speaker}의 페르소나 추측 중...")
                    # 각 화자별로 메트릭 프롬프트 로드
                    persona_num = dialogue_data["persona_info"][speaker]["persona_num"]
                    metric_prompt = self.load_metric_prompt(metric, persona_num)
                    
                    system_prompt, messages = self.create_prompt(dialogue_context, metric_prompt, metric, method)
                    prediction = self.generate_message_vllm(messages, system_prompt)
                    predictions[speaker] = {
                        "turns": [turn["turn"] for turn in dialogues if turn["speaker"] == speaker],
                        "prediction": prediction,
                        "prompt": {
                            "system_prompt": system_prompt,
                            "user_prompt": messages[0]["content"]
                        }
                    }
                
            elif method == "a3":
                print("\n[1/4] 4턴씩 대화 그룹 생성 중...")
                # 4턴씩 묶어서 처리
                turn_groups = [
                    dialogues[0:4],    # 0,1,2,3
                    dialogues[2:6],    # 2,3,4,5
                    dialogues[4:8],    # 4,5,6,7
                    dialogues[6:10]    # 6,7,8,9
                ]
                
                # 각 화자별로 처리
                for speaker in ['Persona1', 'Persona2']:
                    print(f"\n{speaker} 처리 중...")
                    # 각 화자별로 메트릭 프롬프트 로드
                    persona_num = dialogue_data["persona_info"][speaker]["persona_num"]
                    metric_prompt = self.load_metric_prompt('m1', persona_num)
                    
                    group_predictions = []
                    
                    # 각 4턴 그룹에 대해 예측 수행
                    print("[2/4] 각 4턴 그룹별 페르소나 추측 중...")
                    for i, turns in enumerate(turn_groups):
                        print(f"  - 그룹 {i+1} (턴 {turns[0]['turn']}-{turns[-1]['turn']}) 처리 중...")
                        dialogue_context = self.get_dialogue_context_a3(turns)
                        
                        # 그룹 예측에서는 메트릭 프롬프트를 사용하지 않음
                        group_system_prompt = """
당신은 대화를 분석하여 화자의 페르소나를 추측하는 전문가입니다.
화자의 대화를 통해 그 사람의 성격, 관심사, 생활방식 등을 간단명료하게 묘사해주세요.
응답은 다음 형식을 따라주세요:
1. 대화 분석: 화자의 대화에서 드러나는 핵심 특징 2-3가지
2. 페르소나 묘사: 화자의 성격, 관심사, 생활방식 등을 간단히 묘사
3. 결론: 핵심 페르소나 특징 1-2문장으로 요약"""
                        
                        group_messages = [{
                            "role": "user",
                            "content": f"""다음은 대화의 일부 구간입니다. 이 구간을 통해 {speaker}의 페르소나를 추측해주세요.

대화:
{dialogue_context}"""
                        }]
                        
                        prediction = self.generate_message_vllm(group_messages, group_system_prompt)
                        group_predictions.append({
                            "turns": [turn["turn"] for turn in turns],
                            "prediction": prediction,
                            "prompt": {
                                "system_prompt": group_system_prompt,
                                "user_prompt": group_messages[0]["content"]
                            }
                        })
                    
                    print("[3/4] 모든 그룹의 예측 결과 종합 중...")
                    # 모든 그룹의 예측을 종합하여 최종 예측 생성
                    combined_context = "각 대화 구간에 대한 페르소나 추측 결과입니다:\n\n"
                    for i, pred in enumerate(group_predictions):
                        combined_context += f"대화 구간 {i+1} (턴 {pred['turns'][0]}-{pred['turns'][-1]}):\n{pred['prediction']}\n\n"
                    
                    print("[4/4] 최종 페르소나 추측 중...")
                    # 최종 예측을 위한 프롬프트 생성
                    final_system_prompt = f"""당신은 여러 대화 구간의 페르소나 추측 결과를 종합하여 최종 페르소나를 추측하는 전문가입니다.

{metric_prompt}"""
                    final_messages = [{
                        "role": "user",
                        "content": f"""다음은 여러 대화 구간에 대한 페르소나 추측 결과입니다. 이 결과들을 종합하여 {speaker}의 최종 페르소나를 추측해주세요.

{combined_context}"""
                    }]
                    
                    # 최종 예측 생성
                    final_prediction = self.generate_message_vllm(final_messages, final_system_prompt)
                    
                    # 모든 예측 결과 저장
                    predictions[speaker] = {
                        "group_predictions": group_predictions,
                        "final_prediction": final_prediction,
                        "final_prompt": {
                            "system_prompt": final_system_prompt,
                            "user_prompt": final_messages[0]["content"]
                        }
                    }
        
        print(f"\n=== {method} 방법 처리 완료 ===\n")
        return {
            "method": method,
            "metric": metric,
            "dialogue_type": dialogue_type,
            "persona_nums": persona_nums,
            "dialogues": dialogues,
            "predictions": predictions
        }

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='페르소나 추측 스크립트')
    parser.add_argument('--method', type=str, choices=['a1', 'a2', 'a3', 'a4'], required=True,
                      help='페르소나 추측 방법 선택 (a1, a2, a3, a4)')
    parser.add_argument('--type', type=str, choices=['slang', 'non_slang'], required=True,
                      help='대화 유형 선택 (slang, non_slang)')
    parser.add_argument('--metric', type=str, choices=['m1', 'm2', 'm3'], required=True,
                      help='평가 메트릭 선택 (m1, m2, m3)')
    args = parser.parse_args()

    print(f"\n=== 설정 정보 ===")
    print(f"메소드: {args.method}")
    print(f"대화 유형: {args.type}")
    print(f"평가 메트릭: {args.metric}")

    # 설정
    VLLM_SERVER_URL = "http://localhost:8006/v1"  # vLLM 서버 URL
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 모델 이름
    
    # 입력/출력 디렉토리 설정
    base_dir = Path("/data/nlp/output_final/reprocessed_dialogues2")
    output_dir = Path("/data/nlp/output_final/persona_predictions3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = base_dir / f"{args.type}_dialogues"
    
    # PersonaPredictor 초기화
    predictor = PersonaPredictor(VLLM_SERVER_URL, MODEL_NAME)
    
    # 각 대화 파일 처리
    print("\n[2/3] 대화 파일 처리 시작...")
    for dialogue_file in input_dir.glob('dialogue_*.json'):
        dialogue_num = int(dialogue_file.stem.split('_')[1])
        print(f"\n대화 {dialogue_num} 처리 중...")
        
        # 1~150번 다이얼로그만 처리
        if dialogue_num > 150:
            continue
            
        # 대화 데이터 로드
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
        
        # 선택된 메소드로 처리
        predictions = predictor.process_dialogue(dialogue_data, args.method, args.metric, args.type)
        
        # 결과 저장
        print(f"\n[3/3] 결과 저장 중...")
        output_file = output_dir / f"dialogue_{dialogue_num}_{args.method}_{args.type}_{args.metric}_predictions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"대화 {dialogue_num} 처리 완료")

if __name__ == "__main__":
    main() 