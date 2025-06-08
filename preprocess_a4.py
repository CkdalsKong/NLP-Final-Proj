import json
import time
import requests
import argparse
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class DialogueStyleProcessor:
    def __init__(self, vllm_server_url: str, model_name: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.VLLM_SERVER_URL = vllm_server_url
        self.MODEL_NAME = model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def generate_style_summary(self, dialogue_data: Dict, speaker: str = None) -> str:
        """대화의 스타일을 요약"""
        dialogues = dialogue_data["dialogues"]
        
        # 대화 컨텍스트 생성
        dialogue_context = ""
        if speaker:
            # 특정 화자의 대화만 선택
            speaker_turns = [turn for turn in dialogues if turn['speaker'] == speaker]
            for turn in speaker_turns:
                dialogue_context += f"{turn['speaker']}: {turn['response']}\n"
        else:
            # 전체 대화
            for turn in dialogues:
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

    def create_embedding(self, text: str) -> np.ndarray:
        """텍스트를 임베딩으로 변환"""
        return self.embedding_model.encode(text)

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='대화 스타일 전처리 스크립트')
    parser.add_argument('--type', type=str, choices=['slang', 'non_slang'], required=True,
                      help='대화 유형 선택 (slang, non_slang)')
    args = parser.parse_args()

    print(f"\n=== 설정 정보 ===")
    print(f"대화 유형: {args.type}")
    
    # 설정
    VLLM_SERVER_URL = "http://localhost:8006/v1"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 디렉토리 설정
    base_dir = Path("/data/nlp/output_final/reprocessed_dialogues2")
    output_dir = Path("/data/nlp/output_final/reprocessed_dialogues2/style_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = base_dir / f"{args.type}_dialogues"
    
    # 프로세서 초기화
    processor = DialogueStyleProcessor(VLLM_SERVER_URL, MODEL_NAME)
    
    # 스타일 요약 및 임베딩 생성
    print("\n[1/3] 스타일 요약 및 임베딩 생성 중...")
    style_summaries = defaultdict(dict)  # dialogue_id -> speaker -> summary
    embeddings = []  # 모든 임베딩을 하나의 리스트에 저장
    metadata = []  # 각 임베딩에 대한 메타데이터 (dialogue_id, speaker, persona_info)
    
    for dialogue_file in tqdm(list(input_dir.glob('dialogue_*.json'))):
        dialogue_num = int(dialogue_file.stem.split('_')[1])
        
        # 151~200 대화만 처리
        if dialogue_num < 151 or dialogue_num > 200:
            continue
            
        print(f"\n대화 {dialogue_num} 처리 중...")
        
        # 대화 데이터 로드
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
        
        # 각 화자별로 스타일 요약 생성
        speakers = set(turn['speaker'] for turn in dialogue_data["dialogues"])
        for speaker in speakers:
            print(f"  - {speaker}의 스타일 분석 중...")
            style_summary = processor.generate_style_summary(dialogue_data, speaker)
            style_summaries[dialogue_num][speaker] = style_summary
            
            # 임베딩 생성
            embedding = processor.create_embedding(style_summary)
            embeddings.append(embedding)
            
            # 메타데이터 저장 (dialogue_id, speaker, persona_info)
            metadata.append({
                "dialogue_id": dialogue_num,
                "speaker": speaker,
                "persona_info": dialogue_data["persona_info"][speaker],
                "style_summary": style_summary
            })
    
    # FAISS 인덱스 생성
    print("\n[2/3] FAISS 인덱스 생성 중...")
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # 결과 저장
    print("\n[3/3] 결과 저장 중...")
    
    # 스타일 요약 저장
    summary_file = output_dir / f"{args.type}_style_summaries.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(dict(style_summaries), f, ensure_ascii=False, indent=2)
    
    # FAISS 인덱스 저장
    index_file = output_dir / f"{args.type}_style_index.faiss"
    faiss.write_index(index, str(index_file))
    
    # 메타데이터 저장
    metadata_file = output_dir / f"{args.type}_style_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n처리 완료!")
    print(f"총 {len(embeddings)}개의 스타일 임베딩 생성됨")
    print(f"결과가 {output_dir}에 저장됨")

if __name__ == "__main__":
    main() 