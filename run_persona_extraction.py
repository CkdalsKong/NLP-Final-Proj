import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
model = SentenceTransformer("all-MiniLM-L6-v2")

def llm_persona_prompt(name, utterances):
    joined = "\n".join(utterances)
    prompt = f"""
다음은 어떤 사람의 대화 내용입니다. 이 사람의 나이대, 성별, 성격, 취향 등을 추론하세요.

[대화]
{joined}

[출력 형식 예시]
나이대: 20대, 성별: 여성, 성격: 외향적, 취향: 트렌디한 것
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def run_a1(dialogue):
    utterances = [f'{t["speaker"]}: {t["text"]}' for t in dialogue]
    return llm_persona_prompt("Full Dialogue", utterances)


def run_a2(dialogue, speaker):
    speaker_utterances = [t["text"] for t in dialogue if t["speaker"] == speaker]
    return llm_persona_prompt(speaker, speaker_utterances)


def run_a3(dialogue, speaker, chunk_size=3):
    chunks = []
    cur_chunk = []
    for turn in dialogue:
        if turn["speaker"] == speaker:
            cur_chunk.append(turn["text"])
        if len(cur_chunk) == chunk_size:
            chunks.append(cur_chunk)
            cur_chunk = []
    if cur_chunk:
        chunks.append(cur_chunk)

    results = []
    for chunk in chunks:
        result = llm_persona_prompt(speaker, chunk)
        results.append(result)
    return "\n\n".join(results)


def run_a4_rag(dialogue_id, speaker, style_summary_path, embedding_path, index_path, metadata_path):
    # 1. style summary 로드
    style_data = {}
    with open(style_summary_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["dialogue_id"] == dialogue_id and d["speaker"] == speaker:
                style_data = d
                break

    if not style_data:
        return f"❌ style summary not found for {speaker}"

    query = style_data["style_summary"]
    embedding = model.encode([query])

    # 2. FAISS 검색
    index = faiss.read_index(index_path)
    D, I = index.search(embedding, k=3)

    # 3. metadata 검색 결과 반환
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = [json.loads(l) for l in f]
    retrieved = [metadata[i] for i in I[0]]

    # 4. prompt 구성
    retrieved_str = "\n".join([
        f"- 나이대: {r.get('age', '?')}, 성격: ?, 슬랭: {', '.join(r['slang_used'])}, 요약: {r['style_summary']}"
        for r in retrieved
    ])

    prompt = f"""
다음은 화자 {speaker}의 대화 스타일과 유사한 페르소나 예시입니다.
이들을 참고하여 {speaker}의 페르소나를 추론하세요.

[Retrieved Examples]
{retrieved_str}

[Output format]
나이대: ..., 성별: ..., 성격: ..., 취향: ...
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def run_all_methods(dialogue_file: str, base_dir: str):
    dialogue_id = Path(dialogue_file).parent.name
    with open(dialogue_file, "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    speakers = list(set(t["speaker"] for t in dialogue))
    results = {}

    for speaker in speakers:
        print(f"\n🧑‍💬 추론 중: {speaker}")
        results[speaker] = {
            "A1_long_context": run_a1(dialogue),
            "A2_speaker_only": run_a2(dialogue, speaker),
            "A3_chunk_based": run_a3(dialogue, speaker),
            "A4_rag_based": run_a4_rag(
                dialogue_id=dialogue_id,
                speaker=speaker,
                style_summary_path=f"{base_dir}/style_summary.jsonl",
                embedding_path=f"{base_dir}/style_embeddings.npy",
                index_path=f"{base_dir}/style_faiss.index",
                metadata_path=f"{base_dir}/style_metadata.jsonl"
            )
        }

    # 저장
    out_path = Path(base_dir) / "persona_extraction_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 모든 방식 추론 결과 저장 완료 → {out_path}")
