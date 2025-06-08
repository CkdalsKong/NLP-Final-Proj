#!/bin/bash

# 메소드와 대화 유형 배열 정의
methods=("a1 a2 a3 a4")
types=("slang" "non_slang")

# 각 메소드와 대화 유형에 대해 실행
for method in "${methods[@]}"; do
    for type in "${types[@]}"; do
        echo "=== Running $method with $type dialogues ==="
        python vLLM_inference.py --method $method --type $type --metric m1
        echo "=== Completed $method with $type dialogues ==="
        echo ""
    done
done

echo "All predictions completed!" 