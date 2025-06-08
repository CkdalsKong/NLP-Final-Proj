import pandas as pd
from collections import defaultdict
from itertools import combinations

# CSV 파일 읽기
df = pd.read_csv('slang_dataset/slang_sentence/combined_slang_dataset.csv')

# 연령대와 성별을 문자열로 변환
df["연령대"] = df["연령대"].astype(str)
df["성별"] = df["성별"].astype(str)

# 연령대-성별 그룹별 슬랭 수집
group_slangs = defaultdict(set)
for _, row in df.iterrows():
    group = (row["연령대"], row["성별"])
    group_slangs[group].add(row["slang"])

# 결과를 저장할 리스트
results = []

# 각 그룹별 분석
for group, slangs in group_slangs.items():
    age, gender = group
    result = {
        "연령대": age,
        "성별": gender,
        "슬랭_개수": len(slangs),
        "슬랭_목록": list(slangs)
    }
    results.append(result)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 그룹 간 겹침 분석
overlap_results = []
for (group1, slangs1), (group2, slangs2) in combinations(group_slangs.items(), 2):
    age1, gender1 = group1
    age2, gender2 = group2
    
    # 겹치는 슬랭 찾기
    common_slangs = slangs1.intersection(slangs2)
    
    overlap_result = {
        "그룹1_연령대": age1,
        "그룹1_성별": gender1,
        "그룹2_연령대": age2,
        "그룹2_성별": gender2,
        "겹치는_슬랭_개수": len(common_slangs),
        "겹치는_슬랭_목록": list(common_slangs)
    }
    overlap_results.append(overlap_result)

# 겹침 결과를 DataFrame으로 변환
overlap_df = pd.DataFrame(overlap_results)

# 결과 출력
print("\n=== 연령대-성별 그룹별 슬랭 분포 ===")
print(results_df[["연령대", "성별", "슬랭_개수"]].sort_values(["연령대", "성별"]))

print("\n=== 그룹 간 슬랭 겹침 분석 ===")
print(overlap_df[["그룹1_연령대", "그룹1_성별", "그룹2_연령대", "그룹2_성별", "겹치는_슬랭_개수"]].sort_values("겹치는_슬랭_개수", ascending=False))

# 결과를 CSV 파일로 저장
results_df.to_csv("slang_distribution_by_group.csv", index=False, encoding='utf-8-sig')
overlap_df.to_csv("slang_overlap_analysis.csv", index=False, encoding='utf-8-sig')

# 상세 분석 결과 출력
print("\n=== 상세 분석 결과 ===")
for _, row in results_df.iterrows():
    print(f"\n[{row['연령대']}, {row['성별']}] 그룹:")
    print(f"총 슬랭 개수: {row['슬랭_개수']}")
    print("슬랭 목록:", ", ".join(row['슬랭_목록']))

print("\n=== 주요 겹침 분석 ===")
for _, row in overlap_df.iterrows():
    if row['겹치는_슬랭_개수'] > 0:
        print(f"\n[{row['그룹1_연령대']}, {row['그룹1_성별']}] - [{row['그룹2_연령대']}, {row['그룹2_성별']}]")
        print(f"겹치는 슬랭 개수: {row['겹치는_슬랭_개수']}")
        print("겹치는 슬랭:", ", ".join(row['겹치는_슬랭_목록'])) 