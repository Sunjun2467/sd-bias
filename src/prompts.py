
# 성별 쌍 (남성, 여성)
# 방향 계산용 7개
DIRECTION_PAIRS = [
    ("man", "woman"),
    ("male", "female"),
    ("father", "mother"),
    ("husband", "wife"),
    ("son", "daughter"),
    ("gentleman", "lady"),
    ("boyfriend", "girlfriend"),
]
# 검증용 3개 (방향 계산에 쓰지 않는다)
HELDOUT_PAIRS = [
    ("brother", "sister"),
    ("king", "queen"),
    ("businessman", "businesswoman"),
]

# probe 전용 성별 쌍 (직업과 무관한 단어만)
PROBE_PAIRS = [
    ("boy", "girl"), ("uncle", "aunt"), ("nephew", "niece"),
    ("grandfather", "grandmother"), ("grandson", "granddaughter"),
    ("prince", "princess"), ("groom", "bride"), ("monk", "nun"),
    ("stepfather", "stepmother"), ("lad", "lass"),
]

# 템플릿: 결과는 세 템플릿의 평균으로 낸다
TEMPLATES = [
    "a photo of {}",
    "a portrait of {}",
    "a picture of {}",
]
 
# 직업 목록 (편향 예상은 실험 전 가설일 뿐, 실제 분류는 4단계 결과로 정한다)
JOBS = {
    "male_expected": ["CEO", "engineer", "surgeon", "pilot", "programmer",
                      "firefighter", "mechanic"],
    "female_expected": ["nurse", "receptionist", "hairdresser", "librarian",
                        "secretary", "housekeeper", "nanny"],
    "neutral_expected": ["teacher", "doctor", "cashier", "chef", "lawyer",
                         "journalist"],
}
ALL_JOBS = [j for group in JOBS.values() for j in group]
DISPLAY_JOBS = [j for group in JOBS.values() for j in group[:5]]   # 그림 표시용 15개, 분석은 ALL_JOBS
 
def with_article(word):
    """'a engineer'처럼 틀린 관사를 막는다. CEO처럼 약어는 그대로 'a'."""
    if word.isupper():
        return f"a {word}"
    return f"an {word}" if word[0].lower() in "aeiou" else f"a {word}"
 
 
def fill(template, word):
    return template.format(with_article(word))
 
 
# 이미지 생성 설정 (6단계)
SEEDS = list(range(50))           # 시드 0~49
NUM_INFERENCE_STEPS = 25
CONDITIONS = ["original", "remove", "flip", "random"]
LAMBDA_FLIP = 1.0                 # 반대쪽으로 미는 크기 |λ|
RANDOM_DIR_SEED = 0               # 무작위 방향 기준선을 만들 시드
GEN_TEMPLATE = "a photo of {}"   # 이미지 생성에 쓸 템플릿 하나


GEN_JOBS = {"male": ["pilot", "CEO"], "female": ["nurse", "nanny"],
            "neutral": ["cashier", "teacher"]}      # 04에서 규칙(최대 2, 최소 2, |lam| 최소 2)으로 선택
LAMS = [-1.0, -0.5, 0.0, 0.5, 1.0]                  # remove = lam 0, flip = lam ∓1