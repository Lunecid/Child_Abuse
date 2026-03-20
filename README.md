# Child Abuse Detection & Analysis Research Pipeline (v28 Refactor)

한국 아동 상담 세션의 구조화된 JSON 기록을 처리하여, 아동 학대 유형을 분류하고
학대 유형별 차별적 어휘를 식별하는 **학술 논문용 NLP 분석 파이프라인**입니다.

---

## 1. 프로젝트 목적

| 목적 | 설명 |
|------|------|
| **정서군 분류** | 아동을 부정/평범/긍정 정서군으로 분류 (규칙 기반) |
| **학대유형 식별** | 성학대/신체학대/정서학대/방임 4대 유형을 점수 기반으로 판별 |
| **차별적 어휘 탐색** | Log-odds, χ², Correspondence Analysis로 학대유형별 핵심 어휘 추출 |
| **교량 단어(Bridge Words) 탐지** | 여러 학대유형에 걸쳐 출현하는 공유 어휘 식별 |
| **ML 기반 학대유형 예측** | TF-IDF + 전통 ML (LR/RF/SVM) 및 KLUE-BERT 파인튜닝 분류기 비교 |
| **논문용 산출물 생성** | CSV 통계표, 워드클라우드, 레이더 차트, CA 바이플롯 등 |

---

## 2. 전체 파이프라인 흐름

```
run_v28_refactor.py
 │
 ├─ run_pipeline(only_negative=False)  →  ver28_all/
 └─ run_pipeline(only_negative=True)   →  ver28_negOnly/
      │
      ├─ Stage 0: Doc-level 라벨 셔플 퍼뮤테이션 + 브릿지 부트스트랩
      ├─ Stage A: 아동별 발화 추출 → 형태소 분석 (Okt) → 토큰화
      ├─ Stage B: 정서군/학대유형 분포 메타 테이블 생성
      ├─ Stage 1: 정서군 라벨 타당성 검증 (ANOVA, t-test, χ²)
      ├─ Stage 2: 문항별 레이더 차트
      ├─ Stage 3: 토큰-레벨 정서군 분석 (WordCloud, log-odds, χ², HHI)
      ├─ Stage 4: 토큰-레벨 학대유형 분석 (WordCloud, log-odds, χ², Bridge, CA)
      ├─ Stage 5: 워드 임베딩 (Word2Vec/FastText) 학습 + CA 공간 비교
      ├─ BERT-CA: BERT 문맥 임베딩 CA 검증 (Procrustes + Mantel)
      ├─ Stage 6: TF-IDF + Logistic Regression 다항 분류
      ├─ Stage 7: 전처리 예시 저장 (토큰화 샘플, 불용어 목록)
      ├─ Stage 8: 빈도 매칭 베이스라인 (Bridge ablation)
      ├─ Stage 9: Main+Sub 학대유형 통합 분석 (NEG_ONLY 전용)
      └─ Stage 10: 다중라벨 vs 단일라벨 분류기 비교 (NEG_ONLY 전용)
```

---

## 3. 실행 방법

### 전체 파이프라인
```bash
python run_v28_refactor.py [--data_dir /path/to/json/files]
```
- `data/` 폴더의 `*.json` 파일을 읽음
- ALL → `ver28_all/`, NEG_ONLY → `ver28_negOnly/`에 산출물 저장

### 독립 실행 스크립트
```bash
# 다중라벨 vs 단일라벨 분류기 비교
python run_neg_gt_multilabel.py --data_dir /path/to/data [--skip_bert]

# Soft-label vs Single-label 분석
python run_softlabel_vs_singlelabel.py --data_dir /path/to/data

# 임계값 통계 추출 (Appendix용)
python extract_threshold_statistics.py

# GT-Algorithm Gap 진단
python run_gt_alg_gap_diagnosis.py --data_dir /path/to/data

# 부정군 반박 메트릭
python run_abuse_neg_rebuttal_metrics.py --data_dir /path/to/data
```

---

## 4. ML 모델 상세 설정

### 4.1 TF-IDF 하이퍼파라미터 (전역 공유)

`core/common.py`에서 `TFIDF_PARAMS`로 전역 정의되며, 모든 분류기 모듈에서 동일하게 참조합니다.

```python
TFIDF_PARAMS = dict(
    tokenizer    = str.split,      # 사전 토큰화된 텍스트를 공백 분리
    preprocessor = None,           # 별도 전처리 없음 (이미 Okt 처리됨)
    token_pattern= None,           # custom tokenizer 사용 시 반드시 None
    ngram_range  = (1, 2),         # 유니그램 + 바이그램
    min_df       = 2,              # 최소 2개 문서에 출현해야 피처로 포함
    max_features = 20000,          # 최대 피처 수 상한
)
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `tokenizer` | `str.split` | Okt로 이미 형태소 분석된 텍스트를 공백 기준 분리 |
| `preprocessor` | `None` | sklearn 내부 전처리 비활성화 |
| `token_pattern` | `None` | custom tokenizer 사용 시 필수 None 설정 |
| `ngram_range` | `(1, 2)` | 유니그램과 바이그램을 모두 피처로 사용 |
| `min_df` | `2` | 1개 문서에만 등장하는 토큰 제외 (노이즈 방지) |
| `max_features` | `20000` | 상위 20,000개 피처만 유지 (차원 축소) |

**Fallback**: 훈련 데이터가 적어 `min_df=2`로 피처가 0개가 되면 자동으로 `min_df=1`로 완화합니다.

### 4.2 TF-IDF 기반 분류기 3종

| 분류기 | sklearn 클래스 | 주요 하이퍼파라미터 |
|--------|---------------|-------------------|
| **LR** (Logistic Regression) | `LogisticRegression` | `solver="lbfgs"`, `max_iter=300` |
| **RF** (Random Forest) | `RandomForestClassifier` | `n_estimators=200`, `max_depth=None` |
| **SVM** (Linear SVM) | `LinearSVC` | `max_iter=1000`, `dual="auto"` |

**다중라벨 모드** (`make_multilabel_clf`): Binary Relevance 방식으로 라벨별 이진 분류기를 독립 학습합니다.
- LR: `solver="liblinear"`, `max_iter=1000`, `class_weight="balanced"`
- RF: `n_estimators=200`, `class_weight="balanced"`
- SVM: `max_iter=1000`, `class_weight="balanced"`

**평가**: Stratified 5-Fold Cross-Validation (`random_state=42`)

### 4.3 KLUE-BERT 파인튜닝 분류기

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model_name` | `klue/bert-base` | 한국어 사전학습 BERT 모델 |
| `max_length` | `256` | 입력 토큰 최대 길이 |
| `batch_size` | `16` | 학습/추론 배치 크기 |
| `epochs` | `10` | 학습 에포크 수 |
| `learning_rate` | `2e-5` | AdamW 학습률 |
| `weight_decay` | `0.01` | AdamW 가중치 감쇠 |
| `warmup_ratio` | `0.1` | Linear warmup 비율 (전체 스텝의 10%) |
| `grad_clip` | `1.0` | 그래디언트 클리핑 최대 노름 |

**하이퍼파라미터 탐색 공간** (`bert_hyperparameter.py`):
- `learning_rate`: [1e-5, 2e-5, 5e-5]
- `epochs`: [3, 5, 10]
- 총 9가지 조합 × 5-Fold CV

### 4.4 TF-IDF + Multinomial Logistic Regression (Stage 6, plots.py)

파이프라인 Stage 6에서 정서군/학대유형에 대한 TF-IDF + 다항 로지스틱 회귀를 실행합니다.

```python
LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=300,
)
```
- TF-IDF 파라미터는 `TFIDF_PARAMS`와 동일
- sklearn `Pipeline`으로 data leakage 방지 (Fold별 TF-IDF fit)

### 4.5 Soft Label 분석 (revision_v2.py)

```python
TfidfVectorizer(
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    lowercase=False,
    max_features=<parameterized>,
    min_df=<parameterized>,
)
```
- 학대유형별 합산점수를 확률분포(soft label)로 변환
- 다중출력 Ridge 회귀로 soft label 예측
- KL divergence, Brier score, Cosine similarity로 평가

---

## 5. 한국어 NLP 전처리

### 형태소 분석 (core/text.py)

| 항목 | 설정 |
|------|------|
| 토크나이저 | `konlpy.Okt` (형태소 분석 + 어간 추출) |
| POS 필터 | 명사(Noun), 동사(Verb), 형용사(Adjective)만 유지 |
| 부정 바이그램 | "안/못 + 동사" → "안_먹다" 형태로 결합 |
| 불용어 | `STOPWORDS_BASE`에 정의된 도메인 특화 불용어 |
| 최소 길이 | 1글자 토큰 제외 |
| Fallback | Okt 미설치 시 공백 토큰화로 대체 |

---

## 6. 핵심 임계값 및 상수

### 아동 분류 임계값

| 상수 | 값 | 용도 |
|------|-----|------|
| `high_total_thresh` | 45 | 합계점수 → 강제 부정군 |
| `low_total_thresh` | 10 | 합계점수 → 긍정군 (다른 신호 없을 때) |
| `risk_strong` / `risk_weak` | 25 / 10 | 위험점수 임계값 |
| 학대 main 임계값 | >6 | 주 학대유형 판정 |
| `sub_threshold` | 4 | 부 학대유형 판정 |

### 분석 임계값

| 상수 | 값 | 용도 |
|------|-----|------|
| `MIN_TOTAL_COUNT_VALENCE` | 10 | 정서군 분석 최소 토큰 빈도 |
| `MIN_TOTAL_COUNT_ABUSE` | 8 | 학대유형 분석 최소 토큰 빈도 |
| `MIN_DOC_COUNT` | 5 | Doc-level 분석 최소 문서 빈도 |
| `DEFAULT_ZMIN` | 1.96 | Z-score 임계값 (α=0.05) |

### Bridge Word 임계값 (3가지 구성)

| 구성 | `min_p1` | `min_p2` | `max_gap` | 설명 |
|------|----------|----------|-----------|------|
| **B0_baseline** | 0.40 | 0.25 | 0.20 | 기본 설정 |
| **B_loose** | 0.30 | 0.20 | 0.25 | 완화 설정 |
| **B_strict** | 0.45 | 0.30 | 0.20 | 엄격 설정 |

---

## 7. 레포지토리 구조

```
Child_Abuse_Lab/
├── run_v28_refactor.py                    # 메인 진입점: 전체 파이프라인 (ALL + NEG_ONLY)
├── run_neg_gt_multilabel.py               # 다중라벨 vs 단일라벨 분류기 비교
├── run_softlabel_vs_singlelabel.py        # Soft-label vs Single-label 분석
├── run_gt_alg_gap_diagnosis.py            # GT-Algorithm Gap 진단
├── run_abuse_neg_rebuttal_metrics.py      # 부정군 반박 메트릭
├── extract_threshold_statistics.py        # 임계값 통계 추출
├── data/                                  # 입력 JSON 파일 (비공개)
│
└── abuse_pipeline/                        # 메인 Python 패키지
    ├── __init__.py                        # 역호환 re-export + sys.modules shim
    ├── pipeline.py                        # 오케스트레이터: 10-stage 파이프라인
    │
    ├── core/                              # 핵심 유틸리티
    │   ├── common.py                      # 전역 설정, 상수, 임계값, TFIDF_PARAMS
    │   ├── labels.py                      # 정서군 분류, 학대유형 판별
    │   ├── text.py                        # 발화 추출, 형태소 분석, 토큰화
    │   └── plots.py                       # 시각화 (레이더 차트, TF-IDF 로짓)
    │
    ├── data/                              # 데이터 처리
    │   ├── counting.py                    # GT 기반 학대유형 카운팅
    │   ├── doc_level.py                   # 문서-레벨 빈도표, 퍼뮤테이션, 부트스트랩
    │   └── embedding.py                   # Word2Vec / FastText 학습 및 투영
    │
    ├── stats/                             # 통계 분석
    │   ├── stats.py                       # HHI, χ², log-odds, BH-FDR, Bridge 탐지
    │   ├── ca.py                          # Correspondence Analysis + Bridge overlay
    │   ├── bridge_threshold_justification.py  # Bridge 임계값 3-전략 검증
    │   ├── contextual_embedding_ca.py     # BERT CA 검증 (Procrustes + Mantel)
    │   └── weighted_ca_extension.py       # 가중 CA (부-학대유형 기여)
    │
    ├── classifiers/                       # ML 분류기
    │   ├── classifier_utils.py            # TF-IDF/BERT 공용 유틸리티 (팩토리, fold 함수)
    │   ├── tfidf_vs_bert_comparision.py   # TF-IDF 3종 vs KLUE-BERT 비교
    │   ├── neg_gt_multilabel_analysis.py  # 다중라벨 vs 단일라벨 비교
    │   ├── softlabel_vs_singlelabel_analysis.py  # Soft-label 분석
    │   ├── bert_hyperparameter.py         # BERT 하이퍼파라미터 그리드 서치
    │   └── bert_abuse_coloring.py         # BERT 단어-레벨 학대유형 컬러링
    │
    ├── analysis/                          # 확장 분석
    │   ├── compare_abuse_labels.py        # GT vs 알고리즘 라벨 비교
    │   ├── label_comparsion_analysis.py   # 라벨 비교 통계
    │   ├── main_sub_abuse_analysis.py     # 주+부 학대유형 분석
    │   ├── integrated_label_bridge_analysis.py  # 7-stage 통합 분석
    │   ├── main_threshold_sensitivity.py  # 주 학대유형 임계값 민감도
    │   ├── sub_threshold_sensitivity.py   # 부 학대유형 임계값 민감도
    │   ├── gt_alg_gap_diagnosis.py        # GT-Algorithm Gap 진단
    │   └── abuse_neg_rebuttal_metrics.py  # 부정군 반박 메트릭
    │
    ├── investigation/                     # 탐색적 진단
    │   ├── borderline_case_explorer.py    # 경계 사례 탐색
    │   └── no_gt.py                       # GT 없는 사례 분석
    │
    └── revision/                          # 논문 수정 확장
        ├── revision_extensions.py         # 임계값 민감도, FDR, 다중 분류기
        └── revision_v2.py                 # Appendix: soft label, 전처리 강건성, GT 편향
```

---

## 8. 산출물 구조

```
ver28_all/ (또는 ver28_negOnly/)
├── meta/                      # 전처리 예시, 불용어 목록, 분포표
├── valence/
│   ├── stats/                 # Log-odds, χ², HHI, TF-IDF logit 결과 CSV
│   └── figures/               # 레이더 차트, 워드클라우드 이미지
├── abuse/
│   ├── stats/                 # 학대유형 통계, Bridge word 목록, 빈도매칭 기준선
│   └── figures/               # 워드클라우드, CA 바이플롯
├── embeddings/                # Word2Vec/FastText 모델 + 투영 결과
├── ca_prob/                   # Correspondence Analysis 산출물
├── pbridge_ablation/          # Bridge word ablation 연구
├── delta_bridge/              # Log-odds 기반 Bridge ablation
├── bert_ca_validation/        # BERT CA + Procrustes 검증
├── model_comparison/          # TF-IDF vs BERT 분류 성능 비교
├── revision/                  # 논문 수정용 산출물
└── neg_gt_multilabel/         # 다중라벨 분류기 비교 (NEG_ONLY 전용)
```

---

## 9. 입력 데이터 형식

아동 1명당 1개 JSON 파일:

```json
{
  "info": {
    "ID": "...", "합계점수": 30, "위기단계": "상담필요",
    "성별": "...", "나이": "...", "학년": "...",
    "임상진단": "...", "임상가 종합소견": "...", "학대의심": "..."
  },
  "list": [
    {
      "문항": "학대여부", "문항합계": 12,
      "list": [
        {
          "항목": "정서학대", "점수": 5,
          "audio": [
            {"type": "A", "text": "아동 발화 응답"},
            {"type": "Q", "text": "상담사 질문"}
          ]
        }
      ]
    }
  ]
}
```

- `info.위기단계`: 응급, 위기아동, 학대의심, 상담필요, 관찰필요, 정상군
- `audio[].type`: `"A"` = 아동 답변, `"Q"` = 상담사 질문

---

## 10. 의존성

### 필수
- Python 3.9+, `numpy`, `pandas`, `matplotlib`

### 선택 (미설치 시 해당 기능 건너뜀)
| 패키지 | 용도 |
|--------|------|
| `scipy` | 통계 검정 (ANOVA, χ², Spearman) |
| `scikit-learn` | TF-IDF, 분류기 (LR/RF/SVM), 교차 검증 |
| `konlpy` | 한국어 형태소 분석 (Okt) |
| `gensim` | Word2Vec / FastText |
| `prince` | Correspondence Analysis |
| `transformers` + `torch` | KLUE-BERT 파인튜닝 |
| `wordcloud` | 워드클라우드 시각화 |
| `squarify` | 트리맵 시각화 |
| `adjustText` | 텍스트 오버랩 방지 |
| `statsmodels` | 추가 통계 분석 |
| `deep_translator` | 한→영 번역 |

---

## 11. 재현성 보장

| 항목 | 설정 |
|------|------|
| Random seed | `42` (전체 파이프라인 통일) |
| Tie-breaking | `SEVERITY_RANK` (아동 안전 우선 위계) |
| 정렬 | `kind="mergesort"` (안정 정렬) |
| CSV 인코딩 | `utf-8-sig` (Excel 한글 호환) |
| Matplotlib 백엔드 | `"Agg"` (비대화형, 파일 출력 전용) |
