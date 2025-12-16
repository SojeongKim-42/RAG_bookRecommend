# RAG 도서 추천 시스템 - 평가 및 실험 가이드

## 📌 개요

이 문서는 RAG 도서 추천 시스템의 성능을 체계적으로 평가하고 다양한 파라미터 조합을 실험할 수 있는 통합 가이드입니다.

### 주요 기능

- **체계적 평가**: 30개의 다양한 테스트 쿼리로 장르 적합도, 검색 품질 측정
- **배치 실험**: 여러 설정을 한 번에 실험하고 자동으로 비교
- **시각화**: 실험 결과를 그래프로 비교하여 최적 설정 도출
- **Orchestrator 통합**: 모호한 쿼리 처리 능력 평가

---

## 🏗️ 시스템 구조

### 핵심 컴포넌트

```
평가 시스템
├── evaluation_dataset.py       # 30개의 테스트 쿼리 데이터셋
├── evaluation_metrics.py       # 장르 적합도, 검색 품질 메트릭
├── run_evaluation.py          # 단일 평가 실행
└── evaluation_viz.py          # 결과 시각화

실험 시스템
├── experiment_config.py       # 실험 설정 정의 (Presets 포함)
├── run_experiments.py         # 배치 실험 실행
└── compare_experiments.py     # 실험 결과 비교 및 시각화

Orchestrator (모호성 처리)
├── orchestrator.py           # 상태 머신 기반 Orchestrator
└── chains.py                 # 전문화된 LLM Chain들
```

### 파일 간 관계

```
run_evaluation.py
    ↓ uses
evaluation_dataset.py + evaluation_metrics.py
    ↓ produces
evaluation_results/*.json

run_experiments.py
    ↓ uses
experiment_config.py + run_evaluation.py
    ↓ produces
experiment_results/comparison_*.json

compare_experiments.py
    ↓ uses
experiment_results/comparison_*.json
    ↓ produces
comparison_viz/*.png
```

---

## 🚀 빠른 시작

### 1. 환경 준비

```bash
# 환경 활성화
eval "$(mamba shell hook --shell bash)" && mamba activate bkms

# Vector store 확인 (필요시 생성)
python main.py --stats-only
```

### 2. 첫 평가 실행 (3분 소요)

```bash
# 샘플 5개로 빠르게 테스트
python run_evaluation.py --sample 5 --k 5
```

### 3. 첫 실험 실행 (5분 소요)

```bash
# Baseline 실험
python run_experiments.py --preset baseline --sample 5
```

### 4. 결과 확인

```bash
# 결과 파일 확인
ls -lt evaluation_results/
ls -lt experiment_results/

# 시각화 생성
python compare_experiments.py \
  --comparison-file experiment_results/comparison_*.json
```

---

## 📊 평가 시스템 (Evaluation)

### 평가 데이터셋

**30개의 다양한 테스트 쿼리** 포함:

- **SPECIFIC (5개)**: "SF 소설 추천해줘", "마케팅 관련 실용서 필요해"
- **EMOTIONAL (5개)**: "요즘 너무 우울해", "무기력한데 동기부여 받고 싶어"
- **SITUATIONAL (5개)**: "군대 가기 전에 읽을 책", "출퇴근할 때 읽을만한 거"
- **VAGUE (5개)**: "인생에 도움되는 책", "재밌는 책 추천"
- **MULTI_INTENT (5개)**: "재밌고 의미 있는 소설", "가볍게 읽히면서 생각할 거리를 주는 책"
- **기타 (5개)**: 다양한 추가 케이스

각 쿼리는 다음 정보를 포함:
- 기대 장르 (ground truth)
- 기대 테마/분위기 (선택적)
- 관련 도서 리스트 (선택적)

### 평가 메트릭

#### 장르 적합도 메트릭 (GenreEvaluator)

```python
- Genre Precision: 검색된 책 중 올바른 장르 비율 (0-1)
- Genre Recall: 기대 장르를 얼마나 커버했는가 (0-1)
- Genre F1 Score: Precision과 Recall의 조화 평균
- Genre Diversity: 장르 다양성 (unique genres / total books)
```

**해석 가이드:**
- **Precision >= 0.8**: 매우 좋음 - 검색 결과가 정확
- **Recall >= 0.8**: 매우 좋음 - 기대 장르를 잘 커버
- **F1 Score >= 0.7**: 전반적으로 우수한 성능
- **Diversity >= 0.7**: 매우 다양함

#### 검색 품질 메트릭 (RetrievalEvaluator)

```python
- Precision@K: 상위 K개 중 관련 문서 비율
- Recall@K: 전체 관련 문서 중 상위 K개에 포함된 비율
- MRR (Mean Reciprocal Rank): 첫 번째 관련 문서 순위의 역수
- Coverage: 기대 도서를 얼마나 검색했는가
```

#### 의미 유사도 메트릭 (SemanticEvaluator)

```python
- Average Similarity: 쿼리와 검색 결과 간 평균 코사인 유사도
- Max Similarity: 최고 유사도 (가장 관련성 높은 결과)
- Min Similarity: 최저 유사도
```

### 평가 실행 방법

#### 기본 사용법

```bash
# 전체 데이터셋 평가 (k=5)
python run_evaluation.py --k 5

# 특정 쿼리 타입만 평가
python run_evaluation.py --query-type emotional --k 3

# 샘플 평가 (빠른 테스트용)
python run_evaluation.py --sample 10 --k 5

# 결과 저장 안 함 (콘솔 출력만)
python run_evaluation.py --no-save --sample 5
```

#### 주요 옵션

- `--k`: 검색할 문서 개수 (기본값: 5)
- `--query-type`: 평가할 쿼리 타입 (all, specific, emotional, situational, vague, multi_intent)
- `--sample`: 샘플링할 쿼리 개수
- `--output-dir`: 결과 저장 디렉토리 (기본값: evaluation_results)
- `--no-save`: 결과 파일 저장 안 함

### 평가 결과 파일

평가 실행 후 `evaluation_results/` 디렉토리에 생성:

```
evaluation_results/
├── detailed_results_{timestamp}.json     # 각 쿼리별 상세 결과
├── aggregated_results_{timestamp}.json   # 전체 평균 메트릭
└── summary_report_{timestamp}.txt        # 사람이 읽기 쉬운 요약 리포트
```

### 결과 시각화

```bash
# 종합 리포트 생성 (모든 시각화 포함)
python evaluation_viz.py \
  --results-file evaluation_results/detailed_results_TIMESTAMP.json \
  --output-dir evaluation_results

# 실패 분석 (F1 < 0.5인 쿼리 분석)
python evaluation_viz.py \
  --results-file evaluation_results/detailed_results_TIMESTAMP.json \
  --failure-threshold 0.5
```

**생성되는 시각화:**
1. `metrics_by_query_type.png`: 쿼리 타입별 메트릭 비교
2. `f1_distribution.png`: F1 스코어 분포 및 박스플롯
3. `genre_distribution.png`: 검색된 장르 분포
4. `correlation_matrix.png`: 메트릭 간 상관관계

---

## 🧪 실험 시스템 (Experiments)

### 실험 설정 (ExperimentConfig)

실험은 다음 요소들의 조합으로 정의됩니다:

#### RetrievalConfig (검색 파라미터)

```python
k: int = 5                    # 검색 문서 수
use_mmr: bool = True          # MMR 다양성 검색
mmr_lambda: float = 0.8       # 0=max diversity, 1=max relevance
use_reranking: bool = True    # 베스트셀러 기반 재정렬
rank_alpha: float = 0.8       # 유사도 가중치
rank_beta: float = 0.2        # 베스트셀러 순위 가중치
use_adaptive_k: bool = True   # 적응형 K
min_k: int = 2                # 최소 K
max_k: int = 10               # 최대 K
```

#### OrchestratorConfig (Orchestrator 설정)

```python
enabled: bool = False         # Orchestrator 사용 여부
model_name: str = None        # 최종 추천용 LLM 모델
chain_model_name: str = None  # Chain 연산용 경량 모델
verbose: bool = False         # 상세 로그
```

### 실험 Preset

실험 시스템은 자주 사용하는 설정 조합을 Preset으로 제공합니다.

#### 1. `baseline` - 기본 성능 측정

```bash
python run_experiments.py --preset baseline --sample 10
```

- MMR: ON (λ=0.8)
- Reranking: ON (α=0.8, β=0.2)
- Adaptive K: ON
- Orchestrator: OFF

**사용 목적**: 현재 시스템의 기본 성능 파악

#### 2. `orchestrator` - Orchestrator 효과 비교

```bash
python run_experiments.py --preset orchestrator --sample 10
```

- Baseline vs Orchestrator enabled 2가지 실험
- 모호한 쿼리 처리 능력 향상 확인

**사용 목적**: Orchestrator의 성능 개선 효과 측정

#### 3. `ablation` - Ablation Study (기능별 영향)

```bash
python run_experiments.py --preset ablation --sample 10
```

6개 실험:
1. Minimal (모든 기능 OFF)
2. Minimal + MMR
3. Minimal + Reranking
4. Minimal + Adaptive K
5. All features (Baseline)
6. All features + Orchestrator

**사용 목적**: 각 기능이 성능에 미치는 영향 분석

#### 4. `k_sweep` - K 값 변화

```bash
python run_experiments.py --preset k_sweep --sample 10
```

5개 실험: k = 2, 3, 5, 7, 10

**사용 목적**: 검색 문서 개수가 성능에 미치는 영향

#### 5. `lambda_sweep` - Diversity vs Relevance

```bash
python run_experiments.py --preset lambda_sweep --sample 10
```

6개 실험: λ = 0.3, 0.5, 0.7, 0.8, 0.9, 0.95

- **λ 낮음 (0.3-0.5)**: 다양성 ↑, 정확도 ↓
- **λ 높음 (0.8-0.95)**: 관련성 ↑, 다양성 ↓

**사용 목적**: 다양성과 관련성의 트레이드오프 분석

#### 6. `rerank_sweep` - Reranking 가중치 조합

```bash
python run_experiments.py --preset rerank_sweep --sample 10
```

5개 실험: (α, β) = (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)

- **α 높음**: 의미 유사도 중심
- **β 높음**: 베스트셀러 순위 중심

**사용 목적**: 유사도와 인기도의 최적 밸런스 찾기

### 실험 실행 방법

#### 기본 사용법

```bash
# Baseline 실험
python run_experiments.py --preset baseline --sample 10

# 여러 Preset 동시 실행
python run_experiments.py --preset ablation --sample 10
```

#### 고급 옵션

```bash
# 특정 쿼리 타입만 평가
python run_experiments.py \
  --preset baseline \
  --query-types emotional situational \
  --sample 10

# 출력 디렉토리 변경
python run_experiments.py \
  --preset k_sweep \
  --output-dir my_experiments \
  --sample 10

# 개별 결과 저장 생략 (비교 데이터만)
python run_experiments.py \
  --preset k_sweep \
  --no-save-individual
```

### 실험 결과 구조

```
experiment_results/
├── comparison_20251216_123456.json          # 비교 데이터
├── baseline__orchestrator_off_k_5_mmr_λ0.8/ # 각 실험 디렉토리
│   ├── config.json                          # 실험 설정
│   ├── detailed_results_*.json              # 상세 결과
│   ├── aggregated_results_*.json            # 집계 결과
│   └── summary_report_*.txt                 # 요약 리포트
└── comparison_viz/                          # 비교 시각화
    ├── overall_comparison.png               # 전체 메트릭 비교
    ├── f1_ranking.png                       # F1 순위
    ├── feature_impact.png                   # 기능별 영향
    └── tradeoff_analysis.png                # 트레이드오프 분석
```

---

## 📈 실험 결과 비교 및 시각화

### 비교 도구 사용

```bash
# 실험 결과 비교 및 시각화
python compare_experiments.py \
  --comparison-file experiment_results/comparison_TIMESTAMP.json

# 커스텀 출력 디렉토리
python compare_experiments.py \
  --comparison-file experiment_results/comparison_TIMESTAMP.json \
  --output-dir my_comparison_results
```

### 생성되는 시각화

#### 1. Overall Comparison (overall_comparison.png)

모든 실험의 Precision, Recall, F1, Diversity를 나란히 비교하는 막대 그래프

#### 2. F1 Ranking (f1_ranking.png)

F1 스코어 기준으로 실험 순위를 표시하는 수평 막대 그래프

#### 3. Feature Impact (feature_impact.png)

각 기능(MMR, Reranking, Adaptive K, Orchestrator)의 영향도를 비교

#### 4. Tradeoff Analysis (tradeoff_analysis.png)

- Precision-Recall 트레이드오프 산점도
- Diversity-F1 트레이드오프 산점도

### 콘솔 출력 예시

```
====================================================================================================
EXPERIMENT COMPARISON
====================================================================================================

Experiment                                Precision     Recall         F1  Diversity
----------------------------------------------------------------------------------------------------
orchestrator__on__adaptive_k_2-10_...         0.750      0.700      0.723      0.650
baseline__orchestrator_off_k_5_mmr_...        0.720      0.680      0.698      0.640

====================================================================================================
```

---

## 🔬 Orchestrator 통합

### Orchestrator란?

**Ambiguity-Aware Orchestrator**는 모호한 도서 추천 요청을 처리하는 상태 머신 기반 에이전트입니다.

### 주요 특징

1. **모호성 자동 감지**: "요즘 너무 공허해" 같은 감정 기반 쿼리 인식
2. **Query Rewriting**: 모호한 쿼리를 검색 최적화된 문장으로 변환
3. **품질 평가**: 검색 결과가 충분한지 자체 평가
4. **명확화 질문**: 필요시 사용자에게 추가 정보 요청 (최소화)
5. **최종 추천**: 맥락을 고려한 친근한 추천 생성

### 처리 흐름

```
User Query
    ↓
[1] Ambiguity Detection
    ↓
[2] Query Rewriting (if ambiguous)
    ↓
[3] Retrieve from Vector DB
    ↓
[4] Quality Evaluation
    ↓
[5] Clarification (if insufficient) → User Response → [3]
    ↓
[6] Final Recommendation
```

### Orchestrator 실험

```bash
# Orchestrator 유무 비교
python run_experiments.py --preset orchestrator --sample 10

# 모호한 쿼리에서만 테스트
python run_experiments.py \
  --preset orchestrator \
  --query-types emotional vague situational \
  --sample 10
```

**예상 결과:**
- Orchestrator가 모호한 쿼리(emotional, vague)에서 F1 향상
- Query rewriting으로 더 관련성 높은 문서 검색
- 처리 시간 증가 (추가 LLM 호출)

### 세부 구현

#### Chains (chains.py)

5개의 전문화된 Chain:
1. **AmbiguityDetector**: 모호성 감지 및 분류
2. **QueryRewriter**: 검색 최적화 쿼리 생성
3. **RetrieveQualityEvaluator**: 검색 결과 품질 평가
4. **ClarificationQuestionGenerator**: 명확화 질문 생성
5. **FinalRecommender**: 최종 추천 생성

#### 모델 사용 전략

성능과 비용 최적화를 위해 두 가지 모델 사용:

- **Chain 연산용 (경량)**: `gemini-2.0-flash-lite` - 빠른 판단/분류
- **최종 추천용 (고품질)**: `gemini-2.0-flash` - 높은 품질의 추천 텍스트

---

## 💡 추천 워크플로우

### 시나리오 1: 첫 평가 (새 시스템 or 변경 후)

```bash
# 1. 빠른 샘플 테스트
python run_evaluation.py --sample 5 --k 5

# 2. 전체 평가
python run_evaluation.py --k 5

# 3. 시각화
python evaluation_viz.py \
  --results-file evaluation_results/detailed_results_*.json
```

### 시나리오 2: Orchestrator 효과 검증

```bash
# 1. Baseline vs Orchestrator 비교
python run_experiments.py --preset orchestrator --sample 10

# 2. 모호한 쿼리에서 집중 테스트
python run_experiments.py \
  --preset orchestrator \
  --query-types emotional vague \
  --sample 10

# 3. 결과 비교
python compare_experiments.py \
  --comparison-file experiment_results/comparison_*.json
```

### 시나리오 3: 파라미터 튜닝

```bash
# 1. Ablation study로 중요 기능 파악
python run_experiments.py --preset ablation --sample 10

# 2. 중요 파라미터 sweep
python run_experiments.py --preset k_sweep --sample 10
python run_experiments.py --preset lambda_sweep --sample 10

# 3. 최적 조합 찾기
python compare_experiments.py \
  --comparison-file experiment_results/comparison_*.json
```

### 시나리오 4: 쿼리 타입별 최적화

```bash
# 각 쿼리 타입별로 실험
for qtype in emotional situational vague specific multi_intent; do
    python run_experiments.py \
      --preset k_sweep \
      --query-types $qtype \
      --sample 10 \
      --output-dir experiments_${qtype}
done

# 쿼리 타입별 최적 설정 분석
```

---

## 🔍 결과 해석 가이드

### Baseline vs Orchestrator 비교

**확인 사항:**
- Emotional, vague 쿼리에서 F1 개선되었는가?
- Overall F1이 일관되게 향상되었는가?
- 처리 시간 증가는 허용 가능한가?

### K 값 실험

- **K 작음 (2-3)**: Precision ↑, Recall ↓, Diversity ↓
- **K 큼 (7-10)**: Precision ↓, Recall ↑, Diversity ↑
- **최적값**: 대부분 k=5가 적절, Adaptive K로 자동 조정 권장

### MMR Lambda 실험

- **λ = 0.3-0.5 (High Diversity)**: 다양성 중시, 추천 다양화
- **λ = 0.8-0.95 (High Relevance)**: 정확도 중시, 관련성 높은 결과
- **최적값**: λ=0.7~0.8이 좋은 균형

### Reranking 가중치

- **α 높음 (0.8-0.9)**: 의미 유사도 중심
- **β 높음 (0.3-0.5)**: 베스트셀러 인기도 중심
- **추천**: α=0.7-0.8 (의미 유사도 우선)

---

## 🛠️ 고급 사용법

### 커스텀 실험 정의

Python 코드로 직접 실험 정의:

```python
from experiment_config import ExperimentConfig, RetrievalConfig, OrchestratorConfig
from run_experiments import ExperimentBatchRunner

# 커스텀 실험 정의
custom_exp = ExperimentConfig(
    name="my_experiment",
    description="High diversity with orchestrator",
    retrieval=RetrievalConfig(
        k=7,
        use_mmr=True,
        mmr_lambda=0.5,  # High diversity
        use_reranking=True,
        rank_alpha=0.9,
        use_adaptive_k=False,
    ),
    orchestrator=OrchestratorConfig(
        enabled=True,
        verbose=False,
    ),
    sample_size=20,
)

# 실험 실행
runner = ExperimentBatchRunner()
result = runner.run_single_experiment(custom_exp)
```

### 평가 데이터셋 확장

새로운 테스트 쿼리 추가 ([evaluation_dataset.py](evaluation_dataset.py)):

```python
TestQuery(
    query_id="NEW001",
    query="당신의 새로운 쿼리",
    query_type=QueryType.SPECIFIC,
    expected_genres=[GenreCategory.NOVEL],
    expected_themes=["테마1", "테마2"],
    relevant_books=["책 제목1", "책 제목2"],  # 선택적
    notes="추가 설명"
)
```

### 새로운 메트릭 추가

[evaluation_metrics.py](evaluation_metrics.py)에 새 평가자 클래스 추가:

```python
class CustomEvaluator:
    """Custom evaluation metric."""

    def evaluate(self, test_query, retrieved_books):
        # Your custom evaluation logic
        pass
```

---

## 🔧 트러블슈팅

### "Vector store not found" 에러

```bash
# Vector store 먼저 생성
python main.py --stats-only
```

### 실험이 너무 느림

```bash
# 샘플 수 줄이기
python run_experiments.py --preset baseline --sample 3

# 또는 orchestrator 제외
python run_experiments.py --preset k_sweep --sample 5
```

### 메모리 부족

```bash
# 한 번에 하나씩 실험 실행
python run_experiments.py --preset baseline --sample 5
```

### Orchestrator 관련 에러

- `AmbiguityDetector` 결과 확인: verbose=True로 실행
- `confidence` threshold 조정
- Query rewriting 품질 확인

---

## 📋 체크리스트

### 실험 시작 전

- [ ] Vector store 생성 완료
- [ ] 환경 활성화 (mamba activate bkms)
- [ ] 실험 목적 명확히 정의
- [ ] 먼저 작은 샘플로 테스트 (--sample 3~5)

### 실험 실행 중

- [ ] 로그 메시지 확인
- [ ] 오류 없이 완료되는지 체크
- [ ] 한 번에 하나의 변수만 변경 (Ablation study 활용)

### 실험 완료 후

- [ ] 결과 파일 생성 확인
- [ ] 시각화 생성 및 확인
- [ ] F1 ranking 분석
- [ ] Tradeoff 그래프로 균형점 찾기
- [ ] 쿼리 타입별 성능 차이 확인
- [ ] 최적 설정 선택 및 문서화

---

## 📝 개선 방향 제안

평가 결과를 바탕으로 다음과 같은 개선을 고려:

### 낮은 Recall

- 검색 k 값 증가
- Query rewriting 개선
- 임베딩 모델 변경

### 낮은 Precision

- Reranking 알고리즘 개선
- 메타데이터 필터링 강화
- MMR 파라미터 조정

### 낮은 Diversity

- MMR lambda 값 감소 (더 많은 다양성)
- 장르 밸런싱 전략 도입

### 특정 쿼리 타입 성능 저하

- 해당 타입에 특화된 Query rewriting
- 쿼리 타입별 검색 전략 분화
- Orchestrator 프롬프트 개선

---

## 📚 관련 파일

### 평가 시스템

- [evaluation_dataset.py](evaluation_dataset.py) - 테스트 쿼리 데이터셋
- [evaluation_metrics.py](evaluation_metrics.py) - 평가 메트릭
- [run_evaluation.py](run_evaluation.py) - 평가 실행 스크립트
- [evaluation_viz.py](evaluation_viz.py) - 시각화 도구

### 실험 시스템

- [experiment_config.py](experiment_config.py) - 실험 설정 및 Presets
- [run_experiments.py](run_experiments.py) - 배치 실험 실행
- [compare_experiments.py](compare_experiments.py) - 결과 비교 및 시각화

### Orchestrator

- [orchestrator.py](orchestrator.py) - Orchestrator 구현
- [chains.py](chains.py) - 전문화된 Chain들

### 설정

- [config.py](config.py) - 시스템 설정

---

## ✨ 요약

이 통합 문서는 RAG 도서 추천 시스템의 성능을 체계적으로 평가하고 최적의 설정을 찾기 위한 완전한 가이드입니다.

**핵심 포인트:**
1. 30개의 다양한 테스트 쿼리로 실제 사용 패턴 반영
2. 장르 적합도, 검색 품질, 의미 유사도를 다각도로 평가
3. Preset을 활용한 빠른 실험 및 비교
4. Orchestrator로 모호한 쿼리 처리 능력 향상
5. 시각화를 통한 직관적인 결과 분석

**시작하기:**
```bash
# 1. 빠른 테스트
python run_experiments.py --preset baseline --sample 5

# 2. 결과 비교
python compare_experiments.py --comparison-file experiment_results/comparison_*.json

# 3. 최적 설정 선택 및 적용
```

Happy Experimenting! 🧪✨
