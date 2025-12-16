# RAG Book Recommendation System

도서 추천을 위한 고급 RAG(Retrieval-Augmented Generation) 시스템입니다. FAISS 벡터 스토어와 HuggingFace 임베딩을 사용하여 사용자 질의에 맞는 도서를 추천합니다.

## 주요 기능

### 핵심 기능
- **RAG 기반 도서 추천**: 벡터 검색과 LLM을 결합한 지능형 추천
- **고급 검색 전략**: MMR, Adaptive Top-k, 베스트셀러 재랭킹
- **모호성 인식 오케스트레이터**: 불명확한 질의에 대한 자동 명료화
- **대화형 인터페이스**: CLI 및 Streamlit 웹 UI 지원

### 고급 검색 기능
- **MMR (Maximal Marginal Relevance)**: 검색 결과의 다양성 확보
- **메타데이터 필터링**: 장르, 카테고리 등 조건부 검색
- **베스트셀러 재랭킹**: 유사도와 인기도를 결합한 스코어링
- **Adaptive Top-k**: 질의 특성에 따라 동적으로 검색 결과 개수 조정

## 프로젝트 구조

```
RAG_bookRecommend/
├── src/                           # 핵심 애플리케이션 코드
│   ├── config.py                  # 설정 및 환경 변수 관리
│   ├── utils.py                   # 유틸리티 함수
│   ├── core/                      # 핵심 RAG 컴포넌트
│   │   ├── rag_agent.py           # RAG 에이전트
│   │   ├── orchestrator.py        # 모호성 인식 오케스트레이터
│   │   └── chains.py              # LLM 체인 (명료화, 재작성 등)
│   └── data/                      # 데이터 처리
│       ├── vector_store.py        # 벡터 스토어 관리 (고급 검색)
│       └── document_processor.py  # 문서 로드 및 처리
├── evaluation/                    # 평가 및 실험
│   ├── evaluate.py                # 실험 실행기
│   ├── metrics.py                 # 평가 메트릭
│   ├── dataset.py                 # 평가 데이터셋
│   ├── utils.py                   # 평가 유틸리티
│   ├── viz.py                     # 시각화 도구
│   └── experiment_config.py       # 실험 설정
├── main.py                        # CLI 실행 파일
├── streamlit_app.py               # Streamlit 웹 UI
├── data/                          # 데이터 파일
│   └── combined_preprocessed.csv  # 도서 데이터
└── requirements.txt               # 패키지 의존성
```

## 설치 방법

1. **저장소 클론**
```bash
git clone <repository-url>
cd RAG_bookRecommend
```

2. **패키지 설치**
```bash
pip install -r requirements.txt
```

3. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일을 열어 API 키 입력
```

`.env` 파일 예시:
```
GOOGLE_API_KEY=your_google_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

## 사용 방법

### 1. CLI 모드

#### 대화형 모드 (기본)
```bash
python main.py
```

#### 단일 쿼리 모드
```bash
python main.py --query "SF 소설 추천해줘"
```

#### 벡터 스토어 재생성
```bash
python main.py --rebuild
```

#### 통계 확인
```bash
python main.py --stats-only
```

### 2. Streamlit 웹 UI

```bash
streamlit run streamlit_app.py
```

웹 인터페이스에서 다음 기능을 사용할 수 있습니다:
- 실시간 대화형 추천
- RAG Agent / Orchestrator 전환
- 검색 파라미터 동적 조정 (MMR, Adaptive-k 등)
- 대화 히스토리 관리

### 3. 평가 및 실험

#### 사전 정의된 실험 실행
```bash
python evaluation/evaluate.py --preset baseline
python evaluation/evaluate.py --preset ablation
python evaluation/evaluate.py --preset k_sweep
```

#### 커스텀 실험
```bash
python evaluation/evaluate.py --custom --k 5 --mmr-lambda 0.8
```

## 핵심 컴포넌트

### RAGAgent
기본 RAG 에이전트로, 벡터 검색과 LLM을 결합하여 도서를 추천합니다.

```python
from src.core.rag_agent import RAGAgent
from src.data.vector_store import VectorStoreManager

vectorstore_manager = VectorStoreManager()
agent = RAGAgent(vectorstore_manager)
response = agent.query("감동적인 소설 추천해줘")
```

### AmbiguityAwareOrchestrator
모호한 질의를 감지하고 자동으로 명료화하는 고급 오케스트레이터입니다.

```python
from src.core.orchestrator import AmbiguityAwareOrchestrator

orchestrator = AmbiguityAwareOrchestrator(vectorstore_manager)
result = orchestrator.process_query("재미있는 책 추천해줘")
```

**주요 기능**:
- 모호성 감지 (장르 모호성, 감정 기반, 상황 기반 등)
- 질의 재작성
- 검색 품질 평가
- 자동 명료화 질문 생성

### VectorStoreManager
FAISS 기반 벡터 스토어를 관리하며 다양한 고급 검색 기능을 제공합니다.

#### 기본 검색
```python
from src.data.vector_store import VectorStoreManager

manager = VectorStoreManager()
results = manager.similarity_search("간호학", k=5)
```

#### MMR 검색 (다양성)
```python
results = manager.mmr_search(
    query="자기계발 도서",
    k=5,
    fetch_k=20,
    lambda_mult=0.8  # 0=최대 다양성, 1=최대 관련성
)
```

#### 메타데이터 필터링
```python
results = manager.similarity_search(
    query="흥미로운 이야기",
    k=3,
    filter={"구분": "소설"}
)
```

#### 통합 고급 검색
```python
results = manager.advanced_search(
    query="감동적인 베스트셀러",
    use_mmr=True,
    use_reranking=True,
    use_adaptive_k=True,
    filter={"구분": "소설"}
)
```

## 설정 커스터마이징

`src/config.py`에서 다음 설정을 변경할 수 있습니다:

### 모델 설정
```python
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CHAT_MODEL_NAME = "openai:gpt-5-mini"
CHAIN_MODEL_NAME = "google_genai:gemini-2.0-flash-lite"
```

### 검색 파라미터
```python
# MMR 설정
USE_MMR = True
MMR_LAMBDA = 0.8

# 재랭킹 설정
USE_RERANKING = True
RANK_ALPHA = 0.8
RANK_BETA = 0.2

# Adaptive Top-k 설정
USE_ADAPTIVE_K = True
MIN_K = 3
MAX_ADAPTIVE_K = 10
SIMILARITY_THRESHOLD = 0.5
```

## 평가 시스템

### 평가 메트릭
- **장르 정확도**: Precision, Recall, F1 Score
- **장르 다양성**: 검색 결과의 장르 분포
- **의미 유사도**: 쿼리와 검색 결과 간 코사인 유사도

### 실험 프리셋
- `baseline`: 기본 설정
- `ablation`: 각 기능의 영향 분석
- `k_sweep`: k 값 변화에 따른 성능
- `lambda_sweep`: MMR lambda 파라미터 튜닝
- `orchestrator`: 오케스트레이터 비교

### 결과 분석
평가 결과는 `experiment_results/` 디렉토리에 저장됩니다:
- `detailed_results_*.json`: 쿼리별 상세 결과
- `aggregated_results_*.json`: 집계된 메트릭
- `summary_report_*.txt`: 요약 보고서

## 개발 가이드

### 새로운 체인 추가
`src/core/chains.py`에 새로운 LLM 체인을 추가할 수 있습니다:

```python
class CustomChain:
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model_name)
    
    def run(self, query: str) -> str:
        # 체인 로직 구현
        pass
```

### 새로운 평가 메트릭 추가
`evaluation/metrics.py`에 커스텀 평가자를 추가할 수 있습니다:

```python
class CustomEvaluator:
    def evaluate(self, test_query, retrieved_books):
        # 평가 로직 구현
        pass
```

## 문제 해결

### 벡터 스토어 로드 실패
```bash
python main.py --rebuild
```

### API 키 오류
`.env` 파일에 올바른 API 키가 설정되어 있는지 확인하세요.

### 메모리 부족
`src/config.py`에서 `CHUNK_SIZE`를 줄이거나 `DEFAULT_K`를 낮추세요.

## 라이선스

MIT License