# RAG Book Recommendation System

도서 추천을 위한 고급 RAG(Retrieval-Augmented Generation) 시스템입니다. FAISS 벡터 스토어와 HuggingFace 임베딩을 사용하여 사용자 질의에 맞는 도서를 추천합니다.

## 주요 기능

### 기본 기능
- CSV 파일에서 도서 데이터 로드 및 처리
- FAISS 벡터 스토어 지원
- RAG 기반 도서 추천 에이전트
- 대화형 모드 및 단일 쿼리 모드 지원
- 자동 벡터 스토어 저장/로드

### 고급 검색 기능 (4.3 Vector Store & Retriever)
- **MMR (Maximal Marginal Relevance)**: 검색 결과의 다양성 확보
- **메타데이터 필터링**: 카테고리, 장르 등 조건부 검색
- **베스트셀러 랭크 기반 재랭킹**: 유사도와 인기도를 결합한 스코어링
- **Adaptive Top-k**: 질의 특성에 따라 동적으로 검색 결과 개수 조정

## 프로젝트 구조

```
RAG_bookRecommend/
├── config.py                      # 설정 및 환경 변수 관리
├── document_processor.py          # 문서 로드 및 처리
├── vector_store.py                # 벡터 스토어 관리 (고급 검색 기능 포함)
├── rag_agent.py                   # RAG 에이전트
├── utils.py                       # 유틸리티 함수
├── main.py                        # 메인 실행 파일
├── test_advanced_features.py     # 고급 기능 테스트 스크립트
├── .env                          # 환경 변수 (생성 필요)
├── .env.example                  # 환경 변수 예시
└── data/
    └── combined_preprocessed.csv  # 도서 데이터
```

## 설치 방법

1. 저장소 클론 및 디렉토리 이동
```bash
cd RAG_bookRecommend
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
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

### 1. 대화형 모드 (기본)
```bash
python main.py
```

대화형 모드에서는 계속해서 질문을 입력할 수 있으며, `quit`, `exit`, 또는 `q`를 입력하면 종료됩니다.

### 2. 단일 쿼리 모드
```bash
python main.py --query "간호학 책을 추천해줘"
```

### 3. 벡터 스토어 선택
```bash
# FAISS 사용 (기본)
python main.py --store-type faiss
```

### 4. 고급 기능 비활성화
```bash
# MMR, 재랭킹, Adaptive-k 비활성화
python main.py --disable-advanced
```

### 5. 벡터 스토어 재생성
```bash
python main.py --rebuild
```

### 6. 통계만 확인
```bash
python main.py --stats-only
```

### 7. 고급 기능 테스트
```bash
python test_advanced_features.py
```

## 모듈 설명

### config.py
- 시스템 설정 및 환경 변수 관리
- API 키, 모델 이름, 경로 등의 설정 값 정의
- 환경 변수 검증 기능
- **새로운 설정**: MMR, 재랭킹, Adaptive Top-k 관련 파라미터

### document_processor.py
- CSV 파일에서 문서 로드
- 문서를 청크로 분할
- 메타데이터 추가 및 관리

### vector_store.py
- **FAISS 벡터 스토어** 생성 및 관리
- 임베딩 모델 초기화
- **기본 검색**: 유사도 검색, 스코어와 함께 검색
- **MMR 검색**: 다양성을 고려한 검색
- **메타데이터 필터링**: 조건부 검색
- **베스트셀러 재랭킹**: 유사도와 인기도 결합
- **Adaptive Top-k**: 질의 특성 기반 동적 k 조정
- **Advanced Search**: 모든 기능을 통합한 고급 검색
- 벡터 스토어 저장/로드

### rag_agent.py
- RAG 기반 에이전트 구현
- 고급 검색 기능 통합
- 컨텍스트를 활용한 응답 생성
- 대화형 모드 지원

### test_advanced_features.py
- 고급 검색 기능 테스트 및 시연
- 각 기능의 독립적 테스트
- 통합 테스트

### utils.py
- 도서 중복 제거 기능
- 결과 포맷팅
- 문서 검증 및 통계

## 주요 클래스

### Config
시스템 설정을 관리하는 클래스
```python
from config import Config

Config.setup_environment()  # 환경 변수 설정
Config.validate()           # 설정 검증
```

### DocumentProcessor
문서 처리를 담당하는 클래스
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
chunks, original_docs = processor.process()
```

### VectorStoreManager
벡터 스토어를 관리하는 클래스

#### 기본 사용
```python
from vector_store import VectorStoreManager

manager = VectorStoreManager()
vectorstore = manager.get_or_create_vectorstore(chunks)

# 기본 검색
results = manager.similarity_search("간호학", k=5)
```

#### MMR 검색 (다양성)
```python
# 다양한 결과를 얻기 위해 MMR 사용
results = manager.mmr_search(
    query="자기계발 도서",
    k=5,              # 최종 반환할 문서 수
    fetch_k=20,       # MMR 전 가져올 문서 수
    lambda_mult=0.5   # 0=최대 다양성, 1=최대 관련성
)
```

#### 메타데이터 필터링
```python
# 특정 카테고리만 검색
results = manager.similarity_search(
    query="흥미로운 이야기",
    k=3,
    filter={"구분": "소설"}
)
```

#### 베스트셀러 재랭킹
```python
# 유사도와 베스트셀러 랭크를 결합
results_with_scores = manager.similarity_search_with_score("인기 책", k=5)
reranked = manager.rerank_by_bestseller(
    results_with_scores,
    alpha=0.7,  # 유사도 가중치
    beta=0.3    # 랭크 가중치
)
```

#### Adaptive Top-k
```python
# 질의 특성에 따라 자동으로 k 결정
optimal_k = manager.adaptive_top_k(
    query="다양한 장르의 책 추천",
    min_k=2,
    max_k=10,
    threshold=0.7
)
```

#### 통합 고급 검색
```python
# 모든 기능을 한 번에 사용
results = manager.advanced_search(
    query="흥미진진한 다양한 베스트셀러",
    use_mmr=True,
    use_reranking=True,
    use_adaptive_k=True,
    filter={"구분": "소설"}
)
```

### RAGAgent
RAG 에이전트 클래스
```python
from rag_agent import RAGAgent

agent = RAGAgent(vectorstore_manager)
response = agent.query("간호학 책을 추천해줘")
```

## 설정 커스터마이징

config.py에서 다음 설정을 변경할 수 있습니다:

### 기본 설정
```python
# 임베딩 모델
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 채팅 모델
CHAT_MODEL_NAME = "google_genai:gemini-2.0-flash"

# 청크 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# 기본 검색 설정
DEFAULT_K = 2  # 기본 검색 결과 수
```

### 고급 검색 설정
```python
# 벡터 스토어 타입
VECTOR_STORE_TYPE = "faiss"  # "faiss" only

# MMR 설정
USE_MMR = True              # MMR 사용 여부
MMR_FETCH_K = 20           # MMR 전 가져올 문서 수
MMR_LAMBDA = 0.5           # 0=최대 다양성, 1=최대 관련성

# 재랭킹 설정
USE_RERANKING = True       # 재랭킹 사용 여부
RANK_ALPHA = 0.7          # 유사도 가중치
RANK_BETA = 0.3           # 랭크 가중치
RANK_COLUMN = "판매지수"   # 랭크 메타데이터 컬럼명

# Adaptive Top-k 설정
USE_ADAPTIVE_K = True      # 적응형 k 사용 여부
MIN_K = 2                 # 최소 문서 수
MAX_ADAPTIVE_K = 10       # 최대 문서 수
SIMILARITY_THRESHOLD = 0.7 # 유사도 임계값
```

## 고급 기능 상세 설명

### 1. MMR (Maximal Marginal Relevance)

MMR은 검색 결과의 관련성과 다양성을 동시에 최적화합니다.

**동작 원리**:
- 먼저 `fetch_k`개의 후보 문서를 검색
- 관련성이 높으면서도 이미 선택된 문서와 다른 문서를 반복적으로 선택
- `lambda_mult` 파라미터로 관련성과 다양성의 균형 조절

**사용 예시**:
```python
# 다양한 장르의 책을 추천받고 싶을 때
results = manager.mmr_search(
    query="재미있는 책",
    k=5,
    fetch_k=20,
    lambda_mult=0.3  # 다양성 우선
)
```

### 2. 메타데이터 필터링

특정 조건을 만족하는 문서만 검색합니다.

**사용 예시**:
```python
# 소설 카테고리에서만 검색
results = manager.similarity_search(
    query="감동적인 이야기",
    k=3,
    filter={"구분": "소설"}
)

# 여러 조건 조합
results = manager.similarity_search(
    query="베스트셀러",
    k=5,
    filter={"구분": "자기계발", "판매지수": {"$gt": 1000}}
)
```

### 3. 베스트셀러 랭크 기반 재랭킹

유사도 점수와 베스트셀러 랭크를 결합하여 최종 스코어를 계산합니다.

**수식**:
```
final_score = α × similarity_score + β × (1 / rank)
```

- `α` (alpha): 유사도 점수의 가중치 (기본: 0.7)
- `β` (beta): 랭크 점수의 가중치 (기본: 0.3)
- 랭크가 낮을수록(인기가 높을수록) 점수가 높아짐

**사용 예시**:
```python
# 유사도와 인기도를 모두 고려
results_with_scores = manager.similarity_search_with_score("추천 도서", k=10)
reranked = manager.rerank_by_bestseller(
    results_with_scores,
    alpha=0.6,  # 유사도 60%
    beta=0.4    # 인기도 40%
)
```

### 4. Adaptive Top-k 전략

질의의 특성에 따라 동적으로 검색할 문서 수를 결정합니다.

**고려 요소**:
1. **질의 길이**: 긴 질의일수록 더 많은 컨텍스트 필요
2. **질의 모호성**: "추천", "다양한" 등의 키워드 포함 시 k 증가
3. **유사도 분포**: 높은 유사도 문서가 많으면 k 증가

**동작 방식**:
```python
# 자동으로 최적의 k 값 결정
k = manager.adaptive_top_k(
    query="다양한 종류의 흥미진진한 책 추천해줘",  # 모호하고 긴 질의
    min_k=2,
    max_k=10,
    threshold=0.7
)
# k가 높게 결정됨 (예: 7-8)

k = manager.adaptive_top_k(
    query="해리포터",  # 명확하고 짧은 질의
    min_k=2,
    max_k=10,
    threshold=0.7
)
# k가 낮게 결정됨 (예: 2-3)
```

### 5. 통합 Advanced Search

모든 고급 기능을 한 번에 사용하는 통합 메서드입니다.

**처리 순서**:
1. Adaptive Top-k로 최적의 k 결정
2. MMR 또는 기본 검색으로 문서 검색
3. 베스트셀러 재랭킹 적용
4. 최종 결과 반환

**사용 예시**:
```python
results = manager.advanced_search(
    query="감동적이고 인기있는 다양한 소설",
    use_mmr=True,           # MMR 활성화
    use_reranking=True,     # 재랭킹 활성화
    use_adaptive_k=True,    # Adaptive k 활성화
    filter={"구분": "소설"}  # 소설만 검색
)
```

## 개발자 가이드

### 새로운 기능 추가

1. **새로운 임베딩 모델 사용**
   config.py에서 `EMBEDDING_MODEL_NAME` 수정

2. **다른 벡터 스토어 사용**
   vector_store.py의 `VectorStoreManager` 클래스 수정

3. **커스텀 프롬프트 추가**
   rag_agent.py의 `_create_prompt_middleware` 메서드 수정

### 에러 처리

모든 주요 기능에 에러 처리가 구현되어 있습니다:
- 파일 누락: `FileNotFoundError`
- API 키 누락: `ValueError`
- 벡터 스토어 미초기화: `ValueError`

## 라이선스

MIT License