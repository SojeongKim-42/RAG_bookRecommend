# RAG Book Recommendation System

도서 추천을 위한 RAG(Retrieval-Augmented Generation) 시스템입니다. FAISS 벡터 스토어와 HuggingFace 임베딩을 사용하여 사용자 질의에 맞는 도서를 추천합니다.

## 주요 기능

- CSV 파일에서 도서 데이터 로드 및 처리
- FAISS 벡터 스토어를 사용한 효율적인 유사도 검색
- RAG 기반 도서 추천 에이전트
- 대화형 모드 및 단일 쿼리 모드 지원
- 자동 벡터 스토어 저장/로드

## 프로젝트 구조

```
RAG_bookRecommend/
├── config.py              # 설정 및 환경 변수 관리
├── document_processor.py  # 문서 로드 및 처리
├── vector_store.py        # 벡터 스토어 관리
├── rag_agent.py          # RAG 에이전트
├── utils.py              # 유틸리티 함수
├── main.py               # 메인 실행 파일
├── .env                  # 환경 변수 (생성 필요)
├── .env.example          # 환경 변수 예시
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

### 3. 벡터 스토어 재생성
```bash
python main.py --rebuild
```

### 4. 통계만 확인
```bash
python main.py --stats-only
```

## 모듈 설명

### config.py
- 시스템 설정 및 환경 변수 관리
- API 키, 모델 이름, 경로 등의 설정 값 정의
- 환경 변수 검증 기능

### document_processor.py
- CSV 파일에서 문서 로드
- 문서를 청크로 분할
- 메타데이터 추가 및 관리

### vector_store.py
- FAISS 벡터 스토어 생성 및 관리
- 임베딩 모델 초기화
- 유사도 검색 기능
- 벡터 스토어 저장/로드

### rag_agent.py
- RAG 기반 에이전트 구현
- 컨텍스트를 활용한 응답 생성
- 대화형 모드 지원

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
```python
from vector_store import VectorStoreManager

manager = VectorStoreManager()
vectorstore = manager.get_or_create_vectorstore(chunks)
results = manager.similarity_search("간호학", k=5)
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

```python
# 임베딩 모델
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v1"

# 채팅 모델
CHAT_MODEL_NAME = "google_genai:gemini-2.0-flash-lite"

# 청크 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# 검색 설정
DEFAULT_K = 5  # 기본 검색 결과 수
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