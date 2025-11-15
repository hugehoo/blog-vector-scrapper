# Blog Scraper RAG - 프로젝트 요약

## 📋 프로젝트 개요

**Blog Scraper RAG**는 블로그 포스트를 스크래핑하고 의미론적 검색(Semantic Search)을 제공하는 RAG (Retrieval Augmented Generation) 시스템입니다.

- **버전**: 2.0.0
- **개발 환경**: Python 3.13+
- **프레임워크**: FastAPI

## 🏗️ 시스템 아키텍처

```
┌─────────────┐
│  FastAPI    │ ← REST API 서버
└──────┬──────┘
       │
   ┌───┴────────────┐
   │                │
┌──▼────┐      ┌───▼────┐
│MongoDB│      │ Milvus │
│(원본) │      │(벡터DB)│
└───────┘      └────────┘
```

### 데이터 플로우

1. **스크래핑**: crawl4ai를 사용하여 블로그 URL에서 컨텐츠 추출
2. **전처리**: 마크다운 정리 및 청킹 (chunk 단위로 분할)
3. **임베딩**: Hugging Face 모델을 사용하여 텍스트를 벡터로 변환
4. **저장**:
   - MongoDB: 원본 포스트 메타데이터
   - Milvus: 벡터 임베딩
5. **검색**: 쿼리를 벡터화하여 유사한 청크 검색

## 🛠️ 기술 스택

### 백엔드
- **FastAPI** (0.121.2+): REST API 프레임워크
- **Uvicorn** (0.38.0+): ASGI 서버

### 데이터베이스
- **MongoDB** (Motor 3.3.0+): 원본 블로그 포스트 저장
- **Milvus Lite** (2.5.1+): 로컬 벡터 데이터베이스

### AI/ML
- **Sentence Transformers** (3.0.1+): 텍스트 임베딩
- **BAAI/bge-m3**: 기본 임베딩 모델 (한영 혼용 지원)

### 웹 스크래핑
- **crawl4ai** (0.7.7+): 웹 크롤링 및 콘텐츠 추출
- **httpx** (0.27.0+): 비동기 HTTP 클라이언트

### 기타
- **python-dotenv**: 환경 변수 관리
- **tqdm**: 진행률 표시

## 📂 프로젝트 구조

```
blog-scrapper-rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 애플리케이션 진입점
│   ├── config.py            # 환경 변수 설정
│   ├── models.py            # Pydantic 모델 정의
│   ├── database/
│   │   ├── mongodb.py       # MongoDB 클라이언트
│   │   └── milvus.py        # Milvus 벡터 DB 클라이언트
│   └── features/
│       ├── scraping.py      # 스크래핑 및 인덱싱 API
│       ├── embedding.py     # 임베딩 서비스
│       └── search.py        # 시맨틱 검색 API
├── pyproject.toml           # 프로젝트 의존성
├── .env                     # 환경 변수 (gitignore)
└── milvus_blog.db           # Milvus Lite 데이터베이스 파일
```

## 🎯 주요 기능

### 1. 블로그 포스트 관리

- **GET /posts**: 페이지네이션을 통한 포스트 목록 조회
- **GET /posts/count**: 전체 포스트 수 조회

### 2. 스크래핑 & 임베딩

- **POST /posts/scrape-batch**: URL 일괄 스크래핑 (테스트용)
- **POST /posts/scrape-and-embed**: 단일 포스트 스크래핑 + 임베딩 + 저장
- **POST /posts/scrape-and-embed-bulk**: 대량 포스트 처리 (배치 처리 지원)

#### 처리 파이프라인
1. MongoDB에서 포스트 가져오기
2. crawl4ai로 URL 스크래핑
3. 텍스트 정리 (코드 블록, HTML 태그 제거)
4. 청킹 (900자 기준, 250자 최소, 문장 단위 오버랩)
5. 임베딩 생성 (BAAI/bge-m3 모델)
6. Milvus에 벡터 저장

### 3. 시맨틱 검색

- **POST /search**: 자연어 쿼리로 유사 포스트 검색
  - 쿼리 임베딩 생성
  - Milvus에서 코사인 유사도 기반 검색
  - 중복 제거 (같은 문서의 여러 청크)
  - MongoDB에서 전체 메타데이터 조회
  - 결과 반환 (제목, URL, 요약, 유사도 점수)

- **GET /search/stats**: Milvus 컬렉션 통계

## ⚙️ 환경 설정

### 필수 환경 변수 (.env)

```bash
# MongoDB 설정
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=blog_db

# Hugging Face 임베딩 모델 설정
HF_EMBEDDING_MODEL=BAAI/bge-m3
HF_EMBEDDING_DEVICE=cpu
HF_EMBEDDING_NORMALIZE=true

# Milvus 설정
MILVUS_DB_PATH=./milvus_blog.db
```

## 🚀 실행 방법

### 1. 의존성 설치

```bash
uv sync
```

### 2. 서버 실행

```bash
uvicorn app.main:app --reload
```

또는

```bash
python -m uvicorn app.main:app --reload
```

### 3. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 데이터베이스 스키마

### MongoDB (blog_posts 컬렉션)
```json
{
  "_id": "ObjectId",
  "title": "string",
  "url": "string",
  "summary": "string",
  ...
}
```

### Milvus (blog_posts 컬렉션)
```
- id: VARCHAR(150) - Primary Key (format: "{doc_id}:{chunk_index}")
- doc_id: VARCHAR(100) - MongoDB ObjectId
- chunk_index: INT64 - 청크 순서
- vector: FLOAT_VECTOR - 임베딩 벡터
```

## 🔧 최근 커밋 히스토리

```
7dfc634 - convert to batch
e6a93e9 - add post-text cleaning
e224081 - change model to BAAI/bge-m3
7749f01 - project milestone 1 - ingest data to milvus
```

## 📈 성능 최적화

### 청킹 전략
- **최대 길이**: 900자
- **최소 길이**: 250자
- **오버랩**: 1문장 (문맥 보존)
- **문장 분리**: 한글/영문 혼용 지원 (`.!?。！？` 기준)

### 배치 처리
- 대량 처리 시 `batch_size` 파라미터로 동시 처리 수 조절 (기본: 10)
- 실패한 포스트는 건너뛰고 계속 진행
- 처리 시간 및 성공률 통계 제공

### 검색 최적화
- 중복 제거: 같은 문서의 여러 청크 중 가장 관련성 높은 것만 반환
- 후보 확대: `limit * 5`개 검색 후 중복 제거하여 원하는 수만큼 반환

## 🔍 특징

1. **로컬 임베딩 모델**: API 비용 없이 완전 로컬 실행
2. **한영 혼용 지원**: BAAI/bge-m3 모델로 한글/영어 모두 처리
3. **청크 수준 검색**: 전체 문서가 아닌 관련 단락 수준의 정밀 검색
4. **비동기 처리**: FastAPI + Motor로 높은 동시성 지원
5. **완전한 로컬 벡터 DB**: Milvus Lite로 별도 서버 불필요

## 📝 사용 예시

### 포스트 스크래핑 및 임베딩 생성
```http
POST http://localhost:8000/posts/scrape-and-embed-bulk
?skip=0&limit=100&batch_size=10
```

### 시맨틱 검색
```http
POST http://localhost:8000/search
Content-Type: application/json

{
  "query": "머신러닝 모델 최적화 방법",
  "limit": 5
}
```

## 🎓 학습 포인트

이 프로젝트는 다음 개념들을 실습할 수 있습니다:

- RAG (Retrieval Augmented Generation) 파이프라인 구축
- 벡터 데이터베이스 활용
- 텍스트 임베딩 및 시맨틱 검색
- 비동기 웹 스크래핑
- FastAPI를 활용한 REST API 개발
- MongoDB + Milvus 하이브리드 데이터 저장

---

**생성일**: 2025-11-15
**현재 상태**: Main 브랜치, 커밋 상태 클린
