# RAG Chatbot

재무 문서 PDF를 근거로, 수치 중심 질의응답을 수행하는 **검증 강화 RAG Chatbot**입니다.  
단순 벡터 검색 기반 RAG가 아니라, 표 복원/하이브리드 검색/재시도형 품질 평가/숫자 근거 검증까지 포함합니다.

## 핵심 기능

- PDF 업로드 후 자동 Ingestion(Chunking + Embedding + DB 저장)
- 재무 문서 특화 하이브리드 검색 (Vector + Keyword + RRF)
- Multi-Query + Retrieval 품질 재시도 루프
- 표/슬라이드/주석 분리 청킹
- 근거(reference) 검증 + 숫자 정합성 검증 + 단계적 복구
- SSE 기반 실시간 스트리밍 답변
- 세션 저장/검색/선택 삭제

## 아키텍처 개요

- **Backend**: FastAPI, SQLAlchemy Async, PostgreSQL, pgvector, pg_trgm
- **RAG Layer**
  - Ingestion: `backend/app/rag/ingest/parser.py`
  - Retrieval: `backend/app/rag/retrieve/hybrid.py`
  - Pipeline Orchestration: `backend/app/rag/pipeline.py`
  - Generation/Validation: `backend/app/rag/generate/engine.py`
- **Frontend**: React + Vite + React Query + SSE(`fetch-event-source`)

## RAG 파이프라인

### 1) Ingestion

1. `pymupdf4llm`로 페이지 단위 Markdown 추출 (실패 시 PyMuPDF fallback)
2. 표 품질 규칙 기반 감지(저품질 표/레이아웃/수치 밀도)
3. 표 fallback: **Camelot -> Tabula** 순으로 추출
4. 저품질 표 교체/누락 표 병합
5. `slide`, `table`, `note` 타입별 청킹 및 메타데이터 저장
6. OpenAI 임베딩 생성 후 pgvector 컬럼에 저장

### 2) Retrieval

1. 히스토리 요약 + standalone query 재작성
2. Multi-Query 생성
   - 원질문
   - 규칙기반 재무 확장 쿼리
   - LLM 생성 쿼리
   - 키워드 fallback
3. 하이브리드 검색
   - pgvector 코사인 검색
   - pg_trgm/ILIKE 기반 후보 + BM25 점수화
   - RRF 융합
   - 휴리스틱 재랭킹(+옵션 Cross-Encoder)
4. 품질 평가 루프
   - 1차: LLM retrieval evaluator
   - 2차 이후: cheap heuristic 우선, 필요 시 compact LLM evaluator
   - 임계값 미달 시 쿼리 재구성 후 재검색(최대 N회)
5. 선택 chunk가 표일 경우 동일 페이지 표 sibling chunk를 추가 로드

### 3) Generation

1. JSON Schema(`answer`, `references`, `confidence`) 강제 출력
2. reference를 원문 chunk와 대조 검증(테이블은 의미 기반 매칭 포함)
3. 답변 숫자 검증
   - 근거 숫자 포함 여부
   - term-number pair 정합성
   - 핵심 지표(매출액/영업이익/순이익) 하드 검증
   - 성장률은 근거 수치로 계산 가능한 경우만 허용
4. 실패 시 단계적 복구
   - repair(근거 숫자만 사용)
   - compose(reference만으로 재작성)
   - synthesize(code 기반 조립)
   - quote fallback/근거 부족 반환

## 프로젝트 구조

```text
rag-chatbot/
  backend/
    app/
      api/routes/        # chat, upload, sessions API
      rag/               # ingest, retrieve, generate, pipeline
      db/                # models, session, base
      core/              # config, logging, paths
  frontend/
    src/
      api/               # backend 호출
      hooks/             # useChat, useUploadPdf
      app/components/    # ChatInterface
  data/pdfs/             # Ingestion 대상 PDF
  docker-compose.yml     # local postgres(pgvector)
```

## 실행 방법

### 1) 사전 준비

- Python 3.11+
- Node.js 20+
- Docker / Docker Compose

### 2) DB 실행

```bash
docker compose up -d postgres
```

기본 DB 포트는 `5432`입니다.

### 3) 백엔드 실행

```bash
cp backend/.env.example backend/.env
```

`backend/.env`에서 `OPENAI_API_KEY`를 설정한 뒤 실행:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Backend: `http://localhost:8000`

### 4) 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

- Frontend: `http://localhost:5173`
- 기본 API 주소: `http://localhost:8000`

## 주요 API

- `POST /api/upload` : PDF 업로드 + 인제스트
- `POST /api/chat` : 동기 채팅
- `POST /api/chat/stream` : SSE 스트리밍 채팅
- `GET /api/sessions` : 세션 목록
- `GET /api/sessions/{session_id}/messages` : 세션 메시지
- `DELETE /api/sessions` : 전체 세션 삭제
- `DELETE /api/sessions/selected` : 선택 세션 삭제
- `GET /api/sessions/search?keyword=...` : 메시지 검색

## 주요 환경 변수 (`backend/.env`)

- `OPENAI_API_KEY`: OpenAI API 키
- `DATABASE_URL`: PostgreSQL 연결 문자열
- `CORS_ALLOW_ORIGINS`: 허용 Origin
- `AUTO_INGEST_ON_STARTUP`: 서버 시작 시 자동 인제스트
- `LLM_MODEL`, `EMBEDDING_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `VECTOR_TOP_K`, `KEYWORD_TOP_K`, `FUSED_TOP_K`, `RERANK_TOP_K`
- `MULTI_QUERY_COUNT`, `RETRIEVAL_MAX_ATTEMPTS`, `RETRIEVAL_QUALITY_THRESHOLD`
- `ENABLE_RERANKER`, `RERANKER_MODEL`

## 사용 가이드

1. 프론트에서 PDF 업로드
2. 연도/분기/사업부/지표를 명확히 포함해 질문
3. 답변의 `출처`, `confidence` 확인
4. 근거 부족 응답 시 질문을 더 구체화

예시 질문:

- `2024년 2분기 DS 부문 매출액과 영업이익을 원문 인용과 함께 알려줘`

## Troubleshooting

1. `Failed to generate response` / 500

- `OPENAI_API_KEY` 누락 또는 키 오류 확인

2. DB 연결 실패

- `docker compose ps`로 postgres 상태 확인
- `DATABASE_URL`이 `localhost:5432`와 일치하는지 확인

3. CORS 오류

- `CORS_ALLOW_ORIGINS`에 프론트 주소(`http://localhost:5173`) 포함

4. 업로드 실패

- PDF 이외 파일 업로드 여부 확인

5. 스트리밍 끊김

- `/api/chat/stream`은 SSE 기반이므로 프록시/방화벽의 SSE 차단 여부 확인
