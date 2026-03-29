# bge-m3 Embedding Server

로컬에서 `BAAI/bge-m3` 임베딩을 OpenAI 호환 `/v1/embeddings` 형태로 서빙하는 용도입니다.

## 실행

```bash
cd tools/embedding-server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 확인

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/v1/embeddings ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"BAAI/bge-m3\",\"input\":\"벡터 검색\"}"
```

Spring 설정 기본값은 이미 `http://localhost:8000/v1/embeddings` 와 `BAAI/bge-m3` 를 바라보도록 맞춰져 있습니다.
