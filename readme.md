# DBLP Research Assistant

A semantic search and question-answering system built on top of the DBLP computer science
bibliography dataset (~200,000 publications). The project combines unsupervised topic discovery
with a RAG pipeline that lets users ask natural language questions about CS research.

## Project Overview

1. **Data loading** — DBLP XML parsed and filtered to publications from 2018–2025
2. **EDA & Feature Engineering** — title length, author count, publication type, temporal trends
3. **Clustering** — TF-IDF → LSA (SVD, 100 dims) → L2 normalization → MiniBatchKMeans (k=20)
4. **Visualization** — UMAP 2D projection of 50k documents, temporal topic evolution (2018–2025)
5. **RAG pipeline** — SentenceTransformers embeddings → ChromaDB vector store → LLaMA 3.1 via Groq
6. **API + UI** — FastAPI backend (Dockerized), Streamlit chat frontend


## Running the Project

### 1. Generate the ChromaDB index

Run notebooks in order (01 → 04) to produce `data/chromadb/`.

### 2. Build the Docker image

```bash
docker build -t dblp-api .
```

### 3. Start the API

```bash
docker run -p 8000:8000 -e GROQ_API_KEY=<your_key> dblp-api
```

API available at `http://localhost:8000` · Swagger UI at `http://localhost:8000/docs`

### 4. Start the Streamlit UI

```bash
streamlit run app/streamlit/st.py
```

UI available at `http://localhost:8501`


## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | RAG query — returns LLM answer + source papers |
| `GET`  | `/health` | Service status check |

**Example request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are recent advances in graph neural networks?", "n_results": 5}'
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | `pandas`, `lxml` |
| NLP / Vectorization | `scikit-learn` (TF-IDF, TruncatedSVD) |
| Clustering | `scikit-learn` MiniBatchKMeans |
| Visualization | `matplotlib`, `umap-learn` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | `chromadb` |
| LLM | LLaMA 3.1 8B via `langchain-groq` |
| API | `FastAPI` + `uvicorn` |
| UI | `Streamlit` |
| Deployment | `Docker` |
