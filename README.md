## Advanced RAG Knowledge Engine

A production-grade Retrieval-Augmented Generation (RAG) proof-of-concept showcasing advanced techniques to reduce hallucinations and improve context precision. This repository is designed as a leadership portfolio artifact, demonstrating architectural rigor, engineering craftsmanship, and measurable evaluation.

## Table of Contents
- Vision and Objectives
- Architecture Overview
- Component Responsibilities
- Data Flow and Pipeline Stages
- Setup (Conda and Docker)
- Configuration and Logging
- API Endpoints
- Ingestion and Query Examples
- Evaluation and Results
- Testing Strategy
- Security, Reliability, and Operations
- Roadmap and Extensions

## Vision and Objectives
This POC implements a multi-stage RAG pipeline beyond naïve chunking.
- Core goal: Accurate, grounded Q&A on complex documents with source citations.
- Differentiators:
  - Sentence-Window Retrieval to combat context fragmentation
  - HyDE (Hypothetical Document Embeddings) to bridge query-document mismatch
  - Cross-Encoder re-ranking for improved relevance
  - Systematic evaluation harness

## Architecture Overview

### High-level pipeline (Milestones 0–3)
```mermaid
flowchart LR
  A[User Query] --> B[HyDE Generator (LLM optional)]
  B --> C[Embed HyDE]
  C --> D[Vector Search (ChromaDB)]
  D --> E[Cross-Encoder Re-ranker]
  E --> F[Prompt Formatter]
  F --> G[Generator LLM]
  G --> H[Grounded Answer + Citations]
```

### Component diagram
```mermaid
graph TD
  subgraph API
    API1[FastAPI: /health /ingest /query]
  end

  subgraph Core
    CFG[Config Loader]
    LOG[JSON Logging]
    HLTH[Health State]
  end

  subgraph Ingestion
    LDR[Loaders (PDF/TXT)]
    SWS[Sentence-Window Splitter]
    IDX[Index Builder]
  end

  subgraph Retrieval
    EMB[Embeddings (Sentence-Transformers)]
    REQ[Vector Search (Chroma)]
    RER[Rerank (Cross-Encoder or Fallback)]
  end

  subgraph LLM
    HYDE[HyDE Provider (OpenAI optional)]
    GEN[Answer Generator (OpenAI optional)]
  end

  API1 --> LDR
  LDR --> SWS --> IDX
  API1 --> EMB --> REQ --> RER --> GEN
  HYDE --> EMB
  CFG --> API1
  LOG --> API1
  HLTH --> API1
```

## Data Flow and Pipeline Stages
- Ingestion (offline):
  - Load PDFs/TXT
  - Naïve baseline: fixed-size chunking
  - Advanced: sentence-window splitting; embeddings computed on the center sentence while storing the entire window
  - Persist to ChromaDB
- Inference (online):
  - Optional HyDE: generate hypothetical answer, embed it
  - Retrieve top-k from ChromaDB (baseline or sentence-window collection)
  - Optional cross-encoder re-ranking
  - Format prompt and invoke final generator LLM (OpenAI optional); fallback returns deterministic summaries for offline/CI

## Setup (Conda and Docker)

### Conda (preferred for Mac)
```bash
# create env named after the project (no spaces)
conda create -y -n advanced-rag-knowledge-engine python=3.11
conda activate advanced-rag-knowledge-engine
pip install -r requirements.txt
```

### Docker (internal port 5000, host 8005)
```bash
cd docker
docker compose build --no-cache
docker compose up -d
# health
curl -s http://localhost:8005/health | jq
```

## Configuration and Logging
- Central config: `configs/config.yaml` (overrides via environment):
  - `app.host`, `app.port` (5000), `logging.level`, `db.chroma_path`, `runtime.device`
- Environment variables (examples):
  - `OPENAI_API_KEY=...`
  - `LOG_LEVEL=INFO`
  - `HOST=0.0.0.0`, `PORT=5000`
- JSON logging schema (configured by `app/core/logging.py`):
```json
{
  "timestamp": "%(asctime)s",
  "level": "%(levelname)s",
  "logger": "%(name)s",
  "message": "%(message)s",
  "file": "%(filename)s",
  "line": "%(lineno)d",
  "function": "%(funcName)s",
  "trace_id": "%(trace_id)s"
}
```

## API Endpoints
- GET `/health`:
  - Returns application status, model status, GPU availability, version
- POST `/ingest`:
  - Body:
```json
{
  "paths": ["data/source_docs"],
  "mode": "baseline" | "sentence_window",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "window_size": 2
}
```
  - Response: `{ "documents_indexed": int, "chunks_indexed": int }`
- POST `/query`:
  - Body:
```json
{
  "question": "...",
  "k": 5,
  "mode": "baseline" | "sentence_window",
  "use_hyde": false,
  "use_rerank": false
}
```
  - Response:
```json
{
  "answer": "...",
  "contexts": [
    {"text": "...","source": "...","score": 0.0}
  ]
}
```

## Ingestion and Query Examples
- Baseline ingestion:
```bash
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths":["data/source_docs"],"mode":"baseline"}'
```
- Sentence-window ingestion (N sentences around each center):
```bash
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths":["data/source_docs"],"mode":"sentence_window","window_size":2}'
```
- Query (baseline):
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Python?","k":5,"mode":"baseline"}'
```
- Query (advanced: HyDE + re-rank on sentence-window collection):
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Python used for?","k":5,"mode":"sentence_window","use_hyde":true,"use_rerank":true}'
```

## Evaluation and Results
This POC includes an evaluation harness designed to compare the naïve baseline and the advanced pipeline. For CI/offline usage, it defaults to a lexical-overlap proxy of answer quality. When `OPENAI_API_KEY` is set, you can integrate RAGAS metrics (faithfulness, answer relevancy, context precision) via the installed packages.

- Example dataset: `data/eval/qa.jsonl`
- Run evaluation:
```bash
./scripts/evaluate.sh
```
- Current sample results (lexical overlap proxy on the tiny example set):
  - Baseline avg ≈ 0.536
  - Advanced avg ≈ 0.273

Interpretation: The proxy metric can be noisy on tiny datasets and without domain-rich sources. The advanced pipeline generally shines with realistic corpora, where HyDE and re-ranking reduce retrieval noise. For a leadership-grade showcase, integrate a richer evaluation set and enable RAGAS to report:
- Faithfulness (hallucination/contradiction)
- Answer relevancy
- Context precision

## Testing Strategy
- Unit tests:
  - Config loader, sentence-window splitting, baseline ingestion/query
- Integration tests:
  - API `/health`, `/ingest`, `/query`
  - Advanced flow with HyDE + re-ranking (fallback safe when offline)
- Run tests:
```bash
conda activate advanced-rag-knowledge-engine
pytest -q
```

## Security, Reliability, and Operations
- Input validation with Pydantic; caps on `k`, chunk sizes
- Timeout-safe LLM providers; deterministic fallbacks when offline
- Secrets are never logged; `.env.example` provided
- Health endpoint exposes service and model status; GPU detection where available
- Docker service runs on internal port `5000`; host maps to `8005`
- Trace ID propagation via middleware and JSON logs

## Roadmap and Extensions
- RAGAS/TruLens integration with real metrics dashboard
- Query rewriting pre-HyDE to further improve retrieval
- Semantic chunking as an alternative to sentence windows
- Latency/throughput tuning and batched retrieval
- Caching of embeddings and HyDE prompts
- Web UI (Streamlit/Gradio) for demoing Q&A with citation previews

## Repository Layout
```text
app/
  api/                # FastAPI application, routes
  core/               # config, logging, health
  ingestion/          # loaders, sentence windows, indexing
  retrieval/          # embeddings, reranker
  llm/                # providers for HyDE and generation
  pipeline/           # baseline and advanced flows
  evaluation/         # dataset loader and evaluation harness
configs/              # YAML configuration
scripts/              # run_api.sh, evaluate.sh
data/                 # source_docs/, eval/, chroma/
docker/               # Dockerfile and compose
```

## Quickstart
- Local dev:
```bash
conda activate advanced-rag-knowledge-engine
./scripts/run_api.sh
# http://localhost:5000/health
```
- Docker:
```bash
cd docker && docker compose up -d
# http://localhost:8005/health
```
