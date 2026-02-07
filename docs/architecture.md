# Architecture Overview

This document provides visual diagrams of the **Advanced RAG Knowledge Engine** to help you understand how data flows through the system.

## 1. System Context
High-level view of how the User interacts with the API and the underlying RAG components.

```mermaid
graph LR
    User[User] -- HTTP POST --> API[FastAPI Service]
    API -- Ingest --> Ingestion[Ingestion Pipeline]
    API -- Query --> RAG[RAG Pipeline]
    
    subgraph "Knowledge Engine"
        Ingestion -- Writes --> VectorDB[(ChromaDB)]
        RAG -- Reads --> VectorDB
        RAG -- Generates --> LLM["LLM Provider (OpenAI)"]
    end
```

## 2. Ingestion Pipeline (Offline)
How documents are processed, chunked, and indexed. Note the **Sentence Window** strategy.

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Loader
    participant Splitter as Window Splitter
    participant Embedder
    participant DB as ChromaDB

    Client->>API: POST /ingest (paths, mode="sentence_window")
    API->>Loader: Load Documents (PDF/TXT)
    Loader-->>API: Raw Text
    
    loop For each Document
        API->>Splitter: split_into_sentence_windows(text)
        Splitter-->>API: "List[(Center Sentence, Full Window)]"
        
        Note right of Splitter: Embed Center Sentence<br/>Store Full Window
        
        API->>Embedder: embed(Center Sentences)
        Embedder-->>API: Vectors
        API->>DB: add(ids, vectors, metadatas={window: Full Window})
    end
    
    API-->>Client: 200 OK (Count)
```

## 3. Query Pipeline (Online)
The "Advanced RAG" flow featuring **HyDE** and **Cross-Encoder Reranking**.

```mermaid
flowchart TD
    Start([User Question]) --> HyDE{Use HyDE?}
    
    HyDE -- Yes --> GenHyDE[Generate Hypothetical Answer]
    GenHyDE --> Embed[Embed Text]
    HyDE -- No --> Embed
    
    Embed --> Retrieve[Vector Search (Top-K)]
    Retrieve --> Rerank{Use Reranker?}
    
    Rerank -- Yes --> CrossEncoder[Cross-Encoder Scoring]
    CrossEncoder --> Filter[Top-N Relevant]
    Rerank -- No --> Filter
    
    Filter --> Context[Construct Context Window]
    Context --> Prompt[Format Prompt]
    Prompt --> LLM[Generator LLM]
    LLM --> Answer([Final Answer])
```
