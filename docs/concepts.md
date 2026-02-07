# Key Concepts & Techniques

This guide explains the "Advanced" techniques used in this project. For each technique, we explore the **Problem** it solves, the **Solution** implemented, a concrete **Contrast** (Before vs. After), and the associated **Trade-offs**.

---

## 1. Sentence Window Retrieval

### The Problem: Context Fragmentation
In standard RAG, we split text into fixed chunks (e.g., 500 characters). This often cuts thoughts in half.
*   **Issue:** The chunk might contain the *answer* but lack the *context* needed to understand it.
*   **Result:** The LLM hallucinates because it sees "It was 50%," but misses the previous sentence saying "The profit margin."

### The Solution
We decouple **what we search** from **what we give the LLM**.
1.  **Index:** We embed only a single sentence (the "Center"). This splits meanings into atomic units, making search highly precise.
2.  **Retrieve:** When we find a match, we pull the "Window" (e.g., 3 sentences before and after) from metadata.
3.  **Generate:** The LLM sees the full coherent paragraph.

### Contrast: Before vs. After

| Feature | Baseline (Naive Chunking) | Advanced (Sentence Window) |
| :--- | :--- | :--- |
| **Retrieval Unit** | Arbitrary 500-char block | Single atomic sentence |
| **Context Provided** | "margin was 10%. However, the..." (Cut off) | "In 2023, the profit margin was 10%. However, the explicit costs..." (Full thought) |
| **LLM Experience** | Confusing, missing subjects | Coherent, self-contained |

### Trade-offs
*   **Storage Size (High):** We store overlapping windows, increasing database size by factor of $2 \times WindowSize$.
*   **Ingestion Speed (Slower):** More processing to split and manage windows.

---

## 2. HyDE (Hypothetical Document Embeddings)

### The Problem: Query-Document Mismatch
Users ask short, vague questions ("How to run?"), but documents are long and technical ("Execution instructions for the bash script...").
*   **Issue:** The vector for "How to run" implies a question, while the document vector implies a statement. They might not be close in vector space.

### The Solution
We ask an LLM to "hallucinate" a hypothetical answer strictly for retrieval purposes.
1.  **User:** "How to run?"
2.  **HyDE:** "To run the application, execute the `run_api.sh` script in your terminal."
3.  **Search:** We embed that *hypothetical answer* and search for documents matching *it*.

### Contrast: Before vs. After

| Feature | Baseline (Direct Query) | Advanced (HyDE) |
| :--- | :--- | :--- |
| **Search Query** | "challenges in deep learning" | "Deep learning faces issues like vanishing gradients and overfitting..." |
| **Match Quality** | Matches generic "learning" or "deep" keywords | Matches technical discussions on gradients/overfitting |
| **Retrieval** | Surface-level matches | Sementic/Intent-level matches |

### Trade-offs
*   **Latency (High):** Adds an extra LLM call *before* retrieval.
*   **Cost (Higher):** One extra generation token cost per query.

---

## 3. Cross-Encoder Reranking

### The Problem: The "Top-K" Noise
Vector search (Bi-Encoder) is fast but approximate. It collapses all meaning into a single vector.
*   **Issue:** Using `k=5` might return 2 relevant docs and 3 unrelated ones that just happen to share keywords (e.g., "Python" the snake vs. "Python" the code).

### The Solution
We use a two-stage process:
1.  **Retrieve (High Recall):** Get the top 25 docs using fast vector search.
2.  **Rerank (High Precision):** Use a **Cross-Encoder** model (BERT) that reads the Query and Document *together* to output a relevance score (0-1).
3.  **Filter:** Take the top 5 from the reranker.

### Contrast: Before vs. After

| Feature | Baseline (Vector Only) | Advanced (Reranked) |
| :--- | :--- | :--- |
| **Top Result** | "Python is a snake in the jungle..." (High vector similarity due to keyword) | "Python is a programming language..." (Cross-encoder understands context) |
| **Precision** | ~60% (Mix of good and bad) | ~90% (Bad results pushed to bottom) |

### Trade-offs
*   **Latency (Medium):** Running a BERT model on 25 documents takes time (CPU/GPU).
*   **Complexity:** Requires managing a second model in the pipeline.
