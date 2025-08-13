from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import Dataset

from app.pipeline.baseline import answer_question, answer_question_with_collection
from app.pipeline.advanced import answer_with_hyde_and_rerank
from app.ingestion.index import SENTENCE_WINDOW_COLLECTION


@dataclass
class EvalResult:
    question: str
    reference: str
    baseline_answer: str
    advanced_answer: str


def _run_baseline(question: str) -> Tuple[str, List[str]]:
    ans, retrieved = answer_question(question, k=5)
    contexts = [t for t, _m, _s in retrieved]
    return ans, contexts


def _run_advanced(question: str) -> Tuple[str, List[str]]:
    ans, retrieved = answer_with_hyde_and_rerank(question, k=5, rerank_top_k=5)
    contexts = [t for t, _m, _s in retrieved]
    return ans, contexts


def _run_sentence_window_baseline(question: str) -> Tuple[str, List[str]]:
    ans, retrieved = answer_question_with_collection(question, k=5, collection_name=SENTENCE_WINDOW_COLLECTION)
    contexts = [t for t, _m, _s in retrieved]
    return ans, contexts


def _score_pair(reference: str, candidate: str) -> float:
    """Simple lexical overlap as a fallback when ragas cannot be used."""
    ref = set(reference.lower().split())
    cand = set(candidate.lower().split())
    if not ref:
        return 0.0
    return len(ref & cand) / len(ref)


def evaluate_pairs_proxy(qa_pairs: List[Tuple[str, str]]) -> List[EvalResult]:
    results: List[EvalResult] = []
    for q, ref in qa_pairs:
        baseline_ans, _ = _run_baseline(q)
        advanced_ans, _ = _run_advanced(q)
        results.append(EvalResult(question=q, reference=ref, baseline_answer=baseline_ans, advanced_answer=advanced_ans))
    return results


def print_summary_proxy(results: List[EvalResult]) -> None:
    baseline_scores: List[float] = []
    advanced_scores: List[float] = []

    for r in results:
        baseline_scores.append(_score_pair(r.reference, r.baseline_answer))
        advanced_scores.append(_score_pair(r.reference, r.advanced_answer))

    b_avg = sum(baseline_scores) / max(1, len(baseline_scores))
    a_avg = sum(advanced_scores) / max(1, len(advanced_scores))

    print("\nEvaluation Summary (lexical overlap proxy)")
    print("---------------------------------------")
    print(f"Baseline avg: {b_avg:.3f}")
    print(f"Advanced avg: {a_avg:.3f}")


def evaluate_with_ragas(qa_pairs: List[Tuple[str, str]]) -> None:
    """Run RAGAS metrics when OPENAI_API_KEY is set.

    Builds two datasets (baseline vs advanced) with question, answer, contexts, ground_truth.
    Prints aggregated metric means.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot run RAGAS")

    # Prepare datasets
    baseline_rows: List[Dict] = []
    advanced_rows: List[Dict] = []

    for q, ref in qa_pairs:
        b_ans, b_ctx = _run_baseline(q)
        a_ans, a_ctx = _run_advanced(q)
        baseline_rows.append({"question": q, "answer": b_ans, "contexts": b_ctx, "ground_truth": ref})
        advanced_rows.append({"question": q, "answer": a_ans, "contexts": a_ctx, "ground_truth": ref})

    baseline_ds = Dataset.from_list(baseline_rows)
    advanced_ds = Dataset.from_list(advanced_rows)

    # Configure LLM and embeddings for RAGAS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_relevancy
    from ragas import evaluate as ragas_evaluate

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    metrics = [faithfulness, answer_relevancy, context_precision, context_relevancy]

    print("\nRunning RAGAS on baseline dataset...")
    b_result = ragas_evaluate(baseline_ds, metrics=metrics, llm=llm, embeddings=embeddings)
    print("Running RAGAS on advanced dataset...")
    a_result = ragas_evaluate(advanced_ds, metrics=metrics, llm=llm, embeddings=embeddings)

    # Print means
    print("\nRAGAS Summary (means)")
    print("---------------------")
    for m in metrics:
        name = getattr(m, "name", str(m))
        b_mean = float(b_result[m].mean()) if hasattr(b_result[m], "mean") else float(b_result[m])
        a_mean = float(a_result[m].mean()) if hasattr(a_result[m], "mean") else float(a_result[m])
        print(f"{name}: baseline={b_mean:.3f}  advanced={a_mean:.3f}")


if __name__ == "__main__":
    import argparse
    from app.evaluation.dataset import load_qa_pairs

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/eval/qa.jsonl")
    args = parser.parse_args()

    pairs = load_qa_pairs(args.dataset)

    if os.getenv("OPENAI_API_KEY"):
        try:
            evaluate_with_ragas(pairs)
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}. Falling back to proxy metrics.")
            results = evaluate_pairs_proxy(pairs)
            print_summary_proxy(results)
    else:
        results = evaluate_pairs_proxy(pairs)
        print_summary_proxy(results)
