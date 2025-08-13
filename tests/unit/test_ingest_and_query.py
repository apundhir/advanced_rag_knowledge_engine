from __future__ import annotations

from pathlib import Path

from app.pipeline.baseline import ingest_paths, answer_question


def test_ingest_and_query_pipeline(tmp_path: Path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    sample = doc_dir / "sample.txt"
    sample.write_text("Python is a programming language. It is popular for AI.")

    docs, chunks = ingest_paths([str(doc_dir)], chunk_size=64, chunk_overlap=16)
    assert docs == 1
    assert chunks >= 1

    answer, retrieved = answer_question("What is Python?", k=3)
    assert isinstance(answer, str)
    assert len(retrieved) >= 1
