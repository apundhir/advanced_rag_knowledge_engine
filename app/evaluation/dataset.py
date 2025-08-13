from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def load_qa_pairs(path: str | Path) -> List[Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # local import to keep dependencies centralized

        data = yaml.safe_load(p.read_text())
        return [(item["question"], item["answer"]) for item in data]
    # jsonl as default
    items: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append((row["question"], row["answer"]))
    return items
