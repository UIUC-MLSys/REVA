from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class Context:
    doc_id: str
    text: str
    title: str = ""
    rank: int | None = None
    chunk_id: str | None = None

    @property
    def block(self) -> str:
        title = f"(Title: {self.title}) " if self.title else ""
        return f"Doc {self.rank} {title}{self.text.strip()}".strip()


@dataclass(slots=True)
class Example:
    id: str
    question: str
    contexts: list[Context]
    answers: list[str] | None = None

    @property
    def context_text(self) -> str:
        return "\n\n".join(context.block for context in self.contexts).strip()


@dataclass(slots=True)
class CompressionConfig:
    method: str
    budget: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompressionResult:
    method: str
    text: str
    raw_tokens: int
    compressed_tokens: int
    meta: dict[str, Any] | None = None


class Compressor:
    def __init__(self, config: CompressionConfig, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model_name = model_name

    def compress(self, example: Example) -> CompressionResult:
        raise NotImplementedError


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_context(row: dict[str, Any], rank: int) -> Context:
    return Context(
        doc_id=str(row["doc_id"]),
        text=str(row["text"]),
        title=str(row.get("title", "")),
        rank=rank if row.get("rank") is None else int(row["rank"]),
        chunk_id=None if row.get("chunk_id") is None else str(row["chunk_id"]).strip(),
    )


def parse_example(row: dict[str, Any], top_k: int | None = None) -> Example:
    contexts = list(row["contexts"])
    if top_k is not None:
        contexts = contexts[:top_k]
    return Example(
        id=str(row["id"]),
        question=str(row["question"]).strip(),
        contexts=[parse_context(context, rank) for rank, context in enumerate(contexts, start=1)],
        answers=[str(answer).strip() for answer in row.get("answers", []) if str(answer).strip()],
    )


def load_examples(path: str | Path, limit: int | None = None, top_k: int | None = None) -> list[Example]:
    lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    return [parse_example(row, top_k) for row in rows]
