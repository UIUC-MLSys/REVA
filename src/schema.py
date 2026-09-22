from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(slots=True)
class Context:
    doc_id: str
    text: str
    title: str = ""
    rank: int | None = None
    chunk_id: str | None = None
    dataset: str | None = None

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
    dataset: str | None = None
    split: str | None = None
    source_response: str | None = None

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


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path).expanduser()
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_context(row: dict[str, Any], rank: int, dataset: str | None = None) -> Context:
    return Context(
        doc_id=str(row["doc_id"]),
        text=str(row["text"]),
        title=str(row.get("title", "")),
        rank=rank if row.get("rank") is None else int(row["rank"]),
        chunk_id=None if row.get("chunk_id") is None else str(row["chunk_id"]).strip(),
        dataset=dataset,
    )


def parse_example(row: dict[str, Any], top_k: int | None = None) -> Example:
    contexts = list(row["contexts"])
    if top_k is not None:
        contexts = contexts[:top_k]
    return Example(
        id=str(row["id"]),
        question=str(row["question"]).strip(),
        contexts=[parse_context(context, rank, row.get("dataset")) for rank, context in enumerate(contexts, start=1)],
        answers=[str(answer).strip() for answer in row.get("answers", []) if str(answer).strip()],
        dataset=row.get("dataset"),
        split=row.get("split"),
        source_response=None if row.get("source_response") is None else str(row["source_response"]).strip(),
    )


def load_corpus(path: str | Path, doc_ids: set[str]) -> dict[str, dict[str, str]]:
    docs = {}
    if not doc_ids:
        return docs
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id", row.get("id")))
        if doc_id not in doc_ids:
            continue
        if "contents" in row:
            title, separator, text = row["contents"].partition("\n")
            if not separator:
                title, text = "", title
        else:
            title, text = row.get("title", ""), row["text"]
        docs[doc_id] = {"doc_id": doc_id, "title": title.strip(), "text": text.strip()}
        if len(docs) == len(doc_ids):
            break
    missing = doc_ids - docs.keys()
    if missing:
        raise ValueError(f"corpus is missing {len(missing)} document IDs, e.g. {sorted(missing)[:5]}")
    return docs


def load_examples(
    path: str | Path,
    limit: int | None = None,
    top_k: int | None = None,
    corpus_path: str | Path | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> list[Example]:
    rows = []
    if limit == 0:
        return []
    for row in read_jsonl(path):
        if dataset is not None and row.get("dataset") is not None and row["dataset"] != dataset:
            continue
        if split is not None and row.get("split") is not None and row["split"] != split:
            continue
        row["dataset"] = row.get("dataset") or dataset
        row["split"] = row.get("split") or split
        if "contexts" not in row:
            row["doc_ids"] = [str(doc_id) for doc_id in row["doc_ids"][:top_k]]
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break

    doc_ids = {doc_id for row in rows if "contexts" not in row for doc_id in row["doc_ids"]}
    if doc_ids and corpus_path is None:
        raise ValueError("retrieval IDs need document text; pass --corpus PATH to the wiki18 JSONL corpus")
    corpus = load_corpus(corpus_path, doc_ids) if doc_ids else {}
    for row in rows:
        if "contexts" not in row:
            row["contexts"] = [corpus[doc_id] for doc_id in row["doc_ids"]]
    return [parse_example(row, top_k) for row in rows]
