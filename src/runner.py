from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from methods import get_compressor
from metrics import score, summarize_records
from models import build_prompt, generate, load_model
from schema import CompressionConfig, CompressionResult, Example, load_examples, write_json, write_jsonl


def build_record(
    example: Example,
    result: CompressionResult,
    compression_ms: float,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "id": example.id,
        "question": example.question,
        "answers": example.answers or [],
        "method": result.method,
        "raw_text": example.context_text,
        "compressed_text": result.text,
        "raw_tokens": result.raw_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_ms": round(compression_ms, 3),
    }
    if result.meta:
        record["compression_meta"] = result.meta
    if generation:
        record.update(generation)
        record.update(score(generation["prediction"], example.answers))
    return record


def run(
    input_path: str | Path,
    method: str,
    output_dir: str | Path,
    budget: int | None = None,
    model_name: str | None = None,
    options: dict[str, Any] | None = None,
    limit: int | None = None,
    top_k: int | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")

    examples = load_examples(input_path, limit, top_k)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = load_model(model_name) if model_name else None
    tokenizer = None if generator is None else generator.tokenizer
    compressor = get_compressor(CompressionConfig(method, budget, options or {}), tokenizer, model_name)

    records = []
    for example in examples:
        start = time.perf_counter()
        result = compressor.compress(example)
        compression_ms = (time.perf_counter() - start) * 1000.0
        generation = None
        if generator is not None:
            generation = generate(generator, build_prompt(example, result.text, tokenizer), max_new_tokens)
        records.append(build_record(example, result, compression_ms, generation))

    summary = summarize_records(records) if records else {"method": method, "queries": 0}
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    return summary
