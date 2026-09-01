from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines import result, truncate_text
from metrics import token_count
from schema import CompressionResult, Compressor, Context, Example, load_examples, write_json, write_jsonl

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def doc_key(context: Context) -> str:
    return f"{context.doc_id}::chunk={context.chunk_id}" if context.chunk_id else context.doc_id


def optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def dtype_value(value: Any) -> Any:
    return {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(value or "auto").lower())


def model_device(model: Any) -> Any:
    return next(model.parameters()).device


def to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def set_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


@dataclass(slots=True)
class REVAConfig:
    budget: int | None
    score_store_path: str | None = None
    scoring_model_name: str | None = None
    max_scoring_tokens: Any = "model_max_length"
    per_doc_quota: int | None = None
    device_map: Any = "auto"
    dtype: Any = "auto"
    attn_implementation: str | None = "eager"
    layer_subset: Any = "all"
    head_aggregation: str = "mean"
    head_top_k: int = 3
    query_aggregation: str = "sum"
    word_score: str = "max"


@dataclass(slots=True)
class PromptDoc:
    context: Context
    start: int
    end: int


@dataclass(slots=True)
class PromptInput:
    text: str
    docs: list[PromptDoc]
    source_ranges: list[tuple[int, int]]


@dataclass(slots=True)
class WordUnit:
    text: str
    index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_ids: list[int] = field(default_factory=list)
    token_count: int = 1
    score: float = 0.0
    score_sum: float = 0.0
    score_count: int = 0
    prompt_positions: list[int] = field(default_factory=list)


@dataclass(slots=True)
class DocumentUnits:
    context: Context
    units: list[WordUnit]
    source: str = "prompt"
    total_tokens: int | None = None

    @property
    def raw_tokens(self) -> int:
        return self.total_tokens if self.total_tokens is not None else sum(unit.token_count for unit in self.units)


@dataclass(slots=True)
class ScoreDoc:
    doc_id: str
    key: str
    units: list[WordUnit]
    chunk_id: str | None = None
    raw_text: str = ""


@dataclass(slots=True)
class ScoreStore:
    docs: dict[str, ScoreDoc]

    def find(self, context: Context) -> ScoreDoc | None:
        return self.docs.get(doc_key(context))


@dataclass(slots=True)
class SelectionResult:
    docs: list[DocumentUnits]
    quotas: list[int]


@dataclass(slots=True)
class DocAggregate:
    key: str
    context: Context
    units: list[WordUnit]
    query_count: int = 0


@dataclass(slots=True)
class ScoredDocs:
    docs: list[DocumentUnits]
    meta: dict[str, Any]


def tokenizer_max_length(tokenizer: Any) -> int | None:
    length = int(getattr(tokenizer, "model_max_length", 0) or 0)
    return length if 0 < length < 1_000_000 else None


def resolve_max_scoring_tokens(tokenizer: Any, config: REVAConfig) -> int | None:
    value = config.max_scoring_tokens
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "off", "false", "0"}:
            return None
        if normalized in {"auto", "model", "model_max_length"}:
            return tokenizer_max_length(tokenizer)
    limit = int(value)
    return limit if limit > 0 else None


def config_meta(config: REVAConfig) -> dict[str, Any]:
    return {
        "scoring_model_name": config.scoring_model_name,
        "max_scoring_tokens": config.max_scoring_tokens,
        "per_doc_quota": config.per_doc_quota,
        "layer_subset": config.layer_subset,
        "head_aggregation": config.head_aggregation,
        "head_top_k": config.head_top_k,
        "query_aggregation": config.query_aggregation,
        "word_score": config.word_score,
        "word_unit_type": "tokenizer_offset_words",
        "score_normalization": "per_query_doc_max",
    }


def context_header(context: Context) -> str:
    rank = "" if context.rank is None else str(context.rank)
    title = f" | title={context.title}" if context.title else ""
    return f"[DOC {rank} | id={context.doc_id}{title}]".strip()


def build_scoring_prompt(example: Example, contexts: list[Context] | None = None) -> PromptInput:
    text = "Contexts:\n"
    docs = []
    for context in (list(example.contexts) if contexts is None else contexts):
        text += context_header(context) + "\n"
        start = len(text)
        body = context.text.strip()
        text += body
        end = len(text)
        text += "\n\n"
        docs.append(PromptDoc(context, start, end))

    text += "Question:\n"
    query_start = len(text)
    text += example.question.strip()
    query_end = len(text)
    text += "\n\n"
    source_ranges = [(query_start, query_end)]
    return PromptInput(text, docs, source_ranges)


def flat_list(values: Any) -> list[Any]:
    return list(values[0]) if values and isinstance(values[0], list) else list(values)


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(token_id) for token_id in flat_list(encoded["input_ids"])]
    offsets = [tuple(offset) for offset in flat_list(encoded["offset_mapping"])]
    return input_ids, [(int(start), int(end)) for start, end in offsets]


def overlaps(start: int, end: int, target_start: int, target_end: int) -> bool:
    return end > start and start < target_end and end > target_start


def positions_in_ranges(offsets: list[tuple[int, int]], ranges: list[tuple[int, int]]) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if any(overlaps(start, end, target_start, target_end) for target_start, target_end in ranges)
    ]


def count_tokens(tokenizer: Any, text: str) -> int:
    token_ids, _offsets = tokenize_with_offsets(tokenizer, text)
    return len(token_ids)


def limited_scoring_prompt(example: Example, tokenizer: Any, config: REVAConfig) -> tuple[PromptInput, dict[str, Any]]:
    contexts = list(example.contexts)
    prompt = build_scoring_prompt(example, contexts)
    before = count_tokens(tokenizer, prompt.text)
    limit = resolve_max_scoring_tokens(tokenizer, config)
    meta = {
        "max_scoring_tokens": limit,
        "scoring_prompt_tokens_before_limit": before,
        "scoring_prompt_tokens": before,
        "scoring_prompt_truncated": False,
        "scoring_kept_documents": len(contexts),
        "scoring_dropped_documents": 0,
    }
    if limit is None or before <= limit:
        return prompt, meta

    dropped = 0
    while contexts:
        contexts = contexts[1:]
        dropped += 1
        prompt = build_scoring_prompt(example, contexts)
        current = count_tokens(tokenizer, prompt.text)
        if current <= limit:
            meta.update({
                "scoring_prompt_tokens": current,
                "scoring_prompt_truncated": True,
                "scoring_kept_documents": len(contexts),
                "scoring_dropped_documents": dropped,
            })
            return prompt, meta
    raise ValueError(f"max_scoring_tokens={limit} is too small for the scoring question/answer")


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    mode = mode.strip().lower()
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "sum":
        return sum(values)
    raise ValueError("word_score must be one of: max, mean, sum")


def normalize_scores(units: list[WordUnit]) -> list[WordUnit]:
    max_score = max((unit.score for unit in units), default=0.0)
    if max_score > 0:
        for unit in units:
            unit.score = round(unit.score / max_score, 6)
    return units


def build_word_units(text: str, tokenizer: Any | None = None) -> list[WordUnit]:
    if tokenizer is None:
        return [
            WordUnit(match.group(0), index, match.start(), match.end(), index, index + 1)
            for index, match in enumerate(WORD_RE.finditer(text))
        ]
    token_ids, offsets = tokenize_with_offsets(tokenizer, text)
    units = []
    for index, match in enumerate(WORD_RE.finditer(text)):
        positions = positions_in_ranges(offsets, [(match.start(), match.end())])
        if positions:
            start = positions[0]
            end = positions[-1] + 1
            ids = token_ids[start:end]
            units.append(WordUnit(match.group(0), index, match.start(), match.end(), start, end, ids, len(ids)))
    return units


def build_prompt_word_units(
    prompt: str,
    prompt_doc: PromptDoc,
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    scores: dict[int, float],
    config: REVAConfig,
) -> DocumentUnits:
    doc_positions = positions_in_ranges(offsets, [(prompt_doc.start, prompt_doc.end)])
    local_position = {position: index for index, position in enumerate(doc_positions)}
    units = []
    for index, match in enumerate(WORD_RE.finditer(prompt, prompt_doc.start, prompt_doc.end)):
        positions = positions_in_ranges(offsets, [(match.start(), match.end())])
        if not positions:
            continue
        local_start = local_position[positions[0]]
        local_end = local_position[positions[-1]] + 1
        ids = [token_ids[position] for position in positions]
        values = [scores.get(position, 0.0) for position in positions]
        units.append(
            WordUnit(
                text=match.group(0),
                index=index,
                char_start=match.start() - prompt_doc.start,
                char_end=match.end() - prompt_doc.start,
                token_start=local_start,
                token_end=local_end,
                token_ids=ids,
                token_count=len(ids),
                score=aggregate(values, config.word_score),
                prompt_positions=positions,
            )
        )
    return DocumentUnits(prompt_doc.context, normalize_scores(units), "prompt", len(doc_positions))


def select_layers(attentions: tuple[Any, ...] | list[Any], subset: Any) -> list[Any]:
    layers = [layer for layer in attentions if layer is not None]
    if isinstance(subset, str):
        value = subset.strip().lower()
        if value == "all":
            return layers
        if value == "first":
            return layers[:1]
        if value == "last":
            return layers[-1:]
        if value.lstrip("-").isdigit():
            return [layers[int(value)]]
    if isinstance(subset, int):
        return [layers[subset]]
    if isinstance(subset, list):
        return [layers[int(index)] for index in subset]
    raise ValueError("layer_subset must be all, first, last, an integer, or a list of integers")


def reduce_heads(scores: Any, config: REVAConfig) -> Any:
    mode = config.head_aggregation.strip().lower()
    if mode == "mean":
        return scores.mean(dim=0)
    if mode == "max":
        return scores.max(dim=0).values
    if mode in {"topk", "top_k", "top_k_mean"}:
        top_k = min(max(1, config.head_top_k), int(scores.shape[0]))
        return scores.topk(top_k, dim=0).values.mean(dim=0)
    raise ValueError("head_aggregation must be one of: mean, max, topk")


def reduce_sources(scores: Any, config: REVAConfig) -> Any:
    mode = config.query_aggregation.strip().lower()
    if mode == "sum":
        return scores.sum(dim=0)
    if mode == "mean":
        return scores.mean(dim=0)
    if mode == "max":
        return scores.max(dim=0).values
    raise ValueError("query_aggregation must be one of: sum, mean, max")


def run_attentions(model: Any, token_ids: list[int]) -> Any:
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    inputs = to_device({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}, model_device(model))
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    if outputs.attentions is None:
        raise ValueError("reva could not read attention maps from the scoring model")
    return outputs.attentions


def token_scores(
    attentions: tuple[Any, ...] | list[Any],
    source_positions: list[int],
    document_positions: list[int],
    config: REVAConfig,
) -> dict[int, float]:
    if not source_positions or not document_positions:
        return {position: 0.0 for position in document_positions}
    layer_scores = []
    layers = select_layers(attentions, config.layer_subset)
    if not layers:
        raise ValueError("reva found no usable attention layers")
    for layer in layers:
        attention = layer.detach().float()
        device = attention.device
        source_index = torch.tensor(source_positions, dtype=torch.long, device=device)
        document_index = torch.tensor(document_positions, dtype=torch.long, device=device)
        scores = attention[0].index_select(1, source_index).index_select(2, document_index)
        layer_scores.append(reduce_sources(reduce_heads(scores, config), config))
    values = torch.stack(layer_scores).mean(dim=0).detach().cpu().tolist()
    return {position: round(float(value), 6) for position, value in zip(document_positions, values)}


def score_prompt_with_meta(example: Example, tokenizer: Any, model: Any, config: REVAConfig) -> ScoredDocs:
    prompt, meta = limited_scoring_prompt(example, tokenizer, config)
    token_ids, offsets = tokenize_with_offsets(tokenizer, prompt.text)
    document_positions = positions_in_ranges(offsets, [(doc.start, doc.end) for doc in prompt.docs])
    source_positions = positions_in_ranges(offsets, prompt.source_ranges)
    if not source_positions:
        raise ValueError("reva could not find query tokens for attention scoring")
    if document_positions and min(source_positions) <= max(document_positions):
        raise ValueError("reva scoring prompt must place query tokens after document tokens")
    scores = token_scores(run_attentions(model, token_ids), source_positions, document_positions, config)
    docs = [build_prompt_word_units(prompt.text, doc, token_ids, offsets, scores, config) for doc in prompt.docs]
    meta["scored_document_tokens"] = len(document_positions)
    meta["source_tokens"] = len(source_positions)
    return ScoredDocs(docs, meta)


def unit_score(row: dict[str, Any]) -> float:
    if row.get("score_count"):
        return float(row.get("score_sum", 0.0)) / float(row["score_count"])
    return float(row.get("score", 0.0))


def parse_score_doc(row: dict[str, Any]) -> ScoreDoc:
    units = [
        WordUnit(
            text=str(unit["text"]),
            index=int(unit.get("index", index)),
            char_start=int(unit.get("char_start", 0)),
            char_end=int(unit.get("char_end", 0)),
            token_start=int(unit.get("token_start", index)),
            token_end=int(unit.get("token_end", unit.get("token_start", index) + int(unit.get("token_count", 1)))),
            token_ids=[int(token_id) for token_id in unit.get("token_ids", [])],
            token_count=int(unit.get("token_count", 1)),
            score=unit_score(unit),
            score_sum=float(unit.get("score_sum", unit.get("score", 0.0))),
            score_count=int(unit.get("score_count", 0)),
        )
        for index, unit in enumerate(row.get("word_units", []))
        if str(unit.get("text", "")).strip()
    ]
    doc_id = str(row.get("doc_id") or row.get("id"))
    key = str(row.get("doc_key") or row.get("id") or doc_id)
    return ScoreDoc(
        doc_id=doc_id,
        key=key,
        units=units,
        chunk_id=None if row.get("chunk_id") is None else str(row["chunk_id"]),
        raw_text=str(row.get("raw_text", "")),
    )


def copy_store_unit(unit: WordUnit) -> WordUnit:
    return WordUnit(
        unit.text,
        unit.index,
        unit.char_start,
        unit.char_end,
        unit.token_start,
        unit.token_end,
        list(unit.token_ids),
        unit.token_count,
        0.0,
        0.0,
        0,
    )


def add_to_aggregates(aggregates: dict[str, DocAggregate], doc: DocumentUnits) -> None:
    key = doc_key(doc.context)
    if key not in aggregates:
        aggregates[key] = DocAggregate(key, doc.context, [copy_store_unit(unit) for unit in doc.units])
    aggregate = aggregates[key]
    if aggregate.context.text != doc.context.text:
        raise ValueError(f"doc_key={key!r} maps to different context text")
    if len(aggregate.units) != len(doc.units):
        raise ValueError(f"doc_key={key!r} maps to different word-unit counts")
    for index, unit in enumerate(doc.units):
        stored = aggregate.units[index]
        if stored.text != unit.text or stored.char_start != unit.char_start or stored.char_end != unit.char_end:
            raise ValueError(f"doc_key={key!r} maps to different word-unit boundaries")
        if stored.token_ids != unit.token_ids:
            raise ValueError(f"doc_key={key!r} maps to different word-unit token ids")
        stored.score_sum += float(unit.score)
        stored.score_count += 1
        stored.score = stored.score_sum / stored.score_count
    aggregate.query_count += 1


def score_doc_row(aggregate: DocAggregate, config: REVAConfig) -> dict[str, Any]:
    return {
        "id": aggregate.key,
        "doc_key": aggregate.key,
        "doc_id": aggregate.context.doc_id,
        "chunk_id": aggregate.context.chunk_id,
        "title": aggregate.context.title,
        "raw_text": aggregate.context.text,
        "word_units": [
            {
                "index": unit.index,
                "text": unit.text,
                "char_start": unit.char_start,
                "char_end": unit.char_end,
                "token_start": unit.token_start,
                "token_end": unit.token_end,
                "token_ids": unit.token_ids,
                "token_count": unit.token_count,
                "score_sum": round(unit.score_sum, 6),
                "score_count": unit.score_count,
                "score": round(unit.score, 6),
            }
            for unit in aggregate.units
        ],
        "score_store_meta": {
            "scoring": "mean_attention",
            "score_normalization": "per_query_doc_max",
            "word_unit_type": "tokenizer_offset_words",
            "training_hits": aggregate.query_count,
            "raw_tokens": sum(unit.token_count for unit in aggregate.units),
            "raw_units": len(aggregate.units),
            "scoring_model_name": config.scoring_model_name,
        },
    }


def validate_score_doc(score_doc: ScoreDoc, context: Context) -> None:
    if score_doc.doc_id != context.doc_id:
        raise ValueError(f"score_store key={score_doc.key!r} has a different doc_id")
    if score_doc.chunk_id is not None and context.chunk_id is not None and score_doc.chunk_id != context.chunk_id:
        raise ValueError(f"score_store key={score_doc.key!r} has a different chunk_id")
    if score_doc.raw_text and score_doc.raw_text != context.text:
        raise ValueError(f"score_store key={score_doc.key!r} has different raw_text")


def reconstruct_units(score_doc: ScoreDoc) -> list[WordUnit]:
    return [
        WordUnit(
            unit.text,
            unit.index,
            unit.char_start,
            unit.char_end,
            unit.token_start,
            unit.token_end,
            list(unit.token_ids),
            unit.token_count,
            unit.score,
            unit.score_sum,
            unit.score_count,
        )
        for unit in score_doc.units
    ]


def doc_quotas(docs: list[DocumentUnits], budget: int | None, config: REVAConfig) -> list[int]:
    if config.per_doc_quota is not None:
        return [config.per_doc_quota for _ in docs]
    if budget is None:
        return [doc.raw_tokens for doc in docs]
    base, extra = divmod(budget, len(docs))
    return [base + (1 if index < extra else 0) for index in range(len(docs))]


def top_units(units: list[WordUnit], budget: int) -> list[WordUnit]:
    kept = []
    used = 0
    for unit in sorted(units, key=lambda item: (-item.score, item.index)):
        if used + unit.token_count <= budget:
            kept.append(unit)
            used += unit.token_count
    return sorted(kept, key=lambda item: item.index)


def select_docwise(docs: list[DocumentUnits], budget: int | None, config: REVAConfig) -> SelectionResult:
    quotas = doc_quotas(docs, budget, config)
    selected = [
        DocumentUnits(doc.context, top_units(doc.units, quotas[index]), doc.source, doc.raw_tokens)
        for index, doc in enumerate(docs)
    ]
    return SelectionResult(selected, quotas)


def select(docs: list[DocumentUnits], budget: int | None, config: REVAConfig) -> SelectionResult:
    if not docs:
        return SelectionResult([], [])
    return select_docwise(docs, budget, config)


def expand_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and text[start - 1] in "([{'\"":
        start -= 1
    while end < len(text) and text[end] in ".,;:!?)]}'\"":
        end += 1
    return start, end


def render_units(doc: DocumentUnits) -> str:
    spans = [expand_span(doc.context.text, unit.char_start, unit.char_end) for unit in sorted(doc.units, key=lambda item: item.index)]
    if not spans:
        return ""
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        gap = doc.context.text[prev_end:start]
        if start <= prev_end or (len(gap) <= 3 and not WORD_RE.search(gap)):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return " ".join(doc.context.text[start:end].strip() for start, end in merged if doc.context.text[start:end].strip())


def render_docs(selection: SelectionResult) -> str:
    parts = []
    for doc in selection.docs:
        text = render_units(doc)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def selection_meta(selection: SelectionResult, text: str, tokenizer: Any | None = None) -> dict[str, Any]:
    selected_unit_tokens = sum(unit.token_count for doc in selection.docs for unit in doc.units)
    rendered_tokens = token_count(text, tokenizer)
    store_hits = sum(doc.source == "score_store" for doc in selection.docs)
    return {
        "allocation": "docwise",
        "documents": len(selection.docs),
        "score_store_hits": store_hits,
        "fallback_documents": len(selection.docs) - store_hits,
        "realized_budget": rendered_tokens,
        "selected_unit_tokens": selected_unit_tokens,
        "rendered_tokens": rendered_tokens,
    }


def render_selection(selection: SelectionResult, tokenizer: Any | None = None) -> tuple[str, dict[str, Any]]:
    text = render_docs(selection)
    return text, selection_meta(selection, text, tokenizer)


def load_score_store(path: str | Path) -> ScoreStore:
    docs = {}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = parse_score_doc(json.loads(line))
            if doc.key in docs:
                raise ValueError(f"duplicate score_store key: {doc.key}")
            docs[doc.key] = doc
    return ScoreStore(docs)


def load_scoring_model(config: REVAConfig) -> tuple[Any, Any]:
    if not config.scoring_model_name:
        raise ValueError("reva attention scoring requires --model or --option scoring_model_name=...")
    tokenizer = AutoTokenizer.from_pretrained(config.scoring_model_name, use_fast=True, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("reva attention scoring requires a fast tokenizer for offset mapping")
    set_pad_token(tokenizer)
    device_map = config.device_map
    if device_map == "auto" and not torch.cuda.is_available():
        device_map = None
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    dtype = dtype_value(config.dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.scoring_model_name, **kwargs)
    model.eval()
    return tokenizer, model


def read_config(options: dict[str, Any] | None, budget: int | None, model_name: str | None = None) -> REVAConfig:
    options = options or {}
    return REVAConfig(
        budget=budget,
        score_store_path=options.get("score_store_path"),
        scoring_model_name=str(options.get("scoring_model_name") or model_name) if options.get("scoring_model_name") or model_name else None,
        max_scoring_tokens=options.get("max_scoring_tokens", "model_max_length"),
        per_doc_quota=optional_int(options.get("per_doc_token_quota")),
        device_map=options.get("device_map", "auto"),
        dtype=options.get("dtype", "auto"),
        attn_implementation=options.get("attn_implementation", "eager"),
        layer_subset=options.get("layer_subset", "all"),
        head_aggregation=str(options.get("head_aggregation", "mean")),
        head_top_k=int(options.get("head_top_k", 3)),
        query_aggregation=str(options.get("query_aggregation", "sum")),
        word_score=str(options.get("word_score", options.get("word_unit_score_aggregation", "max"))),
    )


def build_store(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
    options: dict[str, Any] | None = None,
    limit: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")
    config = read_config(options, None, model_name)
    tokenizer, model = load_scoring_model(config)
    aggregates: dict[str, DocAggregate] = {}
    examples = load_examples(input_path, limit, top_k)
    truncated_queries = 0
    dropped_documents = 0
    for example in examples:
        scored = score_prompt_with_meta(example, tokenizer, model, config)
        truncated_queries += int(bool(scored.meta["scoring_prompt_truncated"]))
        dropped_documents += int(scored.meta["scoring_dropped_documents"])
        for doc in scored.docs:
            add_to_aggregates(aggregates, doc)

    output_dir = Path(output_dir).expanduser()
    rows = [score_doc_row(aggregates[key], config) for key in sorted(aggregates)]
    write_jsonl(output_dir / "score_store.jsonl", rows)
    summary = {
        "queries": len(examples),
        "documents": len(rows),
        "scoring": "attention",
        "scoring_model_name": config.scoring_model_name,
        "reva_config": config_meta(config),
        "scoring_prompt_truncated_queries": truncated_queries,
        "scoring_dropped_documents": dropped_documents,
        "score_store_path": str(output_dir / "score_store.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


class REVAQueryAwareCompressor(Compressor):
    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.reva_config = read_config(config.options, config.budget, model_name)
        self.scoring_tokenizer, self.scoring_model = load_scoring_model(self.reva_config)

    def compress(self, example: Example) -> CompressionResult:
        scored = score_prompt_with_meta(example, self.scoring_tokenizer, self.scoring_model, self.reva_config)
        text, meta = render_selection(
            select(scored.docs, self.reva_config.budget, self.reva_config),
            self.tokenizer or self.scoring_tokenizer,
        )
        meta["backend"] = "reva"
        meta["scoring"] = "attention"
        meta["scoring_model_name"] = self.reva_config.scoring_model_name
        meta.update(scored.meta)
        return result("reva_query_aware", example.context_text, text, self.tokenizer or self.scoring_tokenizer, meta)


class REVAOfflineCompressor(Compressor):
    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.reva_config = read_config(config.options, config.budget, model_name)
        if not self.reva_config.score_store_path:
            raise ValueError("reva requires --option score_store_path=PATH")
        self.store = load_score_store(str(self.reva_config.score_store_path))

    def compress(self, example: Example) -> CompressionResult:
        docs = []
        for context in example.contexts:
            score_doc = self.store.find(context)
            if score_doc is None:
                text = truncate_text(context.text, self.reva_config.per_doc_quota or self.config.budget, self.tokenizer)
                units = build_word_units(text, self.tokenizer)
                raw_tokens = token_count(context.text, self.tokenizer)
                docs.append(DocumentUnits(context, units, "fallback", raw_tokens))
            else:
                validate_score_doc(score_doc, context)
                units = reconstruct_units(score_doc)
                docs.append(DocumentUnits(context, units, "score_store", sum(unit.token_count for unit in units)))
        text, meta = render_selection(select(docs, self.reva_config.budget, self.reva_config), self.tokenizer)
        meta["score_store_path"] = self.reva_config.score_store_path
        meta["score_store_documents"] = len(self.store.docs)
        return result("reva", example.context_text, text, self.tokenizer, meta)
