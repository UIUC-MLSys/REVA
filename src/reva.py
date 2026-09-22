from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines import result, truncate_text
from metrics import token_count
from schema import CompressionResult, Compressor, Context, Example, load_examples, read_jsonl, write_json, write_jsonl

WORD_RE = re.compile(r"[A-Za-z0-9]+")
PREFIX_TOKENS = 4
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
SEMANTIC_RE = re.compile(
    rf"\d{{1,2}}\s+(?:{MONTHS})\s+\d{{2,4}}"
    rf"|(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{2,4}}"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*)+"
)


def chunk_key(doc_id: Any, chunk_id: Any = None) -> str:
    chunk = "" if chunk_id is None else str(chunk_id).strip()
    return f"{doc_id}::chunk={chunk}" if chunk else str(doc_id)


def doc_key(context: Context) -> str:
    key = chunk_key(context.doc_id, context.chunk_id)
    return f"{context.dataset}::{key}" if context.dataset else key


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
    scoring_mode: str = "query_only"
    scoring_doc_order: str = "original"
    query_weight: float = 0.5
    answer_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.scoring_mode not in {"query_only", "query_plus_answer"}:
            raise ValueError("scoring_mode must be query_only or query_plus_answer")
        if self.scoring_doc_order not in {"original", "reversed"}:
            raise ValueError("scoring_doc_order must be original or reversed")
        if self.scoring_mode == "query_only":
            self.query_weight, self.answer_weight = 1.0, 0.0
        if any(not math.isfinite(weight) or weight < 0 for weight in (self.query_weight, self.answer_weight)):
            raise ValueError("query_weight and answer_weight must be finite and non-negative")
        if self.query_weight + self.answer_weight == 0:
            raise ValueError("query_weight and answer_weight cannot both be zero")


@dataclass(slots=True)
class PromptDoc:
    context: Context
    start: int
    end: int


@dataclass(slots=True)
class PromptInput:
    token_ids: list[int]
    docs: list[PromptDoc]
    query_positions: list[int]
    answer_positions: list[int]


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
    scores: list[float] | None = None
    hits: int = 0


@dataclass(slots=True)
class ScoreStore:
    docs: dict[str, ScoreDoc]
    compact: bool = False

    def find(self, context: Context) -> ScoreDoc | None:
        doc = self.docs.get(doc_key(context))
        if doc is not None:
            return doc
        if self.compact and not context.dataset:
            raise ValueError("compact scores require a dataset; pass --dataset or include it in input rows")
        if context.dataset and chunk_key(context.doc_id, context.chunk_id) in self.docs:
            raise ValueError("unscoped scores require loading with contexts from a single dataset")
        return None


@dataclass(slots=True)
class SelectionResult:
    docs: list[DocumentUnits]
    quotas: list[int]
    budget: int | None = None


@dataclass(slots=True)
class DocAggregate:
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
        "word_unit_type": "tokenizer_word_starts_semantic",
        "scoring_mode": config.scoring_mode,
        "scoring_doc_order": config.scoring_doc_order,
        "query_weight": config.query_weight,
        "answer_weight": config.answer_weight,
        "prompt_tokenization": "segments",
        "score_normalization": "none",
        "exclude_first_n_per_chunk": PREFIX_TOKENS,
    }


def context_header(context: Context, rank: int) -> str:
    rank = rank if context.rank is None else context.rank
    title = f" | title={context.title}" if context.title else ""
    return f"[DOC {rank} | id={context.doc_id}{title}]".strip()


def flat_list(values: Any) -> list[Any]:
    return list(values[0]) if values and isinstance(values[0], list) else list(values)


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(token_id) for token_id in flat_list(encoded["input_ids"])]
    offsets = [tuple(offset) for offset in flat_list(encoded["offset_mapping"])]
    return input_ids, [(int(start), int(end)) for start, end in offsets]


def document_text(text: str) -> str:
    return text.strip() + "\n\n"


def scoring_answer(example: Example, config: REVAConfig) -> str | None:
    if config.scoring_mode == "query_only":
        return None
    if example.source_response and example.source_response.strip():
        return example.source_response.strip()
    answer = next((str(answer).strip() for answer in example.answers or [] if str(answer).strip()), None)
    if answer is None:
        raise ValueError(f"example {example.id!r}: query_plus_answer requires source_response or a non-empty answer")
    return answer


def build_scoring_prompt(
    example: Example,
    tokenizer: Any,
    config: REVAConfig,
    contexts: list[Context] | None = None,
) -> PromptInput:
    if not example.question.strip():
        raise ValueError(f"example {example.id!r}: attention scoring requires a non-empty question")
    answer = scoring_answer(example, config)
    token_ids = tokenizer.encode("Contexts:\n", add_special_tokens=False)
    docs = []
    indexed = list(enumerate(example.contexts if contexts is None else contexts, start=1))
    if config.scoring_doc_order == "reversed":
        indexed.reverse()
    for rank, context in indexed:
        token_ids.extend(tokenizer.encode(context_header(context, rank) + "\n", add_special_tokens=False))
        start = len(token_ids)
        token_ids.extend(tokenizer.encode(document_text(context.text), add_special_tokens=False))
        docs.append(PromptDoc(context, start, len(token_ids)))

    token_ids.extend(tokenizer.encode("Question:\n", add_special_tokens=False))
    query_start = len(token_ids)
    token_ids.extend(tokenizer.encode(document_text(example.question), add_special_tokens=False))
    query_positions = list(range(query_start, len(token_ids)))
    answer_positions = []
    if answer is not None:
        token_ids.extend(tokenizer.encode("Known response:\n", add_special_tokens=False))
        answer_start = len(token_ids)
        token_ids.extend(tokenizer.encode(document_text(answer), add_special_tokens=False))
        answer_positions = list(range(answer_start, len(token_ids)))
    return PromptInput(token_ids, docs, query_positions, answer_positions)


def limited_scoring_prompt(example: Example, tokenizer: Any, config: REVAConfig) -> tuple[PromptInput, dict[str, Any]]:
    contexts = list(example.contexts)
    prompt = build_scoring_prompt(example, tokenizer, config, contexts)
    before = len(prompt.token_ids)
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
        contexts = contexts[:-1] if config.scoring_doc_order == "reversed" else contexts[1:]
        dropped += 1
        prompt = build_scoring_prompt(example, tokenizer, config, contexts)
        current = len(prompt.token_ids)
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


def join_units(units: list[WordUnit]) -> str:
    text = ""
    for unit in units:
        piece = unit.text.strip()
        if not piece:
            continue
        previous = text.rsplit(" ", 1)[-1]
        number = previous[:-1].isdigit() and piece.isdigit()
        joins_number = number and (text.endswith(".") or (text.endswith(",") and len(piece) == 3))
        if not text:
            text = piece
        elif piece in {".", ",", ";", ":", ")", "]", "}"} or text.endswith(("-", "'", "/", "(", "[", "{")) or joins_number:
            text += piece
        else:
            text += " " + piece
    return text


def build_word_units(
    text: str,
    tokenizer: Any,
    scores: list[float] | None = None,
    word_score: str = "max",
) -> list[WordUnit]:
    # Scoring and cache reconstruction use exactly the same document token sequence.
    token_ids, offsets = tokenize_with_offsets(tokenizer, document_text(text))
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    starts = [0] + [
        index for index, piece in enumerate(pieces)
        if index and (piece[:1].isspace() or piece.startswith(("\u0120", "\u2581")))
    ]
    leading = len(text) - len(text.lstrip())
    units = []
    for index, (start, end) in enumerate(zip(starts, starts[1:] + [len(token_ids)])):
        first = pieces[start].replace("\u0120", " ").replace("\u2581", " ")
        if first.startswith("##"):
            first = first[2:]
        tail = "".join(piece.replace("\u0120", "").replace("\u2581", "").replace("##", "") for piece in pieces[start + 1:end])
        units.append(WordUnit(
            text=(first.strip() + tail).strip(),
            index=index,
            char_start=min(len(text), leading + offsets[start][0]),
            char_end=min(len(text), leading + offsets[end - 1][1]),
            token_start=start,
            token_end=end,
            token_ids=token_ids[start:end],
            token_count=end - start,
            score=aggregate(scores[start:end], word_score) if scores is not None else 0.0,
        ))

    merged = []
    index = 0
    while index < len(units):
        best_end = index + 1
        for end in range(index + 2, min(len(units), index + 8) + 1):
            candidate = " ".join(join_units(units[index:end]).split())
            if SEMANTIC_RE.fullmatch(candidate) or SEMANTIC_RE.fullmatch(candidate.rstrip(".;:")):
                best_end = end
        group = units[index:best_end]
        start, end = group[0].token_start, group[-1].token_end
        merged.append(WordUnit(
            text=join_units(group),
            index=len(merged),
            char_start=group[0].char_start,
            char_end=group[-1].char_end,
            token_start=start,
            token_end=end,
            token_ids=token_ids[start:end],
            token_count=end - start,
            score=max(unit.score for unit in group),
        ))
        index = best_end
    return merged


def build_prompt_word_units(
    prompt_doc: PromptDoc,
    tokenizer: Any,
    scores: dict[int, float],
    config: REVAConfig,
) -> DocumentUnits:
    values = [scores.get(position, 0.0) for position in range(prompt_doc.start, prompt_doc.end)]
    units = build_word_units(prompt_doc.context.text, tokenizer, values, config.word_score)
    return DocumentUnits(prompt_doc.context, units, "prompt", len(values))


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


def source_scores(
    attentions: tuple[Any, ...] | list[Any],
    source_positions: list[int],
    document_positions: list[int],
    config: REVAConfig,
) -> Any:
    if not source_positions or not document_positions:
        return torch.zeros(len(document_positions))
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
        layer_scores.append(reduce_sources(reduce_heads(scores, config), config).cpu())
    return torch.stack(layer_scores).mean(dim=0)


def token_scores(
    attentions: tuple[Any, ...] | list[Any],
    source_positions: list[int],
    document_positions: list[int],
    config: REVAConfig,
    answer_positions: list[int] | None = None,
) -> dict[int, float]:
    scores = source_scores(attentions, source_positions, document_positions, config)
    if config.scoring_mode == "query_plus_answer" and answer_positions:
        answer_scores = source_scores(attentions, answer_positions, document_positions, config)
        scores = (config.query_weight * scores + config.answer_weight * answer_scores) / (config.query_weight + config.answer_weight)
    values = scores.tolist()
    return {position: round(float(value), 6) for position, value in zip(document_positions, values)}


def score_prompt_with_meta(example: Example, tokenizer: Any, model: Any, config: REVAConfig) -> ScoredDocs:
    prompt, meta = limited_scoring_prompt(example, tokenizer, config)
    # Match the released scorer: suppress the first four tokens without removing their units.
    document_positions = [
        position for doc in prompt.docs
        for position in range(doc.start + (PREFIX_TOKENS if doc.end - doc.start > PREFIX_TOKENS else 0), doc.end)
    ]
    if not prompt.query_positions:
        raise ValueError("reva could not find query tokens for attention scoring")
    if document_positions and min(prompt.query_positions) <= max(document_positions):
        raise ValueError("reva scoring prompt must place query tokens after document tokens")
    scores = token_scores(
        run_attentions(model, prompt.token_ids), prompt.query_positions, document_positions, config, prompt.answer_positions,
    )
    docs = [build_prompt_word_units(doc, tokenizer, scores, config) for doc in prompt.docs]
    if config.scoring_doc_order == "reversed":
        docs.reverse()
    meta["scored_document_tokens"] = len(document_positions)
    meta["source_tokens"] = len(prompt.query_positions) + len(prompt.answer_positions)
    meta["query_tokens"] = len(prompt.query_positions)
    meta["answer_tokens"] = len(prompt.answer_positions)
    meta["scoring_mode"] = config.scoring_mode
    meta["scoring_doc_order"] = config.scoring_doc_order
    meta["query_weight"] = config.query_weight
    meta["answer_weight"] = config.answer_weight
    if config.scoring_mode == "query_plus_answer":
        meta["scoring_answer_source"] = "source_response" if example.source_response and example.source_response.strip() else "answers"
    return ScoredDocs(docs, meta)


def unit_score(row: dict[str, Any]) -> float:
    if row.get("score_count"):
        return float(row.get("score_sum", 0.0)) / float(row["score_count"])
    return float(row.get("score", 0.0))


def score_key(row: dict[str, Any]) -> str:
    if "scores" in row:
        if not row.get("dataset"):
            raise ValueError("compact score-store rows require dataset")
        key = chunk_key(row["doc_id"], row.get("chunk_id"))
        return f"{row['dataset']}::{key}"
    key = str(row.get("doc_key") or row.get("id") or chunk_key(row["doc_id"], row.get("chunk_id")))
    if row.get("dataset") and not key.startswith(f"{row['dataset']}::"):
        key = f"{row['dataset']}::{key}"
    return key


def parse_score_doc(row: dict[str, Any]) -> ScoreDoc:
    if "scores" not in row and "word_units" not in row:
        raise ValueError("score-store rows must contain scores or word_units")
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
    return ScoreDoc(
        doc_id=doc_id,
        key=score_key(row),
        units=units,
        chunk_id=None if row.get("chunk_id") is None else str(row["chunk_id"]).strip() or None,
        raw_text=str(row.get("raw_text", "")),
        scores=[float(score) for score in row["scores"]] if "scores" in row else None,
        hits=int(row.get("hits", row.get("score_store_meta", {}).get("training_hits", 0))),
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
        aggregates[key] = DocAggregate(doc.context, [copy_store_unit(unit) for unit in doc.units])
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


def score_doc_row(aggregate: DocAggregate) -> dict[str, Any]:
    row = {
        "dataset": aggregate.context.dataset,
        "doc_id": aggregate.context.doc_id,
        "hits": aggregate.query_count,
        "scores": [round(unit.score, 6) for unit in aggregate.units],
    }
    if aggregate.context.chunk_id:
        row["chunk_id"] = aggregate.context.chunk_id
    return row


def validate_score_doc(score_doc: ScoreDoc, context: Context) -> None:
    if score_doc.doc_id != context.doc_id:
        raise ValueError(f"score_store key={score_doc.key!r} has a different doc_id")
    if score_doc.chunk_id is not None and context.chunk_id is not None and score_doc.chunk_id != context.chunk_id:
        raise ValueError(f"score_store key={score_doc.key!r} has a different chunk_id")
    if score_doc.raw_text and score_doc.raw_text != context.text:
        raise ValueError(f"score_store key={score_doc.key!r} has different raw_text")


def reconstruct_units(score_doc: ScoreDoc, context: Context, tokenizer: Any | None = None) -> list[WordUnit]:
    if score_doc.scores is not None and not score_doc.units:
        if tokenizer is None:
            raise ValueError("compact scores require the original tokenizer; pass --model or --option scoring_model_name=MODEL")
        units = build_word_units(context.text, tokenizer)
        if len(units) != len(score_doc.scores):
            raise ValueError(
                f"score-store key={score_doc.key!r}: {len(score_doc.scores)} scores but {len(units)} word units; "
                "use the original corpus text and scoring model tokenizer"
            )
        for unit, score in zip(units, score_doc.scores):
            unit.score = score
        score_doc.units = units
        score_doc.raw_text = context.text
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
    return SelectionResult(selected, quotas, budget)


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


def render_docs(selection: SelectionResult, tokenizer: Any | None = None) -> str:
    parts = []
    for doc, quota in zip(selection.docs, selection.quotas):
        text = truncate_text(doc.context.text, quota, tokenizer) if doc.source == "fallback" else render_units(doc)
        # Retokenization and document separators can exceed the selected units' token cost.
        while text and (
            token_count(text, tokenizer) > quota
            or (selection.budget is not None and token_count("\n\n".join([*parts, text]), tokenizer) > selection.budget)
        ):
            if doc.source == "fallback":
                quota = max(0, quota - 1)
                text = truncate_text(doc.context.text, quota, tokenizer)
            else:
                doc.units.remove(min(doc.units, key=lambda unit: (unit.score, -unit.index)))
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
        "fallback_documents": sum(doc.source == "fallback" for doc in selection.docs),
        "realized_budget": rendered_tokens,
        "selected_unit_tokens": selected_unit_tokens,
        "rendered_tokens": rendered_tokens,
    }


def render_selection(selection: SelectionResult, tokenizer: Any | None = None) -> tuple[str, dict[str, Any]]:
    text = render_docs(selection, tokenizer)
    return text, selection_meta(selection, text, tokenizer)


def load_score_store(path: str | Path, contexts: list[Context] | None = None) -> ScoreStore:
    docs = {}
    compact = False
    wanted = None if contexts is None else {doc_key(context) for context in contexts}
    if wanted == set():
        return ScoreStore(docs)
    unscoped = {chunk_key(context.doc_id, context.chunk_id) for context in contexts or []}
    datasets = {context.dataset for context in contexts or []}
    missing_dataset = {chunk_key(context.doc_id, context.chunk_id) for context in contexts or [] if not context.dataset}
    for row in read_jsonl(path):
        compact = compact or "scores" in row
        if "scores" in row and chunk_key(row["doc_id"], row.get("chunk_id")) in missing_dataset:
            raise ValueError("compact scores are dataset-specific; pass --dataset or include dataset in every input row")
        key = score_key(row)
        if wanted is not None and key not in wanted and not (not row.get("dataset") and key in unscoped):
            continue
        doc = parse_score_doc(row)
        if not row.get("dataset") and key in unscoped:
            if len(datasets) > 1:
                raise ValueError("unscoped scores require a single dataset; run each dataset separately")
            dataset = next(iter(datasets))
            if dataset:
                doc.key = f"{dataset}::{key}"
        if doc.key in docs:
            raise ValueError(f"duplicate score_store key: {doc.key}")
        docs[doc.key] = doc
    # An all-unseen compact cache still needs its tokenizer for prefix fallback.
    compact = any(doc.scores is not None for doc in docs.values()) if docs else compact
    return ScoreStore(docs, compact)


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
        scoring_mode=str(options.get("scoring_mode", "query_only")).strip().lower(),
        scoring_doc_order=str(options.get("scoring_doc_order", "original")).strip().lower(),
        query_weight=float(options.get("query_weight", 0.5)),
        answer_weight=float(options.get("answer_weight", 0.5)),
    )


def build_store(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
    options: dict[str, Any] | None = None,
    limit: int | None = None,
    top_k: int | None = None,
    corpus_path: str | Path | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")
    config = read_config(options, None, model_name)
    aggregates: dict[str, DocAggregate] = {}
    examples = load_examples(input_path, limit, top_k, corpus_path, dataset, split)
    if any(not example.dataset for example in examples):
        raise ValueError("build-store requires --dataset or dataset in every input row")
    for example in examples:
        scoring_answer(example, config)
    tokenizer, model = load_scoring_model(config)
    truncated_queries = 0
    dropped_documents = 0
    for example in examples:
        scored = score_prompt_with_meta(example, tokenizer, model, config)
        truncated_queries += int(bool(scored.meta["scoring_prompt_truncated"]))
        dropped_documents += int(scored.meta["scoring_dropped_documents"])
        for doc in scored.docs:
            add_to_aggregates(aggregates, doc)

    output_dir = Path(output_dir).expanduser()
    store_path = output_dir / "score_store.jsonl.gz"
    write_jsonl(store_path, (score_doc_row(aggregates[key]) for key in sorted(aggregates)))
    summary = {
        "queries": len(examples),
        "documents": len(aggregates),
        "scoring": "attention",
        "scoring_model_name": config.scoring_model_name,
        "reva_config": config_meta(config),
        "score_store_schema": ["dataset", "doc_id", "hits", "scores"],
        "top_k": top_k,
        "scoring_prompt_truncated_queries": truncated_queries,
        "scoring_dropped_documents": dropped_documents,
        "score_store_path": str(store_path),
    }
    if any(aggregate.context.chunk_id for aggregate in aggregates.values()):
        summary["score_store_schema"].append("chunk_id")
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
    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None, contexts: list[Context] | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.reva_config = read_config(config.options, config.budget, model_name)
        if not self.reva_config.score_store_path:
            raise ValueError("reva requires --option score_store_path=PATH")
        self.store = load_score_store(str(self.reva_config.score_store_path), contexts)
        self.store_tokenizer = tokenizer
        name = self.reva_config.scoring_model_name
        if name and (tokenizer is None or name != model_name):
            self.store_tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
        if self.store.compact:
            if self.store_tokenizer is None:
                raise ValueError("compact scores require --model or --option scoring_model_name=MODEL")
            if not self.store_tokenizer.is_fast:
                raise ValueError("compact scores require a fast tokenizer for offset mapping")
        self.tokenizer = self.tokenizer or self.store_tokenizer

    def compress(self, example: Example) -> CompressionResult:
        docs = []
        for context in example.contexts:
            score_doc = self.store.find(context)
            if score_doc is None:
                raw_tokens = token_count(context.text, self.tokenizer)
                docs.append(DocumentUnits(context, [], "fallback", raw_tokens))
            else:
                validate_score_doc(score_doc, context)
                units = reconstruct_units(score_doc, context, self.store_tokenizer)
                docs.append(DocumentUnits(context, units, "score_store", sum(unit.token_count for unit in units)))
        text, meta = render_selection(select(docs, self.reva_config.budget, self.reva_config), self.tokenizer)
        meta["score_store_path"] = self.reva_config.score_store_path
        meta["score_store_documents"] = len(self.store.docs)
        return result("reva", example.context_text, text, self.tokenizer, meta)
