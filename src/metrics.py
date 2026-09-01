from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any

from rouge import Rouge

WORD_RE = re.compile(r"[A-Za-z0-9]+")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
SPECIAL = {"yes", "no", "noanswer"}


def token_count(text: str, tokenizer: Any | None = None) -> int:
    if tokenizer is None:
        return len(WORD_RE.findall(text))
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str] | None) -> float:
    pred = normalize_answer(prediction)
    return max((float(pred == normalize_answer(answer)) for answer in answers or []), default=0.0)


def _pair_f1(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(answer)
    if pred in SPECIAL and pred != gold:
        return 0.0
    if gold in SPECIAL and pred != gold:
        return 0.0
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def f1(prediction: str, answers: list[str] | None) -> float:
    return max((_pair_f1(prediction, answer) for answer in answers or []), default=0.0)


def rouge_l(prediction: str, answers: list[str] | None) -> float:
    scorer = Rouge()
    prediction = " ".join(prediction.split())
    values = []
    for answer in answers or []:
        answer = " ".join(answer.split())
        if prediction and answer:
            values.append(scorer.get_scores(prediction, answer)[0]["rouge-l"]["f"])
    return max(values, default=0.0)


def score(prediction: str, answers: list[str] | None) -> dict[str, float]:
    return {
        "em": exact_match(prediction, answers),
        "f1": f1(prediction, answers),
        "rouge_l": rouge_l(prediction, answers),
    }


def compression_ratio(raw_tokens: int, compressed_tokens: int) -> float:
    return raw_tokens / compressed_tokens if compressed_tokens else 0.0


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    rank = (len(values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(values[low])
    weight = rank - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [record["raw_tokens"] for record in records]
    compressed = [record["compressed_tokens"] for record in records]
    ratios = [compression_ratio(r, c) for r, c in zip(raw, compressed) if c > 0]
    summary = {
        "method": records[0]["method"],
        "queries": len(records),
        "raw_tokens": aggregate(raw),
        "compressed_tokens": aggregate(compressed),
        "compression_ratio": aggregate(ratios) if ratios else None,
        "tokens_saved": aggregate([r - c for r, c in zip(raw, compressed)]),
        "compression_ms": aggregate([record["compression_ms"] for record in records]),
    }
    if "prediction" in records[0]:
        summary["em"] = aggregate([record["em"] for record in records])
        summary["f1"] = aggregate([record["f1"] for record in records])
        summary["rouge_l"] = aggregate([record["rouge_l"] for record in records])
        summary["generation_ms"] = aggregate([record["generation_ms"] for record in records])
    return summary
