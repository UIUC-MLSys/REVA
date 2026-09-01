from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from metrics import token_count
from schema import CompressionResult, Compressor, Context, Example

WORD_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def truncate_text(text: str, budget: int | None, tokenizer: Any | None = None) -> str:
    if budget is None:
        return text.strip()
    if budget <= 0:
        return ""
    if tokenizer is not None:
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        return tokenizer.decode(list(ids)[:budget], skip_special_tokens=True).strip()
    matches = list(WORD_RE.finditer(text))
    return text[: matches[budget - 1].end()].strip() if budget < len(matches) else text.strip()


def result(method: str, raw_text: str, text: str, tokenizer: Any | None = None, meta: dict[str, Any] | None = None) -> CompressionResult:
    return CompressionResult(method, text, token_count(raw_text, tokenizer), token_count(text, tokenizer), meta)


def cap(text: str, budget: int | None, tokenizer: Any | None) -> str:
    return truncate_text(text, budget, tokenizer) if budget is not None and token_count(text, tokenizer) > budget else text.strip()


def shared_budget(total: int | None, docs: int) -> int | None:
    if total is None or docs <= 0:
        return None
    if total == 0:
        return 0
    return max(1, total // docs)


def bool_opt(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "y"}


def checkpoint(options: dict[str, Any], table: dict[str, str]) -> str:
    if options.get("checkpoint"):
        return str(options["checkpoint"])
    dataset = str(options.get("dataset_name") or "nq").lower().replace("_", "")
    return table.get(dataset, table["nq"])


def device_map(options: dict[str, Any]) -> Any:
    if "device_map" in options:
        return options["device_map"]
    return "auto" if torch.cuda.is_available() else None


def load_causal_lm(model_name: str, options: dict[str, Any]) -> Any:
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(options.get("dtype", "auto")).lower())
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if device_map(options) is not None:
        kwargs["device_map"] = device_map(options)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def model_device(model: Any) -> Any:
    return next(model.parameters()).device


def to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def set_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    return list(ids[0] if ids and isinstance(ids[0], list) else ids)


def opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    text = "\n\n".join(f"{message['role'].title()}: {message['content']}" for message in messages)
    return f"{text}\n\nAssistant:" if add_generation_prompt else text


def greedy_budget(parts: list[str], budget: int, tokenizer: Any | None) -> str:
    kept = []
    used = 0
    for part in parts:
        cost = token_count(part, tokenizer)
        if used + cost <= budget:
            kept.append(part)
            used += cost
        elif not kept:
            return truncate_text(part, budget, tokenizer)
    return " ".join(kept).strip()


class RawCompressor(Compressor):
    def compress(self, example: Example) -> CompressionResult:
        return result(self.config.method, example.context_text, example.context_text, self.tokenizer)


class TruncateCompressor(Compressor):
    def compress(self, example: Example) -> CompressionResult:
        text = truncate_text(example.context_text, self.config.budget, self.tokenizer)
        return result(self.config.method, example.context_text, text, self.tokenizer)


class SelectiveContextCompressor(Compressor):
    DEFAULTS = {"model_type": "gpt2", "lang": "en", "reduce_level": "phrase", "reduce_ratio": 0.35}

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        self.reducer = None

    def get_reducer(self):
        if self.reducer is None:
            reducer_cls = importlib.import_module("selective_context").SelectiveContext
            self.reducer = reducer_cls(model_type=str(self.options["model_type"]), lang=str(self.options["lang"]))
        return self.reducer

    def compress_text(self, raw_text: str, budget: int | None) -> tuple[str, dict[str, Any]]:
        raw_tokens = token_count(raw_text, self.tokenizer)
        if budget == 0:
            return "", {"backend": "selective_context", "target_token": 0}
        if budget is not None and budget >= raw_tokens:
            return raw_text.strip(), {"backend": "selective_context", "target_token": budget, "reduced": False}
        ratio = float(self.options["reduce_ratio"]) if budget is None else 1.0 - budget / max(raw_tokens, 1)
        ratio = min(1.0, max(0.0, ratio))
        text, removed = self.get_reducer()(raw_text, reduce_ratio=ratio, reduce_level=str(self.options["reduce_level"]))
        text = cap(text, budget, self.tokenizer)
        return text, {
            "backend": "selective_context",
            "model_type": self.options["model_type"],
            "lang": self.options["lang"],
            "reduce_level": self.options["reduce_level"],
            "reduce_ratio": ratio,
            "removed_count": len(removed or []),
        }

    def compress(self, example: Example) -> CompressionResult:
        text, meta = self.compress_text(example.context_text, self.config.budget)
        return result("selective_context", example.context_text, text, self.tokenizer, meta)


class SelectiveContextDocwiseCompressor(SelectiveContextCompressor):
    def compress(self, example: Example) -> CompressionResult:
        quota = shared_budget(self.config.budget, len(example.contexts))
        parts = []
        docs = []
        for context in example.contexts:
            text, meta = self.compress_text(context.block, quota)
            if text:
                parts.append(text)
            docs.append({"doc_id": context.doc_id, "rank": context.rank, **meta})
        text = cap("\n\n".join(parts), self.config.budget, self.tokenizer)
        return result(
            "selective_context_docwise",
            example.context_text,
            text,
            self.tokenizer,
            {"base_method": "selective_context", "per_doc_budget": quota, "documents": docs},
        )


class LLMLinguaCompressor(Compressor):
    DEFAULTS = {
        "model_name": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "rate": 0.5,
        "use_context_level_filter": False,
        "use_token_level_filter": True,
        "force_tokens": ["\n", "?"],
        "chunk_end_tokens": [".", "\n"],
    }

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        if not 0.0 <= float(self.options["rate"]) <= 1.0:
            raise ValueError("llmlingua2 rate must be between 0 and 1")
        self.compressor = None

    def list_opt(self, value: Any) -> list[Any]:
        return list(value) if isinstance(value, list | tuple) else [value]

    def get_compressor(self):
        if self.compressor is None:
            cls = importlib.import_module("llmlingua").PromptCompressor
            self.compressor = cls(
                model_name=str(self.options["model_name"]),
                device_map=self.options.get("device_map") or ("cuda" if torch.cuda.is_available() else "cpu"),
                use_llmlingua2=True,
            )
        return self.compressor

    def compress_blocks(self, blocks: list[str], budget: int | None) -> tuple[str, dict[str, Any]]:
        raw_text = "\n\n".join(blocks).strip()
        if budget == 0 or not raw_text:
            return "", {"backend": "llmlingua2", "target_token": 0}
        payload = {
            "use_context_level_filter": bool_opt(self.options["use_context_level_filter"]),
            "use_token_level_filter": bool_opt(self.options["use_token_level_filter"]),
            "force_tokens": self.list_opt(self.options["force_tokens"]),
            "chunk_end_tokens": self.list_opt(self.options["chunk_end_tokens"]),
        }
        payload["target_token" if budget is not None else "rate"] = budget if budget is not None else float(self.options["rate"])
        compressor = self.get_compressor()
        response = compressor.compress_prompt_llmlingua2(blocks, **payload) if hasattr(compressor, "compress_prompt_llmlingua2") else compressor.compress_prompt(blocks, **payload)
        text = cap(str(response["compressed_prompt"]), budget, self.tokenizer)
        return text, {
            "backend": "llmlingua2",
            "model_name": self.options["model_name"],
            "target_token": budget,
            "origin_tokens": response.get("origin_tokens"),
            "compressor_tokens": response.get("compressed_tokens"),
        }

    def compress(self, example: Example) -> CompressionResult:
        text, meta = self.compress_blocks([context.block for context in example.contexts], self.config.budget)
        return result("llmlingua2", example.context_text, text, self.tokenizer, meta)


class LLMLingua2DocwiseCompressor(LLMLinguaCompressor):
    def compress(self, example: Example) -> CompressionResult:
        quota = shared_budget(self.config.budget, len(example.contexts))
        parts = []
        docs = []
        for context in example.contexts:
            text, meta = self.compress_blocks([context.block], quota)
            if text:
                parts.append(text)
            docs.append({"doc_id": context.doc_id, "rank": context.rank, **meta})
        text = cap("\n\n".join(parts), self.config.budget, self.tokenizer)
        return result(
            "llmlingua2_docwise",
            example.context_text,
            text,
            self.tokenizer,
            {"base_method": "llmlingua2", "per_doc_budget": quota, "documents": docs},
        )


class LongLLMLinguaCompressor(Compressor):
    DEFAULTS = {
        "model_name": "NousResearch/Llama-2-7b-hf",
        "rate": 0.55,
        "condition_in_question": "after_condition",
        "reorder_context": "sort",
        "rank_method": "longllmlingua",
        "dynamic_context_compression_ratio": 0.3,
    }

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        if not 0.0 <= float(self.options["rate"]) <= 1.0:
            raise ValueError("longllmlingua rate must be between 0 and 1")
        if not 0.0 <= float(self.options["dynamic_context_compression_ratio"]) <= 1.0:
            raise ValueError("longllmlingua dynamic_context_compression_ratio must be between 0 and 1")
        self.compressor = None

    def get_compressor(self):
        if self.compressor is None:
            cls = importlib.import_module("llmlingua").PromptCompressor
            self.compressor = cls(
                model_name=str(self.options["model_name"]),
                device_map=self.options.get("device_map") or ("cuda" if torch.cuda.is_available() else "cpu"),
            )
        return self.compressor

    def compress(self, example: Example) -> CompressionResult:
        if self.config.budget == 0 or not example.context_text:
            return result("longllmlingua", example.context_text, "", self.tokenizer, {"backend": "longllmlingua", "target_token": self.config.budget})
        payload = {
            "question": example.question,
            "condition_in_question": self.options["condition_in_question"],
            "reorder_context": self.options["reorder_context"],
            "rank_method": self.options["rank_method"],
            "dynamic_context_compression_ratio": float(self.options["dynamic_context_compression_ratio"]),
        }
        payload["target_token" if self.config.budget is not None else "rate"] = self.config.budget if self.config.budget is not None else float(self.options["rate"])
        response = self.get_compressor().compress_prompt([context.block for context in example.contexts], **payload)
        text = cap(str(response["compressed_prompt"]), self.config.budget, self.tokenizer)
        return result("longllmlingua", example.context_text, text, self.tokenizer, {
            "backend": "longllmlingua",
            "model_name": self.options["model_name"],
            "target_token": self.config.budget,
            "origin_tokens": response.get("origin_tokens"),
            "compressor_tokens": response.get("compressed_tokens"),
        })


class RecompExtractiveCompressor(Compressor):
    CHECKPOINTS = {
        "nq": "fangyuan/nq_extractive_compressor",
        "tqa": "fangyuan/tqa_extractive_compressor",
        "triviaqa": "fangyuan/tqa_extractive_compressor",
        "hotpotqa": "fangyuan/hotpotqa_extractive_compressor",
    }

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.checkpoint = checkpoint(config.options, self.CHECKPOINTS)
        self.batch_size = int(config.options.get("batch_size", 8))
        self.max_length = int(config.options.get("max_length", 512))
        self.top_sentences = int(config.options.get("top_sentences", 1))
        if self.batch_size <= 0:
            raise ValueError("recomp_extractive batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("recomp_extractive max_length must be positive")
        if self.top_sentences < -1:
            raise ValueError("recomp_extractive top_sentences must be -1 or non-negative")
        self.model = None
        self.infer_tokenizer = None

    def get_model(self):
        if self.model is None:
            self.infer_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.checkpoint, torch_dtype="auto", trust_remote_code=True, device_map=device_map(self.config.options))
            self.model.eval()
        return self.model, self.infer_tokenizer

    def sentence_candidates(self, contexts: list[Context], include_titles: bool) -> list[str]:
        out = []
        for context in contexts:
            for sentence in SENTENCE_RE.split(context.text.strip()):
                sentence = sentence.strip()
                if sentence:
                    title = f"{context.title} " if include_titles and context.title else ""
                    out.append(f"{title}{sentence}".strip())
        return out

    def embed(self, texts: list[str], model: Any, tokenizer: Any) -> Any:
        device = model_device(model)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**encoded)[0]
        mask = encoded["attention_mask"].unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, 0.0)
        return (hidden.sum(dim=1) / mask.sum(dim=1)).detach().cpu()

    def score(self, question: str, candidates: list[str]) -> list[float]:
        model, tokenizer = self.get_model()
        q = self.embed([question], model, tokenizer)[0]
        scores = []
        for start in range(0, len(candidates), self.batch_size):
            embs = self.embed(candidates[start : start + self.batch_size], model, tokenizer)
            scores.extend((q @ emb).item() for emb in embs)
        return scores

    def compress(self, example: Example) -> CompressionResult:
        candidates = self.sentence_candidates(example.contexts, bool_opt(self.config.options.get("include_titles", True)))
        if self.config.budget == 0 or not candidates or (self.config.budget is None and self.top_sentences == 0):
            return result("recomp_extractive", example.context_text, "", self.tokenizer, {"backend": "recomp_extractive", "candidates": len(candidates)})
        scores = self.score(example.question, candidates)
        ranked = [candidates[index] for index, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        selected = ranked if self.top_sentences == -1 else ranked[: self.top_sentences]
        text = greedy_budget(ranked, self.config.budget, self.tokenizer) if self.config.budget is not None else " ".join(selected)
        return result("recomp_extractive", example.context_text, text, self.tokenizer, {
            "backend": "recomp_extractive",
            "checkpoint": self.checkpoint,
            "candidates": len(candidates),
            "selected_tokens": token_count(text, self.tokenizer),
        })


class RecompAbstractiveCompressor(Compressor):
    CHECKPOINTS = {
        "nq": "fangyuan/nq_abstractive_compressor",
        "tqa": "fangyuan/tqa_abstractive_compressor",
        "triviaqa": "fangyuan/tqa_abstractive_compressor",
        "hotpotqa": "fangyuan/hotpotqa_abstractive",
    }

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.checkpoint = checkpoint(config.options, self.CHECKPOINTS)
        self.max_input_tokens = int(config.options.get("max_input_tokens", 1024))
        self.max_output_tokens = int(config.options.get("max_output_tokens", 512))
        if self.max_input_tokens <= 0:
            raise ValueError("recomp_abstractive max_input_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("recomp_abstractive max_output_tokens must be positive")
        self.model = None
        self.infer_tokenizer = None

    def get_model(self):
        if self.model is None:
            self.infer_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, torch_dtype="auto", trust_remote_code=True, device_map=device_map(self.config.options))
            self.model.eval()
        return self.model, self.infer_tokenizer

    def compress(self, example: Example) -> CompressionResult:
        if self.config.budget == 0:
            return result("recomp_abstractive", example.context_text, "", self.tokenizer, {"backend": "recomp_abstractive", "target_token": 0})
        model, tokenizer = self.get_model()
        prompt = f"Question: {example.question}\n Document: {example.context_text}\n Summary: "
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(next(model.parameters()).device)
        cap_tokens = self.config.budget if self.config.budget is not None else self.max_output_tokens
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=cap_tokens, do_sample=False)
        text = cap(tokenizer.decode(output[0], skip_special_tokens=True), self.config.budget, self.tokenizer)
        return result("recomp_abstractive", example.context_text, text, self.tokenizer, {
            "backend": "recomp_abstractive",
            "checkpoint": self.checkpoint,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": cap_tokens,
        })


class FaviCompCompressor(Compressor):
    DEFAULTS = {
        "compression_model_name": None,
        "target_model_name": None,
        "alpha": 0.5,
        "decoding_len": 128,
        "max_input_tokens": None,
        "device_map": "auto",
        "dtype": "auto",
        "add_generation_prompt": True,
        "include_doc_metadata": False,
    }
    EVIDENCE_SYSTEM = (
        "You are an expert in summarization. Given a question and multiple document snippets, "
        "generate one summarized context that is helpful to answer the question. Just summarize, no other words."
    )
    CONTEXT_SYSTEM = (
        "You are an expert in context generation. Given a question, generate a context that is helpful "
        "to answer the question. Just generate the context, no other words."
    )

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        model = self.options["compression_model_name"] or model_name
        if model is None:
            raise ValueError("favicomp needs --model or --option compression_model_name=...")
        self.model_name = str(model)
        self.target_name = str(self.options["target_model_name"] or self.model_name)
        self.alpha = float(self.options["alpha"])
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("favicomp alpha must be between 0 and 1")
        self.steps = int(config.budget if config.budget is not None else self.options["decoding_len"])
        self.max_input_tokens = opt_int(self.options["max_input_tokens"])
        self.add_generation_prompt = bool_opt(self.options["add_generation_prompt"])
        self.include_doc_metadata = bool_opt(self.options["include_doc_metadata"])
        self.infer_tokenizer = None
        self.target_tokenizer = None
        self.model = None
        self.target_model = None

    def chat_input(self, tokenizer: Any, messages: list[dict[str, str]], device: Any) -> dict[str, Any]:
        encoded = tokenizer(chat_text(tokenizer, messages, self.add_generation_prompt), return_tensors="pt")
        encoded = dict(encoded)
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        if self.max_input_tokens is not None and encoded["input_ids"].shape[-1] > self.max_input_tokens:
            encoded["input_ids"] = encoded["input_ids"][:, -self.max_input_tokens:]
            encoded["attention_mask"] = encoded["attention_mask"][:, -self.max_input_tokens:]
        return to_device(encoded, device)

    def same_tokenizer(self, left: Any, right: Any) -> bool:
        if left is right:
            return True
        attrs = ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")
        if any(getattr(left, attr, None) != getattr(right, attr, None) for attr in attrs):
            return False
        probes = ["Question: Who wrote Hamlet?", "Summarized Context:"]
        return all(token_ids(left, probe) == token_ids(right, probe) for probe in probes)

    def append_mask(self, mask: Any | None, ids: Any) -> Any | None:
        return None if mask is None else torch.cat([mask, torch.ones_like(ids)], dim=-1)

    def load(self) -> None:
        if self.model is not None:
            return
        self.infer_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        set_pad_token(self.infer_tokenizer)
        self.model = load_causal_lm(self.model_name, self.options)
        self.model.eval()
        if not self.alpha:
            return
        if self.target_name == self.model_name:
            self.target_tokenizer = self.infer_tokenizer
            self.target_model = self.model
        else:
            self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_name, trust_remote_code=True)
            set_pad_token(self.target_tokenizer)
            if not self.same_tokenizer(self.infer_tokenizer, self.target_tokenizer):
                raise ValueError("favicomp needs compression and target models with the same tokenizer")
            self.target_model = load_causal_lm(self.target_name, self.options)
            self.target_model.eval()

    def decode(self, evidence: list[dict[str, str]], context: list[dict[str, str]]) -> tuple[str, int]:
        evidence_input = self.chat_input(self.infer_tokenizer, evidence, model_device(self.model))
        context_input = None if not self.alpha else self.chat_input(self.target_tokenizer, context, model_device(self.target_model))
        evidence_ids = evidence_input["input_ids"]
        context_ids = None if context_input is None else context_input["input_ids"]
        evidence_mask = evidence_input.get("attention_mask")
        context_mask = None if context_input is None else context_input.get("attention_mask")
        evidence_cache = None
        context_cache = None
        generated = []
        eos_id = getattr(self.infer_tokenizer, "eos_token_id", None)
        for _ in range(self.steps):
            with torch.inference_mode():
                evidence_out = self.model(input_ids=evidence_ids, attention_mask=evidence_mask, past_key_values=evidence_cache, use_cache=True)
                evidence_cache = evidence_out.past_key_values
                logits = evidence_out.logits[:, -1, :]
                if self.alpha:
                    context_out = self.target_model(input_ids=context_ids, attention_mask=context_mask, past_key_values=context_cache, use_cache=True)
                    context_cache = context_out.past_key_values
                    if logits.shape[-1] != context_out.logits.shape[-1]:
                        raise ValueError("favicomp compression and target models need the same vocabulary size")
                    logits = logits * (1.0 - self.alpha) + context_out.logits[:, -1, :].to(logits.device) * self.alpha
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_id[0, 0])
            if eos_id is not None and token_id == eos_id:
                break
            generated.append(token_id)
            evidence_ids = next_id.to(model_device(self.model))
            evidence_mask = self.append_mask(evidence_mask, evidence_ids)
            if self.alpha:
                context_ids = next_id.to(model_device(self.target_model))
                context_mask = self.append_mask(context_mask, context_ids)
        return self.infer_tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)

    def meta(self, generated: int, empty: bool) -> dict[str, Any]:
        return {
            "backend": "favicomp",
            "compression_model_name": self.model_name,
            "target_model_name": self.target_name,
            "alpha": self.alpha,
            "generated_tokens": generated,
            "empty": empty,
        }

    def compress(self, example: Example) -> CompressionResult:
        raw_text = example.context_text
        if self.steps <= 0 or not raw_text:
            return result("favicomp", raw_text, "", self.tokenizer, self.meta(0, True))
        self.load()
        docs = raw_text if self.include_doc_metadata else "\n".join(context.text.strip() for context in example.contexts)
        evidence = [
            {"role": "system", "content": self.EVIDENCE_SYSTEM},
            {"role": "user", "content": f"Question: {example.question}\nDocuments: {docs}\nSummarized Context:"},
        ]
        context = [
            {"role": "system", "content": self.CONTEXT_SYSTEM},
            {"role": "user", "content": f"Question: {example.question}\nContext:"},
        ]
        text, generated = self.decode(evidence, context)
        text = cap(text, self.config.budget, self.tokenizer)
        return result("favicomp", raw_text, text, self.tokenizer, self.meta(generated, False))


class ExitCompressor(Compressor):
    DEFAULTS = {
        "base_model_name": "google/gemma-2b-it",
        "checkpoint": "doubleyyh/exit-gemma-2b",
        "threshold": 0.5,
        "batch_size": 8,
        "max_length": 4096,
        "device_map": "auto",
        "dtype": "float16",
        "include_titles": True,
    }

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        self.base_name = str(self.options["base_model_name"])
        self.checkpoint = self.options["checkpoint"]
        self.threshold = float(self.options["threshold"])
        self.batch_size = int(self.options["batch_size"])
        if self.batch_size <= 0:
            raise ValueError("exit batch_size must be positive")
        self.max_length = int(self.options["max_length"])
        self.include_titles = bool_opt(self.options["include_titles"])
        self.infer_tokenizer = None
        self.model = None
        self.yes_id = None
        self.no_id = None

    def prompt(self, query: str, full_context: str, sentence: str) -> str:
        return (
            "<start_of_turn>user\n"
            f"Query:\n{query}\nFull context:\n{full_context}\nSentence:\n{sentence}\n"
            'Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>\n'
            "<start_of_turn>model\n"
        )

    def sentences(self, example: Example) -> list[dict[str, Any]]:
        rows = []
        for context in example.contexts:
            text = f"{context.title}\n{context.text}" if self.include_titles and context.title else context.text
            for index, sentence in enumerate(SENTENCE_RE.split(text.strip())):
                if sentence.strip():
                    rows.append({"doc_id": context.doc_id, "rank": context.rank, "sentence_index": index, "text": sentence.strip()})
        return rows

    def first_token_id(self, text: str) -> int:
        return int(self.infer_tokenizer.encode(text, add_special_tokens=False)[0])

    def load(self) -> None:
        if self.model is not None:
            return
        self.infer_tokenizer = AutoTokenizer.from_pretrained(self.base_name, use_fast=True, trust_remote_code=True)
        set_pad_token(self.infer_tokenizer)
        self.infer_tokenizer.padding_side = "left"
        self.infer_tokenizer.truncation_side = "left"
        base = load_causal_lm(self.base_name, self.options)
        self.model = importlib.import_module("peft").PeftModel.from_pretrained(base, str(self.checkpoint)) if self.checkpoint else base
        self.model.eval()
        self.yes_id = self.first_token_id("Yes")
        self.no_id = self.first_token_id("No")

    def fit_context(self, query: str, full_context: str, sentence: str) -> str:
        fixed = self.prompt(query, "", sentence)
        room = max(0, self.max_length - len(token_ids(self.infer_tokenizer, fixed)) - 1)
        return truncate_text(full_context, room, self.infer_tokenizer)

    def scores(self, query: str, full_context: str, sentences: list[str]) -> list[float]:
        self.load()
        out = []
        for start in range(0, len(sentences), self.batch_size):
            prompts = [self.prompt(query, self.fit_context(query, full_context, sentence), sentence) for sentence in sentences[start : start + self.batch_size]]
            encoded = self.infer_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            encoded = to_device(dict(encoded), model_device(self.model))
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
            yes_no = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=1)
            out.extend(float(x) for x in torch.softmax(yes_no, dim=1)[:, 0].detach().cpu())
        return out

    def meta(self, sentence_count: int, selected_count: int) -> dict[str, Any]:
        return {
            "backend": "exit",
            "base_model_name": self.base_name,
            "checkpoint": self.checkpoint,
            "threshold": self.threshold,
            "sentence_count": sentence_count,
            "selected_count": selected_count,
        }

    def compress(self, example: Example) -> CompressionResult:
        raw_text = example.context_text
        if self.config.budget == 0 or not raw_text:
            return result("exit", raw_text, "", self.tokenizer, self.meta(0, 0))
        rows = self.sentences(example)
        scores = self.scores(example.question, " ".join(row["text"] for row in rows), [row["text"] for row in rows])
        kept = [row["text"] for row, score in zip(rows, scores) if score >= self.threshold]
        text = greedy_budget(kept, self.config.budget, self.tokenizer) if self.config.budget is not None else " ".join(kept)
        return result("exit", raw_text, text, self.tokenizer, self.meta(len(rows), len(kept)))


class LongRefinerCompressor(Compressor):
    DEFAULTS = {
        "base_model_path": "Qwen/Qwen2.5-3B-Instruct",
        "query_analysis_module_lora_path": "jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct",
        "doc_structuring_module_lora_path": "jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct",
        "global_selection_module_lora_path": "jinjiajie/Global-Selection-Qwen2.5-3B-Instruct",
        "score_model_name": "bge-reranker-v2-m3",
        "score_model_path": "BAAI/bge-reranker-v2-m3",
        "max_model_len": 25000,
        "budget": 2048,
        "ratio": None,
        "module_path": None,
    }

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        if self.options["module_path"]:
            sys.path.insert(0, str(Path(str(self.options["module_path"])).expanduser().resolve()))
        budget_value = config.budget if config.budget is not None else self.options["budget"]
        self.budget = None if budget_value is None else int(budget_value)
        self.ratio = None if self.options["ratio"] is None else float(self.options["ratio"])
        self.refiner = None

    def load(self) -> Any:
        if self.refiner is None:
            self.refiner = importlib.import_module("longrefiner").LongRefiner(
                base_model_path=str(self.options["base_model_path"]),
                query_analysis_module_lora_path=str(self.options["query_analysis_module_lora_path"]),
                doc_structuring_module_lora_path=str(self.options["doc_structuring_module_lora_path"]),
                global_selection_module_lora_path=str(self.options["global_selection_module_lora_path"]),
                score_model_name=str(self.options["score_model_name"]),
                score_model_path=str(self.options["score_model_path"]),
                max_model_len=int(self.options["max_model_len"]),
            )
        return self.refiner

    def documents(self, example: Example) -> list[dict[str, str]]:
        docs = []
        for context in example.contexts:
            title = context.title.strip() or context.doc_id
            contents = f"{title}\n{context.text.strip()}".strip()
            docs.append({"id": context.doc_id, "title": title, "rank": "" if context.rank is None else str(context.rank), "contents": contents})
        return docs

    def meta(self, segment_count: int) -> dict[str, Any]:
        return {
            "backend": "longrefiner",
            "base_model_path": self.options["base_model_path"],
            "budget": self.budget,
            "ratio": self.ratio,
            "segment_count": segment_count,
        }

    def compress(self, example: Example) -> CompressionResult:
        raw_text = example.context_text
        if self.budget == 0 or not raw_text:
            return result("longrefiner", raw_text, "", self.tokenizer, self.meta(0))
        refined = self.load().run(example.question, self.documents(example), budget=self.budget, ratio=self.ratio)
        parts = [refined.strip()] if isinstance(refined, str) else [str(item).strip() for item in refined if str(item).strip()]
        text = cap("\n\n".join(parts), self.budget, self.tokenizer)
        return result("longrefiner", raw_text, text, self.tokenizer, self.meta(len(parts)))


class CompActCompressor(Compressor):
    DEFAULTS = {
        "model_name": "cwyoon99/CompAct-7b",
        "segment_size": 5,
        "max_iteration": 6,
        "max_new_tokens": 900,
        "max_input_tokens": None,
        "device_map": "auto",
        "dtype": "bfloat16",
        "add_generation_prompt": True,
        "include_doc_metadata": True,
        "include_eval_reason": False,
    }
    FIRST_STEP = (
        "Generate a summary of the source documents to answer the question. "
        "Then write Evaluation: [ENOUGH] if the summary has enough detail, otherwise [MORE]."
    )
    NEXT_STEP = (
        "Refine the previous summary using the new source documents and the previous evaluation. "
        "Then write Evaluation: [ENOUGH] if the summary has enough detail, otherwise [MORE]."
    )

    def __init__(self, config, tokenizer: Any | None = None, model_name: str | None = None) -> None:
        super().__init__(config, tokenizer, model_name)
        self.options = {**self.DEFAULTS, **config.options}
        self.model_name = str(self.options["model_name"])
        self.segment_size = int(self.options["segment_size"])
        if self.segment_size <= 0:
            raise ValueError("compact segment_size must be positive")
        self.max_iteration = opt_int(self.options["max_iteration"])
        self.max_new_tokens = int(self.options["max_new_tokens"])
        if self.max_new_tokens <= 0:
            raise ValueError("compact max_new_tokens must be positive")
        self.max_input_tokens = opt_int(self.options["max_input_tokens"])
        self.add_generation_prompt = bool_opt(self.options["add_generation_prompt"])
        self.include_doc_metadata = bool_opt(self.options["include_doc_metadata"])
        self.include_eval_reason = bool_opt(self.options["include_eval_reason"])
        self.infer_tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.infer_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        set_pad_token(self.infer_tokenizer)
        self.model = load_causal_lm(self.model_name, self.options)
        self.model.eval()

    def strip_markers(self, text: str) -> str:
        return text.replace("[MORE]", "").replace("[ENOUGH]", "").replace("\n", " ").strip()

    def parse(self, text: str) -> tuple[str, str]:
        summary = re.search(r"(?:Summary:)?(.*?)(?=Evaluation:|$)", text, re.DOTALL)
        evaluation = re.search(r"Evaluation:(.*?)(?=Summary:|$)", text, re.DOTALL)
        return (summary.group(1).strip() if summary else "", evaluation.group(1).strip() if evaluation else "")

    def document_text(self, context: Context) -> str:
        title = f"{context.title.strip()} " if self.include_doc_metadata and context.title else ""
        return f"{title}{context.text.strip()}".strip()

    def prompt(self, question: str, docs: list[str], prev_summary: str, prev_eval: str) -> str:
        source = "\n".join(docs)
        if prev_summary:
            content = (
                f"{self.NEXT_STEP}\n\nQuestion: {question}\n\nPrevious summary: {prev_summary}\n\n"
                f"Evaluation of previous summary: {self.strip_markers(prev_eval)}\n\nSource documents: {source}\n\nSummary:"
            )
        else:
            content = f"{self.FIRST_STEP}\n\nQuestion: {question}\n\nSource documents: {source}\n\nSummary:"
        return chat_text(self.infer_tokenizer, [{"role": "user", "content": content}], self.add_generation_prompt)

    def generate(self, prompt: str) -> str:
        self.load()
        encoded = self.infer_tokenizer(prompt, return_tensors="pt")
        encoded = to_device(dict(encoded), model_device(self.model))
        if self.max_input_tokens is not None and encoded["input_ids"].shape[-1] > self.max_input_tokens:
            encoded["input_ids"] = encoded["input_ids"][:, -self.max_input_tokens :]
            encoded["attention_mask"] = encoded["attention_mask"][:, -self.max_input_tokens :]
        prompt_len = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.infer_tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.infer_tokenizer, "eos_token_id", None),
            )
        return self.infer_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    def meta(self, iterations: int, complete: bool) -> dict[str, Any]:
        return {
            "backend": "compact",
            "model_name": self.model_name,
            "segment_size": self.segment_size,
            "max_iteration": self.max_iteration,
            "iteration_count": iterations,
            "complete": complete,
        }

    def compress(self, example: Example) -> CompressionResult:
        raw_text = example.context_text
        if self.config.budget == 0 or not raw_text:
            return result("compact", raw_text, "", self.tokenizer, self.meta(0, False))
        self.load()
        docs = [self.document_text(context) for context in example.contexts if context.text.strip()]
        prev_summary = ""
        prev_eval = ""
        complete = False
        turns = 0
        for start in range(0, len(docs), self.segment_size):
            if self.max_iteration is not None and turns >= self.max_iteration:
                break
            output = self.generate(self.prompt(example.question, docs[start : start + self.segment_size], prev_summary, prev_eval))
            prev_summary, prev_eval = self.parse(output)
            complete = "[COMPLETE]" in prev_eval
            turns += 1
            if complete:
                break
        text = f"{prev_summary} {self.strip_markers(prev_eval)}".strip() if self.include_eval_reason else prev_summary
        text = cap(text, self.config.budget, self.tokenizer)
        return result("compact", raw_text, text, self.tokenizer, self.meta(turns, complete))
