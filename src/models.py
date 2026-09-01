from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import Example

SYSTEM_PROMPT = (
    "Answer the question based on the given document. "
    "Only give me the answer and do not output any other words."
    "\nThe following are given documents.\n\n{reference}"
)
USER_PROMPT = "Question: {question}"


@dataclass(slots=True)
class Generator:
    tokenizer: Any
    model: Any
    device: Any
    max_length: int


def build_prompt(example: Example, context_text: str, tokenizer: Any | None = None) -> str:
    system = SYSTEM_PROMPT.format(reference=context_text.strip())
    user = USER_PROMPT.format(question=example.question.strip())
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return "\n\n".join([system, user]).strip()
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_name: str) -> Generator:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    max_length = int(getattr(model.config, "max_position_embeddings", 4096))
    return Generator(tokenizer, model, next(model.parameters()).device, max_length)


def generate(generator: Generator, prompt: str, max_new_tokens: int = 32) -> dict[str, Any]:
    encoded = generator.tokenizer(prompt, return_tensors="pt").to(generator.device)
    budget = max(1, generator.max_length - max_new_tokens)
    if encoded["input_ids"].shape[-1] > budget:
        encoded["input_ids"] = encoded["input_ids"][:, -budget:]
        encoded["attention_mask"] = encoded["attention_mask"][:, -budget:]
    prompt_len = encoded["input_ids"].shape[-1]
    start = time.perf_counter()
    output = generator.model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=generator.tokenizer.pad_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0
    answer_ids = output[0][prompt_len:]
    text = generator.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return {
        "prediction": text.splitlines()[0] if text else "",
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(answer_ids.shape[-1]),
        "generation_ms": ms,
    }
