from __future__ import annotations

from typing import Any

from schema import CompressionConfig, Compressor

METHODS = [
    "raw",
    "truncate",
    "selective_context",
    "selective_context_docwise",
    "llmlingua2",
    "llmlingua2_docwise",
    "longllmlingua",
    "recomp_extractive",
    "recomp_abstractive",
    "reva_query_aware",
    "reva",
    "favicomp",
    "exit",
    "longrefiner",
    "compact",
]


def list_methods() -> list[str]:
    return list(METHODS)


def get_compressor(config: CompressionConfig, tokenizer: Any | None = None, model_name: str | None = None) -> Compressor:
    from baselines import (
        CompActCompressor,
        ExitCompressor,
        FaviCompCompressor,
        LLMLingua2DocwiseCompressor,
        LLMLinguaCompressor,
        LongLLMLinguaCompressor,
        LongRefinerCompressor,
        RawCompressor,
        RecompAbstractiveCompressor,
        RecompExtractiveCompressor,
        SelectiveContextCompressor,
        SelectiveContextDocwiseCompressor,
        TruncateCompressor,
    )
    from reva import REVAOfflineCompressor, REVAQueryAwareCompressor

    classes = {
        "raw": RawCompressor,
        "truncate": TruncateCompressor,
        "selective_context": SelectiveContextCompressor,
        "selective_context_docwise": SelectiveContextDocwiseCompressor,
        "llmlingua2": LLMLinguaCompressor,
        "llmlingua2_docwise": LLMLingua2DocwiseCompressor,
        "longllmlingua": LongLLMLinguaCompressor,
        "recomp_extractive": RecompExtractiveCompressor,
        "recomp_abstractive": RecompAbstractiveCompressor,
        "reva_query_aware": REVAQueryAwareCompressor,
        "reva": REVAOfflineCompressor,
        "favicomp": FaviCompCompressor,
        "exit": ExitCompressor,
        "longrefiner": LongRefinerCompressor,
        "compact": CompActCompressor,
    }
    return classes[config.method](config, tokenizer, model_name)
