from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import faiss
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from schema import write_json


@dataclass(frozen=True, slots=True)
class RetrievalBuildSpec:
    build_id: str
    corpus_id: str
    corpus_repo: str
    corpus_file: str
    embedding_model: str
    model_cache_dir: str
    index_name: str
    text_prefix: str = "passage: "


WIKI18_E5_BUILD_SPEC = RetrievalBuildSpec(
    build_id="wiki18_e5",
    corpus_id="wiki18_100w",
    corpus_repo="RUC-NLPIR/FlashRAG_datasets",
    corpus_file="retrieval-corpus/wiki18_100w.zip",
    embedding_model="intfloat/e5-base-v2",
    model_cache_dir="intfloat__e5-base-v2",
    index_name="e5_Flat.index",
)

SAVE_EVERY = 1_000_000


@dataclass(slots=True)
class RetrievalConfig:
    data_root: Path
    batch_size: int = 256
    max_length: int = 512
    chunk_size: int = 250_000
    overwrite: bool = False


class RetrievalPaths:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def corpora(self) -> Path:
        return self.root / "corpora"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def dataset_dir(self) -> Path:
        return self.downloads / "FlashRAG_datasets"

    def corpus_archive(self, spec: RetrievalBuildSpec) -> Path:
        return self.dataset_dir() / spec.corpus_file

    def corpus_path(self, spec: RetrievalBuildSpec) -> Path:
        return self.corpora / spec.corpus_id / f"{spec.corpus_id}.jsonl"

    def model_path(self, spec: RetrievalBuildSpec) -> Path:
        return self.models / spec.model_cache_dir

    def index_dir(self, spec: RetrievalBuildSpec) -> Path:
        return self.indexes / spec.corpus_id / "e5_flat"

    def embedding_path(self, spec: RetrievalBuildSpec) -> Path:
        return self.index_dir(spec) / "e5_embeddings.f32.memmap"

    def checkpoint_path(self, spec: RetrievalBuildSpec) -> Path:
        return self.index_dir(spec) / "e5_embeddings.ckpt.json"

    def index_path(self, spec: RetrievalBuildSpec) -> Path:
        return self.index_dir(spec) / spec.index_name

    def manifest_path(self, stage: str, spec: RetrievalBuildSpec) -> Path:
        return self.manifests / f"{stage}.{spec.build_id}.json"

    def ensure(self) -> None:
        for path in (self.downloads, self.corpora, self.models, self.indexes, self.manifests):
            path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_path(paths: RetrievalPaths, path: Path) -> str:
    root = paths.root.resolve()
    target = path.resolve()
    return str(target.relative_to(root)) if target.is_relative_to(root) else str(target)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(paths: RetrievalPaths, spec: RetrievalBuildSpec, stage: str, extra: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "created_at": utc_now(),
        "stage": stage,
        "retrieval_build_id": spec.build_id,
        "corpus_id": spec.corpus_id,
        "embedding_model": spec.embedding_model,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    payload.update(extra)
    write_json(paths.manifest_path(stage, spec), payload)
    return payload


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def download_artifacts(paths: RetrievalPaths, spec: RetrievalBuildSpec, overwrite: bool = False) -> dict[str, Any]:
    paths.ensure()
    snapshot_download(
        repo_id=spec.corpus_repo,
        repo_type="dataset",
        local_dir=str(paths.dataset_dir()),
        allow_patterns=[spec.corpus_file],
    )
    snapshot_download(repo_id=spec.embedding_model, repo_type="model", local_dir=str(paths.model_path(spec)))
    corpus_path = extract_corpus(paths, spec, overwrite)
    return write_manifest(
        paths,
        spec,
        "download",
        {
            "corpus_archive": artifact_path(paths, paths.corpus_archive(spec)),
            "corpus_path": artifact_path(paths, corpus_path),
            "model_path": artifact_path(paths, paths.model_path(spec)),
        },
    )


def extract_corpus(paths: RetrievalPaths, spec: RetrievalBuildSpec, overwrite: bool = False) -> Path:
    archive_path = paths.corpus_archive(spec)
    target_path = paths.corpus_path(spec)
    if target_path.exists() and not overwrite:
        return target_path
    if not archive_path.exists():
        raise FileNotFoundError(f"missing corpus archive: {archive_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    extract_root = target_path.parent / "_extract_tmp"
    clean_dir(extract_root)
    extract_root.mkdir(parents=True)
    if target_path.exists():
        target_path.unlink()

    with zipfile.ZipFile(archive_path) as handle:
        handle.extractall(extract_root)
    candidates = sorted(extract_root.rglob("*.jsonl"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no jsonl corpus found under {extract_root}")
    shutil.move(str(candidates[0]), str(target_path))
    clean_dir(extract_root)
    return target_path


def count_lines(path: Path) -> int:
    total = 0
    with path.open(encoding="utf-8") as handle:
        for _line in handle:
            total += 1
    return total


def corpus_text(row: dict[str, Any], prefix: str) -> str:
    if row.get("contents"):
        text = str(row["contents"])
    else:
        title = str(row.get("title", "")).strip()
        body = str(row.get("text", "")).strip()
        text = f"{title}\n{body}".strip() if title else body
    return f"{prefix}{text}" if prefix else text


def text_batches(path: Path, batch_size: int, start_index: int, prefix: str) -> Iterator[tuple[int, list[str]]]:
    batch = []
    batch_start = start_index
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index:
                continue
            if not batch:
                batch_start = index
            batch.append(corpus_text(json.loads(line), prefix))
            if len(batch) == batch_size:
                yield batch_start, batch
                batch = []
    if batch:
        yield batch_start, batch


def load_checkpoint(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {"next_index": 0}


def save_checkpoint(path: Path, next_index: int, corpus_size: int, dim: int, completed: bool) -> None:
    write_json(
        path,
        {
            "updated_at": utc_now(),
            "next_index": next_index,
            "corpus_size": corpus_size,
            "embedding_dim": dim,
            "dtype": "float32",
            "completed": completed,
        },
    )


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def encode_texts(model: Any, tokenizer: Any, texts: list[str], max_length: int, device: torch.device) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
    embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.float().cpu().numpy()


def progress(stage: str, current: int, total: int, started: float) -> str:
    elapsed = max(time.time() - started, 1e-6)
    rate = current / elapsed
    pct = 100.0 * current / max(total, 1)
    return f"{stage}: {current}/{total} ({pct:.2f}%) {rate:.1f}/s"


def encode_corpus(paths: RetrievalPaths, spec: RetrievalBuildSpec, config: RetrievalConfig) -> dict[str, Any]:
    corpus_path = paths.corpus_path(spec)
    model_path = paths.model_path(spec)
    embedding_path = paths.embedding_path(spec)
    checkpoint_path = paths.checkpoint_path(spec)
    if not corpus_path.exists():
        raise FileNotFoundError(f"missing corpus jsonl: {corpus_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing model snapshot: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()
    dim = int(model.config.hidden_size)
    corpus_size = count_lines(corpus_path)

    if config.overwrite:
        for path in (embedding_path, checkpoint_path):
            if path.exists():
                path.unlink()
    checkpoint = load_checkpoint(checkpoint_path)
    start_index = int(checkpoint.get("next_index", 0)) if embedding_path.exists() else 0

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.memmap(embedding_path, shape=(corpus_size, dim), mode="r+" if embedding_path.exists() else "w+", dtype=np.float32)
    started = time.time()
    last_saved = start_index
    for batch_start, texts in text_batches(corpus_path, config.batch_size, start_index, spec.text_prefix):
        batch_end = batch_start + len(texts)
        memmap[batch_start:batch_end] = encode_texts(model, tokenizer, texts, config.max_length, device)
        if batch_end - last_saved >= SAVE_EVERY:
            memmap.flush()
            save_checkpoint(checkpoint_path, batch_end, corpus_size, dim, False)
            last_saved = batch_end
        if batch_end == corpus_size or batch_end % (config.batch_size * 100) == 0:
            print(progress("encode", batch_end, corpus_size, started), flush=True)
    memmap.flush()
    del memmap
    save_checkpoint(checkpoint_path, corpus_size, corpus_size, dim, True)
    return write_manifest(
        paths,
        spec,
        "encode",
        {
            "corpus_path": artifact_path(paths, corpus_path),
            "model_path": artifact_path(paths, model_path),
            "embedding_path": artifact_path(paths, embedding_path),
            "checkpoint_path": artifact_path(paths, checkpoint_path),
            "corpus_size": corpus_size,
            "embedding_dim": dim,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "duration_sec": round(time.time() - started, 3),
        },
    )


def build_faiss_index(paths: RetrievalPaths, spec: RetrievalBuildSpec, config: RetrievalConfig) -> dict[str, Any]:
    encode_manifest = read_json(paths.manifest_path("encode", spec))
    embedding_path = paths.embedding_path(spec)
    index_path = paths.index_path(spec)
    index_manifest_path = paths.manifest_path("index", spec)
    corpus_size = int(encode_manifest["corpus_size"])
    dim = int(encode_manifest["embedding_dim"])
    if index_path.exists() and index_manifest_path.exists() and not config.overwrite:
        return read_json(index_manifest_path)

    embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(corpus_size, dim)
    index = faiss.IndexFlatIP(dim)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for start in range(0, corpus_size, config.chunk_size):
        end = min(start + config.chunk_size, corpus_size)
        index.add(np.asarray(embeddings[start:end]))
        print(progress("index", end, corpus_size, started), flush=True)
    faiss.write_index(index, str(index_path))
    return write_manifest(
        paths,
        spec,
        "index",
        {
            "embedding_path": artifact_path(paths, embedding_path),
            "index_path": artifact_path(paths, index_path),
            "corpus_size": corpus_size,
            "embedding_dim": dim,
            "faiss_type": "Flat",
            "faiss_metric": "inner_product",
            "chunk_size": config.chunk_size,
            "duration_sec": round(time.time() - started, 3),
        },
    )


def build_retrieval(stage: str, config: RetrievalConfig) -> dict[str, Any]:
    if stage not in {"download", "encode", "index", "all"}:
        raise ValueError("stage must be one of: download, encode, index, all")
    paths = RetrievalPaths(config.data_root)
    paths.ensure()
    if stage in {"download", "all"}:
        summary = download_artifacts(paths, WIKI18_E5_BUILD_SPEC, config.overwrite)
    if stage in {"encode", "all"}:
        summary = encode_corpus(paths, WIKI18_E5_BUILD_SPEC, config)
    if stage in {"index", "all"}:
        summary = build_faiss_index(paths, WIKI18_E5_BUILD_SPEC, config)
    return summary
