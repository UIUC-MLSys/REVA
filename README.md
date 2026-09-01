# REVA: Reusable Evidence View Aggregation

This repository contains the artifact for the paper:

> **REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving**  
> Accepted at **IEEE ICDM 2026**.

REVA is a post-retrieval context-compression pipeline for retrieval-augmented generation (RAG). Instead of compressing each request's retrieved contexts from scratch, REVA aggregates historical query-document accesses scored with the target model into reusable document evidence views. At serving time, the score-store-backed `reva` path looks up stored word-unit scores by document or chunk ID and materializes budget-specific plain-text views with low online overhead.

The implementation keeps the downstream RAG interface unchanged: the generator still receives ordinary text context. No KV-cache interface, latent memory, or generator-side architecture change is required.

This artifact lets you build a reusable score store, run score-store-backed context compression, optionally generate and evaluate answers, and compare against thin baseline wrappers. Paper link, citation metadata, and DOI will be added when the proceedings or arXiv page is public.

## Repository Layout

```text
src/
  cli.py              argparse entrypoint
  schema.py           data classes and JSON/JSONL helpers
  models.py           prompt formatting, model loading, generation
  metrics.py          token counts, EM/F1/ROUGE-L, summaries
  runner.py           shared benchmark loop
  methods.py          method registry
  baselines.py        raw, truncate, and baseline adapter shells
  reva.py              REVA score-store construction, selection, rendering, compressors
  retrieval_build.py  wiki18/E5 corpus download, encoding, FAISS index build
```

The main implementation module is `src/reva.py`, and the primary CLI method IDs are `reva` and `reva_query_aware`.

## Setup

```bash
uv sync --locked
```

For retrieval index building:

```bash
uv sync --locked --extra retrieval
```

Optional baseline wrappers require extra packages:

```bash
uv sync --locked --extra baselines
```

Run commands from the repository root with `PYTHONPATH=src`; this repo is kept as a lightweight artifact tree rather than an installed Python package.

## Input Format

Input files are JSONL. Each row contains one question and its retrieved contexts:

```json
{"id":"q1","question":"...","answers":["..."],"contexts":[{"doc_id":"d1","text":"...","title":"...","chunk_id":"..."}]}
```

Use a stable `chunk_id` when multiple chunks share one `doc_id`. REVA validates score-store text and word-unit boundaries before reuse, so changed chunks should rebuild the store instead of silently reusing stale scores.

## Quick Start

Build a reusable score store from historical or training retrieval examples:

```bash
PYTHONPATH=src uv run python -m cli build-store \
  --input data/train.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --option max_scoring_tokens=8192 \
  --output outputs/reva_store
```

Run REVA from the prebuilt store:

```bash
PYTHONPATH=src uv run python -m cli run \
  --input data/test.jsonl \
  --method reva \
  --budget 512 \
  --option score_store_path=outputs/reva_store/score_store.jsonl \
  --output outputs/reva_b512
```

Run the query-aware diagnostic variant without a prebuilt store:

```bash
PYTHONPATH=src uv run python -m cli run \
  --input data/test.jsonl \
  --method reva_query_aware \
  --budget 512 \
  --option scoring_model_name=meta-llama/Llama-3.1-8B-Instruct \
  --output outputs/reva_query_aware_b512
```

`build-store` writes `score_store.jsonl` and `summary.json`. `run` writes `records.jsonl` and `summary.json`.

By default, `run` only compresses contexts. Add `--model ...` to `run` when you also want answer generation and EM/F1/ROUGE-L evaluation.

Expected working tree for a local experiment:

```text
data/train.jsonl
data/test.jsonl
outputs/reva_store/score_store.jsonl
outputs/reva_store/summary.json
outputs/reva_b512/records.jsonl
outputs/reva_b512/summary.json
retrieval_artifacts/
```

`data/train.jsonl` and `data/test.jsonl` are placeholders for your benchmark inputs; sample benchmark data is not shipped in this repository.

## Retrieval Build

`retrieval-build` prepares the wiki18/E5 retrieval artifacts used by the paper pipeline. It downloads the corpus, encodes passages with E5, and builds a FAISS Flat inner-product index from the embeddings.

```text
retrieval_build_id: wiki18_e5
corpus: RUC-NLPIR/FlashRAG_datasets/retrieval-corpus/wiki18_100w.zip
embedding_model: intfloat/e5-base-v2
embeddings: float32 memmap
index: FAISS Flat inner product
```

Build all retrieval artifacts:

```bash
PYTHONPATH=src uv run --extra retrieval python -m cli retrieval-build \
  --stage all \
  --data-root /path/to/retrieval_artifacts
```

The command writes:

```text
downloads/FlashRAG_datasets/retrieval-corpus/wiki18_100w.zip
corpora/wiki18_100w/wiki18_100w.jsonl
models/intfloat__e5-base-v2/
indexes/wiki18_100w/e5_flat/e5_embeddings.f32.memmap
indexes/wiki18_100w/e5_flat/e5_embeddings.ckpt.json
indexes/wiki18_100w/e5_flat/e5_Flat.index
manifests/download.wiki18_e5.json
manifests/encode.wiki18_e5.json
manifests/index.wiki18_e5.json
```

You can also run `--stage download`, `--stage encode`, or `--stage index` separately. `--batch-size` and `--max-length` apply to `encode`; `--chunk-size` applies to `index`. Encoding resumes from `e5_embeddings.ckpt.json` by default; add `--overwrite` to rebuild artifacts for the selected stage.

## How REVA Works

REVA builds one reusable evidence view per document or chunk by scoring word units with the target model on historical or training accesses, then reuses those scores at serving time.

```text
build-store: prompt -> token scores -> word units -> average score store
run reva:     load score store -> select word units -> render compressed context
```

The core implementation uses a document-wise reusable evidence view: each retrieved document gets a quota, selects its own highest-scoring word units, and renders the selected units back in original document order. Documents missing from the score store fall back to prefix truncation under the same quota.

A word unit is an alphanumeric word span aligned to tokenizer offsets:

```text
Doc: "Paris is the capital of France."

index  unit      score
0      Paris     0.91
1      is        0.08
2      the       0.03
3      capital   0.77
4      of        0.04
5      France    0.88
```

For a small token budget, REVA selects high-score word units under the budget and renders them in the original document order:

```text
Selected by score: Paris, France, capital
Rendered context:  Paris capital France
```

The score store keeps one row per document or chunk:

```json
{
  "doc_key": "d1::chunk=0",
  "doc_id": "d1",
  "chunk_id": "0",
  "word_units": [
    {"index": 0, "text": "Paris", "token_count": 1, "score_sum": 2.73, "score_count": 3, "score": 0.91}
  ]
}
```

For each training query, token scores are grouped into word-unit scores and normalized inside each document. The score store then averages normalized word scores as `score = score_sum / score_count`.

## Useful Options

```text
score_store_path       score_store.jsonl for score-store-backed REVA runs
scoring_model_name     model used for score-store construction
max_scoring_tokens     cap for the scoring prompt
per_doc_token_quota    fixed token quota per retrieved document
layer_subset           all, first, last, an index, or a list of indices
head_aggregation       mean, max, or topk
query_aggregation      sum, mean, or max
word_score             max, mean, or sum over the word's tokens
```

`word_score` is the preferred option name. The implementation also accepts the legacy `word_unit_score_aggregation` key for compatibility with earlier experiment scripts.

## Method Names and References

| Method ID | Reference | Code or Checkpoint | Notes |
| --- | --- | --- | --- |
| `raw` | - | [src/baselines.py](src/baselines.py) | Passes the retrieved context through unchanged. |
| `truncate` | - | [src/baselines.py](src/baselines.py) | Prefix truncation baseline under the requested budget. |
| `selective_context` | [Selective Context](https://arxiv.org/abs/2310.06201) | [liyucheng09/Selective_Context](https://github.com/liyucheng09/Selective_Context) | Thin wrapper around the public package. |
| `selective_context_docwise` | [Selective Context](https://arxiv.org/abs/2310.06201) | [liyucheng09/Selective_Context](https://github.com/liyucheng09/Selective_Context) | Applies Selective Context per document. |
| `llmlingua2` | [LLMLingua-2](https://arxiv.org/abs/2403.12968) | [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) | Uses the LLMLingua package with the LLMLingua-2 path. |
| `llmlingua2_docwise` | [LLMLingua-2](https://arxiv.org/abs/2403.12968) | [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) | Applies LLMLingua-2 per document. |
| `longllmlingua` | [LongLLMLingua](https://arxiv.org/abs/2310.06839) | [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) | Query-aware long-context prompt compression. |
| `recomp_extractive` | [RECOMP](https://arxiv.org/abs/2310.04408) | [carriex/recomp](https://github.com/carriex/recomp) | Extractive RECOMP-style sentence selection. |
| `recomp_abstractive` | [RECOMP](https://arxiv.org/abs/2310.04408) | [carriex/recomp](https://github.com/carriex/recomp) | Abstractive RECOMP-style summary generation. |
| `reva_query_aware` | REVA paper link forthcoming | [src/reva.py](src/reva.py) | Online diagnostic variant that scores each query with the target model. |
| `reva` | REVA paper link forthcoming | [src/reva.py](src/reva.py) | Main score-store-backed reusable evidence view path. |
| `favicomp` | [FaviComp](https://arxiv.org/abs/2409.12468) | [luka-group/FaviComp](https://github.com/luka-group/FaviComp) | Familiarity-aware evidence compression wrapper. |
| `exit` | [EXIT](https://arxiv.org/abs/2412.12559) | [ThisIsHwang/EXIT](https://github.com/ThisIsHwang/EXIT) | Context-aware extractive compression wrapper. |
| `longrefiner` | [LongRefiner](https://arxiv.org/abs/2505.10413) | [ignorejjj/LongRefiner](https://github.com/ignorejjj/LongRefiner) | Hierarchical long-document refinement wrapper. |
| `compact` | [CompAct](https://arxiv.org/abs/2407.09014) | [dmis-lab/CompAct](https://github.com/dmis-lab/CompAct) | Active multi-step document compression wrapper. |

Baseline wrappers are implemented for `selective_context`, `selective_context_docwise`, `llmlingua2`, `llmlingua2_docwise`, `longllmlingua`, `recomp_extractive`, `recomp_abstractive`, `favicomp`, `exit`, `longrefiner`, and `compact`. They call public packages or checkpoints at runtime and are intentionally thin, so some methods need extra setup beyond `uv sync --locked --extra baselines`: `selective_context` may need `python -m spacy download en_core_web_sm`, `longrefiner` needs the official LongRefiner package or `--option module_path=/path/to/LongRefiner`, and `compact` / `favicomp` / `exit` need access to their model checkpoints.

`build-store` and `reva_query_aware` require a scoring model through `--model` or `--option scoring_model_name=...`. `max_scoring_tokens` caps the scoring prompt before the model forward pass; long prompts are shortened by dropping whole document blocks from the beginning of the scoring prompt. The base REVA path in this repo does not use rank decay, answer-in-prompt scoring, global selection, or online score-store updates.

## Environment and Hardware

- Python `>=3.10,<3.13` with `uv` is recommended.
- CPU is enough for score-store-backed `reva` runs when a compatible score store already exists.
- GPU is strongly recommended for `build-store`, `reva_query_aware`, and generation/evaluation with open-weight LMs.
- Attention scoring can be VRAM-heavy because it uses model attention maps and, by default, eager attention.
- `retrieval-build` downloads the wiki18 corpus and E5 model, writes float32 memmaps, and builds a FAISS index; plan disk and RAM accordingly.

## Citation

Citation metadata, DOI, and a public paper link will be added when the official ICDM 2026 proceedings or arXiv page is available.
