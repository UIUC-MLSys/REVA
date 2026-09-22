# REVA: Reusable Evidence View Aggregation

> **REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving**  
> Accepted at **IEEE ICDM 2026**.

REVA compresses retrieved documents using reusable word-unit scores mined from historical RAG requests. The generator receives plain text; no architecture or KV-cache changes are required.

![REVA pipeline](figures/reva_pipeline.png)

## Setup

Run commands from the repository root:

```bash
uv sync --locked
export PYTHONPATH=src
```

Optional dependencies: `uv sync --locked --extra retrieval` for corpus/index preparation, or `--extra baselines` for baseline wrappers.

## Data

- [Retrieval top-20 with answers](https://drive.google.com/file/d/1buhg89g4n5j4tGiDj_94K1F0bflNzhFb/view?usp=sharing): NQ, TriviaQA, HotpotQA, and 2WikiMultihopQA.
- [REVA score cache](https://drive.google.com/file/d/1vO7EmnzyV2-Fqg8KudX-oT2haiwY8uPe/view?usp=sharing): precomputed word-unit scores for REVA.

Place the artifacts under `data/`:

```text
data/
  retrieval_top20_4datasets_with_answers.jsonl.gz
  score_cache_reva_v1/
    scores_reva_v1__llama31__qpa__reversed__topk10.jsonl.gz
    ...
```

Use the tokenizer matching each cache:

| Cache model | Model/tokenizer |
| --- | --- |
| `llama31` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen35` | `Qwen/Qwen3.5-9B` |
| `gemma4e4b` | `google/gemma-4-E4B-it` |

Retrieval rows contain document IDs, not text. Download the FlashRAG wiki18 corpus and E5 model:

```bash
uv run --extra retrieval python -m cli retrieval-build \
  --stage download --data-root retrieval_artifacts
```

The corpus is extracted to `retrieval_artifacts/corpora/wiki18_100w/wiki18_100w.jsonl`. Use `--stage all` instead to also encode passages and build the E5/FAISS index.

## Run

Compress NQ test queries with the released Llama Q+A cache:

```bash
uv run python -m cli run \
  --input data/retrieval_top20_4datasets_with_answers.jsonl.gz \
  --corpus retrieval_artifacts/corpora/wiki18_100w/wiki18_100w.jsonl \
  --dataset nq --split test --top-k 10 --limit 10 \
  --method reva --budget 512 \
  --option scoring_model_name=meta-llama/Llama-3.1-8B-Instruct \
  --option score_store_path=data/score_cache_reva_v1/scores_reva_v1__llama31__qpa__reversed__topk10.jsonl.gz \
  --output outputs/reva_nq_b512
```

This loads only the tokenizer. Add `--model meta-llama/Llama-3.1-8B-Instruct` for generation and EM/F1/ROUGE-L evaluation. Remove `--limit` for the full split; use `--split dev` for HotpotQA and 2WikiMultihopQA. Outputs are `records.jsonl` and `summary.json`.

For query-aware scoring without a cache, use `--method reva_query_aware` and omit `score_store_path`.

## Build Scores

Build a Q+A/reversed store from training retrieval:

```bash
uv run python -m cli build-store \
  --input data/retrieval_top20_4datasets_with_answers.jsonl.gz \
  --corpus retrieval_artifacts/corpora/wiki18_100w/wiki18_100w.jsonl \
  --dataset nq --split train --top-k 10 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --option scoring_mode=query_plus_answer \
  --option scoring_doc_order=reversed \
  --option max_scoring_tokens=8192 \
  --output outputs/reva_store
```

Outputs are `score_store.jsonl.gz` and `summary.json` (model and scoring settings). To use this store, set `score_store_path=outputs/reva_store/score_store.jsonl.gz` in the run command.

Scoring options apply to `build-store` and `reva_query_aware`:

- `scoring_mode`: `query_only` (default) or `query_plus_answer`. Q+A requires `source_response` or the first non-empty `answers` entry, in that order.
- `scoring_doc_order`: `original` (default) or `reversed`; output documents retain retrieval order.
- `query_weight` / `answer_weight`: default to `0.5` / `0.5` in Q+A mode; query-only uses Q alone.

Use historical responses or training answers to build Q+A stores. Serving with `reva` never reads the new query's answer; query-aware Q+A with evaluation answers is an oracle diagnostic.

## Formats

Inputs and score stores support JSONL or `.jsonl.gz`. Released retrieval rows:

```json
{"dataset":"nq","split":"test","id":"q1","question":"...","answers":["..."],"doc_ids":["d1"]}
```

Pass `--corpus` to join IDs to text. Alternatively, replace `doc_ids` with inline `contexts`:

```json
{"dataset":"nq","id":"q1","question":"...","answers":["..."],"contexts":[{"doc_id":"d1","title":"...","text":"..."}]}
```

Each compact score row stores one document, keyed by `(dataset, doc_id)`:

```json
{"dataset":"nq","doc_id":"d1","hits":9,"scores":[0.0,0.0,0.000031]}
```

`hits` counts training accesses; `scores[i]` is the mean score of word unit `i`. Include `dataset` in input rows or pass `--dataset`. For multiple chunks of one document, provide a stable `chunk_id`, which is also saved in the cache. Older stores with explicit `word_units` remain readable; unlabelled stores must be used with one dataset at a time.

## How It Works

1. **Score:** Documents precede the question and optional known response. Attention is averaged over heads/layers and summed over source tokens; Q+A mixes separately computed Q and A scores. The first four tokens of documents longer than four tokens are excluded from scoring. No per-document max normalization is applied.
2. **Aggregate:** Tokenizer pieces form word units, with names, dates, and grouped numbers merged. Unit scores use the maximum token score by default, then average across document accesses and are saved to six decimals. Scoring and cache loading share the same unit builder, using the original text/tokenizer and a trailing double newline.
3. **Serve:** Select high-scoring units within document quotas and render in original order. Unseen documents use prefix truncation. Rendered text, including document separators, stays within `--budget`.

Illustrative selection: `Paris is the capital of France` -> select `Paris`, `France`, `capital` by score -> render `Paris capital France`.

Core implementation: [src/reva.py](src/reva.py). [src/cli.py](src/cli.py) exposes the commands, [src/runner.py](src/runner.py) runs evaluation, and [src/baselines.py](src/baselines.py) contains baseline wrappers.
