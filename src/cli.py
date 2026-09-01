from __future__ import annotations

import argparse
import json
from typing import Any

from methods import list_methods


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def option_text(value: str) -> str:
    if "=" not in value or value.startswith("="):
        raise argparse.ArgumentTypeError("must be key=value")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-store")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--model", default=None)
    build.add_argument("--limit", type=non_negative_int, default=None)
    build.add_argument("--top-k", type=non_negative_int, default=None)
    build.add_argument("--option", action="append", type=option_text, default=[])

    retrieval = commands.add_parser("retrieval-build")
    retrieval.add_argument("--stage", choices=("download", "encode", "index", "all"), default="all")
    retrieval.add_argument("--data-root", default="retrieval_artifacts")
    retrieval.add_argument("--batch-size", type=positive_int, default=256)
    retrieval.add_argument("--max-length", type=positive_int, default=512)
    retrieval.add_argument("--chunk-size", type=positive_int, default=250_000)
    retrieval.add_argument("--overwrite", action="store_true")

    bench = commands.add_parser("run")
    bench.add_argument("--input", required=True)
    bench.add_argument("--method", required=True, choices=list_methods())
    bench.add_argument("--output", required=True)
    bench.add_argument("--budget", type=non_negative_int, default=None)
    bench.add_argument("--model", default=None)
    bench.add_argument("--limit", type=non_negative_int, default=None)
    bench.add_argument("--top-k", type=non_negative_int, default=None)
    bench.add_argument("--max-new-tokens", type=positive_int, default=32)
    bench.add_argument("--option", action="append", type=option_text, default=[])
    return root


def parse_options(values: list[str]) -> dict[str, Any]:
    options = {}
    for value in values:
        if "=" not in value or value.startswith("="):
            raise ValueError(f"--option must be key=value, got {value!r}")
        key, _, raw = value.partition("=")
        key = key.strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            options[key] = lowered == "true"
        elif lowered in {"none", "null"}:
            options[key] = None
        elif raw[:1] in "[{\"0123456789":
            options[key] = json.loads(raw)
        else:
            options[key] = raw
    return options


def main() -> int:
    root = parser()
    args = root.parse_args()
    if args.command == "build-store":
        from reva import build_store

        options = parse_options(args.option)
        if not (args.model or options.get("scoring_model_name")):
            root.error("build-store requires --model or --option scoring_model_name=MODEL")
        summary = build_store(
            args.input,
            args.output,
            model_name=args.model,
            options=options,
            limit=args.limit,
            top_k=args.top_k,
        )
        print(summary)
    elif args.command == "retrieval-build":
        from retrieval_build import RetrievalConfig, build_retrieval

        summary = build_retrieval(
            args.stage,
            RetrievalConfig(
                data_root=args.data_root,
                batch_size=args.batch_size,
                max_length=args.max_length,
                chunk_size=args.chunk_size,
                overwrite=args.overwrite,
            ),
        )
        print(summary)
    elif args.command == "run":
        from runner import run

        options = parse_options(args.option)
        if args.method == "reva" and not options.get("score_store_path"):
            root.error(f"{args.method} requires --option score_store_path=PATH")
        if args.method == "reva_query_aware" and not (args.model or options.get("scoring_model_name")):
            root.error("reva_query_aware requires --model or --option scoring_model_name=MODEL")
        summary = run(
            args.input,
            args.method,
            args.output,
            budget=args.budget,
            model_name=args.model,
            options=options,
            limit=args.limit,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
