"""Execute one paper run cell in an isolated lightweight subprocess."""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

from labverse.benchmarks.experiment import ProviderOptions, run_single_capture
from labverse.benchmarks.initial_variants import BASELINE_INITIAL_VARIANT_ID
from labverse.paper_worker_payload import serialize_artifacts, serialize_run_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one LabVerse paper cell and write its serialized result."
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--replication", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--request-timeout-seconds", required=True, type=float)
    parser.add_argument("--client-max-retries", required=True, type=int)
    parser.add_argument("--worker-timeout-seconds", required=True, type=float)
    parser.add_argument("--initial-variant", default=BASELINE_INITIAL_VARIANT_ID)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    args = _build_parser().parse_args()
    provider_options = ProviderOptions(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        client_max_retries=args.client_max_retries,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    result, artifacts = run_single_capture(
        condition=args.condition,
        provider_name=args.provider,
        model=args.model,
        scenario_name=args.scenario,
        replication=args.replication,
        initial_variant_id=args.initial_variant,
        provider_options=provider_options,
    )
    payload = {
        "result": serialize_run_result(result),
        "artifacts": serialize_artifacts(artifacts),
    }
    Path(args.output).write_text(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
