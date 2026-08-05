#!/usr/bin/env python3
"""Create three review-only candidate plans for failed Tau2 traces."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import completion

DEFAULT_TRACES = Path(
    "outputs/telecom_50_individual_traces/traces"
)
DEFAULT_OUTPUT = Path("analysis")
DEFAULT_MODEL = "gpt-5-mini"


def parse_test_numbers(value: str) -> list[int]:
    """Parse values such as '1-9', '1,3,8', or '1-3,7'."""
    numbers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Bad test range: {part}")
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    if not numbers:
        raise ValueError("No test numbers were given")
    return sorted(numbers)


def find_trace(traces_dir: Path, test_number: int) -> Path:
    """Find one trace by its two-digit test number."""
    matches = sorted(traces_dir.glob(f"{test_number:02d}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one trace for test {test_number:02d}, found {len(matches)}"
        )
    return matches[0]


def clean_json_response(content: str) -> dict[str, Any]:
    """Read a JSON object, allowing an accidental Markdown code fence."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("The model response must be one JSON object")
    return value


def validate_plans(value: dict[str, Any]) -> None:
    """Check the small set of fields needed for easy plan review."""
    plans = value.get("plans")
    if not isinstance(plans, list) or len(plans) != 3:
        raise ValueError("The response must contain exactly 3 plans")
    expected_ids = ["plan_1", "plan_2", "plan_3"]
    if [plan.get("plan_id") for plan in plans] != expected_ids:
        raise ValueError(f"Plan IDs must be {expected_ids}")
    for plan in plans:
        if not isinstance(plan.get("steps"), list) or not plan["steps"]:
            raise ValueError(f"{plan['plan_id']} has no steps")
        if not isinstance(plan.get("final_checks"), list):
            raise ValueError(f"{plan['plan_id']} has no final_checks list")


def request_plans(
    *, prompt: str, trace: dict[str, Any], model: str, temperature: float | None
) -> dict[str, Any]:
    """Send the full trace to the model without exposing any executable tools."""
    model_options: dict[str, Any] = {}
    if temperature is not None:
        model_options["temperature"] = temperature
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Full trace JSON:\n" + json.dumps(trace, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_object"},
        **model_options,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response")
    return clean_json_response(content)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Generate review-only plans for failed Tau2 traces."
    )
    parser.add_argument("--tests", default="1-9", help="Tests, for example 1-9")
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=Path("analysis/planning_prompt.txt"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional model temperature. By default the provider chooses it.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace plan files that exist"
    )
    return parser


def main() -> int:
    """Generate and save plans for all selected traces."""
    args = build_parser().parse_args()
    load_dotenv()
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    test_numbers = parse_test_numbers(args.tests)
    failures: list[str] = []

    for test_number in test_numbers:
        label = f"test_{test_number:02d}"
        try:
            source = find_trace(args.traces_dir, test_number)
            trace = json.loads(source.read_text(encoding="utf-8"))
            test_dir = args.output_dir / label
            test_dir.mkdir(parents=True, exist_ok=True)
            saved_trace = test_dir / "trace.json"
            plans_path = test_dir / "candidate_plans.json"
            shutil.copy2(source, saved_trace)

            if plans_path.exists() and not args.overwrite:
                print(f"{label}: kept existing {plans_path}")
                continue

            print(f"{label}: asking {args.model} for 3 plans...", flush=True)
            plans = request_plans(
                prompt=prompt,
                trace=trace,
                model=args.model,
                temperature=args.temperature,
            )
            validate_plans(plans)
            plans["test_id"] = str(trace.get("task_id") or plans.get("test_id") or label)
            plans["source_trace"] = str(source)
            plans["model"] = args.model
            plans["generated_at"] = datetime.now(timezone.utc).isoformat()
            plans["note"] = "Review only. No candidate plan was run."
            plans_path.write_text(
                json.dumps(plans, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"{label}: saved {plans_path}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"{label}: ERROR: {exc}", file=sys.stderr)

    if failures:
        print("\nSome tests failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
