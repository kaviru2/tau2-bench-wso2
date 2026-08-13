#!/usr/bin/env python3
"""Diagnose Tau2 traces and generate grounded agent operating guidelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import completion

DEFAULT_TRACES = Path("outputs/telecom_50_individual_traces/traces")
DEFAULT_OUTPUT = Path("analysis")
DEFAULT_PROMPT = Path("analysis/planning_prompt.txt")
DEFAULT_MODEL = "gpt-5.6-luna"


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


def require_nonempty_list(value: dict[str, Any], field: str) -> list[Any]:
    """Return a required, non-empty list field."""
    items = value.get(field)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{field} must be a non-empty list")
    return items


def failed_actions(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return evaluator actions that were not matched, keyed by action ID."""
    reward_info = trace.get("trace", {}).get("reward_info") or {}
    action_checks = reward_info.get("action_checks") or []
    return {
        str(check["action"]["action_id"]): check["action"]
        for check in action_checks
        if check.get("action_match") is False
        and isinstance(check.get("action"), dict)
        and check["action"].get("action_id")
    }


def source_path_exists(trace: dict[str, Any], source: str) -> bool:
    """Return whether a simple dotted/indexed source path exists in the trace."""
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*",
        source,
    ):
        return False
    current: Any = trace
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", source):
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current):
                return False
            current = current[index]
        else:
            return False
    return True


def normalize_trace_locations(value: dict[str, Any], trace: dict[str, Any]) -> None:
    """Make trace indices and assistant-turn numbers deterministic."""
    messages = trace.get("trace", {}).get("messages") or []
    if not messages:
        raise ValueError("The trace has no messages to anchor instructions to")

    failure = value.get("failure_analysis") or {}
    points: list[tuple[str, dict[str, Any]]] = []
    first_wrong = failure.get("first_wrong_point")
    if isinstance(first_wrong, dict):
        points.append(("first_wrong_point", first_wrong))
    for index, point in enumerate(failure.get("failure_points") or [], start=1):
        if isinstance(point, dict):
            points.append((f"failure_point_{index}", point))
    for guideline in value.get("agent_guidelines") or []:
        if isinstance(guideline, dict) and isinstance(
            guideline.get("applies_at"), dict
        ):
            points.append(
                (str(guideline.get("guideline_id")), guideline["applies_at"])
            )

    for label, point in points:
        message_index = point.get("trace_message_index")
        if message_index is None and point.get("placement") == "before_ending":
            message_index = len(messages) - 1
            point["trace_message_index"] = message_index
        if not isinstance(message_index, int) or not 0 <= message_index < len(messages):
            raise ValueError(f"{label} has an invalid trace_message_index")
        if messages[message_index].get("role") == "assistant":
            point["assistant_turn"] = sum(
                1
                for message in messages[: message_index + 1]
                if message.get("role") == "assistant"
            )
        else:
            point["assistant_turn"] = None


def validate_analysis(value: dict[str, Any], trace: dict[str, Any]) -> None:
    """Validate the fields needed to review and apply the diagnosis."""
    goal = value.get("goal_satisfaction")
    if not isinstance(goal, dict) or not isinstance(goal.get("satisfied"), bool):
        raise ValueError("goal_satisfaction.satisfied must be a boolean")
    require_nonempty_list(goal, "evidence")

    failure = value.get("failure_analysis")
    if not isinstance(failure, dict):
        raise ValueError("failure_analysis must be an object")
    first_wrong = failure.get("first_wrong_point")
    if not isinstance(first_wrong, dict):
        raise ValueError("failure_analysis.first_wrong_point must be an object")
    if not isinstance(first_wrong.get("trace_message_index"), int):
        raise ValueError("first_wrong_point.trace_message_index must be an integer")
    require_nonempty_list(failure, "failure_points")
    require_nonempty_list(failure, "failed_checks")

    guidelines = require_nonempty_list(value, "agent_guidelines")
    expected_ids = [f"guideline_{index}" for index in range(1, len(guidelines) + 1)]
    actual_ids = [item.get("guideline_id") for item in guidelines]
    if actual_ids != expected_ids:
        raise ValueError(f"Guideline IDs must be sequential: {expected_ids}")
    valid_placements = {
        "replace_assistant_turn",
        "after_trace_message",
        "before_next_assistant_action",
        "before_ending",
    }
    valid_categories = {
        "diagnosis",
        "sequencing",
        "tool_use",
        "policy",
        "verification",
        "communication",
        "termination",
    }
    valid_priorities = {"critical", "high", "normal"}
    expected_actions = failed_actions(trace)
    covered_action_ids: list[str] = []
    for guideline in guidelines:
        guideline_id = guideline["guideline_id"]
        location = guideline.get("applies_at")
        if not isinstance(location, dict):
            raise ValueError(f"{guideline_id} has no applies_at location")
        if location.get("placement") not in valid_placements:
            raise ValueError(f"{guideline_id} has an invalid placement")
        if guideline.get("category") not in valid_categories:
            raise ValueError(f"{guideline_id} has an invalid category")
        if guideline.get("priority") not in valid_priorities:
            raise ValueError(f"{guideline_id} has an invalid priority")
        if not isinstance(guideline.get("guideline"), str) or not guideline[
            "guideline"
        ].strip():
            raise ValueError(f"{guideline_id} has no guideline text")
        if not isinstance(guideline.get("agent_decision"), str) or not guideline[
            "agent_decision"
        ].strip():
            raise ValueError(f"{guideline_id} has no agent_decision")
        if not isinstance(guideline.get("success_condition"), str) or not guideline[
            "success_condition"
        ].strip():
            raise ValueError(f"{guideline_id} has no success_condition")

        grounding = require_nonempty_list(guideline, "grounding")
        for evidence in grounding:
            if not isinstance(evidence, dict):
                raise ValueError(f"{guideline_id} has invalid grounding evidence")
            source = evidence.get("source")
            fact = evidence.get("fact")
            if not isinstance(source, str) or not source.startswith(
                ("trace.", "task.", "status", "task_id", "order")
            ):
                raise ValueError(f"{guideline_id} has an invalid grounding source")
            if not source_path_exists(trace, source):
                raise ValueError(
                    f"{guideline_id} grounding source does not exist: {source}"
                )
            if not isinstance(fact, str) or not fact.strip():
                raise ValueError(f"{guideline_id} has an empty grounding fact")

        required_action = guideline.get("required_action")
        if not isinstance(required_action, dict):
            raise ValueError(f"{guideline_id} has no required_action object")
        action_id = required_action.get("action_id")
        if action_id is not None and not isinstance(action_id, str):
            raise ValueError(f"{guideline_id} has invalid action_id")
        if action_id:
            covered_action_ids.append(action_id)
            action = expected_actions.get(action_id)
            if action is None:
                raise ValueError(
                    f"{guideline_id} references unknown failed action ID {action_id}"
                )
            if required_action.get("requestor") != action.get("requestor"):
                raise ValueError(f"{action_id} has the wrong requestor")
            if required_action.get("tool_name") != action.get("name"):
                raise ValueError(f"{action_id} has the wrong tool_name")
            if required_action.get("tool_input") != (action.get("arguments") or {}):
                raise ValueError(f"{action_id} has the wrong tool_input")
        else:
            if required_action.get("requestor") is not None:
                raise ValueError(f"{guideline_id} must have a null requestor")
            if required_action.get("tool_name") is not None:
                raise ValueError(f"{guideline_id} must have a null tool_name")
            if required_action.get("tool_input") != {}:
                raise ValueError(f"{guideline_id} must have an empty tool_input")

    duplicates = {
        action_id
        for action_id in covered_action_ids
        if covered_action_ids.count(action_id) > 1
    }
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates))
        raise ValueError(f"Failed action IDs appear more than once: {duplicate_text}")

    missing_action_ids = set(expected_actions) - set(covered_action_ids)
    if missing_action_ids:
        missing = ", ".join(sorted(missing_action_ids))
        raise ValueError(f"Guidelines do not cover failed action IDs: {missing}")

    require_nonempty_list(value, "global_agent_rules")
    require_nonempty_list(value, "final_verification_guidelines")


def request_analysis(
    *,
    prompt: str,
    trace: dict[str, Any],
    model: str,
    temperature: float | None,
    request_timeout: float,
    validation_feedback: str | None = None,
) -> dict[str, Any]:
    """Send a full trace to the model without exposing executable tools."""
    model_options: dict[str, Any] = {}
    if temperature is not None:
        model_options["temperature"] = temperature
    user_content = "Full trace JSON:\n" + json.dumps(trace, ensure_ascii=False)
    action_manifest = [
        {
            "action_id": action_id,
            "requestor": action.get("requestor"),
            "tool_name": action.get("name"),
            "tool_input": action.get("arguments") or {},
        }
        for action_id, action in failed_actions(trace).items()
    ]
    user_content += (
        "\n\nTrace-derived failed-action manifest. Every item below must be "
        "covered exactly once in agent_guidelines.required_action:\n"
        + json.dumps(action_manifest, ensure_ascii=False)
    )
    if validation_feedback:
        user_content += (
            "\n\nYour previous response was rejected by the output validator: "
            f"{validation_feedback}\nReturn a corrected complete JSON object."
        )
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        timeout=request_timeout,
        max_retries=1,
        **model_options,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response")
    return clean_json_response(content)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Tau2 traces and generate trace-grounded operating guidelines "
            "for the agent at the exact points where its behavior should change."
        )
    )
    parser.add_argument("--tests", default="1-9", help="Tests, for example 1-9")
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Maximum seconds to wait for one model request (default: 180).",
    )
    parser.add_argument(
        "--analysis-attempts",
        type=int,
        default=3,
        help="Maximum schema-validation attempts per trace (default: 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional model temperature. By default the provider chooses it.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace guideline files that exist"
    )
    return parser


def main() -> int:
    """Generate and save a diagnosis and agent guidelines for selected traces."""
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
            analysis_path = test_dir / "failure_instructions.json"
            shutil.copy2(source, saved_trace)

            if analysis_path.exists() and not args.overwrite:
                print(f"{label}: kept existing {analysis_path}")
                continue

            print(f"{label}: asking {args.model} for failure analysis...", flush=True)
            validation_feedback: str | None = None
            for attempt in range(1, args.analysis_attempts + 1):
                analysis = request_analysis(
                    prompt=prompt,
                    trace=trace,
                    model=args.model,
                    temperature=args.temperature,
                    request_timeout=args.request_timeout,
                    validation_feedback=validation_feedback,
                )
                try:
                    normalize_trace_locations(analysis, trace)
                    validate_analysis(analysis, trace)
                    break
                except ValueError as exc:
                    validation_feedback = str(exc)
                    if attempt == args.analysis_attempts:
                        raise
                    print(
                        f"{label}: validation attempt {attempt} failed; retrying: "
                        f"{validation_feedback}",
                        flush=True,
                    )
            analysis["test_id"] = str(
                trace.get("task_id") or analysis.get("test_id") or label
            )
            analysis["source_trace"] = str(source)
            analysis["model"] = args.model
            analysis["source_trace_sha256"] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            analysis["generated_at"] = datetime.now(timezone.utc).isoformat()
            analysis["note"] = (
                "Trace-grounded agent guidelines. These are operational rules for the "
                "agent, not user-facing scripts. No guideline or tool was run."
            )
            analysis_path.write_text(
                json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"{label}: saved {analysis_path}")
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
