#!/usr/bin/env python3
"""Run the nine guided Telecom MMS cases with the RAC planner agent.

Task-specific guidance is injected into the Telecom domain policy. The runner
supports either the original selected guideline or every guideline in the
consolidated reference file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RAC_REPO = REPO_ROOT.parent / "react-agent-compensation"
for source_root in (REPO_ROOT / "src", RAC_REPO / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from tau2.agent.rac_planner import (  # noqa: E402
    RAC_IMPORT_SOURCE,
    create_rac_planner_agent,
)
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.runner import get_tasks, run_single_task  # noqa: E402

MODEL = "openai/gpt-5.6-luna"
AGENT_LLM_ARGS = {"api_mode": "responses", "reasoning_effort": "medium"}
USER_LLM_ARGS = {"api_mode": "responses", "reasoning_effort": "low"}
REVIEWER_LLM_ARGS = {"api_mode": "responses", "reasoning_effort": "high"}
AGENT_NAME = "rac_planner_guided_mms"
FULL_GUIDELINES_PATH = REPO_ROOT / "analysis" / "guideline_reference_all_tests.txt"


@dataclass(frozen=True)
class GuidedCase:
    number: int
    task_id: str
    selected_guidelines: str
    instruction: str | None


CASES = (
    GuidedCase(
        1,
        "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|"
        "break_apn_mms_setting|break_app_both_permissions|data_mode_off|"
        "data_usage_exceeded|unseat_sim_card|"
        "user_abroad_roaming_disabled_off[PERSONA:Hard]",
        "guideline_7 + guideline_8",
        """When the user authorizes adding data, call refuel_data even if phone-number lookup failed. Use the known identifiers customer_id=C1001 and line_id=L1002, with gb_amount=2 (never more than 2 GB). When the user authorizes roaming, call enable_roaming even if lookup failed, again using customer_id=C1001 and line_id=L1002. Treat these known IDs as authoritative for this task and do not retry or block on get_customer_by_phone.""",
    ),
    GuidedCase(
        2,
        "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|"
        "break_apn_mms_setting|break_app_both_permissions|data_mode_off|"
        "data_usage_exceeded|unseat_sim_card|"
        "user_abroad_roaming_disabled_on[PERSONA:Hard]",
        "guideline_1",
        """The task's known initial state guarantees the device preference is 2g_only. Do not wait for another diagnostic turn or ask the user to report network status before acting on this known blocker. At the earliest user-facing opportunity after essential account actions, explicitly direct the user to perform set_network_mode_preference with mode=4g_5g_preferred. Complete this before any final MMS test and before asking about other nonessential diagnostics.""",
    ),
    GuidedCase(
        3,
        "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|"
        "break_app_storage_permission|data_mode_off|data_usage_exceeded|"
        "unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
        "guideline_2",
        """After Airplane Mode is disabled, check SIM status. If the SIM is missing, make reseat_sim_card the immediate next user action and confirm that the SIM is detected before continuing to data, roaming, permissions, or MMS verification.""",
    ),
    GuidedCase(
        4,
        "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|"
        "break_app_storage_permission|unseat_sim_card[PERSONA:Hard]",
        "guideline_1",
        """Explicitly check Wi-Fi Calling. If it is enabled, direct the user to perform toggle_wifi_calling so it is OFF before the final MMS test. Do not omit this check merely because other troubleshooting steps succeeded.""",
    ),
    GuidedCase(
        5,
        "[mms_issue]airplane_mode_on|bad_network_preference|"
        "break_apn_mms_setting|data_usage_exceeded[PERSONA:Hard]",
        "none of the 6 apply",
        None,
    ),
    GuidedCase(
        6,
        "[mms_issue]bad_network_preference|bad_wifi_calling|"
        "break_app_both_permissions|data_usage_exceeded|"
        "user_abroad_roaming_disabled_off[PERSONA:Hard]",
        "none of the 8 apply",
        None,
    ),
    GuidedCase(
        7,
        "[mms_issue]bad_network_preference|bad_wifi_calling|"
        "break_app_sms_permission|data_mode_off|unseat_sim_card|"
        "user_abroad_roaming_enabled_off[PERSONA:Hard]",
        "guideline_1",
        """Act on SIM evidence already returned by tools. If SIM Card Status is missing, immediately direct the user to perform reseat_sim_card and confirm service is restored before proceeding to data, roaming, permissions, or MMS verification.""",
    ),
    GuidedCase(
        8,
        "[mms_issue]bad_wifi_calling|break_apn_mms_setting|"
        "break_app_sms_permission|data_usage_exceeded[PERSONA:Hard]",
        "guideline_3",
        """Inspect the complete messaging-app permission result, not only storage. If SMS is absent (for example, the result lists storage and phone), explicitly direct the user to perform grant_app_permission with app_name=messaging and permission=sms. Confirm it is granted before the final MMS test.""",
    ),
    GuidedCase(
        9,
        "[mms_issue]break_apn_mms_setting|data_mode_off|"
        "data_usage_exceeded|"
        "user_abroad_roaming_disabled_on[PERSONA:Hard]",
        "guideline_3",
        """Do not transfer or end after account-side fixes. After APN reset and reboot, explicitly direct the user to perform toggle_data so mobile data is ON, confirm completion, and only then test MMS.""",
    ),
)

CASE_BY_ID = {case.task_id: case for case in CASES}
ACTIVE_GUIDANCE: dict[str, tuple[str, str | None]] = {
    case.task_id: (case.selected_guidelines, case.instruction) for case in CASES
}


def load_full_guidelines(path: Path) -> dict[str, tuple[str, str]]:
    """Load and validate the TEST_XX blocks in the consolidated reference."""
    text = path.read_text()
    markers = list(re.finditer(r"(?m)^TEST_(\d{2})$", text))
    if len(markers) != len(CASES):
        raise ValueError(
            f"Expected {len(CASES)} TEST blocks in {path}, found {len(markers)}"
        )

    guidance: dict[str, tuple[str, str]] = {}
    for index, marker in enumerate(markers):
        number = int(marker.group(1))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        block = text[marker.start() : end].strip("#\n ")
        case = next((item for item in CASES if item.number == number), None)
        if case is None:
            raise ValueError(f"Unexpected TEST_{number:02d} in {path}")
        expected_id = f"Test ID: {case.task_id}"
        if expected_id not in block:
            raise ValueError(f"TEST_{number:02d} does not match expected task ID")
        guidance[case.task_id] = ("all guidelines", block)
    return guidance


def parse_case_numbers(raw: str) -> list[int]:
    """Parse comma-separated case numbers and inclusive ranges."""
    numbers: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid descending range: {part}")
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    invalid = numbers - {case.number for case in CASES}
    if not numbers or invalid:
        raise ValueError(f"Cases must be from 1 through 9; invalid: {sorted(invalid)}")
    return sorted(numbers)


def guided_agent_factory(
    tools,
    domain_policy: str,
    llm: str,
    llm_args: dict[str, Any] | None = None,
    task=None,
    **kwargs: Any,
):
    """Create RAC and add the selected per-task operating instruction."""
    case = CASE_BY_ID.get(getattr(task, "id", None))
    if case is None:
        raise ValueError(f"No guided MMS case for task: {getattr(task, 'id', None)}")

    selected_guidelines, instruction = ACTIVE_GUIDANCE[case.task_id]
    if instruction:
        domain_policy = (
            f"{domain_policy}\n\n"
            "<selected_rac_test_guidance>\n"
            "This instruction is task-specific and mandatory. Preserve all normal "
            "domain-policy requirements, including required consent. Track this "
            "instruction until its success condition is completed; do not skip it, "
            "prematurely transfer, or end the conversation while it remains actionable.\n"
            f"Selected: {selected_guidelines}\n"
            "Execution requirement: Treat the following as a complete ordered "
            "checklist for this exact known task. Work through every applicable "
            "guideline, one action at a time. Maintain an internal checklist of "
            "completed actions. Do not attempt the final MMS verification or end "
            "the interaction while a required action remains incomplete. For "
            "requestor=user actions, explicitly direct the user to perform the "
            "named tool action with the exact arguments. For requestor=assistant "
            "actions, call the tool yourself after required consent.\n"
            f"Instruction:\n{instruction}\n"
            "</selected_rac_test_guidance>"
        )

    return create_rac_planner_agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
        **kwargs,
    )


def write_json(path: Path, value: Any) -> None:
    """Write JSON via a temporary file so interrupted runs keep prior traces valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    if hasattr(value, "model_dump_json"):
        temporary.write_text(value.model_dump_json(indent=2) + "\n")
    else:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="1-9",
        help="Case numbers/ranges to run (default: 1-9)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--progress-review-interval",
        type=int,
        default=20,
        help="Ask the planning supervisor to review the accumulated trace every N turns",
    )
    parser.add_argument(
        "--progress-review-model",
        default=MODEL,
        help="Model used for periodic progress reviews",
    )
    parser.add_argument(
        "--guideline-set",
        choices=("single", "full"),
        default="single",
        help="Inject the selected guideline or every guideline in the reference",
    )
    args = parser.parse_args()

    if args.progress_review_interval < 0:
        raise ValueError("--progress-review-interval must be zero or greater")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is not set in the environment or {REPO_ROOT / '.env'}"
        )
    if not RAC_REPO.is_dir():
        raise RuntimeError(f"RAC repository not found: {RAC_REPO}")

    selected_numbers = parse_case_numbers(args.cases)
    selected_cases = [case for case in CASES if case.number in selected_numbers]
    global ACTIVE_GUIDANCE
    if args.guideline_set == "full":
        ACTIVE_GUIDANCE = load_full_guidelines(FULL_GUIDELINES_PATH)
        output_prefix = "full_guidelines"
    else:
        ACTIVE_GUIDANCE = {
            case.task_id: (case.selected_guidelines, case.instruction) for case in CASES
        }
        output_prefix = "guided"

    if AGENT_NAME not in registry.get_agents():
        registry.register_agent_factory(guided_agent_factory, AGENT_NAME)

    tasks = get_tasks(
        "telecom",
        task_ids=[case.task_id for case in selected_cases],
    )
    tasks_by_id = {task.id: task for task in tasks}
    config = TextRunConfig(
        domain="telecom",
        agent=AGENT_NAME,
        llm_agent=MODEL,
        llm_user=MODEL,
        llm_args_agent=AGENT_LLM_ARGS,
        llm_args_user=USER_LLM_ARGS,
        max_steps=args.max_steps,
        seed=args.seed,
        log_level="INFO",
    )

    started_at = datetime.now(timezone.utc).isoformat()
    summary_path = REPO_ROOT / "analysis" / f"{output_prefix}_rac_summary.json"
    existing_runs: list[dict[str, Any]] = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing_runs = [
            record
            for record in existing.get("runs", [])
            if record.get("case") not in selected_numbers
        ]
    summary: dict[str, Any] = {
        "started_at": started_at,
        "model": MODEL,
        "agent": AGENT_NAME,
        "rac_import_source": RAC_IMPORT_SOURCE,
        "rac_repository": str(RAC_REPO),
        "guideline_set": args.guideline_set,
        "guideline_source": (
            str(FULL_GUIDELINES_PATH) if args.guideline_set == "full" else None
        ),
        "progress_review_interval": args.progress_review_interval,
        "progress_review_model": args.progress_review_model,
        "agent_llm_args": AGENT_LLM_ARGS,
        "user_llm_args": USER_LLM_ARGS,
        "progress_reviewer_llm_args": REVIEWER_LLM_ARGS,
        "runs": existing_runs,
    }
    failures = 0

    print(f"RAC source: {RAC_IMPORT_SOURCE}")
    print(f"Model (agent and user): {MODEL}")
    for case in selected_cases:
        output_dir = REPO_ROOT / "analysis" / f"test_{case.number:02d}"
        progress_review_dir = output_dir / f"{output_prefix}_progress_reviews"
        os.environ["TAU2_RAC_TRACE_DIR"] = str(output_dir)
        os.environ["TAU2_RAC_PROGRESS_REVIEW_INTERVAL"] = str(
            args.progress_review_interval
        )
        os.environ["TAU2_RAC_PROGRESS_REVIEW_MODEL"] = args.progress_review_model
        os.environ["TAU2_RAC_PROGRESS_REVIEW_DIR"] = str(progress_review_dir)
        selected_guidelines, instruction = ACTIVE_GUIDANCE[case.task_id]
        metadata = {
            "case": case.number,
            "task_id": case.task_id,
            "selected_guidelines": selected_guidelines,
            "injected_instruction": instruction,
            "guideline_set": args.guideline_set,
            "guideline_source": (
                str(FULL_GUIDELINES_PATH) if args.guideline_set == "full" else None
            ),
            "model": MODEL,
            "agent": AGENT_NAME,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "progress_review_interval": args.progress_review_interval,
            "progress_review_model": args.progress_review_model,
            "agent_llm_args": AGENT_LLM_ARGS,
            "user_llm_args": USER_LLM_ARGS,
            "progress_reviewer_llm_args": REVIEWER_LLM_ARGS,
            "progress_review_dir": str(progress_review_dir),
        }
        write_json(output_dir / f"{output_prefix}_run_config.json", metadata)
        print(f"\n[{case.number}/9] Running {case.task_id}", flush=True)
        try:
            simulation = run_single_task(
                config,
                tasks_by_id[case.task_id],
                seed=args.seed,
                save_dir=output_dir,
                verbose_logs=True,
            )
            trace_path = output_dir / f"{output_prefix}_trace.json"
            write_json(trace_path, simulation)
            reward = simulation.reward_info.reward if simulation.reward_info else None
            record = {
                **metadata,
                "status": "completed",
                "simulation_id": simulation.id,
                "reward": reward,
                "trace": str(trace_path),
            }
            error_path = output_dir / f"{output_prefix}_run_error.json"
            if error_path.exists():
                error_path.unlink()
            print(f"[{case.number}/9] Completed: reward={reward}", flush=True)
        except Exception as exc:
            failures += 1
            record = {
                **metadata,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json(output_dir / f"{output_prefix}_run_error.json", record)
            print(f"[{case.number}/9] ERROR: {type(exc).__name__}: {exc}", flush=True)
        summary["runs"].append(record)
        summary["runs"].sort(key=lambda item: item["case"])
        write_json(summary_path, summary)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["completed"] = sum(
        record.get("status") == "completed" for record in summary["runs"]
    )
    summary["failed"] = sum(
        record.get("status") == "error" for record in summary["runs"]
    )
    write_json(summary_path, summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
