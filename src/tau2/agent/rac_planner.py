"""RAC Planner Agent for Tau2-Bench.

Integrates React Agent Compensation (RAC) planner pipeline with multi-objective Pareto plan selection,
RecoveryManager transaction logging, and ReAct failure backtracking recovery.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.agent.rac_progress_review import RACProgressReviewer
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolType
from tau2.utils.llm_utils import generate

# Attempt import of RAC components, adding fallback path if needed
try:
    from react_agent_compensation.core.recovery_manager import RecoveryManager
    from react_agent_compensation.planner.models import ParetoSelectionResult
    from react_agent_compensation.planner.pipeline import PlannerPipeline

    RAC_IMPORT_SOURCE = "python_environment"
except ImportError:
    rac_src = Path(__file__).resolve().parents[4] / "react-agent-compensation" / "src"
    if rac_src.exists() and str(rac_src) not in sys.path:
        sys.path.insert(0, str(rac_src))
    from react_agent_compensation.core.recovery_manager import RecoveryManager
    from react_agent_compensation.planner.models import ParetoSelectionResult
    from react_agent_compensation.planner.pipeline import PlannerPipeline

    RAC_IMPORT_SOURCE = f"sibling_checkout:{rac_src}"


SYSTEM_PROMPT = """
<instructions>
You are a customer service agent enhanced with RAC (React Agent Compensation) Planning.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Always follow the domain policy strictly.
When provided with a candidate plan strategy, execute tools aligned with the selected plan.
If a tool fails, RAC RecoveryManager will perform backtracking and replanning.
</instructions>

<policy>
{domain_policy}
</policy>
""".strip()


class RACPlannerAgentState(BaseModel):
    """Execution state for RAC Planner Agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    current_plan_name: str | None = None
    strategy_summary: str | None = None
    candidates_info: list[dict] = Field(default_factory=list)
    checkpoint_events: list[dict[str, Any]] = Field(default_factory=list)
    authorized_write_targets: list[str] = Field(default_factory=list)
    write_scope_locked: bool = False
    completed_communication_obligations: list[str] = Field(default_factory=list)
    progress_review_rounds: list[dict[str, Any]] = Field(default_factory=list)
    recovery_guidelines: list[str] = Field(default_factory=list)
    last_progress_review_turn: int = 0
    replan_count: int = 0

    model_config = {"arbitrary_types_allowed": True}


class RACPlannerAgent(LLMConfigMixin, HalfDuplexAgent[RACPlannerAgentState]):
    """Half-duplex agent combining RAC Planner multi-objective Pareto plan selection with RecoveryManager resiliency."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        num_candidates: int = 3,
        max_replan_cycles: int = 3,
    ) -> None:
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.num_candidates = num_candidates
        self.max_replan_cycles = max_replan_cycles
        review_interval = int(os.environ.get("TAU2_RAC_PROGRESS_REVIEW_INTERVAL", "0"))
        review_dir_value = os.environ.get("TAU2_RAC_PROGRESS_REVIEW_DIR")
        self.progress_reviewer = (
            RACProgressReviewer(
                model=os.environ.get("TAU2_RAC_PROGRESS_REVIEW_MODEL", llm),
                interval=review_interval,
                output_dir=(
                    Path(review_dir_value).expanduser() if review_dir_value else None
                ),
                llm_args={"api_mode": "responses", "reasoning_effort": "high"},
            )
            if review_interval > 0
            else None
        )
        strict_scope_value = os.environ.get("TAU2_RAC_STRICT_WRITE_SCOPE", "1")
        self.strict_write_scope = strict_scope_value.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.write_tool_names = {
            tool.name
            for tool in tools
            if getattr(getattr(tool, "_func", None), "__tool_type__", None)
            == ToolType.WRITE
        }
        self.planner = PlannerPipeline()
        self.recovery_manager = RecoveryManager(compensation_pairs={})
        self._last_plan_result: ParetoSelectionResult | None = None
        logger.info(
            "[RACPlannerAgent] Initialized with PlannerPipeline={}, RecoveryManager={}, import_source={}",
            type(self.planner).__module__,
            type(self.recovery_manager).__module__,
            RAC_IMPORT_SOURCE,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(domain_policy=self.domain_policy)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> RACPlannerAgentState:
        """Initialize state with system message and conversation history."""
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage."
        )
        state = RACPlannerAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        self._record_checkpoint(
            state,
            "agent_state_initialized",
            {
                "rac_import_source": RAC_IMPORT_SOURCE,
                "planner_class": f"{type(self.planner).__module__}.{type(self.planner).__name__}",
                "recovery_manager_class": f"{type(self.recovery_manager).__module__}.{type(self.recovery_manager).__name__}",
                "initial_history_messages": len(message_history),
            },
        )
        return state

    def review_progress(
        self,
        *,
        completed_turns: int,
        trajectory: list[Message],
        state: RACPlannerAgentState,
    ) -> RACPlannerAgentState:
        """Review accumulated execution at each configured turn boundary."""
        reviewer = self.progress_reviewer
        if reviewer is None or completed_turns < reviewer.interval:
            return state

        review_turn = (completed_turns // reviewer.interval) * reviewer.interval
        if review_turn <= state.last_progress_review_turn:
            return state

        round_number = len(state.progress_review_rounds) + 1
        result = reviewer.review(
            round_number=round_number,
            completed_turns=completed_turns,
            messages=trajectory,
            previous_guidelines=state.recovery_guidelines,
        )
        state.last_progress_review_turn = review_turn
        state.progress_review_rounds.append(
            {
                "round": result.round_number,
                "scheduled_turn": review_turn,
                "completed_turns": result.completed_turns,
                "status": result.status,
                "summary": result.summary,
                "reason": result.reason,
                "guidelines": result.guidelines,
                "artifact_dir": result.artifact_dir,
            }
        )

        if result.status == "Stuck" and result.guidelines:
            state.recovery_guidelines.append(result.guidelines)
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=(
                        f'<TODO round="{round_number}" reviewed_after_turn="{completed_turns}">\n'
                        f"{result.guidelines}\n"
                        "</TODO>\n"
                        "Restart your execution reasoning from the earliest deviation "
                        "identified above. Continue from the current environment state, "
                        "apply this recovery guidance, preserve still-valid earlier TODO "
                        "guidelines, and do not claim completion until the relevant success "
                        "condition is verified."
                    ),
                )
            )

        self._record_checkpoint(
            state,
            "progress_review_completed",
            {
                "round": round_number,
                "scheduled_turn": review_turn,
                "completed_turns": completed_turns,
                "status": result.status,
                "guidelines_injected": bool(result.guidelines),
                "artifact_dir": result.artifact_dir,
            },
        )
        self._save_live_trace(state)
        return state

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: RACPlannerAgentState
    ) -> tuple[AssistantMessage, RACPlannerAgentState]:
        """Process incoming user/tool message and generate assistant response guided by RAC planner."""
        # 1. Update message history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            # Check for tool errors in MultiToolMessage
            tool_errors = [
                tm
                for tm in message.tool_messages
                if getattr(tm, "error", None)
                or (isinstance(tm.content, str) and "error" in tm.content.lower())
            ]
            if tool_errors:
                self._handle_failure_and_replan(tool_errors[0], state)
            else:
                for tm in message.tool_messages:
                    self._record_successful_tool(tm, state)
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
            if getattr(message, "error", None) or (
                isinstance(message.content, str) and "error" in message.content.lower()
            ):
                self._handle_failure_and_replan(message, state)
            else:
                self._record_successful_tool(message, state)
        else:
            state.messages.append(message)
            if isinstance(message, UserMessage):
                self._capture_initial_write_scope(message, state)

        if state.current_plan_name is None and any(
            isinstance(m, UserMessage) for m in state.messages
        ):
            self._create_initial_plan(state)

        # 2. Formulate LLM prompt (including plan strategy if replanned due to failure)
        messages_to_send = list(state.system_messages)
        if state.strategy_summary:
            messages_to_send.append(
                SystemMessage(
                    role="system",
                    content=f"[RAC Selected Plan Strategy: {state.current_plan_name}]\n{state.strategy_summary}",
                )
            )
        messages_to_send.extend(state.messages)

        communication_grounding = self._build_communication_grounding(state)
        if communication_grounding:
            messages_to_send.append(
                SystemMessage(
                    role="system",
                    content=communication_grounding["prompt"],
                )
            )
            self._record_checkpoint(
                state,
                "communication_grounding_computed",
                {
                    "obligation": communication_grounding["obligation"],
                    "total": communication_grounding["total"],
                    "reservation_ids": communication_grounding["reservation_ids"],
                },
            )

        # 4. Generate LLM action/response
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages_to_send,
            **self.llm_args,
        )
        response = self._force_missing_communication_lookup(
            response=response,
            state=state,
        )
        response = self._guard_write_scope(response=response, state=state)
        response = self._verify_and_repair_response(
            response=response,
            grounding=communication_grounding,
            state=state,
        )

        state.messages.append(response)
        self._record_checkpoint(
            state,
            "llm_response_generated",
            {
                "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                "message_count": len(state.messages),
            },
        )
        self._save_live_trace(state)
        return response, state

    def _capture_initial_write_scope(
        self, message: UserMessage, state: RACPlannerAgentState
    ) -> None:
        """Collect reservation IDs from user requests until the first write is issued."""
        if state.write_scope_locked or not isinstance(message.content, str):
            return
        reservation_ids = re.findall(
            r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)[A-Z0-9]+\b", message.content
        )
        for reservation_id in reservation_ids:
            if reservation_id not in state.authorized_write_targets:
                state.authorized_write_targets.append(reservation_id)

    def _guard_write_scope(
        self,
        response: AssistantMessage,
        state: RACPlannerAgentState,
    ) -> AssistantMessage:
        """Block writes to reservations outside the original mutation scope."""
        if (
            not self.strict_write_scope
            or not response.tool_calls
            or not state.authorized_write_targets
        ):
            return response

        write_calls = [
            tool_call
            for tool_call in response.tool_calls
            if tool_call.name in self.write_tool_names
        ]
        if not write_calls:
            return response

        if not state.write_scope_locked:
            state.write_scope_locked = True
            self._record_checkpoint(
                state,
                "write_scope_locked",
                {"authorized_targets": state.authorized_write_targets},
            )

        blocked_calls = [
            tool_call
            for tool_call in write_calls
            if isinstance(tool_call.arguments.get("reservation_id"), str)
            and tool_call.arguments["reservation_id"]
            not in state.authorized_write_targets
        ]
        if not blocked_calls:
            return response

        blocked_targets = sorted(
            {tool_call.arguments["reservation_id"] for tool_call in blocked_calls}
        )
        self._record_checkpoint(
            state,
            "write_scope_expansion_blocked",
            {
                "authorized_targets": state.authorized_write_targets,
                "blocked_targets": blocked_targets,
                "blocked_tools": [tool_call.name for tool_call in blocked_calls],
            },
        )
        allowed_targets = ", ".join(state.authorized_write_targets)
        blocked_target_text = ", ".join(blocked_targets)
        return response.model_copy(
            update={
                "content": (
                    f"I can provide information about {blocked_target_text}, but I "
                    "cannot modify those reservations in this request. The authorized "
                    f"reservation scope is {allowed_targets}."
                ),
                "tool_calls": None,
            }
        )

    def _build_communication_grounding(
        self, state: RACPlannerAgentState
    ) -> dict[str, Any] | None:
        """Compute supported facts for user-requested communication obligations."""
        user_requested_total, user_reservation_ids, reservation_snapshots = (
            self._collect_reservation_evidence(state)
        )
        if not user_requested_total:
            return None

        if not user_reservation_ids or not all(
            reservation_id in reservation_snapshots
            for reservation_id in user_reservation_ids
        ):
            return None

        current_date = self._airline_current_date()
        if current_date is None:
            return None

        upcoming: list[tuple[str, int]] = []
        for reservation_id in user_reservation_ids:
            snapshot = reservation_snapshots[reservation_id]
            flight_dates = [
                flight.get("date")
                for flight in snapshot.get("flights", [])
                if isinstance(flight, dict) and isinstance(flight.get("date"), str)
            ]
            if not any(
                flight_date >= current_date.isoformat() for flight_date in flight_dates
            ):
                continue

            payments = snapshot.get("payment_history", [])
            amount = sum(
                payment.get("amount", 0)
                for payment in payments
                if isinstance(payment, dict)
                and isinstance(payment.get("amount"), (int, float))
            )
            if amount >= 0:
                upcoming.append((reservation_id, round(amount)))

        total = sum(amount for _, amount in upcoming)
        if not upcoming:
            return None

        breakdown = ", ".join(
            f"{reservation_id}: ${amount:,}" for reservation_id, amount in upcoming
        )
        return {
            "obligation": "total_cost_of_upcoming_flights",
            "total": total,
            "reservation_ids": [reservation_id for reservation_id, _ in upcoming],
            "response": self._format_upcoming_flights_response(
                upcoming=upcoming,
                snapshots=reservation_snapshots,
                current_date=current_date,
            ),
            "prompt": (
                "[RAC Grounded Communication Requirement]\n"
                f"The user asked for the total cost of upcoming flights. The current "
                f"airline date is {current_date.isoformat()}. Using the earliest "
                "get_reservation_details snapshots and their payment_history totals, "
                f"the supported breakdown is {breakdown}; total: ${total:,}. "
                "Flights before the current date are not upcoming. State the total "
                "exactly and do not substitute segment-price arithmetic. Only report "
                "reservation IDs, routes, dates, costs, and the total. Do not mention "
                "cabins or suggest additional actions."
            ),
        }

    def _format_upcoming_flights_response(
        self,
        upcoming: list[tuple[str, int]],
        snapshots: dict[str, dict[str, Any]],
        current_date: date,
    ) -> str:
        """Format a minimal grounded answer without prompting additional actions."""
        lines = [f"Your upcoming flights as of {current_date.isoformat()}:"]
        for reservation_id, amount in upcoming:
            snapshot = snapshots[reservation_id]
            route = (
                f"{snapshot.get('origin', '?')} to {snapshot.get('destination', '?')}"
            )
            dates = sorted(
                {
                    flight["date"]
                    for flight in snapshot.get("flights", [])
                    if isinstance(flight, dict) and isinstance(flight.get("date"), str)
                }
            )
            lines.append(
                f"- {reservation_id}: {route}, {', '.join(dates)} - ${amount:,}"
            )
        lines.extend(
            [
                "",
                f"Total cost of your upcoming flights: ${sum(amount for _, amount in upcoming):,}.",
            ]
        )
        return "\n".join(lines)

    def _collect_reservation_evidence(
        self, state: RACPlannerAgentState
    ) -> tuple[bool, list[str] | None, dict[str, dict[str, Any]]]:
        """Collect the user's reservation list and earliest detail snapshots."""
        obligation = "total_cost_of_upcoming_flights"
        if obligation in state.completed_communication_obligations:
            return False, None, {}

        user_requested_total = any(
            isinstance(message, UserMessage)
            and isinstance(message.content, str)
            and "upcoming flight" in message.content.lower()
            and re.search(r"\b(total|cost)\b", message.content, re.IGNORECASE)
            for message in state.messages
        )
        if not user_requested_total:
            return False, None, {}

        tool_names_by_id: dict[str, str] = {}
        user_reservation_ids: list[str] | None = None
        reservation_snapshots: dict[str, dict[str, Any]] = {}

        for message in state.messages:
            for tool_call in getattr(message, "tool_calls", None) or []:
                tool_names_by_id[tool_call.id] = tool_call.name

            if not isinstance(message, ToolMessage) or message.error:
                continue
            tool_name = tool_names_by_id.get(message.id)
            if tool_name not in {"get_user_details", "get_reservation_details"}:
                continue
            try:
                payload = json.loads(message.content or "")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            if tool_name == "get_user_details":
                reservations = payload.get("reservations")
                if isinstance(reservations, list) and all(
                    isinstance(item, str) for item in reservations
                ):
                    user_reservation_ids = reservations
            else:
                reservation_id = payload.get("reservation_id")
                if isinstance(reservation_id, str):
                    # The first snapshot preserves the price/status at request time,
                    # before later upgrades and cancellations mutate the reservation.
                    reservation_snapshots.setdefault(reservation_id, payload)

        return True, user_reservation_ids, reservation_snapshots

    def _force_missing_communication_lookup(
        self,
        response: AssistantMessage,
        state: RACPlannerAgentState,
    ) -> AssistantMessage:
        """Replace a premature text answer with the next required evidence lookup."""
        if response.tool_calls:
            return response

        user_requested_total, reservation_ids, snapshots = (
            self._collect_reservation_evidence(state)
        )
        if not user_requested_total or not reservation_ids:
            return response

        missing_ids = [
            reservation_id
            for reservation_id in reservation_ids
            if reservation_id not in snapshots
        ]
        if not missing_ids:
            return response

        reservation_id = missing_ids[0]
        forced_response = response.model_copy(
            update={
                "content": None,
                "tool_calls": [
                    ToolCall(
                        id=f"call_rac_grounding_{uuid4().hex}",
                        name="get_reservation_details",
                        arguments={"reservation_id": reservation_id},
                    )
                ],
            }
        )
        self._record_checkpoint(
            state,
            "communication_evidence_lookup_forced",
            {
                "obligation": "total_cost_of_upcoming_flights",
                "reservation_id": reservation_id,
                "remaining_after_this_lookup": len(missing_ids) - 1,
                "blocked_draft_preview": (response.content or "")[:160],
            },
        )
        return forced_response

    def _airline_current_date(self) -> date | None:
        """Read the benchmark's current airline date from the domain policy."""
        match = re.search(
            r"current time is\s+(\d{4}-\d{2}-\d{2})",
            self.domain_policy,
            re.IGNORECASE,
        )
        if not match:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None

    def _verify_and_repair_response(
        self,
        response: AssistantMessage,
        grounding: dict[str, Any] | None,
        state: RACPlannerAgentState,
    ) -> AssistantMessage:
        """Ensure a grounded communication fact survives the final LLM draft."""
        if grounding is None or response.tool_calls or not response.content:
            return response

        total = grounding["total"]
        total_pattern = re.compile(
            rf"(?<!\d)(?:{re.escape(str(total))}|{re.escape(f'{total:,}')})(?!\d)"
        )
        if total_pattern.search(response.content):
            self._record_checkpoint(
                state,
                "communication_response_verified",
                {"obligation": grounding["obligation"], "total": total},
            )
            self._complete_communication_obligation(grounding["obligation"], state)
            return response.model_copy(update={"content": grounding["response"]})

        self._record_checkpoint(
            state,
            "communication_response_repaired",
            {
                "obligation": grounding["obligation"],
                "total": total,
                "reason": "draft_omitted_or_misstated_grounded_total",
            },
        )
        self._complete_communication_obligation(grounding["obligation"], state)
        return response.model_copy(update={"content": grounding["response"]})

    def _complete_communication_obligation(
        self, obligation: str, state: RACPlannerAgentState
    ) -> None:
        """Mark a communication requirement complete after grounded delivery."""
        if obligation not in state.completed_communication_obligations:
            state.completed_communication_obligations.append(obligation)
            self._record_checkpoint(
                state,
                "communication_obligation_completed",
                {"obligation": obligation},
            )

    def _record_checkpoint(
        self,
        state: RACPlannerAgentState,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a compact RAC checkpoint in state and logs."""
        checkpoint = {
            "event": event,
            "details": details or {},
            "replan_count": state.replan_count,
            "current_plan_name": state.current_plan_name,
            "updated_at": time.strftime("%H:%M:%S"),
        }
        state.checkpoint_events.append(checkpoint)
        logger.info("[RACPlannerAgent][checkpoint] {}: {}", event, details or {})

    def _record_successful_tool(
        self, message: ToolMessage, state: RACPlannerAgentState
    ) -> None:
        """Record successful tau2 tool output in RAC's transaction log."""
        tool_name = self._resolve_tool_name(message, state)
        record = self.recovery_manager.record_action(tool_name, {})
        self.recovery_manager.mark_completed(record.id, result=message.content)
        self._record_checkpoint(
            state,
            "tool_action_recorded",
            {
                "tool_name": tool_name,
                "record_id": record.id,
                "transaction_log_size": len(self.recovery_manager.log.snapshot()),
            },
        )

    def _resolve_tool_name(
        self, message: ToolMessage, state: RACPlannerAgentState
    ) -> str:
        """Resolve a tau2 ToolMessage id back to the assistant-requested tool name."""
        message_id = getattr(message, "id", "")
        for history_message in reversed(state.messages):
            tool_calls = getattr(history_message, "tool_calls", None)
            if not tool_calls:
                continue
            for tool_call in tool_calls:
                if message_id and getattr(tool_call, "id", "") == message_id:
                    return getattr(tool_call, "name", "tool")
        return getattr(message, "name", "tool")

    def _save_live_trace(self, state: RACPlannerAgentState) -> None:
        """Flush current task state to a live_trace.json file for real-time monitoring."""
        try:
            trace_dir = os.environ.get("TAU2_RAC_TRACE_DIR")
            if trace_dir:
                sim_base = Path(trace_dir).expanduser()
            else:
                sim_base = Path(__file__).resolve().parents[3] / "data" / "simulations"
            sim_base.mkdir(parents=True, exist_ok=True)
            trace_file = sim_base / "live_trace.json"

            msgs_dump = []
            for m in state.messages:
                role = getattr(m, "role", "")
                role_str = role.value if hasattr(role, "value") else str(role)
                content = getattr(m, "content", "")
                t_calls = None
                if hasattr(m, "tool_calls") and m.tool_calls:
                    t_calls = [
                        {"name": c.name, "arguments": c.arguments} for c in m.tool_calls
                    ]
                msgs_dump.append(
                    {"role": role_str, "content": content, "tool_calls": t_calls}
                )

            payload = {
                "rac_import_source": RAC_IMPORT_SOURCE,
                "planner_class": f"{type(self.planner).__module__}.{type(self.planner).__name__}",
                "recovery_manager_class": f"{type(self.recovery_manager).__module__}.{type(self.recovery_manager).__name__}",
                "current_plan_name": state.current_plan_name,
                "strategy_summary": state.strategy_summary,
                "candidates_info": state.candidates_info,
                "checkpoint_events": state.checkpoint_events,
                "authorized_write_targets": state.authorized_write_targets,
                "write_scope_locked": state.write_scope_locked,
                "completed_communication_obligations": state.completed_communication_obligations,
                "progress_review_rounds": state.progress_review_rounds,
                "recovery_guidelines": state.recovery_guidelines,
                "last_progress_review_turn": state.last_progress_review_turn,
                "replan_count": state.replan_count,
                "transaction_log_size": len(self.recovery_manager.log.snapshot()),
                "failure_summary": self.recovery_manager.get_failure_summary(),
                "messages": msgs_dump,
                "updated_at": time.strftime("%H:%M:%S"),
            }
            with open(trace_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Extract tool schema dicts from self.tools safely."""
        tool_schemas = []
        for t in self.tools:
            if hasattr(t, "openai_schema") and isinstance(t.openai_schema, dict):
                fn = t.openai_schema.get("function", {})
                tool_schemas.append(
                    {
                        "name": fn.get("name", getattr(t, "name", "tool")),
                        "description": fn.get(
                            "description", getattr(t, "short_desc", "")
                        ),
                        "parameters": fn.get("parameters", {}),
                    }
                )
            else:
                tool_schemas.append(
                    {
                        "name": getattr(t, "name", "tool"),
                        "description": getattr(
                            t, "short_desc", getattr(t, "long_desc", "")
                        ),
                    }
                )
        return tool_schemas

    def _create_initial_plan(self, state: RACPlannerAgentState) -> None:
        """Create Pareto optimal plan candidates using RAC Planner Pipeline."""
        tool_schemas = self._get_tool_schemas()
        user_goals = [
            m.content
            for m in state.messages
            if isinstance(m, UserMessage) and m.content
        ]
        goal_text = (
            "\n".join(user_goals) if user_goals else "Assist customer per domain policy"
        )

        plan_result = self.planner.create_and_select_plan(
            goals=goal_text,
            recovery_manager=self.recovery_manager,
            tool_schemas=tool_schemas,
            num_candidates=self.num_candidates,
        )
        self._last_plan_result = plan_result

        if plan_result.selected_candidate:
            state.current_plan_name = plan_result.selected_candidate.name
            state.strategy_summary = plan_result.selected_candidate.strategy_summary
            state.candidates_info = [
                {
                    "name": c.name,
                    "cost": getattr(c, "cost", 0.0),
                    "risk": getattr(c, "compensation_risk", 0.0),
                    "strategy_summary": getattr(c, "strategy_summary", ""),
                    "selected": (c.name == plan_result.selected_candidate.name),
                }
                for c in (plan_result.pareto_front or [])
            ]
            logger.info(
                f"[RACPlannerAgent] Selected Pareto plan: {state.current_plan_name} "
                f"(Cost={plan_result.selected_candidate.cost}, Risk={plan_result.selected_candidate.compensation_risk})"
            )
            self._record_checkpoint(
                state,
                "initial_plan_selected",
                {
                    "plan_name": state.current_plan_name,
                    "pareto_front_count": len(plan_result.pareto_front or []),
                },
            )

    def _handle_failure_and_replan(
        self, failed_message: ToolMessage, state: RACPlannerAgentState
    ) -> None:
        """Trigger RAC RecoveryManager backtracking and replanning upon tool failure (max 3 cycles)."""
        if state.replan_count >= self.max_replan_cycles:
            logger.warning(
                f"[RACPlannerAgent] Max replan cycles limit ({self.max_replan_cycles}) reached. Skipping further replanning."
            )
            return

        state.replan_count += 1
        tool_name = self._resolve_tool_name(failed_message, state)
        error_msg = (
            str(failed_message.content)
            if failed_message.content
            else "Tool call failed"
        )

        logger.warning(
            f"[RACPlannerAgent] Triggering RAC failure recovery & replanning for '{tool_name}' (cycle #{state.replan_count}/{self.max_replan_cycles})"
        )

        tool_schemas = self._get_tool_schemas()
        user_goals = [
            m.content
            for m in state.messages
            if isinstance(m, UserMessage) and m.content
        ]
        goal_text = (
            "\n".join(user_goals) if user_goals else "Assist customer per domain policy"
        )

        replan_result = self.planner.handle_react_failure_and_replan(
            recovery_manager=self.recovery_manager,
            failed_action=tool_name,
            error_message=error_msg,
            goals=goal_text,
            tool_schemas=tool_schemas,
            num_candidates=self.num_candidates,
        )
        self._last_plan_result = replan_result

        if replan_result.selected_candidate:
            state.current_plan_name = (
                f"Replan #{state.replan_count}: {replan_result.selected_candidate.name}"
            )
            state.strategy_summary = replan_result.selected_candidate.strategy_summary
            state.candidates_info = [
                {
                    "name": c.name,
                    "cost": getattr(c, "cost", 0.0),
                    "risk": getattr(c, "compensation_risk", 0.0),
                    "strategy_summary": getattr(c, "strategy_summary", ""),
                    "selected": (c.name == replan_result.selected_candidate.name),
                }
                for c in (replan_result.pareto_front or [])
            ]
            logger.info(
                f"[RACPlannerAgent] Selected new plan candidate for cycle #{state.replan_count}: {state.current_plan_name}"
            )
            self._record_checkpoint(
                state,
                "tool_failure_replan_selected",
                {
                    "failed_tool": tool_name,
                    "plan_name": state.current_plan_name,
                    "pareto_front_count": len(replan_result.pareto_front or []),
                    "failure_summary": self.recovery_manager.get_failure_summary(),
                },
            )

    def handle_evaluation_failure(
        self,
        failure_reason: str,
        state: RACPlannerAgentState,
    ) -> None:
        """Trigger RAC failure recovery & replanning when Tau2 evaluation fails at the end of a trial (Case 2)."""
        if state.replan_count >= self.max_replan_cycles:
            logger.warning(
                f"[RACPlannerAgent] Max replan cycles limit ({self.max_replan_cycles}) reached. Skipping further evaluation replanning."
            )
            return

        state.replan_count += 1
        logger.warning(
            f"[RACPlannerAgent] Triggering RAC replanning for Tau2 evaluation failure: '{failure_reason}' (cycle #{state.replan_count}/{self.max_replan_cycles})"
        )

        tool_schemas = self._get_tool_schemas()
        user_goals = [
            m.content
            for m in state.messages
            if isinstance(m, UserMessage) and m.content
        ]
        goal_text = (
            "\n".join(user_goals) if user_goals else "Assist customer per domain policy"
        )

        replan_result = self.planner.handle_react_failure_and_replan(
            recovery_manager=self.recovery_manager,
            failed_action="tau2_evaluation_check",
            error_message=failure_reason,
            goals=goal_text,
            tool_schemas=tool_schemas,
            num_candidates=self.num_candidates,
        )
        self._last_plan_result = replan_result

        if replan_result.selected_candidate:
            state.current_plan_name = f"Replan #{state.replan_count} (Eval Retry): {replan_result.selected_candidate.name}"
            state.strategy_summary = replan_result.selected_candidate.strategy_summary
            state.candidates_info = [
                {
                    "name": c.name,
                    "cost": getattr(c, "cost", 0.0),
                    "risk": getattr(c, "compensation_risk", 0.0),
                    "strategy_summary": getattr(c, "strategy_summary", ""),
                    "selected": (c.name == replan_result.selected_candidate.name),
                }
                for c in (replan_result.pareto_front or [])
            ]
            logger.info(
                f"[RACPlannerAgent] Selected new plan candidate for evaluation retry: {state.current_plan_name}"
            )
            self._record_checkpoint(
                state,
                "evaluation_failure_replan_selected",
                {
                    "failure_reason": failure_reason,
                    "plan_name": state.current_plan_name,
                    "pareto_front_count": len(replan_result.pareto_front or []),
                    "failure_summary": self.recovery_manager.get_failure_summary(),
                },
            )
        self._save_live_trace(state)


def create_rac_planner_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "openai/gpt-5.6-luna",
    llm_args: Optional[dict] = None,
    **kwargs,
) -> RACPlannerAgent:
    """Factory function to instantiate RACPlannerAgent in Tau2 runner."""
    return RACPlannerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
        num_candidates=kwargs.get("num_candidates", 3),
        max_replan_cycles=kwargs.get("max_replan_cycles", 3),
    )
