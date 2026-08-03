"""RAC Planner Agent for Tau2-Bench.

Integrates React Agent Compensation (RAC) planner pipeline with multi-objective Pareto plan selection,
RecoveryManager transaction logging, and ReAct failure backtracking recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

# Attempt import of RAC components, adding fallback path if needed
try:
    from react_agent_compensation.core.recovery_manager import RecoveryManager
    from react_agent_compensation.planner.models import ParetoSelectionResult, PlanCandidate
    from react_agent_compensation.planner.pipeline import PlannerPipeline
except ImportError:
    rac_src = Path(__file__).resolve().parents[4] / "react-agent-compensation" / "src"
    if rac_src.exists() and str(rac_src) not in sys.path:
        sys.path.insert(0, str(rac_src))
    from react_agent_compensation.core.recovery_manager import RecoveryManager
    from react_agent_compensation.planner.models import ParetoSelectionResult, PlanCandidate
    from react_agent_compensation.planner.pipeline import PlannerPipeline


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
        self.planner = PlannerPipeline()
        self.recovery_manager = RecoveryManager(compensation_pairs={})
        self._last_plan_result: ParetoSelectionResult | None = None

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
        return RACPlannerAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: RACPlannerAgentState
    ) -> tuple[AssistantMessage, RACPlannerAgentState]:
        """Process incoming user/tool message and generate assistant response guided by RAC planner."""
        # 1. Update message history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            # Check for tool errors in MultiToolMessage
            tool_errors = [tm for tm in message.tool_messages if getattr(tm, "error", None) or (isinstance(tm.content, str) and "error" in tm.content.lower())]
            if tool_errors:
                self._handle_failure_and_replan(tool_errors[0], state)
            else:
                for tm in message.tool_messages:
                    tool_name = getattr(tm, "name", "tool_call")
                    self.recovery_manager.record_action(tool_name, {})
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
            if getattr(message, "error", None) or (isinstance(message.content, str) and "error" in message.content.lower()):
                self._handle_failure_and_replan(message, state)
            else:
                self.recovery_manager.record_action(getattr(message, "name", "tool"), {})
        else:
            state.messages.append(message)

        # 2. Formulate LLM prompt (including plan strategy if replanned due to failure)
        messages_to_send = list(state.system_messages)
        if state.strategy_summary:
            messages_to_send.append(
                SystemMessage(
                    role="system",
                    content=f"[RAC Selected Plan Strategy: {state.current_plan_name}]\n{state.strategy_summary}",
                )
            )

        # 3. Pre-Response Verification & Numerical Refinement
        pre_response_guidance = self.planner.verify_and_refine_pre_response_plan(
            goals=[], messages=state.messages
        )
        if pre_response_guidance:
            messages_to_send.append(
                SystemMessage(
                    role="system",
                    content=pre_response_guidance,
                )
            )

        messages_to_send.extend(state.messages)

        # 4. Generate LLM action/response
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages_to_send,
            **self.llm_args,
        )

        state.messages.append(response)
        self._save_live_trace(state)
        return response, state

    def _save_live_trace(self, state: RACPlannerAgentState) -> None:
        """Flush current task state to data/simulations/live_trace.json for real-time step monitoring."""
        try:
            import json, time
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
                    t_calls = [{"name": c.name, "arguments": c.arguments} for c in m.tool_calls]
                msgs_dump.append({"role": role_str, "content": content, "tool_calls": t_calls})

            payload = {
                "current_plan_name": state.current_plan_name,
                "strategy_summary": state.strategy_summary,
                "candidates_info": state.candidates_info,
                "replan_count": state.replan_count,
                "messages": msgs_dump,
                "updated_at": time.strftime("%H:%M:%S"),
            }
            with open(trace_file, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Extract tool schema dicts from self.tools safely."""
        tool_schemas = []
        for t in self.tools:
            if hasattr(t, "openai_schema") and isinstance(t.openai_schema, dict):
                fn = t.openai_schema.get("function", {})
                tool_schemas.append({
                    "name": fn.get("name", getattr(t, "name", "tool")),
                    "description": fn.get("description", getattr(t, "short_desc", "")),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                tool_schemas.append({
                    "name": getattr(t, "name", "tool"),
                    "description": getattr(t, "short_desc", getattr(t, "long_desc", "")),
                })
        return tool_schemas

    def _create_initial_plan(self, state: RACPlannerAgentState) -> None:
        """Create Pareto optimal plan candidates using RAC Planner Pipeline."""
        tool_schemas = self._get_tool_schemas()
        user_goals = [
            m.content for m in state.messages if isinstance(m, UserMessage) and m.content
        ]
        goal_text = "\n".join(user_goals) if user_goals else "Assist customer per domain policy"

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
        tool_name = getattr(failed_message, "name", "tool")
        error_msg = str(failed_message.content) if failed_message.content else "Tool call failed"

        logger.warning(
            f"[RACPlannerAgent] Triggering RAC failure recovery & replanning for '{tool_name}' (cycle #{state.replan_count}/{self.max_replan_cycles})"
        )

        tool_schemas = self._get_tool_schemas()
        user_goals = [
            m.content for m in state.messages if isinstance(m, UserMessage) and m.content
        ]
        goal_text = "\n".join(user_goals) if user_goals else "Assist customer per domain policy"

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
            state.current_plan_name = f"Replan #{state.replan_count}: {replan_result.selected_candidate.name}"
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
            m.content for m in state.messages if isinstance(m, UserMessage) and m.content
        ]
        goal_text = "\n".join(user_goals) if user_goals else "Assist customer per domain policy"

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


def create_rac_planner_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "openai/gpt-5-mini",
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
