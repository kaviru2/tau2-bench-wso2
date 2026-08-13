"""Focused tests for RAC planner communication grounding."""

import json

from tau2.agent.rac_planner import RACPlannerAgent, RACPlannerAgentState
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def _tool_exchange(name: str, call_id: str, payload: dict) -> list:
    return [
        AssistantMessage.text(
            "",
            tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
        ),
        ToolMessage(id=call_id, role="tool", content=json.dumps(payload)),
    ]


def _agent() -> RACPlannerAgent:
    agent = object.__new__(RACPlannerAgent)
    agent.domain_policy = "The current time is 2024-05-15 15:00:00 EST."
    agent.strict_write_scope = True
    agent.write_tool_names = {"cancel_reservation", "update_reservation_flights"}
    return agent


def _state() -> RACPlannerAgentState:
    messages = [
        UserMessage(
            role="user",
            content="Check my other upcoming flights and tell me the total cost.",
        )
    ]
    messages += _tool_exchange(
        "get_user_details",
        "user",
        {"reservations": ["UP1", "UP2", "PAST"]},
    )
    messages += _tool_exchange(
        "get_reservation_details",
        "up1",
        {
            "reservation_id": "UP1",
            "flights": [{"date": "2024-05-20", "price": 10}],
            "payment_history": [{"amount": 296}],
        },
    )
    messages += _tool_exchange(
        "get_reservation_details",
        "up2",
        {
            "reservation_id": "UP2",
            "flights": [{"date": "2024-05-28", "price": 20}],
            "payment_history": [{"amount": 1332}],
        },
    )
    messages += _tool_exchange(
        "get_reservation_details",
        "past",
        {
            "reservation_id": "PAST",
            "flights": [{"date": "2024-05-14", "price": 9999}],
            "payment_history": [{"amount": 9999}],
        },
    )
    return RACPlannerAgentState(
        system_messages=[SystemMessage(role="system", content="policy")],
        messages=messages,
    )


def test_upcoming_flight_total_uses_payments_and_excludes_past_flights() -> None:
    grounding = _agent()._build_communication_grounding(_state())

    assert grounding is not None
    assert grounding["total"] == 1628
    assert grounding["reservation_ids"] == ["UP1", "UP2"]
    assert "segment-price arithmetic" in grounding["prompt"]


def test_wrong_upcoming_total_is_repaired() -> None:
    agent = _agent()
    state = _state()
    grounding = agent._build_communication_grounding(state)
    draft = AssistantMessage.text(
        "Summary\nTotal cost for those upcoming flights: $6,120."
    )

    repaired = agent._verify_and_repair_response(draft, grounding, state)

    assert "Total cost of your upcoming flights: $1,628." in repaired.content
    assert "$6,120" not in repaired.content
    assert state.checkpoint_events[-1]["event"] == "communication_obligation_completed"
    assert state.completed_communication_obligations == [
        "total_cost_of_upcoming_flights"
    ]


def test_larger_number_does_not_count_as_the_grounded_total() -> None:
    agent = _agent()
    state = _state()
    grounding = agent._build_communication_grounding(state)
    draft = AssistantMessage.text("Total cost for those upcoming flights: $16,280.")

    repaired = agent._verify_and_repair_response(draft, grounding, state)

    assert repaired.content.endswith("Total cost of your upcoming flights: $1,628.")


def test_grounding_waits_until_all_reservations_are_fetched() -> None:
    state = _state()
    state.messages = state.messages[:-2]

    assert _agent()._build_communication_grounding(state) is None


def test_premature_answer_is_replaced_with_missing_reservation_lookup() -> None:
    agent = _agent()
    state = _state()
    state.messages = state.messages[:-2]
    draft = AssistantMessage.text("The total is $1,628.")

    response = agent._force_missing_communication_lookup(draft, state)

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_reservation_details"
    assert response.tool_calls[0].arguments == {"reservation_id": "PAST"}
    assert (
        state.checkpoint_events[-1]["event"] == "communication_evidence_lookup_forced"
    )


def test_existing_tool_call_is_not_replaced() -> None:
    agent = _agent()
    state = _state()
    state.messages = state.messages[:-2]
    draft = AssistantMessage.text(
        "",
        tool_calls=[
            ToolCall(
                id="existing",
                name="get_reservation_details",
                arguments={"reservation_id": "PAST"},
            )
        ],
    )

    response = agent._force_missing_communication_lookup(draft, state)

    assert response is draft
    assert state.checkpoint_events == []


def test_write_scope_blocks_new_reservations_after_first_write() -> None:
    agent = _agent()
    state = RACPlannerAgentState(
        system_messages=[SystemMessage(role="system", content="policy")],
        messages=[],
    )
    initial_request = UserMessage(
        role="user",
        content="Cancel reservations XEHM4B and 59XX6W.",
    )
    agent._capture_initial_write_scope(initial_request, state)

    allowed_write = AssistantMessage.text(
        "",
        tool_calls=[
            ToolCall(
                id="allowed",
                name="cancel_reservation",
                arguments={"reservation_id": "59XX6W"},
            )
        ],
    )
    assert agent._guard_write_scope(allowed_write, state) is allowed_write
    assert state.write_scope_locked is True
    assert state.authorized_write_targets == ["XEHM4B", "59XX6W"]

    expanded_request = UserMessage(
        role="user",
        content="Upgrade and cancel 7WPL39 and 3EMQJ6.",
    )
    agent._capture_initial_write_scope(expanded_request, state)
    blocked_write = AssistantMessage.text(
        "",
        tool_calls=[
            ToolCall(
                id="blocked",
                name="update_reservation_flights",
                arguments={"reservation_id": "7WPL39", "cabin": "business"},
            )
        ],
    )

    response = agent._guard_write_scope(blocked_write, state)

    assert response.tool_calls is None
    assert "cannot modify" in response.content
    assert "7WPL39" in response.content
    assert state.authorized_write_targets == ["XEHM4B", "59XX6W"]
    assert state.checkpoint_events[-1]["event"] == "write_scope_expansion_blocked"


def test_completed_grounding_obligation_does_not_repeat() -> None:
    agent = _agent()
    state = _state()
    grounding = agent._build_communication_grounding(state)
    response = AssistantMessage.text("The supported total is $1,628.")

    agent._verify_and_repair_response(response, grounding, state)

    assert agent._build_communication_grounding(state) is None
