from types import SimpleNamespace

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.utils import llm_utils


def test_to_responses_input_replays_reasoning_and_tool_output():
    prior_output = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "check_status",
            "arguments": "{}",
        },
    ]
    messages = [
        SystemMessage(role="system", content="Follow policy."),
        UserMessage(role="user", content="Please check."),
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call_1", name="check_status", arguments={})],
            raw_data={"output": prior_output},
        ),
        ToolMessage(role="tool", id="call_1", content="connected"),
    ]

    instructions, response_input = llm_utils.to_responses_input(messages)

    assert instructions == "Follow policy."
    assert response_input == [
        {"role": "user", "content": "Please check."},
        *prior_output,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "connected",
        },
    ]


def test_generate_responses_uses_reasoning_effort_and_flat_tools(monkeypatch):
    captured = {}

    class FakeResponse:
        model = "gpt-5.6-luna"
        output_text = ""

        def get(self, key):
            if key == "usage":
                return SimpleNamespace(input_tokens=10, output_tokens=4)
            return None

        def model_dump(self, mode="json"):
            assert mode == "json"
            return {
                "model": self.model,
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": []},
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "check_status",
                        "arguments": '{"line_id": "L1002"}',
                    },
                ],
            }

    def fake_responses(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    class FakeTool:
        openai_schema = {
            "type": "function",
            "function": {
                "name": "check_status",
                "description": "Check a line.",
                "parameters": {"type": "object"},
            },
        }

    monkeypatch.setattr(llm_utils.litellm, "responses", fake_responses, raising=False)
    monkeypatch.setattr(llm_utils, "completion_cost", lambda **kwargs: 0.01)

    message = llm_utils.generate(
        model="openai/gpt-5.6-luna",
        messages=[UserMessage(role="user", content="Check my line.")],
        tools=[FakeTool()],
        api_mode="responses",
        reasoning_effort="medium",
    )

    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["tools"] == [
        {
            "type": "function",
            "name": "check_status",
            "description": "Check a line.",
            "parameters": {"type": "object"},
        }
    ]
    assert message.tool_calls == [
        ToolCall(
            id="call_1",
            name="check_status",
            arguments={"line_id": "L1002"},
        )
    ]
    assert message.usage == {"completion_tokens": 4, "prompt_tokens": 10}
    assert message.raw_data["output"][0]["type"] == "reasoning"
