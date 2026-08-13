"""Tests for periodic RAC trace review and guideline reinjection."""

from pathlib import Path

from tau2.agent.rac_planner import RACPlannerAgent, RACPlannerAgentState
from tau2.agent.rac_progress_review import (
    NEW_INSTRUCTIONS_PROMPT,
    ProgressReviewResult,
    RACProgressReviewer,
    parse_review_response,
)
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage


def test_parse_making_progress_discards_guidelines() -> None:
    response = """
<analysis><progress>
<SufficentProgress>MakingProgress</SufficentProgress>
<OneSentenceSummary>Evidence is still accumulating.</OneSentenceSummary>
<Reason>A new diagnostic result was obtained.</Reason>
</progress><guidelinesToRecover>Do not send this.</guidelinesToRecover></analysis>
"""

    status, summary, reason, guidelines = parse_review_response(response)

    assert status == "MakingProgress"
    assert summary == "Evidence is still accumulating."
    assert reason == "A new diagnostic result was obtained."
    assert guidelines is None


def test_reviewer_saves_complete_round_and_previous_guidelines(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_generate(**kwargs):
        assert NEW_INSTRUCTIONS_PROMPT in kwargs["messages"][0].content
        assert '<TODO round="1">' in kwargs["messages"][0].content
        assert "<completedTurns>40</completedTurns>" in kwargs["messages"][0].content
        return AssistantMessage.text(
            """<analysis><progress>
<SufficentProgress>Stuck</SufficentProgress>
<OneSentenceSummary>The same failed test is repeating.</OneSentenceSummary>
<Reason>No corrective action followed the tool observation.</Reason>
</progress><guidelinesToRecover><guideline>
<description>Resolve the observed blocker before testing again.</description>
<SIngleSentenceReasoningForGuideline>The trace shows a repeated test.</SIngleSentenceReasoningForGuideline>
<whereGuidelineApplied>Before the next test.</whereGuidelineApplied>
</guideline></guidelinesToRecover></analysis>"""
        )

    monkeypatch.setattr("tau2.agent.rac_progress_review.generate", fake_generate)
    reviewer = RACProgressReviewer(model="test-model", interval=20, output_dir=tmp_path)

    result = reviewer.review(
        round_number=2,
        completed_turns=40,
        messages=[UserMessage(role="user", content="It still fails.")],
        previous_guidelines=["Check the network preference first."],
    )

    round_dir = tmp_path / "round_02_turn_0040"
    assert result.status == "Stuck"
    assert result.guidelines is not None
    assert (round_dir / "prompt_sent.txt").exists()
    assert (round_dir / "trace_sent.json").exists()
    assert (round_dir / "previous_guidelines_sent.json").exists()
    assert (round_dir / "planner_response.txt").exists()
    assert (round_dir / "guidelines_sent_to_agent.txt").read_text()
    assert (round_dir / "review_result.json").exists()


def test_rac_agent_injects_stuck_guidelines_once_per_boundary(monkeypatch) -> None:
    class FakeReviewer:
        interval = 20

        def review(self, **kwargs):
            return ProgressReviewResult(
                round_number=kwargs["round_number"],
                completed_turns=kwargs["completed_turns"],
                status="Stuck",
                summary="The flow is cycling.",
                reason="A blocker remains unresolved.",
                raw_response="response",
                guidelines="<guideline><description>Resolve the blocker.</description></guideline>",
                artifact_dir="/tmp/review",
            )

    agent = object.__new__(RACPlannerAgent)
    agent.progress_reviewer = FakeReviewer()
    monkeypatch.setattr(agent, "_save_live_trace", lambda state: None)
    state = RACPlannerAgentState(
        system_messages=[SystemMessage(role="system", content="policy")],
        messages=[],
    )
    trajectory = [UserMessage(role="user", content="Still broken")]

    state = agent.review_progress(
        completed_turns=20, trajectory=trajectory, state=state
    )
    state = agent.review_progress(
        completed_turns=21, trajectory=trajectory, state=state
    )

    assert state.last_progress_review_turn == 20
    assert len(state.progress_review_rounds) == 1
    assert len(state.recovery_guidelines) == 1
    assert len(state.system_messages) == 2
    assert '<TODO round="1"' in state.system_messages[-1].content
    assert (
        "Continue from the current environment state"
        in state.system_messages[-1].content
    )
