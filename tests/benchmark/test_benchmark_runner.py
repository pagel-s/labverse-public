"""Tests for the benchmark runner and aggregate scoring."""

from __future__ import annotations

from typing import Any

from labverse.agent.interface import AgentInterface, Observation
from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage
from labverse.agent.text_planner import TextPlannerAgent
from labverse.benchmarks import BenchmarkRunner
from labverse.benchmarks.scenarios.base import BenchmarkScenario
from labverse.models.action import ActionSpec
from labverse.world.state import WorldState


def test_benchmark_runner_executes_all_scripted_scenarios() -> None:
    runner = BenchmarkRunner()
    scenario_count = len(runner.list_scenarios())

    aggregate = runner.run_all()

    assert aggregate.total_scenarios == scenario_count
    assert aggregate.passed == scenario_count
    assert aggregate.mean_completion == 1.0
    assert aggregate.mean_efficiency == 1.0
    assert aggregate.total_blocked_actions == 0
    assert {score.scenario_name for score in aggregate.per_scenario} == set(runner.list_scenarios())


class _InvalidJsonPlannerProvider(LLMProvider):
    """Provider that always returns non-JSON, exhausting the planner's retry budget."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, system
        self.call_count += 1
        return LLMResponse(
            text="I cannot comply.",
            usage=TokenUsage(prompt_tokens=17, completion_tokens=5),
        )


class _BrokenNoAuditAgent(AgentInterface):
    def __init__(self) -> None:
        self._agent_id = "grad_kim"

    def observe(self, observation: Observation) -> None:
        del observation

    def act(self, action_spec: ActionSpec):
        del action_spec
        raise RuntimeError("synthetic pre-provider crash")

    def query(self, question: str) -> str:
        del question
        return "broken"

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def name(self) -> str:
        return "BrokenNoAuditAgent"

    @property
    def audit_steps(self) -> list[dict[str, Any]]:
        return []


def test_one_shot_planner_failure_preserves_prompt_surface_audit() -> None:
    """Regression: a one-shot planner that never produces a valid plan must still leave
    its failed plan-prompt audit in the capture, so paper-status can read the prompt
    surface from the run metadata. See BenchmarkRunner._run_agents_with_replay's
    StopIteration handler."""

    runner = BenchmarkRunner()
    provider = _InvalidJsonPlannerProvider()

    def factory(
        scenario: BenchmarkScenario,
        world: WorldState,
    ) -> AgentInterface:
        del scenario
        return TextPlannerAgent(
            agent_id="grad_kim",
            provider=provider,
            goal="Set up the PCR thermocycler protocol.",
            world_state=world,
            name="TextPlanner-test",
        )

    capture = runner.run_scenario_capture("pcr_setup", agent_factory=factory)

    # The planner exhausts MAX_RETRIES (1 + 2 = 3 attempts) then raises StopIteration.
    assert provider.call_count == 3
    assert capture.step_audits, "Failed one-shot plan must still be harvested as an audit step"

    plan_audit = capture.step_audits[0]
    assert plan_audit["prompt_surface_kind"] == "global_one_shot_planner"
    assert plan_audit["system_prompt"], "system_prompt must survive on the failed plan audit"
    assert plan_audit["user_message"], "user_message must survive on the failed plan audit"
    assert plan_audit.get("status") == "failed"


def test_runner_synthesizes_artifact_step_for_zero_step_agent_failure() -> None:
    runner = BenchmarkRunner()

    def factory(
        scenario: BenchmarkScenario,
        world: WorldState,
    ) -> AgentInterface:
        del scenario, world
        return _BrokenNoAuditAgent()

    capture = runner.run_scenario_capture("pcr_setup", agent_factory=factory)

    assert capture.step_audits, "Zero-step agent failures must emit an audit artifact"
    failure_step = capture.step_audits[0]
    assert failure_step["status"] == "artifact_integrity_failure"
    assert failure_step["prompt_surface_kind"] == "artifact_integrity"
    assert "synthetic pre-provider crash" in failure_step["error"]
    assert capture.score.details["runner_error"].endswith("synthetic pre-provider crash")
    assert capture.score.completion_ratio == 0.0
    assert capture.score.steps_taken == 0


def test_runner_preserves_traceback_for_top_level_pre_step_failure() -> None:
    runner = BenchmarkRunner()

    class _BrokenScenario:
        name = "broken_pre_step"

        def configured_world(self):
            raise TypeError("tuple_iterator object does not support the context manager protocol")

    capture = runner.run_scenario_capture(_BrokenScenario())

    assert not capture.score.success
    assert capture.step_audits, "Top-level runner failures must emit an audit artifact"
    failure_step = capture.step_audits[0]
    assert failure_step["status"] == "artifact_integrity_failure"
    assert failure_step["prompt_surface_kind"] == "artifact_integrity"
    assert "runner_traceback" in capture.score.details
    assert "TypeError: tuple_iterator object does not support the context manager protocol" in (
        capture.score.details["runner_traceback"]
    )
