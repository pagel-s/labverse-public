"""Benchmark execution over scenarios and agent implementations."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from labverse.agent.audit import (
    attach_action_result,
    build_artifact_integrity_step,
    fail_step_audit,
    mark_post_completion_noop,
    materialize_audit_steps,
    read_actor_audit_steps,
)
from labverse.agent.interface import AgentInterface
from labverse.agent.prompt import TurnStatus
from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.initial_variants import (
    BASELINE_INITIAL_VARIANT_ID,
    apply_initial_condition_variant,
)
from labverse.benchmarks.replay import build_replay_frames_for_steps
from labverse.benchmarks.scenarios import SCENARIO_REGISTRY
from labverse.benchmarks.scenarios.base import BenchmarkScenario
from labverse.benchmarks.scorer import AggregateScore, BenchmarkScore, BenchmarkScorer
from labverse.gamemaster import GameMaster
from labverse.world.state import WorldState

AgentFactory = Callable[
    [BenchmarkScenario, WorldState],
    Sequence[AgentInterface] | AgentInterface,
]


def _clock_tick(game_master: GameMaster) -> int:
    clock = getattr(game_master, "clock", None)
    return int(getattr(clock, "tick", 0) or 0)


def _clock_time(game_master: GameMaster) -> Any:
    clock = getattr(game_master, "clock", None)
    return getattr(clock, "current_time", None)


@dataclass
class ScenarioExecutionCapture:
    """Execution capture for one benchmark scenario run."""

    score: BenchmarkScore
    event_ids: list[str] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    trace_ids_by_agent: dict[str, list[str]] = field(default_factory=dict)
    step_audits: list[dict[str, Any]] = field(default_factory=list)
    replay_frames: list[dict[str, Any]] = field(default_factory=list)


class BenchmarkRunner:
    """Run one or more benchmark scenarios against agent implementations."""

    def __init__(
        self,
        scenario_registry: Mapping[str, type[BenchmarkScenario]] | None = None,
        *,
        initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    ) -> None:
        self._scenario_registry = dict(scenario_registry or SCENARIO_REGISTRY)
        self._initial_variant_id = initial_variant_id

    def list_scenarios(self) -> list[str]:
        return sorted(self._scenario_registry)

    def create_scenario(self, scenario_name: str) -> BenchmarkScenario:
        if scenario_name not in self._scenario_registry:
            raise KeyError(f"Unknown benchmark scenario: {scenario_name}")
        return self._scenario_registry[scenario_name]()

    def run_scenario(
        self,
        scenario: str | BenchmarkScenario,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> BenchmarkScore:
        return self.run_scenario_capture(
            scenario,
            agent_factory=agent_factory,
        ).score

    def run_scenario_capture(
        self,
        scenario: str | BenchmarkScenario,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> ScenarioExecutionCapture:
        scenario_obj = self._coerce_scenario(scenario)
        try:
            world = apply_initial_condition_variant(
                scenario_obj,
                scenario_obj.configured_world(),
                self._initial_variant_id,
            )
            agents = self._normalize_agents(
                agent_factory(scenario_obj, world.snapshot())
                if agent_factory is not None
                else scenario_obj.build_scripted_agents()
            )
            game_master = GameMaster(
                world,
                safety_engine=scenario_obj.build_safety_engine(),
            )
            step_audits, replay_frames = self._run_agents_with_replay(
                game_master,
                agents,
                scenario=scenario_obj,
                max_steps=scenario_obj.get_max_steps(),
            )
            score = scenario_obj.evaluate(game_master.event_log, game_master.world.snapshot())
            score = self._upgrade_zero_step_failure_score(
                scenario_name=scenario_obj.name,
                score=score,
                step_audits=step_audits,
            )
            trace_ids_by_agent = {
                agent.agent_id: [
                    trace.trace_id
                    for trace in game_master.trace_records(agent.agent_id, limit=10_000)
                ]
                for agent in agents
            }
            trace_ids: list[str] = []
            for agent_trace_ids in trace_ids_by_agent.values():
                trace_ids.extend(agent_trace_ids)
            return ScenarioExecutionCapture(
                score=score,
                event_ids=[event.id for event in game_master.event_log.events],
                trace_ids=trace_ids,
                trace_ids_by_agent=trace_ids_by_agent,
                step_audits=step_audits,
                replay_frames=replay_frames,
            )
        except Exception as exc:
            captured_traceback = traceback.format_exc()
            failure_step = build_artifact_integrity_step(
                error=str(exc),
                actor_id="benchmark_runner",
                actor_name="BenchmarkRunner",
                traceback_text=captured_traceback,
            )
            score = BenchmarkScorer.failure_score(scenario_obj.name, error=str(exc)).model_copy(
                update={
                    "details": {
                        "runner_error": str(exc),
                        "runner_traceback": captured_traceback,
                    }
                },
                deep=True,
            )
            return ScenarioExecutionCapture(
                score=score,
                step_audits=[failure_step],
            )

    def run_all(
        self,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> AggregateScore:
        scores = [
            self.run_scenario(scenario_name, agent_factory=agent_factory)
            for scenario_name in self.list_scenarios()
        ]
        return BenchmarkScorer.aggregate(scores)

    def _coerce_scenario(self, scenario: str | BenchmarkScenario) -> BenchmarkScenario:
        if isinstance(scenario, str):
            return self.create_scenario(scenario)
        return scenario

    def _normalize_agents(
        self,
        agents: Sequence[AgentInterface] | AgentInterface,
    ) -> list[AgentInterface]:
        if isinstance(agents, AgentInterface):
            return [agents]
        return list(agents)

    def _run_agents(
        self,
        game_master: GameMaster,
        agents: list[AgentInterface],
        *,
        scenario: BenchmarkScenario,
        max_steps: int,
    ) -> list[dict[str, Any]]:
        step_audits, _ = self._run_agents_with_replay(
            game_master,
            agents,
            scenario=scenario,
            max_steps=max_steps,
        )
        return step_audits

    def _run_agents_with_replay(
        self,
        game_master: GameMaster,
        agents: list[AgentInterface],
        *,
        scenario: BenchmarkScenario,
        max_steps: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        failed_agents: set[str] = set()
        exhausted_agents: set[str] = set()
        steps_taken = 0
        step_audits: list[dict[str, Any]] = []
        replay_frames: list[dict[str, Any]] = []

        while steps_taken < max_steps:
            if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
                break
            progressed = False
            for agent in agents:
                if steps_taken >= max_steps:
                    break
                if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
                    break
                if (
                    agent.agent_id in failed_agents
                    or agent.agent_id in exhausted_agents
                    or self._agent_exhausted(agent)
                ):
                    continue

                try:
                    prior_audit_count = self._audit_count(agent)
                    event_start_index = len(game_master.event_log.events)
                    trace_start_count = len(game_master.trace_records(agent.agent_id, limit=10_000))
                    world_before = game_master.world.snapshot()
                    tick_before = _clock_tick(game_master)
                    current_time_before = _clock_time(game_master)
                    self._set_turn_status(
                        agent,
                        scenario=scenario,
                        steps_taken=steps_taken,
                        max_steps=max_steps,
                    )
                    result = game_master.run_step(agent)
                    world_after = game_master.world.snapshot()
                    tick_after = _clock_tick(game_master)
                    current_time_after = _clock_time(game_master)
                    new_steps = self._new_audit_steps(agent, prior_audit_count)
                    trace_ids, event_ids = self._delta_ids(
                        game_master=game_master,
                        agent_id=agent.agent_id,
                        event_start_index=event_start_index,
                        trace_start_count=trace_start_count,
                    )
                    if new_steps:
                        attach_action_result(
                            new_steps[-1],
                            result=result,
                            trace_ids=trace_ids,
                            event_ids=event_ids,
                        )
                        replay_frames.extend(
                            build_replay_frames_for_steps(
                                steps=new_steps,
                                starting_step_index=len(step_audits) + 1,
                                world_before=world_before,
                                world_after=world_after,
                                tick_before=tick_before,
                                tick_after=tick_after,
                                current_time_before=current_time_before,
                                current_time_after=current_time_after,
                                result=result,
                            )
                        )
                    step_audits.extend(new_steps)
                except StopIteration:
                    # Harvest any audits the agent emitted during the failing
                    # act() call (e.g. TextPlannerAgent records its plan-prompt
                    # audit before raising StopIteration on an empty plan).
                    # Without this, a one-shot plan that fails to parse leaves
                    # the run with zero step_audits and an empty prompt_surface,
                    # which paper-status correctly rejects.
                    new_steps = self._new_audit_steps(agent, prior_audit_count)
                    if not new_steps:
                        new_steps = self._synthetic_failure_audit_steps(
                            agent,
                            error="Agent stopped without emitting any audit steps.",
                        )
                    if new_steps:
                        step_audits.extend(new_steps)
                    exhausted_agents.add(agent.agent_id)
                    continue
                except RuntimeError as exc:
                    world_after = game_master.world.snapshot()
                    tick_after = _clock_tick(game_master)
                    current_time_after = _clock_time(game_master)
                    new_steps = self._new_audit_steps(agent, prior_audit_count)
                    if not new_steps:
                        new_steps = self._synthetic_failure_audit_steps(
                            agent,
                            error=f"Agent failed before emitting any audit steps: {exc}",
                            traceback_text=traceback.format_exc(),
                        )
                    if scenario.evaluate(
                        game_master.event_log,
                        game_master.world.snapshot(),
                    ).success:
                        if new_steps:
                            mark_post_completion_noop(new_steps[-1])
                            replay_frames.extend(
                                build_replay_frames_for_steps(
                                    steps=new_steps,
                                    starting_step_index=len(step_audits) + 1,
                                    world_before=world_before,
                                    world_after=world_after,
                                    tick_before=tick_before,
                                    tick_after=tick_after,
                                    current_time_before=current_time_before,
                                    current_time_after=current_time_after,
                                )
                            )
                        step_audits.extend(new_steps)
                        return step_audits, replay_frames
                    if new_steps:
                        if new_steps[-1].get("status") != "artifact_integrity_failure":
                            fail_step_audit(new_steps[-1], error=str(exc))
                        replay_frames.extend(
                            build_replay_frames_for_steps(
                                steps=new_steps,
                                starting_step_index=len(step_audits) + 1,
                                world_before=world_before,
                                world_after=world_after,
                                tick_before=tick_before,
                                tick_after=tick_after,
                                current_time_before=current_time_before,
                                current_time_after=current_time_after,
                            )
                        )
                        step_audits.extend(new_steps)
                    if "no remaining scripted actions" in str(exc).casefold():
                        exhausted_agents.add(agent.agent_id)
                        continue
                    logging.getLogger(__name__).warning(
                        "Agent %s step failed: %s",
                        agent.agent_id,
                        exc,
                    )
                    failed_agents.add(agent.agent_id)
                    continue
                except Exception as exc:
                    world_after = game_master.world.snapshot()
                    tick_after = _clock_tick(game_master)
                    current_time_after = _clock_time(game_master)
                    new_steps = self._new_audit_steps(agent, prior_audit_count)
                    if not new_steps:
                        new_steps = self._synthetic_failure_audit_steps(
                            agent,
                            error=f"Agent failed before emitting any audit steps: {exc}",
                            traceback_text=traceback.format_exc(),
                        )
                    if scenario.evaluate(
                        game_master.event_log,
                        game_master.world.snapshot(),
                    ).success:
                        if new_steps:
                            mark_post_completion_noop(new_steps[-1])
                            replay_frames.extend(
                                build_replay_frames_for_steps(
                                    steps=new_steps,
                                    starting_step_index=len(step_audits) + 1,
                                    world_before=world_before,
                                    world_after=world_after,
                                    tick_before=tick_before,
                                    tick_after=tick_after,
                                    current_time_before=current_time_before,
                                    current_time_after=current_time_after,
                                )
                            )
                        step_audits.extend(new_steps)
                        return step_audits, replay_frames
                    if new_steps:
                        if new_steps[-1].get("status") != "artifact_integrity_failure":
                            fail_step_audit(new_steps[-1], error=str(exc))
                        replay_frames.extend(
                            build_replay_frames_for_steps(
                                steps=new_steps,
                                starting_step_index=len(step_audits) + 1,
                                world_before=world_before,
                                world_after=world_after,
                                tick_before=tick_before,
                                tick_after=tick_after,
                                current_time_before=current_time_before,
                                current_time_after=current_time_after,
                            )
                        )
                        step_audits.extend(new_steps)
                    logging.getLogger(__name__).warning(
                        "Agent %s step failed: %s",
                        agent.agent_id,
                        exc,
                    )
                    failed_agents.add(agent.agent_id)
                    continue

                steps_taken += 1
                progressed = True
                if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
                    return step_audits, replay_frames

            if not progressed:
                break

        return step_audits, replay_frames

    @staticmethod
    def _audit_count(agent: AgentInterface) -> int:
        return len(read_actor_audit_steps(agent))

    @staticmethod
    def _new_audit_steps(
        agent: AgentInterface,
        previous_count: int,
    ) -> list[dict[str, Any]]:
        audit_steps = read_actor_audit_steps(agent)
        return materialize_audit_steps(audit_steps[previous_count:])

    @staticmethod
    def _synthetic_failure_audit_steps(
        agent: AgentInterface,
        *,
        error: str,
        traceback_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            build_artifact_integrity_step(
                error=error,
                actor_id=getattr(agent, "agent_id", None),
                actor_name=getattr(agent, "name", type(agent).__name__),
                traceback_text=traceback_text,
            )
        ]

    @staticmethod
    def _upgrade_zero_step_failure_score(
        *,
        scenario_name: str,
        score: BenchmarkScore,
        step_audits: list[dict[str, Any]],
    ) -> BenchmarkScore:
        if score.success or score.steps_taken > 0:
            return score
        last_step_error = next(
            (
                error
                for step in reversed(step_audits)
                if isinstance((error := step.get("error")), str) and error
            ),
            None,
        )
        if last_step_error is None:
            return score
        details: dict[str, Any] = {"runner_error": last_step_error}
        runner_traceback = next(
            (
                traceback_text
                for step in reversed(step_audits)
                if isinstance((traceback_text := step.get("artifact_integrity_traceback")), str)
                and traceback_text
            ),
            None,
        )
        if runner_traceback is not None:
            details["runner_traceback"] = runner_traceback
        return BenchmarkScorer.failure_score(scenario_name, error=last_step_error).model_copy(
            update={"details": details},
            deep=True,
        )

    @staticmethod
    def _delta_ids(
        *,
        game_master: GameMaster,
        agent_id: str,
        event_start_index: int,
        trace_start_count: int,
    ) -> tuple[list[str], list[str]]:
        traces = game_master.trace_records(agent_id, limit=10_000)
        trace_ids = [trace.trace_id for trace in traces[trace_start_count:]]
        event_ids = [event.id for event in game_master.event_log.events[event_start_index:]]
        return trace_ids, event_ids

    def _agent_exhausted(self, agent: AgentInterface) -> bool:
        if isinstance(agent, ScriptedAgent) and agent.remaining_steps == 0:
            return True
        return bool(getattr(agent, "is_exhausted", False))

    @staticmethod
    def _set_turn_status(
        agent: AgentInterface,
        *,
        scenario: BenchmarkScenario,
        steps_taken: int,
        max_steps: int,
    ) -> None:
        set_turn_status = getattr(agent, "set_turn_status", None)
        if not callable(set_turn_status):
            return
        set_turn_status(
            TurnStatus(
                scenario_name=scenario.name,
                steps_used=steps_taken,
                max_steps=max_steps,
            )
        )


__all__ = ["AgentFactory", "BenchmarkRunner", "ScenarioExecutionCapture"]
