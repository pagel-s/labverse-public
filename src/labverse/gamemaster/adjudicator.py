"""GameMaster implementation for the core action adjudication pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from labverse.agent.action_binding import (
    build_action_from_available,
    resolve_action_choice,
    resolve_available_action,
)
from labverse.agent.action_space import ActionSpaceCalculator
from labverse.agent.interface import AgentInterface, Observation
from labverse.agent.observation import ObservationBuilder
from labverse.agent.trace import AgentMemorySnapshot, AgentTraceRecord, AgentTraceStore
from labverse.core.clock import SimulationClock
from labverse.core.events import Event, EventBus, EventLog, EventType
from labverse.equipment.backends.simulated import SimulatedBackend
from labverse.gamemaster.action_resolver import ActionResolver, ResolverContext
from labverse.gamemaster.time_manager import TimedEvent, TimeManager
from labverse.gamemaster.validators import ValidationPipeline
from labverse.models.action import (
    Action,
    ActionResult,
    ActionResultStatus,
    ActionSpec,
    ChangedEntity,
)
from labverse.models.safety import SafetyCheckResult, SafetyVerdict
from labverse.safety import SafetyEngine
from labverse.world.spatial import SpatialIndex
from labverse.world.state import WorldState


class GameMaster:
    """Central orchestrator for the minimal semantic simulation loop."""

    def __init__(
        self,
        world_state: WorldState,
        *,
        clock: SimulationClock | None = None,
        event_bus: EventBus | None = None,
        safety_engine: SafetyEngine | None = None,
        equipment_backend: SimulatedBackend | None = None,
    ) -> None:
        self._world = world_state
        self._clock = clock or SimulationClock()
        self._events = event_bus or EventBus()
        self._event_log = EventLog()
        self._events.subscribe("*", self._event_log.append)
        self._safety = safety_engine or SafetyEngine()
        self._equipment_backend = equipment_backend or SimulatedBackend(world_state)
        self._spatial = SpatialIndex(world_state)
        self._trace_store = AgentTraceStore()
        self._action_resolver = ActionResolver()
        self._validators = ValidationPipeline(self._action_resolver.handlers)
        self._time_manager = TimeManager(self._clock)
        self._action_space = ActionSpaceCalculator(
            self._world,
            self._clock,
            self._spatial,
        )
        self._observation_builder = ObservationBuilder(
            self._world,
            self._clock,
            self._event_log,
            working_memory_provider=self._trace_store.working_memory,
            last_action_provider=self._last_action_id,
        )

    @property
    def world(self) -> WorldState:
        return self._world

    @property
    def clock(self) -> SimulationClock:
        return self._clock

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    @property
    def pending_timed_events(self) -> int:
        return len(self._time_manager)

    def trace_records(self, agent_id: str, *, limit: int = 20) -> list[AgentTraceRecord]:
        return self._trace_store.recent(agent_id, limit=limit)

    def memory_snapshot(self, agent_id: str) -> AgentMemorySnapshot:
        return self._trace_store.snapshot(agent_id)

    def set_working_memory(self, agent_id: str, value: str) -> AgentMemorySnapshot:
        self._world.get_person(agent_id)
        return self._trace_store.set_working_memory(agent_id, value)

    def submit_action_by_id(
        self,
        agent_id: str,
        action_id: str,
        *,
        parameters: dict[str, object] | None = None,
    ) -> ActionResult:
        observation = self.get_observation(agent_id)
        action_spec = self.get_available_actions(agent_id)
        available_action = resolve_available_action(action_id, action_spec.available_actions)
        action = build_action_from_available(
            agent_id=agent_id,
            available_action=available_action,
            provided_parameters=parameters or {},
        )
        return self.submit_action(action, observation=observation, action_spec=action_spec)

    def submit_action_by_candidate_ref(
        self,
        agent_id: str,
        candidate_ref: str,
        *,
        verb: str,
        target_id: str | None = None,
        parameters: dict[str, object] | None = None,
    ) -> ActionResult:
        observation = self.get_observation(agent_id)
        action_spec = self.get_available_actions(agent_id)
        resolution = resolve_action_choice(
            {
                "candidate_ref": candidate_ref,
                "verb": verb,
                **({"target_id": target_id} if target_id is not None else {}),
            },
            action_spec.available_actions,
        )
        action = build_action_from_available(
            agent_id=agent_id,
            available_action=resolution.available_action,
            provided_parameters=parameters or {},
        )
        return self.submit_action(action, observation=observation, action_spec=action_spec)

    def submit_action(
        self,
        action: Action,
        *,
        observation: Observation | None = None,
        action_spec: ActionSpec | None = None,
    ) -> ActionResult:
        action.timestamp = self._clock.current_time
        event_start_index = len(self._event_log.events)
        self._publish(
            Event(
                event_type=EventType.ACTION_SUBMITTED,
                tick=self._clock.tick,
                timestamp=self._clock.current_time,
                source_id=action.agent_id,
                target_id=action.target_id or "",
                description=f"{action.agent_id} submitted {action.verb}",
                data={"action_id": action.id, "parameters": dict(action.parameters)},
            )
        )

        validation_error = self._validators.validate_action(action, self._world)
        if validation_error is not None:
            result = self._blocked_result(action, validation_error)
            return self._finalize_action(
                action,
                result,
                observation=observation,
                action_spec=action_spec,
                event_start_index=event_start_index,
            )

        warnings, safety_violations = self._evaluate_safety(action)
        if safety_violations:
            result = self._blocked_result(
                action,
                "; ".join(safety_violations),
                warnings=warnings,
                safety_violations=safety_violations,
            )
            return self._finalize_action(
                action,
                result,
                observation=observation,
                action_spec=action_spec,
                event_start_index=event_start_index,
            )

        context = ResolverContext(
            world=self._world,
            clock=self._clock,
            spatial=self._spatial,
            equipment_backend=self._equipment_backend,
        )

        precondition_failures = self._action_resolver.preconditions(action, context)
        if precondition_failures:
            result = self._blocked_result(
                action,
                "; ".join(precondition_failures),
                warnings=warnings,
            )
            return self._finalize_action(
                action,
                result,
                observation=observation,
                action_spec=action_spec,
                event_start_index=event_start_index,
            )

        try:
            resolved = self._action_resolver.resolve(action, context)
        except Exception as exc:
            result = self._failed_result(action, str(exc), warnings=warnings)
            return self._finalize_action(
                action,
                result,
                observation=observation,
                action_spec=action_spec,
                event_start_index=event_start_index,
            )

        for cancellation in resolved.cancellations:
            self._time_manager.cancel(
                event_type=cancellation.event_type,
                target_id=cancellation.target_id,
            )
        for scheduled_effect in resolved.scheduled_effects:
            self._time_manager.schedule_in_seconds(
                event_type=scheduled_effect.event_type,
                target_id=scheduled_effect.target_id,
                delay_seconds=scheduled_effect.delay_seconds,
                callback_data=scheduled_effect.callback_data,
            )
        for event in resolved.emitted_events:
            self._publish(event)

        if resolved.time_elapsed_seconds > 0:
            ticks_advanced = self._clock.advance_by_seconds(resolved.time_elapsed_seconds)
            self._publish(
                Event(
                    event_type=EventType.CLOCK_ADVANCED,
                    tick=self._clock.tick,
                    timestamp=self._clock.current_time,
                    description=f"Clock advanced by {ticks_advanced} ticks",
                    data={
                        "ticks_advanced": ticks_advanced,
                        "seconds_elapsed": resolved.time_elapsed_seconds,
                    },
                )
            )

        self._process_due_events()

        result = ActionResult(
            action_id=action.id,
            success=True,
            status=ActionResultStatus.SUCCEEDED,
            outcome_description=resolved.outcome_description,
            state_changes=resolved.state_changes,
            changed_entities=self._changed_entities(resolved.state_changes),
            time_elapsed_seconds=resolved.time_elapsed_seconds,
            warnings=warnings + resolved.warnings,
        )
        self._publish(
            Event(
                event_type=EventType.ACTION_RESOLVED,
                tick=self._clock.tick,
                timestamp=self._clock.current_time,
                source_id=action.agent_id,
                target_id=action.target_id or "",
                description=f"{action.verb} resolved successfully",
                data={"action_id": action.id, "success": True},
            )
        )
        return self._finalize_action(
            action,
            result,
            observation=observation,
            action_spec=action_spec,
            event_start_index=event_start_index,
        )

    def tick(self, ticks: int = 1) -> list[Event]:
        if ticks < 0:
            raise ValueError(f"Cannot tick backwards: {ticks}")
        if ticks > 0:
            self._clock.advance(ticks)
            self._publish(
                Event(
                    event_type=EventType.CLOCK_ADVANCED,
                    tick=self._clock.tick,
                    timestamp=self._clock.current_time,
                    description=f"Clock advanced by {ticks} ticks",
                    data={"ticks_advanced": ticks},
                )
            )
        return self._process_due_events()

    def get_available_actions(self, agent_id: str) -> ActionSpec:
        return self._action_space.calculate(agent_id)

    def get_observation(self, agent_id: str) -> Observation:
        return self._observation_builder.build(agent_id)

    def run_step(self, agent: AgentInterface) -> ActionResult:
        observation = self.get_observation(agent.agent_id)
        agent.observe(observation)
        action_spec = self.get_available_actions(agent.agent_id)
        action = agent.act(action_spec)
        if action.agent_id != agent.agent_id:
            action = action.model_copy(update={"agent_id": agent.agent_id}, deep=True)
        result = self.submit_action(action, observation=observation, action_spec=action_spec)
        notify_result = getattr(agent, "notify_result", None)
        if callable(notify_result):
            notify_result(
                result.success,
                result.outcome_description or result.blocked_reason or result.failure_reason or "",
            )
        return result

    def _blocked_result(
        self,
        action: Action,
        blocked_reason: str,
        *,
        warnings: list[str] | None = None,
        safety_violations: list[str] | None = None,
    ) -> ActionResult:
        result = ActionResult(
            action_id=action.id,
            success=False,
            status=ActionResultStatus.BLOCKED,
            outcome_description=blocked_reason,
            blocked_reason=blocked_reason,
            warnings=warnings or [],
            safety_violations=safety_violations or [],
        )
        self._publish(
            Event(
                event_type=EventType.ACTION_BLOCKED,
                tick=self._clock.tick,
                timestamp=self._clock.current_time,
                source_id=action.agent_id,
                target_id=action.target_id or "",
                description=f"{action.verb} was blocked",
                data={"action_id": action.id, "reason": blocked_reason},
            )
        )
        self._publish(
            Event(
                event_type=EventType.ACTION_RESOLVED,
                tick=self._clock.tick,
                timestamp=self._clock.current_time,
                source_id=action.agent_id,
                target_id=action.target_id or "",
                description=f"{action.verb} was blocked",
                data={"action_id": action.id, "success": False, "reason": blocked_reason},
            )
        )
        return result

    def _failed_result(
        self,
        action: Action,
        failure_reason: str,
        *,
        warnings: list[str] | None = None,
    ) -> ActionResult:
        result = ActionResult(
            action_id=action.id,
            success=False,
            status=ActionResultStatus.FAILED,
            outcome_description=failure_reason,
            failure_reason=failure_reason,
            warnings=warnings or [],
        )
        self._publish(
            Event(
                event_type=EventType.ACTION_RESOLVED,
                tick=self._clock.tick,
                timestamp=self._clock.current_time,
                source_id=action.agent_id,
                target_id=action.target_id or "",
                description=f"{action.verb} failed",
                data={"action_id": action.id, "success": False, "reason": failure_reason},
            )
        )
        return result

    def _evaluate_safety(
        self,
        action: Action,
        *,
        publish_events: bool = True,
    ) -> tuple[list[str], list[str]]:
        raw_results = self._safety.evaluate(action, self._world)
        results: Iterable[SafetyCheckResult]
        if isinstance(raw_results, SafetyCheckResult):
            results = [raw_results]
        else:
            results = list(raw_results)

        warnings: list[str] = []
        blocked_messages: list[str] = []
        for result in results:
            if result.verdict == SafetyVerdict.WARN:
                warnings.append(result.message)
                if publish_events:
                    self._publish(
                        Event(
                            event_type=EventType.SAFETY_WARNING,
                            tick=self._clock.tick,
                            timestamp=self._clock.current_time,
                            source_id=action.agent_id,
                            target_id=action.target_id or "",
                            description=result.message,
                        )
                    )
            elif result.verdict == SafetyVerdict.BLOCK:
                blocked_messages.append(result.message)
                if publish_events:
                    self._publish(
                        Event(
                            event_type=EventType.SAFETY_VIOLATION,
                            tick=self._clock.tick,
                            timestamp=self._clock.current_time,
                            source_id=action.agent_id,
                            target_id=action.target_id or "",
                            description=result.message,
                        )
                    )
        return warnings, blocked_messages

    def _process_due_events(self) -> list[Event]:
        published: list[Event] = []
        for timed_event in self._time_manager.pop_due_events():
            published.extend(self._handle_timed_event(timed_event))
        return published

    def _handle_timed_event(self, timed_event: TimedEvent) -> list[Event]:
        emitted: list[Event] = []
        if timed_event.event_type == "equipment_complete":
            equipment = self._world.get_equipment(timed_event.target_id)
            command_name = timed_event.callback_data.get("command_name", equipment.running_command)
            self._equipment_backend.complete_command(equipment.id)
            emitted.append(
                Event(
                    event_type=EventType.TIMED_EVENT,
                    tick=self._clock.tick,
                    timestamp=self._clock.current_time,
                    source_id=equipment.id,
                    target_id=equipment.id,
                    description=f"{equipment.name} completed {command_name}",
                    data=timed_event.callback_data,
                )
            )
            emitted.append(
                Event(
                    event_type=EventType.EQUIPMENT_STATE,
                    tick=self._clock.tick,
                    timestamp=self._clock.current_time,
                    source_id=equipment.id,
                    target_id=equipment.id,
                    description=f"{equipment.name} entered {equipment.status.value}",
                    data={"status": equipment.status.value, "command_name": command_name},
                )
            )
        else:
            emitted.append(
                Event(
                    event_type=EventType.TIMED_EVENT,
                    tick=self._clock.tick,
                    timestamp=self._clock.current_time,
                    source_id=timed_event.target_id,
                    target_id=timed_event.target_id,
                    description=f"Timed event fired: {timed_event.event_type}",
                    data=timed_event.callback_data,
                )
            )

        for event in emitted:
            self._publish(event)
        return emitted

    def _finalize_action(
        self,
        action: Action,
        result: ActionResult,
        *,
        observation: Observation | None,
        action_spec: ActionSpec | None,
        event_start_index: int,
    ) -> ActionResult:
        emitted_events = self._event_log.events[event_start_index:]
        post_observation: Observation | None = None
        try:
            post_observation = self.get_observation(action.agent_id)
        except Exception:
            post_observation = None

        finalized_result = result.model_copy(
            update={
                "emitted_event_ids": [event.id for event in emitted_events],
                "post_observation_id": (
                    post_observation.observation_id if post_observation is not None else None
                ),
            },
            deep=True,
        )

        if observation is not None and action_spec is not None:
            self._trace_store.append(
                AgentTraceRecord(
                    agent_id=action.agent_id,
                    tick=self._clock.tick,
                    timestamp=self._clock.current_time,
                    observation=observation.model_copy(deep=True),
                    action_spec=action_spec.model_copy(deep=True),
                    action=action.model_copy(deep=True),
                    result=finalized_result.model_copy(deep=True),
                    event_ids=[event.id for event in emitted_events],
                    post_observation=post_observation.model_copy(deep=True)
                    if post_observation is not None
                    else None,
                )
            )

        return finalized_result

    def _last_action_id(self, agent_id: str) -> str | None:
        trace = self._trace_store.last_trace(agent_id)
        return trace.action.id if trace is not None else None

    def _changed_entities(
        self,
        state_changes: list[dict[str, object]],
    ) -> list[ChangedEntity]:
        changed: dict[tuple[str, str], set[str]] = {}
        for change in state_changes:
            entity_type = change.get("entity_type")
            entity_id = change.get("entity_id")
            field = change.get("field")
            if not isinstance(entity_type, str) or not isinstance(entity_id, str):
                continue
            changed.setdefault((entity_type, entity_id), set())
            if isinstance(field, str):
                changed[(entity_type, entity_id)].add(field)

        summaries: list[ChangedEntity] = []
        for (entity_type, entity_id), fields in changed.items():
            summaries.append(
                ChangedEntity(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    summary=", ".join(sorted(fields)),
                    room_id=self._entity_room_id(entity_type, entity_id),
                    zone_id=self._entity_zone_id(entity_type, entity_id),
                )
            )
        return summaries

    def _entity_room_id(self, entity_type: str, entity_id: str) -> str | None:
        if entity_type == "person" and entity_id in self._world.persons:
            return self._world.persons[entity_id].current_room_id
        if entity_type == "equipment" and entity_id in self._world.equipment:
            return self._world.equipment[entity_id].room_id
        if entity_type == "material" and entity_id in self._world.materials:
            return self._world.materials[entity_id].room_id
        if entity_type == "sample" and entity_id in self._world.samples:
            return self._world.samples[entity_id].room_id
        return None

    def _entity_zone_id(self, entity_type: str, entity_id: str) -> str | None:
        if entity_type == "person" and entity_id in self._world.persons:
            return self._world.persons[entity_id].current_zone_id
        if entity_type == "equipment" and entity_id in self._world.equipment:
            return self._world.equipment[entity_id].zone_id
        if entity_type == "material" and entity_id in self._world.materials:
            return self._world.materials[entity_id].zone_id
        if entity_type == "sample" and entity_id in self._world.samples:
            return self._world.samples[entity_id].zone_id
        return None

    def _publish(self, event: Event) -> None:
        self._events.publish(event)
