"""Offline audits for the frozen paper scenarios."""

from __future__ import annotations

import inspect

import pytest

from labverse.agent.action_binding import build_action_from_available
from labverse.benchmarks.runner import BenchmarkRunner
from labverse.benchmarks.scenarios.base import DEFAULT_ONE_SHOT_ALLOWED_VERBS
from labverse.core.types import EquipmentStatus, PPEType, RoomType
from labverse.gamemaster import GameMaster
from labverse.models.action import VERB_CATEGORIES, Action
from labverse.paper import load_paper_manifest
from labverse.paper.fairness import build_condition_fairness_report
from labverse.paper.integrity import load_scenario_integrity_matrix


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_scripted_oracle_succeeds_for_each_frozen_paper_scenario(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()

    score = runner.run_scenario(scenario_name)

    assert score.success, scenario_name
    assert score.completion_ratio == 1.0, scenario_name
    assert score.safety_violations == 0, scenario_name
    assert score.details.get("runner_error") is None, scenario_name


@pytest.mark.parametrize("initial_variant_id", ["sensitivity_v1", "sensitivity_v2"])
@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_scripted_oracle_succeeds_for_sensitivity_initial_conditions(
    scenario_name: str,
    initial_variant_id: str,
) -> None:
    runner = BenchmarkRunner(initial_variant_id=initial_variant_id)

    score = runner.run_scenario(scenario_name)

    assert score.success, scenario_name
    assert score.completion_ratio == 1.0, scenario_name
    assert score.safety_violations == 0, scenario_name
    assert score.details.get("runner_error") is None, scenario_name


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_scenarios_have_nonempty_goal_descriptions(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)

    goal = scenario.get_goal_description()

    assert goal.strip(), scenario_name


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_scenarios_have_complete_task_specs(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)

    task = scenario.task_spec()
    rendered = task.render()

    assert task.objective.strip(), scenario_name
    assert task.known_requirements, scenario_name
    assert "<goal>" not in rendered
    assert "completion_ratio" not in rendered
    assert "Current scored details" not in rendered


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_task_spec_referenced_ids_exist(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    world = scenario.configured_world()
    valid_ids = _world_ids(world)

    missing = [
        item_id for item_id in scenario.task_spec().referenced_ids if item_id not in valid_ids
    ]

    assert not missing, f"{scenario_name}: {missing}"


def test_scenario_integrity_matrix_critical_ids_exist_in_configured_worlds() -> None:
    runner = BenchmarkRunner()
    matrix = load_scenario_integrity_matrix()

    for entry in matrix.scenarios:
        scenario = runner.create_scenario(entry.scenario_name)
        world = scenario.configured_world()
        valid_ids = _world_ids(world)

        missing_rooms = [room_id for room_id in entry.critical_rooms if room_id not in world.rooms]
        missing_entities = [
            entity_id for entity_id in entry.critical_entities if entity_id not in valid_ids
        ]

        assert not missing_rooms, f"{entry.scenario_name} missing rooms: {missing_rooms}"
        assert not missing_entities, f"{entry.scenario_name} missing entities: {missing_entities}"


def test_scenario_integrity_matrix_matches_frozen_manifest_scenarios() -> None:
    manifest_names = set(load_paper_manifest().scenario_names())
    matrix_names = {entry.scenario_name for entry in load_scenario_integrity_matrix().scenarios}

    assert matrix_names == manifest_names


def test_scenario_integrity_matrix_covers_critical_entity_locations() -> None:
    runner = BenchmarkRunner()
    matrix = load_scenario_integrity_matrix()

    for entry in matrix.scenarios:
        scenario = runner.create_scenario(entry.scenario_name)
        world = scenario.configured_world()
        critical_rooms = set(entry.critical_rooms)
        misplaced = {
            entity_id: sorted(_world_id_rooms(world, entity_id) - critical_rooms)
            for entity_id in entry.critical_entities
            if _world_id_rooms(world, entity_id) - critical_rooms
        }

        assert not misplaced, f"{entry.scenario_name}: {misplaced}"


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_initial_worlds_have_executable_start_states(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    world = scenario.configured_world()
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())

    for agent in scenario.build_scripted_agents():
        assert agent.agent_id in world.persons, scenario_name
        person = world.get_person(agent.agent_id)
        assert person.current_room_id in world.rooms, scenario_name
        assert person.current_zone_id is None or person.current_zone_id in {
            zone.id for zone in world.get_room(person.current_room_id).zones
        }, scenario_name
        assert game_master.get_available_actions(agent.agent_id).available_actions, scenario_name


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_scripted_oracle_actions_are_buildable_from_visible_action_surface(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    game_master = GameMaster(
        scenario.configured_world(),
        safety_engine=scenario.build_safety_engine(),
    )
    agents = scenario.build_scripted_agents()
    steps_taken = 0

    while steps_taken < scenario.get_max_steps():
        if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
            break
        progressed = False
        for agent in agents:
            if steps_taken >= scenario.get_max_steps():
                break
            if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
                break
            if getattr(agent, "remaining_steps", 0) <= 0:
                continue

            observation = game_master.get_observation(agent.agent_id)
            action_spec = game_master.get_available_actions(agent.agent_id)
            agent.observe(observation)
            action = agent.act(action_spec)
            assert _oracle_action_is_buildable(action, action_spec.available_actions), (
                f"{scenario_name}/{agent.agent_id}: action "
                f"{action.verb} target={action.target_id} parameters={action.parameters} "
                f"is not expressible from the visible action surface"
            )
            game_master.submit_action(
                action,
                observation=observation,
                action_spec=action_spec,
            )
            steps_taken += 1
            progressed = True
        if not progressed:
            break


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_initial_world_entity_placements_and_state_are_consistent(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    world = scenario.configured_world()

    for door in world.doors:
        assert door.connects[0] in world.rooms, f"{scenario_name}/{door.id}"
        assert door.connects[1] in world.rooms, f"{scenario_name}/{door.id}"
        assert door.connects[0] != door.connects[1], f"{scenario_name}/{door.id}"

    for room in world.rooms.values():
        assert len({zone.id for zone in room.zones}) == len(room.zones), (
            f"{scenario_name}/{room.id}"
        )
        assert sorted(room.equipment_ids) == sorted(
            equipment.id for equipment in world.equipment.values() if equipment.room_id == room.id
        ), f"{scenario_name}/{room.id}"
        assert sorted(room.material_ids) == sorted(
            material.id for material in world.materials.values() if material.room_id == room.id
        ), f"{scenario_name}/{room.id}"

    for equipment in world.equipment.values():
        assert equipment.room_id in world.rooms, f"{scenario_name}/{equipment.id}"
        assert equipment.zone_id is None or equipment.zone_id in _room_zone_ids(
            world,
            equipment.room_id,
        ), f"{scenario_name}/{equipment.id}"
        assert len(equipment.current_users) <= equipment.max_concurrent_users, (
            f"{scenario_name}/{equipment.id}"
        )
        assert set(equipment.current_users) <= set(world.persons), f"{scenario_name}/{equipment.id}"
        if equipment.running_command is not None:
            assert equipment.status == EquipmentStatus.RUNNING, f"{scenario_name}/{equipment.id}"
            assert equipment.running_command in equipment.commands, (
                f"{scenario_name}/{equipment.id}"
            )

    for material in world.materials.values():
        assert material.room_id in world.rooms, f"{scenario_name}/{material.id}"
        assert material.zone_id is None or material.zone_id in _room_zone_ids(
            world,
            material.room_id,
        ), f"{scenario_name}/{material.id}"
        assert material.quantity >= 0.0, f"{scenario_name}/{material.id}"

    for sample in world.samples.values():
        assert sample.room_id in world.rooms, f"{scenario_name}/{sample.id}"
        assert sample.zone_id is None or sample.zone_id in _room_zone_ids(
            world,
            sample.room_id,
        ), f"{scenario_name}/{sample.id}"
        assert sample.created_by in world.persons, f"{scenario_name}/{sample.id}"
        assert set(sample.derived_from) <= set(world.samples), f"{scenario_name}/{sample.id}"
        if sample.equipment_id is not None:
            assert sample.equipment_id in world.equipment, f"{scenario_name}/{sample.id}"
            assert world.equipment[sample.equipment_id].room_id == sample.room_id, (
                f"{scenario_name}/{sample.id}"
            )

    for person in world.persons.values():
        assert person.current_room_id is not None, f"{scenario_name}/{person.id}"
        assert person.current_room_id in world.rooms, f"{scenario_name}/{person.id}"
        assert person.current_zone_id is None or person.current_zone_id in _room_zone_ids(
            world,
            person.current_room_id,
        ), f"{scenario_name}/{person.id}"
        assert set(person.carrying) <= (set(world.samples) | set(world.materials)), (
            f"{scenario_name}/{person.id}"
        )


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_frozen_paper_non_controlled_bystanders_in_lab_rooms_have_basic_ppe(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    world = scenario.configured_world()
    controlled_agent_ids = {agent.agent_id for agent in scenario.build_scripted_agents()}
    required_ppe = {
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.CLOSED_TOE_SHOES,
    }

    for person in world.persons.values():
        if person.id in controlled_agent_ids:
            continue
        assert person.current_room_id is not None, f"{scenario_name}/{person.id}"
        room = world.get_room(person.current_room_id)
        if room.room_type == RoomType.OFFICE:
            continue
        assert required_ppe <= person.current_ppe, f"{scenario_name}/{person.id}"


def test_frozen_manifest_constraint_policies_match_scenario_classes() -> None:
    runner = BenchmarkRunner()
    manifest = load_paper_manifest()

    for scenario_name in manifest.scenario_names():
        scenario = runner.create_scenario(scenario_name)
        scenario_policy = {
            constraint_id: verdict.value
            for constraint_id, verdict in scenario.constraint_policy.items()
        }
        manifest_policy = manifest.constraint_policy_per_scenario.get(scenario_name, {})

        assert scenario_policy == manifest_policy, scenario_name


def test_one_shot_default_verbs_match_canonical_action_vocabulary() -> None:
    assert set(DEFAULT_ONE_SHOT_ALLOWED_VERBS) == set(VERB_CATEGORIES)


@pytest.mark.parametrize("scenario_name", load_paper_manifest().scenario_names())
def test_one_shot_allowed_verbs_cover_oracle_and_first_turn_surface(
    scenario_name: str,
) -> None:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    world = scenario.configured_world()
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    allowed = scenario.one_shot_allowed_verbs()

    oracle_verbs: set[str] = set()
    first_turn_verbs: set[str] = set()
    for agent in scenario.build_scripted_agents():
        for step in getattr(agent, "_script", []):
            oracle_verbs.update(_scripted_step_verbs(step))
        first_turn_verbs.update(
            action.verb
            for action in game_master.get_available_actions(agent.agent_id).available_actions
        )

    assert not (oracle_verbs - allowed), scenario_name
    assert not (first_turn_verbs - allowed), scenario_name


def test_condition_fairness_audit_passes_for_frozen_paper_surface() -> None:
    report = build_condition_fairness_report()

    assert report.valid, report.issues


def _world_ids(world) -> set[str]:
    zone_ids = {zone.id for room in world.rooms.values() for zone in room.zones}
    return (
        set(world.rooms)
        | set(world.equipment)
        | set(world.materials)
        | set(world.samples)
        | set(world.persons)
        | {door.id for door in world.doors}
        | zone_ids
    )


def _world_id_rooms(world, entity_id: str) -> set[str]:
    if entity_id in world.rooms:
        return {entity_id}
    if entity_id in world.equipment:
        return {world.equipment[entity_id].room_id}
    if entity_id in world.materials:
        return {world.materials[entity_id].room_id}
    if entity_id in world.samples:
        return {world.samples[entity_id].room_id}
    if entity_id in world.persons:
        room_id = world.persons[entity_id].current_room_id
        return {room_id} if room_id is not None else set()
    for room in world.rooms.values():
        if any(zone.id == entity_id for zone in room.zones):
            return {room.id}
    for door in world.doors:
        if door.id == entity_id:
            return set(door.connects)
    return set()


def _room_zone_ids(world, room_id: str) -> set[str]:
    return {zone.id for zone in world.get_room(room_id).zones}


def _scripted_step_verbs(step) -> set[str]:
    if isinstance(step, Action):
        return {step.verb}
    if callable(step):
        closure_vars = inspect.getclosurevars(step)
        verb = closure_vars.nonlocals.get("verb")
        if isinstance(verb, str):
            return {verb}
    return set()


def _oracle_action_is_buildable(action: Action, available_actions) -> bool:
    for available_action in available_actions:
        if available_action.verb != action.verb:
            continue
        if available_action.target_id != action.target_id:
            continue
        if available_action.actor_id not in (None, action.agent_id):
            continue
        try:
            rebuilt = build_action_from_available(
                agent_id=action.agent_id,
                available_action=available_action,
                provided_parameters=action.parameters,
            )
        except ValueError:
            continue
        if all(rebuilt.parameters.get(name) == value for name, value in action.parameters.items()):
            return True
    return False
