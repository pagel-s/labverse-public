"""Unit tests for the safety engine and built-in constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from labverse.core.types import (
    ActionCategory,
    CertificationType,
    HazardPictogram,
    MaterialPhase,
    PersonRole,
    PPEType,
    WasteCategory,
)
from labverse.models.action import Action
from labverse.models.material import GHSData, Material
from labverse.models.person import Person
from labverse.models.safety import SafetyCheckResult, SafetyVerdict
from labverse.safety import (
    BSLAccessConstraint,
    CertificationConstraint,
    ChemicalCompatibilityConstraint,
    EmergencyEquipmentConstraint,
    EquipmentConcurrencyConstraint,
    FumeHoodRequirementConstraint,
    MaterialExpiryConstraint,
    PPERequirementConstraint,
    RoomOccupancyConstraint,
    SafetyEngine,
    SampleContaminationConstraint,
    WasteDisposalConstraint,
)
from labverse.world.state import WorldState


def _action(
    action_id: str,
    agent_id: str,
    category: ActionCategory,
    verb: str,
    *,
    target_id: str | None = None,
    parameters: dict[str, object] | None = None,
    timestamp: datetime | None = None,
) -> Action:
    return Action(
        id=action_id,
        agent_id=agent_id,
        category=category,
        verb=verb,
        target_id=target_id,
        parameters=parameters or {},
        timestamp=timestamp,
    )


def _result_by_id(results: list[SafetyCheckResult]) -> dict[str, SafetyCheckResult]:
    return {result.constraint_id: result for result in results}


def test_safety_engine_has_expected_default_constraints() -> None:
    engine = SafetyEngine()

    assert [constraint.constraint_id for constraint in engine.constraints] == [
        "ppe_requirement",
        "bsl_access",
        "chemical_compatibility",
        "certification",
        "room_occupancy",
        "equipment_concurrency",
        "fume_hood_requirement",
        "waste_disposal",
        "sample_contamination",
        "emergency_equipment",
        "material_expiry",
    ]


def test_ppe_requirement_constraint_blocks_missing_required_ppe(
    research_world: WorldState,
) -> None:
    person = research_world.get_person("grad_kim")
    person.current_ppe = {
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.CLOSED_TOE_SHOES,
    }

    result = PPERequirementConstraint().check(
        _action(
            "ppe_block",
            "grad_kim",
            ActionCategory.HANDLE_MATERIAL,
            "handle_material",
            target_id="ethanol_70",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "nitrile_gloves" in result.message


def test_ppe_requirement_constraint_allows_when_required_ppe_is_worn(
    research_world: WorldState,
) -> None:
    person = research_world.get_person("grad_kim")
    person.current_ppe = {
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.NITRILE_GLOVES,
        PPEType.CLOSED_TOE_SHOES,
    }

    result = PPERequirementConstraint().check(
        _action(
            "ppe_allow",
            "grad_kim",
            ActionCategory.HANDLE_MATERIAL,
            "handle_material",
            target_id="ethanol_70",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.ALLOW


def test_bsl_access_constraint_blocks_uncertified_entry(research_world: WorldState) -> None:
    research_world.add_person(
        Person(
            id="visitor_1",
            name="Visitor",
            role=PersonRole.UNDERGRAD,
            current_room_id="main_wet_lab",
        )
    )

    result = BSLAccessConstraint().check(
        _action(
            "bsl_block",
            "visitor_1",
            ActionCategory.MOVE,
            "move_to",
            target_id="tissue_culture",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "bsl2_training" in result.message


def test_chemical_compatibility_constraint_blocks_incompatible_materials(
    research_world: WorldState,
) -> None:
    result = ChemicalCompatibilityConstraint().check(
        _action(
            "compat_block",
            "tech_johnson",
            ActionCategory.HANDLE_MATERIAL,
            "combine_materials",
            target_id="ethanol_70",
            parameters={"other_material_ids": ["hcl_conc"]},
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "incompatible" in result.message


def test_certification_constraint_blocks_uncertified_equipment_use(
    research_world: WorldState,
) -> None:
    result = CertificationConstraint().check(
        _action(
            "cert_block",
            "grad_kim",
            ActionCategory.USE_EQUIPMENT,
            "start_equipment",
            target_id="autoclave_1",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert CertificationType.AUTOCLAVE_OPERATION.value in result.message


def test_room_occupancy_constraint_blocks_full_room(research_world: WorldState) -> None:
    for index in range(3):
        research_world.add_person(
            Person(
                id=f"occupant_{index}",
                name=f"Occupant {index}",
                role=PersonRole.TECHNICIAN,
                current_room_id="tissue_culture",
            )
        )
    research_world.add_person(
        Person(
            id="visitor_2",
            name="Visitor 2",
            role=PersonRole.UNDERGRAD,
            current_room_id="main_wet_lab",
        )
    )

    result = RoomOccupancyConstraint().check(
        _action(
            "occupancy_block",
            "visitor_2",
            ActionCategory.MOVE,
            "move_to",
            target_id="tissue_culture",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "max occupancy" in result.message


def test_equipment_concurrency_constraint_blocks_busy_instrument(
    research_world: WorldState,
) -> None:
    centrifuge = research_world.get_equipment("centrifuge_1")
    centrifuge.current_users = ["pi_chen"]
    centrifuge.max_concurrent_users = 1

    result = EquipmentConcurrencyConstraint().check(
        _action(
            "concurrency_block",
            "grad_kim",
            ActionCategory.USE_EQUIPMENT,
            "start_equipment",
            target_id="centrifuge_1",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "max concurrent users" in result.message


def test_fume_hood_requirement_constraint_blocks_hazardous_material_outside_hood(
    research_world: WorldState,
) -> None:
    research_world.add_material(
        Material(
            id="acetone_1",
            name="Acetone",
            phase=MaterialPhase.LIQUID,
            room_id="main_wet_lab",
            zone_id="bench_1",
            ghs=GHSData(pictograms=[HazardPictogram.GHS02_FLAMMABLE]),
        )
    )
    research_world.get_person("grad_kim").current_zone_id = "bench_1"

    result = FumeHoodRequirementConstraint().check(
        _action(
            "fume_block",
            "grad_kim",
            ActionCategory.HANDLE_MATERIAL,
            "handle_material",
            target_id="acetone_1",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "fume hood" in result.message


def test_waste_disposal_constraint_blocks_wrong_stream(research_world: WorldState) -> None:
    result = WasteDisposalConstraint().check(
        _action(
            "waste_block",
            "tech_johnson",
            ActionCategory.HANDLE_MATERIAL,
            "dispose_material",
            target_id="ethanol_70",
            parameters={"waste_category": WasteCategory.GENERAL.value},
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.BLOCK
    assert "chemical" in result.message


def test_sample_contamination_constraint_warns_for_nonsterile_environment(
    research_world: WorldState,
) -> None:
    research_world.get_person("grad_kim").current_room_id = "main_wet_lab"

    result = SampleContaminationConstraint().check(
        _action(
            "sample_warn",
            "grad_kim",
            ActionCategory.HANDLE_SAMPLE,
            "handle_sample",
            target_id="hela_p12",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.WARN
    assert "contaminated" in result.message


def test_emergency_equipment_constraint_warns_when_room_lacks_safety_equipment(
    research_world: WorldState,
) -> None:
    person = research_world.get_person("tech_johnson")
    person.current_room_id = "chemical_storage"

    result = EmergencyEquipmentConstraint().check(
        _action(
            "emergency_warn",
            "tech_johnson",
            ActionCategory.HANDLE_MATERIAL,
            "handle_material",
            target_id="hcl_conc",
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.WARN
    assert "eyewash" in result.message
    assert "emergency shower" in result.message


def test_material_expiry_constraint_warns_on_expired_reagent(research_world: WorldState) -> None:
    research_world.add_material(
        Material(
            id="expired_buffer",
            name="Expired Buffer",
            room_id="main_wet_lab",
            expiry_date=datetime(2026, 1, 10, tzinfo=UTC).date(),
        )
    )

    result = MaterialExpiryConstraint().check(
        _action(
            "expiry_warn",
            "grad_kim",
            ActionCategory.HANDLE_MATERIAL,
            "handle_material",
            target_id="expired_buffer",
            timestamp=datetime(2026, 1, 11, 9, 0, tzinfo=UTC),
        ),
        research_world,
    )

    assert result is not None
    assert result.verdict == SafetyVerdict.WARN
    assert "expired" in result.message


def test_safety_engine_aggregates_multiple_findings_for_same_action(
    research_world: WorldState,
    safety_engine: SafetyEngine,
) -> None:
    person = research_world.get_person("tech_johnson")
    person.current_room_id = "chemical_storage"
    person.current_ppe = {
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.NITRILE_GLOVES,
        PPEType.FACE_SHIELD,
        PPEType.CHEMICAL_APRON,
        PPEType.CLOSED_TOE_SHOES,
    }
    research_world.get_material("hcl_conc").expiry_date = (
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC) - timedelta(days=1)
    ).date()

    results = _result_by_id(
        safety_engine.evaluate(
            _action(
                "engine_multi",
                "tech_johnson",
                ActionCategory.HANDLE_MATERIAL,
                "handle_material",
                target_id="hcl_conc",
                timestamp=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            ),
            research_world,
        )
    )

    assert results["ppe_requirement"].verdict == SafetyVerdict.ALLOW
    assert results["fume_hood_requirement"].verdict == SafetyVerdict.BLOCK
    assert results["emergency_equipment"].verdict == SafetyVerdict.WARN
    assert results["material_expiry"].verdict == SafetyVerdict.WARN
