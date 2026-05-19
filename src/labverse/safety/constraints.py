"""Built-in safety constraints for action evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import ClassVar

from labverse.core.types import (
    ActionCategory,
    BSLLevel,
    CertificationType,
    EquipmentStatus,
    HazardPictogram,
    RoomType,
    WasteCategory,
)
from labverse.models.action import Action
from labverse.models.material import Material
from labverse.models.safety import SafetyCheckResult, SafetyVerdict
from labverse.models.sample import Sample
from labverse.safety.bsl_rules import required_certifications_for_bsl
from labverse.safety.chemical_compat import assess_chemical_compatibility
from labverse.safety.ppe import action_room, action_zone_type, missing_ppe, resolve_required_ppe
from labverse.world.state import WorldState


class ConstraintEvaluator(ABC):
    """Abstract base class for safety constraints."""

    constraint_id: ClassVar[str]
    constraint_name: ClassVar[str]

    def applies_to(self, action: Action) -> bool:
        return True

    @abstractmethod
    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None: ...

    def _result(
        self,
        verdict: SafetyVerdict,
        message: str,
        *,
        remediation: str | None = None,
    ) -> SafetyCheckResult:
        return SafetyCheckResult(
            verdict=verdict,
            constraint_id=self.constraint_id,
            constraint_name=self.constraint_name,
            message=message,
            remediation=remediation,
        )


class PPERequirementConstraint(ConstraintEvaluator):
    constraint_id = "ppe_requirement"
    constraint_name = "PPERequirementConstraint"

    def applies_to(self, action: Action) -> bool:
        return action.verb not in {"don_ppe", "doff_ppe", "wait", "log_observation"}

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        person = world.get_person(action.agent_id)
        required_ppe = resolve_required_ppe(action, world)
        if not required_ppe:
            return None

        missing = sorted(ppe.value for ppe in missing_ppe(person, required_ppe))
        if missing:
            return self._result(
                SafetyVerdict.BLOCK,
                f"Person {person.id} is missing required PPE: {', '.join(missing)}",
                remediation="Don the missing PPE before proceeding.",
            )
        return self._result(SafetyVerdict.ALLOW, f"Person {person.id} is wearing the required PPE")


class BSLAccessConstraint(ConstraintEvaluator):
    constraint_id = "bsl_access"
    constraint_name = "BSLAccessConstraint"

    def applies_to(self, action: Action) -> bool:
        return action.verb == "move_to"

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        if action.target_id is None or action.target_id not in world.rooms:
            return None

        room = world.get_room(action.target_id)
        if room.bsl_level <= BSLLevel.BSL1:
            return self._result(
                SafetyVerdict.ALLOW, f"Room {room.id} does not require elevated BSL access"
            )

        person = world.get_person(action.agent_id)
        required = required_certifications_for_bsl(room.bsl_level)
        missing = sorted(cert.value for cert in required if not person.has_certification(cert))
        if missing:
            return self._result(
                SafetyVerdict.BLOCK,
                f"Room {room.id} requires certifications: {', '.join(missing)}",
                remediation="Obtain the required biosafety training before entering.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Person {person.id} meets the BSL-{room.bsl_level.value} access requirement",
        )


class ChemicalCompatibilityConstraint(ConstraintEvaluator):
    constraint_id = "chemical_compatibility"
    constraint_name = "ChemicalCompatibilityConstraint"

    def applies_to(self, action: Action) -> bool:
        return (
            action.category == ActionCategory.HANDLE_MATERIAL or action.verb == "pipette_transfer"
        )

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        primary = _target_material(action, world)
        if primary is None:
            return None

        comparisons = _comparison_materials(action, world, primary)
        if not comparisons:
            return self._result(
                SafetyVerdict.ALLOW,
                f"No incompatible co-materials identified for {primary.id}",
            )

        warning_message: str | None = None
        for other in comparisons:
            finding = assess_chemical_compatibility(primary, other)
            if finding is None:
                continue
            verdict, reason = finding
            if verdict == SafetyVerdict.BLOCK:
                return self._result(
                    SafetyVerdict.BLOCK,
                    f"{primary.name} is incompatible with {other.name}: {reason}",
                    remediation="Separate the materials or choose a compatible reagent set.",
                )
            if warning_message is None:
                warning_message = (
                    f"{primary.name} and {other.name} should be handled with "
                    f"extra caution: {reason}"
                )

        if warning_message is not None:
            return self._result(SafetyVerdict.WARN, warning_message)
        return self._result(
            SafetyVerdict.ALLOW,
            f"No chemical compatibility conflicts detected for {primary.id}",
        )


class CertificationConstraint(ConstraintEvaluator):
    constraint_id = "certification"
    constraint_name = "CertificationConstraint"

    def applies_to(self, action: Action) -> bool:
        return action.category == ActionCategory.USE_EQUIPMENT

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        if action.target_id is None or action.target_id not in world.equipment:
            return None

        person = world.get_person(action.agent_id)
        equipment = world.get_equipment(action.target_id)
        required = set(equipment.required_certifications) | _required_certifications_from_action(
            action
        )
        if not required:
            return self._result(
                SafetyVerdict.ALLOW,
                f"Equipment {equipment.id} does not require extra certification for this action",
            )

        missing = sorted(cert.value for cert in required if not person.has_certification(cert))
        if missing:
            return self._result(
                SafetyVerdict.BLOCK,
                f"Person {person.id} lacks certifications for {equipment.id}: {', '.join(missing)}",
                remediation="Use a certified operator or complete the required training.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Person {person.id} satisfies the certification requirements for {equipment.id}",
        )


class RoomOccupancyConstraint(ConstraintEvaluator):
    constraint_id = "room_occupancy"
    constraint_name = "RoomOccupancyConstraint"

    def applies_to(self, action: Action) -> bool:
        return action.verb == "move_to"

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        if action.target_id is None or action.target_id not in world.rooms:
            return None

        person = world.get_person(action.agent_id)
        room = world.get_room(action.target_id)
        occupancy = len(world.persons_in_room(room.id))
        if person.current_room_id != room.id and occupancy >= room.max_occupancy:
            return self._result(
                SafetyVerdict.BLOCK,
                f"Room {room.id} is at max occupancy ({occupancy}/{room.max_occupancy})",
                remediation="Wait until the room occupancy drops before entering.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Room {room.id} occupancy is within limits ({occupancy}/{room.max_occupancy})",
        )


class EquipmentConcurrencyConstraint(ConstraintEvaluator):
    constraint_id = "equipment_concurrency"
    constraint_name = "EquipmentConcurrencyConstraint"

    def applies_to(self, action: Action) -> bool:
        return action.category == ActionCategory.USE_EQUIPMENT

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        if action.target_id is None or action.target_id not in world.equipment:
            return None

        equipment = world.get_equipment(action.target_id)
        if (
            len(equipment.current_users) >= equipment.max_concurrent_users
            and action.agent_id not in equipment.current_users
        ):
            return self._result(
                SafetyVerdict.BLOCK,
                f"Equipment {equipment.id} is already at max concurrent users",
                remediation="Wait for the current run to finish or use another instrument.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Equipment {equipment.id} has available user capacity",
        )


class FumeHoodRequirementConstraint(ConstraintEvaluator):
    constraint_id = "fume_hood_requirement"
    constraint_name = "FumeHoodRequirementConstraint"

    def applies_to(self, action: Action) -> bool:
        return (
            action.category == ActionCategory.HANDLE_MATERIAL or action.verb == "pipette_transfer"
        )

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        material = _target_material(action, world)
        if material is None or not _requires_fume_hood(material):
            return None

        zone_type = action_zone_type(action, world)
        if zone_type == "fume_hood_area":
            return self._result(
                SafetyVerdict.ALLOW,
                f"Material {material.id} is being handled under a fume hood",
            )
        return self._result(
            SafetyVerdict.BLOCK,
            f"Material {material.id} must be handled in a fume hood zone",
            remediation="Move the procedure to a fume hood before proceeding.",
        )


class WasteDisposalConstraint(ConstraintEvaluator):
    constraint_id = "waste_disposal"
    constraint_name = "WasteDisposalConstraint"

    def applies_to(self, action: Action) -> bool:
        return "dispose" in action.verb or action.parameters.get("waste_category") is not None

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        expected = _expected_waste_category(action, world)
        if expected is None:
            return None

        actual_raw = action.parameters.get("waste_category", action.parameters.get("waste_stream"))
        if actual_raw is None:
            return self._result(
                SafetyVerdict.WARN,
                "Waste stream was not specified for disposal",
                remediation="Classify the waste stream explicitly before disposal.",
            )

        try:
            actual = WasteCategory(str(actual_raw))
        except ValueError:
            return self._result(
                SafetyVerdict.WARN,
                f"Unknown waste stream classification: {actual_raw!r}",
                remediation="Use a supported waste stream classification.",
            )

        if actual != expected:
            return self._result(
                SafetyVerdict.BLOCK,
                f"Waste must go to {expected.value}, not {actual.value}",
                remediation="Dispose of the material in the correct waste stream.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Waste stream {actual.value} matches the required classification",
        )


class SampleContaminationConstraint(ConstraintEvaluator):
    constraint_id = "sample_contamination"
    constraint_name = "SampleContaminationConstraint"

    def __init__(self, *, enforcement_level: SafetyVerdict = SafetyVerdict.WARN) -> None:
        self._enforcement_level = enforcement_level

    def applies_to(self, action: Action) -> bool:
        # Only open-liquid exposure is meaningful for contamination. Sealed
        # transport (retrieve_sample / store_sample move closed vessels in and
        # out of incubators) does not expose the sample to room air and must
        # not be gated by sterile-environment checks, or the agent cannot move
        # cell cultures between rooms at all.
        return action.verb == "pipette_transfer"

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        sample = _target_sample(action, world)
        if sample is None or not _sample_requires_sterile_handling(sample):
            return None

        room = action_room(action, world)
        if room is None:
            return self._result(
                self._enforcement_level,
                f"Unable to verify sterile handling environment for sample {sample.id}",
            )

        if _sterile_environment_available(action, world, sample):
            return self._result(
                SafetyVerdict.ALLOW,
                f"Sample {sample.id} is being handled in a sterile environment",
            )
        return self._result(
            self._enforcement_level,
            f"Sample {sample.id} may be contaminated outside a sterile handling environment",
            remediation="Handle the sample in a tissue culture room or active biosafety cabinet.",
        )


class EmergencyEquipmentConstraint(ConstraintEvaluator):
    constraint_id = "emergency_equipment"
    constraint_name = "EmergencyEquipmentConstraint"

    def applies_to(self, action: Action) -> bool:
        return (
            action.category == ActionCategory.HANDLE_MATERIAL or action.verb == "pipette_transfer"
        )

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        material = _target_material(action, world)
        room = action_room(action, world)
        if material is None or room is None:
            return None

        missing: list[str] = []
        if HazardPictogram.GHS05_CORROSIVE in material.ghs.pictograms:
            if not room.has_eyewash:
                missing.append("eyewash")
            if not room.has_emergency_shower:
                missing.append("emergency shower")
        elif material.is_hazardous and not room.has_eyewash:
            missing.append("eyewash")

        if missing:
            return self._result(
                SafetyVerdict.WARN,
                f"Room {room.id} lacks emergency equipment for {material.id}: {', '.join(missing)}",
                remediation="Relocate the work or ensure emergency equipment is available.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Room {room.id} has the emergency equipment needed for {material.id}",
        )


class MaterialExpiryConstraint(ConstraintEvaluator):
    constraint_id = "material_expiry"
    constraint_name = "MaterialExpiryConstraint"

    def __init__(self, *, enforcement_level: SafetyVerdict = SafetyVerdict.WARN) -> None:
        self._enforcement_level = enforcement_level

    def applies_to(self, action: Action) -> bool:
        # Disposing an expired material IS the correct response to expiry,
        # so dispose_material is deliberately out of scope — otherwise the
        # constraint would block the cleanup path.
        if action.verb == "dispose_material":
            return False
        return (
            action.category == ActionCategory.HANDLE_MATERIAL or action.verb == "pipette_transfer"
        )

    def check(self, action: Action, world: WorldState) -> SafetyCheckResult | None:
        material = _target_material(action, world)
        if material is None or material.expiry_date is None:
            return None

        current_date = action.timestamp.date() if action.timestamp is not None else date.today()
        if material.is_expired_on(current_date):
            return self._result(
                self._enforcement_level,
                f"Material {material.id} is expired as of {current_date.isoformat()}",
                remediation="Replace the reagent or explicitly document the exception.",
            )
        return self._result(
            SafetyVerdict.ALLOW,
            f"Material {material.id} is within its expiry window",
        )


BUILTIN_CONSTRAINTS: tuple[type[ConstraintEvaluator], ...] = (
    PPERequirementConstraint,
    BSLAccessConstraint,
    ChemicalCompatibilityConstraint,
    CertificationConstraint,
    RoomOccupancyConstraint,
    EquipmentConcurrencyConstraint,
    FumeHoodRequirementConstraint,
    WasteDisposalConstraint,
    SampleContaminationConstraint,
    EmergencyEquipmentConstraint,
    MaterialExpiryConstraint,
)


def _target_material(action: Action, world: WorldState) -> Material | None:
    for entity_id in _action_entity_ids(action):
        if entity_id in world.materials:
            return world.get_material(entity_id)
    return None


def _comparison_materials(action: Action, world: WorldState, primary: Material) -> list[Material]:
    material_ids: list[str] = []
    for key in ("other_material_ids", "with_material_ids", "co_located_material_ids"):
        raw_value = action.parameters.get(key)
        if isinstance(raw_value, list):
            material_ids.extend(str(value) for value in raw_value if isinstance(value, str))

    single_other = action.parameters.get("other_material_id")
    if isinstance(single_other, str):
        material_ids.append(single_other)

    if not material_ids:
        material_ids.extend(
            entity_id
            for entity_id in _action_entity_ids(action)
            if entity_id in world.materials and entity_id != primary.id
        )

    materials = [
        world.get_material(material_id)
        for material_id in material_ids
        if material_id in world.materials
    ]
    if materials:
        return materials

    destination_room = action_room(action, world)
    if destination_room is None:
        return []
    return [
        material
        for material in world.materials_in_room(destination_room.id)
        if material.id != primary.id
    ]


def _required_certifications_from_action(action: Action) -> set[CertificationType]:
    raw_value = action.parameters.get("required_certifications")
    if not isinstance(raw_value, list):
        return set()

    required: set[CertificationType] = set()
    for item in raw_value:
        if not isinstance(item, str):
            continue
        try:
            required.add(CertificationType(item))
        except ValueError:
            continue
    return required


def _requires_fume_hood(material: Material) -> bool:
    hazardous_pictograms = {
        HazardPictogram.GHS02_FLAMMABLE,
        HazardPictogram.GHS04_COMPRESSED_GAS,
        HazardPictogram.GHS05_CORROSIVE,
        HazardPictogram.GHS06_TOXIC,
        HazardPictogram.GHS08_HEALTH_HAZARD,
    }
    return material.phase.value == "gas" or any(
        pictogram in hazardous_pictograms for pictogram in material.ghs.pictograms
    )


def _expected_waste_category(action: Action, world: WorldState) -> WasteCategory | None:
    if action.target_id is not None and action.target_id in world.materials:
        return world.get_material(action.target_id).waste_category
    if action.target_id is not None and action.target_id in world.samples:
        sample = world.get_sample(action.target_id)
        return (
            WasteCategory.BIOHAZARD if sample.bsl_level >= BSLLevel.BSL2 else WasteCategory.GENERAL
        )
    return None


def _target_sample(action: Action, world: WorldState) -> Sample | None:
    for entity_id in _action_entity_ids(action):
        if entity_id in world.samples:
            return world.get_sample(entity_id)
    return None


def _action_entity_ids(action: Action) -> list[str]:
    ids: list[str] = []
    if action.target_id is not None:
        ids.append(action.target_id)
    for value in action.parameters.values():
        if isinstance(value, str):
            ids.append(value)
        elif isinstance(value, list):
            ids.extend(item for item in value if isinstance(item, str))
    return ids


def _sample_requires_sterile_handling(sample: Sample) -> bool:
    metadata_flag = sample.metadata.get("sterile", "").casefold() == "true"
    return metadata_flag or sample.sample_type in {"cell_culture", "primary_cells", "sterile_media"}


def _sterile_environment_available(action: Action, world: WorldState, sample: Sample) -> bool:
    if bool(action.parameters.get("inside_bsc")) or bool(
        action.parameters.get("sterile_technique")
    ):
        return True

    if sample.equipment_id is not None and sample.equipment_id in world.equipment:
        equipment = world.get_equipment(sample.equipment_id)
        if equipment.equipment_type == "biosafety_cabinet" and equipment.status in {
            EquipmentStatus.READY,
            EquipmentStatus.RUNNING,
        }:
            return True

    room = action_room(action, world)
    if room is None or room.room_type != RoomType.TISSUE_CULTURE:
        return False

    return any(
        equipment.equipment_type == "biosafety_cabinet"
        and equipment.status in {EquipmentStatus.READY, EquipmentStatus.RUNNING}
        for equipment in world.equipment_in_room(room.id)
    )


__all__ = [
    "BUILTIN_CONSTRAINTS",
    "BSLAccessConstraint",
    "ChemicalCompatibilityConstraint",
    "CertificationConstraint",
    "ConstraintEvaluator",
    "EmergencyEquipmentConstraint",
    "EquipmentConcurrencyConstraint",
    "FumeHoodRequirementConstraint",
    "MaterialExpiryConstraint",
    "PPERequirementConstraint",
    "RoomOccupancyConstraint",
    "SampleContaminationConstraint",
    "WasteDisposalConstraint",
]
