"""Concrete equipment state machine implementations and factory helpers."""

from labverse.equipment.devices.autoclave import AutoclaveStateMachine
from labverse.equipment.devices.biosafety_cabinet import BiosafetyCabinetStateMachine
from labverse.equipment.devices.centrifuge import CentrifugeStateMachine
from labverse.equipment.devices.incubator import IncubatorStateMachine
from labverse.equipment.devices.pcr_thermocycler import PcrThermocyclerStateMachine
from labverse.equipment.devices.pipette import PipetteStateMachine
from labverse.equipment.state_machine import EquipmentStateMachine
from labverse.models.equipment import Equipment

STATE_MACHINE_REGISTRY: dict[str, type[EquipmentStateMachine]] = {
    "autoclave": AutoclaveStateMachine,
    "biosafety_cabinet": BiosafetyCabinetStateMachine,
    "centrifuge": CentrifugeStateMachine,
    "incubator": IncubatorStateMachine,
    "pcr_thermocycler": PcrThermocyclerStateMachine,
    "pipette": PipetteStateMachine,
}


def create_state_machine(equipment: Equipment) -> EquipmentStateMachine:
    machine_cls = STATE_MACHINE_REGISTRY.get(equipment.equipment_type, EquipmentStateMachine)
    return machine_cls(equipment)


__all__ = [
    "AutoclaveStateMachine",
    "BiosafetyCabinetStateMachine",
    "CentrifugeStateMachine",
    "IncubatorStateMachine",
    "PcrThermocyclerStateMachine",
    "PipetteStateMachine",
    "STATE_MACHINE_REGISTRY",
    "create_state_machine",
]
