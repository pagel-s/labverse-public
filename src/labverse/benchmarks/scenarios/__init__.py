"""Benchmark scenario implementations."""

from labverse.benchmarks.scenarios.base import BenchmarkScenario
from labverse.benchmarks.scenarios.branch_selection_under_uncertainty import (
    BranchSelectionUnderUncertaintyScenario,
)
from labverse.benchmarks.scenarios.cell_culture_passage import CellCulturePassageScenario
from labverse.benchmarks.scenarios.chemical_spill_response import ChemicalSpillResponseScenario
from labverse.benchmarks.scenarios.cold_chain_time_critical_transfer import (
    ColdChainTimeCriticalTransferScenario,
)
from labverse.benchmarks.scenarios.contamination_source_tracing import (
    ContaminationSourceTracingScenario,
)
from labverse.benchmarks.scenarios.emergency_triage_response import (
    EmergencyTriageResponseScenario,
)
from labverse.benchmarks.scenarios.equipment_failure_recovery import (
    EquipmentFailureRecoveryScenario,
)
from labverse.benchmarks.scenarios.inventory_management import InventoryManagementScenario
from labverse.benchmarks.scenarios.multi_step_protocol import MultiStepProtocolScenario
from labverse.benchmarks.scenarios.pcr_setup import PCRSetupScenario
from labverse.benchmarks.scenarios.reagent_preparation import ReagentPreparationScenario
from labverse.benchmarks.scenarios.safety_violation_detection import (
    SafetyViolationDetectionScenario,
)
from labverse.benchmarks.scenarios.sample_tracking import SampleTrackingScenario
from labverse.benchmarks.scenarios.shared_equipment_scheduling import (
    SharedEquipmentSchedulingScenario,
)

SCENARIO_REGISTRY: dict[str, type[BenchmarkScenario]] = {
    BranchSelectionUnderUncertaintyScenario.name: BranchSelectionUnderUncertaintyScenario,
    PCRSetupScenario.name: PCRSetupScenario,
    CellCulturePassageScenario.name: CellCulturePassageScenario,
    ChemicalSpillResponseScenario.name: ChemicalSpillResponseScenario,
    ColdChainTimeCriticalTransferScenario.name: ColdChainTimeCriticalTransferScenario,
    ContaminationSourceTracingScenario.name: ContaminationSourceTracingScenario,
    ReagentPreparationScenario.name: ReagentPreparationScenario,
    EquipmentFailureRecoveryScenario.name: EquipmentFailureRecoveryScenario,
    EmergencyTriageResponseScenario.name: EmergencyTriageResponseScenario,
    InventoryManagementScenario.name: InventoryManagementScenario,
    MultiStepProtocolScenario.name: MultiStepProtocolScenario,
    SafetyViolationDetectionScenario.name: SafetyViolationDetectionScenario,
    SharedEquipmentSchedulingScenario.name: SharedEquipmentSchedulingScenario,
    SampleTrackingScenario.name: SampleTrackingScenario,
}

__all__ = [
    "BenchmarkScenario",
    "BranchSelectionUnderUncertaintyScenario",
    "CellCulturePassageScenario",
    "ChemicalSpillResponseScenario",
    "ColdChainTimeCriticalTransferScenario",
    "ContaminationSourceTracingScenario",
    "EmergencyTriageResponseScenario",
    "EquipmentFailureRecoveryScenario",
    "InventoryManagementScenario",
    "MultiStepProtocolScenario",
    "PCRSetupScenario",
    "ReagentPreparationScenario",
    "SCENARIO_REGISTRY",
    "SafetyViolationDetectionScenario",
    "SampleTrackingScenario",
    "SharedEquipmentSchedulingScenario",
]
