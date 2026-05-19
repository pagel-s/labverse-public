"""Agent interface: observation, action space, scripted agents."""

from labverse.agent.interface import (
    AgentInterface,
    DoorSummary,
    EquipmentSummary,
    MaterialSummary,
    Observation,
    PersonSummary,
    SampleSummary,
    SelfState,
)
from labverse.agent.observation import ObservationBuilder
from labverse.agent.scripted import ScriptedAgent, ScriptedStep
from labverse.agent.trace import AgentMemorySnapshot, AgentTraceRecord, AgentTraceStore

__all__ = [
    "ActionSpaceCalculator",
    "AgentInterface",
    "AgentMemorySnapshot",
    "AgentTraceRecord",
    "AgentTraceStore",
    "DoorSummary",
    "EquipmentSummary",
    "GlobalStructuredInteractiveAgent",
    "InteractiveTextAgent",
    "LLMAgent",
    "MaterialSummary",
    "Observation",
    "ObservationBuilder",
    "PersonSummary",
    "SampleSummary",
    "ScriptedAgent",
    "ScriptedStep",
    "SelfState",
    "TextPlannerAgent",
]


def __getattr__(name: str) -> object:
    if name == "ActionSpaceCalculator":
        from labverse.agent.action_space import ActionSpaceCalculator

        return ActionSpaceCalculator
    if name == "LLMAgent":
        from labverse.agent.llm import LLMAgent

        return LLMAgent
    if name == "TextPlannerAgent":
        from labverse.agent.text_planner import TextPlannerAgent

        return TextPlannerAgent
    if name == "InteractiveTextAgent":
        from labverse.agent.interactive_text import InteractiveTextAgent

        return InteractiveTextAgent
    if name == "GlobalStructuredInteractiveAgent":
        from labverse.agent.global_structured import GlobalStructuredInteractiveAgent

        return GlobalStructuredInteractiveAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
