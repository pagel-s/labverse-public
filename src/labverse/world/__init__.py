"""World state management: state container, entity registry, spatial index."""

from labverse.world.registry import EntityRegistry
from labverse.world.spatial import SpatialIndex
from labverse.world.state import WorldState

__all__ = ["EntityRegistry", "SpatialIndex", "WorldState"]
