"""Spatial index: room adjacency graph and pathfinding."""

from __future__ import annotations

from collections import deque

from labverse.core.types import BSLLevel
from labverse.models.space import Door
from labverse.world.state import WorldState


class SpatialIndex:
    """Room adjacency graph built from doors, with pathfinding and access checks."""

    def __init__(self, world: WorldState) -> None:
        self._world = world
        self._adjacency: dict[str, list[str]] = {}
        self._door_map: dict[tuple[str, str], Door] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild adjacency from current doors."""
        self._adjacency.clear()
        self._door_map.clear()
        for room_id in self._world.rooms:
            self._adjacency[room_id] = []
        for door in self._world.doors:
            a, b = door.connects
            if a in self._adjacency and b in self._adjacency:
                self._adjacency[a].append(b)
                self._adjacency[b].append(a)
                self._door_map[(a, b)] = door
                self._door_map[(b, a)] = door

    def neighbors(self, room_id: str) -> list[str]:
        """Get all rooms directly connected to the given room."""
        return list(self._adjacency.get(room_id, []))

    def get_door(self, room_a: str, room_b: str) -> Door | None:
        """Get the door connecting two rooms, or None."""
        return self._door_map.get((room_a, room_b))

    def are_adjacent(self, room_a: str, room_b: str) -> bool:
        """Check if two rooms are directly connected."""
        return room_b in self._adjacency.get(room_a, [])

    def can_traverse(
        self,
        room_a: str,
        room_b: str,
        person_bsl_clearance: BSLLevel = BSLLevel.BSL1,
    ) -> tuple[bool, str]:
        """Check if a person can move from room_a to room_b through the connecting door.

        Returns (can_move, reason). reason is empty string if allowed.
        """
        if not self.are_adjacent(room_a, room_b):
            return False, f"Rooms {room_a} and {room_b} are not adjacent"
        door = self.get_door(room_a, room_b)
        if door is None:
            return False, "No door found"
        if door.locked:
            return False, f"Door {door.id} is locked"
        if not door.is_open:
            return False, f"Door {door.id} is closed"
        if door.requires_bsl_clearance is not None:
            if person_bsl_clearance < door.requires_bsl_clearance:
                required_bsl = door.requires_bsl_clearance.value
                current_bsl = person_bsl_clearance.value
                return False, (
                    f"Door {door.id} requires BSL-{required_bsl} clearance, "
                    f"person has BSL-{current_bsl}"
                )
        return True, ""

    def find_path(
        self,
        start: str,
        end: str,
        person_bsl_clearance: BSLLevel = BSLLevel.BSL1,
        ignore_locks: bool = False,
    ) -> list[str] | None:
        """BFS shortest path from start to end, respecting access constraints.

        Returns list of room IDs from start to end (inclusive), or None if no path.
        """
        if start == end:
            return [start]
        if start not in self._adjacency or end not in self._adjacency:
            return None

        visited: set[str] = {start}
        queue: deque[list[str]] = deque([[start]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor in self._adjacency.get(current, []):
                if neighbor in visited:
                    continue
                door = self.get_door(current, neighbor)
                if door is not None:
                    if (door.locked or not door.is_open) and not ignore_locks:
                        continue
                    if door.requires_bsl_clearance is not None:
                        if person_bsl_clearance < door.requires_bsl_clearance:
                            continue
                visited.add(neighbor)
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                queue.append(new_path)

        return None

    def reachable_rooms(
        self,
        start: str,
        person_bsl_clearance: BSLLevel = BSLLevel.BSL1,
    ) -> set[str]:
        """Get all rooms reachable from start, respecting access constraints."""
        if start not in self._adjacency:
            return set()

        visited: set[str] = {start}
        queue: deque[str] = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency.get(current, []):
                if neighbor in visited:
                    continue
                door = self.get_door(current, neighbor)
                if door is not None:
                    if door.locked or not door.is_open:
                        continue
                    if door.requires_bsl_clearance is not None:
                        if person_bsl_clearance < door.requires_bsl_clearance:
                            continue
                visited.add(neighbor)
                queue.append(neighbor)

        return visited

    def room_distance(self, room_a: str, room_b: str) -> int | None:
        """Minimum number of doors between two rooms, or None if unreachable."""
        path = self.find_path(room_a, room_b, BSLLevel.BSL4, ignore_locks=True)
        if path is None:
            return None
        return len(path) - 1
