"""Tests for WorldState, EntityRegistry, and SpatialIndex."""

from datetime import UTC, date, datetime

import pytest

from labverse.core.types import (
    BSLLevel,
    CertificationType,
    EquipmentStatus,
    HazardPictogram,
    PersonRole,
    RoomType,
)
from labverse.models.equipment import Equipment
from labverse.models.material import GHSData, Material
from labverse.models.person import Certification, Person
from labverse.models.sample import Sample
from labverse.models.space import Door, LabLayout, Room, Zone
from labverse.world.registry import EntityRegistry
from labverse.world.spatial import SpatialIndex
from labverse.world.state import WorldState

NOW = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def lab_world() -> WorldState:
    """A realistic lab with 5 rooms, 4 doors, equipment, materials, persons.

    Layout:
        office -- wet_lab -- tissue_culture (BSL-2 gated)
                    |
                 cold_room
                    |
                 chem_storage
    """
    rooms = {
        "office": Room(id="office", name="Office", room_type=RoomType.OFFICE),
        "wet_lab": Room(
            id="wet_lab",
            name="Main Wet Lab",
            room_type=RoomType.WET_LAB,
            has_emergency_shower=True,
            has_eyewash=True,
            zones=[
                Zone(id="bench_1", name="Bench 1", zone_type="bench", capacity=2),
                Zone(id="fume_hood_1", name="Fume Hood 1", zone_type="fume_hood_area"),
            ],
        ),
        "tc_room": Room(
            id="tc_room",
            name="Tissue Culture",
            room_type=RoomType.TISSUE_CULTURE,
            bsl_level=BSLLevel.BSL2,
            max_occupancy=3,
        ),
        "cold_room": Room(
            id="cold_room",
            name="Cold Room",
            room_type=RoomType.COLD_ROOM,
            temperature_celsius=4.0,
        ),
        "chem_storage": Room(
            id="chem_storage",
            name="Chemical Storage",
            room_type=RoomType.CHEMICAL_STORAGE,
        ),
    }
    doors = [
        Door(id="d_office_wet", connects=("office", "wet_lab")),
        Door(id="d_wet_tc", connects=("wet_lab", "tc_room"), requires_bsl_clearance=BSLLevel.BSL2),
        Door(id="d_wet_cold", connects=("wet_lab", "cold_room")),
        Door(id="d_cold_chem", connects=("cold_room", "chem_storage")),
    ]
    layout = LabLayout(
        id="test_lab",
        name="Test Lab",
        rooms=rooms,
        doors=doors,
        default_entry_room="office",
    )

    world = WorldState(lab=layout)

    # Equipment
    world.add_equipment(
        Equipment(
            id="centrifuge_1",
            name="Centrifuge",
            equipment_type="centrifuge",
            room_id="wet_lab",
            status=EquipmentStatus.IDLE,
        )
    )
    world.add_equipment(
        Equipment(
            id="pcr_1",
            name="PCR",
            equipment_type="pcr_thermocycler",
            room_id="wet_lab",
            status=EquipmentStatus.RUNNING,
        )
    )
    world.add_equipment(
        Equipment(
            id="bsc_1",
            name="BSC",
            equipment_type="biosafety_cabinet",
            room_id="tc_room",
            status=EquipmentStatus.IDLE,
            required_certifications=[CertificationType.BSL2_TRAINING],
        )
    )
    world.add_equipment(
        Equipment(
            id="incubator_1",
            name="Incubator",
            equipment_type="incubator",
            room_id="tc_room",
            status=EquipmentStatus.RUNNING,
        )
    )

    # Materials
    world.add_material(
        Material(
            id="ethanol",
            name="Ethanol 70%",
            room_id="wet_lab",
            ghs=GHSData(pictograms=[HazardPictogram.GHS02_FLAMMABLE]),
            quantity=2500.0,
        )
    )
    world.add_material(
        Material(
            id="pbs",
            name="PBS 1X",
            room_id="cold_room",
            quantity=500.0,
        )
    )
    world.add_material(
        Material(
            id="hcl",
            name="HCl Conc.",
            room_id="chem_storage",
            ghs=GHSData(
                pictograms=[HazardPictogram.GHS05_CORROSIVE, HazardPictogram.GHS07_IRRITANT]
            ),
            quantity=500.0,
        )
    )

    # Persons
    world.add_person(
        Person(
            id="pi_chen",
            name="Dr. Chen",
            role=PersonRole.PI,
            current_room_id="office",
            certifications=[
                Certification(cert_type=CertificationType.BSL2_TRAINING, granted_date="2020-01-01"),
                Certification(
                    cert_type=CertificationType.CHEMICAL_SAFETY, granted_date="2020-01-01"
                ),
            ],
        )
    )
    world.add_person(
        Person(
            id="grad_kim",
            name="Alex Kim",
            role=PersonRole.GRAD_STUDENT,
            current_room_id="wet_lab",
            certifications=[
                Certification(cert_type=CertificationType.BSL2_TRAINING, granted_date="2024-09-01"),
            ],
        )
    )
    world.add_person(
        Person(
            id="undergrad_lee",
            name="Jordan Lee",
            role=PersonRole.UNDERGRAD,
            current_room_id="wet_lab",
        )
    )

    # Samples
    world.add_sample(
        Sample(
            id="sample_1",
            name="HeLa P12",
            sample_type="cell_culture",
            room_id="tc_room",
            equipment_id="incubator_1",
            created_at=NOW,
            created_by="grad_kim",
            bsl_level=BSLLevel.BSL2,
        )
    )

    return world


# ── WorldState Tests ───────────────────────────────────────────────────────


class TestWorldState:
    def test_entity_counts(self, lab_world: WorldState) -> None:
        counts = lab_world.entity_counts
        assert counts["rooms"] == 5
        assert counts["doors"] == 4
        assert counts["equipment"] == 4
        assert counts["materials"] == 3
        assert counts["persons"] == 3
        assert counts["samples"] == 1

    def test_get_room(self, lab_world: WorldState) -> None:
        room = lab_world.get_room("wet_lab")
        assert room.name == "Main Wet Lab"
        assert room.room_type == RoomType.WET_LAB

    def test_get_room_not_found(self, lab_world: WorldState) -> None:
        with pytest.raises(KeyError, match="Room not found"):
            lab_world.get_room("nonexistent")

    def test_get_equipment(self, lab_world: WorldState) -> None:
        eq = lab_world.get_equipment("centrifuge_1")
        assert eq.equipment_type == "centrifuge"

    def test_get_person(self, lab_world: WorldState) -> None:
        p = lab_world.get_person("pi_chen")
        assert p.role == PersonRole.PI

    def test_equipment_in_room(self, lab_world: WorldState) -> None:
        wet_eq = lab_world.equipment_in_room("wet_lab")
        assert len(wet_eq) == 2
        types = {eq.equipment_type for eq in wet_eq}
        assert types == {"centrifuge", "pcr_thermocycler"}

    def test_room_indexes_stay_in_sync(self, lab_world: WorldState) -> None:
        wet_lab = lab_world.get_room("wet_lab")
        assert wet_lab.equipment_ids == ["centrifuge_1", "pcr_1"]
        assert wet_lab.material_ids == ["ethanol"]

        lab_world.remove_equipment("pcr_1")
        lab_world.remove_material("ethanol")

        assert wet_lab.equipment_ids == ["centrifuge_1"]
        assert wet_lab.material_ids == []

    def test_materials_in_room(self, lab_world: WorldState) -> None:
        mats = lab_world.materials_in_room("wet_lab")
        assert len(mats) == 1
        assert mats[0].id == "ethanol"

    def test_persons_in_room(self, lab_world: WorldState) -> None:
        persons = lab_world.persons_in_room("wet_lab")
        assert len(persons) == 2
        names = {p.name for p in persons}
        assert names == {"Alex Kim", "Jordan Lee"}

    def test_samples_in_room(self, lab_world: WorldState) -> None:
        samples = lab_world.samples_in_room("tc_room")
        assert len(samples) == 1

    def test_add_and_remove_equipment(self, lab_world: WorldState) -> None:
        eq = Equipment(id="new_eq", name="New", equipment_type="balance", room_id="wet_lab")
        lab_world.add_equipment(eq)
        assert lab_world.get_equipment("new_eq").name == "New"
        removed = lab_world.remove_equipment("new_eq")
        assert removed.id == "new_eq"
        with pytest.raises(KeyError):
            lab_world.get_equipment("new_eq")

    def test_remove_nonexistent_raises(self, lab_world: WorldState) -> None:
        with pytest.raises(KeyError):
            lab_world.remove_equipment("nope")
        with pytest.raises(KeyError):
            lab_world.remove_material("nope")
        with pytest.raises(KeyError):
            lab_world.remove_sample("nope")
        with pytest.raises(KeyError):
            lab_world.remove_person("nope")

    def test_snapshot_is_deep_copy(self, lab_world: WorldState) -> None:
        snap = lab_world.snapshot()
        # Mutate original
        lab_world.get_person("pi_chen").current_room_id = "wet_lab"
        # Snapshot is unaffected
        assert snap.get_person("pi_chen").current_room_id == "office"

    def test_repr(self, lab_world: WorldState) -> None:
        r = repr(lab_world)
        assert "Test Lab" in r
        assert "rooms=5" in r

    def test_empty_world(self) -> None:
        w = WorldState()
        assert w.entity_counts["rooms"] == 0
        assert w.entity_counts["equipment"] == 0


# ── EntityRegistry Tests ──────────────────────────────────────────────────


class TestEntityRegistry:
    def test_generic_get(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        eq = reg.get("equipment", "centrifuge_1")
        assert eq.name == "Centrifuge"
        room = reg.get("room", "wet_lab")
        assert room.room_type == RoomType.WET_LAB

    def test_generic_get_bad_type(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        with pytest.raises(ValueError, match="Unknown entity type"):
            reg.get("widget", "x")

    def test_rooms_by_type(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        wet_labs = reg.rooms_by_type(RoomType.WET_LAB)
        assert len(wet_labs) == 1
        assert wet_labs[0].id == "wet_lab"

    def test_rooms_by_bsl(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        bsl2_rooms = reg.rooms_by_bsl(2)
        assert len(bsl2_rooms) == 1
        assert bsl2_rooms[0].id == "tc_room"

    def test_equipment_by_type(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        centrifuges = reg.equipment_by_type("centrifuge")
        assert len(centrifuges) == 1

    def test_equipment_by_status(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        running = reg.equipment_by_status(EquipmentStatus.RUNNING)
        assert len(running) == 2  # PCR + incubator

    def test_available_equipment(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        available = reg.available_equipment()
        assert len(available) == 2  # centrifuge (IDLE) + BSC (IDLE)
        available_centrifuge = reg.available_equipment("centrifuge")
        assert len(available_centrifuge) == 1

    def test_hazardous_materials(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        haz = reg.hazardous_materials()
        assert len(haz) == 2  # ethanol, HCl

    def test_persons_by_role(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        grads = reg.persons_by_role(PersonRole.GRAD_STUDENT)
        assert len(grads) == 1
        assert grads[0].name == "Alex Kim"

    def test_expired_materials_accept_explicit_date(self, lab_world: WorldState) -> None:
        lab_world.add_material(
            Material(
                id="expired_buffer",
                name="Expired Buffer",
                room_id="wet_lab",
                expiry_date=date(2026, 1, 10),
            )
        )
        reg = EntityRegistry(lab_world)
        expired = reg.expired_materials(current_date=date(2026, 1, 15))
        assert [material.id for material in expired] == ["expired_buffer"]
        assert reg.expired_materials(current_date=date(2026, 1, 9)) == []

    def test_active_persons(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        assert len(reg.active_persons()) == 3

    def test_active_samples(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        assert len(reg.active_samples()) == 1

    def test_all_in_room(self, lab_world: WorldState) -> None:
        reg = EntityRegistry(lab_world)
        contents = reg.all_in_room("wet_lab")
        assert len(contents["equipment"]) == 2
        assert len(contents["materials"]) == 1
        assert len(contents["persons"]) == 2
        assert len(contents["samples"]) == 0


# ── SpatialIndex Tests ────────────────────────────────────────────────────


class TestSpatialIndex:
    def test_neighbors(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        neighbors = spatial.neighbors("wet_lab")
        assert set(neighbors) == {"office", "tc_room", "cold_room"}

    def test_are_adjacent(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        assert spatial.are_adjacent("office", "wet_lab")
        assert spatial.are_adjacent("wet_lab", "office")
        assert not spatial.are_adjacent("office", "tc_room")
        assert not spatial.are_adjacent("office", "chem_storage")

    def test_get_door(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        door = spatial.get_door("wet_lab", "tc_room")
        assert door is not None
        assert door.id == "d_wet_tc"
        assert door.requires_bsl_clearance == BSLLevel.BSL2
        # reverse direction
        assert spatial.get_door("tc_room", "wet_lab") is door

    def test_get_door_nonexistent(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        assert spatial.get_door("office", "chem_storage") is None

    def test_can_traverse_normal(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        can, reason = spatial.can_traverse("office", "wet_lab")
        assert can
        assert reason == ""

    def test_can_traverse_bsl_blocked(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        can, reason = spatial.can_traverse("wet_lab", "tc_room", BSLLevel.BSL1)
        assert not can
        assert "BSL-2 clearance" in reason

    def test_can_traverse_bsl_allowed(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        can, reason = spatial.can_traverse("wet_lab", "tc_room", BSLLevel.BSL2)
        assert can

    def test_can_traverse_locked(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        # Lock a door
        door = spatial.get_door("office", "wet_lab")
        assert door is not None
        door.locked = True
        can, reason = spatial.can_traverse("office", "wet_lab")
        assert not can
        assert "locked" in reason

    def test_can_traverse_closed(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        door = spatial.get_door("office", "wet_lab")
        assert door is not None
        door.is_open = False
        can, reason = spatial.can_traverse("office", "wet_lab")
        assert not can
        assert "closed" in reason

    def test_can_traverse_not_adjacent(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        can, reason = spatial.can_traverse("office", "chem_storage")
        assert not can
        assert "not adjacent" in reason

    def test_find_path_simple(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        path = spatial.find_path("office", "wet_lab")
        assert path == ["office", "wet_lab"]

    def test_find_path_multi_hop(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        path = spatial.find_path("office", "chem_storage")
        assert path == ["office", "wet_lab", "cold_room", "chem_storage"]

    def test_find_path_same_room(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        path = spatial.find_path("wet_lab", "wet_lab")
        assert path == ["wet_lab"]

    def test_find_path_bsl_blocked(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        # BSL-1 person can't reach TC room
        path = spatial.find_path("office", "tc_room", BSLLevel.BSL1)
        assert path is None

    def test_find_path_bsl_allowed(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        path = spatial.find_path("office", "tc_room", BSLLevel.BSL2)
        assert path == ["office", "wet_lab", "tc_room"]

    def test_find_path_nonexistent_room(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        path = spatial.find_path("office", "nonexistent")
        assert path is None

    def test_reachable_rooms_bsl1(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        reachable = spatial.reachable_rooms("office", BSLLevel.BSL1)
        assert reachable == {"office", "wet_lab", "cold_room", "chem_storage"}

    def test_reachable_rooms_bsl2(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        reachable = spatial.reachable_rooms("office", BSLLevel.BSL2)
        assert reachable == {"office", "wet_lab", "tc_room", "cold_room", "chem_storage"}

    def test_reachable_rooms_isolated(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        reachable = spatial.reachable_rooms("nonexistent")
        assert reachable == set()

    def test_room_distance(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        assert spatial.room_distance("office", "office") == 0
        assert spatial.room_distance("office", "wet_lab") == 1
        assert spatial.room_distance("office", "chem_storage") == 3
        assert spatial.room_distance("office", "tc_room") == 2

    def test_room_distance_unreachable(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        assert spatial.room_distance("office", "nonexistent") is None

    def test_find_path_locked_bypass(self, lab_world: WorldState) -> None:
        spatial = SpatialIndex(lab_world)
        door = spatial.get_door("office", "wet_lab")
        assert door is not None
        door.locked = True
        # Without ignore_locks: can't pass
        path = spatial.find_path("office", "cold_room")
        assert path is None
        # With ignore_locks: can pass
        path = spatial.find_path("office", "cold_room", ignore_locks=True)
        assert path == ["office", "wet_lab", "cold_room"]
