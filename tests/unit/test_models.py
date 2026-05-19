"""Comprehensive tests for all domain models."""

from datetime import UTC, date, datetime, timedelta

from labverse.core.types import (
    ActionCategory,
    BSLLevel,
    CertificationType,
    EquipmentStatus,
    HazardPictogram,
    MaterialPhase,
    PersonRole,
    PPEType,
    RoomType,
    SignalWord,
    WasteCategory,
)
from labverse.models.action import Action, ActionResult, ActionSpec, AvailableAction
from labverse.models.equipment import Equipment, EquipmentCommand, EquipmentParameter
from labverse.models.material import GHSData, Material
from labverse.models.person import Certification, Person
from labverse.models.protocol import Protocol, ProtocolStep
from labverse.models.safety import (
    BSLRequirements,
    SafetyCheckResult,
    SafetyVerdict,
)
from labverse.models.sample import CustodyRecord, Sample
from labverse.models.space import Door, LabLayout, Position, Room, Zone

# ── Helpers ────────────────────────────────────────────────────────────────

NOW = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)


def _room(id: str = "room_1", **kw) -> Room:
    defaults = dict(name="Test Room", room_type=RoomType.WET_LAB)
    return Room(id=id, **(defaults | kw))


def _equipment(id: str = "eq_1", **kw) -> Equipment:
    defaults = dict(name="Test Device", equipment_type="centrifuge", room_id="room_1")
    return Equipment(id=id, **(defaults | kw))


def _material(id: str = "mat_1", **kw) -> Material:
    defaults = dict(name="Test Reagent", room_id="room_1")
    return Material(id=id, **(defaults | kw))


def _person(id: str = "person_1", **kw) -> Person:
    defaults = dict(name="Test Person", role=PersonRole.GRAD_STUDENT)
    return Person(id=id, **(defaults | kw))


def _sample(id: str = "sample_1", **kw) -> Sample:
    defaults = dict(
        name="Test Sample",
        sample_type="cell_culture",
        room_id="room_1",
        created_at=NOW,
        created_by="person_1",
    )
    return Sample(id=id, **(defaults | kw))


# ── Space Models ───────────────────────────────────────────────────────────


class TestPosition:
    def test_defaults(self) -> None:
        p = Position()
        assert p.x == 0.0 and p.y == 0.0

    def test_values(self) -> None:
        p = Position(x=3.5, y=7.2)
        assert p.x == 3.5

    def test_round_trip(self) -> None:
        p = Position(x=1.0, y=2.0)
        assert Position.model_validate(p.model_dump()) == p


class TestDoor:
    def test_basic_door(self) -> None:
        d = Door(id="d1", connects=("room_a", "room_b"))
        assert d.connects == ("room_a", "room_b")
        assert d.is_open is True
        assert d.locked is False
        assert d.requires_bsl_clearance is None

    def test_bsl_gated_door(self) -> None:
        d = Door(id="d1", connects=("a", "b"), requires_bsl_clearance=BSLLevel.BSL2)
        assert d.requires_bsl_clearance == BSLLevel.BSL2

    def test_round_trip(self) -> None:
        d = Door(id="d1", connects=("a", "b"), locked=True)
        assert Door.model_validate(d.model_dump()) == d


class TestZone:
    def test_defaults(self) -> None:
        z = Zone(id="z1", name="Bench 1")
        assert z.zone_type == "bench"
        assert z.capacity == 1

    def test_custom_zone(self) -> None:
        z = Zone(id="z1", name="Fume Hood", zone_type="fume_hood_area", capacity=1, width=1.5)
        assert z.zone_type == "fume_hood_area"
        assert z.width == 1.5


class TestRoom:
    def test_defaults(self) -> None:
        r = _room()
        assert r.bsl_level == BSLLevel.BSL1
        assert r.temperature_celsius == 22.0
        assert r.max_occupancy == 10
        assert r.zones == []
        assert r.equipment_ids == []

    def test_cold_room(self) -> None:
        r = _room(room_type=RoomType.COLD_ROOM, temperature_celsius=4.0)
        assert r.temperature_celsius == 4.0
        assert r.room_type == RoomType.COLD_ROOM

    def test_with_zones(self) -> None:
        z = Zone(id="z1", name="Bench 1")
        r = _room(zones=[z])
        assert len(r.zones) == 1
        assert r.zones[0].id == "z1"

    def test_round_trip(self) -> None:
        z = Zone(id="z1", name="B1")
        r = _room(zones=[z], equipment_ids=["eq_1"], bsl_level=BSLLevel.BSL2)
        restored = Room.model_validate(r.model_dump())
        assert restored == r
        assert restored.bsl_level == BSLLevel.BSL2


class TestLabLayout:
    def test_basic_layout(self) -> None:
        r1 = _room("r1", name="Lab A")
        r2 = _room("r2", name="Lab B", room_type=RoomType.OFFICE)
        d = Door(id="d1", connects=("r1", "r2"))
        layout = LabLayout(
            id="lab_1",
            name="Test Lab",
            rooms={"r1": r1, "r2": r2},
            doors=[d],
            default_entry_room="r1",
        )
        assert len(layout.rooms) == 2
        assert len(layout.doors) == 1
        assert layout.default_entry_room == "r1"

    def test_round_trip(self) -> None:
        r = _room("r1")
        layout = LabLayout(id="l1", name="L", rooms={"r1": r}, default_entry_room="r1")
        assert LabLayout.model_validate(layout.model_dump()) == layout


# ── Equipment Models ───────────────────────────────────────────────────────


class TestEquipmentParameter:
    def test_basic(self) -> None:
        p = EquipmentParameter(
            name="speed_rpm",
            param_type="int",
            unit="rpm",
            min_value=100,
            max_value=15000,
        )
        assert p.unit == "rpm"
        assert p.min_value == 100

    def test_enum_param(self) -> None:
        p = EquipmentParameter(
            name="rotor",
            param_type="enum",
            allowed_values=["fixed_angle", "swinging_bucket"],
        )
        assert p.allowed_values is not None
        assert len(p.allowed_values) == 2


class TestEquipment:
    def test_defaults(self) -> None:
        eq = _equipment()
        assert eq.status == EquipmentStatus.OFF
        assert eq.max_concurrent_users == 1
        assert eq.current_users == []
        assert eq.is_available is False  # OFF is not available

    def test_available_when_idle(self) -> None:
        eq = _equipment(status=EquipmentStatus.IDLE)
        assert eq.is_available is True

    def test_not_available_when_running(self) -> None:
        eq = _equipment(status=EquipmentStatus.RUNNING)
        assert eq.is_available is False
        assert eq.is_running is True

    def test_not_available_when_full(self) -> None:
        eq = _equipment(status=EquipmentStatus.IDLE, current_users=["p1"], max_concurrent_users=1)
        assert eq.is_available is False

    def test_with_certifications(self) -> None:
        eq = _equipment(required_certifications=[CertificationType.AUTOCLAVE_OPERATION])
        assert CertificationType.AUTOCLAVE_OPERATION in eq.required_certifications

    def test_with_commands(self) -> None:
        cmd = EquipmentCommand(name="spin", description="Start centrifuge", duration_seconds=600)
        eq = _equipment(commands={"spin": cmd})
        assert eq.commands["spin"].duration_seconds == 600

    def test_round_trip(self) -> None:
        eq = _equipment(
            status=EquipmentStatus.RUNNING,
            running_command="spin",
            running_command_started_at=NOW,
            running_command_duration=timedelta(seconds=600),
        )
        restored = Equipment.model_validate(eq.model_dump())
        assert restored.running_command == "spin"
        assert restored.running_command_duration == timedelta(seconds=600)


# ── Material Models ────────────────────────────────────────────────────────


class TestGHSData:
    def test_defaults(self) -> None:
        g = GHSData()
        assert g.pictograms == []
        assert g.signal_word == SignalWord.NONE

    def test_hazardous(self) -> None:
        g = GHSData(
            pictograms=[HazardPictogram.GHS02_FLAMMABLE],
            signal_word=SignalWord.DANGER,
            h_statements=["H225", "H319"],
        )
        assert len(g.h_statements) == 2


class TestMaterial:
    def test_defaults(self) -> None:
        m = _material()
        assert m.phase == MaterialPhase.LIQUID
        assert m.quantity == 0.0
        assert not m.is_hazardous
        assert not m.is_expired

    def test_hazardous_material(self) -> None:
        ghs = GHSData(pictograms=[HazardPictogram.GHS05_CORROSIVE])
        m = _material(ghs=ghs)
        assert m.is_hazardous

    def test_expired_material(self) -> None:
        m = _material(expiry_date=date(2020, 1, 1))
        assert m.is_expired

    def test_expired_material_on_explicit_date(self) -> None:
        m = _material(expiry_date=date(2026, 1, 10))
        assert m.is_expired_on(date(2026, 1, 11))
        assert not m.is_expired_on(date(2026, 1, 10))

    def test_not_expired(self) -> None:
        m = _material(expiry_date=date(2099, 12, 31))
        assert not m.is_expired

    def test_no_expiry(self) -> None:
        m = _material()
        assert not m.is_expired

    def test_with_ppe(self) -> None:
        m = _material(required_ppe=[PPEType.NITRILE_GLOVES, PPEType.SAFETY_GLASSES])
        assert len(m.required_ppe) == 2

    def test_round_trip(self) -> None:
        ghs = GHSData(
            pictograms=[HazardPictogram.GHS02_FLAMMABLE, HazardPictogram.GHS07_IRRITANT],
            signal_word=SignalWord.DANGER,
        )
        m = _material(
            cas_number="64-17-5",
            formula="C2H5OH",
            ghs=ghs,
            waste_category=WasteCategory.CHEMICAL,
            incompatible_with=["oxidizers"],
        )
        restored = Material.model_validate(m.model_dump())
        assert restored == m
        assert len(restored.ghs.pictograms) == 2


# ── Sample Models ──────────────────────────────────────────────────────────


class TestCustodyRecord:
    def test_basic(self) -> None:
        cr = CustodyRecord(
            timestamp=NOW,
            person_id="p1",
            action="created",
            location_room_id="room_1",
        )
        assert cr.action == "created"
        assert cr.location_zone_id is None


class TestSample:
    def test_defaults(self) -> None:
        s = _sample()
        assert s.status == "active"
        assert s.is_active
        assert s.container_type == "microcentrifuge_tube"
        assert s.chain_of_custody == []
        assert s.last_custody_record is None

    def test_with_custody(self) -> None:
        cr = CustodyRecord(timestamp=NOW, person_id="p1", action="created", location_room_id="r1")
        s = _sample(chain_of_custody=[cr])
        assert s.last_custody_record is not None
        assert s.last_custody_record.action == "created"

    def test_derived_sample(self) -> None:
        s = _sample(derived_from=["parent_1", "parent_2"])
        assert len(s.derived_from) == 2

    def test_inactive(self) -> None:
        s = _sample(status="disposed")
        assert not s.is_active

    def test_round_trip(self) -> None:
        cr = CustodyRecord(timestamp=NOW, person_id="p1", action="stored", location_room_id="r1")
        s = _sample(
            volume_ul=50.0,
            bsl_level=BSLLevel.BSL2,
            chain_of_custody=[cr],
            metadata={"passage": "5"},
        )
        restored = Sample.model_validate(s.model_dump())
        assert restored == s
        assert restored.bsl_level == BSLLevel.BSL2


# ── Person Models ──────────────────────────────────────────────────────────


class TestCertification:
    def test_basic(self) -> None:
        c = Certification(cert_type=CertificationType.BSL2_TRAINING, granted_date="2024-01-15")
        assert c.cert_type == CertificationType.BSL2_TRAINING
        assert c.expiry_date is None


class TestPerson:
    def test_defaults(self) -> None:
        p = _person()
        assert p.role == PersonRole.GRAD_STUDENT
        assert p.certifications == []
        assert p.current_ppe == set()
        assert p.carrying == []

    def test_has_certification(self) -> None:
        cert = Certification(cert_type=CertificationType.BSL2_TRAINING, granted_date="2024-01-01")
        p = _person(certifications=[cert])
        assert p.has_certification(CertificationType.BSL2_TRAINING)
        assert not p.has_certification(CertificationType.BSL3_TRAINING)

    def test_certification_types(self) -> None:
        certs = [
            Certification(cert_type=CertificationType.BSL2_TRAINING, granted_date="2024-01-01"),
            Certification(cert_type=CertificationType.CHEMICAL_SAFETY, granted_date="2024-01-01"),
        ]
        p = _person(certifications=certs)
        assert p.certification_types == {
            CertificationType.BSL2_TRAINING,
            CertificationType.CHEMICAL_SAFETY,
        }

    def test_has_ppe(self) -> None:
        p = _person(current_ppe={PPEType.LAB_COAT, PPEType.NITRILE_GLOVES})
        assert p.has_ppe(PPEType.LAB_COAT)
        assert not p.has_ppe(PPEType.FACE_SHIELD)

    def test_round_trip(self) -> None:
        cert = Certification(cert_type=CertificationType.BSL2_TRAINING, granted_date="2024-01-01")
        p = _person(
            role=PersonRole.PI,
            certifications=[cert],
            current_ppe={PPEType.LAB_COAT, PPEType.SAFETY_GLASSES},
            current_room_id="room_1",
            carrying=["sample_1"],
        )
        restored = Person.model_validate(p.model_dump())
        assert restored.role == PersonRole.PI
        assert restored.has_certification(CertificationType.BSL2_TRAINING)
        assert restored.current_ppe == {PPEType.LAB_COAT, PPEType.SAFETY_GLASSES}


# ── Protocol Models ────────────────────────────────────────────────────────


class TestProtocolStep:
    def test_basic(self) -> None:
        s = ProtocolStep(id="s1", order=1, description="Add reagent", action_type="handle_material")
        assert s.duration_seconds is None
        assert s.substeps == []

    def test_with_params(self) -> None:
        s = ProtocolStep(
            id="s1",
            order=1,
            description="Centrifuge",
            action_type="use_equipment",
            equipment_type="centrifuge",
            parameters={"speed_rpm": 12000, "duration_seconds": 600},
            duration_seconds=600,
        )
        assert s.parameters["speed_rpm"] == 12000


class TestProtocol:
    def test_basic(self) -> None:
        p = Protocol(id="p1", name="PCR", estimated_duration_minutes=120)
        assert p.total_steps == 0

    def test_with_steps(self) -> None:
        steps = [
            ProtocolStep(id="s1", order=1, description="Step 1", action_type="handle_material"),
            ProtocolStep(id="s2", order=2, description="Step 2", action_type="use_equipment"),
        ]
        p = Protocol(id="p1", name="PCR", steps=steps, estimated_duration_minutes=120)
        assert p.total_steps == 2

    def test_nested_steps(self) -> None:
        substeps = [
            ProtocolStep(id="s1a", order=1, description="Sub A", action_type="handle_material"),
            ProtocolStep(id="s1b", order=2, description="Sub B", action_type="handle_material"),
        ]
        steps = [
            ProtocolStep(
                id="s1",
                order=1,
                description="Step 1",
                action_type="use_equipment",
                substeps=substeps,
            ),
            ProtocolStep(id="s2", order=2, description="Step 2", action_type="observe"),
        ]
        p = Protocol(id="p1", name="Complex", steps=steps, estimated_duration_minutes=60)
        assert p.total_steps == 4

    def test_round_trip(self) -> None:
        steps = [ProtocolStep(id="s1", order=1, description="Do it", action_type="safety")]
        p = Protocol(
            id="p1",
            name="Test",
            bsl_level_required=BSLLevel.BSL2,
            steps=steps,
            required_certifications=[CertificationType.BSL2_TRAINING],
            safety_notes=["Wear gloves"],
            estimated_duration_minutes=30,
        )
        restored = Protocol.model_validate(p.model_dump())
        assert restored == p


# ── Action Models ──────────────────────────────────────────────────────────


class TestAction:
    def test_basic(self) -> None:
        a = Action(
            id="a1",
            agent_id="p1",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="room_2",
        )
        assert a.verb == "move_to"
        assert a.timestamp is None

    def test_with_parameters(self) -> None:
        a = Action(
            id="a1",
            agent_id="p1",
            category=ActionCategory.USE_EQUIPMENT,
            verb="start_centrifuge",
            target_id="eq_1",
            parameters={"speed_rpm": 12000, "duration_seconds": 600},
        )
        assert a.parameters["speed_rpm"] == 12000


class TestActionResult:
    def test_success(self) -> None:
        r = ActionResult(action_id="a1", success=True, outcome_description="Moved to room 2")
        assert r.success
        assert r.safety_violations == []
        assert r.blocked_reason is None

    def test_blocked(self) -> None:
        r = ActionResult(
            action_id="a1",
            success=False,
            outcome_description="Blocked",
            blocked_reason="Missing BSL-2 certification",
            safety_violations=["bsl_access"],
        )
        assert not r.success
        assert len(r.safety_violations) == 1


class TestAvailableAction:
    def test_basic(self) -> None:
        a = AvailableAction(
            action_id="move_to:room_2",
            verb="move_to",
            category=ActionCategory.MOVE,
            description="Move to adjacent room",
        )
        assert a.action_id == "move_to:room_2"
        assert a.verb == "move_to"


class TestActionSpec:
    def test_empty(self) -> None:
        spec = ActionSpec()
        assert spec.available_actions == []

    def test_with_actions(self) -> None:
        actions = [
            AvailableAction(
                action_id="move_to:room_2",
                verb="move_to",
                category=ActionCategory.MOVE,
            ),
            AvailableAction(
                action_id="don_ppe:self",
                verb="don_ppe",
                category=ActionCategory.SAFETY,
            ),
        ]
        spec = ActionSpec(available_actions=actions, context="You are in the wet lab.")
        assert len(spec.available_actions) == 2


# ── Safety Models ──────────────────────────────────────────────────────────


class TestSafetyVerdict:
    def test_values(self) -> None:
        assert SafetyVerdict.ALLOW == "allow"
        assert SafetyVerdict.WARN == "warn"
        assert SafetyVerdict.BLOCK == "block"


class TestSafetyCheckResult:
    def test_allow(self) -> None:
        r = SafetyCheckResult(verdict=SafetyVerdict.ALLOW)
        assert r.verdict == SafetyVerdict.ALLOW

    def test_block_with_remediation(self) -> None:
        r = SafetyCheckResult(
            verdict=SafetyVerdict.BLOCK,
            constraint_name="ppe_requirement",
            message="Missing nitrile gloves",
            remediation="Don nitrile gloves before handling chemicals",
        )
        assert r.remediation is not None


class TestBSLRequirements:
    def test_bsl1(self) -> None:
        r = BSLRequirements(
            level=BSLLevel.BSL1,
            min_ppe={PPEType.LAB_COAT, PPEType.SAFETY_GLASSES, PPEType.CLOSED_TOE_SHOES},
        )
        assert not r.requires_biosafety_cabinet
        assert not r.requires_double_door_entry

    def test_bsl3(self) -> None:
        r = BSLRequirements(
            level=BSLLevel.BSL3,
            min_ppe={
                PPEType.LAB_COAT,
                PPEType.SAFETY_GLASSES,
                PPEType.NITRILE_GLOVES,
                PPEType.RESPIRATOR,
            },
            requires_biosafety_cabinet=True,
            required_certifications=[CertificationType.BSL3_TRAINING],
            requires_directional_airflow=True,
            requires_double_door_entry=True,
            decontamination_required_on_exit=True,
        )
        assert r.requires_biosafety_cabinet
        assert r.decontamination_required_on_exit

    def test_round_trip(self) -> None:
        r = BSLRequirements(
            level=BSLLevel.BSL2,
            min_ppe={PPEType.NITRILE_GLOVES},
            requires_biosafety_cabinet=True,
        )
        restored = BSLRequirements.model_validate(r.model_dump())
        assert restored == r


# ── Cross-model serialization ──────────────────────────────────────────────


class TestCrossModelSerialization:
    """Verify that complex nested models serialize and deserialize correctly."""

    def test_full_lab_layout_round_trip(self) -> None:
        z1 = Zone(id="z1", name="Bench 1", zone_type="bench", capacity=2)
        z2 = Zone(id="z2", name="Fume Hood", zone_type="fume_hood_area")
        r1 = _room("r1", name="Wet Lab", zones=[z1, z2], equipment_ids=["eq_1", "eq_2"])
        r2 = _room("r2", name="Office", room_type=RoomType.OFFICE)
        d = Door(id="d1", connects=("r1", "r2"))
        layout = LabLayout(
            id="l1",
            name="Test",
            rooms={"r1": r1, "r2": r2},
            doors=[d],
            default_entry_room="r1",
        )

        data = layout.model_dump()
        restored = LabLayout.model_validate(data)
        assert restored == layout
        assert len(restored.rooms["r1"].zones) == 2

    def test_json_round_trip(self) -> None:
        """Verify JSON string serialization works."""
        eq = _equipment(status=EquipmentStatus.RUNNING)
        json_str = eq.model_dump_json()
        restored = Equipment.model_validate_json(json_str)
        assert restored == eq
