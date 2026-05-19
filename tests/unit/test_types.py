"""Tests for core type enumerations."""

import pytest

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


class TestEnumValues:
    """Verify enum string/int values match expectations for serialization stability."""

    def test_room_types(self) -> None:
        assert RoomType.WET_LAB == "wet_lab"
        assert RoomType.TISSUE_CULTURE == "tissue_culture"
        assert RoomType.OFFICE == "office"
        assert len(RoomType) == 12

    def test_bsl_levels(self) -> None:
        assert BSLLevel.BSL1 == 1
        assert BSLLevel.BSL2 == 2
        assert BSLLevel.BSL3 == 3
        assert BSLLevel.BSL4 == 4
        assert BSLLevel.BSL2 > BSLLevel.BSL1

    def test_equipment_status(self) -> None:
        assert EquipmentStatus.OFF == "off"
        assert EquipmentStatus.RUNNING == "running"
        assert EquipmentStatus.ERROR == "error"
        assert len(EquipmentStatus) == 9

    def test_hazard_pictograms(self) -> None:
        assert HazardPictogram.GHS02_FLAMMABLE == "GHS02"
        assert HazardPictogram.GHS05_CORROSIVE == "GHS05"
        assert len(HazardPictogram) == 9

    def test_ppe_types(self) -> None:
        assert PPEType.LAB_COAT == "lab_coat"
        assert PPEType.NITRILE_GLOVES == "nitrile_gloves"

    def test_person_roles(self) -> None:
        assert PersonRole.PI == "pi"
        assert PersonRole.GRAD_STUDENT == "grad_student"

    def test_action_categories(self) -> None:
        assert ActionCategory.MOVE == "move"
        assert ActionCategory.USE_EQUIPMENT == "use_equipment"
        assert ActionCategory.SAFETY == "safety"


class TestEnumSerialization:
    """Verify enums serialize to/from strings correctly."""

    @pytest.mark.parametrize(
        "enum_cls,value",
        [
            (RoomType, "wet_lab"),
            (EquipmentStatus, "running"),
            (MaterialPhase, "liquid"),
            (PPEType, "nitrile_gloves"),
            (WasteCategory, "biohazard"),
            (SignalWord, "danger"),
            (ActionCategory, "use_equipment"),
            (CertificationType, "bsl2_training"),
        ],
    )
    def test_string_enum_round_trip(self, enum_cls: type, value: str) -> None:
        member = enum_cls(value)
        assert member.value == value
        assert str(member.value) == value

    def test_bsl_level_from_int(self) -> None:
        assert BSLLevel(2) == BSLLevel.BSL2

    def test_invalid_enum_value_raises(self) -> None:
        with pytest.raises(ValueError):
            RoomType("nonexistent_room")
