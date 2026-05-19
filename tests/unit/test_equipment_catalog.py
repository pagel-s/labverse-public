"""Tests for equipment catalog loading and catalog-backed defaults."""

from __future__ import annotations

from pathlib import Path

from labverse.equipment import load_equipment_catalog
from labverse.models.equipment import EquipmentParameter

ROOT = Path(__file__).resolve().parents[2]


def test_load_equipment_catalog_directory() -> None:
    catalog = load_equipment_catalog(ROOT / "configs" / "equipment")

    definitions = catalog.definitions
    assert set(definitions) == {
        "autoclave",
        "biosafety_cabinet",
        "centrifuge",
        "incubator",
        "pcr_thermocycler",
        "pipette",
    }
    assert definitions["centrifuge"].state_machine == "centrifuge"
    assert "spin" in definitions["centrifuge"].commands


def test_catalog_definition_contains_parameter_models() -> None:
    catalog = load_equipment_catalog(ROOT / "configs" / "equipment")
    centrifuge = catalog.get_definition("centrifuge")

    assert centrifuge is not None
    speed = centrifuge.properties["speed_rpm"]
    assert isinstance(speed, EquipmentParameter)
    assert speed.unit == "rpm"
    assert speed.max_value == 15000
