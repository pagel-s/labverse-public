"""Tests for YAML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from labverse.config import ConfigError, load_lab_config, load_world
from labverse.models.equipment import EquipmentParameter

ROOT = Path(__file__).resolve().parents[2]


def test_load_minimal_world_from_yaml() -> None:
    world = load_world(ROOT / "configs" / "labs" / "minimal_test.yaml")

    assert world.lab.id == "minimal_test_lab"
    assert world.lab.default_entry_room == "office"
    assert world.entity_counts == {
        "rooms": 2,
        "doors": 1,
        "equipment": 1,
        "materials": 1,
        "samples": 0,
        "persons": 1,
    }
    assert world.get_room("wet_lab").equipment_ids == ["centrifuge_1"]
    assert world.get_room("wet_lab").material_ids == ["pbs_1x"]
    centrifuge = world.get_equipment("centrifuge_1")
    assert "spin" in centrifuge.commands
    assert centrifuge.commands["spin"].duration_seconds == 600


def test_loader_normalizes_equipment_properties() -> None:
    world = load_world(ROOT / "configs" / "labs" / "research_bsl2.yaml")
    centrifuge = world.get_equipment("centrifuge_1")

    max_rpm = centrifuge.properties["max_rpm"]
    assert isinstance(max_rpm, EquipmentParameter)
    assert max_rpm.current_value == 21130.0

    temperature_range = centrifuge.properties["temperature_range"]
    assert isinstance(temperature_range, EquipmentParameter)
    assert temperature_range.param_type == "range"
    assert temperature_range.min_value == -10.0
    assert temperature_range.max_value == 40.0

    rcf_g = centrifuge.properties["rcf_g"]
    assert isinstance(rcf_g, EquipmentParameter)
    assert rcf_g.current_value == 1000


def test_invalid_config_rejects_unknown_room_reference(tmp_path: Path) -> None:
    invalid_config = {
        "lab": {"id": "broken_lab", "name": "Broken Lab"},
        "rooms": [{"id": "office", "name": "Office", "room_type": "office"}],
        "equipment": [
            {
                "id": "centrifuge_1",
                "name": "Centrifuge",
                "equipment_type": "centrifuge",
                "room_id": "missing_room",
            }
        ],
    }
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(yaml.safe_dump(invalid_config), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown room 'missing_room'"):
        load_lab_config(config_path)
