"""Equipment type catalog loader and default-definition registry."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError

from labverse.core.types import CertificationType
from labverse.models.equipment import EquipmentCommand, EquipmentParameter


class EquipmentCatalogError(ValueError):
    """Raised when an equipment catalog is invalid."""


class EquipmentDefinition(BaseModel):
    """Catalog definition for an equipment type."""

    equipment_type: str
    description: str = ""
    state_machine: str | None = None
    commands: dict[str, EquipmentCommand] = Field(default_factory=dict)
    properties: dict[str, EquipmentParameter] = Field(default_factory=dict)
    required_certifications: list[CertificationType] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class EquipmentCatalog:
    """Lookup table of equipment type definitions."""

    def __init__(self, definitions: dict[str, EquipmentDefinition] | None = None) -> None:
        self._definitions = definitions or {}

    @property
    def definitions(self) -> dict[str, EquipmentDefinition]:
        return dict(self._definitions)

    def get_definition(self, equipment_type: str) -> EquipmentDefinition | None:
        return self._definitions.get(equipment_type)

    def add_definition(self, definition: EquipmentDefinition) -> None:
        if definition.equipment_type in self._definitions:
            raise EquipmentCatalogError(
                f"duplicate equipment_type in catalog: {definition.equipment_type}"
            )
        self._definitions[definition.equipment_type] = definition


def load_equipment_catalog(catalog_dir: str | Path) -> EquipmentCatalog:
    """Load all equipment type definitions from a directory of YAML files."""
    directory = Path(catalog_dir)
    if not directory.exists():
        raise EquipmentCatalogError(f"equipment catalog directory not found: {directory}")
    if not directory.is_dir():
        raise EquipmentCatalogError(f"equipment catalog path is not a directory: {directory}")

    paths = sorted(directory.glob("*.yaml"))
    signature = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    definitions = _load_equipment_catalog_cached(str(directory.resolve()), signature)
    return EquipmentCatalog(
        {
            definition.equipment_type: EquipmentDefinition.model_validate(
                definition.model_dump(mode="python", round_trip=True),
            )
            for definition in definitions
        }
    )


@lru_cache(maxsize=32)
def _load_equipment_catalog_cached(
    resolved_dir: str,
    signature: tuple[tuple[str, int, int], ...],
) -> tuple[EquipmentDefinition, ...]:
    """Parse equipment YAML once per catalog revision.

    Paper audits repeatedly materialize the same frozen worlds. Caching avoids
    repeated YAML scanner churn while the file-stat signature still invalidates
    ordinary edits during development.
    """
    directory = Path(resolved_dir)
    del signature

    catalog = EquipmentCatalog()
    for path in sorted(directory.glob("*.yaml")):
        for definition in _load_catalog_file(path):
            catalog.add_definition(definition)
    return tuple(catalog.definitions.values())


def _load_catalog_file(path: Path) -> list[EquipmentDefinition]:
    try:
        raw_data = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=yaml.SafeLoader,
        )
    except yaml.YAMLError as exc:
        raise EquipmentCatalogError(f"invalid YAML in equipment catalog {path}: {exc}") from exc

    if raw_data is None:
        return []
    if not isinstance(raw_data, dict):
        raise EquipmentCatalogError(f"catalog file must be a mapping: {path}")

    raw_definitions = raw_data.get("equipment_types", [])
    if not isinstance(raw_definitions, list):
        raise EquipmentCatalogError(f"equipment_types must be a list in catalog {path}")

    definitions: list[EquipmentDefinition] = []
    for raw_definition in raw_definitions:
        try:
            definitions.append(EquipmentDefinition.model_validate(raw_definition))
        except ValidationError as exc:
            messages = "; ".join(error["msg"] for error in exc.errors())
            raise EquipmentCatalogError(f"invalid equipment catalog {path}: {messages}") from exc
    return definitions


__all__ = [
    "EquipmentCatalog",
    "EquipmentCatalogError",
    "EquipmentDefinition",
    "load_equipment_catalog",
]
