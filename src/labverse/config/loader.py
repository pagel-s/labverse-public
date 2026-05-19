"""YAML configuration loader and validation helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from labverse.config.builder import build_world
from labverse.config.schema import ConfigError, LabConfig
from labverse.equipment.catalog import (
    EquipmentCatalog,
    EquipmentCatalogError,
    load_equipment_catalog,
)
from labverse.world.state import WorldState


def load_lab_config(path: str | Path) -> LabConfig:
    """Load and validate a lab configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"config path is not a file: {config_path}")
    stat = config_path.stat()
    cached_config = _load_lab_config_cached(
        str(config_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )
    return LabConfig.model_validate(
        cached_config.model_dump(mode="python", round_trip=True),
    )


@lru_cache(maxsize=128)
def _load_lab_config_cached(
    resolved_path: str,
    mtime_ns: int,
    size: int,
) -> LabConfig:
    """Parse and validate a config file once per file revision.

    Paper prompt probes repeatedly materialize the same frozen lab configs.
    Caching the validated config avoids repeated YAML scanner churn while still
    invalidating on ordinary file edits via the stat tuple.
    """
    del mtime_ns, size
    config_path = Path(resolved_path)
    try:
        raw_data = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=yaml.SafeLoader,
        )
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if raw_data is None:
        raise ConfigError(f"config file is empty: {config_path}")
    if not isinstance(raw_data, dict):
        raise ConfigError(f"top-level config must be a mapping: {config_path}")

    try:
        return LabConfig.model_validate(raw_data)
    except ValidationError as exc:
        messages = "; ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"invalid config {config_path}: {messages}") from exc


def load_world(
    path: str | Path,
    *,
    equipment_catalog_dir: str | Path | None = None,
) -> WorldState:
    """Load a YAML config and build a runtime `WorldState`."""
    config_path = Path(path)
    equipment_catalog = _load_adjacent_catalog(config_path, equipment_catalog_dir)
    return build_world(load_lab_config(config_path), equipment_catalog)


def _load_adjacent_catalog(
    config_path: Path,
    equipment_catalog_dir: str | Path | None,
) -> EquipmentCatalog | None:
    if equipment_catalog_dir is not None:
        try:
            return load_equipment_catalog(equipment_catalog_dir)
        except EquipmentCatalogError as exc:
            raise ConfigError(str(exc)) from exc

    inferred_dir = config_path.parent.parent / "equipment"
    if not inferred_dir.exists():
        return None

    try:
        return load_equipment_catalog(inferred_dir)
    except EquipmentCatalogError as exc:
        raise ConfigError(str(exc)) from exc
