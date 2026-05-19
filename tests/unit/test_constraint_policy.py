"""Tests for per-scenario constraint enforcement overrides."""

from __future__ import annotations

import pytest

from labverse.models.safety import SafetyVerdict
from labverse.safety.constraints import (
    MaterialExpiryConstraint,
    SampleContaminationConstraint,
)
from labverse.safety.engine import SafetyEngine
from labverse.safety.policy import ConstraintPolicyError, parse_constraint_policy


class TestParseConstraintPolicy:
    def test_none_returns_empty_dict(self) -> None:
        assert parse_constraint_policy(None) == {}

    def test_empty_mapping_returns_empty_dict(self) -> None:
        assert parse_constraint_policy({}) == {}

    def test_parses_block_and_warn_verdicts(self) -> None:
        result = parse_constraint_policy(
            {"material_expiry": "block", "sample_contamination": "warn"}
        )
        assert result == {
            "material_expiry": SafetyVerdict.BLOCK,
            "sample_contamination": SafetyVerdict.WARN,
        }

    def test_accepts_mixed_case_and_whitespace(self) -> None:
        result = parse_constraint_policy({"material_expiry": "  BLOCK  "})
        assert result == {"material_expiry": SafetyVerdict.BLOCK}

    def test_accepts_safety_verdict_enum_values(self) -> None:
        result = parse_constraint_policy({"material_expiry": SafetyVerdict.BLOCK})
        assert result == {"material_expiry": SafetyVerdict.BLOCK}

    def test_rejects_unknown_verdict(self) -> None:
        with pytest.raises(ConstraintPolicyError, match="unknown verdict"):
            parse_constraint_policy({"material_expiry": "maybe"})

    def test_rejects_non_string_key(self) -> None:
        with pytest.raises(ConstraintPolicyError, match="non-empty strings"):
            parse_constraint_policy({42: "block"})  # type: ignore[dict-item]

    def test_rejects_empty_string_key(self) -> None:
        with pytest.raises(ConstraintPolicyError, match="non-empty strings"):
            parse_constraint_policy({"": "block"})

    def test_rejects_non_mapping_input(self) -> None:
        with pytest.raises(ConstraintPolicyError, match="must be a mapping"):
            parse_constraint_policy([("material_expiry", "block")])  # type: ignore[arg-type]


class TestMaterialExpiryEnforcementLevel:
    def test_default_enforcement_is_warn(self) -> None:
        constraint = MaterialExpiryConstraint()
        assert constraint._enforcement_level == SafetyVerdict.WARN

    def test_can_be_constructed_with_block_enforcement(self) -> None:
        constraint = MaterialExpiryConstraint(enforcement_level=SafetyVerdict.BLOCK)
        assert constraint._enforcement_level == SafetyVerdict.BLOCK


class TestSampleContaminationEnforcementLevel:
    def test_default_enforcement_is_warn(self) -> None:
        constraint = SampleContaminationConstraint()
        assert constraint._enforcement_level == SafetyVerdict.WARN

    def test_can_be_constructed_with_block_enforcement(self) -> None:
        constraint = SampleContaminationConstraint(enforcement_level=SafetyVerdict.BLOCK)
        assert constraint._enforcement_level == SafetyVerdict.BLOCK


class TestSafetyEngineOverrides:
    def test_default_engine_has_warn_level_tunable_constraints(self) -> None:
        engine = SafetyEngine()
        expiry = next(c for c in engine.constraints if isinstance(c, MaterialExpiryConstraint))
        contamination = next(
            c for c in engine.constraints if isinstance(c, SampleContaminationConstraint)
        )
        assert expiry._enforcement_level == SafetyVerdict.WARN
        assert contamination._enforcement_level == SafetyVerdict.WARN

    def test_engine_promotes_matching_constraints_via_overrides(self) -> None:
        engine = SafetyEngine(
            enforcement_overrides={
                "material_expiry": SafetyVerdict.BLOCK,
                "sample_contamination": SafetyVerdict.BLOCK,
            }
        )
        expiry = next(c for c in engine.constraints if isinstance(c, MaterialExpiryConstraint))
        contamination = next(
            c for c in engine.constraints if isinstance(c, SampleContaminationConstraint)
        )
        assert expiry._enforcement_level == SafetyVerdict.BLOCK
        assert contamination._enforcement_level == SafetyVerdict.BLOCK

    def test_unknown_override_keys_are_silently_ignored(self) -> None:
        # A typo'd constraint id should not explode engine construction — the
        # override simply finds no matching constraint type and is dropped.
        engine = SafetyEngine(
            enforcement_overrides={"definitely_not_a_real_constraint": SafetyVerdict.BLOCK}
        )
        expiry = next(c for c in engine.constraints if isinstance(c, MaterialExpiryConstraint))
        assert expiry._enforcement_level == SafetyVerdict.WARN

    def test_override_does_not_affect_constraints_without_kwarg(self) -> None:
        # ppe_requirement does not advertise an enforcement_level kwarg, so it
        # should be constructed with no-args even when an override is supplied.
        engine = SafetyEngine(enforcement_overrides={"ppe_requirement": SafetyVerdict.WARN})
        # It still appears in the pipeline with its default behavior.
        ppe_ids = [c.constraint_id for c in engine.constraints]
        assert "ppe_requirement" in ppe_ids
