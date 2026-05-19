"""Tests for ID generation."""

from labverse.core.ids import generate_id


class TestGenerateId:
    def test_basic_id(self) -> None:
        id_ = generate_id()
        assert len(id_) == 8
        assert id_.isalnum()

    def test_prefixed_id(self) -> None:
        id_ = generate_id("room")
        assert id_.startswith("room_")
        assert len(id_) == 13  # "room_" + 8 hex chars

    def test_uniqueness(self) -> None:
        ids = {generate_id("eq") for _ in range(1000)}
        assert len(ids) == 1000

    def test_empty_prefix(self) -> None:
        id_ = generate_id("")
        assert "_" not in id_
        assert len(id_) == 8
