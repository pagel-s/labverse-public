"""Entity ID generation with optional prefixes for readability."""

import uuid


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix.

    Examples:
        generate_id("room") -> "room_a1b2c3d4"
        generate_id("eq") -> "eq_e5f6a7b8"
        generate_id() -> "a1b2c3d4"
    """
    short = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_{short}"
    return short
