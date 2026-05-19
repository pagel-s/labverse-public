"""Person models: lab members with roles, certifications, and state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labverse.core.types import CertificationType, PersonRole, PPEType


class Certification(BaseModel):
    """A training or certification held by a person."""

    cert_type: CertificationType
    granted_date: str
    expiry_date: str | None = None
    granted_by: str | None = None


class Person(BaseModel):
    """A lab member."""

    id: str
    name: str
    role: PersonRole
    certifications: list[Certification] = Field(default_factory=list)
    current_room_id: str | None = None
    current_zone_id: str | None = None
    current_ppe: set[PPEType] = Field(default_factory=set)
    carrying: list[str] = Field(default_factory=list)
    is_active: bool = True
    lab_group: str | None = None
    current_activity: str | None = None

    def has_certification(self, cert_type: CertificationType) -> bool:
        """Check if the person holds a specific certification."""
        return any(c.cert_type == cert_type for c in self.certifications)

    def has_ppe(self, ppe_type: PPEType) -> bool:
        """Check if the person is currently wearing specific PPE."""
        return ppe_type in self.current_ppe

    @property
    def certification_types(self) -> set[CertificationType]:
        return {c.cert_type for c in self.certifications}
