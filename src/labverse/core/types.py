"""Shared enumerations and type aliases used across the entire LabVerse system."""

from enum import Enum, StrEnum


class RoomType(StrEnum):
    """Types of rooms found in a laboratory facility."""

    WET_LAB = "wet_lab"
    DRY_LAB = "dry_lab"
    TISSUE_CULTURE = "tissue_culture"
    COLD_ROOM = "cold_room"
    FREEZER_ROOM = "freezer_room"
    CHEMICAL_STORAGE = "chemical_storage"
    AUTOCLAVE_ROOM = "autoclave_room"
    SHARED_EQUIPMENT = "shared_equipment"
    MEDIA_PREP = "media_prep"
    OFFICE = "office"
    SERVER_ROOM = "server_room"
    HALLWAY = "hallway"


class BSLLevel(int, Enum):
    """Biosafety containment levels."""

    BSL1 = 1
    BSL2 = 2
    BSL3 = 3
    BSL4 = 4


class EquipmentStatus(StrEnum):
    """Operational states for lab equipment."""

    OFF = "off"
    IDLE = "idle"
    WARMING_UP = "warming_up"
    READY = "ready"
    RUNNING = "running"
    COOLING_DOWN = "cooling_down"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    RESERVED = "reserved"


class MaterialPhase(StrEnum):
    """Physical phase of a material."""

    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    SUSPENSION = "suspension"
    POWDER = "powder"
    GEL = "gel"


class HazardPictogram(StrEnum):
    """GHS hazard pictogram codes."""

    GHS01_EXPLOSIVE = "GHS01"
    GHS02_FLAMMABLE = "GHS02"
    GHS03_OXIDIZER = "GHS03"
    GHS04_COMPRESSED_GAS = "GHS04"
    GHS05_CORROSIVE = "GHS05"
    GHS06_TOXIC = "GHS06"
    GHS07_IRRITANT = "GHS07"
    GHS08_HEALTH_HAZARD = "GHS08"
    GHS09_ENVIRONMENT = "GHS09"


class SignalWord(StrEnum):
    """GHS signal words indicating hazard severity."""

    DANGER = "danger"
    WARNING = "warning"
    NONE = "none"


class WasteCategory(StrEnum):
    """Laboratory waste stream classifications."""

    GENERAL = "general"
    CHEMICAL = "chemical"
    BIOHAZARD = "biohazard"
    SHARPS = "sharps"
    RADIOACTIVE = "radioactive"


class PPEType(StrEnum):
    """Personal protective equipment types."""

    LAB_COAT = "lab_coat"
    SAFETY_GLASSES = "safety_glasses"
    NITRILE_GLOVES = "nitrile_gloves"
    FACE_SHIELD = "face_shield"
    FUME_HOOD = "fume_hood"
    RESPIRATOR = "respirator"
    FULL_BODY_SUIT = "full_body_suit"
    CHEMICAL_APRON = "chemical_apron"
    CLOSED_TOE_SHOES = "closed_toe_shoes"


class PersonRole(StrEnum):
    """Roles of lab personnel."""

    PI = "pi"
    POSTDOC = "postdoc"
    GRAD_STUDENT = "grad_student"
    UNDERGRAD = "undergrad"
    TECHNICIAN = "technician"
    LAB_MANAGER = "lab_manager"
    SAFETY_OFFICER = "safety_officer"


class CertificationType(StrEnum):
    """Training and certification types."""

    BSL2_TRAINING = "bsl2_training"
    BSL3_TRAINING = "bsl3_training"
    BSL4_TRAINING = "bsl4_training"
    CHEMICAL_SAFETY = "chemical_safety"
    RADIATION_SAFETY = "radiation_safety"
    AUTOCLAVE_OPERATION = "autoclave_operation"
    LASER_SAFETY = "laser_safety"
    CRYOGEN_HANDLING = "cryogen_handling"
    EQUIPMENT_SPECIFIC = "equipment_specific"


class ActionCategory(StrEnum):
    """Categories of agent actions."""

    MOVE = "move"
    USE_EQUIPMENT = "use_equipment"
    HANDLE_MATERIAL = "handle_material"
    HANDLE_SAMPLE = "handle_sample"
    OBSERVE = "observe"
    COMMUNICATE = "communicate"
    SAFETY = "safety"
    ADMINISTRATIVE = "administrative"
