import type { Room } from "../api/client";

export type RoomPattern = "wet_grid" | "sterile_grid" | "cold_racks" | "office_mesh" | "hazmat";

export type RoomFixtureKind =
  | "bench"
  | "hood"
  | "island"
  | "cabinet"
  | "biosafety"
  | "incubator"
  | "rack"
  | "desk"
  | "table"
  | "storage";

export interface RoomFixture {
  kind: RoomFixtureKind;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  tone?: "steel" | "sage" | "oak" | "cold" | "amber";
}

export interface RoomVisualProfile {
  key: string;
  suiteLabel: string;
  narrative: string;
  accent: string;
  accentSoft: string;
  line: string;
  shell: string;
  headerFrom: string;
  headerTo: string;
  bodyFrom: string;
  bodyTo: string;
  text: string;
  muted: string;
  meta: string;
  badgeBg: string;
  badgeText: string;
  hazard: string;
  fixtureFill: string;
  fixtureStroke: string;
  slotFill: string;
  pattern: RoomPattern;
  ambientTint: string;
  ambientAlpha: number;
  particleColor: string | null;
  particleCount: number;
  floorColor: string;
  wallColor: string;
  thresholdColor: string;
}

export type RoomLayoutBand = "primary" | "support";

export interface RoomLayoutFootprint {
  widthBias: number;
  heightBias: number;
}

const BASE_PROFILE: RoomVisualProfile = {
  key: "operations",
  suiteLabel: "Laboratory operations suite",
  narrative: "General laboratory coordination and workflow support.",
  accent: "#2f7d72",
  accentSoft: "#d7ebe6",
  line: "#6e8794",
  shell: "#f4f5f3",
  headerFrom: "#dcefe9",
  headerTo: "#f6faf8",
  bodyFrom: "#f8faf8",
  bodyTo: "#eef3f0",
  text: "#17222e",
  muted: "#526577",
  meta: "#6e8390",
  badgeBg: "#eff5f2",
  badgeText: "#284239",
  hazard: "#c65c2b",
  fixtureFill: "#d8e4e2",
  fixtureStroke: "#58717c",
  slotFill: "#ffffff",
  pattern: "office_mesh",
  ambientTint: "#5a8895",
  ambientAlpha: 0.03,
  particleColor: null,
  particleCount: 0,
  floorColor: "#d8ddd8",
  wallColor: "#aeb7b2",
  thresholdColor: "#6b5c49",
};

function toTitle(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function roomVisualProfile(room: Room): RoomVisualProfile {
  switch (room.room_type) {
    case "wet_lab":
      return {
        ...BASE_PROFILE,
        key: "wet-lab",
        suiteLabel: "Molecular biology wet bench",
        narrative:
          "Bench-led prep, shared instruments, and quick access to washdown and solvent handling.",
        accent: "#1e7f74",
        accentSoft: "#d4ebe6",
        line: "#4f7f79",
        shell: "#eef5f2",
        headerFrom: "#cfe8e0",
        headerTo: "#f7fbfa",
        bodyFrom: "#f7faf8",
        bodyTo: "#e7f0ed",
        badgeBg: "#e6f2ee",
        badgeText: "#1f4f47",
        fixtureFill: "#d5e1df",
        fixtureStroke: "#55747b",
        pattern: "wet_grid",
        ambientTint: "#38a286",
        ambientAlpha: 0.05,
        particleColor: "#7acfb1",
        particleCount: 10,
        floorColor: "#31574f",
        wallColor: "#5f736e",
        thresholdColor: "#8d6e46",
      };
    case "tissue_culture":
      return {
        ...BASE_PROFILE,
        key: "tissue-culture",
        suiteLabel: "Aseptic cell culture suite",
        narrative:
          "Sterile workflow, containment-first handling, incubator support, and clean entry discipline.",
        accent: "#4f8b78",
        accentSoft: "#e4f1eb",
        line: "#78988a",
        shell: "#f6faf7",
        headerFrom: "#dcefe7",
        headerTo: "#fbfefd",
        bodyFrom: "#fbfcfb",
        bodyTo: "#edf5f1",
        badgeBg: "#eef7f2",
        badgeText: "#33574a",
        fixtureFill: "#dbe8e3",
        fixtureStroke: "#769285",
        pattern: "sterile_grid",
        ambientTint: "#8fc0ac",
        ambientAlpha: 0.04,
        particleColor: "#d7efe3",
        particleCount: 12,
        floorColor: "#32493f",
        wallColor: "#8ca59a",
        thresholdColor: "#c4d2cb",
      };
    case "cold_room":
      return {
        ...BASE_PROFILE,
        key: "cold-room",
        suiteLabel: "Temperature-controlled storage",
        narrative:
          "Dense cold-storage staging with narrow aisles, inventory turnover, and refrigeration monitoring.",
        accent: "#4f83b7",
        accentSoft: "#dbe7f4",
        line: "#6f8ea6",
        shell: "#edf4f9",
        headerFrom: "#d6e3f2",
        headerTo: "#f8fbfe",
        bodyFrom: "#f4f8fc",
        bodyTo: "#e8f0f7",
        badgeBg: "#e8f0f8",
        badgeText: "#345270",
        fixtureFill: "#d5e0eb",
        fixtureStroke: "#678197",
        pattern: "cold_racks",
        ambientTint: "#77a9df",
        ambientAlpha: 0.06,
        particleColor: "#cfe3f9",
        particleCount: 14,
        floorColor: "#314556",
        wallColor: "#7b90a1",
        thresholdColor: "#8ea7ba",
      };
    case "office":
      return {
        ...BASE_PROFILE,
        key: "office",
        suiteLabel: "Write-up and coordination area",
        narrative:
          "Protocol review, scheduling, data entry, and operator handoff before bench work.",
        accent: "#7a6b58",
        accentSoft: "#ece4db",
        line: "#8b7d69",
        shell: "#f5f1eb",
        headerFrom: "#e7ddd0",
        headerTo: "#fbf7f2",
        bodyFrom: "#faf7f2",
        bodyTo: "#f0ebe3",
        badgeBg: "#f2ece4",
        badgeText: "#584b3d",
        fixtureFill: "#e4dbcf",
        fixtureStroke: "#847462",
        pattern: "office_mesh",
        ambientTint: "#b49361",
        ambientAlpha: 0.03,
        particleColor: null,
        particleCount: 0,
        floorColor: "#5b4b3b",
        wallColor: "#9a8e80",
        thresholdColor: "#74624d",
      };
    case "chemical_storage":
      return {
        ...BASE_PROFILE,
        key: "chemical-storage",
        suiteLabel: "Chemical segregation storage",
        narrative:
          "Hazard-coded storage, corrosive segregation, spill readiness, and restricted handling.",
        accent: "#af6c2b",
        accentSoft: "#f0e1cf",
        line: "#9c7443",
        shell: "#f7f1e7",
        headerFrom: "#edd9bb",
        headerTo: "#fbf6ee",
        bodyFrom: "#fbf7f0",
        bodyTo: "#f3eadc",
        badgeBg: "#f5ebda",
        badgeText: "#6c4b1e",
        hazard: "#b4481d",
        fixtureFill: "#ead8bf",
        fixtureStroke: "#8d6b44",
        pattern: "hazmat",
        ambientTint: "#d6a25d",
        ambientAlpha: 0.045,
        particleColor: "#f0d4a2",
        particleCount: 6,
        floorColor: "#544635",
        wallColor: "#937d60",
        thresholdColor: "#8d5d26",
      };
    default:
      return BASE_PROFILE;
  }
}

export function roomOperationalNarrative(room: Room): string {
  return roomVisualProfile(room).narrative;
}

export function roomLayoutBand(room: Room): RoomLayoutBand {
  switch (room.room_type) {
    case "wet_lab":
    case "tissue_culture":
    case "shared_equipment":
    case "fume_hood_lab":
    case "autoclave_room":
      return "primary";
    default:
      return "support";
  }
}

export function roomLayoutPriority(room: Room): number {
  switch (room.room_type) {
    case "office":
      return 0;
    case "wet_lab":
      return 1;
    case "tissue_culture":
      return 2;
    case "shared_equipment":
      return 3;
    case "fume_hood_lab":
      return 4;
    case "control_room":
      return 5;
    case "dry_lab":
      return 6;
    case "chemical_storage":
      return 7;
    case "cold_room":
      return 8;
    case "autoclave_room":
      return 9;
    default:
      return 10;
  }
}

export function roomLayoutFootprint(room: Room): RoomLayoutFootprint {
  switch (room.room_type) {
    case "wet_lab":
      return { widthBias: 1.38, heightBias: 1.14 };
    case "tissue_culture":
      return { widthBias: 1.16, heightBias: 1.1 };
    case "shared_equipment":
      return { widthBias: 1.18, heightBias: 1.05 };
    case "fume_hood_lab":
      return { widthBias: 1.08, heightBias: 1.08 };
    case "office":
      return { widthBias: 1.02, heightBias: 0.94 };
    case "chemical_storage":
      return { widthBias: 0.86, heightBias: 0.98 };
    case "cold_room":
      return { widthBias: 0.88, heightBias: 0.96 };
    case "control_room":
    case "dry_lab":
      return { widthBias: 0.96, heightBias: 0.94 };
    default:
      return { widthBias: 1, heightBias: 1 };
  }
}

export function roomZoneBadges(room: Room): string[] {
  const badges = room.zones.map((zone) => `${toTitle(zone.zone_type)}: ${zone.name}`);
  if (room.has_eyewash) {
    badges.push("Eyewash station");
  }
  if (room.has_emergency_shower) {
    badges.push("Emergency shower");
  }
  if (room.has_ventilation) {
    badges.push("Ventilated");
  }
  return badges.slice(0, 4);
}

export function roomVisualFixtures(room: Room): RoomFixture[] {
  const hasFumeHood = room.zones.some((zone) => zone.zone_type === "fume_hood_area");
  switch (room.room_type) {
    case "wet_lab": {
      const serviceFixture: RoomFixture = hasFumeHood
        ? { kind: "hood", label: "fume hood", x: 0.66, y: 0.1, width: 0.24, height: 0.2 }
        : { kind: "cabinet", label: "reagent wall", x: 0.68, y: 0.08, width: 0.2, height: 0.24 };
      return [
        { kind: "bench", label: "prep bench west", x: 0.05, y: 0.12, width: 0.52, height: 0.16 },
        serviceFixture,
        { kind: "bench", label: "buffer bench east", x: 0.08, y: 0.32, width: 0.44, height: 0.12 },
        { kind: "island", label: "shared instrument island", x: 0.15, y: 0.5, width: 0.48, height: 0.16 },
        { kind: "storage", label: "washdown line", x: 0.08, y: 0.74, width: 0.3, height: 0.1 },
        { kind: "cabinet", label: "consumables wall", x: 0.44, y: 0.72, width: 0.24, height: 0.12 },
        { kind: "rack", label: "tip and tube rack", x: 0.72, y: 0.5, width: 0.16, height: 0.26 },
      ];
    }
    case "tissue_culture":
      return [
        { kind: "biosafety", label: "biosafety cabinet", x: 0.06, y: 0.1, width: 0.54, height: 0.21, tone: "sage" },
        { kind: "incubator", label: "CO2 incubator bank", x: 0.68, y: 0.1, width: 0.18, height: 0.34, tone: "sage" },
        { kind: "bench", label: "sterile prep lane", x: 0.16, y: 0.48, width: 0.52, height: 0.13, tone: "sage" },
        { kind: "rack", label: "media and flask rack", x: 0.1, y: 0.72, width: 0.26, height: 0.12, tone: "sage" },
        { kind: "storage", label: "waste transfer point", x: 0.42, y: 0.72, width: 0.2, height: 0.12, tone: "sage" },
        { kind: "cabinet", label: "PPE staging", x: 0.72, y: 0.56, width: 0.16, height: 0.2, tone: "sage" },
      ];
    case "cold_room":
      return [
        { kind: "rack", label: "left cold rack", x: 0.06, y: 0.1, width: 0.22, height: 0.7, tone: "cold" },
        { kind: "rack", label: "right cold rack", x: 0.72, y: 0.1, width: 0.2, height: 0.7, tone: "cold" },
        { kind: "storage", label: "transfer cart", x: 0.38, y: 0.34, width: 0.2, height: 0.12, tone: "cold" },
        { kind: "cabinet", label: "freezer bank", x: 0.38, y: 0.56, width: 0.22, height: 0.18, tone: "cold" },
      ];
    case "office":
      return [
        { kind: "desk", label: "protocol desks", x: 0.08, y: 0.12, width: 0.24, height: 0.18, tone: "oak" },
        { kind: "desk", label: "coordination desks", x: 0.36, y: 0.12, width: 0.24, height: 0.18, tone: "oak" },
        { kind: "table", label: "review table", x: 0.18, y: 0.5, width: 0.42, height: 0.16, tone: "oak" },
        { kind: "cabinet", label: "records wall", x: 0.72, y: 0.12, width: 0.16, height: 0.54, tone: "oak" },
        { kind: "storage", label: "dispatch credenza", x: 0.7, y: 0.74, width: 0.18, height: 0.1, tone: "oak" },
      ];
    case "chemical_storage":
      return [
        { kind: "cabinet", label: "hazmat cabinet west", x: 0.08, y: 0.12, width: 0.2, height: 0.58, tone: "amber" },
        { kind: "cabinet", label: "hazmat cabinet east", x: 0.32, y: 0.12, width: 0.2, height: 0.58, tone: "amber" },
        { kind: "storage", label: "segregated shelving", x: 0.58, y: 0.18, width: 0.28, height: 0.16, tone: "amber" },
        { kind: "storage", label: "spill tray zone", x: 0.58, y: 0.5, width: 0.28, height: 0.13, tone: "amber" },
        { kind: "rack", label: "waste hold", x: 0.62, y: 0.72, width: 0.2, height: 0.1, tone: "amber" },
      ];
    default:
      return [
        { kind: "bench", label: "work surface", x: 0.1, y: 0.18, width: 0.48, height: 0.14 },
        { kind: "cabinet", label: "storage", x: 0.7, y: 0.14, width: 0.18, height: 0.5 },
      ];
  }
}
