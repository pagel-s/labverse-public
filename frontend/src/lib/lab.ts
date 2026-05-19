import type {
  AvailableAction,
  Equipment,
  Material,
  Person,
  Room,
  Sample,
  SimulationEvent,
  WorldState,
} from "../api/client";

export interface FacilityMetrics {
  totalOccupancy: number;
  totalCapacity: number;
  activeEquipment: number;
  attentionEquipment: number;
  hazardMaterials: number;
  elevatedContainmentRooms: number;
  sampleCount: number;
}

export interface RoomDirectoryEntry {
  room: Room;
  summary: RoomSummary;
  occupants: Person[];
}

export interface OperationalNotice {
  id: string;
  tone: "critical" | "elevated" | "informational";
  title: string;
  detail: string;
  roomId?: string;
}

export interface RoomRoutePlan {
  path: string[] | null;
  blockedReason: string | null;
}

export interface RoomSummary {
  occupancy: number;
  capacity: number;
  occupancyRatio: number;
  activeEquipment: number;
  attentionEquipment: number;
  hazardMaterials: number;
  sampleCount: number;
  certificationsRequired: string[];
}

export function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

export function titleize(value: string): string {
  return humanize(value).replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function initials(name: string): string {
  return name
    .split(" ")
    .map((part) => part[0] ?? "")
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export function formatTimestamp(
  value: string | null,
  options: Intl.DateTimeFormatOptions = {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  },
): string {
  if (value === null) {
    return "--:--:--";
  }
  return new Date(value).toLocaleString([], options);
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat([], {
    maximumFractionDigits: value >= 10 ? 0 : 1,
  }).format(value);
}

export function formatQuantity(value: number, unit: string): string {
  return `${formatCompactNumber(value)} ${unit}`;
}

export function isHazardousMaterial(material: Material): boolean {
  return (
    material.ghs.pictograms.length > 0 ||
    material.ghs.signal_word !== "none" ||
    material.incompatible_with.length > 0 ||
    material.is_controlled
  );
}

export function materialRiskTone(material: Material): "critical" | "elevated" | "standard" {
  if (material.ghs.signal_word === "danger" || material.incompatible_with.length > 0) {
    return "critical";
  }
  if (isHazardousMaterial(material) || material.is_controlled) {
    return "elevated";
  }
  return "standard";
}

export function equipmentTone(status: string): "running" | "attention" | "ready" | "idle" {
  switch (status) {
    case "running":
      return "running";
    case "error":
    case "maintenance":
      return "attention";
    case "ready":
      return "ready";
    default:
      return "idle";
  }
}

export function roomTone(room: Room): "critical" | "elevated" | "standard" | "support" {
  if (room.bsl_level >= 3) {
    return "critical";
  }
  if (room.bsl_level === 2) {
    return "elevated";
  }
  if (room.room_type === "chemical_storage") {
    return "critical";
  }
  if (room.room_type === "office") {
    return "support";
  }
  return "standard";
}

export function summarizeRoom(world: WorldState, room: Room): RoomSummary {
  const people = Object.values(world.persons).filter((person) => person.current_room_id === room.id);
  const equipment = Object.values(world.equipment).filter((item) => item.room_id === room.id);
  const materials = Object.values(world.materials).filter((item) => item.room_id === room.id);
  const samples = Object.values(world.samples).filter((item) => item.room_id === room.id);
  const certificationsRequired = new Set<string>();

  for (const item of equipment) {
    for (const certification of item.required_certifications) {
      certificationsRequired.add(certification);
    }
  }

  return {
    occupancy: people.length,
    capacity: room.max_occupancy,
    occupancyRatio: room.max_occupancy > 0 ? people.length / room.max_occupancy : 0,
    activeEquipment: equipment.filter((item) => equipmentTone(item.status) === "running").length,
    attentionEquipment: equipment.filter((item) => equipmentTone(item.status) === "attention").length,
    hazardMaterials: materials.filter(isHazardousMaterial).length,
    sampleCount: samples.length,
    certificationsRequired: [...certificationsRequired].sort(),
  };
}

export function computeFacilityMetrics(world: WorldState): FacilityMetrics {
  const rooms = Object.values(world.lab.rooms);
  const totalOccupancy = Object.values(world.persons).filter((person) => person.current_room_id !== null).length;
  const totalCapacity = rooms.reduce((sum, room) => sum + room.max_occupancy, 0);
  const equipment = Object.values(world.equipment);

  return {
    totalOccupancy,
    totalCapacity,
    activeEquipment: equipment.filter((item) => equipmentTone(item.status) === "running").length,
    attentionEquipment: equipment.filter((item) => equipmentTone(item.status) === "attention").length,
    hazardMaterials: Object.values(world.materials).filter(isHazardousMaterial).length,
    elevatedContainmentRooms: rooms.filter((room) => room.bsl_level >= 2).length,
    sampleCount: Object.keys(world.samples).length,
  };
}

export function eventTone(event: SimulationEvent): "critical" | "elevated" | "informational" {
  const type = event.event_type.toLowerCase();
  if (type.includes("safety") || type.includes("violation") || type.includes("error")) {
    return "critical";
  }
  if (type.includes("equipment") || type.includes("material") || type.includes("sample")) {
    return "elevated";
  }
  return "informational";
}

export function personRoleTone(role: string): "lead" | "operator" | "specialist" | "default" {
  switch (role) {
    case "pi":
      return "lead";
    case "technician":
      return "specialist";
    case "grad_student":
      return "operator";
    default:
      return "default";
  }
}

export function suggestedPpeFromActionDescription(description: string): string | null {
  const normalized = description.trim();
  const match = /^(?:Don|Doff)\s+(.+)$/.exec(normalized);
  if (match === null) {
    return null;
  }
  return match[1]?.trim().replaceAll(" ", "_").toLowerCase() || null;
}

export function describeActionTarget(
  world: WorldState,
  action: AvailableAction,
): string | null {
  if (!action.target_id) {
    return null;
  }
  if (action.target_type === "room") {
    return world.lab.rooms[action.target_id]?.name ?? titleize(action.target_id);
  }
  if (action.target_type === "equipment") {
    return world.equipment[action.target_id]?.name ?? titleize(action.target_id);
  }
  if (action.target_type === "material") {
    return world.materials[action.target_id]?.name ?? titleize(action.target_id);
  }
  if (action.target_type === "sample") {
    return world.samples[action.target_id]?.name ?? titleize(action.target_id);
  }
  if (action.target_type === "door") {
    const door = world.lab.doors.find((candidate) => candidate.id === action.target_id);
    if (!door) {
      return titleize(action.target_id);
    }
    const [leftRoomId, rightRoomId] = door.connects;
    const leftRoom = world.lab.rooms[leftRoomId]?.name ?? titleize(leftRoomId);
    const rightRoom = world.lab.rooms[rightRoomId]?.name ?? titleize(rightRoomId);
    return `${leftRoom} ↔ ${rightRoom}`;
  }
  return titleize(action.target_id);
}

export function findRoomPath(
  world: WorldState,
  startRoomId: string,
  endRoomId: string,
): string[] | null {
  if (startRoomId === endRoomId) {
    return [startRoomId];
  }
  if (!world.lab.rooms[startRoomId] || !world.lab.rooms[endRoomId]) {
    return null;
  }

  const adjacency = new Map<string, string[]>();
  for (const roomId of Object.keys(world.lab.rooms)) {
    adjacency.set(roomId, []);
  }
  for (const door of world.lab.doors) {
    const [left, right] = door.connects;
    adjacency.get(left)?.push(right);
    adjacency.get(right)?.push(left);
  }

  const queue: string[][] = [[startRoomId]];
  const visited = new Set<string>([startRoomId]);

  while (queue.length > 0) {
    const path = queue.shift();
    if (!path) {
      continue;
    }
    const current = path[path.length - 1];
    const neighbors = adjacency.get(current) ?? [];
    for (const neighbor of neighbors) {
      if (visited.has(neighbor)) {
        continue;
      }
      const nextPath = [...path, neighbor];
      if (neighbor === endRoomId) {
        return nextPath;
      }
      visited.add(neighbor);
      queue.push(nextPath);
    }
  }

  return null;
}

function personBslClearance(person: Person | null): number {
  if (person === null) {
    return 1;
  }
  const certifications = new Set(person.certifications.map((item) => item.cert_type));
  if (certifications.has("bsl4_training")) {
    return 4;
  }
  if (certifications.has("bsl3_training")) {
    return 3;
  }
  if (certifications.has("bsl2_training")) {
    return 2;
  }
  return 1;
}

function firstBlockedDoorReason(
  world: WorldState,
  path: string[],
  bslClearance: number,
): string | null {
  for (let index = 0; index < path.length - 1; index += 1) {
    const left = path[index];
    const right = path[index + 1];
    const door = world.lab.doors.find((candidate) => {
      const [a, b] = candidate.connects;
      return (a === left && b === right) || (a === right && b === left);
    });
    if (!door) {
      return `No door currently connects ${titleize(left)} and ${titleize(right)}.`;
    }
    if (door.locked) {
      return `Door ${door.id} is locked.`;
    }
    if (!door.is_open) {
      return `Door ${door.id} is closed.`;
    }
    if (
      door.requires_bsl_clearance !== null &&
      bslClearance < door.requires_bsl_clearance
    ) {
      return `Door ${door.id} requires BSL-${door.requires_bsl_clearance} clearance.`;
    }
  }
  return null;
}

export function findTraversableRoomRoute(
  world: WorldState,
  startRoomId: string,
  endRoomId: string,
  person: Person | null,
): RoomRoutePlan {
  const bslClearance = personBslClearance(person);
  const traversablePath = (() => {
    if (startRoomId === endRoomId) {
      return [startRoomId];
    }
    if (!world.lab.rooms[startRoomId] || !world.lab.rooms[endRoomId]) {
      return null;
    }

    const adjacency = new Map<string, string[]>();
    for (const roomId of Object.keys(world.lab.rooms)) {
      adjacency.set(roomId, []);
    }
    for (const door of world.lab.doors) {
      const [left, right] = door.connects;
      if (
        door.locked ||
        !door.is_open ||
        (door.requires_bsl_clearance !== null && bslClearance < door.requires_bsl_clearance)
      ) {
        continue;
      }
      adjacency.get(left)?.push(right);
      adjacency.get(right)?.push(left);
    }

    const queue: string[][] = [[startRoomId]];
    const visited = new Set<string>([startRoomId]);

    while (queue.length > 0) {
      const path = queue.shift();
      if (!path) {
        continue;
      }
      const current = path[path.length - 1];
      const neighbors = adjacency.get(current) ?? [];
      for (const neighbor of neighbors) {
        if (visited.has(neighbor)) {
          continue;
        }
        const nextPath = [...path, neighbor];
        if (neighbor === endRoomId) {
          return nextPath;
        }
        visited.add(neighbor);
        queue.push(nextPath);
      }
    }
    return null;
  })();

  if (traversablePath !== null) {
    return { path: traversablePath, blockedReason: null };
  }

  const topologyPath = findRoomPath(world, startRoomId, endRoomId);
  if (topologyPath === null) {
    return {
      path: null,
      blockedReason: "No route exists in the current room graph.",
    };
  }

  return {
    path: null,
    blockedReason:
      firstBlockedDoorReason(world, topologyPath, bslClearance)
      ?? "No currently traversable route exists from the selected operator position.",
  };
}

export function formatRoomPath(world: WorldState, roomIds: string[]): string {
  return roomIds
    .map((roomId) => world.lab.rooms[roomId]?.name ?? titleize(roomId))
    .join(" -> ");
}

export function findCommandDefaults(
  equipment: Equipment | null,
  commandName: string | null,
): Record<string, string> {
  if (equipment === null || commandName === null) {
    return {};
  }
  const command = equipment.commands[commandName];
  if (!command) {
    return {};
  }

  const parameters = Array.isArray(command.parameters) ? command.parameters : [];

  return Object.fromEntries(
    parameters.flatMap((parameter) => {
      if (parameter.default_value !== null && parameter.default_value !== undefined) {
        return [[parameter.name, String(parameter.default_value)]];
      }
      const equipmentProperty = equipment.properties[parameter.name];
      if (
        equipmentProperty?.current_value !== null &&
        equipmentProperty?.current_value !== undefined
      ) {
        return [[parameter.name, String(equipmentProperty.current_value)]];
      }
      if (
        equipmentProperty?.default_value !== null &&
        equipmentProperty?.default_value !== undefined
      ) {
        return [[parameter.name, String(equipmentProperty.default_value)]];
      }
      return [];
    }),
  );
}

export function inferCommandName(equipment: Equipment | null): string | null {
  if (equipment === null) {
    return null;
  }
  return Object.keys(equipment.commands)[0] ?? null;
}

export function roomPeople(world: WorldState, roomId: string): Person[] {
  return Object.values(world.persons)
    .filter((person) => person.current_room_id === roomId)
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function roomEquipment(world: WorldState, roomId: string): Equipment[] {
  return Object.values(world.equipment)
    .filter((item) => item.room_id === roomId)
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function roomMaterials(world: WorldState, roomId: string): Material[] {
  return Object.values(world.materials)
    .filter((item) => item.room_id === roomId)
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function roomSamples(world: WorldState, roomId: string): Sample[] {
  return Object.values(world.samples)
    .filter((item) => item.room_id === roomId)
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function buildRoomDirectory(world: WorldState): RoomDirectoryEntry[] {
  return Object.values(world.lab.rooms)
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((room) => ({
      room,
      summary: summarizeRoom(world, room),
      occupants: roomPeople(world, room.id),
    }));
}

export function buildOperationalNotices(
  world: WorldState,
  recentEvents: SimulationEvent[],
  selectedRoomId: string | null,
  selectedAgentId: string | null,
): OperationalNotice[] {
  const notices: OperationalNotice[] = [];
  const directory = buildRoomDirectory(world);

  for (const entry of directory) {
    if (entry.summary.occupancyRatio >= 0.8) {
      notices.push({
        id: `occupancy:${entry.room.id}`,
        tone: "elevated",
        title: `${entry.room.name} approaching occupancy limit`,
        detail: `${entry.summary.occupancy}/${entry.summary.capacity} occupants currently assigned.`,
        roomId: entry.room.id,
      });
    }

    if (entry.summary.hazardMaterials > 0) {
      notices.push({
        id: `hazard:${entry.room.id}`,
        tone: roomTone(entry.room) === "critical" ? "critical" : "elevated",
        title: `${entry.room.name} contains controlled materials`,
        detail: `${entry.summary.hazardMaterials} hazard-tracked materials require supervised handling.`,
        roomId: entry.room.id,
      });
    }

    if (entry.summary.attentionEquipment > 0) {
      notices.push({
        id: `equipment:${entry.room.id}`,
        tone: "critical",
        title: `${entry.room.name} has equipment needing attention`,
        detail: `${entry.summary.attentionEquipment} asset${entry.summary.attentionEquipment === 1 ? "" : "s"} report error or maintenance state.`,
        roomId: entry.room.id,
      });
    }
  }

  if (selectedAgentId) {
    const person = world.persons[selectedAgentId];
    if (
      person &&
      person.current_room_id &&
      roomTone(world.lab.rooms[person.current_room_id]) !== "support" &&
      person.current_ppe.length === 0
    ) {
      notices.push({
        id: `operator:${person.id}`,
        tone: "elevated",
        title: `${person.name} has no PPE recorded`,
        detail: `Operator is currently in ${world.lab.rooms[person.current_room_id].name} without active PPE.`,
        roomId: person.current_room_id,
      });
    }
  }

  if (selectedRoomId) {
    const room = world.lab.rooms[selectedRoomId];
    const summary = summarizeRoom(world, room);
    if (summary.certificationsRequired.length > 0) {
      notices.push({
        id: `selected-room:${room.id}`,
        tone: "informational",
        title: `${room.name} requires trained operators`,
        detail: `Current equipment requires ${summary.certificationsRequired.map(titleize).join(", ")}.`,
        roomId: room.id,
      });
    }
  }

  for (const event of recentEvents.slice(0, 4)) {
    const tone = eventTone(event);
    if (tone === "informational") {
      continue;
    }
    const eventRoomId = event.target_id && world.lab.rooms[event.target_id]
      ? event.target_id
      : event.source_id && world.lab.rooms[event.source_id]
        ? event.source_id
        : undefined;
    notices.push({
      id: `event:${event.timestamp}:${event.event_type}`,
      tone,
      title: titleize(event.event_type),
      detail: event.description,
      roomId: eventRoomId,
    });
  }

  const deduped = new Map<string, OperationalNotice>();
  for (const notice of notices) {
    if (!deduped.has(notice.id)) {
      deduped.set(notice.id, notice);
    }
  }

  return [...deduped.values()].slice(0, 8);
}
