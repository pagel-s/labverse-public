import type { Door, Person, Room, WorldState } from "../../api/client";
import {
  roomLayoutFootprint,
  roomLayoutPriority,
} from "../../lib/roomVisuals";

export const TILE = 32;

export interface TileRect {
  tx: number;
  ty: number;
  tw: number;
  th: number;
}

export type FrontageSide = "north" | "south";
export type RoomEntryKind = "standard" | "controlled" | "shared";

export interface GeneratedCorridorSegment extends TileRect {
  kind: "connection";
  door_id: string;
  room_id?: string;
  entry_id?: string;
}

export interface GeneratedRoomEntry {
  id: string;
  room_id: string;
  side: FrontageSide;
  kind: RoomEntryKind;
  tx: number;
  ty: number;
  tw: number;
  th: number;
  anchor: { x: number; y: number };
  door_ids: string[];
}

export interface GeneratedDoorEndpoint {
  room_id: string;
  entry_id: string;
  x: number;
  y: number;
}

export interface GeneratedDoorRoute {
  door_id: string;
  connects: [string, string];
  endpoints: GeneratedDoorEndpoint[];
  visual_anchor: { x: number; y: number };
  spine_y: number | null;
}

export interface GeneratedLabLayout {
  kind: "generated";
  roomRects: Record<string, TileRect>;
  corridorSegments: GeneratedCorridorSegment[];
  roomEntries: Record<string, GeneratedRoomEntry[]>;
  doorRoutes: Record<string, GeneratedDoorRoute>;
  hubRoomId: string | null;
  mapWidthTiles: number;
  mapHeightTiles: number;
}

export type LabLayoutConfig = GeneratedLabLayout;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function rowWidth(roomIds: string[], sizes: Map<string, { tw: number; th: number }>, gap: number): number {
  if (roomIds.length === 0) {
    return 0;
  }
  return roomIds.reduce((total, roomId) => total + (sizes.get(roomId)?.tw ?? 0), 0) + gap * (roomIds.length - 1);
}

function roomEntryCount(room: Room, degree: number, isHub: boolean): { count: number; width: number; kind: RoomEntryKind } {
  if (degree <= 0) {
    return { count: 0, width: 0, kind: "standard" };
  }
  if (isHub) {
    return {
      count: room.room_type === "wet_lab" || room.room_type === "shared_equipment" ? 2 : 1,
      width: room.room_type === "wet_lab" ? 3 : 2,
      kind: "shared",
    };
  }
  if (room.room_type === "tissue_culture") {
    return { count: 1, width: 2, kind: "controlled" };
  }
  if (room.room_type === "wet_lab") {
    return { count: 1, width: 3, kind: "standard" };
  }
  return { count: 1, width: 2, kind: "standard" };
}

function buildDegreeMap(doors: Door[]): Map<string, number> {
  const degrees = new Map<string, number>();
  for (const door of doors) {
    for (const roomId of door.connects) {
      degrees.set(roomId, (degrees.get(roomId) ?? 0) + 1);
    }
  }
  return degrees;
}

function detectHubRoom(rooms: Room[], doors: Door[]): string | null {
  if (doors.length === 0) {
    return null;
  }
  const degrees = buildDegreeMap(doors);
  const candidates = rooms.filter(
    (room) =>
      (degrees.get(room.id) ?? 0) === doors.length &&
      doors.every((door) => door.connects.includes(room.id)),
  );
  return candidates.sort((left, right) => {
    const leftPreference = left.room_type === "wet_lab" ? 0 : left.room_type === "office" ? 2 : 1;
    const rightPreference = right.room_type === "wet_lab" ? 0 : right.room_type === "office" ? 2 : 1;
    return leftPreference - rightPreference || left.name.localeCompare(right.name);
  })[0]?.id ?? null;
}

function roomSize(
  room: Room,
  degree: number,
  isHub: boolean,
): { tw: number; th: number } {
  const zones = room.zones.length;
  const equipmentCount = room.equipment_ids.length;
  const footprint = roomLayoutFootprint(room);

  let tw = Math.round(7 * footprint.widthBias) + Math.floor(zones * 1.25) + Math.floor(equipmentCount * 0.35);
  let th = Math.round(7 * footprint.heightBias) + Math.floor(zones * 0.75);

  switch (room.room_type) {
    case "wet_lab":
      tw = clamp(tw, 11, 18);
      th = clamp(th, 8, 11);
      break;
    case "tissue_culture":
      tw = clamp(tw, 9, 12);
      th = clamp(th, 8, 10);
      break;
    case "cold_room":
      tw = clamp(tw, 7, 9);
      th = clamp(th, 8, 10);
      break;
    case "chemical_storage":
      tw = clamp(tw, 7, 9);
      th = clamp(th, 7, 9);
      break;
    case "office":
      tw = clamp(tw, 8, 10);
      th = clamp(th, 7, 8);
      break;
    default:
      tw = clamp(tw, 8, 12);
      th = clamp(th, 7, 10);
      break;
  }

  if (isHub) {
    tw = clamp(Math.max(tw, 16 + Math.min(6, Math.max(0, degree - 2) * 2)), 16, 24);
    th = clamp(Math.max(th, 10), 10, 12);
  }

  return { tw, th };
}

function buildRoomEntry(
  room: Room,
  rect: TileRect,
  side: FrontageSide,
  centerX: number,
  doorId: string,
  index: number,
  isHub: boolean,
): GeneratedRoomEntry {
  const { width, kind } = roomEntryCount(room, 1, isHub);
  const usableStart = rect.tx + 2;
  const usableEnd = rect.tx + rect.tw - 3;
  const entryWidth = Math.min(2, width, Math.max(1, usableEnd - usableStart + 1));
  const start = clamp(
    Math.round(centerX - entryWidth / 2),
    usableStart,
    usableEnd - entryWidth + 1,
  );
  const ty = side === "south" ? rect.ty + rect.th - 1 : rect.ty;
  return {
    id: `${room.id}:entry:${index + 1}`,
    room_id: room.id,
    side,
    kind,
    tx: start,
    ty,
    tw: entryWidth,
    th: 1,
    anchor: {
      x: (start + entryWidth / 2) * TILE,
      y: (ty + 0.5) * TILE,
    },
    door_ids: [doorId],
  };
}

function buildGeneratedLayout(world: WorldState): GeneratedLabLayout {
  const sorted = Object.values(world.lab.rooms).sort(
    (left, right) =>
      roomLayoutPriority(left) - roomLayoutPriority(right) || left.name.localeCompare(right.name),
  );
  const margin = 2;
  const gap = 1;
  const connectionDepth = 2;
  const degrees = buildDegreeMap(world.lab.doors);
  const hubRoomId = detectHubRoom(sorted, world.lab.doors);
  const hubRoom = sorted.find((room) => room.id === hubRoomId) ?? sorted[0];
  if (!hubRoom) {
    throw new Error("Cannot generate a lab layout without rooms.");
  }
  const leafRooms = sorted.filter((room) => room.id !== hubRoom.id);
  const splitIndex = Math.ceil(leafRooms.length / 2);
  const topRooms = leafRooms.slice(0, splitIndex);
  const bottomRooms = leafRooms.slice(splitIndex);

  const sizes = new Map<string, { tw: number; th: number }>();
  for (const room of sorted) {
    sizes.set(
      room.id,
      roomSize(room, degrees.get(room.id) ?? 0, room.id === hubRoom.id),
    );
  }

  const topWidth = rowWidth(topRooms.map((room) => room.id), sizes, gap);
  const bottomWidth = rowWidth(bottomRooms.map((room) => room.id), sizes, gap);
  const naturalHubSize = sizes.get(hubRoom.id)!;
  const totalRowWidth = Math.max(topWidth, bottomWidth, naturalHubSize.tw);
  sizes.set(hubRoom.id, { ...naturalHubSize, tw: totalRowWidth });
  const roomRects: Record<string, TileRect> = {};
  const topY = margin;

  const placeRow = (rooms: Room[], y: number): void => {
    const ids = rooms.map((room) => room.id);
    const width = rowWidth(ids, sizes, gap);
    let x = margin + Math.floor((totalRowWidth - width) / 2);
    for (const room of rooms) {
      const size = sizes.get(room.id)!;
      roomRects[room.id] = { tx: x, ty: y, tw: size.tw, th: size.th };
      x += size.tw + gap;
    }
  };

  if (topRooms.length > 0) {
    placeRow(topRooms, topY);
  }
  const maxTopH = Math.max(...topRooms.map((room) => sizes.get(room.id)?.th ?? 0), 0);
  const hubY = topY + maxTopH + (topRooms.length > 0 ? connectionDepth : 0);
  placeRow([hubRoom], hubY);
  const hubRect = roomRects[hubRoom.id]!;
  const bottomY = hubY + hubRect.th + (bottomRooms.length > 0 ? connectionDepth : 0);
  if (bottomRooms.length > 0) {
    placeRow(bottomRooms, bottomY);
  }

  const roomEntries: Record<string, GeneratedRoomEntry[]> = {};
  for (const room of sorted) {
    roomEntries[room.id] = [];
  }

  const minTx = Math.min(...Object.values(roomRects).map((rect) => rect.tx));
  const maxTx = Math.max(...Object.values(roomRects).map((rect) => rect.tx + rect.tw));
  const corridorSegments: GeneratedCorridorSegment[] = [];

  const doorRoutes: Record<string, GeneratedDoorRoute> = {};
  for (const [doorIndex, door] of world.lab.doors.entries()) {
    const leafRoomId = door.connects.find((roomId) => roomId !== hubRoom.id);
    const leafRoom = leafRooms.find((room) => room.id === leafRoomId);
    if (!leafRoom) {
      continue;
    }
    const leafRect = roomRects[leafRoom.id]!;
    const leafIsTop = topRooms.some((room) => room.id === leafRoom.id);
    const leafSide: FrontageSide = leafIsTop ? "south" : "north";
    const hubSide: FrontageSide = leafIsTop ? "north" : "south";
    const leafCenterTiles = leafRect.tx + leafRect.tw / 2;
    const leafEntry = buildRoomEntry(
      leafRoom,
      leafRect,
      leafSide,
      leafCenterTiles,
      door.id,
      0,
      false,
    );
    const hubEntry = buildRoomEntry(
      hubRoom,
      hubRect,
      hubSide,
      leafCenterTiles,
      door.id,
      doorIndex,
      true,
    );
    roomEntries[leafRoom.id].push(leafEntry);
    roomEntries[hubRoom.id].push(hubEntry);

    const segmentStart = Math.min(leafEntry.ty, hubEntry.ty) + 1;
    const segmentEnd = Math.max(leafEntry.ty, hubEntry.ty);
    corridorSegments.push({
      kind: "connection",
      door_id: door.id,
      room_id: leafRoom.id,
      entry_id: leafEntry.id,
      tx: leafEntry.tx,
      ty: segmentStart,
      tw: leafEntry.tw,
      th: Math.max(1, segmentEnd - segmentStart),
    });

    const endpointByRoom = new Map<string, GeneratedRoomEntry>([
      [hubRoom.id, hubEntry],
      [leafRoom.id, leafEntry],
    ]);
    const endpoints = door.connects.map((roomId) => {
      const entry = endpointByRoom.get(roomId)!;
      return {
        room_id: roomId,
        entry_id: entry.id,
        x: entry.anchor.x,
        y: entry.anchor.y,
      };
    });
    const visualAnchor = {
      x: leafEntry.anchor.x,
      y: (leafEntry.anchor.y + hubEntry.anchor.y) / 2,
    };

    doorRoutes[door.id] = {
      door_id: door.id,
      connects: [...door.connects] as [string, string],
      endpoints,
      visual_anchor: visualAnchor,
      spine_y: null,
    };
  }

  const mapWidthTiles = Math.max(
    maxTx + margin,
    ...corridorSegments.map((segment) => segment.tx + segment.tw + margin),
  );
  const mapHeightTiles = Math.max(
    ...Object.values(roomRects).map((rect) => rect.ty + rect.th + margin),
    ...corridorSegments.map((segment) => segment.ty + segment.th + margin),
  );

  return {
    kind: "generated",
    roomRects,
    corridorSegments,
    roomEntries,
    doorRoutes,
    hubRoomId: hubRoom.id,
    mapWidthTiles,
    mapHeightTiles,
  };
}

/** Get full LabVille layout config for a given world. */
export function getLabLayout(world: WorldState): LabLayoutConfig {
  return buildGeneratedLayout(world);
}

/** Get room tile rects for a given world. */
export function getRoomRects(world: WorldState): Record<string, TileRect> {
  return getLabLayout(world).roomRects;
}

/**
 * Get pixel position for a person inside their room/zone.
 * Offsets by personIndex to avoid stacking.
 */
export function getPersonPixelPos(
  rects: Record<string, TileRect>,
  room: Room,
  roomId: string,
  zoneId: string | null,
  personIndex: number,
): { x: number; y: number } {
  const rect = rects[roomId];
  if (!rect) return { x: TILE * 4, y: TILE * 4 };

  const roomPx = rect.tx * TILE;
  const roomPy = rect.ty * TILE;
  const roomPw = rect.tw * TILE;
  const roomPh = rect.th * TILE;
  const minX = roomPx + TILE * 2;
  const maxX = roomPx + roomPw - TILE * 2;
  const minY = roomPy + TILE * 3;
  const maxY = roomPy + roomPh - TILE * 2;

  if (zoneId && room.zones) {
    const zone = room.zones.find((candidate) => candidate.id === zoneId);
    if (zone) {
      const scaleX = roomPw / Math.max(room.width, 1);
      const scaleY = roomPh / Math.max(room.height, 1);
      const zoneWidth = Math.max(TILE * 1.5, zone.width * scaleX);
      const columns = Math.max(1, Math.min(3, Math.floor(zoneWidth / (TILE * 1.4))));
      const col = personIndex % columns;
      const row = Math.floor(personIndex / columns);
      const zx = roomPx + TILE + zone.position.x * scaleX;
      const zy =
        roomPy +
        TILE * 2.4 +
        (zone.position.y + zone.height) * scaleY +
        TILE;
      const rowSpacing = TILE * 1.45;
      const baseY = clamp(zy, minY, Math.max(minY, maxY - row * rowSpacing));
      return {
        x: clamp(zx + ((col + 1) / (columns + 1)) * zoneWidth, minX, maxX),
        y: baseY + row * rowSpacing,
      };
    }
  }

  const col = personIndex % 3;
  const row = Math.floor(personIndex / 3);
  return {
    x: clamp(roomPx + TILE * 3 + col * TILE * 1.6, minX, maxX),
    y: clamp(roomPy + roomPh - TILE * 2.5 - row * TILE * 1.5, minY, maxY),
  };
}

/** Get the pixel center of a room for camera targeting. */
export function getRoomCenter(
  rects: Record<string, TileRect>,
  roomId: string,
): { x: number; y: number } | null {
  const rect = rects[roomId];
  if (!rect) return null;
  return {
    x: (rect.tx + rect.tw / 2) * TILE,
    y: (rect.ty + rect.th / 2) * TILE,
  };
}

/** Group persons by room for position index calculation. */
export function groupPersonsByRoom(world: WorldState): Map<string, Person[]> {
  const byRoom = new Map<string, Person[]>();
  for (const person of Object.values(world.persons)) {
    if (!person.current_room_id) continue;
    const current = byRoom.get(person.current_room_id) ?? [];
    current.push(person);
    byRoom.set(person.current_room_id, current);
  }
  for (const people of byRoom.values()) {
    people.sort((left, right) => left.id.localeCompare(right.id));
  }
  return byRoom;
}
