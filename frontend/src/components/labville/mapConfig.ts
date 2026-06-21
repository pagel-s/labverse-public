import type { Door, Person, Room, WorldState } from "../../api/client";
import {
  roomLayoutBand,
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
  kind: "spine" | "branch";
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

function roomCenterX(rect: TileRect): number {
  return (rect.tx + rect.tw / 2) * TILE;
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

function detectHubRoom(roomIds: string[], doors: Door[]): string | null {
  const degrees = buildDegreeMap(doors);
  let hubRoomId: string | null = null;
  let maxDegree = 0;
  for (const roomId of roomIds) {
    const degree = degrees.get(roomId) ?? 0;
    if (degree > maxDegree) {
      maxDegree = degree;
      hubRoomId = roomId;
    }
  }
  if (maxDegree < 2 || maxDegree < roomIds.length - 2) {
    return null;
  }
  return hubRoomId;
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

function buildRoomEntries(
  room: Room,
  rect: TileRect,
  degree: number,
  isHub: boolean,
): GeneratedRoomEntry[] {
  const { count, width, kind } = roomEntryCount(room, degree, isHub);
  if (count <= 0) {
    return [];
  }

  const side: FrontageSide = roomLayoutBand(room) === "primary" ? "south" : "north";
  const usableStart = rect.tx + 2;
  const usableEnd = rect.tx + rect.tw - 3;
  const centers =
    count === 1
      ? [rect.tx + rect.tw / 2]
      : [rect.tx + rect.tw * 0.36, rect.tx + rect.tw * 0.64];

  const entries: GeneratedRoomEntry[] = [];
  let previousEnd = usableStart - 2;
  centers.forEach((center, index) => {
    let start = clamp(Math.round(center - width / 2), usableStart, usableEnd - width + 1);
    if (start <= previousEnd + 1) {
      start = previousEnd + 2;
    }
    start = clamp(start, usableStart, usableEnd - width + 1);
    previousEnd = start + width - 1;
    const ty = side === "south" ? rect.ty + rect.th - 1 : rect.ty;
    entries.push({
      id: `${room.id}:entry:${index + 1}`,
      room_id: room.id,
      side,
      kind,
      tx: start,
      ty,
      tw: width,
      th: 1,
      anchor: {
        x: (start + width / 2) * TILE,
        y: (ty + 0.5) * TILE,
      },
      door_ids: [],
    });
  });
  return entries;
}

function chooseNearestEntry(
  entries: GeneratedRoomEntry[],
  targetX: number,
): GeneratedRoomEntry | null {
  if (entries.length === 0) {
    return null;
  }
  return [...entries].sort(
    (left, right) => Math.abs(left.anchor.x - targetX) - Math.abs(right.anchor.x - targetX),
  )[0] ?? null;
}

function buildGeneratedLayout(world: WorldState): GeneratedLabLayout {
  const sorted = Object.values(world.lab.rooms).sort(
    (left, right) =>
      roomLayoutPriority(left) - roomLayoutPriority(right) || left.name.localeCompare(right.name),
  );
  const margin = 2;
  const gap = 1;
  const corridorHeight = 2;
  const degrees = buildDegreeMap(world.lab.doors);
  const hubRoomId = detectHubRoom(sorted.map((room) => room.id), world.lab.doors);

  const primaryRooms = sorted.filter((room) => roomLayoutBand(room) === "primary");
  const supportRooms = sorted.filter((room) => roomLayoutBand(room) === "support");
  const topRooms = primaryRooms.length > 0 ? [...primaryRooms] : sorted.slice(0, Math.ceil(sorted.length / 2));
  let bottomRooms = primaryRooms.length > 0 ? supportRooms : sorted.filter((room) => !topRooms.includes(room));
  if (bottomRooms.length === 0 && topRooms.length > 1) {
    bottomRooms = topRooms.splice(Math.ceil(topRooms.length / 2));
  }

  const sizes = new Map<string, { tw: number; th: number }>();
  for (const room of sorted) {
    sizes.set(
      room.id,
      roomSize(room, degrees.get(room.id) ?? 0, room.id === hubRoomId),
    );
  }

  const totalRowWidth = Math.max(rowWidth(topRooms.map((room) => room.id), sizes, gap), rowWidth(bottomRooms.map((room) => room.id), sizes, gap));
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

  placeRow(topRooms, topY);
  const maxTopH = Math.max(...topRooms.map((room) => sizes.get(room.id)?.th ?? 0), 0);
  const corridorTy = topY + maxTopH + 1;
  const bottomY = corridorTy + corridorHeight + 1;
  placeRow(bottomRooms, bottomY);

  const roomEntries: Record<string, GeneratedRoomEntry[]> = {};
  for (const room of sorted) {
    roomEntries[room.id] = buildRoomEntries(
      room,
      roomRects[room.id]!,
      degrees.get(room.id) ?? 0,
      room.id === hubRoomId,
    );
  }

  const minTx = Math.min(...Object.values(roomRects).map((rect) => rect.tx));
  const maxTx = Math.max(...Object.values(roomRects).map((rect) => rect.tx + rect.tw));
  const corridorSegments: GeneratedCorridorSegment[] = [
    {
      kind: "spine",
      tx: minTx,
      ty: corridorTy,
      tw: maxTx - minTx,
      th: corridorHeight,
    },
  ];

  for (const entries of Object.values(roomEntries)) {
    for (const entry of entries) {
      if (entry.side === "south") {
        const branchTy = entry.ty + 1;
        const branchHeight = corridorTy - branchTy;
        if (branchHeight > 0) {
          corridorSegments.push({
            kind: "branch",
            room_id: entry.room_id,
            entry_id: entry.id,
            tx: entry.tx,
            ty: branchTy,
            tw: entry.tw,
            th: branchHeight,
          });
        }
      } else {
        const branchTy = corridorTy + corridorHeight;
        const branchHeight = entry.ty - branchTy;
        if (branchHeight > 0) {
          corridorSegments.push({
            kind: "branch",
            room_id: entry.room_id,
            entry_id: entry.id,
            tx: entry.tx,
            ty: branchTy,
            tw: entry.tw,
            th: branchHeight,
          });
        }
      }
    }
  }

  const doorRoutes: Record<string, GeneratedDoorRoute> = {};
  const spineY = (corridorTy + corridorHeight / 2) * TILE;

  for (const door of world.lab.doors) {
    const [leftRoomId, rightRoomId] = door.connects;
    const leftEntries = roomEntries[leftRoomId] ?? [];
    const rightEntries = roomEntries[rightRoomId] ?? [];
    const leftRect = roomRects[leftRoomId];
    const rightRect = roomRects[rightRoomId];
    if (!leftRect || !rightRect) {
      continue;
    }

    const leftIsHub = hubRoomId === leftRoomId;
    const rightIsHub = hubRoomId === rightRoomId;
    const leftEntry = chooseNearestEntry(leftEntries, roomCenterX(rightRect));
    const rightEntry = chooseNearestEntry(rightEntries, roomCenterX(leftRect));

    const endpoints: GeneratedDoorEndpoint[] = [];
    if (leftEntry) {
      leftEntry.door_ids.push(door.id);
      endpoints.push({
        room_id: leftRoomId,
        entry_id: leftEntry.id,
        x: leftEntry.anchor.x,
        y: leftEntry.anchor.y,
      });
    }
    if (rightEntry) {
      rightEntry.door_ids.push(door.id);
      endpoints.push({
        room_id: rightRoomId,
        entry_id: rightEntry.id,
        x: rightEntry.anchor.x,
        y: rightEntry.anchor.y,
      });
    }

    const visualAnchor =
      leftIsHub && rightEntry
        ? { ...rightEntry.anchor }
        : rightIsHub && leftEntry
          ? { ...leftEntry.anchor }
          : endpoints.length === 2
            ? {
                x: (endpoints[0].x + endpoints[1].x) / 2,
                y: (endpoints[0].y + endpoints[1].y) / 2,
              }
            : endpoints[0]
              ? { x: endpoints[0].x, y: endpoints[0].y }
              : { x: roomCenterX(leftRect), y: (corridorTy + 0.5) * TILE };

    doorRoutes[door.id] = {
      door_id: door.id,
      connects: [leftRoomId, rightRoomId],
      endpoints,
      visual_anchor: visualAnchor,
      spine_y: spineY,
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
    hubRoomId,
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
