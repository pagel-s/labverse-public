import Phaser from "phaser";

import type { Door, Equipment, Room, WorldState } from "../../api/client";
import { equipmentTone, titleize } from "../../lib/lab";
import {
  roomVisualFixtures,
  roomVisualProfile,
  type RoomFixture,
} from "../../lib/roomVisuals";
import {
  TILE,
  getLabLayout,
  getPersonPixelPos,
  getRoomCenter,
  groupPersonsByRoom,
  type GeneratedLabLayout,
  type GeneratedRoomEntry,
  type TileRect,
} from "./mapConfig";
import { AgentSprite } from "./sprites";

/* ── Constants ───────────────────────────────────────────── */

const BG_COLOR = 0x10171d;

const EQUIP_STATUS_COLORS: Record<string, number> = {
  running: 0x0f8b6d,
  attention: 0xba5a36,
  ready: 0x2f7ebc,
  idle: 0x7b8793,
};

const BSL_BADGE_COLORS: Record<number, number> = {
  1: 0x2d8a4e,
  2: 0xb8860b,
  3: 0xc0392b,
};

const TILE_IDS = {
  floorLab: 1,
  floorCorridor: 2,
  floorCold: 3,
  floorStorage: 4,
  floorOffice: 5,
  threshold: 25,
  wallTop: 17,
  wallBottom: 18,
  wallLeft: 19,
  wallRight: 20,
  wallTopLeft: 21,
  wallTopRight: 22,
  wallBottomLeft: 23,
  wallBottomRight: 24,
  objectCabinet: 33,
  objectHood: 34,
  objectRackTop: 35,
  objectDeskTop: 36,
  objectDeskBottom: 37,
  objectMachine: 38,
  objectBench: 49,
  objectSink: 50,
  objectFreezer: 51,
  objectIsland: 52,
  objectStationLeft: 65,
  objectStationRight: 66,
  objectRackBottom: 67,
  objectAutoclave: 68,
  objectMonitor: 69,
  objectWaste: 70,
} as const;

type ScenePulseShape = "circle" | "rect";

function hexToNumber(hex: string): number {
  return Number.parseInt(hex.replace("#", ""), 16);
}

function drawProceduralFixture(
  g: Phaser.GameObjects.Graphics,
  fixture: RoomFixture,
  px: number,
  py: number,
  pw: number,
  ph: number,
  fill: number,
  stroke: number,
): void {
  const fx = px + fixture.x * pw;
  const fy = py + fixture.y * ph;
  const fw = fixture.width * pw;
  const fh = fixture.height * ph;
  const radius = Math.max(3, Math.min(8, Math.min(fw, fh) * 0.12));

  let fixtureFill = fill;
  if (fixture.tone === "cold") {
    fixtureFill = 0xc9dff5;
  } else if (fixture.tone === "oak") {
    fixtureFill = 0xd8c3a8;
  } else if (fixture.tone === "sage") {
    fixtureFill = 0xcde1d7;
  } else if (fixture.tone === "amber") {
    fixtureFill = 0xe0c59c;
  }

  g.fillStyle(fixtureFill, 0.98);
  g.fillRoundedRect(fx, fy, fw, fh, radius);
  g.lineStyle(1, stroke, 0.65);
  g.strokeRoundedRect(fx, fy, fw, fh, radius);

  g.fillStyle(0xffffff, 0.06);
  g.fillRoundedRect(fx + 3, fy + 3, Math.max(10, fw - 6), Math.max(3, fh * 0.18), 4);

  g.lineStyle(1, stroke, 0.2);
  if (fixture.kind === "rack" || fixture.kind === "storage" || fixture.kind === "cabinet") {
    for (let iy = fy + fh * 0.28; iy < fy + fh - 4; iy += Math.max(8, fh * 0.2)) {
      g.lineBetween(fx + 4, iy, fx + fw - 4, iy);
    }
  }
  if (fixture.kind === "bench" || fixture.kind === "island" || fixture.kind === "desk" || fixture.kind === "table") {
    g.lineBetween(fx + 5, fy + fh - 4, fx + 5, fy + fh + 3);
    g.lineBetween(fx + fw - 5, fy + fh - 4, fx + fw - 5, fy + fh + 3);
  }
  if (fixture.kind === "hood" || fixture.kind === "biosafety") {
    g.fillStyle(0xf4fbff, 0.14);
    g.fillRect(fx + fw * 0.12, fy + fh * 0.28, fw * 0.76, fh * 0.38);
    g.strokeRect(fx + fw * 0.12, fy + fh * 0.28, fw * 0.76, fh * 0.38);
  }
  if (fixture.kind === "incubator") {
    g.fillStyle(stroke, 0.16);
    g.fillRect(fx + fw * 0.62, fy + fh * 0.18, fw * 0.12, fh * 0.58);
  }
  if (fixture.kind === "bench" || fixture.kind === "island") {
    g.fillStyle(0xffffff, 0.08);
    g.fillRect(fx + fw * 0.16, fy + fh * 0.28, fw * 0.68, fh * 0.12);
  }
  if (fixture.kind === "storage" || fixture.kind === "cabinet") {
    g.fillStyle(stroke, 0.12);
    g.fillRect(fx + fw * 0.16, fy + fh * 0.14, fw * 0.08, fh * 0.66);
    g.fillRect(fx + fw * 0.76, fy + fh * 0.14, fw * 0.08, fh * 0.66);
  }
}

function zonePalette(zoneType: string): { stroke: number; fill: number; alpha: number } {
  switch (zoneType) {
    case "fume_hood_area":
      return { stroke: 0xc97a1d, fill: 0xe8b96f, alpha: 0.18 };
    case "bench":
      return { stroke: 0x4287c2, fill: 0x7eaed6, alpha: 0.14 };
    case "storage":
      return { stroke: 0x7d68b8, fill: 0xb7a8e7, alpha: 0.14 };
    default:
      return { stroke: 0x4f8b78, fill: 0x9fd1bf, alpha: 0.12 };
  }
}

function roomFloorTile(roomType: string): number {
  switch (roomType) {
    case "cold_room":
      return TILE_IDS.floorCold;
    case "chemical_storage":
      return TILE_IDS.floorStorage;
    case "office":
    case "control_room":
    case "dry_lab":
      return TILE_IDS.floorOffice;
    default:
      return TILE_IDS.floorLab;
  }
}

function putTile(
  layer: Phaser.Tilemaps.TilemapLayer,
  tileIndex: number,
  x: number,
  y: number,
): void {
  layer.putTileAt(tileIndex, x, y);
}

function stampVerticalPair(
  layer: Phaser.Tilemaps.TilemapLayer,
  x: number,
  y: number,
  topTile: number,
  bottomTile: number,
): void {
  putTile(layer, topTile, x, y);
  putTile(layer, bottomTile, x, y + 1);
}

function drawZoneFrames(
  g: Phaser.GameObjects.Graphics,
  room: Room,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  if (room.zones.length === 0) {
    return;
  }

  const scaleX = width / Math.max(room.width, 1);
  const scaleY = height / Math.max(room.height, 1);
  for (const zone of room.zones.slice(0, 3)) {
    const palette = zonePalette(zone.zone_type);
    const zx = x + zone.position.x * scaleX + 4;
    const zy = y + zone.position.y * scaleY + 4;
    const zw = Math.max(42, zone.width * scaleX - 8);
    const zh = Math.max(18, zone.height * scaleY + 12);

    g.fillStyle(palette.fill, palette.alpha);
    g.fillRoundedRect(zx, zy, zw, zh, 8);
    g.lineStyle(1.2, palette.stroke, 0.7);
    g.strokeRoundedRect(zx, zy, zw, zh, 8);
  }
}

/* ── Depth layers (explicit z-ordering) ─────────────────── */
const DEPTH = {
  GRID: 1,
  AMBIENT: 2,
  PARTICLES: 3,
  ROOM_HIGHLIGHT: 5,
  EQUIP: 10,
  ACCESS: 12,
  ROOM_LABELS: 15,
  FOCUS: 18,
  AGENTS: 20,
  HUD_BG: 1000,
  HUD_TEXT: 1001,
} as const;

type ClickCallback = (type: "room" | "agent" | "equipment", id: string) => void;

/* ── Scene ───────────────────────────────────────────────── */

export class LabScene extends Phaser.Scene {
  private agents = new Map<string, AgentSprite>();
  private agentRooms = new Map<string, string | null>();
  private roomRects: Record<string, TileRect> = {};
  private roomLabels: Phaser.GameObjects.GameObject[] = [];
  private roomHighlight: Phaser.GameObjects.Graphics | null = null;
  private focusGraphics: Phaser.GameObjects.Graphics | null = null;
  private roomHitAreas: Phaser.GameObjects.Rectangle[] = [];
  private equipOverlays: Phaser.GameObjects.Container[] = [];
  private ambientGraphics: Phaser.GameObjects.Graphics | null = null;
  private particleGraphics: Phaser.GameObjects.Graphics | null = null;
  private gridGraphics: Phaser.GameObjects.Graphics | null = null;
  private selectedRoomId: string | null = null;
  private hoveredRoomId: string | null = null;
  private selectedAgentId: string | null = null;
  private selectedEquipmentId: string | null = null;
  private hoveredEquipmentId: string | null = null;
  private onClick: ClickCallback | null = null;
  private onCreateCallback: (() => void) | null = null;
  private isDragging = false;
  private dragStart = { x: 0, y: 0 };
  private cameraDragStart = { x: 0, y: 0 };
  private mapReady = false;
  private mapW = 0;
  private mapH = 0;
  private highlightTime = 0;
  private hudText: Phaser.GameObjects.Text | null = null;
  private hudBg: Phaser.GameObjects.Graphics | null = null;
  private readonly hudVisible = false;
  private particles: Array<{
    x: number; y: number; vx: number; vy: number;
    color: number; alpha: number; roomType: string;
    // Bounds for wrapping particles inside their room
    bx: number; by: number; bw: number; bh: number;
  }> = [];
  private lastWorld: WorldState | null = null;
  private proceduralGraphics: Phaser.GameObjects.Graphics | null = null;
  private generatedTilemap: Phaser.Tilemaps.Tilemap | null = null;
  private generatedTileLayers: Phaser.Tilemaps.TilemapLayer[] = [];
  private generatedLayoutKey: string | null = null;
  private generatedLayout: GeneratedLabLayout | null = null;
  private equipmentAnchors = new Map<string, { x: number; y: number; label: string; status: string }>();
  private doorAnchors = new Map<string, { x: number; y: number; door: Door }>();

  constructor() {
    super({ key: "LabScene" });
  }

  setClickCallback(cb: ClickCallback): void {
    this.onClick = cb;
  }

  onceReady(cb: () => void): void {
    if (this.mapReady) {
      cb();
    } else {
      this.onCreateCallback = cb;
    }
  }

  /* ── Lifecycle ──────────────────────────────────────── */

  preload(): void {
    this.load.image("lab-tiles", "/labville/lab-tiles.png");
    this.load.image("characters", "/labville/characters.png");
  }

  create(): void {
    this.cameras.main.setBackgroundColor(BG_COLOR);
    this.mapW = 64 * TILE;
    this.mapH = 32 * TILE;

    // Grid overlay
    this.gridGraphics = this.add.graphics();
    this.gridGraphics.setDepth(DEPTH.GRID);
    this.drawGrid();

    // Ambient room tinting
    this.ambientGraphics = this.add.graphics();
    this.ambientGraphics.setDepth(DEPTH.AMBIENT);

    // Particle layer
    this.particleGraphics = this.add.graphics();
    this.particleGraphics.setDepth(DEPTH.PARTICLES);

    // Highlight graphics
    this.roomHighlight = this.add.graphics();
    this.roomHighlight.setDepth(DEPTH.ROOM_HIGHLIGHT);
    this.focusGraphics = this.add.graphics();
    this.focusGraphics.setDepth(DEPTH.FOCUS);

    // Camera controls
    this.setupCamera(this.mapW, this.mapH);

    // HUD overlay
    this.hudBg = this.add.graphics();
    this.hudBg.setScrollFactor(0);
    this.hudBg.setDepth(DEPTH.HUD_BG);
    this.hudText = this.add.text(16, 12, "", {
      fontSize: "13px",
      fontFamily: '"Avenir Next", "Segoe UI", sans-serif',
      color: "#e0e8f0",
      lineSpacing: 4,
    });
    this.hudText.setScrollFactor(0);
    this.hudText.setDepth(DEPTH.HUD_TEXT);

    this.mapReady = true;

    if (this.onCreateCallback) {
      this.onCreateCallback();
      this.onCreateCallback = null;
    }
  }

  update(_time: number, delta: number): void {
    this.highlightTime += delta;
    this.updateRoomHighlight();
    this.updateParticles(delta);
  }

  /* ── External API ───────────────────────────────────── */

  syncWorld(world: WorldState): void {
    if (!this.mapReady) return;

    this.lastWorld = world;
    const labLayout = getLabLayout(world);
    this.roomRects = labLayout.roomRects;
    this.generatedLayout = labLayout;
    this.ensureGeneratedTilemap(world);
    this.drawProceduralRooms(world);
    this.mapW = this.generatedLayout.mapWidthTiles * TILE;
    this.mapH = this.generatedLayout.mapHeightTiles * TILE;
    this.updateCameraExtents();

    this.drawAmbient(world);
    this.setupRoomLabels(world);
    this.setupRoomHitAreas(world);
    this.setupDoorOverlays(world);
    this.syncEquipment(world);
    this.syncPersons(world);
    this.updateRoomHighlight();
    this.spawnParticles(world);
    this.updateHud(world);
  }

  syncPersons(world: WorldState): void {
    if (!this.mapReady) return;

    const byRoom = groupPersonsByRoom(world);
    const seen = new Set<string>();

    for (const [roomId, persons] of byRoom) {
      const room = world.lab.rooms[roomId];
      if (!room) continue;

      persons.forEach((person, idx) => {
        seen.add(person.id);
        const pos = getPersonPixelPos(this.roomRects, room, roomId, person.current_zone_id, idx);
        const existing = this.agents.get(person.id);

        if (existing) {
          existing.updatePerson(person);
          const previousRoomId = this.agentRooms.get(person.id) ?? null;
          if (previousRoomId && previousRoomId !== roomId) {
            existing.tweenAlong(this.buildTravelPath(previousRoomId, roomId, pos));
          } else {
            existing.tweenTo(pos.x, pos.y);
          }
        } else {
          const sprite = new AgentSprite(this, person, pos.x, pos.y);
          sprite.setDepth(DEPTH.AGENTS);
          sprite.on("pointerup", () => {
            if (!this.isDragging) this.onClick?.("agent", person.id);
          });
          sprite.showActivity(person.current_activity);
          this.agents.set(person.id, sprite);
        }
        this.agentRooms.set(person.id, roomId);
      });
    }

    for (const [id, sprite] of this.agents) {
      if (!seen.has(id)) {
        sprite.destroy();
        this.agents.delete(id);
        this.agentRooms.delete(id);
      }
    }
  }

  selectRoom(roomId: string | null): void {
    this.selectedRoomId = roomId;
    this.updateRoomHighlight();
    if (roomId) {
      const center = getRoomCenter(this.roomRects, roomId);
      if (center) this.cameras.main.pan(center.x, center.y, 400, "Quad.easeInOut");
    }
  }

  selectAgent(agentId: string | null): void {
    const prev = this.selectedAgentId;
    this.selectedAgentId = agentId;
    if (prev) this.agents.get(prev)?.setHighlighted(false);
    if (agentId) {
      const sprite = this.agents.get(agentId);
      if (sprite) {
        sprite.setHighlighted(true);
        this.cameras.main.pan(sprite.x, sprite.y, 400, "Quad.easeInOut");
      }
    }
    this.updateRoomHighlight();
  }

  selectEquipment(equipmentId: string | null): void {
    this.selectedEquipmentId = equipmentId;
    this.updateRoomHighlight();
    if (equipmentId) {
      const anchor = this.equipmentAnchors.get(equipmentId);
      if (anchor) {
        this.cameras.main.pan(anchor.x, anchor.y, 360, "Quad.easeInOut");
      }
    }
  }

  showActionFeedback(feedback: {
    agentId: string;
    verb: string;
    targetId: string | null;
    phase: "executing" | "resolved" | "blocked" | "failed";
    outcome: string;
  }): void {
    const sprite = this.agents.get(feedback.agentId);
    const activity =
      feedback.phase === "executing"
        ? feedback.outcome
        : feedback.phase === "resolved"
          ? feedback.outcome
          : `${titleize(feedback.phase)}: ${feedback.outcome}`;
    sprite?.showActivity(activity);

    const targetId = feedback.targetId;
    const color =
      feedback.phase === "executing"
        ? 0xd4a34a
        : feedback.phase === "resolved"
          ? 0x2ae8c0
          : feedback.phase === "blocked"
            ? 0xd97243
            : 0xc25555;
    if (targetId && this.equipmentAnchors.has(targetId)) {
      const anchor = this.equipmentAnchors.get(targetId)!;
      this.spawnScenePulse(anchor.x, anchor.y, color, "circle");
      return;
    }
    if (targetId && this.doorAnchors.has(targetId)) {
      const anchor = this.doorAnchors.get(targetId)!;
      this.spawnScenePulse(anchor.x, anchor.y, color, "rect");
      return;
    }
    if (targetId && this.roomRects[targetId]) {
      const center = getRoomCenter(this.roomRects, targetId);
      if (center) {
        this.spawnScenePulse(center.x, center.y, color, "rect");
        return;
      }
    }
    if (sprite) {
      this.spawnScenePulse(sprite.x, sprite.y - 14, color, "circle");
    }
  }

  /* ── Private helpers ────────────────────────────────── */

  private setupCamera(mapW: number, mapH: number): void {
    const cam = this.cameras.main;
    cam.setBounds(-TILE * 2, -TILE * 2, mapW + TILE * 4, mapH + TILE * 4);
    this.fitCameraToReadableLab();

    this.scale.on(Phaser.Scale.Events.RESIZE, () => {
      this.fitCameraToReadableLab(false);
    });

    // Drag-to-pan
    this.input.on("pointerdown", (p: Phaser.Input.Pointer) => {
      this.isDragging = false;
      this.dragStart = { x: p.x, y: p.y };
      this.cameraDragStart = { x: cam.scrollX, y: cam.scrollY };
    });
    this.input.on("pointermove", (p: Phaser.Input.Pointer) => {
      if (!p.isDown) return;
      const dx = p.x - this.dragStart.x;
      const dy = p.y - this.dragStart.y;
      if (Math.abs(dx) > 4 || Math.abs(dy) > 4) this.isDragging = true;
      if (this.isDragging) {
        cam.scrollX = this.cameraDragStart.x - dx / cam.zoom;
        cam.scrollY = this.cameraDragStart.y - dy / cam.zoom;
      }
    });

    // Scroll-to-zoom
    this.input.on("wheel", (_p: unknown, _gos: unknown, _dx: number, dy: number) => {
      cam.setZoom(Phaser.Math.Clamp(cam.zoom + (dy > 0 ? -0.05 : 0.05), 0.3, 2.5));
    });

    // Pinch-to-zoom for touch
    let lastPinchDist = 0;
    this.input.on("pointerdown", () => { lastPinchDist = 0; });
    this.input.on("pointermove", () => {
      if (this.input.pointer1.isDown && this.input.pointer2.isDown) {
        const dx = this.input.pointer1.x - this.input.pointer2.x;
        const dy2 = this.input.pointer1.y - this.input.pointer2.y;
        const dist = Math.sqrt(dx * dx + dy2 * dy2);
        if (lastPinchDist > 0) {
          const delta = (dist - lastPinchDist) * 0.003;
          cam.setZoom(Phaser.Math.Clamp(cam.zoom + delta, 0.3, 2.5));
          this.isDragging = true;
        }
        lastPinchDist = dist;
      } else {
        lastPinchDist = 0;
      }
    });
  }

  private updateCameraExtents(): void {
    const cam = this.cameras.main;
    cam.setBounds(-TILE * 2, -TILE * 2, this.mapW + TILE * 4, this.mapH + TILE * 4);
    this.fitCameraToReadableLab();
    this.drawGrid();
  }

  private fitCameraToReadableLab(center = true): void {
    const cam = this.cameras.main;
    const zoomX = cam.width / Math.max(this.mapW, 1);
    const zoomY = cam.height / Math.max(this.mapH, 1);
    const fitZoom = Math.min(zoomX, zoomY) * 0.95;
    const fillHeightZoom = zoomY * 0.98;
    const readableZoom = cam.width < 760 ? 0.9 : 0.82;
    cam.setZoom(Phaser.Math.Clamp(Math.max(fitZoom, fillHeightZoom, readableZoom), 0.45, 1.5));
    if (center) {
      cam.centerOn(this.mapW / 2, this.mapH / 2);
    }
  }

  private spawnScenePulse(
    x: number,
    y: number,
    color: number,
    shape: ScenePulseShape,
  ): void {
    const pulse =
      shape === "circle"
        ? this.add.circle(x, y, 14, color, 0.16)
        : this.add.rectangle(x, y, 34, 24, color, 0.12);
    pulse.setDepth(DEPTH.FOCUS + 1);

    if ("setStrokeStyle" in pulse) {
      pulse.setStrokeStyle(2, color, 0.88);
    }

    const secondary =
      shape === "circle"
        ? this.add.circle(x, y, 9, color, 0.2)
        : this.add.rectangle(x, y, 20, 12, color, 0.16);
    secondary.setDepth(DEPTH.FOCUS + 2);
    if ("setStrokeStyle" in secondary) {
      secondary.setStrokeStyle(1.5, 0xf7fbff, 0.72);
    }

    this.tweens.add({
      targets: [pulse, secondary],
      alpha: 0,
      scaleX: shape === "circle" ? 2.6 : 1.9,
      scaleY: shape === "circle" ? 2.6 : 1.9,
      duration: 620,
      ease: "Quad.easeOut",
      onComplete: () => {
        pulse.destroy();
        secondary.destroy();
      },
    });
  }

  private drawGrid(): void {
    const g = this.gridGraphics;
    if (!g) return;
    g.clear();
    g.lineStyle(1, 0xd4e2ed, 0.028);
    for (let x = 0; x <= this.mapW; x += TILE) {
      g.moveTo(x, 0);
      g.lineTo(x, this.mapH);
    }
    for (let y = 0; y <= this.mapH; y += TILE) {
      g.moveTo(0, y);
      g.lineTo(this.mapW, y);
    }
    g.strokePath();
  }

  private generatedWorldKey(world: WorldState): string {
    return Object.entries(this.roomRects)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([roomId, rect]) => {
        const room = world.lab.rooms[roomId];
        return `${roomId}:${room?.room_type ?? "unknown"}:${rect.tx},${rect.ty},${rect.tw},${rect.th}`;
      })
      .join("|");
  }

  private clearGeneratedTilemap(): void {
    for (const layer of this.generatedTileLayers) {
      layer.destroy();
    }
    this.generatedTileLayers = [];
    this.generatedTilemap?.destroy();
    this.generatedTilemap = null;
    this.generatedLayoutKey = null;
  }

  private generatedRoomEntries(roomId: string): GeneratedRoomEntry[] {
    return this.generatedLayout?.roomEntries[roomId] ?? [];
  }

  private roomTravelEntry(
    roomId: string,
    targetRoomId: string | null = null,
  ): GeneratedRoomEntry | null {
    const entries = this.generatedRoomEntries(roomId);
    if (entries.length > 0) {
      if (targetRoomId) {
        const targetCenter = getRoomCenter(this.roomRects, targetRoomId);
        if (targetCenter) {
          const nearest = [...entries].sort(
            (left, right) => Math.abs(left.anchor.x - targetCenter.x) - Math.abs(right.anchor.x - targetCenter.x),
          )[0];
          if (nearest) {
            return nearest;
          }
        }
      }
      return entries[0] ?? null;
    }
    return null;
  }

  private entryThresholdAnchor(entry: GeneratedRoomEntry): { x: number; y: number } {
    return { ...entry.anchor };
  }

  private entryInteriorAnchor(entry: GeneratedRoomEntry): { x: number; y: number } {
    return {
      x: entry.anchor.x,
      y: entry.side === "south" ? entry.anchor.y - TILE : entry.anchor.y + TILE,
    };
  }

  private entryCorridorAnchor(entry: GeneratedRoomEntry): { x: number; y: number } {
    return {
      x: entry.anchor.x,
      y: entry.side === "south" ? entry.anchor.y + TILE : entry.anchor.y - TILE,
    };
  }

  private buildTravelPath(
    fromRoomId: string,
    toRoomId: string,
    destination: { x: number; y: number },
  ): Array<{ x: number; y: number }> {
    if (!this.lastWorld || fromRoomId === toRoomId) {
      return [destination];
    }

    const directDoor = this.lastWorld.lab.doors.find(
      (door) => door.connects.includes(fromRoomId) && door.connects.includes(toRoomId),
    );
    if (directDoor && this.generatedLayout) {
      const route = this.generatedLayout.doorRoutes[directDoor.id];
      if (route) {
        const fromEndpoint = route.endpoints.find((endpoint) => endpoint.room_id === fromRoomId);
        const toEndpoint = route.endpoints.find((endpoint) => endpoint.room_id === toRoomId);
        const fromEntry = fromEndpoint
          ? this.generatedRoomEntries(fromRoomId).find((entry) => entry.id === fromEndpoint.entry_id) ?? null
          : null;
        const toEntry = toEndpoint
          ? this.generatedRoomEntries(toRoomId).find((entry) => entry.id === toEndpoint.entry_id) ?? null
          : null;
        const points: Array<{ x: number; y: number }> = [];
        if (fromEndpoint && fromEntry) {
          points.push(this.entryInteriorAnchor(fromEntry));
          points.push(this.entryThresholdAnchor(fromEntry));
          points.push(this.entryCorridorAnchor(fromEntry));
          if (route.spine_y !== null && Math.abs(this.entryCorridorAnchor(fromEntry).y - route.spine_y) > 4) {
            points.push({ x: fromEndpoint.x, y: route.spine_y });
          }
        }
        if (toEndpoint && toEntry) {
          if (route.spine_y !== null) {
            points.push({ x: toEndpoint.x, y: route.spine_y });
          }
          points.push(this.entryCorridorAnchor(toEntry));
          points.push(this.entryThresholdAnchor(toEntry));
          points.push(this.entryInteriorAnchor(toEntry));
        }
        points.push(destination);
        return points.filter((point, index) => {
          if (index === 0) {
            return true;
          }
          const previous = points[index - 1];
          return Phaser.Math.Distance.Between(previous.x, previous.y, point.x, point.y) > 4;
        });
      }
    }
    if (directDoor) {
      const doorAnchor = this.doorAnchors.get(directDoor.id);
      if (doorAnchor) {
        return [{ x: doorAnchor.x, y: doorAnchor.y }, destination];
      }
    }

    const spine = this.generatedLayout?.corridorSegments.find((segment) => segment.kind === "spine");
    if (!spine) {
      return [destination];
    }

    const fromEntry = this.roomTravelEntry(fromRoomId, toRoomId);
    const toEntry = this.roomTravelEntry(toRoomId, fromRoomId);
    if (!fromEntry || !toEntry) {
      return [destination];
    }

    const corridorY = (spine.ty + spine.th / 2) * TILE;
    const path = [
      this.entryInteriorAnchor(fromEntry),
      this.entryThresholdAnchor(fromEntry),
      this.entryCorridorAnchor(fromEntry),
      { x: this.entryCorridorAnchor(fromEntry).x, y: corridorY },
      { x: this.entryCorridorAnchor(toEntry).x, y: corridorY },
      this.entryCorridorAnchor(toEntry),
      this.entryThresholdAnchor(toEntry),
      this.entryInteriorAnchor(toEntry),
      destination,
    ];

    return path.filter((point, index) => {
      if (index === 0) {
        return true;
      }
      const previous = path[index - 1];
      return Phaser.Math.Distance.Between(previous.x, previous.y, point.x, point.y) > 4;
    });
  }

  private ensureGeneratedTilemap(world: WorldState): void {
    const layout = this.generatedLayout;
    if (!layout) {
      return;
    }
    const layoutKey = this.generatedWorldKey(world);
    if (layoutKey === this.generatedLayoutKey && this.generatedTileLayers.length > 0) {
      for (const layer of this.generatedTileLayers) {
        layer.setVisible(true);
      }
      return;
    }

    this.clearGeneratedTilemap();

    const rects = Object.values(this.roomRects);
    if (rects.length === 0) {
      return;
    }

    const map = this.make.tilemap({
      tileWidth: TILE,
      tileHeight: TILE,
      width: layout.mapWidthTiles,
      height: layout.mapHeightTiles,
    });
    const tileset = map.addTilesetImage("lab-tiles", "lab-tiles", TILE, TILE, 0, 0, 1);
    if (!tileset) {
      return;
    }

    const floorLayer = map.createBlankLayer("generated-floor", tileset);
    const wallLayer = map.createBlankLayer("generated-walls", tileset);
    const objectLayer = map.createBlankLayer("generated-objects", tileset);
    if (!floorLayer || !wallLayer || !objectLayer) {
      return;
    }

    floorLayer.setDepth(0);
    wallLayer.setDepth(0.05);
    objectLayer.setDepth(0.1);
    this.generatedTilemap = map;
    this.generatedTileLayers = [floorLayer, wallLayer, objectLayer];

    for (const segment of layout.corridorSegments) {
      for (let y = segment.ty; y < segment.ty + segment.th; y += 1) {
        for (let x = segment.tx; x < segment.tx + segment.tw; x += 1) {
          putTile(floorLayer, TILE_IDS.floorCorridor, x, y);
        }
      }
    }

    for (const [roomId, rect] of Object.entries(this.roomRects)) {
      const room = world.lab.rooms[roomId];
      if (!room) {
        continue;
      }
      this.stampRoomTiles(
        floorLayer,
        wallLayer,
        objectLayer,
        room,
        rect,
        layout.roomEntries[room.id] ?? [],
      );
    }

    this.generatedLayoutKey = layoutKey;
  }

  private stampRoomTiles(
    floorLayer: Phaser.Tilemaps.TilemapLayer,
    wallLayer: Phaser.Tilemaps.TilemapLayer,
    objectLayer: Phaser.Tilemaps.TilemapLayer,
    room: Room,
    rect: TileRect,
    entries: GeneratedRoomEntry[],
  ): void {
    const floorTile = roomFloorTile(room.room_type);
    const topY = rect.ty;
    const bottomY = rect.ty + rect.th - 1;
    const leftX = rect.tx;
    const rightX = rect.tx + rect.tw - 1;
    const opensOnWall = (side: "north" | "south", x: number): boolean =>
      entries.some((entry) => entry.side === side && x >= entry.tx && x < entry.tx + entry.tw);

    for (let y = rect.ty + 1; y < rect.ty + rect.th - 1; y += 1) {
      for (let x = rect.tx + 1; x < rect.tx + rect.tw - 1; x += 1) {
        putTile(floorLayer, floorTile, x, y);
      }
    }

    putTile(wallLayer, TILE_IDS.wallTopLeft, leftX, topY);
    putTile(wallLayer, TILE_IDS.wallTopRight, rightX, topY);
    putTile(wallLayer, TILE_IDS.wallBottomLeft, leftX, bottomY);
    putTile(wallLayer, TILE_IDS.wallBottomRight, rightX, bottomY);

    for (let x = leftX + 1; x < rightX; x += 1) {
      if (opensOnWall("north", x)) {
        putTile(floorLayer, floorTile, x, topY);
      } else {
        putTile(wallLayer, TILE_IDS.wallTop, x, topY);
      }

      if (opensOnWall("south", x)) {
        putTile(floorLayer, floorTile, x, bottomY);
      } else {
        putTile(wallLayer, TILE_IDS.wallBottom, x, bottomY);
      }
    }

    for (let y = topY + 1; y < bottomY; y += 1) {
      putTile(wallLayer, TILE_IDS.wallLeft, leftX, y);
      putTile(wallLayer, TILE_IDS.wallRight, rightX, y);
    }

    for (const entry of entries) {
      const transitionY = entry.side === "south" ? bottomY - 1 : topY + 1;
      if (transitionY <= topY || transitionY >= bottomY) {
        continue;
      }
      for (let x = entry.tx; x < entry.tx + entry.tw; x += 1) {
        putTile(floorLayer, floorTile, x, transitionY);
      }
    }

    this.stampRoomObjects(objectLayer, room, rect);
  }

  private stampRoomObjects(
    objectLayer: Phaser.Tilemaps.TilemapLayer,
    room: Room,
    rect: TileRect,
  ): void {
    const innerWidth = Math.max(1, rect.tw - 2);
    const innerHeight = Math.max(1, rect.th - 2);
    const fixtures = roomVisualFixtures(room);

    const putFixtureTile = (tile: number, tileX: number, tileY: number): void => {
      if (tileX <= rect.tx || tileX >= rect.tx + rect.tw - 1 || tileY <= rect.ty || tileY >= rect.ty + rect.th - 1) {
        return;
      }
      putTile(objectLayer, tile, tileX, tileY);
    };

    for (const fixture of fixtures) {
      const baseX = rect.tx + 1 + Math.round(fixture.x * (innerWidth - 1));
      const baseY = rect.ty + 1 + Math.round(fixture.y * (innerHeight - 1));
      const spanX = Math.max(1, Math.round(fixture.width * innerWidth));
      const clampedX = Math.min(rect.tx + rect.tw - 2, baseX);
      const clampedY = Math.min(rect.ty + rect.th - 2, baseY);

      switch (fixture.kind) {
        case "bench":
          for (let x = clampedX; x < clampedX + spanX; x += 2) {
            putFixtureTile(TILE_IDS.objectBench, x, clampedY);
          }
          break;
        case "hood":
        case "biosafety":
          for (let x = clampedX; x < clampedX + spanX; x += 2) {
            putFixtureTile(TILE_IDS.objectHood, x, clampedY);
          }
          break;
        case "island":
          for (let x = clampedX; x < clampedX + spanX; x += 2) {
            putFixtureTile(TILE_IDS.objectIsland, x, clampedY);
          }
          break;
        case "desk":
        case "table":
          stampVerticalPair(objectLayer, clampedX, clampedY, TILE_IDS.objectDeskTop, TILE_IDS.objectDeskBottom);
          if (fixture.kind === "desk") {
            putFixtureTile(TILE_IDS.objectMonitor, Math.min(clampedX + 2, rect.tx + rect.tw - 2), clampedY + 3);
          }
          break;
        case "rack":
        case "storage":
          for (let x = clampedX; x < clampedX + spanX; x += 2) {
            stampVerticalPair(objectLayer, x, clampedY, TILE_IDS.objectRackTop, TILE_IDS.objectRackBottom);
          }
          break;
        case "cabinet":
          for (let x = clampedX; x < clampedX + spanX; x += 2) {
            putFixtureTile(TILE_IDS.objectCabinet, x, clampedY);
          }
          break;
        case "incubator":
          putFixtureTile(
            room.room_type === "cold_room" ? TILE_IDS.objectFreezer : TILE_IDS.objectMachine,
            clampedX,
            clampedY,
          );
          break;
      }
    }

    switch (room.room_type) {
      case "wet_lab":
        putFixtureTile(TILE_IDS.objectStationLeft, rect.tx + 1, rect.ty + rect.th - 2);
        putFixtureTile(TILE_IDS.objectStationRight, rect.tx + 3, rect.ty + rect.th - 2);
        putFixtureTile(TILE_IDS.objectSink, rect.tx + rect.tw - 4, rect.ty + 5);
        break;
      case "tissue_culture":
        putFixtureTile(TILE_IDS.objectWaste, rect.tx + rect.tw - 3, rect.ty + rect.th - 4);
        putFixtureTile(TILE_IDS.objectMachine, rect.tx + rect.tw - 4, rect.ty + 2);
        break;
      case "chemical_storage":
        putFixtureTile(TILE_IDS.objectWaste, rect.tx + rect.tw - 3, rect.ty + rect.th - 3);
        break;
      case "office":
        putFixtureTile(TILE_IDS.objectMonitor, rect.tx + rect.tw - 4, rect.ty + 5);
        break;
      case "cold_room":
        putFixtureTile(TILE_IDS.objectFreezer, rect.tx + 1, rect.ty + rect.th - 3);
        break;
    }
  }

  private drawAmbient(world: WorldState): void {
    const g = this.ambientGraphics;
    if (!g) return;
    g.clear();

    // Very subtle room-type tinting only — no shadows or vignettes
    for (const [roomId, rect] of Object.entries(this.roomRects)) {
      const room = world.lab.rooms[roomId];
      if (!room) continue;

      const px = rect.tx * TILE;
      const py = rect.ty * TILE;
      const pw = rect.tw * TILE;
      const ph = rect.th * TILE;

      const profile = roomVisualProfile(room);
      const tint = hexToNumber(profile.ambientTint);
      const alpha = profile.ambientAlpha;
      if (alpha > 0) {
        g.fillStyle(tint, alpha);
        g.fillRect(px + TILE, py + TILE, pw - TILE * 2, ph - TILE * 2);
      }
    }
  }

  private syncEquipment(world: WorldState): void {
    for (const c of this.equipOverlays) c.destroy();
    this.equipOverlays = [];
    this.equipmentAnchors.clear();

    const byRoom = new Map<string, Equipment[]>();
    for (const eq of Object.values(world.equipment)) {
      const arr = byRoom.get(eq.room_id) ?? [];
      arr.push(eq);
      byRoom.set(eq.room_id, arr);
    }

    for (const [roomId, eqs] of byRoom) {
      const rect = this.roomRects[roomId];
      const room = world.lab.rooms[roomId];
      if (!rect || !room) continue;

      const px = rect.tx * TILE;
      const py = rect.ty * TILE;
      const pw = rect.tw * TILE;
      const ph = rect.th * TILE;

      eqs.forEach((eq, idx) => {
        const tone = equipmentTone(eq.status);
        const color = EQUIP_STATUS_COLORS[tone] ?? EQUIP_STATUS_COLORS.idle;

        // Position: zone-based or grid fallback
        let ex: number, ey: number;
        if (eq.zone_id && room.zones) {
          const zone = room.zones.find((z) => z.id === eq.zone_id);
          if (zone) {
            const scaleX = pw / Math.max(room.width, 1);
            const scaleY = ph / Math.max(room.height, 1);
            ex = px + TILE + zone.position.x * scaleX + (idx % 2) * TILE * 1.5 + TILE * 0.5;
            ey = py + TILE * 1.5 + zone.position.y * scaleY + Math.floor(idx / 2) * TILE + TILE;
          } else {
            ex = px + TILE * 2 + (idx % 4) * TILE * 2;
            ey = py + TILE * 3 + Math.floor(idx / 4) * TILE * 2;
          }
        } else {
          ex = px + TILE * 2 + (idx % 4) * TILE * 2;
          ey = py + TILE * 3 + Math.floor(idx / 4) * TILE * 2;
        }
        this.equipmentAnchors.set(eq.id, {
          x: ex,
          y: ey,
          label: eq.name,
          status: eq.status,
        });

        const container = this.add.container(ex, ey);
        container.setDepth(DEPTH.EQUIP);

        // Glow behind running equipment
        if (tone === "running") {
          const glow = this.add.circle(0, 0, 14, color, 0.12);
          container.add(glow);
          this.tweens.add({
            targets: glow,
            scaleX: 1.3, scaleY: 1.3, alpha: 0.06,
            duration: 1500, yoyo: true, repeat: -1,
            ease: "Sine.easeInOut",
          });
        }

        // Status dot
        const pad = this.add.circle(0, 0, 11, 0x0f161b, 0.78);
        pad.setStrokeStyle(1.2, 0xe7f1f7, 0.26);
        container.add(pad);
        const dot = this.add.circle(0, 0, 7, color, 0.94);
        dot.setStrokeStyle(1.5, 0xffffff, 0.68);
        container.add(dot);
        const gleam = this.add.circle(-2, -2, 2.4, 0xffffff, 0.38);
        container.add(gleam);

        // Pulse ring for running
        if (tone === "running") {
          const ring = this.add.circle(0, 0, 10, color, 0);
          ring.setStrokeStyle(2, color, 0.6);
          container.add(ring);
          this.tweens.add({
            targets: ring,
            scaleX: 2.0, scaleY: 2.0, alpha: 0,
            duration: 1400, repeat: -1, ease: "Quad.easeOut",
          });
        }

        // Error flash
        if (tone === "attention") {
          this.tweens.add({
            targets: dot, alpha: 0.3,
            duration: 500, yoyo: true, repeat: -1,
          });
        }

        // Name label
        const shortName = eq.name.length > 14 ? eq.name.slice(0, 12) + ".." : eq.name;
        const nameLabel = this.add.text(0, 11, shortName, {
          fontSize: "6px",
          fontFamily: '"IBM Plex Mono", monospace',
          color: "#9aa8b5",
          stroke: "#2a2a2e",
          strokeThickness: 2,
        });
        nameLabel.setOrigin(0.5, 0);
        container.add(nameLabel);

        // Tooltip on hover
        const tooltip = this.add.text(12, -8, `${eq.name}\n${eq.status}`, {
          fontSize: "8px",
          fontFamily: '"IBM Plex Mono", monospace',
          color: "#e0e0e0",
          backgroundColor: "#222222ee",
          padding: { x: 5, y: 3 },
        });
        tooltip.setVisible(false);
        container.add(tooltip);

        dot.setInteractive({ useHandCursor: true });
        dot.on("pointerover", () => {
          tooltip.setVisible(true);
          this.hoveredEquipmentId = eq.id;
          this.tweens.add({
            targets: container,
            scaleX: 1.08,
            scaleY: 1.08,
            duration: 120,
            ease: "Quad.easeOut",
          });
        });
        dot.on("pointerout", () => {
          tooltip.setVisible(false);
          if (this.hoveredEquipmentId === eq.id) {
            this.hoveredEquipmentId = null;
          }
          this.tweens.add({
            targets: container,
            scaleX: 1,
            scaleY: 1,
            duration: 140,
            ease: "Quad.easeOut",
          });
        });
        dot.on("pointerup", () => {
          if (!this.isDragging) {
            this.onClick?.("equipment", eq.id);
          }
        });

        this.equipOverlays.push(container);
      });
    }
  }

  private setupRoomLabels(world: WorldState): void {
    for (const t of this.roomLabels) t.destroy();
    this.roomLabels = [];

    for (const [roomId, rect] of Object.entries(this.roomRects)) {
      const room = world.lab.rooms[roomId];
      if (!room) continue;

      const px = (rect.tx + 0.4) * TILE;
      const py = (rect.ty + 0.2) * TILE;
      const profile = roomVisualProfile(room);
      const headerColor = 0x182028;
      const borderColor = hexToNumber(profile.accent);

      // Room name with background
      const bgG = this.add.graphics();
      bgG.setDepth(DEPTH.ROOM_LABELS);
      const name = this.add.text(px + 8, py + 4, room.name, {
        fontSize: "13px",
        fontFamily: '"Avenir Next", "Segoe UI", sans-serif',
        color: "#eef4f8",
        fontStyle: "bold",
      });
      name.setDepth(DEPTH.ROOM_LABELS);
      const bgW = name.width + 16;
      const bgH = name.height + 8;
      bgG.fillStyle(headerColor, 0.94);
      bgG.fillRoundedRect(px, py, bgW, bgH, 4);
      bgG.lineStyle(1, borderColor, 0.4);
      bgG.strokeRoundedRect(px, py, bgW, bgH, 4);
      bgG.fillStyle(hexToNumber(profile.accent), 0.85);
      bgG.fillRect(px, py, bgW, 3);
      this.roomLabels.push(bgG, name);

      // BSL badge
      const bslColor = BSL_BADGE_COLORS[room.bsl_level] ?? 0x7b8793;
      const bslX = px + bgW + 6;
      const bslBg = this.add.graphics();
      bslBg.setDepth(DEPTH.ROOM_LABELS);
      bslBg.fillStyle(bslColor, 0.85);
      bslBg.fillRoundedRect(bslX, py + 1, 38, bgH - 2, 3);
      const bslText = this.add.text(bslX + 19, py + bgH / 2, `BSL-${room.bsl_level}`, {
        fontSize: "8px",
        fontFamily: '"IBM Plex Mono", monospace',
        color: "#ffffff",
        fontStyle: "bold",
      });
      bslText.setOrigin(0.5, 0.5);
      bslText.setDepth(DEPTH.ROOM_LABELS);
      this.roomLabels.push(bslBg, bslText);

      // Room type subtitle
      const meta = this.add.text(px + 8, py + bgH + 2, titleize(room.room_type), {
        fontSize: "8px",
        fontFamily: '"IBM Plex Mono", monospace',
        color: "#b6c5cf",
        stroke: "#1a1e24",
        strokeThickness: 2,
      });
      meta.setDepth(DEPTH.ROOM_LABELS);
      this.roomLabels.push(meta);

      // Occupancy dots
      const people = Object.values(world.persons).filter(
        (p) => p.current_room_id === roomId,
      );
      const occX = (rect.tx + rect.tw - 0.5) * TILE;
      const occY = py + 4;
      const occG = this.add.graphics();
      occG.setDepth(DEPTH.ROOM_LABELS);
      const maxDots = Math.min(room.max_occupancy, 8);
      for (let i = 0; i < maxDots; i++) {
        const dx = occX - (maxDots - i) * 8;
        if (i < people.length) {
          occG.fillStyle(0x2d8a4e, 0.9);
          occG.fillCircle(dx, occY + 5, 3);
        } else {
          occG.fillStyle(0x4a5568, 0.4);
          occG.fillCircle(dx, occY + 5, 3);
        }
      }
      this.roomLabels.push(occG);
    }
  }

  private setupRoomHitAreas(world: WorldState): void {
    for (const r of this.roomHitAreas) r.destroy();
    this.roomHitAreas = [];

    for (const [roomId, rect] of Object.entries(this.roomRects)) {
      if (!world.lab.rooms[roomId]) continue;

      const hitArea = this.add.rectangle(
        (rect.tx + rect.tw / 2) * TILE,
        (rect.ty + rect.th / 2) * TILE,
        rect.tw * TILE,
        rect.th * TILE,
        0x000000,
        0,
      );
      hitArea.setInteractive({ useHandCursor: true });
      hitArea.on("pointerup", () => {
        if (!this.isDragging) this.onClick?.("room", roomId);
      });
      hitArea.on("pointerover", () => { this.hoveredRoomId = roomId; });
      hitArea.on("pointerout", () => {
        if (this.hoveredRoomId === roomId) this.hoveredRoomId = null;
      });
      this.roomHitAreas.push(hitArea);
    }
  }

  private setupDoorOverlays(world: WorldState): void {
    this.doorAnchors.clear();

    for (const door of world.lab.doors) {
      const anchor = this.generatedLayout?.doorRoutes[door.id]?.visual_anchor ?? this.fallbackDoorAnchor(door);
      if (!anchor) {
        continue;
      }
      this.doorAnchors.set(door.id, { x: anchor.x, y: anchor.y, door });
    }
  }

  private fallbackDoorAnchor(door: Door): { x: number; y: number } | null {
    const left = this.roomRects[door.connects[0]];
    const right = this.roomRects[door.connects[1]];
    if (!left || !right) {
      return null;
    }

    const leftPx = {
      x: left.tx * TILE,
      y: left.ty * TILE,
      w: left.tw * TILE,
      h: left.th * TILE,
    };
    const rightPx = {
      x: right.tx * TILE,
      y: right.ty * TILE,
      w: right.tw * TILE,
      h: right.th * TILE,
    };

    const horizontalOverlapStart = Math.max(leftPx.x, rightPx.x);
    const horizontalOverlapEnd = Math.min(leftPx.x + leftPx.w, rightPx.x + rightPx.w);
    const verticalOverlapStart = Math.max(leftPx.y, rightPx.y);
    const verticalOverlapEnd = Math.min(leftPx.y + leftPx.h, rightPx.y + rightPx.h);

    if (horizontalOverlapEnd > horizontalOverlapStart) {
      const centerX = (horizontalOverlapStart + horizontalOverlapEnd) / 2;
      const centerY =
        leftPx.y < rightPx.y
          ? (leftPx.y + leftPx.h + rightPx.y) / 2
          : (rightPx.y + rightPx.h + leftPx.y) / 2;
      return { x: centerX, y: centerY };
    }

    if (verticalOverlapEnd > verticalOverlapStart) {
      const centerY = (verticalOverlapStart + verticalOverlapEnd) / 2;
      const centerX =
        leftPx.x < rightPx.x
          ? (leftPx.x + leftPx.w + rightPx.x) / 2
          : (rightPx.x + rightPx.w + leftPx.x) / 2;
      return { x: centerX, y: centerY };
    }

    return {
      x: (leftPx.x + leftPx.w / 2 + rightPx.x + rightPx.w / 2) / 2,
      y: (leftPx.y + leftPx.h / 2 + rightPx.y + rightPx.h / 2) / 2,
    };
  }

  private updateRoomHighlight(): void {
    const g = this.roomHighlight;
    const focus = this.focusGraphics;
    if (!g || !focus) return;
    g.clear();
    focus.clear();

    // Hover highlight
    if (this.hoveredRoomId && this.hoveredRoomId !== this.selectedRoomId) {
      const hRect = this.roomRects[this.hoveredRoomId];
      if (hRect) {
        const hpx = hRect.tx * TILE;
        const hpy = hRect.ty * TILE;
        const hpw = hRect.tw * TILE;
        const hph = hRect.th * TILE;
        g.fillStyle(0x1f7a6f, 0.03);
        g.fillRect(hpx, hpy, hpw, hph);
        g.lineStyle(1.5, 0x1f7a6f, 0.35);
        g.strokeRect(hpx - 1, hpy - 1, hpw + 2, hph + 2);
      }
    }

    if (this.selectedEquipmentId) {
      const equipmentAnchor = this.equipmentAnchors.get(this.selectedEquipmentId);
      if (equipmentAnchor) {
        const pulse = 0.5 + 0.5 * Math.sin(this.highlightTime * 0.004);
        focus.lineStyle(2.2, 0xa7ffeb, 0.72 + pulse * 0.2);
        focus.strokeCircle(equipmentAnchor.x, equipmentAnchor.y, 18 + pulse * 2);
        focus.fillStyle(0x3ac6a5, 0.12);
        focus.fillCircle(equipmentAnchor.x, equipmentAnchor.y, 12);
      }
    }

    if (this.hoveredEquipmentId && this.hoveredEquipmentId !== this.selectedEquipmentId) {
      const equipmentAnchor = this.equipmentAnchors.get(this.hoveredEquipmentId);
      if (equipmentAnchor) {
        focus.lineStyle(1.8, 0xc8f8ef, 0.56);
        focus.strokeCircle(equipmentAnchor.x, equipmentAnchor.y, 15);
      }
    }

    if (!this.selectedRoomId) return;
    const rect = this.roomRects[this.selectedRoomId];
    if (!rect) return;

    const px = rect.tx * TILE;
    const py = rect.ty * TILE;
    const pw = rect.tw * TILE;
    const ph = rect.th * TILE;

    const pulse = 0.5 + 0.5 * Math.sin(this.highlightTime * 0.003);

    // Selection border
    const borderAlpha = 0.6 + 0.3 * pulse;
    g.lineStyle(2.5, 0x1f7a6f, borderAlpha);
    g.strokeRect(px - 2, py - 2, pw + 4, ph + 4);

    // Corner accents
    const cornerLen = 12;
    g.lineStyle(3, 0x2ae8c0, borderAlpha * 0.8);
    g.moveTo(px - 3, py - 3 + cornerLen); g.lineTo(px - 3, py - 3); g.lineTo(px - 3 + cornerLen, py - 3);
    g.moveTo(px + pw + 3 - cornerLen, py - 3); g.lineTo(px + pw + 3, py - 3); g.lineTo(px + pw + 3, py - 3 + cornerLen);
    g.moveTo(px - 3, py + ph + 3 - cornerLen); g.lineTo(px - 3, py + ph + 3); g.lineTo(px - 3 + cornerLen, py + ph + 3);
    g.moveTo(px + pw + 3 - cornerLen, py + ph + 3); g.lineTo(px + pw + 3, py + ph + 3); g.lineTo(px + pw + 3, py + ph + 3 - cornerLen);
    g.strokePath();

    // Subtle inner tint
    g.fillStyle(0x1f7a6f, 0.04);
    g.fillRect(px, py, pw, ph);
  }

  private spawnParticles(world: WorldState): void {
    this.particles = [];
    for (const [roomId, rect] of Object.entries(this.roomRects)) {
      const room = world.lab.rooms[roomId];
      if (!room) continue;

      const px = rect.tx * TILE + TILE;
      const py = rect.ty * TILE + TILE;
      const pw = rect.tw * TILE - TILE * 2;
      const ph = rect.th * TILE - TILE * 2;
      const profile = roomVisualProfile(room);
      if (profile.particleColor === null || profile.particleCount === 0) {
        continue;
      }
      const count = profile.particleCount;
      const color = hexToNumber(profile.particleColor);
      const roomType = room.room_type;

      for (let i = 0; i < count; i++) {
        const speed = room.room_type === "cold_room" ? 0.06 : 0.12;
        this.particles.push({
          x: px + Math.random() * pw,
          y: py + Math.random() * ph,
          vx: (Math.random() - 0.5) * speed,
          vy: (Math.random() - 0.5) * speed - (room.room_type === "fume_hood_lab" ? 0.04 : 0),
          color, alpha: 0.1 + Math.random() * 0.15, roomType,
          bx: px, by: py, bw: pw, bh: ph,
        });
      }
    }
  }

  private updateParticles(delta: number): void {
    const g = this.particleGraphics;
    if (!g || this.particles.length === 0) return;
    g.clear();

    const dt = delta * 0.06;
    for (const p of this.particles) {
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx += (Math.random() - 0.5) * 0.003 * dt;
      p.vy += (Math.random() - 0.5) * 0.003 * dt;

      // Wrap within room bounds (not map bounds)
      if (p.x < p.bx) p.x = p.bx + p.bw;
      if (p.x > p.bx + p.bw) p.x = p.bx;
      if (p.y < p.by) p.y = p.by + p.bh;
      if (p.y > p.by + p.bh) p.y = p.by;

      const shimmer = p.alpha * (0.7 + 0.3 * Math.sin(this.highlightTime * 0.002 + p.x * 0.1));
      const radius = p.roomType === "cold_room" ? 1.8 : 1.3;
      g.fillStyle(p.color, shimmer);
      g.fillCircle(p.x, p.y, radius);
    }
  }

  private updateHud(world: WorldState): void {
    if (!this.hudText || !this.hudBg) return;
    if (!this.hudVisible) {
      this.hudText.setVisible(false);
      this.hudBg.clear();
      this.hudBg.setVisible(false);
      return;
    }

    const labName = world.lab.name || "Lab";
    const personCount = Object.keys(world.persons).length;
    const roomCount = Object.keys(world.lab.rooms).length;
    const runningEquipment = Object.values(world.equipment).filter(
      (equipment) => equipmentTone(equipment.status) === "running",
    ).length;
    const attentionEquipment = Object.values(world.equipment).filter(
      (equipment) => equipmentTone(equipment.status) === "attention",
    ).length;

    let focusLabel = "Facility overview";
    let accent = 0x2ae8c0;
    if (this.selectedEquipmentId) {
      const equipment = world.equipment[this.selectedEquipmentId];
      if (equipment) {
        focusLabel = `Equipment · ${equipment.name} · ${titleize(equipment.status)}`;
        accent = EQUIP_STATUS_COLORS[equipmentTone(equipment.status)] ?? accent;
      }
    } else if (this.selectedAgentId) {
      const person = world.persons[this.selectedAgentId];
      if (person) {
        focusLabel = `Operator · ${person.name}${person.current_activity ? ` · ${person.current_activity}` : ""}`;
        accent = 0x78d0bf;
      }
    } else if (this.selectedRoomId) {
      const room = world.lab.rooms[this.selectedRoomId];
      if (room) {
        focusLabel = `Room · ${room.name} · BSL-${room.bsl_level} ${titleize(room.room_type)}`;
        accent = hexToNumber(roomVisualProfile(room).accent);
      }
    }

    this.hudText.setText(
      [
        labName,
        focusLabel,
        `${personCount} personnel · ${roomCount} rooms · ${runningEquipment} running`,
        `${attentionEquipment} attention · ${Object.keys(world.samples).length} samples in system`,
      ].join("\n"),
    );

    this.hudBg.clear();
    const tw = this.hudText.width + 28;
    const th = this.hudText.height + 20;
    this.hudBg.fillStyle(0x0f1418, 0.84);
    this.hudBg.fillRoundedRect(8, 6, tw, th, 6);
    this.hudBg.fillStyle(accent, 0.92);
    this.hudBg.fillRoundedRect(8, 6, tw, 4, 4);
    this.hudBg.lineStyle(1, 0x415464, 0.58);
    this.hudBg.strokeRoundedRect(8, 6, tw, th, 6);
  }

  /** Draw rooms procedurally for non-Cronin labs. */
  private drawProceduralRooms(world: WorldState): void {
    if (!this.proceduralGraphics) {
      this.proceduralGraphics = this.add.graphics();
      this.proceduralGraphics.setDepth(4);
    }
    const g = this.proceduralGraphics;
    g.clear();

    const entries = Object.entries(this.roomRects).filter(([roomId]) => Boolean(world.lab.rooms[roomId]));
    if (entries.length === 0) {
      return;
    }
    const corridorSegments = this.generatedLayout?.corridorSegments ?? [];

    for (const segment of corridorSegments) {
      const sx = segment.tx * TILE;
      const sy = segment.ty * TILE;
      const sw = segment.tw * TILE;
      const sh = segment.th * TILE;
      const isSpine = segment.kind === "spine";
      g.fillStyle(isSpine ? 0x091014 : 0x0b1318, isSpine ? 0.2 : 0.13);
      g.fillRoundedRect(sx, sy, sw, sh, isSpine ? 8 : 5);
      g.lineStyle(1, isSpine ? 0x5c6e79 : 0x41535e, isSpine ? 0.18 : 0.12);
      g.strokeRoundedRect(sx, sy, sw, sh, isSpine ? 8 : 5);
    }

    for (const [roomId, rect] of entries) {
      const room = world.lab.rooms[roomId];
      if (!room) continue;

      const px = rect.tx * TILE;
      const py = rect.ty * TILE;
      const pw = rect.tw * TILE;
      const ph = rect.th * TILE;
      const profile = roomVisualProfile(room);
      const innerX = px + TILE;
      const innerY = py + TILE * 1.6;
      const innerW = pw - TILE * 2;
      const innerH = ph - TILE * 2.2;
      const roomEntries = this.generatedRoomEntries(roomId);

      const drawHorizontalWall = (side: "north" | "south", y: number): void => {
        const openings = roomEntries
          .filter((entry) => entry.side === side)
          .sort((left, right) => left.tx - right.tx);
        let cursor = px + 10;
        const wallEnd = px + pw - 10;
        for (const entry of openings) {
          const openStart = entry.tx * TILE + 4;
          const openEnd = (entry.tx + entry.tw) * TILE - 4;
          if (openStart > cursor) {
            g.lineBetween(cursor, y, openStart, y);
          }
          cursor = openEnd;
        }
        if (cursor < wallEnd) {
          g.lineBetween(cursor, y, wallEnd, y);
        }
      };

      g.lineStyle(1.2, hexToNumber(profile.accent), 0.16);
      g.lineBetween(px + 2, py + 10, px + 2, py + ph - 10);
      g.lineBetween(px + pw - 2, py + 10, px + pw - 2, py + ph - 10);
      drawHorizontalWall("north", py + 2);
      drawHorizontalWall("south", py + ph - 2);

      drawZoneFrames(g, room, innerX, innerY, innerW, innerH);
    }
  }
}
