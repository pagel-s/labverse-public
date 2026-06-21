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
  ROOM_HIGHLIGHT: 5,
  EQUIP: 10,
  ACCESS: 12,
  ROOM_LABELS: 15,
  FOCUS: 18,
  AGENTS: 20,
} as const;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function equipmentMarkerPosition(
  room: Room,
  rect: TileRect,
  equipment: Equipment,
  index: number,
  count: number,
): { x: number; y: number } {
  const px = rect.tx * TILE;
  const py = rect.ty * TILE;
  const pw = rect.tw * TILE;
  const ph = rect.th * TILE;
  const minX = px + TILE * 1.75;
  const maxX = px + pw - TILE * 1.75;
  const minY = py + TILE * 2.6;
  const maxY = py + ph - TILE * 2.2;
  const zone = equipment.zone_id
    ? room.zones.find((candidate) => candidate.id === equipment.zone_id)
    : null;

  if (zone) {
    const scaleX = pw / Math.max(room.width, 1);
    const scaleY = ph / Math.max(room.height, 1);
    const zoneWidth = Math.max(TILE * 1.5, zone.width * scaleX);
    const columns = Math.max(1, Math.min(4, Math.floor(zoneWidth / (TILE * 1.35))));
    const col = index % columns;
    const row = Math.floor(index / columns);
    const rowCount = Math.max(1, Math.ceil(count / columns));
    const rowSpacing = Math.min(TILE * 1.25, (maxY - minY) / rowCount);
    const rawY = py + TILE * 2.1 + zone.position.y * scaleY;
    const baseY = clamp(
      rawY,
      minY,
      Math.max(minY, maxY - (rowCount - 1) * rowSpacing),
    );
    return {
      x: clamp(
        px + TILE + zone.position.x * scaleX + ((col + 1) / (columns + 1)) * zoneWidth,
        minX,
        maxX,
      ),
      y: baseY + row * rowSpacing,
    };
  }

  const columns = Math.max(1, Math.min(4, count));
  const col = index % columns;
  const row = Math.floor(index / columns);
  const usableWidth = Math.max(TILE, maxX - minX);
  return {
    x: clamp(minX + ((col + 1) / (columns + 1)) * usableWidth, minX, maxX),
    y: clamp(minY + TILE * 0.8 + row * TILE * 1.35, minY, maxY),
  };
}

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

    // Highlight graphics
    this.roomHighlight = this.add.graphics();
    this.roomHighlight.setDepth(DEPTH.ROOM_HIGHLIGHT);
    this.focusGraphics = this.add.graphics();
    this.focusGraphics.setDepth(DEPTH.FOCUS);

    // Camera controls
    this.setupCamera(this.mapW, this.mapH);

    this.mapReady = true;

    if (this.onCreateCallback) {
      this.onCreateCallback();
      this.onCreateCallback = null;
    }
  }

  update(_time: number, delta: number): void {
    this.highlightTime += delta;
    this.updateRoomHighlight();
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
  }

  syncPersons(world: WorldState): void {
    if (!this.mapReady) return;

    const byRoom = groupPersonsByRoom(world);
    const seen = new Set<string>();

    for (const [roomId, persons] of byRoom) {
      const room = world.lab.rooms[roomId];
      if (!room) continue;

      const zoneIndexes = new Map<string, number>();
      persons.forEach((person) => {
        seen.add(person.id);
        const zoneKey = person.current_zone_id ?? "__room__";
        const zoneIndex = zoneIndexes.get(zoneKey) ?? 0;
        zoneIndexes.set(zoneKey, zoneIndex + 1);
        const pos = getPersonPixelPos(
          this.roomRects,
          room,
          roomId,
          person.current_zone_id,
          zoneIndex,
        );
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

  captureScene(): Promise<string> {
    for (const agent of this.agents.values()) {
      agent.refreshActivityBubble();
    }

    this.cameras.main.setBackgroundColor(0xfbfaf7);
    return new Promise((resolve, reject) => {
      this.game.renderer.snapshot((snapshot) => {
        this.cameras.main.setBackgroundColor(BG_COLOR);
        if (snapshot instanceof HTMLImageElement) {
          resolve(this.prepareCaptureImage(snapshot));
          return;
        }
        reject(new Error("The replay renderer did not return an image."));
      }, "image/png");
    });
  }

  private prepareCaptureImage(snapshot: HTMLImageElement): string {
    const source = document.createElement("canvas");
    source.width = snapshot.naturalWidth;
    source.height = snapshot.naturalHeight;
    const sourceContext = source.getContext("2d", { willReadFrequently: true });
    if (!sourceContext) return snapshot.src;
    sourceContext.drawImage(snapshot, 0, 0);

    const pixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
    let minX = source.width;
    let minY = source.height;
    let maxX = -1;
    let maxY = -1;
    const stride = 2;
    for (let y = 0; y < source.height; y += stride) {
      for (let x = 0; x < source.width; x += stride) {
        const offset = (y * source.width + x) * 4;
        const differsFromPaper =
          Math.abs(pixels[offset] - 251) > 8 ||
          Math.abs(pixels[offset + 1] - 250) > 8 ||
          Math.abs(pixels[offset + 2] - 247) > 8;
        if (pixels[offset + 3] > 16 && differsFromPaper) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    if (maxX < minX || maxY < minY) return snapshot.src;
    const padding = 20;
    const cropX = Math.max(0, minX - padding);
    const cropY = Math.max(0, minY - padding);
    const cropWidth = Math.min(source.width - cropX, maxX - minX + padding * 2);
    const cropHeight = Math.min(source.height - cropY, maxY - minY + padding * 2);
    const scale = Math.max(1, Math.min(2, 4096 / cropWidth, 4096 / cropHeight));

    const output = document.createElement("canvas");
    output.width = Math.round(cropWidth * scale);
    output.height = Math.round(cropHeight * scale);
    const outputContext = output.getContext("2d");
    if (!outputContext) return snapshot.src;
    outputContext.imageSmoothingEnabled = false;
    outputContext.drawImage(
      source,
      cropX,
      cropY,
      cropWidth,
      cropHeight,
      0,
      0,
      output.width,
      output.height,
    );
    return output.toDataURL("image/png");
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

      const zoneIndexes = new Map<string, number>();
      const zoneCounts = new Map<string, number>();
      const orderedEquipment = [...eqs].sort((left, right) => left.id.localeCompare(right.id));
      for (const equipment of orderedEquipment) {
        const key = equipment.zone_id ?? "__room__";
        zoneCounts.set(key, (zoneCounts.get(key) ?? 0) + 1);
      }

      orderedEquipment.forEach((eq) => {
        const tone = equipmentTone(eq.status);
        const color = EQUIP_STATUS_COLORS[tone] ?? EQUIP_STATUS_COLORS.idle;
        const zoneKey = eq.zone_id ?? "__room__";
        const zoneIndex = zoneIndexes.get(zoneKey) ?? 0;
        zoneIndexes.set(zoneKey, zoneIndex + 1);
        const { x: ex, y: ey } = equipmentMarkerPosition(
          room,
          rect,
          eq,
          zoneIndex,
          zoneCounts.get(zoneKey) ?? 1,
        );
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

        // Tooltip on hover
        const tooltip = this.add.text(12, -8, `${eq.name}\n${eq.status}`, {
          fontSize: "10px",
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
          container.setDepth(DEPTH.AGENTS + 2);
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
          container.setDepth(DEPTH.EQUIP);
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

      const roomX = rect.tx * TILE;
      const roomWidth = rect.tw * TILE;
      const px = roomX + 10;
      const py = (rect.ty + 0.2) * TILE;
      const profile = roomVisualProfile(room);
      const headerColor = 0x182028;
      const borderColor = hexToNumber(profile.accent);
      const bslWidth = 38;
      const bslX = roomX + roomWidth - bslWidth - 10;
      const maxNameWidth = Math.max(52, bslX - px - 6);

      // Room name with background
      const bgG = this.add.graphics();
      bgG.setDepth(DEPTH.ROOM_LABELS);
      const name = this.add.text(px + 8, py + 4, room.name, {
        fontSize: "15px",
        fontFamily: '"Avenir Next", "Segoe UI", sans-serif',
        color: "#eef4f8",
        fontStyle: "bold",
      });
      name.setDepth(DEPTH.ROOM_LABELS);
      while (name.width + 16 > maxNameWidth && name.text.length > 8) {
        name.setText(name.text.slice(0, -2).trimEnd() + "…");
      }
      const bgW = Math.min(name.width + 16, maxNameWidth);
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
      const bslBg = this.add.graphics();
      bslBg.setDepth(DEPTH.ROOM_LABELS);
      bslBg.fillStyle(bslColor, 0.85);
      bslBg.fillRoundedRect(bslX, py + 1, bslWidth, bgH - 2, 3);
      const bslText = this.add.text(bslX + 19, py + bgH / 2, `BSL-${room.bsl_level}`, {
        fontSize: "9px",
        fontFamily: '"IBM Plex Mono", monospace',
        color: "#ffffff",
        fontStyle: "bold",
      });
      bslText.setOrigin(0.5, 0.5);
      bslText.setDepth(DEPTH.ROOM_LABELS);
      this.roomLabels.push(bslBg, bslText);

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
