#!/usr/bin/env python3
"""Render a polished LabVerse replay video from a frozen paper trace.

The script is intentionally browser-free. It consumes the same replay JSONL that
drives the public trace viewer, renders a clean publication-style lab plan with
Pillow, and encodes an MP4 with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_BUNDLE = Path("paper_runs/initial-state-perturbation-2")
DEFAULT_RUN = Path(
    "runs/global_one_shot_planner/ollama__gemma4_31b/chemical_spill_response/rep_1"
)
DEFAULT_OUTPUT = Path("paper/videos/headline_labverse_replay.mp4")

W, H = 1920, 1080
FPS = 24

PAPER = "#f7f3ea"
INK = "#202631"
MUTED = "#687181"
SOFT = "#e5dfd3"
LAB_BG = "#101820"
GRID = "#1b2932"
ROOM_FILL = "#e9e5d8"
ROOM_TILE = "#d8d2c2"
ROOM_WALL = "#40515d"
CORRIDOR = "#b6b2a8"
ACCENT = "#2c7c72"
BLUE = "#476f8f"
DANGER = "#c94f44"
WARN = "#d5963f"
OK = "#3b8f62"


ROOM_LAYOUT = {
    "main_wet_lab": (250, 120, 680, 315),
    "tissue_culture": (970, 120, 300, 315),
    "office": (115, 615, 300, 240),
    "chemical_storage": (475, 615, 300, 240),
    "cold_room": (835, 615, 300, 240),
}


@dataclass(frozen=True)
class Fonts:
    title: ImageFont.FreeTypeFont
    h1: ImageFont.FreeTypeFont
    h2: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont
    mono_small: ImageFont.FreeTypeFont


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def load_fonts() -> Fonts:
    sans = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    sans_bold = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    return Fonts(
        title=font(sans_bold, 54),
        h1=font(sans_bold, 38),
        h2=font(sans_bold, 27),
        body=font(sans, 25),
        small=font(sans, 19),
        mono=font(mono, 22),
        mono_small=font(mono, 17),
    )


def load_replay(bundle: Path, run: Path) -> list[dict[str, Any]]:
    replay_path = bundle / run / "replay.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")
    frames = [json.loads(line) for line in replay_path.read_text().splitlines() if line.strip()]
    if not frames:
        raise ValueError(f"Replay file is empty: {replay_path}")
    return frames


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 1, r: int = 16) -> None:
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font_obj: ImageFont.ImageFont, fill: str = INK, anchor: str | None = None) -> None:
    draw.text(xy, value, font=font_obj, fill=fill, anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, value: str, font_obj: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font_obj) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def room_name(room_id: str, world: dict[str, Any]) -> str:
    return world["lab"]["rooms"].get(room_id, {}).get("name", room_id.replace("_", " ").title())


def room_rect(room_id: str) -> tuple[int, int, int, int]:
    return ROOM_LAYOUT.get(room_id, (100, 100, 240, 180))


def room_center(room_id: str) -> tuple[int, int]:
    x, y, w, h = room_rect(room_id)
    return x + w // 2, y + h // 2


def draw_lab_base(draw: ImageDraw.ImageDraw, world: dict[str, Any], fonts: Fonts) -> None:
    lab = (56, 142, 1288, 918)
    rounded(draw, lab, LAB_BG, None, r=22)
    for x in range(lab[0], lab[2], 32):
        draw.line((x, lab[1], x, lab[3]), fill=GRID, width=1)
    for y in range(lab[1], lab[3], 32):
        draw.line((lab[0], y, lab[2], y), fill=GRID, width=1)

    corridor = (110, 470, 1235, 590)
    rounded(draw, corridor, CORRIDOR, None, r=2)
    for x in range(corridor[0] + 6, corridor[2], 24):
        draw.line((x, corridor[1], x - 18, corridor[3]), fill="#a9a59b", width=1)

    for room_id in world["lab"]["rooms"]:
        x, y, w, h = room_rect(room_id)
        draw.rectangle((x, y, x + w, y + h), fill=ROOM_WALL)
        inset = 14
        draw.rectangle((x + inset, y + inset, x + w - inset, y + h - inset), fill=ROOM_FILL)
        for tx in range(x + inset, x + w - inset, 24):
            draw.line((tx, y + inset, tx, y + h - inset), fill=ROOM_TILE, width=1)
        for ty in range(y + inset, y + h - inset, 24):
            draw.line((x + inset, ty, x + w - inset, ty), fill=ROOM_TILE, width=1)
        draw.rectangle((x + 18, y + 18, x + min(w - 18, 235), y + 48), fill="#263744")
        text(draw, (x + 28, y + 25), room_name(room_id, world), fonts.mono_small, "#f4f0e8")
        draw.rectangle((x + w - 78, y + 18, x + w - 22, y + 48), fill=ACCENT)
        text(draw, (x + w - 50, y + 25), "BSL", fonts.mono_small, "#ecfff7", anchor=None)

    # Short doorway connectors to the central corridor. Avoid long diagonal
    # graph edges; the video should read like a lab floorplan, not a network map.
    corridor_top = corridor[1]
    corridor_bottom = corridor[3]
    for room_id, side in [
        ("main_wet_lab", "bottom"),
        ("tissue_culture", "bottom"),
        ("office", "top"),
        ("chemical_storage", "top"),
        ("cold_room", "top"),
    ]:
        if room_id not in world["lab"]["rooms"]:
            continue
        x, y, w, h = room_rect(room_id)
        cx = x + w // 2
        if side == "bottom":
            start_y, end_y = y + h, corridor_top
        else:
            start_y, end_y = y, corridor_bottom
        draw.line((cx, start_y, cx, end_y), fill="#6c7c86", width=6)
        draw.line((cx, start_y, cx, end_y), fill="#2f4654", width=2)


def slot(room_id: str, index: int, kind: str) -> tuple[int, int]:
    x, y, w, h = room_rect(room_id)
    if kind == "person":
        positions = [
            (x + w * 0.28, y + h * 0.70),
            (x + w * 0.42, y + h * 0.70),
            (x + w * 0.56, y + h * 0.70),
            (x + w * 0.70, y + h * 0.70),
        ]
    elif kind == "equipment":
        positions = [
            (x + w * 0.24, y + h * 0.34),
            (x + w * 0.42, y + h * 0.34),
            (x + w * 0.60, y + h * 0.34),
            (x + w * 0.78, y + h * 0.34),
        ]
    else:
        positions = [
            (x + w * 0.24, y + h * 0.78),
            (x + w * 0.38, y + h * 0.78),
            (x + w * 0.52, y + h * 0.78),
            (x + w * 0.66, y + h * 0.78),
        ]
    px, py = positions[index % len(positions)]
    return int(px), int(py)


def draw_equipment(draw: ImageDraw.ImageDraw, world: dict[str, Any], fonts: Fonts) -> None:
    for room_id, room in world["lab"]["rooms"].items():
        for idx, equipment_id in enumerate(room.get("equipment_ids") or []):
            x, y = slot(room_id, idx, "equipment")
            equipment = world.get("equipment", {}).get(equipment_id, {})
            status = equipment.get("status", "ready")
            fill = "#8ba0a7" if status == "ready" else OK
            rounded(draw, (x - 31, y - 33, x + 31, y + 31), "#546a76", "#2d3d47", r=6)
            draw.rectangle((x - 22, y - 23, x + 22, y + 5), fill=fill)
            draw.rectangle((x - 16, y + 11, x + 16, y + 20), fill="#d9d3c6")
            text(draw, (x, y - 55), equipment_id.replace("_", " "), fonts.mono_small, "#29313b", anchor="mm")


def draw_materials(draw: ImageDraw.ImageDraw, world: dict[str, Any], highlight_removed: bool) -> None:
    for room_id, room in world["lab"]["rooms"].items():
        for idx, material_id in enumerate(room.get("material_ids") or []):
            x, y = slot(room_id, idx, "material")
            color = DANGER if "hcl" in material_id else "#2ba7a3"
            if material_id == "hcl_conc":
                draw.ellipse((x - 34, y - 34, x + 34, y + 34), outline=DANGER, width=5)
            draw.rectangle((x - 9, y - 25, x + 9, y - 15), fill="#d8eef2")
            rounded(draw, (x - 17, y - 16, x + 17, y + 24), color, "#263744", r=4)
            draw.rectangle((x - 12, y - 5, x + 12, y + 7), fill="#f8f2df")
    if highlight_removed:
        x, y = slot("chemical_storage", 0, "material")
        draw.line((x - 34, y - 34, x + 34, y + 34), fill=OK, width=7)
        draw.line((x - 34, y + 34, x + 34, y - 34), fill=OK, width=7)


def draw_people(draw: ImageDraw.ImageDraw, world: dict[str, Any], actor_id: str, fonts: Fonts) -> tuple[int, int] | None:
    by_room: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for person_id, person in world.get("persons", {}).items():
        by_room.setdefault(person.get("current_room_id", "office"), []).append((person_id, person))
    actor_pos: tuple[int, int] | None = None
    for room_id, people in by_room.items():
        for idx, (person_id, person) in enumerate(sorted(people)):
            x, y = slot(room_id, idx, "person")
            active = person_id == actor_id
            if active:
                actor_pos = (x, y)
                draw.ellipse((x - 34, y - 48, x + 34, y + 42), outline=ACCENT, width=5)
            draw.ellipse((x - 16, y - 38, x + 16, y - 6), fill="#c88a52", outline="#27343d", width=2)
            coat = "#e7edf2" if "lab_coat" in person.get("current_ppe", []) else "#6d7a87"
            rounded(draw, (x - 23, y - 6, x + 23, y + 36), coat, "#27343d", r=8)
            if "safety_glasses" in person.get("current_ppe", []):
                draw.line((x - 13, y - 25, x + 13, y - 25), fill=BLUE, width=4)
            if "face_shield" in person.get("current_ppe", []):
                draw.arc((x - 20, y - 42, x + 20, y - 2), 200, 340, fill=WARN, width=3)
            text(draw, (x, y + 52), person.get("name", person_id).split()[0], fonts.mono_small, "#202631", anchor="mm")
    return actor_pos


def action_label(frame: dict[str, Any]) -> str:
    candidate = str(frame.get("resolved_candidate_ref") or "")
    params = frame.get("resolved_action_parameters") or {}
    verb = str(frame.get("resolved_verb") or candidate.replace("planned::", ""))
    if candidate == "global_one_shot_plan":
        return "Generate complete action plan"
    if verb == "don_ppe":
        return f"Put on {str(params.get('ppe_type', 'PPE')).replace('_', ' ')}"
    if verb == "move_to":
        before = frame.get("world_before", {}).get("persons", {}).get(frame.get("actor_id"), {}).get("current_room_id")
        after = frame.get("world_after", {}).get("persons", {}).get(frame.get("actor_id"), {}).get("current_room_id")
        return f"Move from {pretty_id(before)} to {pretty_id(after)}"
    if verb == "dispose_material":
        return "Dispose concentrated HCl as chemical waste"
    if verb == "log_observation":
        return "Record incident report"
    return pretty_id(verb)


def pretty_id(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).replace("_", " ").replace("planned::", "").title()


def draw_header(draw: ImageDraw.ImageDraw, fonts: Fonts, frame: dict[str, Any] | None = None) -> None:
    text(draw, (64, 44), "LabVerse", fonts.title, INK)
    text(draw, (64, 103), "Stateful replay of a local LLM laboratory plan", fonts.h2, MUTED)
    text(draw, (1510, 64), "frozen trace", fonts.mono, ACCENT)
    text(draw, (1510, 96), "chemical spill response", fonts.mono_small, MUTED)
    if frame:
        text(draw, (1150, 97), f"step {frame.get('step_index', '?'):>2}", fonts.mono, INK)


def draw_story_panel(draw: ImageDraw.ImageDraw, frame: dict[str, Any], frames: list[dict[str, Any]], fonts: Fonts) -> None:
    panel = (1328, 142, 1864, 918)
    rounded(draw, panel, "#fffdf8", SOFT, width=2, r=22)
    text(draw, (1364, 184), "What the replay shows", fonts.h2, INK)
    bullets = [
        "Every model output is replayed against the current action surface.",
        "The world state updates after each executable lab action.",
        "PPE, location, materials, and reports remain inspectable step by step.",
    ]
    y = 232
    for bullet in bullets:
        draw.ellipse((1366, y + 8, 1378, y + 20), fill=ACCENT)
        for line in wrap(draw, bullet, fonts.small, 430):
            text(draw, (1395, y), line, fonts.small, INK)
            y += 25
        y += 12

    y += 24
    text(draw, (1364, y), "Current action", fonts.h2, INK)
    y += 48
    rounded(draw, (1364, y, 1828, y + 118), "#f3eee4", "#ddd3c2", width=2, r=16)
    for line in wrap(draw, action_label(frame), fonts.body, 410)[:3]:
        text(draw, (1390, y + 24), line, fonts.body, INK)
        y += 30

    y = 650
    text(draw, (1364, y), "Trace outcome", fonts.h2, INK)
    y += 48
    progress = max(0, min(1, int(frame.get("step_index", 1)) / max(1, len(frames))))
    draw.rounded_rectangle((1364, y, 1828, y + 18), radius=9, fill="#e5dfd3")
    draw.rounded_rectangle((1364, y, 1364 + int(464 * progress), y + 18), radius=9, fill=ACCENT)
    y += 44
    text(draw, (1364, y), "successful run: all required criteria satisfied", fonts.small, OK)
    text(draw, (1364, y + 34), "model: gemma4:31b  |  planner: global one-shot", fonts.mono_small, MUTED)


def draw_timeline(draw: ImageDraw.ImageDraw, frames: list[dict[str, Any]], current_index: int, fonts: Fonts) -> None:
    x0, y0, x1 = 72, 970, 1848
    draw.line((x0, y0, x1, y0), fill=SOFT, width=5)
    count = len(frames)
    for i, frame in enumerate(frames):
        x = x0 + int((x1 - x0) * i / max(1, count - 1))
        active = i == current_index
        radius = 16 if active else 10
        draw.ellipse((x - radius, y0 - radius, x + radius, y0 + radius), fill=ACCENT if active else "#c9c2b5", outline=PAPER, width=4)
        if active:
            text(draw, (x, y0 + 32), str(frame.get("step_index", i + 1)), fonts.mono_small, INK, anchor="mm")


def render_replay_frame(frame: dict[str, Any], frames: list[dict[str, Any]], index: int, fonts: Fonts) -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, frame)
    world = frame["world_after"]
    draw_lab_base(draw, world, fonts)
    draw_equipment(draw, world, fonts)
    draw_materials(draw, world, highlight_removed=frame.get("resolved_candidate_ref") == "planned::dispose_material")
    actor_pos = draw_people(draw, world, str(frame.get("actor_id") or "tech_johnson"), fonts)
    if actor_pos:
        x, y = actor_pos
        rounded(draw, (x - 145, max(150, y - 135), x + 145, max(218, y - 58)), "#202a31", "#2ae8c0", width=2, r=12)
        for n, line in enumerate(wrap(draw, action_label(frame), fonts.mono_small, 250)[:2]):
            text(draw, (x - 125, max(168, y - 118) + n * 22), line, fonts.mono_small, "#f7fff9")
    draw_story_panel(draw, frame, frames, fonts)
    draw_timeline(draw, frames, index, fonts)
    return img


def render_title(fonts: Fonts) -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts)
    rounded(draw, (220, 230, 1700, 790), "#fffdf8", SOFT, width=2, r=32)
    text(draw, (310, 330), "From model text to laboratory state", fonts.title, INK)
    body = (
        "A frozen LabVerse trace replays each planned action, updates the simulated "
        "laboratory, and keeps every safety-relevant state transition inspectable."
    )
    y = 430
    for line in wrap(draw, body, fonts.h2, 1160):
        text(draw, (310, y), line, fonts.h2, MUTED)
        y += 44
    text(draw, (310, 650), "Scenario: chemical spill response", fonts.mono, ACCENT)
    text(draw, (310, 692), "Planner: global one-shot  |  Model: gemma4:31b", fonts.mono_small, MUTED)
    return img


def render_end(fonts: Fonts) -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts)
    rounded(draw, (280, 250, 1640, 760), "#fffdf8", SOFT, width=2, r=32)
    text(draw, (390, 360), "Trace complete", fonts.title, INK)
    text(draw, (390, 455), "PPE satisfied. Hazard removed. Incident documented.", fonts.h1, OK)
    text(draw, (390, 555), "The same replay artifact links prompts, model outputs, actions, and world states.", fonts.h2, MUTED)
    return img


def write_repeated(img: Image.Image, frames_dir: Path, start: int, count: int) -> int:
    for offset in range(count):
        img.save(frames_dir / f"frame_{start + offset:05d}.png")
    return start + count


def encode_video(frames_dir: Path, output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the headline video.")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seconds-per-step", type=float, default=1.15)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fonts = load_fonts()
    frames = load_replay(args.bundle, args.run)
    step_frame_count = max(1, math.ceil(args.seconds_per_step * args.fps))
    with tempfile.TemporaryDirectory(prefix="labverse-video-") as tmp:
        frames_dir = Path(tmp)
        cursor = 0
        cursor = write_repeated(render_title(fonts), frames_dir, cursor, args.fps * 2)
        for index, frame in enumerate(frames):
            cursor = write_repeated(
                render_replay_frame(frame, frames, index, fonts),
                frames_dir,
                cursor,
                step_frame_count,
            )
        cursor = write_repeated(render_end(fonts), frames_dir, cursor, args.fps * 2)
        encode_video(frames_dir, args.output, args.fps)
        if args.keep_frames:
            saved = args.output.with_suffix("")
            if saved.exists():
                shutil.rmtree(saved)
            shutil.copytree(frames_dir, saved)
            print(f"frames: {saved}")
    print(f"video: {args.output}")


if __name__ == "__main__":
    main()
