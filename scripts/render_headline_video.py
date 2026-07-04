#!/usr/bin/env python3
"""Render the README replay preview from the real LabVerse frontend.

The script opens the public paper viewer in capture mode, lets the existing
Phaser/LabScene renderer draw each replay step, screenshots those captured
scenes with Chrome, and encodes the frames with ffmpeg.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("paper_runs/initial-state-perturbation-2")
DEFAULT_RUN = Path(
    "runs/global_one_shot_planner/ollama__gemma4_31b/chemical_spill_response/rep_1"
)
DEFAULT_MP4 = Path("paper/videos/headline_labverse_replay.mp4")
DEFAULT_GIF = Path("paper/videos/headline_labverse_replay_preview.gif")
BACKEND_URL = "http://127.0.0.1:8001/api/paper/health"
FRONTEND_URL = "http://127.0.0.1:5173"


def http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def wait_for(url: str, label: str, timeout_s: float = 45.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if http_ready(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"{label} did not become ready at {url}")


def start_backend() -> subprocess.Popen[str] | None:
    if http_ready(BACKEND_URL):
        return None
    labverse = ROOT / ".venv" / "bin" / "labverse"
    if not labverse.exists():
        labverse = Path(shutil.which("labverse") or "")
    if not labverse:
        raise RuntimeError("Could not find labverse. Install the project or create .venv first.")
    process = subprocess.Popen(
        [str(labverse), "paper-serve", "--host", "0.0.0.0", "--port", "8001"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    wait_for(BACKEND_URL, "paper API")
    return process


def start_frontend() -> subprocess.Popen[str] | None:
    if http_ready(FRONTEND_URL):
        return None
    process = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "0.0.0.0"],
        cwd=ROOT / "frontend",
        env={"LABVERSE_PAPER_API_URL": "http://127.0.0.1:8001", **dict(os.environ)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    wait_for(FRONTEND_URL, "frontend")
    return process


def find_chrome() -> Path:
    candidates = [
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise RuntimeError("Could not find Chrome/Chromium. Install Chromium or Chrome first.")


def windows_path(path: Path) -> str:
    if not str(path).startswith("/mnt/"):
        return str(path)
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def frame_dir() -> Path:
    windows_tmp = Path("/mnt/c/Users/pagel/AppData/Local/Temp")
    if windows_tmp.exists():
        return Path(tempfile.mkdtemp(prefix="labverse-real-capture-", dir=windows_tmp))
    return Path(tempfile.mkdtemp(prefix="labverse-real-capture-"))


def replay_steps(bundle: Path, run: Path) -> list[int]:
    replay = ROOT / bundle / run / "replay.jsonl"
    if not replay.exists():
        raise FileNotFoundError(f"Replay file not found: {replay}")
    count = sum(1 for line in replay.read_text().splitlines() if line.strip())
    if count == 0:
        raise RuntimeError(f"Replay file is empty: {replay}")
    return list(range(1, count + 1))


def capture_url(bundle: Path, run: Path, step: int) -> str:
    query = urllib.parse.urlencode(
        {
            "capture": "1",
            "bundle": str(bundle),
            "run": str(run),
            "step": str(step),
            "mode": "after",
        }
    )
    return f"{FRONTEND_URL}/?{query}"


def capture_frames(chrome: Path, bundle: Path, run: Path, steps: list[int], out_dir: Path) -> list[Path]:
    frames: list[Path] = []
    is_windows_browser = str(chrome).startswith("/mnt/")
    for index, step in enumerate(steps, start=1):
        frame = out_dir / f"frame_{index:03d}.png"
        screenshot = windows_path(frame) if is_windows_browser else str(frame)
        cmd = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1000,900",
            "--virtual-time-budget=9000",
            f"--screenshot={screenshot}",
            capture_url(bundle, run, step),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not frame.exists() or frame.stat().st_size == 0:
            raise RuntimeError(f"Chrome did not write frame: {frame}")
        frames.append(frame)
    return frames


def encode(frames: list[Path], mp4: Path, gif: Path) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames[0].parent / "frame_%03d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "1",
            "-i",
            pattern,
            "-vf",
            "scale=1920:-2:flags=lanczos,format=yuv420p",
            "-r",
            "24",
            str(mp4),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "1",
            "-i",
            pattern,
            "-vf",
            "scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
            str(gif),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_MP4)
    parser.add_argument("--preview", type=Path, default=DEFAULT_GIF)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = frontend = None
    try:
        backend = start_backend()
        frontend = start_frontend()
        chrome = find_chrome()
        steps = replay_steps(args.bundle, args.run)
        out_dir = frame_dir()
        frames = capture_frames(chrome, args.bundle, args.run, steps, out_dir)
        encode(frames, args.output, args.preview)
        print(f"Rendered {len(frames)} real LabVerse frames")
        print(f"MP4: {args.output}")
        print(f"Preview: {args.preview}")
        return 0
    finally:
        for process in (frontend, backend):
            if process is not None:
                process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
