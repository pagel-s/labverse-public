#!/usr/bin/env python3
"""Render the README replay preview from the real LabVerse frontend.

The script opens the public paper viewer in capture mode, lets the existing
Phaser/LabScene renderer draw each replay step, screenshots those captured
scenes with Chrome, and encodes the frames with ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import socket
import struct
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
CAPTURE_FPS = 8
STEP_SECONDS = 1.9
INTRO_SECONDS = 0.8


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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def devtools_ports(is_windows_browser: bool) -> list[int]:
    if is_windows_browser:
        return [9333, 9334, 9335, 9336, 19333, 9222]
    return [free_port()]


def cleanup_windows_capture_chrome() -> None:
    powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not powershell.exists():
        return
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'chrome.exe' "
        "-and $_.CommandLine -match '--headless=new' "
        "-and $_.CommandLine -match 'labverse-real-capture' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(
        [str(powershell), "-NoProfile", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def devtools_port_in_use(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=0.4):
            return True
    except Exception:
        return False


def replay_steps(bundle: Path, run: Path) -> list[int]:
    replay = ROOT / bundle / run / "replay.jsonl"
    if not replay.exists():
        raise FileNotFoundError(f"Replay file not found: {replay}")
    count = sum(1 for line in replay.read_text().splitlines() if line.strip())
    if count == 0:
        raise RuntimeError(f"Replay file is empty: {replay}")
    return list(range(1, count + 1))


def capture_url(bundle: Path, run: Path, step: int, *, autoplay: bool) -> str:
    query = urllib.parse.urlencode(
        {
            "capture": "1",
            "bundle": str(bundle),
            "run": str(run),
            "step": str(step),
            "mode": "after",
            "autoplay": "1" if autoplay else "0",
        }
    )
    return f"{FRONTEND_URL}/?{query}"


class CdpClient:
    def __init__(self, websocket_url: str) -> None:
        parsed = urllib.parse.urlparse(websocket_url)
        if parsed.scheme != "ws" or not parsed.hostname or not parsed.port:
            raise RuntimeError(f"Unsupported DevTools websocket URL: {websocket_url}")
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=5)
        self.next_id = 1
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError("Chrome DevTools websocket handshake failed.")

    def close(self) -> None:
        self.sock.close()

    def call(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        message_id = self.next_id
        self.next_id += 1
        self._send_json({"id": message_id, "method": method, "params": params or {}})
        while True:
            message = json.loads(self._recv_text())
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"CDP call failed for {method}: {message['error']}")
                result = message.get("result", {})
                return result if isinstance(result, dict) else {}

    def _send_json(self, value: dict[str, object]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = secrets.token_bytes(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_exact(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self.sock.recv(count - len(chunks))
            if not chunk:
                raise RuntimeError("DevTools websocket closed unexpectedly.")
            chunks.extend(chunk)
        return bytes(chunks)

    def _recv_text(self) -> str:
        chunks: list[bytes] = []
        while True:
            first, second = self._recv_exact(2)
            fin = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 8:
                raise RuntimeError("DevTools websocket closed.")
            if opcode in (1, 0):
                chunks.append(payload)
            if fin and chunks:
                return b"".join(chunks).decode("utf-8")


def candidate_devtools_hosts(is_windows_browser: bool) -> list[str]:
    hosts = ["127.0.0.1", "localhost"]
    if not is_windows_browser:
        return hosts
    for command in (["hostname", "-I"],):
        try:
            output = subprocess.check_output(command, text=True).strip()
        except Exception:
            continue
        hosts.extend(part for part in output.split() if "." in part)
    try:
        resolv = Path("/etc/resolv.conf").read_text()
    except Exception:
        resolv = ""
    for line in resolv.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "nameserver":
            hosts.append(parts[1])
    return list(dict.fromkeys(hosts))


def wait_for_devtools(port: int, hosts: list[str]) -> tuple[str, list[dict[str, object]]]:
    deadline = time.time() + 20
    last_error: Exception | None = None
    while time.time() < deadline:
        for host in hosts:
            url = f"http://{host}:{port}/json"
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    if isinstance(data, list) and data:
                        return host, data
            except Exception as exc:
                last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Chrome DevTools did not become ready: {last_error}")


def wait_for_capture_ready(client: CdpClient) -> None:
    deadline = time.time() + 25
    while time.time() < deadline:
        result = client.call(
            "Runtime.evaluate",
            {
                "expression": "document.body.dataset.captureReady === '1'",
                "returnByValue": True,
            },
        )
        result_value = result.get("result")
        value = result_value.get("value") if isinstance(result_value, dict) else None
        if value is True:
            return
        time.sleep(0.25)
    raise RuntimeError("Capture page did not report ready state.")


def capture_frames(chrome: Path, bundle: Path, run: Path, steps: list[int], out_dir: Path) -> list[Path]:
    frames: list[Path] = []
    profile_dir = frame_dir() / "chrome-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    is_windows_browser = str(chrome).startswith("/mnt/")
    chrome_process: subprocess.Popen[str] | None = None
    client: CdpClient | None = None
    try:
        if is_windows_browser:
            cleanup_windows_capture_chrome()
        devtools_host: str | None = None
        targets: list[dict[str, object]] | None = None
        last_error: Exception | None = None
        for port in devtools_ports(is_windows_browser):
            if devtools_port_in_use(port):
                continue
            chrome_process = subprocess.Popen(
                [
                    str(chrome),
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--mute-audio",
                    "--window-size=860,920",
                    "--remote-debugging-address=0.0.0.0",
                    f"--remote-debugging-port={port}",
                    f"--user-data-dir={windows_path(profile_dir) if is_windows_browser else str(profile_dir)}",
                    "about:blank",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                devtools_host, targets = wait_for_devtools(
                    port,
                    candidate_devtools_hosts(is_windows_browser),
                )
                break
            except Exception as exc:
                last_error = exc
                chrome_process.terminate()
                try:
                    chrome_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    chrome_process.kill()
                chrome_process = None
        if devtools_host is None or targets is None:
            raise RuntimeError(f"Chrome DevTools did not become ready: {last_error}")
        page = next((target for target in targets if target.get("type") == "page"), targets[0])
        websocket_url = page.get("webSocketDebuggerUrl")
        if not isinstance(websocket_url, str):
            raise RuntimeError("Chrome did not expose a page websocket.")
        websocket_url = websocket_url.replace("ws://127.0.0.1:", f"ws://{devtools_host}:")
        websocket_url = websocket_url.replace("ws://localhost:", f"ws://{devtools_host}:")
        client = CdpClient(websocket_url)
        client.call("Page.enable")
        client.call("Runtime.enable")
        client.call("Page.navigate", {"url": capture_url(bundle, run, steps[0], autoplay=True)})
        wait_for_capture_ready(client)
        time.sleep(0.4)

        total_seconds = INTRO_SECONDS + len(steps) * STEP_SECONDS
        frame_count = int(total_seconds * CAPTURE_FPS)
        next_frame_at = time.monotonic()
        for index in range(1, frame_count + 1):
            data_url: str | None = None
            for _attempt in range(10):
                result = client.call(
                    "Runtime.evaluate",
                    {
                        "expression": "document.querySelector('.paper-capture-stage canvas')?.toDataURL('image/png') ?? ''",
                        "returnByValue": True,
                    },
                )
                result_value = result.get("result")
                candidate = result_value.get("value") if isinstance(result_value, dict) else None
                if isinstance(candidate, str) and "," in candidate:
                    data_url = candidate
                    break
                time.sleep(0.1)
            if data_url is None:
                raise RuntimeError("Chrome returned an empty canvas frame.")
            data = data_url.split(",", 1)[1]
            if not isinstance(data, str):
                raise RuntimeError("Chrome returned an empty canvas frame.")
            frame = out_dir / f"frame_{index:03d}.png"
            frame.write_bytes(base64.b64decode(data))
            frames.append(frame)
            next_frame_at += 1 / CAPTURE_FPS
            time.sleep(max(0, next_frame_at - time.monotonic()))
        return frames
    finally:
        if client is not None:
            client.close()
        if chrome_process is not None:
            chrome_process.terminate()
            try:
                chrome_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                chrome_process.kill()
        if is_windows_browser:
            cleanup_windows_capture_chrome()


def encode(frames: list[Path], mp4: Path, gif: Path) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames[0].parent / "frame_%03d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(CAPTURE_FPS),
            "-i",
            pattern,
            "-vf",
            "scale=1440:-2:flags=lanczos,format=yuv420p",
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
            str(CAPTURE_FPS),
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
