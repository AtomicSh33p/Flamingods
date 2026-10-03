"""Shared audio I/O and timeline helpers for the song analyzer."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
TOOL_DIR = Path(__file__).parent


def data_dir() -> Path:
    """Where songs.json, songs/ and output/ live: config.json's "data_dir", else results/."""
    config = TOOL_DIR / "config.json"
    if config.exists():
        return Path(json.loads(config.read_text(encoding="utf-8"))["data_dir"])
    return TOOL_DIR / "results"


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # winget installs an alias here, but it only reaches PATH in new shells
    link = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links/ffmpeg.exe"
    if link.exists():
        return str(link)
    raise SystemExit("ffmpeg not found. Install it with: winget install Gyan.FFmpeg")


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    """Decode any audio file to float32 stereo, shape (2, samples)."""
    cmd = [find_ffmpeg(), "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def save_mp3(path: Path, audio: np.ndarray, sr: int = SR, bitrate: str = "192k") -> None:
    cmd = [find_ffmpeg(), "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(sr),
           "-i", "-", "-b:a", bitrate, str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(audio.T, np.float32).tobytes(), check=True)


def mask_to_ranges(mask: np.ndarray, hop_s: float, min_len: float = 0.0,
                   merge_gap: float = 0.0) -> list[tuple[float, float]]:
    """Turn a per-frame boolean mask into (start, end) ranges in seconds."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    ranges = [(s * hop_s, e * hop_s) for s, e in zip(edges[::2], edges[1::2])]
    merged: list[tuple[float, float]] = []
    for s, e in ranges:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


def overlap(a0: float, a1: float, ranges: list[tuple[float, float]]) -> float:
    """Total seconds of [a0, a1] covered by ranges."""
    return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in ranges)


def snap(t: float, grid: np.ndarray) -> float:
    if len(grid) == 0:
        return t
    return float(grid[np.argmin(np.abs(grid - t))])


def r3(x: float) -> float:
    return round(float(x), 3)
