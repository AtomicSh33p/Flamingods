"""Render the single-file HTML review page for an analysis."""

import json
import os
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).parent / "review_template.html"


def waveform_peaks(mono: np.ndarray, points: int = 1600) -> list[float]:
    chunk = max(1, len(mono) // points)
    usable = len(mono) // chunk * chunk
    peaks = np.abs(mono[:usable]).reshape(-1, chunk).max(axis=1)
    return [round(float(p), 3) for p in peaks / (peaks.max() or 1.0)]


def song_flags(analysis: dict) -> list[str]:
    """Reasons a human should look at this song before trusting its analysis."""
    source, sections = analysis["source"], analysis["sections"]
    duration = source["duration"]
    vocal = sum(a["end"] - a["start"] for a in analysis["activity"] if a["type"] == "vocals")
    words = sum(len(line["words"]) for line in analysis["lyrics"])
    cut = sum(c["end"] - c["start"] for c in analysis["suggested_cuts"])
    flags = []
    if vocal == 0:
        flags.append("no vocals found")
    elif vocal / duration < 0.25:
        flags.append("few vocals")
    if vocal and words / (vocal / 60) < 40:
        flags.append("sparse lyrics")
    if (source.get("language_probability") or 0) < 0.6:
        flags.append("language unsure")
    if cut / duration > 0.35:
        flags.append("large cut")
    if not any(s["label"] == "chorus" for s in sections):
        flags.append("no chorus")
    # a shaky label on an instrumental gap does not matter; one on a sung section does
    if any(s["confidence"] == "low" and s["vocal_coverage"] >= 0.25 for s in sections):
        flags.append("low-confidence label")
    return flags


def write_index(root: Path, manifest: list[dict]) -> Path:
    """Overview page of every song in the manifest, linking to the review pages."""
    failures = {}
    failures_path = root / "failures.jsonl"
    if failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            failures[entry["id"]] = entry["error"]

    rows = []
    for song in manifest:
        row = {**song, "status": "pending", "flags": [], "sections": []}
        path = root / "output" / song["id"] / f"{song['id']}.analysis.json"
        if path.exists():
            analysis = json.loads(path.read_text(encoding="utf-8"))
            row.update(
                status="done", flags=song_flags(analysis),
                language=analysis["source"].get("language"),
                duration=analysis["source"]["duration"], bpm=analysis["tempo"]["bpm"],
                labels=analysis["sections"][0]["source"],
                sections=[[s["label"], round(s["end"] - s["start"], 1)]
                          for s in analysis["sections"]],
                cut=round(sum(c["end"] - c["start"] for c in analysis["suggested_cuts"]), 1),
                review=f"output/{song['id']}/{song['id']}.review.html")
        elif song["id"] in failures:
            row.update(status="failed", flags=[failures[song["id"]][:120]])
        rows.append(row)

    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = (TEMPLATE.parent / "index_template.html").read_text(encoding="utf-8")
    index_path = root / "index.html"
    index_path.write_text(html.replace("__DATA__", payload), encoding="utf-8")
    return index_path


def write_review(analysis: dict, audio_path: Path, html_path: Path,
                 mono: np.ndarray | None) -> None:
    """mono is the decoded audio for the waveform; None draws the page without one."""
    data = {
        "analysis": analysis,
        "peaks": waveform_peaks(mono) if mono is not None else [],
        "audio": Path(os.path.relpath(audio_path, html_path.parent)).as_posix(),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)
    html_path.write_text(html, encoding="utf-8")
