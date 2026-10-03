"""Cut the less interesting parts out of a song using its analysis JSON.

By default applies the "suggested_cuts" stored in the analysis, so edit that list (or the
sections) by hand first if you want different cuts. Writes <song>.trimmed.mp3, a remapped
<song>.trimmed.analysis.json and a review page for the trimmed version.

Usage (inside WSL): ~/.venvs/song-analyzer/bin/python trim.py songs/song.mp3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cuts import remap_analysis, suggest_cuts
from review import write_review
from songlib import SR, data_dir, load_audio, save_mp3


def apply_cuts(audio: np.ndarray, cuts: list[dict], crossfade_s: float, fade_out_s: float
               ) -> np.ndarray:
    total = audio.shape[1]
    keep, cursor = [], 0
    for cut in cuts:
        start, end = int(cut["start"] * SR), min(int(cut["end"] * SR), total)
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        keep.append((cursor, total))
    if not keep:
        raise SystemExit("The cuts remove the whole song.")

    # equal-power crossfade centred on each joint, borrowing audio from inside the cut,
    # so the output length equals the sum of the kept ranges
    half = int(crossfade_s * SR / 2)
    shortest = min([e - s for s, e in keep] + [b[0] - a[1] for a, b in zip(keep, keep[1:])])
    half = max(1, min(half, shortest // 2))
    ramp = np.linspace(0, np.pi / 2, 2 * half, dtype=np.float32)
    pieces = []
    for i, (start, end) in enumerate(keep):
        first, last = i == 0, i == len(keep) - 1
        pieces.append(audio[:, start + (0 if first else half):end - (0 if last else half)])
        if not last:
            next_start = keep[i + 1][0]
            outgoing = audio[:, end - half:end + half]
            incoming = audio[:, next_start - half:next_start + half]
            pieces.append(outgoing * np.cos(ramp) + incoming * np.sin(ramp))
    out = np.concatenate(pieces, axis=1)

    if keep[0][0] > 0:
        n = min(int(0.02 * SR), out.shape[1])
        out[:, :n] *= np.linspace(0, 1, n, dtype=np.float32)
    if keep[-1][1] < total:
        n = min(int(fade_out_s * SR), out.shape[1])
        out[:, -n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--analysis", type=Path,
                        help="default: output/<song name>/<song name>.analysis.json")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore suggested_cuts in the JSON and derive cuts from the flags below")
    parser.add_argument("--keep-bars", type=int, default=1)
    parser.add_argument("--min-cut", type=float, default=4.0, help="intro/outro, seconds")
    parser.add_argument("--min-mid-cut", type=float, default=15.0, help="mid-song, seconds")
    parser.add_argument("--crossfade", type=float, default=0.04, help="seconds, at each joint")
    parser.add_argument("--fade-out", type=float, default=1.5,
                        help="seconds, only when the outro is cut")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    audio_path = args.audio.resolve()
    stem = audio_path.stem
    analysis_path = args.analysis or (
        data_dir() / "output" / stem / f"{stem}.analysis.json")
    if not analysis_path.exists():
        raise SystemExit(f"No analysis at {analysis_path}. Run analyze.py first.")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    if args.recompute:
        vocal_ranges = [(a["start"], a["end"]) for a in analysis["activity"]
                        if a["type"] == "vocals"]
        cuts = suggest_cuts(analysis["source"]["duration"], vocal_ranges,
                            analysis["tempo"]["downbeats"], keep_bars=args.keep_bars,
                            min_cut=args.min_cut, min_mid_cut=args.min_mid_cut)
    else:
        cuts = sorted(analysis["suggested_cuts"], key=lambda c: c["start"])
    if not cuts:
        raise SystemExit("Nothing to cut.")

    audio = load_audio(audio_path)
    trimmed = apply_cuts(audio, cuts, args.crossfade, args.fade_out)
    remapped = remap_analysis(analysis, cuts)
    remapped["source"]["file"] = f"{stem}.trimmed.mp3"
    remapped["source"]["duration"] = round(trimmed.shape[1] / SR, 3)

    out_dir = analysis_path.parent
    mp3_path = out_dir / f"{stem}.trimmed.mp3"
    save_mp3(mp3_path, trimmed)
    json_path = out_dir / f"{stem}.trimmed.analysis.json"
    json_path.write_text(json.dumps(remapped, ensure_ascii=False, indent=1), encoding="utf-8")
    html_path = out_dir / f"{stem}.trimmed.review.html"
    write_review(remapped, mp3_path, html_path, trimmed.mean(0))

    before, after = audio.shape[1] / SR, trimmed.shape[1] / SR
    for cut in cuts:
        print(f"  cut {cut['start']:7.2f} - {cut['end']:7.2f}  {cut['reason']}")
    print(f"{before:.1f}s -> {after:.1f}s ({before - after:.1f}s removed)")
    print(f"Wrote {mp3_path}\nWrote {json_path}\nWrote {html_path}")


if __name__ == "__main__":
    main()
