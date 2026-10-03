"""Decide which parts of a song to remove, and remap an analysis after removing them."""

import copy

import numpy as np

from songlib import r3


def suggest_cuts(duration: float, vocal_ranges: list[tuple[float, float]], downbeats: list[float],
                 *, keep_bars: int = 1, min_cut: float = 4.0, min_mid_cut: float = 15.0
                 ) -> list[dict]:
    """Cut the intro, the outro and long vocal-free breaks, keeping ~keep_bars around the vocals.

    A cut in the middle of a song is audible, so a break is only cut when at least min_mid_cut
    seconds would be removed; the intro and outro only need min_cut. Cut points sit on
    downbeats so whole bars are removed and the beat stays continuous.
    """
    if not vocal_ranges:
        return []
    grid = np.asarray(downbeats, dtype=float)
    bar = float(np.median(np.diff(grid))) if len(grid) > 1 else 2.0

    def lead_in(t: float) -> float:
        if len(grid) == 0:
            return max(0.0, t - keep_bars * bar)
        i = int(np.searchsorted(grid, t, side="right")) - 1
        if i < 0:
            return 0.0
        # a vocal starting right on a downbeat still gets a full bar of lead-in
        i -= keep_bars if t - grid[i] < 0.5 * bar else keep_bars - 1
        return float(grid[i]) if i >= 0 else 0.0

    def tail_out(t: float) -> float:
        if len(grid) == 0:
            return min(duration, t + keep_bars * bar)
        j = int(np.searchsorted(grid, t, side="left"))
        if j >= len(grid):
            return duration
        j += keep_bars if grid[j] - t < 0.5 * bar else keep_bars - 1
        return float(grid[j]) if j < len(grid) else duration

    candidates = [(0.0, lead_in(vocal_ranges[0][0]), "intro", min_cut),
                  (tail_out(vocal_ranges[-1][1]), duration, "outro", min_cut)]
    for (_, prev_end), (next_start, _) in zip(vocal_ranges, vocal_ranges[1:]):
        candidates.append((tail_out(prev_end), lead_in(next_start), "instrumental break",
                           min_mid_cut))

    cuts = [{"start": r3(s), "end": r3(e), "reason": why}
            for s, e, why, shortest in candidates if e - s >= shortest]
    return sorted(cuts, key=lambda c: c["start"])


def remap_analysis(analysis: dict, cuts: list[dict]) -> dict:
    """Return a copy of the analysis with timestamps matching the audio after the cuts."""
    spans = sorted((c["start"], c["end"]) for c in cuts)

    def removed_before(t: float) -> float:
        return sum(max(0.0, min(t, e) - s) for s, e in spans)

    def tmap(t: float) -> float:
        return r3(t - removed_before(t))

    def inside(t: float) -> bool:
        return any(s <= t < e for s, e in spans)

    def remap_ranges(items: list[dict]) -> list[dict]:
        out = []
        for item in items:
            start, end = tmap(item["start"]), tmap(item["end"])
            if end - start >= 0.05:
                out.append({**item, "start": start, "end": end})
        return out

    result = copy.deepcopy(analysis)
    result["source"]["duration"] = tmap(analysis["source"]["duration"])
    result["activity"] = remap_ranges(analysis["activity"])
    result["sections"] = remap_ranges(analysis["sections"])
    result["lyrics"] = [
        {**line, "words": remap_ranges(line["words"])} for line in remap_ranges(analysis["lyrics"])
    ]
    for key in ("beats", "downbeats"):
        result["tempo"][key] = [tmap(t) for t in analysis["tempo"][key] if not inside(t)]
    result["highlights"] = [
        {**h, "time": tmap(h["time"])} for h in analysis["highlights"] if not inside(h["time"])
    ]
    hop = analysis["energy"]["hop_seconds"]
    result["energy"]["values"] = [
        v for i, v in enumerate(analysis["energy"]["values"]) if not inside(i * hop)
    ]
    result["suggested_cuts"] = []
    result["cuts_applied"] = cuts
    return result
