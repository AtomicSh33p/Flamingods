"""The section-labelling request sent to Claude, and applying its answer to an analysis."""

from difflib import SequenceMatcher

from songlib import overlap, r3

LABELS = ["intro", "verse", "pre_chorus", "chorus", "post_chorus", "bridge", "instrumental", "outro"]

SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "label": {"type": "string", "enum": LABELS},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "note": {"type": "string"},
                },
                "required": ["start", "label", "confidence", "note"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["sections", "summary"],
    "additionalProperties": False,
}

SYSTEM = """You label the structure of a song from machine analysis of its audio.

You receive JSON with: the song duration, the tempo, downbeat times (bar starts), ranges where \
vocals are active, lyric lines with timestamps from automatic transcription of the sung vocals, \
and candidate segments from a local heuristic with a first-guess label, vocal coverage and \
relative energy.

Return the song's sections in order. Each section has a start time in seconds; it ends where the \
next one starts, and the last one ends at the song's end. The first section starts at 0.

How to decide:
- The transcription has wrong words but reliable timing. Judge repetition by approximate \
similarity of lines, not exact matches.
- A chorus is a block of lyrics that returns with nearly the same words, usually at higher \
energy. Verses share melody but have different words each time.
- The candidate boundaries are hints. Move, merge or split them when the lyrics show the real \
boundary is elsewhere. Put vocal section boundaries where a lyric line starts, and prefer times \
that are on or near a downbeat.
- Stretches without vocals are intro (start), outro (end) or instrumental (middle). Do not split \
off vocal-free gaps shorter than about 4 seconds.
- Use pre_chorus, post_chorus and bridge only when the evidence is clear; otherwise use verse or \
chorus.
- Set confidence to low when the evidence for a label is thin, and say why in the note. Keep \
notes to one short sentence. The summary is one or two sentences on the overall form.
- Never quote or paraphrase the lyrics in notes or the summary. Refer to lines by their start \
time or describe them ("the repeated hook", "the spoken passage").
- Lyric text may be replaced by group tags such as "[A, 7 words]": lines with the same letter \
have nearly the same words. Judge repetition from the tags then."""


def describe(analysis: dict, start: float, end: float) -> dict:
    """Statistics for one time range, computed from what the analysis already holds."""
    span = max(end - start, 1e-6)
    vocal_ranges = [(a["start"], a["end"]) for a in analysis["activity"] if a["type"] == "vocals"]
    words = [w for line in analysis["lyrics"] for w in line["words"]
             if start <= (w["start"] + w["end"]) / 2 < end]
    hop = analysis["energy"]["hop_seconds"]
    first = int(start / hop)
    energy = analysis["energy"]["values"][first:max(int(end / hop), first + 1)]
    return {
        "start": r3(start), "end": r3(end),
        "vocal_coverage": round(overlap(start, end, vocal_ranges) / span, 2),
        "energy": round(sum(energy) / len(energy), 2) if energy else 0.0,
        "words_per_second": round(len(words) / span, 2),
        "lyrics": " ".join(w["text"] for w in words),
    }


def number_sections(sections: list[dict]) -> None:
    counts: dict[str, int] = {}
    for seg in sections:
        counts[seg["label"]] = counts.get(seg["label"], 0) + 1
        seg["index"] = counts[seg["label"]]


def line_groups(lines: list[dict], threshold: float = 0.7) -> list[str]:
    """A letter per line; lines with nearly the same text share a letter."""
    texts = ["".join(c for c in ln["text"].lower() if c.isalnum() or c == " ") for ln in lines]
    reps: list[str] = []
    tags = []
    for text in texts:
        match = next((i for i, rep in enumerate(reps)
                      if SequenceMatcher(None, text, rep).ratio() >= threshold), None)
        if match is None:
            reps.append(text)
            match = len(reps) - 1
        tags.append(chr(65 + match % 26) + (str(match // 26) if match >= 26 else ""))
    return tags


def build_payload(analysis: dict, redact: bool = False) -> dict:
    """The labelling request. redact replaces lyric text with repetition tags, for songs whose
    lyrics the API's copyright filter will not let through."""
    lines = analysis["lyrics"]
    if redact:
        texts = [f"[{tag}, {len(ln['words'])} words]"
                 for tag, ln in zip(line_groups(lines), lines)]
    else:
        texts = [ln["text"] for ln in lines]
    return {
        "duration": analysis["source"]["duration"],
        "bpm": analysis["tempo"]["bpm"],
        "downbeats": [round(t, 2) for t in analysis["tempo"]["downbeats"]],
        "vocal_ranges": [[round(a["start"], 2), round(a["end"], 2)]
                         for a in analysis["activity"] if a["type"] == "vocals"],
        "lyric_lines": [{"start": ln["start"], "end": ln["end"], "text": text}
                        for ln, text in zip(lines, texts)],
        "candidate_segments": [{k: seg[k] for k in ("start", "end", "label", "vocal_coverage",
                                                    "energy")}
                               for seg in analysis["sections"]],
    }


def apply_labels(analysis: dict, result: dict) -> bool:
    """Replace the analysis sections with Claude's. Returns False if the answer is unusable."""
    duration = analysis["source"]["duration"]
    picked = sorted((s for s in result.get("sections", []) if s.get("label") in LABELS),
                    key=lambda s: s["start"])
    if not picked:
        return False
    picked[0]["start"] = 0.0
    kept = [picked[0]]
    for sec in picked[1:]:
        if sec["start"] - kept[-1]["start"] >= 1.0 and sec["start"] < duration - 1.0:
            kept.append(sec)
    ends = [s["start"] for s in kept[1:]] + [duration]
    analysis["sections"] = [
        {**describe(analysis, sec["start"], end), "label": sec["label"], "source": "claude",
         "confidence": sec["confidence"], "note": sec["note"]}
        for sec, end in zip(kept, ends)
    ]
    number_sections(analysis["sections"])
    analysis["summary"] = result.get("summary", "")
    return True
