"""Analyze a song for the lip-sync stage.

Writes <song>.analysis.json (silence, vocal/instrumental ranges, sections, timed lyrics,
beat grid, energy, highlights, suggested cuts) and <song>.review.html next to it.

Usage (inside WSL): ~/.venvs/song-analyzer/bin/python analyze.py songs/song.mp3
"""

import argparse
import builtins
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from cuts import suggest_cuts
from review import write_review
from sections import apply_labels, build_payload, number_sections
from songlib import SR, data_dir, load_audio, mask_to_ranges, overlap, r3, snap

ANALYSIS_SR = 22050
HOP = 512
FRAME_S = HOP / ANALYSIS_SR
ENERGY_HOP_S = 0.5
# phrases Whisper invents over wordless singing ("ooh", "aah")
WHISPER_FILLER = {"thank you.", "thank you", "thanks for watching!", "thank you for watching.",
                  "thank you for watching!", "you"}


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def rms_db(y: np.ndarray) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    return 20 * np.log10(np.maximum(rms, 1e-7))


class Models:
    """Loads each model once, so a batch does not pay the start-up cost for every song."""

    def __init__(self, device: str, hebrew_model: str, general_model: str):
        self.device = device
        self.hebrew_model = hebrew_model
        self.general_model = general_model
        self._demucs = None
        self._whisper: dict[tuple[str, str], object] = {}

    def demucs(self):
        if self._demucs is None:
            # demucs checkpoints pickle their model class, which torch>=2.6 refuses by default
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            from demucs.pretrained import get_model

            self._demucs = get_model("htdemucs")
            self._demucs.eval()
        return self._demucs

    def whisper(self, name: str, device: str | None = None):
        device = device or self.device
        if (name, device) not in self._whisper:
            if device == "cuda":
                expose_cuda_libs()
            from faster_whisper import WhisperModel

            self._whisper[name, device] = WhisperModel(
                name, device=device, compute_type="int8_float16" if device == "cuda" else "int8")
        return self._whisper[name, device]


def separate_vocals(mix: np.ndarray, models: Models, progress: bool) -> np.ndarray:
    """Return the vocal stem, shape (2, samples) at SR."""
    import torch
    from demucs.apply import apply_model

    model, device = models.demucs(), models.device
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    normalized = ((wav - mean) / std)[None]

    def run(dev: str) -> "torch.Tensor":
        with torch.no_grad():
            return apply_model(model, normalized, device=dev, split=True, overlap=0.25,
                               progress=progress)[0]

    try:
        sources = run(device)
    except torch.cuda.OutOfMemoryError:
        print("  GPU out of memory, retrying on CPU (slower).")
        torch.cuda.empty_cache()
        sources = run("cpu")
    vocals = sources[model.sources.index("vocals")].cpu() * std + mean
    model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    return vocals.numpy()


def detect_vocal_ranges(vocals_mono: np.ndarray, mix_db: np.ndarray, threshold_db: float,
                        ratio_db: float) -> list[tuple[float, float]]:
    """Vocals are active where the stem is loud, or quiet but still dominating the mix.

    The second rule catches soft passages in dynamic songs (a quiet ending after a loud
    middle); instrument bleed never passes it because bleed sits far below the mix.
    """
    from scipy.ndimage import median_filter

    db = rms_db(vocals_mono)
    peak = np.percentile(db, 95)
    # a stem this far below the mix is separation residue, not a singer
    if peak < np.percentile(mix_db, 95) - 25:
        return []
    n = min(len(db), len(mix_db))
    db, dominant = db[:n], db[:n] - mix_db[:n] > -ratio_db
    active = median_filter((db > peak - threshold_db) | (dominant & (db > peak - 30)), size=13)
    return mask_to_ranges(active, FRAME_S, min_len=0.6, merge_gap=1.5)


def expose_cuda_libs() -> None:
    """ctranslate2 needs cuBLAS and cuDNN but ships neither; reuse the copies torch installs."""
    import torch

    if sys.platform == "win32":
        lib = str(Path(torch.__file__).parent / "lib")
        os.add_dll_directory(lib)
        os.environ["PATH"] = lib + os.pathsep + os.environ["PATH"]
        return
    import ctypes

    nvidia = Path(torch.__file__).parent.parent / "nvidia"
    for pattern in ("cublas/lib/libcublas*.so.*", "cudnn/lib/libcudnn*.so.*"):
        for lib in sorted(nvidia.glob(pattern)):
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def transcribe(vocals_mono: np.ndarray, vocal_ranges: list[tuple[float, float]], duration: float,
               models: Models, language: str, hebrew_hint: bool) -> dict:
    """Word-level transcription of the vocal stem, restricted to where vocals are active.

    Returns {"language", "language_probability", "words"}. With language "auto" the language
    is detected from the longest vocal passage; Hebrew uses the Hebrew-tuned model.
    """
    import librosa

    audio = librosa.resample(vocals_mono, orig_sr=SR, target_sr=16000)
    clips = [t for s, e in vocal_ranges for t in (max(0.0, s - 0.3), min(duration, e + 0.3))]

    def run(dev: str | None) -> dict:
        lang, prob = language, 1.0
        if language == "auto" and hebrew_hint:
            # detection on sung vocals is shaky; a Hebrew title is the better evidence
            lang = "he"
        elif language == "auto":
            start = max(vocal_ranges, key=lambda r: r[1] - r[0])[0]
            sample = audio[int(start * 16000):int((start + 30) * 16000)]
            lang, prob, _ = models.whisper(models.general_model, dev).detect_language(sample)
        name = models.hebrew_model if lang == "he" else models.general_model
        segments, _ = models.whisper(name, dev).transcribe(
            audio, language=lang, word_timestamps=True, clip_timestamps=clips,
            condition_on_previous_text=False, beam_size=5)
        words = [{"start": r3(w.start), "end": r3(w.end), "text": w.word.strip()}
                 for seg in segments if seg.text.strip().lower() not in WHISPER_FILLER
                 for w in (seg.words or []) if w.word.strip()]
        return {"language": lang, "language_probability": round(float(prob), 2), "words": words}

    try:
        return run(None)
    except RuntimeError as e:
        if models.device != "cuda":
            raise
        print(f"  Whisper failed on GPU ({e}); retrying on CPU (slower).")
        return run("cpu")


def words_to_lines(words: list[dict], max_gap: float = 0.7, max_words: int = 12) -> list[dict]:
    """Group words into lines at pauses and at the punctuation Whisper inserts."""
    lines: list[list[dict]] = []
    for word in words:
        if lines:
            line, prev = lines[-1], lines[-1][-1]
            phrase_end = prev["text"][-1] in ".!?" or (prev["text"][-1] == "," and len(line) >= 3)
            if not (word["start"] - prev["end"] > max_gap or phrase_end or len(line) >= max_words):
                line.append(word)
                continue
        lines.append([word])
    return [{"start": ws[0]["start"], "end": ws[-1]["end"],
             "text": " ".join(w["text"] for w in ws), "words": ws} for ws in lines]


def mark_repeated(lines: list[dict], min_distance: float = 30.0, threshold: float = 0.7
                  ) -> list[bool]:
    """Flag lines whose words come back elsewhere in the song, which is what a chorus does.

    Compares characters rather than words so transcription errors still match, and ignores
    matches closer than min_distance so a hook repeated inside one verse does not count.
    """
    texts = ["".join(c for c in ln["text"] if c.isalnum() or c == " ") for ln in lines]
    long_enough = [len(t.split()) >= 3 for t in texts]
    flags = [
        long_enough[i] and any(
            long_enough[j] and abs(lines[i]["start"] - lines[j]["start"]) >= min_distance
            and SequenceMatcher(None, texts[i], texts[j]).ratio() >= threshold
            for j in range(len(lines)))
        for i in range(len(lines))
    ]
    for i in range(1, len(flags) - 1):
        if flags[i - 1] == flags[i + 1] != flags[i]:
            flags[i] = flags[i - 1]
    return flags


def track_beats(instrumental: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (bpm, beat frames, downbeat frames). Downbeats assume 4/4 and are an estimate."""
    import librosa

    sr = ANALYSIS_SR
    onset_env = librosa.onset.onset_strength(y=instrumental, sr=sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    if len(beats) < 8:
        return float(np.atleast_1d(tempo)[0]), beats, beats[:0]

    mel = librosa.feature.melspectrogram(y=instrumental, sr=sr, hop_length=HOP, fmax=150, n_mels=8)
    low_onsets = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr)
    chroma = librosa.util.sync(
        librosa.feature.chroma_cqt(y=instrumental, sr=sr, hop_length=HOP), beats)
    # bars tend to start with a kick and a chord change
    harmonic_change = np.linalg.norm(np.diff(chroma, axis=1), axis=0)[:len(beats)]
    kick = low_onsets[beats]

    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    score = z(kick) + z(harmonic_change)
    phase = int(np.argmax([score[p::4].mean() for p in range(4)]))
    return float(np.atleast_1d(tempo)[0]), beats, beats[phase::4]


def audio_boundaries(y: np.ndarray, beats: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Laplacian structural segmentation (McFee & Ellis 2014).

    Returns (beat-synchronous frame starts, cluster id per beat interval).
    """
    import librosa
    import scipy
    from sklearn.cluster import KMeans

    sr = ANALYSIS_SR
    cqt = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y=y, sr=sr, hop_length=HOP, bins_per_octave=36, n_bins=252)), ref=np.max)
    starts = librosa.util.fix_frames(beats, x_min=0)
    csync = librosa.util.sync(cqt, starts, aggregate=np.median)
    rec = librosa.segment.recurrence_matrix(csync, width=3, mode="affinity", sym=True)
    rec = librosa.segment.timelag_filter(scipy.ndimage.median_filter)(rec, size=(1, 7))

    msync = librosa.util.sync(librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP), starts)
    path_dist = np.sum(np.diff(msync, axis=1) ** 2, axis=0)
    path_sim = np.exp(-path_dist / np.median(path_dist))
    path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    deg_path, deg_rec = path.sum(axis=1), rec.sum(axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)
    affinity = mu * rec + (1 - mu) * path

    laplacian = scipy.sparse.csgraph.laplacian(affinity, normed=True)
    _, evecs = scipy.linalg.eigh(laplacian)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    norm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    x = evecs[:, :k] / (norm[:, k - 1:k] + 1e-9)
    ids = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
    return starts[:len(ids)], ids


def merge_boundaries(duration: float, audio_bounds: list[float], lyric_bounds: list[float],
                     vocal_ranges: list[tuple[float, float]], downbeats: np.ndarray, bar: float
                     ) -> list[float]:
    """Combine vocal entry/exit points and lyric changes (trusted) with audio boundaries (hints)."""
    trusted = [0.0]
    if vocal_ranges:
        trusted.append(snap(vocal_ranges[0][0], downbeats))
        for (_, prev_end), (next_start, _) in zip(vocal_ranges, vocal_ranges[1:]):
            if next_start - prev_end > 5.0:
                trusted += [snap(prev_end, downbeats), snap(next_start, downbeats)]
        if duration - vocal_ranges[-1][1] > 5.0:
            trusted.append(snap(vocal_ranges[-1][1], downbeats))
    trusted += [snap(t, downbeats) for t in lyric_bounds]

    bounds: list[float] = []
    for t in trusted + [snap(t, downbeats) for t in audio_bounds]:
        if t < duration - bar and all(abs(t - b) >= 1.5 * bar for b in bounds):
            bounds.append(t)
    return sorted(bounds)


def heuristic_labels(segments: list[dict], lines: list[dict], repeated: list[bool]) -> None:
    """Label segments in place: vocal-free ones by position, vocal ones by lyric repetition."""
    vocal = [s for s in segments if s["vocal_coverage"] >= 0.25]
    for seg in segments:
        if seg in vocal:
            seg["label"] = "verse"
        elif seg is segments[0]:
            seg["label"] = "intro"
        elif seg is segments[-1]:
            seg["label"] = "outro"
        else:
            seg["label"] = "instrumental"

    choruses = []
    for seg in vocal:
        inside = [(len(ln["words"]), rep) for ln, rep in zip(lines, repeated)
                  if seg["start"] <= (ln["start"] + ln["end"]) / 2 < seg["end"]]
        total = sum(n for n, _ in inside)
        if total and sum(n for n, rep in inside if rep) / total >= 0.5:
            choruses.append(seg)
    if not choruses and vocal:
        # no usable lyric repetition: fall back to the loudest recurring audio cluster
        by_cluster: dict[int, list[dict]] = {}
        for seg in vocal:
            by_cluster.setdefault(seg["cluster"], []).append(seg)
        recurring = [group for group in by_cluster.values() if len(group) >= 2]
        if recurring:
            choruses = max(recurring, key=lambda g: np.mean([s["energy"] for s in g]))
    for seg in choruses:
        seg["label"] = "chorus"


def find_highlights(bar_db: np.ndarray, downbeats: np.ndarray, threshold_db: float = 4.0
                    ) -> list[dict]:
    """Bars where the energy jumps or collapses relative to the two bars before."""
    out = []
    for i in range(2, len(bar_db)):
        change = bar_db[i] - bar_db[i - 2:i].mean()
        if abs(change) >= threshold_db:
            out.append({"time": r3(downbeats[i]),
                        "type": "energy_rise" if change > 0 else "energy_drop",
                        "change_db": round(float(change), 1)})
    if len(bar_db) >= 4:
        windows = np.convolve(bar_db, np.ones(4) / 4, mode="valid")
        out.append({"time": r3(downbeats[int(np.argmax(windows))]), "type": "peak_energy",
                    "change_db": 0.0})
    return sorted(out, key=lambda h: h["time"])


def add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--language", default="auto",
                        help="lyrics language code such as he or en; default: detect per song")
    parser.add_argument("--hebrew-model", default="ivrit-ai/whisper-large-v3-turbo-ct2",
                        help="Whisper model for Hebrew songs")
    parser.add_argument("--whisper-model", default="large-v3-turbo",
                        help="Whisper model for every other language and for detection")
    parser.add_argument("--no-llm", action="store_true", help="skip the Claude labelling pass")
    parser.add_argument("--force", action="store_true", help="ignore cached stems and lyrics")
    parser.add_argument("--silence-db", type=float, default=40.0,
                        help="silence = this many dB below the song's loud level")
    parser.add_argument("--vocal-db", type=float, default=10.0,
                        help="vocals = within this many dB of the vocal stem's loud level; "
                             "raise it if quiet singing is missed, lower it if instrument "
                             "bleed is detected as vocals")
    parser.add_argument("--vocal-ratio-db", type=float, default=16.0,
                        help="quiet vocals still count when the stem is within this many dB "
                             "of the full mix")
    parser.add_argument("--clusters", type=int, default=5, help="audio similarity clusters")
    parser.add_argument("--min-cut", type=float, default=4.0,
                        help="shortest intro/outro cut worth suggesting (seconds)")
    parser.add_argument("--min-mid-cut", type=float, default=15.0,
                        help="shortest mid-song cut worth suggesting (seconds)")
    parser.add_argument("--keep-bars", type=int, default=1,
                        help="bars to keep around vocals when cutting")


def analyze_song(audio_path: Path, out_dir: Path, args: argparse.Namespace, models: Models, *,
                 meta: dict | None = None, keep_stem: bool = True, verbose: bool = True) -> dict:
    """Analyze one song and write its JSON and review page. meta adds title etc. to "source"."""
    import librosa
    import soundfile as sf

    def print(*a, **k):  # noqa: A001 - quiet mode for batch runs
        if verbose:
            builtins.print(*a, **k)

    meta = meta or {}
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading {audio_path.name}")
    mix = load_audio(audio_path)
    duration = mix.shape[1] / SR
    y = librosa.resample(mix.mean(0), orig_sr=SR, target_sr=ANALYSIS_SR)
    mix_db = rms_db(y)
    loud = np.percentile(mix_db, 95)
    silence = mask_to_ranges(mix_db < loud - args.silence_db, FRAME_S, min_len=0.4)

    print(f"[2/6] Separating vocals ({models.device})")
    stem_path = cache / "vocals.flac"
    if stem_path.exists() and not args.force:
        vocals = sf.read(stem_path, dtype="float32")[0].T
    else:
        vocals = separate_vocals(mix, models, progress=verbose)
        if keep_stem:
            sf.write(stem_path, vocals.T, SR, subtype="PCM_16")
    vocals_mono = vocals.mean(0)
    vocal_ranges = detect_vocal_ranges(
        librosa.resample(vocals_mono, orig_sr=SR, target_sr=ANALYSIS_SR), mix_db, args.vocal_db,
        args.vocal_ratio_db)
    instrumental = librosa.resample((mix - vocals).mean(0), orig_sr=SR, target_sr=ANALYSIS_SR)

    print("[3/6] Transcribing lyrics")
    words_path = cache / "words.json"
    transcript = {"language": None, "language_probability": 0.0, "words": []}
    if words_path.exists() and not args.force:
        transcript = json.loads(words_path.read_text(encoding="utf-8"))
        if isinstance(transcript, list):  # cache written before language detection existed
            transcript = {"language": "he", "language_probability": 1.0, "words": transcript}
    elif vocal_ranges:
        hebrew_hint = any("֐" <= c <= "׿" for c in meta.get("title", ""))
        transcript = transcribe(vocals_mono, vocal_ranges, duration, models, args.language,
                                hebrew_hint)
        words_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    words = transcript["words"]
    print(f"  language: {transcript['language']}")
    # the cache may come from a run with looser vocal detection
    words = [w for w in words
             if overlap(w["start"] - 0.3, w["end"] + 0.3, vocal_ranges) > 0]
    lines = words_to_lines(words)

    print("[4/6] Beats, energy and structure")
    bpm, beat_frames, downbeat_frames = track_beats(instrumental)
    beats = librosa.frames_to_time(beat_frames, sr=ANALYSIS_SR, hop_length=HOP)
    downbeats = librosa.frames_to_time(downbeat_frames, sr=ANALYSIS_SR, hop_length=HOP)
    bar = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 2.0

    per_hop = int(round(ENERGY_HOP_S / FRAME_S))
    energy_db = np.array([mix_db[i:i + per_hop].mean() for i in range(0, len(mix_db), per_hop)])
    energy = np.clip((energy_db - (loud - 30)) / 30, 0, 1)
    bar_db = np.array([mix_db[a:b].mean() for a, b in zip(downbeat_frames, downbeat_frames[1:])])
    highlights = find_highlights(bar_db, downbeats)

    if len(beat_frames) >= 16:
        starts, ids = audio_boundaries(y, beat_frames, args.clusters)
        change = 1 + np.flatnonzero(ids[:-1] != ids[1:])
        audio_bounds = list(librosa.frames_to_time(starts[change], sr=ANALYSIS_SR, hop_length=HOP))
        start_times = librosa.frames_to_time(starts, sr=ANALYSIS_SR, hop_length=HOP)
    else:
        audio_bounds, start_times, ids = [], np.array([0.0]), np.array([0])

    def describe(start: float, end: float) -> dict:
        inside = [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]
        span = max(end - start, 1e-6)
        seg_ids = ids[(start_times >= start) & (start_times < end)]
        return {
            "start": r3(start), "end": r3(end),
            "vocal_coverage": round(overlap(start, end, vocal_ranges) / span, 2),
            "energy": round(float(energy[int(start / ENERGY_HOP_S):
                                         max(int(end / ENERGY_HOP_S), int(start / ENERGY_HOP_S) + 1)
                                         ].mean()), 2),
            "words_per_second": round(len(inside) / span, 2),
            "cluster": int(np.bincount(seg_ids).argmax()) if len(seg_ids) else -1,
            "lyrics": " ".join(w["text"] for w in inside),
        }

    repeated = mark_repeated(lines)
    lyric_bounds = [ln["start"] for ln, rep, prev in zip(lines[1:], repeated[1:], repeated)
                    if rep != prev]
    bounds = merge_boundaries(duration, audio_bounds, lyric_bounds, vocal_ranges, downbeats, bar)
    candidates = [describe(s, e) for s, e in zip(bounds, bounds[1:] + [duration])]
    heuristic_labels(candidates, lines, repeated)
    sections = []
    for seg in candidates:
        if sections and sections[-1]["label"] == seg["label"]:
            merged = describe(sections[-1]["start"], seg["end"])
            sections[-1] = {**merged, "label": seg["label"]}
        else:
            sections.append(seg)
    for seg in sections:
        seg.update(source="heuristic", confidence="medium", note="")
    number_sections(sections)

    activity = []
    step = 0.05
    grid_t = np.arange(0, duration, step)
    kind = np.full(len(grid_t), "instrumental", dtype=object)
    for s, e in vocal_ranges:
        kind[(grid_t >= s) & (grid_t < e)] = "vocals"
    for s, e in silence:
        kind[(grid_t >= s) & (grid_t < e)] = "silence"
    edges = [0] + list(1 + np.flatnonzero(kind[:-1] != kind[1:])) + [len(kind)]
    for a, b in zip(edges, edges[1:]):
        activity.append({"start": r3(a * step), "end": r3(min(b * step, duration)),
                         "type": kind[a]})

    analysis = {
        "version": 1,
        "source": {"file": audio_path.name, "duration": r3(duration), **meta,
                   "language": transcript["language"],
                   "language_probability": transcript["language_probability"]},
        "summary": "",
        "tempo": {"bpm": round(bpm, 1), "time_signature": "4/4 (assumed)",
                  "beats": [r3(t) for t in beats], "downbeats": [r3(t) for t in downbeats]},
        "sections": sections,
        "activity": activity,
        "lyrics": lines,
        "energy": {"hop_seconds": ENERGY_HOP_S, "values": [round(float(v), 2) for v in energy]},
        "highlights": highlights,
        "suggested_cuts": suggest_cuts(duration, vocal_ranges, list(downbeats),
                                       keep_bars=args.keep_bars, min_cut=args.min_cut,
                                       min_mid_cut=args.min_mid_cut),
    }

    print("[5/6] Claude labelling pass (API)")
    if args.no_llm:
        print("  Skipped (--no-llm).")
    elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("  Skipped: no ANTHROPIC_API_KEY in .env. Heuristic labels kept; "
              "run label.py to label through a Claude subscription.")
    else:
        from llm import label_sections

        result = label_sections(build_payload(analysis))
        if result and not apply_labels(analysis, result):
            print("  Claude returned no usable sections; heuristic labels kept.")
    json_path = out_dir / f"{audio_path.stem}.analysis.json"
    if analysis["sections"][0]["source"] == "heuristic" and json_path.exists() and not args.force:
        # a rerun must not throw away labels that label.py already paid for
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        if previous["sections"] and previous["sections"][0].get("source") == "claude":
            apply_labels(analysis, previous)
            print("  Kept the Claude labels from the previous run (--force discards them).")
    sections = analysis["sections"]

    print("[6/6] Writing output")
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    html_path = out_dir / f"{audio_path.stem}.review.html"
    write_review(analysis, audio_path, html_path, mix.mean(0))

    print(f"\n{audio_path.name}: {duration:.1f}s, {bpm:.0f} BPM")
    for seg in sections:
        print(f"  {seg['start']:7.2f} - {seg['end']:7.2f}  {seg['label']:<13}{seg['index']:<2} "
              f"vocals {seg['vocal_coverage']:.0%}  energy {seg['energy']:.2f}  "
              f"[{seg['source']}, {seg['confidence']}]")
    cut_total = sum(c["end"] - c["start"] for c in analysis["suggested_cuts"])
    print(f"Suggested cuts: {len(analysis['suggested_cuts'])} ({cut_total:.1f}s)")
    print(f"Wrote {json_path}\nWrote {html_path}")
    if not keep_stem:
        stem_path.unlink(missing_ok=True)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out-dir", type=Path, help="default: <data dir>/output/<song name>/")
    add_analysis_options(parser)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise SystemExit(f"File not found: {audio_path}")
    out_dir = args.out_dir or data_dir() / "output" / audio_path.stem
    models = Models(pick_device(args.device), args.hebrew_model, args.whisper_model)
    analyze_song(audio_path, out_dir, args, models)


if __name__ == "__main__":
    main()
