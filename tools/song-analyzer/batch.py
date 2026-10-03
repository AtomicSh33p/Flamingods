"""Work on the whole song collection.

  batch.py run              download and analyze every song in songs.json not done yet
  batch.py add <url>        add one YouTube song to songs.json, analyze it, label it
  batch.py fetch            download the audio for songs that already have results
  batch.py pages            rebuild every review page and the overview (index.html)
  batch.py manifest         rebuild songs.json from the stage's YouTube playlists
  batch.py export [dir]     copy songs.json and the analysis files (no audio or caches) to a
                            folder, to share them privately (default: results/ here)

Everything lives in the data folder: results/ next to this script, or the folder named in
config.json. Resumable: finished songs are skipped and failures go to failures.jsonl.

Usage: ~/.venvs/song-analyzer/bin/python batch.py run --max-minutes 110
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from analyze import Models, add_analysis_options, analyze_song, pick_device
from review import write_index, write_review
from songlib import TOOL_DIR, data_dir, load_audio

PLAYLISTS = {
    "solo": "PLYen2IdQeeBs",
    "duets": "PLDcOSFMnoiyg",
    "steal the thunder": "PLPyFu2eULVbY",
}


def manifest_path() -> Path:
    return data_dir() / "songs.json"


def load_manifest() -> list[dict]:
    if not manifest_path().exists():
        raise SystemExit(f"No {manifest_path()} yet. Run: batch.py manifest")
    return json.loads(manifest_path().read_text(encoding="utf-8"))


def save_manifest(manifest: list[dict]) -> None:
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                               encoding="utf-8")


def song_entry(info: dict, lists: list[str]) -> dict:
    return {"id": info["id"], "title": info.get("title") or "",
            "channel": info.get("channel") or info.get("uploader") or "",
            "duration": info.get("duration"), "lists": lists}


def build_manifest() -> list[dict]:
    from yt_dlp import YoutubeDL

    songs: dict[str, dict] = {}
    with YoutubeDL({"extract_flat": True, "quiet": True, "no_warnings": True}) as ydl:
        for name, playlist in PLAYLISTS.items():
            info = ydl.extract_info(f"https://www.youtube.com/playlist?list={playlist}",
                                    download=False)
            for entry in info["entries"]:
                song = songs.setdefault(entry["id"], song_entry(entry, []))
                if name not in song["lists"]:
                    song["lists"].append(name)
    # keep songs added by hand with "batch.py add"
    if manifest_path().exists():
        for song in load_manifest():
            if song["id"] not in songs and set(song["lists"]) - set(PLAYLISTS):
                songs[song["id"]] = song
    manifest = list(songs.values())
    save_manifest(manifest)
    return manifest


def audio_file(song_id: str) -> Path | None:
    found = [p for p in (data_dir() / "songs").glob(f"{song_id}.*") if p.suffix != ".part"]
    return found[0] if found else None


def analysis_file(song_id: str) -> Path:
    return data_dir() / "output" / song_id / f"{song_id}.analysis.json"


def download(song: dict) -> Path:
    existing = audio_file(song["id"])
    if existing:
        return existing
    from yt_dlp import YoutubeDL

    songs_dir = data_dir() / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)
    options = {"format": "bestaudio[ext=m4a]/bestaudio", "noplaylist": True, "quiet": True,
               "no_warnings": True, "noprogress": True,
               "outtmpl": str(songs_dir / "%(id)s.%(ext)s")}
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={song['id']}", download=True)
        path = Path(ydl.prepare_filename(info))
    # be a polite client; rapid-fire downloads get throttled
    time.sleep(random.uniform(3, 6))
    return path


def run(args: argparse.Namespace, manifest: list[dict]) -> None:
    root = data_dir()
    failures_path = root / "failures.jsonl"
    wanted = [s for s in manifest if not args.ids or s["id"] in args.ids]
    todo = [s for s in wanted if args.force or not analysis_file(s["id"]).exists()]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(wanted)} songs selected, {len(wanted) - len(todo)} already done, "
          f"{len(todo)} to process.", flush=True)
    if not todo:
        return

    models = Models(pick_device(args.device), args.hebrew_model, args.whisper_model)
    started, done, failed, streak = time.time(), 0, 0, 0
    for n, song in enumerate(todo, 1):
        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            print(f"Time budget of {args.max_minutes} minutes reached.", flush=True)
            break
        t0, stage = time.time(), "download"
        try:
            audio = download(song)
            stage = "analyze"
            meta = {k: song[k] for k in ("id", "title", "channel", "lists")}
            analysis = analyze_song(audio, root / "output" / song["id"], args, models, meta=meta,
                                    keep_stem=args.keep_stems, verbose=False)
        except Exception as e:  # one bad song must not stop the batch
            failed, streak = failed + 1, streak + 1
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": song["id"], "title": song["title"], "stage": stage,
                                    "error": f"{type(e).__name__}: {e}",
                                    "trace": traceback.format_exc()[-1500:]},
                                   ensure_ascii=False) + "\n")
            print(f"[{n}/{len(todo)}] FAILED ({stage}) {song['title']}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            if streak >= 5:
                print("Five failures in a row; stopping, this looks systemic.", flush=True)
                break
            continue
        done, streak = done + 1, 0
        form = " ".join(s["label"] for s in analysis["sections"])
        print(f"[{n}/{len(todo)}] {song['title']} | {analysis['source']['language']} | "
              f"{time.time() - t0:.0f}s | {form}", flush=True)
        if done % 20 == 0:
            write_index(root, manifest)

    write_index(root, manifest)
    minutes = (time.time() - started) / 60
    print(f"\nDone {done}, failed {failed}, in {minutes:.0f} min "
          f"({minutes * 60 / max(done, 1):.0f}s per song). "
          f"Remaining: {len(todo) - done - failed}.", flush=True)


def add(args: argparse.Namespace) -> None:
    from yt_dlp import YoutubeDL

    with YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(args.url, download=False)
    manifest = load_manifest() if manifest_path().exists() else []
    song = next((s for s in manifest if s["id"] == info["id"]), None)
    if song is None:
        song = song_entry(info, [args.list])
        manifest.append(song)
    elif args.list not in song["lists"]:
        song["lists"].append(args.list)
    save_manifest(manifest)
    print(f"{song['title']} ({song['channel']}) is in songs.json as {song['id']}.")

    args.ids = [song["id"]]
    run(args, manifest)
    if not analysis_file(song["id"]).exists():
        raise SystemExit("Analysis failed; see failures.jsonl.")
    if args.label:
        label = subprocess.run([sys.executable, str(TOOL_DIR / "label.py"),
                                str(analysis_file(song["id"]))])
        if label.returncode:
            print("Labelling did not complete; heuristic labels kept. Retry with label.py.")
    write_index(data_dir(), manifest)
    review = analysis_file(song["id"]).with_name(f"{song['id']}.review.html")
    print(f"Review page: {review}")


def pages(manifest: list[dict]) -> None:
    built = 0
    for song in manifest:
        path = analysis_file(song["id"])
        if not path.exists():
            continue
        analysis = json.loads(path.read_text(encoding="utf-8"))
        audio = audio_file(song["id"])
        mono = load_audio(audio).mean(0) if audio else None
        audio = audio or data_dir() / "songs" / f"{song['id']}.m4a"
        write_review(analysis, audio.resolve(), path.with_name(f"{song['id']}.review.html"),
                     mono)
        built += 1
    print(f"Built {built} review pages and {write_index(data_dir(), manifest)}")


def fetch(manifest: list[dict], ids: list[str] | None) -> None:
    have = [s for s in manifest
            if analysis_file(s["id"]).exists() and (not ids or s["id"] in ids)]
    todo = [s for s in have if not audio_file(s["id"])]
    print(f"{len(have)} songs have results, {len(todo)} need audio.", flush=True)
    failed = 0
    for n, song in enumerate(todo, 1):
        try:
            download(song)
            print(f"[{n}/{len(todo)}] {song['title']}", flush=True)
        except Exception as e:  # keep going; YouTube refuses some requests at random
            failed += 1
            print(f"[{n}/{len(todo)}] FAILED {song['title']}: {str(e)[:150]}", flush=True)
    if failed:
        print(f"{failed} downloads failed; run fetch again later to retry them.")
    pages(manifest)


def export(target: Path, manifest: list[dict]) -> None:
    if target.resolve() == data_dir().resolve():
        raise SystemExit("The export folder is the data folder itself; nothing to do.")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path(), target / "songs.json")
    copied = 0
    for song in manifest:
        source = analysis_file(song["id"])
        if source.exists():
            (target / "output" / song["id"]).mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / "output" / song["id"] / source.name)
            copied += 1
    print(f"Copied songs.json and {copied} analyses to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["run", "add", "fetch", "pages", "manifest", "export", "index"])
    parser.add_argument("url", nargs="?", help="add: a YouTube link; export: target folder")
    parser.add_argument("--list", default="added", help="add: which list the song belongs to")
    parser.add_argument("--no-label", dest="label", action="store_false",
                        help="add: skip the Claude labelling step")
    parser.add_argument("--ids", nargs="*", help="run, fetch: only these YouTube video ids")
    parser.add_argument("--limit", type=int, help="run: process at most this many songs")
    parser.add_argument("--max-minutes", type=float,
                        help="run: stop starting new songs after this")
    parser.add_argument("--keep-stems", action="store_true",
                        help="keep each vocal stem (about 25 MB per song)")
    add_analysis_options(parser)
    args = parser.parse_args()
    args.no_llm = True  # labelling is a separate step: label.py

    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "manifest":
        print(f"Wrote {manifest_path()} with {len(build_manifest())} songs.")
    elif args.command == "add":
        if not args.url:
            parser.error("add needs a YouTube link")
        add(args)
    elif args.command == "run":
        run(args, load_manifest())
    elif args.command == "fetch":
        fetch(load_manifest(), args.ids)
    elif args.command == "pages":
        pages(load_manifest())
    elif args.command == "export":
        export(Path(args.url) if args.url else TOOL_DIR / "results", load_manifest())
    else:
        print(f"Wrote {write_index(data_dir(), load_manifest())}")


if __name__ == "__main__":
    main()
