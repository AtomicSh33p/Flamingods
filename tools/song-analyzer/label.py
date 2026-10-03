"""Label song sections with Claude through the Claude Code CLI (uses a Claude subscription).

Goes through the analysis files that still carry heuristic labels, asks Claude for the
sections, and writes them back. Safe to rerun: labelled songs are skipped. It stops as soon
as Claude is unavailable (usage limit, logged out) so the rest can be picked up later; a
problem with a single song only skips that song.

Usage (inside WSL):
  ~/.venvs/song-analyzer/bin/python label.py                 every unlabelled song in output/
  ~/.venvs/song-analyzer/bin/python label.py --limit 50      at most 50 songs this run
  ~/.venvs/song-analyzer/bin/python label.py output/x/x.analysis.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from review import write_review
from sections import SCHEMA, SYSTEM, apply_labels, build_payload
from songlib import data_dir, load_audio

ROOT = data_dir()
# replies that mean no further song can succeed right now (usage limit, logged out)
BLOCKING = ("limit", "log in", "login", "logged in", "credit", "overloaded", "authenticat")


def find_claude() -> str:
    exe = shutil.which("claude")
    if exe:
        return exe
    local = Path.home() / ".local/bin/claude"
    if local.exists():
        return str(local)
    raise SystemExit("Claude Code CLI not found. Install it with:\n"
                     "  curl -fsSL https://claude.ai/install.sh | bash\nthen run `claude` once "
                     "and log in.")


def ask_claude(claude: str, payload: dict, model: str, timeout: int) -> tuple[dict | None, dict]:
    """Returns (sections result or None, the CLI's reply envelope)."""
    cmd = [claude, "-p", "--model", model, "--output-format", "json",
           "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA),
           "--tools", "", "--no-session-persistence"]
    # run outside the repo so no project instructions are loaded into the request
    proc = subprocess.run(cmd, input=json.dumps(payload, ensure_ascii=False), text=True,
                          encoding="utf-8", capture_output=True, timeout=timeout,
                          cwd=tempfile.gettempdir())
    try:
        reply = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, {"is_error": True,
                      "result": (proc.stderr or proc.stdout).strip()[-500:] or "no output"}
    if reply.get("is_error"):
        return None, reply
    result = reply.get("structured_output")
    if result is None:
        try:
            result = json.loads(reply.get("result", ""))
        except json.JSONDecodeError:
            return None, {**reply, "is_error": True,
                          "result": "reply was not JSON: " + str(reply.get("result"))[:300]}
    return result, reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analyses", nargs="*", type=Path,
                        help="analysis JSON files; default: every one under output/")
    parser.add_argument("--model", default="opus", help="opus (default) or sonnet")
    parser.add_argument("--limit", type=int, help="label at most this many songs")
    parser.add_argument("--relabel", action="store_true",
                        help="also redo songs that already have Claude labels")
    parser.add_argument("--songs-dir", type=Path, default=ROOT / "songs",
                        help="where the audio files are, for refreshing the review pages")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per song")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    claude = find_claude()
    paths = args.analyses or sorted(
        p for p in (ROOT / "output").glob("*/*.analysis.json")
        if not p.name.endswith(".trimmed.analysis.json"))

    todo = []
    for path in paths:
        analysis = json.loads(path.read_text(encoding="utf-8"))
        labelled = analysis["sections"] and analysis["sections"][0].get("source") == "claude"
        if args.relabel or not labelled:
            todo.append((path, analysis))
    skipped = len(paths) - len(todo)
    # songs that failed before go last, and give up after three tries, so they cannot block
    # the queue on every run
    fail_log = ROOT / "label_failures.json"
    fails = json.loads(fail_log.read_text(encoding="utf-8")) if fail_log.exists() else {}
    todo = sorted((t for t in todo if args.relabel or fails.get(t[0].name, 0) < 3),
                  key=lambda t: fails.get(t[0].name, 0))
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(paths)} analyses found, {skipped} already labelled, labelling {len(todo)} "
          f"with {args.model}.")

    done, failed, streak, cost = 0, 0, 0, 0.0
    for path, analysis in todo:
        name = analysis["source"]["file"]
        shown = analysis["source"].get("title", name)
        try:
            result, reply = ask_claude(claude, build_payload(analysis), args.model, args.timeout)
            if result is None and "content filtering" in str(reply.get("result", "")):
                result, reply = ask_claude(claude, build_payload(analysis, redact=True),
                                           args.model, args.timeout)
                if result is not None:
                    print(f"  {shown}: labelled from repetition tags (lyrics withheld)",
                          flush=True)
        except subprocess.TimeoutExpired:
            result, reply = None, {"result": f"no answer within {args.timeout}s"}
        if result is None or not apply_labels(analysis, result):
            message = str(reply.get("result") or "no usable sections returned")
            if any(k in message.lower() for k in BLOCKING):
                print(f"\nStopped at {shown}: {message}", flush=True)
                break
            # a problem with this one song: skip it, it is retried on the next run
            failed, streak = failed + 1, streak + 1
            fails[path.name] = fails.get(path.name, 0) + 1
            fail_log.write_text(json.dumps(fails, indent=1), encoding="utf-8")
            print(f"  skipped {shown}: {message[:200]}", flush=True)
            if streak >= 3:
                print("\nThree songs in a row failed; stopping.", flush=True)
                break
            continue
        streak = 0

        path.write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
        audio = args.songs_dir / name
        if audio.exists():
            html = path.with_name(path.name.replace(".analysis.json", ".review.html"))
            write_review(analysis, audio.resolve(), html, load_audio(audio).mean(0))
        done += 1
        cost += reply.get("total_cost_usd") or 0.0
        form = " ".join(s["label"] for s in analysis["sections"])
        print(f"[{done}/{len(todo)}] {shown}: {form}", flush=True)

    left = len(todo) - done
    print(f"\nLabelled {done} song(s), {failed} skipped, {left} not labelled in this run. "
          f"API-equivalent cost of this run: ${cost:.2f} (not charged on a subscription).",
          flush=True)
    if left:
        print("If this was a usage limit, rerun later; finished songs are skipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
