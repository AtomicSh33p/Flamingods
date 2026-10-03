# Song Analyzer

Finds the structure of songs for the Flamingods lip-sync stage. For each song it produces a
JSON file with timestamps for:

- silence, and where someone is singing versus instrumental parts
- sections: intro, verse, pre-chorus, chorus, bridge, instrumental, outro
- the lyrics, line by line and word by word (automatic transcription)
- tempo, beats and bar starts, an energy curve, and energy jumps and drops
- suggested cuts: long intros, outros and instrumental breaks to skip on stage

It also builds a review page per song (waveform, coloured sections, lyrics, playback, a
"skip the suggested cuts" preview) and an overview page of all songs with warning flags.

The analysis runs on your computer. Only the optional labelling step talks to Claude, and it
sends text (timed lyrics and section candidates), never audio.

## What is already done

The stage's three playlists (Solo, Duets, Steal the Thunder) are already analyzed: 387 songs,
labelled with Claude. Those results are not in this repo, because they include transcribed
lyrics of copyrighted songs; they are shared privately as `flamingods-results.zip`. Unzip it
into this folder so you get `results/songs.json` and `results/output/`. Nothing under
`results/` is ever committed.

The audio is not included either (copyrighted, and 1.4 GB); you download it yourself with one
command, below.

## Setup

You need about 4 GB of disk for the Python packages and models, plus 1.4 GB if you download
all the audio. An NVIDIA GPU makes analysis fast (under a minute per song); without one it
runs on the CPU at roughly 3 to 5 minutes per song.

**macOS:** install [Homebrew](https://brew.sh), then `brew install python ffmpeg`.

**Linux:** `sudo apt install -y python3-venv ffmpeg`.

**Windows:** use WSL (Ubuntu). Native Windows is not supported: on machines with Smart App
Control turned on, Windows blocks several of the packages this tool needs. In PowerShell,
`wsl --install -d Ubuntu`, restart, then inside Ubuntu `sudo apt install -y python3-venv
ffmpeg`. Your Windows drive is at `/mnt/c/` inside Ubuntu. WSL uses an NVIDIA GPU
automatically if the Windows driver is installed.

Then, in this folder:

```bash
bash setup.sh
```

It creates the environment at `~/.venvs/song-analyzer`. Every command below starts with that
environment's Python; to save typing:

```bash
alias songpy=~/.venvs/song-analyzer/bin/python
```

## Using the shared results

After unzipping the results:

```bash
songpy batch.py fetch
```

Downloads the audio for every analyzed song into `results/songs/` (git-ignored), then builds
the review pages and `results/index.html`, the overview. Open that file in a browser. YouTube
sometimes refuses a download; run `fetch` again later to retry those.

Without the audio you can still read the JSON files, or run `songpy batch.py pages` to build
the pages without waveforms or playback.

## Adding songs

One song from YouTube:

```bash
songpy batch.py add "https://www.youtube.com/watch?v=VIDEO_ID" --list solo
```

It adds the song to `results/songs.json`, downloads and analyzes it, labels it with Claude if
available (see below), and prints the review page's path.

If the stage playlists changed (or you are starting without the results zip), build the list
and analyze whatever has no result yet:

```bash
songpy batch.py manifest
```

```bash
nohup bash run_all.sh > results/launcher.log 2>&1 &
```

`run_all.sh` runs unattended: it downloads and analyzes every song that has no result yet,
labels them with Claude in parallel, waits and retries when a Claude usage limit is hit, and
writes `analysis.log` and `labels.log` into `results/`. It is resumable; if the computer
sleeps or restarts, run it again.

To share your new results, run `songpy batch.py export ~/results-to-share` and send that
folder (zipped) to the others; it holds `songs.json` and the analysis files without audio or
caches. Do not commit results: this repo is public and the results contain lyrics.

## Labels: with or without Claude

Section names come from one of three sources; each section records which one in `source`.

- **Your Claude subscription (recommended):** install Claude Code with
  `curl -fsSL https://claude.ai/install.sh | bash`, run `~/.local/bin/claude auth login` once,
  and `add` / `run_all.sh` use it automatically. Labelling one song takes a few seconds and
  counts against your plan's usage limits. To label songs that only have local labels:
  `songpy label.py` (add `--model sonnet` to use less of your allowance).
- **An Anthropic API key (pay per use, a few cents per song):** copy `env.example` to `.env`,
  set the key there, and `analyze.py` labels each song as part of the analysis.
- **Neither:** the tool guesses labels locally from lyric repetition. Vocals, lyrics, beats and
  cuts are unaffected, but verse/chorus names are often wrong.

If a famous song's lyrics trip Claude's copyright filter, `label.py` automatically retries
with the lyric text replaced by tags that only show which lines repeat.

## Other tools

```bash
songpy trim.py results/songs/VIDEO_ID.m4a
```

Writes a shortened copy of the song with the suggested cuts applied (crossfaded, cut on bar
starts) and a matching JSON. To change what gets cut, edit `suggested_cuts` in the song's
analysis JSON first.

```bash
songpy analyze.py some-file.mp3
```

Analyzes any local audio file. Useful flags: `--language he` or `en` (default: Hebrew if the
title is Hebrew, otherwise detected), `--vocal-db` and `--vocal-ratio-db` for vocal
detection sensitivity, `--min-cut`, `--min-mid-cut` and `--keep-bars` for the cut rule,
`--device cpu`, `--force` to ignore cached work.

To keep the data somewhere other than `results/` (for example a bigger drive), copy
`config.example.json` to `config.json` and set `data_dir`.

## How it works

1. **Silence:** loudness of the full mix relative to the song's loud level.
2. **Vocals:** Demucs separates the vocal track. Singing is "active" where that track is loud,
   or quiet but still dominating the mix (soft endings).
3. **Lyrics:** Whisper on the vocal track with word timestamps; the ivrit-ai model for Hebrew,
   `large-v3-turbo` for other languages.
4. **Beats:** librosa beat tracking on the instrumental. Bar starts assume 4/4 and are
   estimated.
5. **Sections:** candidate boundaries from where vocals start and stop, lyric repetition and
   audio similarity; then Claude names them from the timed lyrics.
6. **Cuts:** intro and outro when at least 4 seconds can go, vocal-free breaks mid-song only
   when at least 15 seconds can go (short mid-song cuts sound wrong). About one bar is kept
   around the vocals, and cuts sit on bar starts.

## Accuracy and the overview flags

Silence, vocal ranges, lyric timing and tempo are reliable. Transcribed words contain errors,
and section labels and bar starts are estimates. The overview flags the songs worth a listen:
`no vocals found`, `few vocals` (often a karaoke or wrong upload), `sparse lyrics`,
`language unsure`, `large cut` (over 35% of the song), `no chorus`, and
`low-confidence label` (Claude was unsure about a sung section).

## JSON format

```
source          file, duration, id, title, channel, lists, language, language_probability
summary         one or two sentences on the song's form (Claude labels only)
tempo           bpm, time_signature, beats[], downbeats[]
sections[]      start, end, label, index, vocal_coverage, energy (0-1), words_per_second,
                lyrics, source (heuristic|claude), confidence, note
activity[]      start, end, type (vocals|instrumental|silence); covers the whole song
lyrics[]        start, end, text, words[] (start, end, text)
energy          hop_seconds, values[] (0-1)
highlights[]    time, type (energy_rise|energy_drop|peak_energy), change_db
suggested_cuts[] start, end, reason
```

All times are seconds from the start of the original YouTube audio.
