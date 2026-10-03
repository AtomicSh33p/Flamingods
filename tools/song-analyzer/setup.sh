#!/usr/bin/env bash
# Creates the analyzer's Python environment at ~/.venvs/song-analyzer.
# Works on Linux, WSL Ubuntu and macOS. Needs Python 3.11+ with venv, and ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"

missing=()
python3 -c "import ensurepip" 2>/dev/null || missing+=("python3-venv")
command -v ffmpeg >/dev/null || missing+=("ffmpeg")
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing: ${missing[*]}"
  if [ "$(uname)" = "Darwin" ]; then
    echo "Install with: brew install python ffmpeg"
  else
    echo "Install with: sudo apt install -y ${missing[*]}"
    echo "(In WSL without your Ubuntu password: wsl -u root -e apt-get install -y ${missing[*]})"
  fi
  exit 1
fi

# the venv lives in the home folder: in WSL, /mnt/c is too slow for thousands of small files
VENV="$HOME/.venvs/song-analyzer"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check -q --upgrade pip
"$VENV/bin/python" -m pip install --disable-pip-version-check -q -r requirements.txt
"$VENV/bin/python" -c "import torch; print('Ready. GPU:', torch.cuda.is_available())"
echo "Run the tools with: $VENV/bin/python <script>.py"
