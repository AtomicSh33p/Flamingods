#!/usr/bin/env bash
# Unattended full run: analyzes every song (no Claude usage) while labelling whatever is
# analyzed with Claude. When the Claude plan's usage limit is hit, labelling waits and
# retries every 20 minutes, and the analysis carries on regardless. Logs go to the data dir.
#
#   run_all.sh [pilot video ids to redo first...]
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
PY="$HOME/.venvs/song-analyzer/bin/python -u"
DATA=$($PY -c "from songlib import data_dir; print(data_dir())")
DEADLINE=$(( $(date +%s) + 36 * 3600 ))
rm -f "$DATA/analysis.done"
log() { echo "[$(date '+%F %T')] $*"; }

(
  if [ $# -gt 0 ]; then
    log "redoing pilot songs with the current code"
    $PY batch.py run --force --ids "$@"
  fi
  # later passes retry songs that failed, e.g. throttled downloads
  for pass in 1 2 3; do
    log "analysis pass $pass"
    $PY batch.py run
    sleep 600
  done
  touch "$DATA/analysis.done"
  log "analysis finished"
) >> "$DATA/analysis.log" 2>&1 &

(
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    log "labelling run"
    if $PY label.py && [ -f "$DATA/analysis.done" ]; then
      $PY label.py && break  # one last sweep for songs analyzed during the previous run
    fi
    $PY batch.py index > /dev/null
    sleep 1200
  done
  log "labelling loop finished"
) >> "$DATA/labels.log" 2>&1 &

wait
$PY batch.py index > /dev/null
log "all done" >> "$DATA/analysis.log"
