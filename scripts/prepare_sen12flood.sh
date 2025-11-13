# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

#!/usr/bin/env bash
set -euo pipefail

### conda: load shell functions, then activate env
if [ -f "/home/<USER_ID>/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/<USER_ID>/miniconda/etc/profile.d/conda.sh"
else
  eval "$(/home/<USER_ID>/miniconda/bin/conda shell.bash hook)"
fi
conda activate /home/<USER_ID>/miniconda/envs/sarwmix

# -----------------------------
# SEN12FLOOD end-to-end prep:
#  1) unzip (if a zip is given)
#  2) ALIGN: S1 -> S2 grid (temp)
#  3) PAIR:  build pair .npy (temp)
#  4) CLEAN: drop partial-coverage pairs (final only)
# -----------------------------

# Defaults (leave empty; pass ONE of --zip or --raw-root; args on command prompt)
ZIP_PATH=""
RAW_ROOT=""
WORK_DIR=""
OUT_DIR=""
KEEP_INTERMEDIATE=0

log() { echo -e "[$(date +'%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--zip /path/SEN12FLOOD.zip | --raw-root /path/SEN12FLOOD] \\
                   --work /scratch/sen12_work --out /data/CURATED_SEN12FLOOD \\
                   [--keep-intermediate]
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP_PATH="$2"; shift 2 ;;
    --raw-root) RAW_ROOT="$2"; shift 2 ;;
    --work) WORK_DIR="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --keep-intermediate) KEEP_INTERMEDIATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -z "$WORK_DIR" || -z "$OUT_DIR" ]] && { usage; exit 1; }
[[ -n "$ZIP_PATH" && -n "$RAW_ROOT" ]] && die "Provide either --zip or --raw-root (not both)."
[[ -z "$ZIP_PATH" && -z "$RAW_ROOT" ]] && die "Provide one of --zip or --raw-root."

mkdir -p "$WORK_DIR" "$OUT_DIR"
TMP_ALIGN="$WORK_DIR/_aligned"
TMP_PAIRS="$WORK_DIR/_pairs"

# detect repo root (this script is expected under repo/scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"

# make these visible to inline python; also run from repo root so rel paths work
export REPO_ROOT RAW_ROOT
cd "$REPO_ROOT"

# helpers
count_files() {
  local dir="$1"; local pattern="$2"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f -name "$pattern" -print 2>/dev/null | wc -l | awk '{print $1}'
  else
    echo 0
  fi
}
progress_bar() {
  local current="$1" total="$2" width="${3:-48}"
  (( total==0 )) && total=1
  local pct=$(( 100 * current / total ))
  local filled=$(( width * current / total ))
  (( filled > width )) && filled=$width
  local unfilled=$(( width - filled ))
  printf -v bar_f "%${filled}s"; bar_f=${bar_f// /#}
  printf -v bar_u "%${unfilled}s"; bar_u=${bar_u// /.}
  printf "\r[%s%s] %3d%% (%d/%d)" "$bar_f" "$bar_u" "$pct" "$current" "$total"
}
remove_dir() {
  local p="$1"
  if [[ -e "$p" ]]; then
    rm -rf -- "$p" || true
    if [[ -e "$p" ]]; then
      log "warn: failed to remove $p (check permissions/open files)"
    else
      log "removed: $p"
    fi
  else
    log "already gone: $p"
  fi
}

# 0) If ZIP provided: summary + unzip
if [[ -n "$ZIP_PATH" ]]; then
  [[ -f "$ZIP_PATH" ]] || die "ZIP not found: $ZIP_PATH"
  log "zip provided → $ZIP_PATH"
  if command -v zipinfo >/dev/null 2>&1; then
    log "zip summary:"
    echo "ZIP File Summary:"
    echo "-----------------"
    echo "File Size: $(stat -c %s "$ZIP_PATH") bytes"
    file_count=$(zipinfo -1 "$ZIP_PATH" | wc -l)
    dir_count=$(zipinfo -1 "$ZIP_PATH" | awk -F/ '{if(NF>1) print $1"/"$2}' | sort -u | wc -l)
    echo "Number of Files: $file_count"
    echo "Number of Directories: $dir_count"
    echo "File Types:"
    zipinfo -1 "$ZIP_PATH" | awk -F. 'NF>1 {ext=$NF; count[ext]++} END {for (ext in count) print count[ext], "." ext " files"}' | sort -n
  else
    log "zipinfo not found; skipping summary."
  fi

  RAW_ROOT="$WORK_DIR/SEN12FLOOD"
  log "unzip → $RAW_ROOT"
  mkdir -p "$RAW_ROOT"
  unzip -q -o "$ZIP_PATH" -d "$WORK_DIR"
fi

[[ -d "$RAW_ROOT" ]] || die "RAW_ROOT not found: $RAW_ROOT (expected SEN12FLOOD root with train/test folders)"

# sanity: required files
[[ -f "$DATASETS_DIR/S1list.json"     ]] || die "Missing $DATASETS_DIR/S1list.json"
[[ -f "$DATASETS_DIR/sen12_train.csv" ]] || die "Missing $DATASETS_DIR/sen12_train.csv"
[[ -f "$DATASETS_DIR/sen12_test.csv"  ]] || die "Missing $DATASETS_DIR/sen12_test.csv"

# Ensure PYTHONPATH contains repo for relative imports
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/sarwmix:$REPO_ROOT/util:${PYTHONPATH:-}"

# 1) ALIGN (S1 -> S2 grid) → TMP_ALIGN
log "step 1/3: ALIGN (S1 -> S2 grid)"
rm -rf "$TMP_ALIGN"; mkdir -p "$TMP_ALIGN"

total_s1=$(count_files "$RAW_ROOT" "S1*.tif")
log "found $total_s1 S1 GeoTIFFs to align"
python "$REPO_ROOT/sarwmix/sen12_align_s1_to_s2.py" \
  --dataset-root "$RAW_ROOT" \
  --save-root "$TMP_ALIGN" \
  > "$WORK_DIR/_align.out" 2>&1 &

ALIGN_PID=$!
while kill -0 "$ALIGN_PID" 2>/dev/null; do
  done_now=$(count_files "$TMP_ALIGN" "S1*.tif")
  progress_bar "$done_now" "$total_s1"
  sleep 5
done
wait "$ALIGN_PID" || die "alignment failed (see $WORK_DIR/_align.out)"
progress_bar "$total_s1" "$total_s1"; echo
log "alignment done."

# 2) PAIR (build .npy pairs) → TMP_PAIRS
log "step 2/3: PAIR (create .npy pairs)"
rm -rf "$TMP_PAIRS"; mkdir -p "$TMP_PAIRS"

log "estimating expected pair count (approx)…"
expected_pairs=$(python - "$DATASETS_DIR" <<'PY'
import json, csv, sys
from pathlib import Path

datasets = Path(sys.argv[1])  # absolute path passed from bash
with open(datasets / "S1list.json", "r") as f:
    meta = json.load(f)

def load_list(name: str):
    out = []
    with open(datasets / name) as f:
        for r in csv.reader(f):
            if r: out.append(r[0])
    return out

def count_pairs(split_csv: str):
    folders = load_list(split_csv)
    count = 0
    for folder in folders:
        entries = meta.get(folder)
        if not entries:
            continue
        false_items, all_items = [], []
        for k, v in entries.items():
            if not k.isdigit():
                continue
            fn = v["filename"]; flood = v["FLOODING"]
            all_items.append((fn, flood))
            if not flood:
                false_items.append((fn, flood))
        seen = set()
        for a in false_items:
            for b in all_items:
                if a[0] == b[0]:
                    continue
                pair = tuple(sorted([a[0], b[0]]))
                if pair in seen:
                    continue
                seen.add(pair)
        count += len(seen)
    return count

print(count_pairs("sen12_train.csv") + count_pairs("sen12_test.csv"))
PY
) || expected_pairs=0
log "expected pairs (approx): $expected_pairs"

python "$REPO_ROOT/sarwmix/sen12_data_prep.py" \
  --dataset-root "$TMP_ALIGN" \
  --save-root "$TMP_PAIRS" \
  > "$WORK_DIR/_pairs.out" 2>&1 &

PAIR_PID=$!
while kill -0 "$PAIR_PID" 2>/dev/null; do
  made_now=$(count_files "$TMP_PAIRS" "*.npy")
  if [[ "${expected_pairs:-0}" -gt 0 ]]; then
    progress_bar "$made_now" "$expected_pairs"
  else
    printf "\rcreated %d pair files…" "$made_now"
  fi
  sleep 5
done
wait "$PAIR_PID" || die "pairing failed (see $WORK_DIR/_pairs.out)"
[[ "${expected_pairs:-0}" -gt 0 ]] && { progress_bar "$expected_pairs" "$expected_pairs"; echo; }
log "pairing done."

# 3) CLEAN (drop partial coverage) → OUT_DIR
log "step 3/3: CLEAN (remove partial-coverage pairs ≥25% zeros)"
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
# pre-create subdirs so 'find' doesn't fail before the python script creates them
mkdir -p "$OUT_DIR/train" "$OUT_DIR/test"

# totals per split for a nicer progress model
total_train_pairs=$(count_files "$TMP_PAIRS/train" "*.npy")
total_test_pairs=$(count_files "$TMP_PAIRS/test" "*.npy")
total_pairs=$(( total_train_pairs + total_test_pairs ))
log "total raw pairs: train=$total_train_pairs, test=$total_test_pairs, all=$total_pairs"

python "$REPO_ROOT/sarwmix/sen12_prune_partial_pairs.py" \
  --dataset-root "$TMP_PAIRS" \
  --save-root   "$OUT_DIR" \
  > "$WORK_DIR/_clean.out" 2>&1 &

CLEAN_PID=$!
while kill -0 "$CLEAN_PID" 2>/dev/null; do
  kept_train=$(count_files "$OUT_DIR/train" "*.npy")
  kept_test=$(count_files "$OUT_DIR/test"  "*.npy")

  # progress: train contributes 0..90, test contributes 0..10
  pct_train=0
  pct_test=0
  if (( total_train_pairs > 0 )); then
    pct_train=$(( kept_train * 90 / total_train_pairs ))
    (( pct_train > 90 )) && pct_train=90
  fi
  if (( total_test_pairs > 0 )); then
    pct_test=$(( kept_test * 10 / total_test_pairs ))
    (( pct_test > 10 )) && pct_test=10
  fi
  pct=$(( pct_train + pct_test ))
  progress_bar "$pct" 100
  sleep 5
done
wait "$CLEAN_PID" || die "clean step failed (see $WORK_DIR/_clean.out)"
progress_bar 100 100; echo
log "clean done."

# Summary
final_train=$(count_files "$OUT_DIR/train" "*.npy")
final_test=$(count_files "$OUT_DIR/test" "*.npy")
train_pos=$(find "$OUT_DIR/train" -type f -name "*__1.npy" 2>/dev/null | wc -l | awk '{print $1}')
train_neg=$(( final_train - train_pos ))
test_pos=$(find "$OUT_DIR/test" -type f -name "*__1.npy" 2>/dev/null | wc -l | awk '{print $1}')
test_neg=$(( final_test - test_pos ))

log "final counts:"
echo "  train: $final_train (pos=$train_pos, neg=$train_neg)"
echo "  test : $final_test  (pos=$test_pos,  neg=$test_neg)"
echo "  total: $((final_train+final_test))"

# Cleanup
if [[ "$KEEP_INTERMEDIATE" -eq 0 ]]; then
  log "removing intermediates: $TMP_ALIGN and $TMP_PAIRS"
  # remove with verification
  remove_dir "$TMP_ALIGN"
  remove_dir "$TMP_PAIRS"
else
  log "keeping intermediates under $WORK_DIR"
fi

log "done. final dataset: $OUT_DIR (train/ and test/)"

