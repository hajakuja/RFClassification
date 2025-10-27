#!/usr/bin/bash
set -euo pipefail

ROOT_DIR="${1:-DroneRF}"

# Prefer unrar; fall back to 7z if available.
extract_cmd() {
  local archive="$1" outdir="$2"
  if command -v unrar >/dev/null 2>&1; then
    # -o+ overwrite, -idq quiet log, -inul suppress errors to stderr (non-fatal)
    unrar x -o+ -idq -inul -- "$archive" "$outdir/"
  elif command -v 7z >/dev/null 2>&1; then
    7z x -y -o"$outdir" -- "$archive" >/dev/null
  else
    echo "Error: need unrar or 7z installed." >&2
    exit 1
  fi
}

# Find RARs (case-insensitive). Handle spaces/newlines in names safely.
# Skip multi-part volumes that are NOT the first one (e.g., .part2.rar).
find "$ROOT_DIR" -type f \( -iname '*.rar' \) -print0 |
while IFS= read -r -d '' f; do
  bn="$(basename -- "$f")"
  dn="$(dirname -- "$f")"

  # If this is a multi-part archive like name.partN.rar, only extract part1
  if [[ "$bn" =~ \.part([0-9]+)\.rar$ ]]; then
    partnum="${BASH_REMATCH[1]}"
    if [[ "$partnum" != "1" ]]; then
      # skip non-first parts; unrar/7z will use other parts automatically
      continue
    fi
    base="${bn%.part${partnum}.rar}"
  else
    base="${bn%.rar}"
  fi

  out="$dn/$base"
  mkdir -p -- "$out"
  echo "Extracting: $f -> $out"
  extract_cmd "$f" "$out"
done

echo "Done."
