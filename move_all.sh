set -euo pipefail

ROOT="${1:-DroneRF}"
HIGH_DIR="$ROOT/High"
LOW_DIR="$ROOT/Low"

mkdir -p "$HIGH_DIR" "$LOW_DIR"

# Helper: move one file to target, adding _N suffix if needed
move_with_unique_name() {
  local src="$1" tgt_dir="$2"
  local base ext name out i
  base="$(basename -- "$src")"
  name="${base%.*}"
  ext="${base##*.}"
  if [[ "$ext" == "$base" ]]; then  # no extension
    ext=""
    out="$tgt_dir/$name"
  else
    out="$tgt_dir/$name.$ext"
  fi

  i=1
  while [[ -e "$out" ]]; do
    if [[ -z "$ext" ]]; then
      out="$tgt_dir/${name}_$i"
    else
      out="$tgt_dir/${name}_$i.$ext"
    fi
    ((i++))
  done

  mv -- "$src" "$out"
}

# Find all dirs ending with _h/_l (case-insensitive), skip High/Low destinations
find "$ROOT" -type d -regextype posix-extended -iregex '.*/[^/]+_[hl]$' \
  ! -path "$HIGH_DIR" ! -path "$LOW_DIR" -print0 |
while IFS= read -r -d '' d; do
  shopt -s nocasematch
  if [[ "$d" == *_h ]]; then
    dest="$HIGH_DIR"
  else
    dest="$LOW_DIR"
  fi
  shopt -u nocasematch

  find "$d" -type f -iname '*.csv' -print0 |
  while IFS= read -r -d '' f; do
    move_with_unique_name "$f" "$dest"
  done
done

echo "✅ Done. All CSVs moved into:"
echo "   High -> $HIGH_DIR"
echo "   Low  -> $LOW_DIR"
