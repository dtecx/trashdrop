#!/usr/bin/env bash
# Fetch only the SO-ARM100 model used as the SO-101 kinematic stand-in.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$repo_root/vendor/mujoco_menagerie"

if [[ -f "$target/trs_so_arm100/so_arm100.xml" ]]; then
  echo "SO-ARM model already available at $target/trs_so_arm100"
  exit 0
fi

if [[ -e "$target" ]]; then
  echo "Refusing to modify unexpected path: $target" >&2
  exit 1
fi

mkdir -p "$(dirname "$target")"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git "$target"
git -C "$target" sparse-checkout set trs_so_arm100

test -f "$target/trs_so_arm100/so_arm100.xml"
find "$target/trs_so_arm100/assets" -name '*.stl' -print -quit | grep -q .
echo "Fetched SO-ARM100 model into $target/trs_so_arm100"
