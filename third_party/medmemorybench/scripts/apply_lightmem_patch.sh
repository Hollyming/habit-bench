#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
LIGHTMEM_ROOT="$PROJECT_ROOT/methods/LightMem"
PATCH_FILE="$PROJECT_ROOT/patches/lightmem-medmemorybench.patch"

if [[ ! -d "$LIGHTMEM_ROOT/.git" && ! -f "$LIGHTMEM_ROOT/.git" ]]; then
  echo "LightMem submodule is not initialized: $LIGHTMEM_ROOT" >&2
  exit 2
fi

if git -C "$LIGHTMEM_ROOT" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "LightMem compatibility patch is already applied."
  exit 0
fi

git -C "$LIGHTMEM_ROOT" apply --check "$PATCH_FILE"
git -C "$LIGHTMEM_ROOT" apply "$PATCH_FILE"
echo "Applied LightMem compatibility patch."
