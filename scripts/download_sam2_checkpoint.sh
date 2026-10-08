#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt}"
mkdir -p "$(dirname "$DESTINATION")"
curl -L --fail --retry 2 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o "$DESTINATION"
echo "Downloaded official SAM2.1 Hiera Tiny checkpoint to $DESTINATION"
