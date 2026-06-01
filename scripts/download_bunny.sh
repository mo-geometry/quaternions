#!/bin/bash
# Download the Stanford bunny OBJ file.
# Run from the project root: bash scripts/download_bunny.sh

set -e

ASSETS_DIR="$(dirname "$0")/../assets"
mkdir -p "$ASSETS_DIR"

URL="https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
DEST="$ASSETS_DIR/bunny.obj"

if [ -f "$DEST" ]; then
    echo "bunny.obj already exists at $DEST"
    exit 0
fi

echo "Downloading Stanford bunny from $URL ..."
curl -L "$URL" -o "$DEST"
echo "Saved to $DEST"

# Quick sanity check
VERTS=$(grep -c "^v " "$DEST")
FACES=$(grep -c "^f " "$DEST")
echo "Vertices: $VERTS, Faces: $FACES"
