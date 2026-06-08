#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default storage location
STABLEWM_HOME="$HOME/.stable-wm"

# Configure environment variables
export STABLEWM_HOME="$(realpath "$STABLEWM_HOME")"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"

echo "=================================================="
echo "Launching Evaluation Session:"
echo "  STABLEWM_HOME: $STABLEWM_HOME"
echo "  Arguments:     $@"
echo "=================================================="
echo ""

python eval.py "$@"
