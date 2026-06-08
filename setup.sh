#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values
DATASET="tworoom"
STABLEWM_HOME="$HOME/.stable-wm"

show_help() {
    echo "Usage: ./setup.sh [options]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset VAL       Dataset to set up. Options: tworoom, pusht, reacher, cube, all, none"
    echo "                          (default: tworoom - the smallest dataset for fast testing)"
    echo "  -s, --stablewm-home VAL Path to the local stable-wm directory"
    echo "                          (default: $HOME/.stable-wm)"
    echo "  -h, --help              Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--dataset)
            DATASET="$2"
            shift; shift
            ;;
        -s|--stablewm-home)
            STABLEWM_HOME="$2"
            shift; shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate dataset selection
if [[ "$DATASET" != "pusht" && "$DATASET" != "reacher" && "$DATASET" != "tworoom" && "$DATASET" != "cube" && "$DATASET" != "all" && "$DATASET" != "none" ]]; then
    echo "Error: Invalid dataset '$DATASET'."
    echo "Must be one of: tworoom, pusht, reacher, cube, all, none"
    exit 1
fi

# Set up storage directory
export STABLEWM_HOME="$(realpath "$STABLEWM_HOME")"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
mkdir -p "$STABLEWM_HOME"

echo "=================================================="
echo "Configuring environment paths:"
echo "  STABLEWM_HOME:     $STABLEWM_HOME"
echo "  LOCAL_DATASET_DIR: $LOCAL_DATASET_DIR"
echo "=================================================="

# Check and install system packages (zstd)
if ! command -v zstd &> /dev/null; then
    echo "zstd not found. Installing system package..."
    if command -v apt-get &> /dev/null; then
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update && apt-get install -y zstd
        elif command -v sudo &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y zstd
        else
            echo "Warning: Could not install zstd. Please install it manually."
        fi
    elif command -v yum &> /dev/null; then
        if [ "$(id -u)" -eq 0 ]; then
            yum install -y zstd
        elif command -v sudo &> /dev/null; then
            sudo yum install -y zstd
        else
            echo "Warning: Could not install zstd. Please install it manually."
        fi
    else
        echo "Warning: Package manager not recognized. Please install zstd manually."
    fi
fi

# Install Python dependencies
echo "=================================================="
echo "Setting up Python dependencies..."
echo "=================================================="
pip install --upgrade pip wheel
pip install "stable-worldmodel[all]"
pip install lightning hydra-core omegaconf wandb huggingface_hub

# Helper function to download and extract dataset
download_and_extract() {
    local repo="$1"
    local filename="$2"
    
    echo ""
    echo "--------------------------------------------------"
    echo "Downloading $filename from Hugging Face dataset $repo..."
    echo "--------------------------------------------------"
    huggingface-cli download --repo-type dataset "$repo" "$filename" --local-dir "$STABLEWM_HOME"
    
    echo "Extracting $filename..."
    tar --zstd -xvf "$STABLEWM_HOME/$filename" -C "$STABLEWM_HOME"
}

# Download specified dataset(s)
download_dataset() {
    local ds="$1"
    
    case $ds in
        pusht)
            download_and_extract "quentinll/lewm-pusht" "pusht_expert_train.h5.zst"
            ;;
        reacher)
            download_and_extract "quentinll/lewm-reacher" "reacher.tar.zst"
            # Adjust path if nested
            if [ -f "$STABLEWM_HOME/reacher/reacher.h5" ]; then
                mv "$STABLEWM_HOME/reacher/reacher.h5" "$STABLEWM_HOME/reacher.h5"
                rmdir "$STABLEWM_HOME/reacher" || true
            fi
            ;;
        tworoom)
            download_and_extract "quentinll/lewm-tworooms" "tworoom.tar.zst"
            # Adjust path if nested
            if [ -f "$STABLEWM_HOME/tworoom/tworoom.h5" ]; then
                mv "$STABLEWM_HOME/tworoom/tworoom.h5" "$STABLEWM_HOME/tworoom.h5"
                rmdir "$STABLEWM_HOME/tworoom" || true
            fi
            ;;
        cube)
            download_and_extract "quentinll/lewm-cube" "cube_single_expert.tar.zst"
            # Adjust path to match expected directory structure: ogbench/cube_single_expert.h5
            mkdir -p "$STABLEWM_HOME/ogbench"
            if [ -f "$STABLEWM_HOME/cube_single_expert.h5" ]; then
                mv "$STABLEWM_HOME/cube_single_expert.h5" "$STABLEWM_HOME/ogbench/cube_single_expert.h5"
            elif [ -f "$STABLEWM_HOME/cube_single_expert/cube_single_expert.h5" ]; then
                mv "$STABLEWM_HOME/cube_single_expert/cube_single_expert.h5" "$STABLEWM_HOME/ogbench/cube_single_expert.h5"
                rmdir "$STABLEWM_HOME/cube_single_expert" || true
            fi
            ;;
        all)
            download_dataset "tworoom"
            download_dataset "pusht"
            download_dataset "reacher"
            download_dataset "cube"
            ;;
        none)
            echo "Skipping dataset download."
            ;;
    esac
}

if [[ "$DATASET" != "none" ]]; then
    echo "=================================================="
    echo "Downloading and preparing dataset: $DATASET"
    echo "=================================================="
    download_dataset "$DATASET"
fi

echo ""
echo "=================================================="
echo "Setup complete!"
echo "You can now run training using: ./train.sh"
echo "=================================================="
