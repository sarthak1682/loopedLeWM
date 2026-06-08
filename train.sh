#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values
DATASET="tworoom"
PREDICTOR_TYPE="looped"
NUM_LOOPS=6
MAX_EPOCHS=""
STABLEWM_HOME="$HOME/.stable-wm"

show_help() {
    echo "Usage: ./train.sh [options]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset VAL       Dataset/environment to train on. Options: tworoom, pusht, reacher, cube"
    echo "                          (default: tworoom)"
    echo "  -p, --predictor VAL     Predictor type to use: standard or looped"
    echo "                          (default: looped)"
    echo "  -k, --num-loops VAL     Number of loop iterations (K) for the looped predictor"
    echo "                          (default: 6)"
    echo "  -e, --epochs VAL        Override trainer.max_epochs (e.g. 2 epochs for quick testing)"
    echo "                          (default: uses the yaml config default, e.g. 100)"
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
        -p|--predictor)
            PREDICTOR_TYPE="$2"
            shift; shift
            ;;
        -k|--num-loops)
            NUM_LOOPS="$2"
            shift; shift
            ;;
        -e|--epochs)
            MAX_EPOCHS="$2"
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

# Validate dataset
if [[ "$DATASET" != "pusht" && "$DATASET" != "reacher" && "$DATASET" != "tworoom" && "$DATASET" != "cube" ]]; then
    echo "Error: Invalid dataset '$DATASET'."
    echo "Must be one of: tworoom, pusht, reacher, cube"
    exit 1
fi

# Set up environment variables for dataset lookup
export STABLEWM_HOME="$(realpath "$STABLEWM_HOME")"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"

# Map dataset name to the config/train/data/ yaml filename
case $DATASET in
    pusht)
        DATA_CFG="pusht"
        ;;
    reacher)
        DATA_CFG="dmc"
        ;;
    tworoom)
        DATA_CFG="tworoom"
        ;;
    cube)
        DATA_CFG="ogb"
        ;;
esac

# Build training command
CMD="python train.py data=$DATA_CFG predictor_type=$PREDICTOR_TYPE num_loops=$NUM_LOOPS"

if [ -n "$MAX_EPOCHS" ]; then
    CMD="$CMD trainer.max_epochs=$MAX_EPOCHS"
fi

echo "=================================================="
echo "Launching Training Session:"
echo "  Dataset:       $DATASET (config: data=$DATA_CFG)"
echo "  Predictor:     $PREDICTOR_TYPE (K = $NUM_LOOPS)"
echo "  Max Epochs:    ${MAX_EPOCHS:-Config Default}"
echo "  STABLEWM_HOME: $STABLEWM_HOME"
echo "=================================================="
echo "Running: $CMD"
echo "=================================================="
echo ""

eval "$CMD"
