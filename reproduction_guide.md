# Standard and Looped LeWM Setup & Execution Guide

This document details how to clone, set up, and train the Looped Predictor (weight-tied) and Standard Predictor models of LeWM.

---

## 1. Clone and Configure Paths

Clone the repository to the destination server:

```bash
git clone https://github.com/sarthak1682/loopedLeWM.git
cd loopedLeWM
```

Initialize the cache directory and link the default path to prevent internal file-not-found errors:

```bash
mkdir -p ~/.stable-wm
ln -sf ~/.stable-wm ~/.stable_worldmodel
```

---

## 2. Dependencies

Make sure `zstd` is installed for dataset extraction:
* Ubuntu / Debian: `sudo apt-get install -y zstd`
* CentOS / RHEL: `sudo yum install -y zstd`

Set up the Python environment (Python 3.10+):

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel
pip install "stable-worldmodel[all]"
pip install lightning hydra-core omegaconf wandb huggingface_hub
```

---

## 3. Dataset Setup

Run the setup script to download and extract datasets to `~/.stable-wm/datasets/`. 

Datasets ordered by size (smallest to largest):

### TwoRoom (~1.8 GB)
```bash
./setup.sh --dataset tworoom
```

### Reacher (~3.5 GB)
```bash
./setup.sh --dataset reacher
```

### PushT (~7.5 GB)
```bash
./setup.sh --dataset pusht
```

### Cube (~12.5 GB)
```bash
./setup.sh --dataset cube
```

---

## 4. Training

To utilize higher VRAM GPUs (A100/H100), override the batch size to `512` and scale the `SIGReg` weight to `0.0225`.

### Looped Predictor (K=6)
```bash
./train.sh \
  --dataset tworoom \
  --predictor looped \
  --num-loops 6 \
  --epochs 5 \
  loader.batch_size=512 \
  loader.num_workers=8 \
  output_model_name=lewm_looped \
  loss.sigreg.weight=0.0225
```

### Standard Predictor (Baseline, depth=6)
```bash
./train.sh \
  --dataset tworoom \
  --predictor standard \
  --epochs 5 \
  loader.batch_size=512 \
  loader.num_workers=8 \
  output_model_name=lewm_standard \
  loss.sigreg.weight=0.0225
```

---

## 5. Evaluation

To evaluate a model, symlink the target epoch's checkpoint to `weights.pt` inside the run directory.

### Evaluate Looped Predictor
```bash
ln -sf ~/.stable-wm/checkpoints/lewm_looped/weights_epoch_5.pt ~/.stable-wm/checkpoints/lewm_looped/weights.pt
./eval.sh --config-name=tworoom.yaml policy=lewm_looped/weights.pt
```

### Evaluate Standard Predictor
```bash
ln -sf ~/.stable-wm/checkpoints/lewm_standard/weights_epoch_5.pt ~/.stable-wm/checkpoints/lewm_standard/weights.pt
./eval.sh --config-name=tworoom.yaml policy=lewm_standard/weights.pt
```

---

## 6. Explanations of Crucial Fixes

### SIGReg Loss Scaling
The `SIGReg` module multiplies its output statistic by the batch size ($B$). When scaling the batch size from `128` to `512` (a 4x increase), the loss weight must be scaled down by 4x (from `0.09` to `0.0225`) to prevent regularizer dominance.

### Double Directory Nesting
The library appends a `/datasets` directory suffix internally. Set `LOCAL_DATASET_DIR` to the parent folder `~/.stable-wm` rather than `~/.stable-wm/datasets` to avoid path resolution errors.

### NaN Monitoring
Added `DetectAnomalyCallback` to `train.py` to monitor weights, predictions, and gradients at each training step to catch and report NaNs immediately.
