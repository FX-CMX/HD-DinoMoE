# HD-DinoMoE

**Hierarchical Dual-stream MoE for Multi-label Ocular Surface Segmentation**

A class-aware hierarchical dual-stream mixture-of-experts architecture built on DINOv3 vision foundation models for composite multi-label segmentation of the ocular surface.

---

## Architecture Overview

HD-DinoMoE introduces four key innovations:

1. **CA-DSGF Encoder** — Class-Aware Dual-Stream Gated Fusion encoder with SAT and LVD backbones
2. **CS-MED Decoder** — Class-Specific Multi-Expert Decoder with 4 heterogeneous experts (DPT, SAM-MLP, D2S, LinearAttn)
3. **PCP Loss** — Progressive Confidence Penalty Loss for glare region suppression
4. **CA-ASW** — Class-Aware Adaptive Sample Weighting with balanced strategy

The model employs a 3-stage training strategy:
- **Stage 1**: Train SAT branch (freeze LVD)
- **Stage 2**: Train LVD branch (freeze SAT)
- **Stage 3**: Joint fine-tuning with HD-MoE decoder

---

## Project Structure

```
HD-DinoMoE/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── dinov3/                       # DINOv3 model repository (see Setup)
├── pretrained/                   # Pre-trained weights (see Setup)
│   ├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── datasets/                     # Datasets (see Setup)
│   └── datasets_new/
│       └── 515/
│           ├── positive/         # Positive samples (train/test)
│           └── negative/         # Negative/glare samples
├── modules/                      # Core model modules
│   ├── __init__.py
│   ├── hd_moe_model.py          # HD-MoE main model
│   ├── decoder_moe.py           # MoE decoder (DecoderGate, MultiDecoderMoE)
│   ├── decoders.py              # 8 base decoder architectures
│   ├── losses.py                # Loss functions
│   ├── metrics.py               # Dice, IoU, HD95 metrics
│   └── visualize.py             # Visualization utilities
├── tools/                        # Training & automation scripts
│   ├── train_hd_moe_staged_v2.py    # Main training script
│   ├── train_hd_moe_joint.py        # Joint training script
│   ├── model.py                     # Single-backbone model builder
│   ├── model_multienc_v2.py         # Dual-backbone model builder
│   ├── dataset.py                   # Multi-label dataset loader
│   ├── dataset_glare_v2.py          # Glare-aware dataset loader
│   ├── sample_weighting.py          # Sample weighting strategies
│   ├── class_aware_weighting.py     # Class-aware sample weighting
│   ├── experiment_config.py         # Experiment configuration
│   ├── experiment_queue.py          # Experiment queue manager
│   ├── experiment_runner.py         # Experiment runner (tmux-based)
│   ├── experiment_launcher.py       # TUI experiment launcher
│   └── experiment_monitor.py        # TUI experiment monitor
└── runs/                         # Training outputs (auto-created)
```

---

## Setup

### 1. Create Conda Environment

```bash
conda create -n HD-DinoMoE python=3.10 -y
conda activate HD-DinoMoE
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare DINOv3 Repository

Clone the DINOv3 repository into the project root:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

> The `dinov3/` directory is used by `torch.hub.load()` to build the ViT-L/16 backbone.

### 5. Download Pre-trained Weights

Download the DINOv3 ViT-L/16 pre-trained weights and place them in the `pretrained/` directory:

| Weight File | Description |
|---|---|
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | SAT-493M pre-trained |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | LVD-1689M pre-trained |

```bash
cd pretrained/
# Download weights to pretrained/ directory
```

### 6. Prepare Dataset

Download the dataset and place it in the `datasets/` directory. The expected structure is:

```
datasets/
└── datasets_new/
    └── 515/
        ├── positive/
        │   ├── train/
        │   │   ├── image/    # Training images (.jpg/.png)
        │   │   └── mask/     # Multi-label masks (.npy, shape: H×W×C)
        │   └── test/
        │       ├── image/
        │       └── mask/
        └── negative/         # Glare/negative samples (same structure)
```

> **Note:** The dataset is not publicly available. Please contact the authors via email to request access.

---

## Usage

### Quick Start: Single Training Run

Navigate to the `tools/` directory and run:

```bash
cd tools/

# Single-backbone (SAT) training, 50 epochs
python train_hd_moe_staged_v2.py \
    --data_dir ../datasets/datasets_new/515/positive \
    --num_classes 3 \
    --dino_ckpt_sat ../pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
    --dino_size l \
    --repo_dir ../dinov3 \
    --epochs 50 \
    --batch_size 1 \
    --lr 1e-4 \
    --input_h 1024 --input_w 1024 \
    --decoder_mode single \
    --single_decoder_type dpt \
    --loss_type bce_dice \
    --save_dir ../runs
```

### Full 3-Stage Dual-Backbone Training

```bash
python train_hd_moe_staged_v2.py \
    --data_dir ../datasets/datasets_new/515/positive \
    --num_classes 3 \
    --dino_ckpt_sat ../pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
    --dino_ckpt_lvd ../pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --dino_size l \
    --repo_dir ../dinov3 \
    --epochs_stage1 30 --epochs_stage2 30 --epochs_stage3 50 \
    --batch_size 1 \
    --lr 1e-4 \
    --decoder_mode multi_moe \
    --separate_projects \
    --loss_type glare_progressive \
    --glare_data_dir ../datasets/datasets_new/515/negative \
    --glare_penalty 2.0 --glare_gamma 1.0 \
    --sample_weighting class_aware \
    --focus_mode balanced \
    --sample_temp 1.0 \
    --gate_entropy_lambda 0.5 \
    --mixed_precision fp16 \
    --save_dir ../runs
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--decoder_mode` | `shared_moe` | `single` / `shared_moe` / `multi_moe` |
| `--separate_projects` | `False` | Use per-class projection layers (multi_moe only) |
| `--loss_type` | `bce_dice` | `bce` / `bce_dice` / `focal_dice` / `glare_progressive` |
| `--sample_weighting` | `none` | `none` / `loss_based` / `focal` / `curriculum` / `class_aware` |
| `--focus_mode` | `hard` | `hard` / `easy` / `balanced` |
| `--stage` | `0` | Run specific stage (1/2/3), 0 = all stages |
| `--mixed_precision` | `no` | `no` / `fp16` / `bf16` |

---

## Automated Experiment System

HD-DinoMoE includes a TUI-based experiment automation system for batch experiment management.

### Launch Experiment Configurator

```bash
cd tools/
python experiment_launcher.py
```

This opens an interactive terminal UI where you can:
- Configure backbone mode (SAT / LVD / Dual)
- Select decoder architecture
- Toggle glare suppression and sample weighting
- Add experiments to queue with GPU assignment

### Monitor Running Experiments

```bash
cd tools/
python experiment_monitor.py
```

The monitor provides real-time status of all queued, running, and completed experiments with progress tracking.

### How It Works

1. **`experiment_launcher.py`** — Interactive TUI for configuring and queuing experiments
2. **`experiment_queue.py`** — Manages the experiment queue (`experiments_queue.json`)
3. **`experiment_runner.py`** — Executes experiments via tmux sessions with GPU isolation
4. **`experiment_config.py`** — Defines all configuration enums and parameter mappings

> **Prerequisite:** The automation system requires `tmux` to be installed (`sudo apt install tmux`).

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{hddinomoe2026,
  title={HD-DinoMoE: Hierarchical Dual-stream Mixture-of-Experts for Multi-label Ocular Surface Segmentation},
  author={...},
  journal={...},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
