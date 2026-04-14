# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarVLA is a modular "Lego-like" codebase for adapting Vision-Language Models (VLMs) into Vision-Language-Action (VLA) models for robotic manipulation. The core philosophy is high cohesion / low coupling — each component (model, data, trainer, config, evaluation) can be developed and smoke-tested independently.

## Common Commands

### Environment Setup
```bash
conda create -n starVLA python=3.10 -y && conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

### Code Quality
```bash
make check        # Validate formatting (black + ruff)
make autoformat   # Auto-fix formatting
make clean        # Remove cache files
```

### Smoke Testing (per layer)
```bash
# Test a framework module in isolation
python starVLA/model/framework/QwenGR00T.py

# Test the dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml starvla_cotrain_oxe.yaml

# Launch distributed training
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml
```

## Architecture

### Configuration System
All configs are YAML files under `starVLA/config/` loaded via **OmegaConf**. Top-level keys are `framework`, `datasets`, and `trainer`. DeepSpeed configs (`deepspeed_zero2.yaml`, `deepspeed_zero3.yaml`) are passed separately to `accelerate launch`.

`trainer_utils/config_tracker.py` provides `AccessTrackedConfig` for logging which config keys are actually accessed during a run.

### Model Layer (`starVLA/model/`)

**`framework/`** — Complete VLA model implementations. Each file is a self-contained framework combining a VLM backbone with an action head:
- `QwenGR00T.py`: Qwen2.5-VL + flow-matching diffusion head (System2 reasoning + System1 action)
- `QwenOFT.py`: Qwen2.5-VL + MLP head (parallel continuous actions)
- `QwenFast.py`: Qwen2.5-VL + Fast action tokenizer (autoregressive discrete tokens)
- `QwenPI.py`: Qwen2.5-VL + flow-matching (π₀ style)
- `base_framework.py`: Abstract base — pretrained weight loading, action normalization/denormalization, trainable module discovery

**`modules/`** — Reusable components plugged into frameworks:
- `vlm/` — VLM wrappers (Qwen2.5-VL, InternVL, Florence-2)
- `action_model/` — Action heads: `fast_ActionHeader.py` (discrete), `MLP_ActionHeader.py` (regression), `GR00T_ActionHeader.py` / `flow_matching_head/` (diffusion), `DiTActionHeader.py` (DiT)
- `dino_model/` — DINO encoder for dense spatial tokens
- `projector/` — Cross-modal projection layers
- `memory/` — Temporal memory modules

**`tools.py`** — `FRAMEWORK_REGISTRY` plugin dict; `auto_get_trainable_modules()` for recursive trainable-submodule discovery.

### Data Layer (`starVLA/dataloader/`)

**Contract**: dataloaders return model-agnostic dicts with keys `image` (PIL), `lang` (str), `action` (normalized np array), and optionally `state`. Frameworks are responsible for converting these into model inputs.

- `lerobot_datasets.py` — Robot demonstration data via the LeRobot ecosystem; supports multi-dataset weighted mixing and LeRobot v2.0/v3.0 formats
- `vlm_datasets.py` — VLM pretraining data in LLaVA JSON format (COCO, VQA, grounding)
- `gr00t_lerobot/` — Robot-type configs, embodiment tags, named dataset mixtures, and data transforms

### Training Layer (`starVLA/training/`)

Explicit PyTorch training loops (not HuggingFace Trainer) for hackability, using **Accelerate + DeepSpeed** for distributed runs.

- `train_starvla.py` — Single-task VLA training
- `train_starvla_cotrain.py` — Multitask co-training (VLM pretraining + VLA simultaneously)
- `train_starvlm.py` — VLM-only pretraining
- `trainer_utils/trainer_tools.py` — `build_param_lr_groups()` (per-module LR), `freeze_backbones()` (regex-based freezing), checkpoint save/load

### Deployment (`deployment/`)
WebSocket-based inference server providing a unified policy interface for real-robot and simulation evaluation.

### Examples (`examples/`)
Benchmark-specific evaluation guides and scripts: `LIBERO/`, `SimplerEnv/`, `RoboCasa_tabletop/`, `Robotwin/`, `Franka/` (real robot), `CoTrainVLM/`.

## Adding a New Framework

1. Create `starVLA/model/framework/MyFramework.py` subclassing `base_framework.py`
2. Register it: add to `FRAMEWORK_REGISTRY` in `starVLA/model/tools.py`
3. Add a `if __name__ == "__main__":` smoke test at the bottom of the file
4. Create a corresponding YAML under `starVLA/config/training/`

## Key Conventions

- **Per-module learning rates**: specified in `trainer.learning_rate` as a dict keyed by module name; applied via `build_param_lr_groups()`
- **Module freezing**: regex patterns in `trainer.freeze_modules`; applied via `freeze_backbones()`
- **Action normalization stats** are stored alongside pretrained checkpoints and loaded by `base_framework.py`
- **Video decoding**: prefer `decord` backend; fall back to `torchvision_av` for AV1-encoded data
