# HybridVLA

A Vision-Language-Action model prototype for robotic manipulation that unifies ideas from three recent architectures into a single, edge-deployable model.

| Source | What We Borrow | Why |
|--------|---------------|-----|
| **BitVLA** | 1.58-bit ternary quantization, distillation-aware QAT, action chunking | Edge deployment — full model fits in <2 GB VRAM |
| **RynnBrain / RynnVLA** | Spatiotemporal memory, Chain-of-Point spatial reasoning | Multi-step task continuity, spatial grounding |
| **Qwen 2.5-VL** | M-RoPE, dynamic resolution, window attention, SwiGLU, DeepStack | State-of-the-art multimodal backbone |

**Key principle:** No pretraining from scratch. We load pretrained Qwen2.5-VL / SigLIP weights and apply post-training only (SFT with LoRA, QAT).

For the full architecture rationale, component explanations, relationship to RynnBrain/RynnVLA, and the case for Qwen2.5-VL over Qwen 3.5, see [DESIGN.md](DESIGN.md).

---

## Architecture

```
Image (384x384) --> [Quantized ViT] --> [2x2 Token Merger] --> 182 visual tokens
                     window attn         4x compression            |
                     M-RoPE 2D pos                                 |
                     SwiGLU + RMSNorm                              v
                                                          [Spatiotemporal Memory]
                                                           gated write / read
                                                                   |
                                                                   v
Instruction --> [Tokenizer] --> [vis_tokens + text_tokens] --> [Qwen2.5 LLM]
                                  M-RoPE position IDs           GQA + LoRA
                                                                DeepStack injection
                                                                     |
                                                          +----------+----------+
                                                          v                     v
                                                 [Chain-of-Point]     [Action Chunk Head]
                                                  (x,y,z) coords      10 future actions
                                                  spatial grounding    parallel decoding
```

The model operates in three modes:

| Mode | Output | Use case |
|------|--------|----------|
| **Action** | Chunk of future motor actions | Real-time 10 Hz robot control |
| **Planning** | Chain-of-Point spatial coordinates | Task planning with spatial grounding |
| **VLM** | Language logits | Image captioning, VQA, instruction understanding |

---

## Quick Start

### Installation

```bash
# Clone and install in editable mode
git clone <repo-url> && cd hybrid-vla
pip install -e .

# With LoRA support (recommended)
pip install -e ".[lora]"

# With data conversion tools
pip install -e ".[data]"
```

### Smoke Tests

All 8 tests run on CPU with no pretrained weights required.

```bash
python -m hybrid_vla.tests.test_smoke

# Or with pytest
pytest hybrid_vla/tests/test_smoke.py -v
```

### Basic Usage

```python
import torch
from hybrid_vla.model.config import get_config
from hybrid_vla.model.hybrid_vla import HybridVLA

# Create a model (no pretrained weights — random init for demo)
config = get_config("base")          # ~3B params
config.vis_quantize = False          # skip quantization for now
config.llm_quantize = False
model = HybridVLA(config)

# Single-step inference
pixel_values = torch.randn(1, 3, 384, 384)   # camera image
input_ids = torch.randint(0, 151936, (1, 12)) # tokenized instruction

actions, memory_state = model.predict_action(pixel_values, input_ids)
# actions: [1, 10, 7] — 10 future timesteps, 7-DoF (6-DoF + gripper)
```

### Multi-Step Inference with Memory

```python
memory_state = None
for t, frame in enumerate(camera_stream):
    pixel_values = preprocess(frame)
    actions, memory_state = model.predict_action(
        pixel_values, input_ids,
        memory_state=memory_state,
        timestep=t,
    )
    execute(actions[0, 0])  # execute first action of the chunk
```

The spatiotemporal memory persists across timesteps, allowing the model to recall object locations even when occluded by the robot's arm.

---

## Training

### Data Preparation

```bash
# 1. Convert LIBERO dataset to training manifest
python -m hybrid_vla.data.prepare_data convert-libero \
    --libero-dir /data/libero --output data/libero_train.jsonl

# 2. Compute per-dimension action normalization statistics
python -m hybrid_vla.data.prepare_data compute-stats \
    --manifest data/libero_train.jsonl --output data/action_stats.json
```

### Training (Recommended: Stage 4 Only)

Starting from Qwen2.5-VL pretrained weights means Stages 1-3 can often be skipped.

```bash
python -m hybrid_vla.training.train \
    --config base --stage 4 \
    --train-data data/libero_train.jsonl \
    --action-stats data/action_stats.json \
    --epochs 10 --lr 5e-5 \
    --output-dir checkpoints/hybridvla_libero
```

### Training Stages

| Stage | What Trains | What's Frozen | Skip If... | Duration |
|-------|------------|--------------|------------|----------|
| 1: Visual Alignment | Connector MLP (~4M params) | ViT, LLM, all new modules | Using Qwen2.5-VL init (already aligned) | ~2 hrs |
| 2: Instruction SFT | LLM via LoRA + memory + CoP | ViT | Qwen2.5-VL already has instruction following | ~1-2 days |
| 3: QAT | ViT via distillation-aware quantization | LLM, others | Don't need 1.58-bit compression | ~3-5 days |
| **4: Robotics SFT** | Action head + memory + CoP + LLM LoRA | ViT | **Never skip** | ~4 hrs - 3 days |

---

## Model Configurations

Three pre-defined scales targeting different deployment scenarios:

| Config | Pretrained Source | Total Params | After 1.58-bit QAT | Target Hardware |
|--------|-----------------|-------------|---------------------|-----------------|
| `small` | SigLIP-B + Qwen2.5-0.5B | ~1.5B | ~400 MB | Embedded / Jetson |
| `base` | Qwen2.5-VL-3B-Instruct | ~3B | ~800 MB | Consumer GPU (4 GB) |
| `large` | Qwen2.5-VL-7B-Instruct | ~8B | ~2 GB | Consumer GPU (8 GB) |

```python
from hybrid_vla.model.config import get_config

config = get_config("base")   # or "small", "large"
```

To use Qwen3-VL as the backbone instead (recommended upgrade path):

```python
from hybrid_vla.model.config import HybridVLAConfig, PretrainedSources

config = get_config("base")
config.pretrained.vlm_model_id = "Qwen/Qwen3-VL-2B"  # or 8B, 32B
```

---

## Key Components

| Module | File | Params | Origin |
|--------|------|--------|--------|
| Quantized Vision Encoder | `model/vision_encoder.py` | ~400M | SigLIP / Qwen2.5-VL ViT with window attention, M-RoPE, SwiGLU |
| Quantized Language Backbone | `model/language_backbone.py` | ~1.5-7B | Qwen2.5 with GQA, DeepStack injection, LoRA adapters |
| Spatiotemporal Memory | `model/spatiotemporal_memory.py` | ~2M | RynnBrain-inspired fixed-size slot bank with gated cross-attention |
| Chain-of-Point Reasoner | `model/spatiotemporal_memory.py` | ~1M | RynnBrain-CoP interleaved spatial coordinate prediction |
| Action Chunk Head | `model/action_head.py` | ~3M | BitVLA/OpenVLA-OFT parallel decoding with bidirectional attention |
| M-RoPE | `model/mrope.py` | 0 | Qwen2-VL unified 1D/2D/3D rotary position embedding |
| BitNet 1.58-bit Quantization | `model/quantization.py` | 0 | BitVLA ternary weight quantization with STE |
| Pretrained Weight Loader | `model/pretrained_loader.py` | 0 | Loads SigLIP / Qwen2.5-VL from HuggingFace |
| Data Pipeline | `data/prepare_data.py` | — | LIBERO and Open X-Embodiment conversion + CoP annotation |
| Training Pipeline | `training/train.py` | — | Multi-stage training with stage-specific freezing schedules |

---

## Project Structure

```
hybrid-vla/
├── setup.py
├── requirements.txt
├── DESIGN.md                         # Full architecture rationale
├── README.md                         # This file
└── hybrid_vla/
    ├── __init__.py
    ├── model/
    │   ├── config.py                 # Model configs with pretrained sources
    │   ├── quantization.py           # BitNet 1.58-bit quantization
    │   ├── mrope.py                  # Multimodal Rotary Position Embedding
    │   ├── vision_encoder.py         # ViT with window attention + M-RoPE
    │   ├── language_backbone.py      # LLM with GQA + DeepStack
    │   ├── spatiotemporal_memory.py  # Memory bank + Chain-of-Point
    │   ├── action_head.py            # Action chunk parallel decoding
    │   ├── hybrid_vla.py             # Main model (ties everything together)
    │   └── pretrained_loader.py      # Weight loading from HuggingFace
    ├── data/
    │   └── prepare_data.py           # Dataset conversion (OXE, LIBERO)
    ├── training/
    │   └── train.py                  # Multi-stage training pipeline
    └── tests/
        └── test_smoke.py             # 8 smoke tests, all CPU
```

---

## Requirements

**Core:**
```
torch>=2.1
transformers>=4.40
numpy
Pillow
torchvision
```

**Optional:**
```
peft>=0.10              # LoRA (has manual fallback if missing)
tensorflow-datasets     # Open X-Embodiment data
h5py                    # LIBERO data
```

Python 3.10+ is required.

---

## Comparison with Source Architectures

| Feature | BitVLA | RynnBrain | RynnVLA | Qwen2.5-VL | **HybridVLA** |
|---------|--------|-----------|---------|------------|---------------|
| 1.58-bit quantization | Yes | — | — | — | **Yes** (optional) |
| Spatiotemporal memory | — | **Yes** | — | — | **Yes** |
| Chain-of-Point reasoning | — | **Yes** | — | — | **Yes** |
| Action chunking | **Yes** | — | Yes | — | **Yes** |
| Window attention ViT | — | Yes | — | **Yes** | **Yes** |
| M-RoPE | — | Yes | — | **Yes** | **Yes** |
| DeepStack injection | — | Yes | — | — | **Yes** |
| World model | — | — | Yes (v2) | — | — |
| Active params | 3B | 3-30B | 7B | 3-72B | **3-8B** |
| Memory footprint | **1.4 GB** | 6-60 GB | ~14 GB | 6-144 GB | **0.8-2 GB** |

---

## References

- **BitVLA** — Wang et al., "BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation" ([arXiv:2506.07530](https://arxiv.org/abs/2506.07530))
- **RynnBrain** — Alibaba DAMO Academy ([GitHub](https://github.com/alibaba-damo-academy/RynnBrain))
- **RynnVLA-001** — "Using Human Demonstrations to Improve Robot Manipulation" ([arXiv:2509.15212](https://arxiv.org/abs/2509.15212))
- **RynnVLA-002** — "A Unified Vision-Language-Action and World Model" ([arXiv:2511.17502](https://arxiv.org/abs/2511.17502))
- **Qwen2.5-VL** — Technical Report ([arXiv:2502.13923](https://arxiv.org/abs/2502.13923))
- **BitNet b1.58** — Ma et al., "The Era of 1-bit LLMs" (Microsoft, 2024)

---

## License

TBD
