# HybridVLA: Design Document

## Overview

HybridVLA is a Vision-Language-Action model prototype that combines the strengths
of three recent architectures for robotic manipulation:

| Source | What we take | Why |
|--------|-------------|-----|
| **BitVLA** | 1.58-bit ternary quantization, distillation-aware QAT, action chunking | Extreme memory efficiency (<2GB) enabling edge deployment |
| **RynnBrain / RynnVLA** | Spatiotemporal memory, Chain-of-Point reasoning, hierarchical design | Multi-step task continuity and spatial grounding |
| **Qwen VLM (2.5-VL / 3-VL)** | M-RoPE, dynamic resolution, window attention, SwiGLU, DeepStack | State-of-the-art multimodal architecture components |

**Critical constraint**: We do NOT pretrain from scratch. We load pretrained
open-source weights (Qwen2.5-VL, SigLIP) and apply post-training (SFT, LoRA, QAT).

---

## Relationship to RynnBrain and RynnVLA

### What are RynnBrain and RynnVLA?

RynnBrain and RynnVLA are separate but complementary model families from
Alibaba DAMO Academy that together form a **hierarchical "cerebrum-cerebellum"
architecture** for embodied intelligence:

```
┌─────────────────────────────────────────────────────────────────────┐
│  RynnBrain ("cerebrum" — high-level brain)                         │
│                                                                     │
│  Perception → Spatiotemporal Memory → CoP Reasoning → Task Plan    │
│                                                                     │
│  Built on: Qwen3-VL                                                 │
│  Variants: 2B, 8B, 30B-A3B (MoE)                                   │
│  Specialists: RynnBrain-Plan, RynnBrain-Nav, RynnBrain-CoP         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ high-level plan
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RynnVLA ("cerebellum" — low-level executor)                       │
│                                                                     │
│  Plan → Visual Observations → Motor Actions                        │
│                                                                     │
│  Built on: Chameleon (Meta's AR model), NOT Qwen                   │
│  RynnVLA-001: ActionVAE + human demo pretraining (12M videos)      │
│  RynnVLA-002: Unified VLA + world model, hybrid discrete+continuous│
└─────────────────────────────────────────────────────────────────────┘
```

**RynnBrain** handles perception, understanding, memory, and planning. It takes
egocentric video streams and language instructions and produces high-level plans
with spatial coordinates (via Chain-of-Point). It sets SOTA on 16 embodied AI
leaderboards, outperforming Gemini Robotics ER 1.5 and NVIDIA Cosmos Reason 2.

**RynnVLA** handles low-level motor execution. It translates high-level plans
into continuous robot actions. RynnVLA-001 is notable for its large-scale
pretraining on 12M ego-centric human manipulation videos, which gives it strong
manipulation priors. RynnVLA-002 adds a world model (next-frame prediction)
jointly trained with action prediction.

### How HybridVLA Relates

HybridVLA is **not a re-implementation** of RynnBrain or RynnVLA. It is a
**single unified model** that collapses their two-tier hierarchy into one,
while borrowing their key innovations:

```
                     RynnBrain/RynnVLA                    HybridVLA
                     (2-model hierarchy)               (single model)
                 ┌─────────────────────┐          ┌─────────────────────┐
                 │  RynnBrain          │          │                     │
 From RynnBrain: │  - Spatiotemporal   │ -------> │  Spatiotemporal     │
                 │    Memory           │          │  Memory module      │
                 │  - Chain-of-Point   │ -------> │  CoP module         │
                 │  - Scene understand.│          │  (inherited from    │
                 └────────┬────────────┘          │   Qwen2.5-VL init) │
                          │ plan                  │                     │
                 ┌────────▼────────────┐          │                     │
 From RynnVLA:   │  RynnVLA            │          │                     │
                 │  - Action chunks    │ -------> │  ActionChunkHead    │
                 │  - Continuous ctrl  │          │  (parallel decode)  │
                 └─────────────────────┘          │                     │
                                                  │  + BitVLA quant.    │
                                                  │  + Qwen VLM arch.   │
                                                  └─────────────────────┘
```

**What we take from RynnBrain:**

1. **Spatiotemporal Memory**: RynnBrain's defining innovation — the ability to
   recall where objects were across the full episode history. Standard VLAs are
   "amnesiac" (they only see the current frame). Our `SpatiotemporalMemory`
   module implements this as a fixed-size slot bank with gated cross-attention
   updates. This is a simplified version of RynnBrain's full memory system, but
   captures the essential capability: if the robot's arm occludes an object in
   frame 30, the memory still knows where it was from frame 15.

2. **Chain-of-Point (CoP) Reasoning**: RynnBrain-CoP interleaves textual
   reasoning with physical point predictions in 3D space. Instead of pure
   text reasoning ("move the block to the plate"), CoP grounds each step:
   "grasp at (0.3, 0.5, 0.1) → move to (0.6, 0.4, 0.3) → place at (0.6, 0.4, 0.05)".
   This directly addresses the #1 failure mode in VLAs: spatial localization
   errors. Our `ChainOfPointReasoner` implements this with a point prediction
   MLP + confidence head + point re-embedding for iterative refinement.

**What we take from RynnVLA:**

1. **Action chunk prediction**: RynnVLA-001 uses ActionVAE to compress action
   chunks into latent embeddings. We take the concept of predicting coherent
   multi-step trajectories but use a simpler approach (cross-attention action
   queries + MLP projection, following BitVLA/OpenVLA-OFT) without the VAE.
   This is more straightforward to train and debug.

2. **Continuous control**: RynnVLA-002 introduced hybrid discrete+continuous
   action heads. Our action head outputs continuous values directly (like
   RynnVLA-002's continuous head) with L1 loss, avoiding the discretization
   artifacts of tokenized actions.

**What we do NOT take (and why):**

- **Two-model hierarchy**: RynnBrain + RynnVLA requires running two separate
  models in sequence. This doubles inference latency and memory. For real-time
  10Hz control on edge hardware, a single model is preferable.

- **Chameleon backbone**: RynnVLA uses Meta's Chameleon (an autoregressive
  image generation model) as its backbone. This is optimized for next-frame
  prediction (world modeling) but doesn't have the strong instruction-following
  of Qwen-family models. We use Qwen2.5-VL instead.

- **12M human video pretraining**: RynnVLA-001's strength comes from pretraining
  on 12M ego-centric human manipulation videos. This requires massive compute.
  Since we can't pretrain from scratch, we rely on Qwen2.5-VL's visual
  understanding (trained on billions of image-text pairs) as our foundation.

- **World model**: RynnVLA-002's joint VLA + world model is elegant but adds
  significant complexity and compute cost. We focus on action prediction only.

---

## Why Not Qwen 3.5?

Qwen3.5 was released on February 16, 2026 — just 2 days ago. Here is why we
use Qwen2.5-VL instead:

### 1. Scale mismatch

Qwen3.5's only released model is **Qwen3.5-397B-A17B** — a 397 billion
parameter MoE model with 17B active parameters per forward pass. Even with
MoE sparsity, 17B active params is far too large for our use case:

| Model | Active Params | FP16 Memory | After 1.58-bit QAT |
|-------|--------------|-------------|---------------------|
| Qwen2.5-VL-3B | 3B | ~6 GB | ~0.8 GB |
| Qwen2.5-VL-7B | 7B | ~14 GB | ~2 GB |
| Qwen3.5-397B-A17B | 17B active | ~34 GB active | ~4 GB (theoretical) |

Our target is **edge deployment on consumer GPUs (4-8GB VRAM)**. Even
quantized, 17B active parameters won't fit. We need the 2-7B scale.

### 2. No small Qwen3.5 variants yet

Qwen3.5 currently only has one model (397B-A17B). There are no 2B, 3B, or 7B
variants. The Qwen2.5-VL family provides exactly the scales we need:
- Qwen2.5-VL-3B for the base config
- Qwen2.5-VL-7B for the large config

When smaller Qwen3.5 models are released, the architecture can be updated —
the weight loading code (`pretrained_loader.py`) is model-agnostic.

### 3. Qwen3.5 is natively multimodal but not VL-specialized

Qwen3.5 integrates vision natively via early fusion, which is elegant. However,
this means there's no separate ViT that we can independently quantize. Our
architecture specifically relies on an extractable vision encoder for:
- Stage 3 distillation-aware QAT (teacher ViT vs. student ViT)
- DeepStack multi-level feature injection
- Independent vision encoder freezing during LLM fine-tuning

Qwen2.5-VL and Qwen3-VL both have separable ViT + LLM components that fit
our modular architecture.

### 4. Maturity and ecosystem

Qwen2.5-VL-3B/7B have been available since January 2025, with:
- Extensive community testing and benchmarking
- HuggingFace Transformers integration
- Known compatibility with PEFT/LoRA
- Documented performance baselines on robotics benchmarks

Qwen3.5 was released 2 days ago. Its integration with downstream tools (PEFT,
vLLM, Transformers) may not be fully mature.

### 5. The Qwen3-VL option

If you want more recent Qwen capabilities, **Qwen3-VL** (released September
2025) is a better fit than Qwen3.5:
- Available in 2B, 4B, 8B, 32B dense + 30B-A3B MoE variants
- Has the interleaved M-RoPE improvement
- Has DeepStack multi-layer visual injection
- Still has a separable ViT for our QAT pipeline
- RynnBrain itself is built on Qwen3-VL

To use Qwen3-VL, just update the config:
```python
pretrained=PretrainedSources(
    vlm_model_id="Qwen/Qwen3-VL-2B",  # or 8B, 32B, 30B-A3B
)
```

### Summary: When to upgrade

| If you need... | Use... |
|----------------|--------|
| Smallest deployment (<1GB) | Qwen2.5-VL-3B + 1.58-bit QAT |
| Best quality at 7B scale | Qwen3-VL-8B (recommended upgrade path) |
| Cutting-edge (when available) | Qwen3.5-xB when small variants release |
| Match RynnBrain exactly | Qwen3-VL (same base as RynnBrain) |

---

## Architecture Components

### 1. Quantized Vision Encoder (`vision_encoder.py`)

**What**: A ViT (Vision Transformer) that processes images into visual tokens.

**Why each design choice**:

- **SigLIP initialization**: SigLIP is the best open-source vision encoder for
  visual grounding. It was trained with sigmoid loss (not softmax contrastive)
  on 4B image-text pairs, giving stronger spatial understanding than CLIP.
  BitVLA chose SigLIP for the same reason. When initializing from Qwen2.5-VL,
  we use its built-in ViT instead (already aligned with the LLM).

- **Window attention** (from Qwen2.5-VL): Processing a 384x384 image produces
  729 patches. Full self-attention is O(729^2) = 531K operations per layer.
  Window attention (8x8 = 64 tokens per window) reduces this to O(729 x 64) =
  47K — an 11x speedup. Only 4 layers use global attention for cross-region
  reasoning. This makes high-resolution input practical.

- **M-RoPE 2D position encoding** (from Qwen2-VL): Unlike absolute position
  embeddings (which fix the ViT to one resolution), M-RoPE encodes height and
  width as rotary embeddings. This allows the same ViT to process any
  resolution — a 224x224 test image or a 768x512 training image — without
  retraining. This is the same position encoding used throughout the Qwen VL
  family and inherited by RynnBrain.

- **SwiGLU activation** (from Qwen2.5-VL): SwiGLU = SiLU(xW1) x xW2. Compared
  to GELU, it provides smoother gradients and ~1-2% better performance on VQA
  benchmarks at the same parameter count.

- **2x2 Token Merger** (from Qwen2-VL): Merges adjacent 2x2 patches into a
  single token via a 2-layer MLP, reducing the visual token count by 4x. A
  384x384 image goes from 729 -> 182 tokens. This is critical because LLM
  inference cost scales linearly with context length — 4x fewer visual tokens
  means 4x faster inference for the LLM.

- **1.58-bit quantization** (from BitVLA): Applied in Stage 3 via QAT. Each
  weight is constrained to {-1, 0, +1}. The ViT shrinks from ~800MB to ~100MB
  (8x reduction) with only ~1.5% accuracy loss thanks to distillation from a
  full-precision teacher.

### 2. Quantized Language Backbone (`language_backbone.py`)

**What**: A decoder-only transformer that processes interleaved visual + text tokens.

**Why each design choice**:

- **Qwen2.5 initialization**: Qwen2.5 is among the best open-source LLM
  families at the 1.5B-7B scale. The Instruct variants already understand
  natural language instructions, spatial concepts, and multi-step reasoning.
  Starting from Qwen2.5 means our model immediately understands "pick up the
  red cube and place it on the blue plate" without any language training.

- **Grouped-Query Attention (GQA)**: Qwen2.5 uses GQA with a 4:1 or 8:1 ratio
  (16 query heads, 2-4 KV heads). This reduces KV cache memory by 4-8x during
  inference with minimal quality loss. For a robotics model running at 10Hz
  control frequency, this matters — every ms of inference latency directly
  impacts control performance.

- **DeepStack injection** (from Qwen3-VL): Standard VLMs inject visual tokens
  at a single layer (typically the input). DeepStack injects multi-level ViT
  features at multiple LLM layers (e.g., layers 6, 12, 18, 24). Early LLM
  layers see low-level visual features (edges, textures), while deep layers
  see high-level semantic features (objects, spatial relations). This gives
  the LLM richer visual understanding without increasing the token count.
  Controlled by a learned sigmoid gate so the model can learn how much visual
  injection is helpful at each layer.

- **LoRA for fine-tuning**: We add LoRA adapters (rank 64, ~2% extra params)
  rather than full fine-tuning. This prevents catastrophic forgetting of the
  pretrained language knowledge while efficiently learning robotics-specific
  adaptations. LoRA weights can be merged back for inference (zero overhead)
  or swapped for different tasks.

- **1.58-bit quantization** (from BitVLA): The LLM backbone is the largest
  component (~1.5-7B params). Quantizing from FP16 to 1.58-bit reduces it
  from ~3-14GB to ~0.4-1.7GB. We can either:
  - Apply QAT in Stage 3 (best quality, requires training)
  - Or use the natively 1-bit BitNet b1.58 2B4T as backbone (zero
    quantization cost, but less capable than Qwen2.5)

### 3. Spatiotemporal Memory (`spatiotemporal_memory.py`)

**What**: A fixed-size memory bank that persists across timesteps during
multi-step tasks. Inspired by RynnBrain's spatiotemporal memory capability.

**Why this exists**: Standard VLAs are "memoryless" — they see only the current
frame and instruction. But real manipulation requires temporal context:
- "Did I already grasp the object?" (action history)
- "Where was the bowl before the arm occluded it?" (object persistence)
- "Am I making progress on this multi-step task?" (plan tracking)

RynnBrain demonstrated that spatiotemporal memory enables "global recall" —
if interrupted during task A to perform task B, the robot can accurately
remember the temporal and spatial state of task A and seamlessly resume.
This solved the longstanding "instantaneous amnesia" problem.

Our simplified version uses:

- **Fixed-size slot bank** (64 slots x D dimensions): A compressed "working
  memory" that doesn't grow with episode length. This is crucial for real-time
  inference — we can't store all past frames. RynnBrain's full memory is
  richer, but our slot-based approach captures the core capability with
  minimal parameters.

- **Write path** (cross-attention): Memory slots attend to incoming visual
  features to absorb new information. A gated update mechanism (sigmoid gate)
  controls how much each slot updates, preventing catastrophic overwriting of
  important old information.

- **Read path** (cross-attention): Action tokens and visual tokens attend to
  memory slots to retrieve relevant historical context. This lets the action
  head "remember" where it saw objects in previous frames.

- **~2M parameters**: Tiny relative to the ViT (400M) or LLM (1.5B+). Trains
  quickly during Stage 4. This is one of the newly-initialized modules.

### 4. Chain-of-Point Reasoning (`spatiotemporal_memory.py`)

**What**: Predicts (x, y, z) spatial coordinates interleaved with language
reasoning. Directly inspired by RynnBrain-CoP.

**Why this exists**: Standard VLAs reason in language space ("I should pick up
the red block") but there's a gap between language and physical coordinates.
RynnBrain introduced CoP to bridge this gap — alternating between textual
reasoning and spatial point grounding within egocentric video streams.

CoP forces every reasoning step to have concrete spatial grounding:

```
Reasoning: "I need to grasp the red block"  -> Point: (0.3, 0.5, 0.1)
Reasoning: "Then move it above the plate"   -> Point: (0.6, 0.4, 0.3)
Reasoning: "And place it down"              -> Point: (0.6, 0.4, 0.05)
```

This reduces "physical hallucinations" — cases where the language plan is
correct but the model picks the wrong spatial location. BitVLA's failure
analysis confirmed spatial localization as the #1 failure mode in VLAs.

Our implementation:
- **Point prediction head**: MLP from hidden state -> (x, y, z)
- **Confidence head**: Predicts how reliable each point prediction is
- **Point embedding**: Encodes predicted points back into hidden space so
  subsequent reasoning can reference them
- **~1M parameters**: Another newly-initialized module.

### 5. Action Chunk Head (`action_head.py`)

**What**: Predicts a chunk of future actions (e.g., next 10 timesteps) in a
single forward pass.

**Why chunking over autoregressive**: Autoregressive action generation
(predicting one action token at a time, like OpenVLA) is slow — for 7-DoF
actions at 10 timesteps, you need 70 sequential decoding steps. Both
RynnVLA (via ActionVAE) and BitVLA (via parallel decoding) address this.
We follow the BitVLA/OpenVLA-OFT approach:

- **Bidirectional attention**: Action queries can attend to each other (not
  causal), so position 5 can inform position 3. This produces smoother, more
  coherent trajectories. RynnVLA-002 uses a similar bidirectional continuous
  Action Transformer head.

- **Cross-attention to LLM context**: Action queries attend to the full LLM
  hidden state (visual + text + memory context) to inform predictions.

- **L1 loss**: Actions are continuous (not discrete tokens), so we use L1
  loss instead of cross-entropy. L1 is more robust to outliers than MSE.
  This matches both BitVLA and RynnVLA-001's approach.

- **Action normalization**: Per-dimension standardization (zero mean, unit
  variance) is critical because position (meters), rotation (radians), and
  gripper (binary) span different scales.

### 6. M-RoPE Position Encoding (`mrope.py`)

**What**: Multimodal Rotary Position Embedding that unifies 1D text, 2D image,
and 3D video positions. From the Qwen VL family (Qwen2-VL through Qwen3-VL).

**Why this over separate position encodings**:
- Each RoPE dimension is decomposed into temporal (t), height (h), width (w)
- For text: t = h = w = sequential position (reduces to standard 1D RoPE)
- For images: t = 0, h/w = patch spatial coordinates (2D positioning)
- For video: t = frame index, h/w = spatial (3D positioning)
- **Interleaved** (Qwen3-VL improvement): t/h/w components interleaved across
  frequency bands instead of contiguous blocks. This prevents temporal
  information from being concentrated in high-frequency dimensions.

This means the same attention mechanism natively handles all three modalities
without architectural modifications. RynnBrain inherits this from Qwen3-VL.

### 7. Quantization Utilities (`quantization.py`)

**What**: BitNet-style 1.58-bit ternary quantization from the BitVLA paper.

**How it works**:
1. **Weights**: Scaled by `mean(|w|)`, then rounded to {-1, 0, +1}. Each
   weight needs only 1.58 bits (log2(3)).
2. **Activations**: Per-token INT8 quantization via absmax scaling. 8-bit
   activations preserve enough precision for computation.
3. **STE**: Straight-through estimator approximates gradients through the
   rounding operation (which has zero gradient everywhere).
4. **Training**: Full-precision weights maintained in the optimizer for
   stable gradient descent. Quantization is applied only in the forward pass.

**Memory impact**: A 3B parameter model:
- FP16: 6 GB
- INT4: 1.5 GB
- 1.58-bit: ~0.6 GB (plus ~0.2 GB for FP16 embeddings and connectors)

---

## Training Pipeline

### Key Principle: Build on Existing Knowledge

We do NOT pretrain from scratch. ~95% of parameters come from pretrained
checkpoints. Only ~10M new parameters need training from scratch.

```
┌─────────────────────────────────────────────────────┐
│  Pretrained Qwen2.5-VL-3B-Instruct                  │
│  (Already has: visual understanding, language,       │
│   vision-language alignment, instruction following)  │
└──────────────────────┬──────────────────────────────┘
                       │ Load weights
                       ▼
┌─────────────────────────────────────────────────────┐
│  HybridVLA (our architecture)                        │
│  ┌──────────┐ ┌─────┐ ┌────────┐ ┌───────┐ ┌─────┐ │
│  │ViT       │ │LLM  │ │Memory  │ │CoP    │ │Act  │ │
│  │(loaded)  │ │(load)│ │(random)│ │(rand) │ │Head │ │
│  │          │ │      │ │~2M     │ │~1M    │ │~3M  │ │
│  └──────────┘ └─────┘ └────────┘ └───────┘ └─────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
   Stage 2 (opt)   Stage 3 (opt)    Stage 4
   Instruction     QAT 1.58-bit    Robotics
   SFT + LoRA     (ViT compress)   SFT
   ~1-2 days       ~3-5 days       ~4hrs-3days
```

### Stage Overview

| Stage | What trains | What's frozen | Data needed | Duration | Skip if... |
|-------|------------|---------------|-------------|----------|-----------|
| 1 | Connector MLP | ViT, LLM, all new | 600K captions | ~2hrs | Using Qwen2.5-VL init |
| 2 | LLM (LoRA), memory, CoP | ViT | 1M instructions | ~1-2 days | Qwen2.5-VL + only need actions |
| 3 | ViT (QAT) | LLM, others | 5M images | ~3-5 days | Don't need 1.58-bit |
| 4 | LLM (LoRA), action head, memory, CoP | ViT | Robot demos | ~4hrs-3days | Never skip |

### Recommended Quick-Start Path

```bash
# 1. Prepare LIBERO data
python -m hybrid_vla.data.prepare_data convert-libero \
    --libero-dir /data/libero --output data/libero_train.jsonl

# 2. Compute action statistics
python -m hybrid_vla.data.prepare_data compute-stats \
    --manifest data/libero_train.jsonl --output data/action_stats.json

# 3. Train (Stage 4 only, using Qwen2.5-VL pretrained weights)
python -m hybrid_vla.training.train \
    --config base --stage 4 \
    --train-data data/libero_train.jsonl \
    --action-stats data/action_stats.json \
    --epochs 10 --lr 5e-5 \
    --output-dir checkpoints/hybridvla_libero
```

---

## Comparison Table

How HybridVLA compares to its source architectures:

| Feature | BitVLA | RynnBrain | RynnVLA-001 | RynnVLA-002 | Qwen2.5-VL | **HybridVLA** |
|---------|--------|-----------|-------------|-------------|------------|---------------|
| ViT backbone | SigLIP-L | Qwen3-VL ViT | Chameleon | Chameleon | Custom ViT | SigLIP or Qwen2.5-VL ViT |
| LLM backbone | BitNet 2B4T | Qwen3-VL LLM | Chameleon | Chameleon | Qwen2.5 | Qwen2.5 (+ LoRA) |
| 1.58-bit quantization | Yes | No | No | No | No | **Yes** (optional) |
| Spatiotemporal memory | No | **Yes** | No | No | No | **Yes** |
| Chain-of-Point | No | **Yes** | No | No | No | **Yes** |
| Action chunking | **Yes** | N/A (planning only) | Yes (ActionVAE) | Yes (hybrid) | N/A | **Yes** |
| Window attention ViT | No | Yes (via Qwen3-VL) | No | No | **Yes** | **Yes** |
| M-RoPE | No | Yes (via Qwen3-VL) | No | No | **Yes** | **Yes** |
| DeepStack | No | Yes (via Qwen3-VL) | No | No | No | **Yes** |
| World model | No | No | No | **Yes** | No | No |
| Params (active) | 3B | 3B-30B | 7B | 7B | 3B-72B | **3B-8B** |
| Memory footprint | **1.4 GB** | ~6-60 GB | ~14 GB | ~14 GB | ~6-144 GB | **~0.8-2 GB** |

---

## File Structure

```
hybrid_vla/
├── __init__.py
├── DESIGN.md                          # This file
├── model/
│   ├── __init__.py
│   ├── config.py                      # Model configs with pretrained sources
│   ├── quantization.py                # BitNet 1.58-bit quantization
│   ├── mrope.py                       # Multimodal Rotary Position Embedding
│   ├── vision_encoder.py              # Quantized ViT with window attention
│   ├── language_backbone.py           # Quantized LLM with GQA + DeepStack
│   ├── spatiotemporal_memory.py       # Memory bank + Chain-of-Point
│   ├── action_head.py                 # Action chunk parallel decoding
│   ├── hybrid_vla.py                  # Main model (ties everything together)
│   └── pretrained_loader.py           # Weight loading from HuggingFace models
├── data/
│   ├── __init__.py
│   └── prepare_data.py                # Dataset conversion + CoP annotation
└── training/
    ├── __init__.py
    └── train.py                       # Multi-stage training pipeline
```

## Dependencies

```
torch>=2.1
transformers>=4.40
peft>=0.10           # for LoRA (optional, has manual fallback)
Pillow
torchvision
numpy
```

Optional (for specific data sources):
```
tensorflow-datasets  # for Open X-Embodiment
h5py                 # for LIBERO
```

## References

- **BitVLA**: Wang et al., "BitVLA: 1-bit Vision-Language-Action Models for
  Robotics Manipulation", arXiv:2506.07530 (June 2025)
- **RynnBrain**: Alibaba DAMO Academy, released Feb 2026.
  [GitHub](https://github.com/alibaba-damo-academy/RynnBrain)
- **RynnVLA-001**: "Using Human Demonstrations to Improve Robot Manipulation",
  arXiv:2509.15212 (September 2025)
- **RynnVLA-002**: "A Unified Vision-Language-Action and World Model",
  arXiv:2511.17502 (November 2025)
- **Qwen2-VL**: Wang et al., "Qwen2-VL: Enhancing Vision-Language Model's
  Perception of the World at Any Resolution", arXiv:2409.12191 (2024)
- **Qwen2.5-VL**: Technical Report, arXiv:2502.13923 (2025)
- **Qwen3-VL**: Technical Report, arXiv:2511.21631 (2025)
- **Qwen3.5**: Released Feb 16, 2026 (397B-A17B MoE, natively multimodal)
- **OpenVLA-OFT**: Kim et al., fine-tuned OpenVLA with action chunking
- **BitNet b1.58**: Ma et al., "The Era of 1-bit LLMs" (Microsoft, 2024)
