# HybridVLA v2: Design Document

## 1. Motivation & Goals

HybridVLA v1 combined BitVLA (1.58-bit quantization), RynnBrain (spatiotemporal
memory, Chain-of-Point), and Qwen2.5-VL into a single unified model. This was a
solid foundation, but recent breakthroughs — particularly **LingBot-VLA** (Jan 2026)
and **π0.5/π0.6** (Physical Intelligence, 2025) — have established new best
practices for VLA design that we should adopt.

### What v1 gets right (keep):
- Qwen2.5-VL backbone with M-RoPE, window attention, SwiGLU
- Spatiotemporal memory for multi-step task continuity
- Chain-of-Point (CoP) spatial reasoning
- BitNet 1.58-bit quantization option for edge deployment
- LoRA-based parameter-efficient fine-tuning
- No pretraining from scratch — build on open-source weights

### What v1 is missing (gaps to fill):
| Gap | Why it matters | Source of solution |
|-----|---------------|-------------------|
| Simple MLP action head with L1 loss | Can't model multimodal action distributions (e.g., go left OR right around obstacle) | LingBot-VLA / π0: **Flow Matching** |
| Action head is a thin add-on | No dedicated capacity for action reasoning, relies entirely on VLM hidden states | LingBot-VLA / π0: **Action Expert (MoT)** |
| Single camera view only | Most real robots have 2-3 cameras (wrist, left, right) | LingBot-VLA: **Multi-view encoding** |
| No depth perception | Manipulation requires depth reasoning for grasping | LingBot-VLA: **Depth distillation** |
| VLM backbone degrades during robot training | Action loss gradients corrupt language/vision knowledge | π0.6: **Knowledge Insulation** |
| Short action horizon (10 steps) | Modern VLAs predict 50-step chunks for smoother trajectories | LingBot-VLA / π0: **Horizon 50** |
| No proprioceptive state input | Robot joint state is a critical input for action prediction | LingBot-VLA / π0: **Proprioception encoding** |

### Design principles:
1. **No pretraining from scratch** — load pretrained VLM + pretrained depth model
2. **Modular** — each new capability is a separable module that can be toggled
3. **Backward compatible** — v1 configs still work (action expert disabled = v1 behavior)
4. **Training data exists** — every feature must have a corresponding data source

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            HybridVLA v2                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  OBSERVATION BRANCH (VLM Backbone — Qwen2.5-VL, frozen or LoRA)       │    │
│  │                                                                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐ │    │
│  │  │ Camera 1 │  │ Camera 2 │  │ Camera 3 │  │ Depth   │  │ Language │ │    │
│  │  │ (left)   │  │ (right)  │  │ (wrist)  │  │ Tokens  │  │ Instruct │ │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘  └─────┬────┘ │    │
│  │       │ViT+Merge    │ViT+Merge    │ViT+Merge    │proj         │embed  │    │
│  │       └──────────────┼─────────────┼─────────────┘             │       │    │
│  │                      ▼                                         │       │    │
│  │              [vis_1 | vis_2 | vis_3 | depth | text]            │       │    │
│  │                      │                                                 │    │
│  │                ┌─────▼──────┐                                          │    │
│  │                │Spatiotempo-│                                          │    │
│  │                │ral Memory  │◄── memory_state (from previous step)     │    │
│  │                └─────┬──────┘                                          │    │
│  │                      ▼                                                 │    │
│  │              observation_tokens  (bidirectional self-attention)         │    │
│  └──────────────────────┬─────────────────────────────────────────────────┘    │
│                         │                                                      │
│           ┌─────────────▼──────────────┐                                       │
│           │   SHARED SELF-ATTENTION    │  ◄── Mixture-of-Transformers (MoT)    │
│           │   (layer-wise joint        │      Obs tokens ↔ Action tokens       │
│           │    sequence modeling)       │      share attention, separate FFNs   │
│           └─────────────┬──────────────┘                                       │
│                         │                                                      │
│  ┌──────────────────────▼─────────────────────────────────────────────────┐    │
│  │  ACTION EXPERT BRANCH (new transformer, trainable)                     │    │
│  │                                                                         │    │
│  │  ┌────────────┐  ┌────────────────┐  ┌──────────────────────────────┐  │    │
│  │  │ Proprio-   │  │ Past Action    │  │ Noised Action Chunk          │  │    │
│  │  │ ception    │  │ Chunk (t-1)    │  │ (flow matching input)        │  │    │
│  │  │ Encoder    │  │ Embedding      │  │ a_t = (1-σ)·ε + σ·a_gt     │  │    │
│  │  └─────┬──────┘  └──────┬─────────┘  └──────────────┬───────────────┘  │    │
│  │        └────────────────┼────────────────────────────┘                  │    │
│  │                         ▼                                               │    │
│  │                  action_tokens  (causal w.r.t. self, full w.r.t. obs)  │    │
│  │                         │                                               │    │
│  │                   ┌─────▼──────┐                                        │    │
│  │                   │  Flow      │                                        │    │
│  │                   │  Matching  │  predict velocity v(a_t, σ, obs)       │    │
│  │                   │  Head      │  loss = ||v - (a_gt - ε)||²            │    │
│  │                   └─────┬──────┘                                        │    │
│  │                         ▼                                               │    │
│  │                   predicted_actions [B, 50, action_dim]                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  AUXILIARY MODULES                                                      │    │
│  │  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────────┐  │    │
│  │  │ Chain-of-Point   │  │ Depth Distillation│  │ BitNet 1.58-bit    │  │    │
│  │  │ Reasoner         │  │ (from DepthAnythg)│  │ Quantization (opt) │  │    │
│  │  └──────────────────┘  └───────────────────┘  └─────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Architectural Changes (v1 → v2)

### 3.1 Mixture-of-Transformers (MoT) with Action Expert

**What**: Replace the thin `ActionChunkHead` (cross-attention + MLP, ~3M params)
with a full **Action Expert** transformer branch (~200-800M params depending on
scale) that shares self-attention layers with the VLM backbone.

**Why**: The current action head has no dedicated reasoning capacity — it simply
projects the VLM's hidden states to actions. This works for simple tasks but fails
on precise manipulation where the action distribution is multimodal or requires
multi-step kinematic reasoning. LingBot-VLA and π0/π0.5/π0.6 all independently
converged on this "action expert" design.

**How** (following LingBot-VLA's MoT architecture):

```
For each transformer layer l:
    # Observation branch (VLM backbone)
    obs_tokens = VLM_FFN_l(SharedAttention_l(obs_tokens, action_tokens))

    # Action branch (Action Expert)
    action_tokens = ActionExpert_FFN_l(SharedAttention_l(action_tokens, obs_tokens))
```

- **Shared**: Self-attention weights (Q, K, V projections) are shared between
  both branches. This enables cross-modal information flow at every layer.
- **Separate**: Feed-forward networks (FFN) are modality-specific. The VLM FFN
  preserves language/vision knowledge. The Action Expert FFN learns action-specific
  transformations.
- **Attention mask**: Blockwise causal — observation tokens attend bidirectionally
  to each other; action tokens attend to all observation tokens but only to past
  action tokens (prevents future action leakage).

**Parameter budget**:
| Scale | VLM Backbone | Action Expert FFN | Total New Params |
|-------|-------------|-------------------|-----------------|
| Base (3B VLM) | 3B (frozen/LoRA) | ~200M | ~200M |
| Large (7B VLM) | 7B (frozen/LoRA) | ~500M | ~500M |

**Initialization**: Action Expert FFN layers are initialized from the VLM's FFN
layers (warm start) then diverge during training. This is much better than random
init — verified by both LingBot-VLA and π0.

**Backward compatibility**: Setting `use_action_expert=False` in config falls back
to the v1 `ActionChunkHead` (cross-attention + MLP). This is useful for quick
experiments and ablation.

### 3.2 Flow Matching Action Objective

**What**: Replace L1 loss on actions with **Conditional Flow Matching** — a
continuous normalizing flow that transports Gaussian noise to the ground-truth
action distribution.

**Why**: L1 loss assumes a unimodal action distribution. If two equally valid
grasping strategies exist (approach from left vs. right), L1 loss averages them
(approach from center — which may collide with the object). Flow matching models
the full distribution and can sample diverse, valid trajectories.

**Formulation**:

```
Training:
    σ ~ Uniform(0, 1)                           # noise level
    ε ~ N(0, I)                                  # Gaussian noise
    a_σ = (1 - σ) · ε + σ · a_gt                # interpolated (noised) action
    v_pred = model(a_σ, σ, observation)          # predicted velocity field
    loss = ||v_pred - (a_gt - ε)||²              # flow matching loss

Inference (10 denoising steps):
    a_0 ~ N(0, I)                                # start from noise
    for i in 0..9:
        σ = i / 10
        v = model(a_σ, σ, observation)           # predict velocity
        a_{σ+Δ} = a_σ + v · Δσ                  # Euler step
    return a_1.0                                  # final action
```

**Key details**:
- Linear probability path (not cosine or sigmoid schedule) — simplest and works
  well per LingBot-VLA and π0
- 10 denoising steps at inference (same as π0) — good quality/speed tradeoff
- The noise level σ is injected into the action expert via sinusoidal embedding
  added to each action token, similar to diffusion timestep embedding
- Only the Action Expert branch runs the 10 denoising steps; the VLM backbone
  runs once and its KV cache is reused (critical for efficiency)

**Performance**: LingBot-VLA's flow matching achieves 17.3% SR on GM-100 vs 13%
for π0.5 and 7.6% for GR00T N1.6. The smooth continuous actions are especially
important for high-precision bimanual manipulation.

### 3.3 Multi-View Camera Support

**What**: Process 1-4 camera views through the same ViT encoder and concatenate
their tokens in the observation sequence.

**Why**: Most real robot setups use multiple cameras (e.g., left shoulder, right
shoulder, wrist). Single-view models lose critical depth and occlusion information.
LingBot-VLA uses 3 views; π0.6 uses up to 4.

**How**:
```python
# Each view encoded independently through the same ViT
view_tokens = []
for i, image in enumerate(camera_images):       # 1-4 views
    tokens = vit_encode(image)                   # [B, N_vis, D]
    tokens += view_position_embedding[i]         # learned per-view offset
    view_tokens.append(tokens)
obs_visual = concat(view_tokens, dim=1)          # [B, num_views * N_vis, D]
```

- Same ViT weights shared across views (parameter efficient)
- Per-view learnable position offset (4 embeddings, ~4*D params) distinguishes views
- View count is configurable at both train and inference time
- Single-view (v1 behavior) works by setting `num_views=1`

### 3.4 Depth Perception via Distillation

**What**: Integrate metric depth understanding by distilling from a pretrained
monocular depth model (Depth Anything V2 or LingBot-Depth).

**Why**: RGB alone makes depth estimation ambiguous. For grasping, the robot needs
to know how far away objects are. LingBot-VLA showed that adding depth tokens
improved their SR from ~15% to 17.3%.

**How** (two options, can use either):

**Option A: Depth Token Injection (lightweight, recommended)**
```
1. Run pretrained depth model (frozen) on each camera view
2. Get depth feature tokens from the depth model's encoder
3. Project depth tokens to VLM dimension via a small MLP
4. Add depth tokens to the observation sequence after visual tokens
5. Train projection MLP + distillation alignment loss
```

**Option B: Depth Image as Extra Channel (simpler)**
```
1. Run pretrained depth model on each camera view → depth map
2. Concatenate depth map as 4th channel (RGBD) to the image
3. Modify ViT patch embedding input channels: 3 → 4
4. Initialize the new channel weights from the mean of RGB weights
```

We recommend **Option A** for maximum quality (it's what LingBot-VLA uses) with
**Option B** as a simpler fallback.

**Pretrained depth models** (open source, no retraining needed):
- Depth Anything V2 (small/base/large) — best general-purpose monocular depth
- LingBot-Depth — specifically designed for robot manipulation depth
- Metric3D v2 — metric depth with scale awareness

### 3.5 Knowledge Insulation

**What**: During robot training, prevent action expert gradients from flowing back
into the VLM backbone. The VLM only receives gradients from its own objectives
(language modeling, FAST token prediction).

**Why**: π0.6 demonstrated that without insulation, the VLM backbone's language
and vision capabilities degrade during robotics fine-tuning. With insulation, the
VLM maintains its generalization while the action expert specializes.

**How**:
```python
# In the shared attention layer:
obs_hidden = shared_attention(obs_tokens, action_tokens)
action_hidden = shared_attention(action_tokens, obs_tokens)

# Knowledge insulation: stop gradient from action branch to VLM
obs_for_action = obs_hidden.detach()  # action expert sees VLM output but
                                       # doesn't push gradients back

# VLM FFN only gets VLM gradients
obs_out = vlm_ffn(obs_hidden)

# Action FFN gets action gradients only
action_out = action_ffn(action_hidden)
```

This is simple to implement (one `.detach()` call) but has a large impact on
maintaining VLM quality during multi-task training.

### 3.6 Proprioceptive State Encoding

**What**: Encode the robot's current joint state (positions, velocities, gripper
state) as input tokens to the action expert.

**Why**: Actions are relative to current state. Without proprioception, the model
must infer the robot's state purely from visual observation, which is noisy and
lossy. Both LingBot-VLA and π0 include proprioception.

**How**:
```python
class ProprioceptionEncoder(nn.Module):
    def __init__(self, proprio_dim, hidden_dim):
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, proprio_state):
        # proprio_state: [B, proprio_dim] (joint angles, velocities, gripper)
        return self.mlp(proprio_state).unsqueeze(1)  # [B, 1, D]
```

- `proprio_dim` is configurable per robot embodiment (e.g., 14 for 7-DoF bimanual)
- The encoded proprioception token is prepended to the action sequence
- A past-action embedding (previous chunk) is also included, following LingBot-VLA

### 3.7 Extended Action Horizon

**What**: Increase default `action_chunk_size` from 10 to 50.

**Why**: Longer horizons produce smoother trajectories and reduce the control
frequency requirement. LingBot-VLA uses horizon=50; π0 uses 50Hz chunks.
Short horizons cause jerky motion at chunk boundaries.

**Config change**:
```python
action_chunk_size: int = 50    # was 10 in v1
action_horizon_train: int = 50 # training horizon
action_horizon_infer: int = 50 # inference horizon (can differ)
```

Note: The actual executed horizon can be shorter than predicted (execute first K
of 50 predicted steps, then replan). This "receding horizon" approach combines
the smoothness of long predictions with the reactivity of frequent replanning.

---

## 4. Updated Model Configurations

### 4.1 v2 Config Variants

| Config | VLM Backbone | Action Expert | Depth | Total Params | Quantized |
|--------|-------------|--------------|-------|-------------|-----------|
| v2-small | Qwen2.5-VL-3B | 12 layers, ~200M | Depth Anything V2-S | ~3.4B | ~0.9 GB |
| v2-base | Qwen2.5-VL-3B | 24 layers, ~400M | Depth Anything V2-B | ~3.6B | ~1.0 GB |
| v2-large | Qwen2.5-VL-7B | 28 layers, ~800M | Depth Anything V2-L | ~8.2B | ~2.2 GB |
| v2-max | Qwen3-VL-8B | 28 layers, ~860M | LingBot-Depth | ~9.3B | ~2.5 GB |

### 4.2 New Config Fields

```python
@dataclass
class HybridVLAv2Config(HybridVLAConfig):
    """Extended configuration for HybridVLA v2."""

    # === Action Expert (MoT) ===
    use_action_expert: bool = True          # False = v1 behavior
    action_expert_depth: int = 24           # number of transformer layers
    action_expert_dim: int = 1024           # hidden dim (can differ from VLM)
    action_expert_num_heads: int = 16
    action_expert_init_from_vlm: bool = True  # warm-start from VLM FFN

    # === Flow Matching ===
    use_flow_matching: bool = True          # False = v1 L1 loss
    flow_matching_steps: int = 10           # denoising steps at inference
    noise_schedule: str = "linear"          # "linear" | "cosine"

    # === Multi-View ===
    num_views: int = 3                      # 1 = single-view (v1 compat)
    view_names: list[str] = field(
        default_factory=lambda: ["left_shoulder", "right_shoulder", "wrist"]
    )

    # === Depth ===
    use_depth: bool = True
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Base-hf"
    depth_integration: str = "token"        # "token" | "channel"

    # === Knowledge Insulation ===
    knowledge_insulation: bool = True

    # === Proprioception ===
    proprio_dim: int = 14                   # 7-DoF x 2 arms (bimanual)
    use_past_actions: bool = True           # feed previous action chunk

    # === Action horizon ===
    action_chunk_size: int = 50             # was 10
    action_execute_horizon: int = 10        # execute first K, then replan
```

---

## 5. Training Pipeline

### 5.1 Overview: 3-Phase Post-Training

We keep the principle of NO pretraining from scratch. The training pipeline is
redesigned around the new MoT architecture:

```
Phase 0: Weight Loading (no training)
    ├── VLM backbone ← Qwen2.5-VL-3B/7B (pretrained)
    ├── Action Expert FFN ← copied from VLM FFN layers (warm start)
    ├── Depth encoder ← Depth Anything V2 (pretrained, frozen)
    ├── Memory, CoP ← random init (small, ~3M params)
    └── Proprio encoder, view embeddings ← random init (~1M params)

Phase 1: Multimodal Alignment + Action Expert Warmup  (~1-3 days, 8xA100)
    ├── Train: Action Expert FFN, proprio encoder, view embeddings,
    │          depth projector, memory, CoP
    ├── Freeze: VLM backbone (or LoRA rank 16), depth encoder
    ├── Data: Mixed robotics demonstration data (multi-view, with proprio)
    ├── Loss: Flow matching on actions + CoP point loss
    └── Purpose: Align new modules to VLM representations

Phase 2: Full Fine-Tuning with Knowledge Insulation  (~3-7 days, 8xA100)
    ├── Train: Everything (VLM via LoRA rank 64, Action Expert full)
    ├── Freeze: Depth encoder
    ├── Gradient: Knowledge insulation enabled (action grad ↛ VLM)
    ├── Data: Large-scale robot data + language co-training data
    ├── Loss: Flow matching + CoP + language modeling (on co-training data)
    └── Purpose: Maximize action quality while preserving VLM capabilities

Phase 3: Optional Quantization (QAT)  (~2-5 days, 8xA100)
    ├── Apply 1.58-bit QAT to VLM backbone + Action Expert
    ├── Distillation from full-precision teacher
    └── Purpose: Edge deployment
```

### 5.2 Quick-Start Path (Minimum Viable Training)

For users who want to get started quickly with limited compute:

```bash
# Skip Phase 1 (use pretrained alignment from Qwen2.5-VL)
# Skip Phase 3 (no quantization needed for GPU inference)
# Only do Phase 2 with LoRA:

python -m hybrid_vla.training.train_v2 \
    --config v2-base \
    --phase 2 \
    --train-data data/your_robot_data.jsonl \
    --lora-rank 64 \
    --epochs 20 \
    --lr 1e-4 \
    --output-dir checkpoints/hybridvla_v2
```

Minimum data requirement: ~100 hours of teleoperation trajectories with
multi-view images + proprioceptive state. If single-view only, set `num_views=1`.

---

## 6. Training Data

### 6.1 Data Format

Each training sample is one timestep in a trajectory:

```json
{
    "episode_id": "ep_00042",
    "timestep": 15,
    "task_instruction": "Pick up the red block and place it on the blue plate",

    "images": {
        "left_shoulder": "path/to/left_shoulder_0015.jpg",
        "right_shoulder": "path/to/right_shoulder_0015.jpg",
        "wrist": "path/to/wrist_0015.jpg"
    },

    "depth_images": {
        "left_shoulder": "path/to/left_shoulder_depth_0015.png",
        "right_shoulder": "path/to/right_shoulder_depth_0015.png",
        "wrist": "path/to/wrist_depth_0015.png"
    },

    "proprioception": [0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 1.0,
                        0.2, -0.1, 0.3, 0.5, -0.2, 0.1, 0.0],

    "action_chunk": [
        [0.01, -0.02, 0.03, 0.00, 0.01, -0.01, 1.0],
        [0.02, -0.03, 0.04, 0.00, 0.01, -0.01, 1.0],
        ...
    ],

    "past_action_chunk": [
        ...
    ],

    "cop_points": [
        {"text": "grasp red block", "point": [0.3, 0.5, 0.1], "confidence": 0.95},
        {"text": "move to plate", "point": [0.6, 0.4, 0.3], "confidence": 0.90}
    ],

    "robot_type": "bimanual_franka",
    "control_frequency_hz": 50
}
```

### 6.2 Data Sources

| Source | Views | Depth | Proprio | Hours | Robot Type |
|--------|-------|-------|---------|-------|-----------|
| **DROID** (Toyota) | 2-3 | some | yes | 1,000+ | Franka Panda |
| **Open X-Embodiment** | varies | rare | varies | 1,000+ | 22+ robots |
| **BridgeData V2** | 1-2 | no | yes | 100+ | WidowX |
| **LIBERO** (sim) | 1 | yes | yes | 50+ | Franka (sim) |
| **RoboSet** (MIT) | 3 | no | yes | 100+ | Franka |
| **ALOHA** (Stanford) | 4 | no | yes | 50+ | Bimanual ViperX |
| **Custom teleoperation** | 1-4 | optional | yes | user-provided | any |

### 6.3 Data Conversion Pipeline

We need converters for each source format. The `prepare_data.py` module will be
extended with:

```python
# New converters for v2 data format
convert_droid()          # DROID dataset → v2 JSONL
convert_oxe()            # Open X-Embodiment → v2 JSONL
convert_aloha()          # ALOHA/ACT format → v2 JSONL
convert_bridgedata()     # BridgeData V2 → v2 JSONL
convert_roboset()        # RoboSet → v2 JSONL
convert_custom()         # Generic HDF5/zarr → v2 JSONL

# Depth estimation for datasets without depth
estimate_depth()         # Run Depth Anything V2 on RGB images → depth maps

# CoP annotation for datasets without spatial grounding
annotate_cop()           # Use VLM (Qwen2.5-VL) to generate CoP annotations
```

### 6.4 Language Co-Training Data

To maintain VLM capabilities during Phase 2 (knowledge insulation helps, but
co-training is additional insurance):

| Source | Purpose | Size |
|--------|---------|------|
| LLaVA-Instruct-665K | Visual instruction following | 665K samples |
| ShareGPT-4V | Detailed image descriptions | 100K samples |
| RefCOCO/g/+ | Referring expression grounding | 300K samples |
| Robotics QA (generated) | Robot-specific VQA | ~50K samples |

The language co-training data uses the standard VLM loss (cross-entropy on text
tokens) and only flows through the VLM backbone (not the action expert).

### 6.5 Data Mixing Strategy

During Phase 2, each training batch is mixed:
```
70% robot demonstration data    (flow matching loss → action expert)
20% language co-training data   (LM loss → VLM backbone only)
10% CoP/grounding data          (point loss → CoP module + VLM)
```

---

## 7. File Structure Changes

```
hybrid_vla/
├── DESIGN.md                          # v1 design (kept for reference)
├── DESIGN_V2.md                       # THIS FILE
├── model/
│   ├── config.py                      # EXTEND with v2 config fields
│   ├── quantization.py                # unchanged
│   ├── mrope.py                       # unchanged
│   ├── vision_encoder.py              # MODIFY: add multi-view support
│   ├── language_backbone.py           # MODIFY: add MoT shared attention
│   ├── spatiotemporal_memory.py       # unchanged (memory + CoP)
│   ├── action_head.py                 # KEEP for v1 compat
│   ├── action_expert.py               # NEW: Action Expert transformer
│   ├── flow_matching.py               # NEW: Flow matching scheduler + loss
│   ├── depth_encoder.py               # NEW: Depth model wrapper + projector
│   ├── proprioception.py              # NEW: Proprioception encoder
│   ├── hybrid_vla.py                  # MODIFY: integrate v2 components
│   ├── hybrid_vla_v2.py               # NEW: v2 model (extends HybridVLA)
│   └── pretrained_loader.py           # MODIFY: add depth model loading
├── data/
│   ├── prepare_data.py                # EXTEND: new dataset converters
│   ├── dataset_v2.py                  # NEW: v2 dataloader (multi-view, depth)
│   └── data_mixing.py                 # NEW: multi-source data mixer
└── training/
    ├── train.py                       # KEEP for v1
    └── train_v2.py                    # NEW: v2 training pipeline (3 phases)
```

---

## 8. Inference Pipeline

### 8.1 Single-Step Inference

```python
# Initialize
model = HybridVLAv2.from_pretrained("checkpoints/hybridvla_v2")
memory_state = None

# Control loop at ~10Hz (replan every 10 steps of 50-step chunk)
while not done:
    images = robot.get_camera_images()          # dict of 1-4 views
    proprio = robot.get_proprioception()         # joint state vector
    instruction = "pick up the red block"

    # Model predicts 50-step action chunk (10 denoising steps internally)
    actions, memory_state = model.predict_action(
        images=images,
        instruction=instruction,
        proprio=proprio,
        memory_state=memory_state,
    )

    # Execute first 10 steps, then replan
    for t in range(10):
        robot.execute(actions[t])
```

### 8.2 Inference Latency Budget

The MoT architecture is designed so the VLM backbone runs ONCE per replan cycle,
and only the action expert runs the 10 flow matching denoising steps:

```
VLM backbone (once):     ~30ms  (3B, FP16)  /  ~15ms (1.58-bit)
Action expert (x10):     ~5ms × 10 = ~50ms  (200M, FP16) / ~25ms (1.58-bit)
Depth model (once):      ~10ms  (frozen, small)
Memory update:           ~2ms
────────────────────────────────────────────────────
Total per replan:        ~92ms (FP16)  /  ~52ms (1.58-bit)
Control frequency:       ~10-20 Hz ✓
```

---

## 9. Migration Path from v1

### For existing v1 users:
1. v1 configs continue to work — `use_action_expert=False` by default for v1
2. v1-trained checkpoints can be loaded into v2 model (extra modules ignored)
3. Incremental upgrade: enable features one at a time
   - Step 1: Add multi-view (`num_views=3`)
   - Step 2: Add flow matching (`use_flow_matching=True`, keep v1 action head)
   - Step 3: Add action expert (`use_action_expert=True`)
   - Step 4: Add depth (`use_depth=True`)

### For new users:
Start with the `v2-base` config and Phase 2 training (skip Phase 1 if using
Qwen2.5-VL initialization).

---

## 10. Comparison with State of the Art

| Feature | HybridVLA v1 | **HybridVLA v2** | LingBot-VLA | π0.5/π0.6 | GR00T N1 |
|---------|-------------|-----------------|-------------|-----------|----------|
| VLM backbone | Qwen2.5-VL | Qwen2.5-VL | Qwen2.5-VL | PaliGemma/Gemma3 | custom VLM |
| Action head | MLP + cross-attn | **MoT Action Expert** | MoT Action Expert | Action Expert | DiT |
| Action loss | L1 | **Flow Matching** | Flow Matching | Flow Matching | DDPM |
| Multi-view | No | **Yes (1-4)** | Yes (3) | Yes (up to 4) | Yes |
| Depth | No | **Yes (distill)** | Yes (LingBot-Depth) | No | No |
| Memory | Yes (RynnBrain) | **Yes (RynnBrain)** | No | No | No |
| CoP reasoning | Yes (RynnBrain) | **Yes (RynnBrain)** | No | No | No |
| Knowledge insulation | No | **Yes (π0.6)** | Unknown | Yes | N/A |
| Proprioception | No | **Yes** | Yes | Yes | Yes |
| 1.58-bit quant | Yes (BitVLA) | **Yes (BitVLA)** | No | No | No |
| Action horizon | 10 | **50** | 50 | 50 | 16 |
| Edge deployment | Yes (<2GB) | **Yes (<1-2.5GB)** | No (~16GB+) | No (~8GB+) | No (~8GB+) |

**Key differentiators of HybridVLA v2**:
1. Only VLA with both MoT action expert AND spatiotemporal memory + CoP
2. Only VLA with 1.58-bit quantization option for edge deployment
3. Combines the best of LingBot-VLA (MoT + flow matching + depth) with RynnBrain
   (memory + CoP) and BitVLA (quantization) — no other model has all three

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MoT shared attention is hard to train | Medium | High | Warm-start from VLM FFN; Phase 1 warmup |
| Flow matching adds inference latency | Low | Medium | Only 10 steps; KV cache reuse; can quantize action expert |
| Multi-view increases memory usage | Medium | Medium | Token merging; can use fewer views at inference |
| Depth model adds latency | Low | Low | Small model (Depth Anything V2-S); can cache across replan |
| Training data insufficient for new features | Medium | High | Depth estimation pipeline fills depth gaps; CoP annotation fills grounding gaps |
| v1 backward compat breaks | Low | Medium | All new features are off by default; extensive config validation |

---

## 12. References

- **LingBot-VLA**: Robbyant/Ant Group, "A Pragmatic VLA Foundation Model",
  arXiv:2601.18692, Jan 2026.
  [GitHub](https://github.com/Robbyant/lingbot-vla)
- **π0**: Hejna et al., "A Vision-Language-Action Flow Model for General Robot
  Control", arXiv:2410.24164, Oct 2024.
- **π0.5**: Physical Intelligence, "A Vision-Language-Action Model with Open-World
  Generalization", 2025. [Blog](https://www.physicalintelligence.company/blog/pi05)
- **π0.6**: Physical Intelligence, Nov 2025.
  [Model Card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)
- **OpenPI**: [GitHub](https://github.com/Physical-Intelligence/openpi)
- **GR00T N1**: NVIDIA, "GR00T N1: An Open Foundation Model for Generalist
  Humanoid Robots", 2025.
- **BitVLA**: Wang et al., "BitVLA: 1-bit Vision-Language-Action Models",
  arXiv:2506.07530, June 2025.
- **RynnBrain**: Alibaba DAMO Academy, Feb 2026.
- **RynnVLA-001**: arXiv:2509.15212, Sep 2025.
- **RynnVLA-002**: arXiv:2511.17502, Nov 2025.
- **Depth Anything V2**: Yang et al., 2024.
- **DROID**: Khazatsky et al., "DROID: A Large-Scale In-the-Wild Robot
  Manipulation Dataset", 2024.
