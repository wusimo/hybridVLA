"""
Configuration for HybridVLA model variants.

DESIGN PRINCIPLE: We do NOT pretrain from scratch. Instead, we load weights
from existing open-source models and apply post-training (SFT, QAT, LoRA):

  - Vision encoder: Initialize from SigLIP (google/siglip-large-patch16-384)
    or Qwen2.5-VL's ViT. These are already strong visual encoders.
  - LLM backbone: Initialize from Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct.
    These already understand language and instructions.
  - New components (memory, CoP, action head): Randomly initialized but small
    (~5% of total params), so they train quickly during SFT.

The quantization to 1.58-bit happens AFTER loading pretrained weights:
  Stage 1: Load full-precision pretrained weights
  Stage 2: SFT on multimodal data (still full precision)
  Stage 3: Apply distillation-aware QAT to compress to 1.58-bit
  Stage 4: Robotics SFT on the quantized model

Provides pre-defined configurations at different scales:
- HybridVLA-Small  (~1.5B -> ~400MB quantized): Uses SigLIP-B + Qwen2.5-0.5B
- HybridVLA-Base   (~3B -> ~800MB quantized): Uses SigLIP-L + Qwen2.5-1.5B
- HybridVLA-Large  (~8B -> ~2GB quantized): Uses SigLIP-SO + Qwen2.5-7B
"""

from dataclasses import dataclass, field


@dataclass
class PretrainedSources:
    """
    References to pretrained model weights for initialization.

    WHY: Training a ViT or LLM from scratch requires 100s of GPU-days and
    trillions of tokens. By starting from pretrained checkpoints, we only
    need to:
    1. Train the small connector/merger MLP to align representations
    2. SFT the LLM on robotics instruction data
    3. Train the new modules (memory, CoP, action head) which are small
    """
    # Vision encoder source (HuggingFace model ID)
    vision_model_id: str = "google/siglip-large-patch16-384"
    # LLM backbone source (HuggingFace model ID)
    llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Optional: full VLM source to initialize both ViT + LLM together
    vlm_model_id: str | None = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Teacher model for distillation-aware QAT (Stage 3)
    teacher_vision_model_id: str = "google/siglip-large-patch16-384"

    # Whether to prefer loading from the unified VLM checkpoint
    # (gets both ViT and LLM in one shot with already-aligned representations)
    prefer_vlm_init: bool = True


@dataclass
class HybridVLAConfig:
    """Complete configuration for the HybridVLA model."""

    # === Model identity ===
    model_name: str = "hybrid-vla-base"

    # === Pretrained model sources ===
    pretrained: PretrainedSources = field(default_factory=PretrainedSources)

    # === Vision encoder (quantized ViT with window attention) ===
    img_size: int = 384  # Match SigLIP default; supports dynamic resolution
    patch_size: int = 14
    in_channels: int = 3
    vis_embed_dim: int = 1024
    vis_depth: int = 24
    vis_num_heads: int = 16
    vis_mlp_ratio: float = 4.0
    vis_window_size: int = 8  # 8x8 window attention (Qwen2.5-VL)
    vis_global_layers: list[int] | None = None  # auto-set if None
    vis_quantize: bool = False  # starts full-precision; QAT applied in Stage 3

    # === Language backbone ===
    vocab_size: int = 151936  # Qwen2.5 tokenizer
    llm_dim: int = 1536  # Qwen2.5-1.5B hidden dim
    llm_depth: int = 28  # Qwen2.5-1.5B depth
    llm_num_heads: int = 12  # Qwen2.5-1.5B heads
    llm_num_kv_heads: int = 2  # Qwen2.5-1.5B GQA
    llm_mlp_ratio: float = 4.0
    llm_max_seq_len: int = 4096
    llm_quantize: bool = False  # starts full-precision; QAT applied in Stage 3
    deepstack_layers: list[int] | None = None  # auto-set if None

    # === Action head (v1 legacy) ===
    action_dim: int = 7  # 6-DoF + gripper
    action_chunk_size: int = 10
    action_num_heads: int = 8

    # === Spatiotemporal memory (RynnBrain) ===
    memory_num_slots: int = 64
    memory_num_heads: int = 8
    use_memory: bool = True

    # === Chain-of-Point reasoning (RynnBrain) ===
    cop_num_points: int = 8
    use_chain_of_point: bool = True

    # === Training ===
    distillation_gamma: float = 0.1  # weight for distillation alignment loss

    # === LoRA config (for parameter-efficient fine-tuning) ===
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
    )

    # =====================================================================
    # v2 EXTENSIONS (LingBot-VLA / π0 / GR00T N1 inspired)
    # All default to False/disabled so v1 behavior is preserved.
    # =====================================================================

    # === Action Expert (Mixture-of-Transformers) ===
    use_action_expert: bool = False  # True enables MoT; False = v1 action head
    action_expert_depth: int = 24    # number of Action Expert FFN layers
    action_expert_dim: int | None = None  # None = same as llm_dim
    action_expert_num_heads: int = 16
    action_expert_num_kv_heads: int = 4
    action_expert_mlp_ratio: float = 4.0
    action_expert_init_from_vlm: bool = True  # warm-start FFN from VLM layers

    # === Flow Matching ===
    use_flow_matching: bool = False  # True = flow matching loss; False = v1 L1 loss
    flow_matching_steps: int = 10    # denoising steps at inference
    flow_noise_schedule: str = "linear"  # "linear" | "cosine"

    # === Multi-View ===
    num_views: int = 1  # 1 = single-view (v1 compat); 2-4 for multi-view
    view_names: list[str] = field(
        default_factory=lambda: ["primary"]
    )

    # === Depth Perception ===
    use_depth: bool = False
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Base-hf"
    depth_integration: str = "token"  # "token" (project depth features) | "channel" (RGBD)
    depth_token_dim: int = 384  # hidden dim of the depth model features

    # === Knowledge Insulation (π0.6) ===
    knowledge_insulation: bool = False  # stop action expert grad from flowing to VLM

    # === Proprioception ===
    proprio_dim: int = 14   # joint state dim (e.g. 7-DoF x 2 for bimanual)
    use_proprio: bool = False
    use_past_actions: bool = False  # feed previous action chunk to action expert

    # === Action horizon (v2 default is longer) ===
    action_execute_horizon: int = 10  # execute first K steps then replan

    # === Derived ===
    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_visual_tokens(self) -> int:
        """Number of visual tokens after 2x2 merging."""
        g = self.grid_size
        return (g // 2) * (g // 2)

    @property
    def effective_action_expert_dim(self) -> int:
        """Action expert hidden dim, defaulting to llm_dim if not set."""
        return self.action_expert_dim if self.action_expert_dim is not None else self.llm_dim

    def validate(self):
        """Validate configuration consistency."""
        assert self.vis_embed_dim % self.vis_num_heads == 0, \
            "vis_embed_dim must be divisible by vis_num_heads"
        assert self.llm_dim % self.llm_num_heads == 0, \
            "llm_dim must be divisible by llm_num_heads"
        assert self.llm_num_heads % self.llm_num_kv_heads == 0, \
            "llm_num_heads must be divisible by llm_num_kv_heads"
        head_dim = self.vis_embed_dim // self.vis_num_heads
        assert head_dim % 6 == 0, \
            f"ViT head_dim ({head_dim}) must be divisible by 6 for M-RoPE"
        llm_head_dim = self.llm_dim // self.llm_num_heads
        assert llm_head_dim % 6 == 0, \
            f"LLM head_dim ({llm_head_dim}) must be divisible by 6 for M-RoPE"
        # v2 validations
        if self.use_action_expert:
            ae_dim = self.effective_action_expert_dim
            assert ae_dim % self.action_expert_num_heads == 0, \
                "action_expert_dim must be divisible by action_expert_num_heads"
            assert self.action_expert_num_heads % self.action_expert_num_kv_heads == 0, \
                "action_expert_num_heads must be divisible by action_expert_num_kv_heads"
        assert 1 <= self.num_views <= 4, "num_views must be between 1 and 4"
        assert self.action_execute_horizon <= self.action_chunk_size, \
            "action_execute_horizon must be <= action_chunk_size"


def hybrid_vla_small() -> HybridVLAConfig:
    """
    ~1.5B parameter config for rapid prototyping.

    Uses SigLIP-B (86M ViT) + Qwen2.5-0.5B (494M LLM).
    After 1.58-bit QAT: ~400MB memory footprint.
    """
    return HybridVLAConfig(
        model_name="hybrid-vla-small",
        pretrained=PretrainedSources(
            vision_model_id="google/siglip-base-patch16-224",
            llm_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            vlm_model_id=None,  # no small Qwen-VL, load separately
            teacher_vision_model_id="google/siglip-base-patch16-224",
        ),
        img_size=224,
        vis_embed_dim=768,
        vis_depth=12,
        vis_num_heads=12,
        llm_dim=896,  # Qwen2.5-0.5B
        llm_depth=24,  # Qwen2.5-0.5B
        llm_num_heads=14,  # Qwen2.5-0.5B -- but 14 not div by 6 for mrope
        llm_num_kv_heads=2,
        memory_num_slots=32,
        action_chunk_size=5,
    )


def hybrid_vla_base() -> HybridVLAConfig:
    """
    ~3B parameter config. Primary target for this prototype.

    Uses Qwen2.5-VL-3B-Instruct as unified source (already has aligned
    ViT + LLM). This is ideal because:
    1. ViT and LLM are already aligned (no Stage 1 needed)
    2. Already trained on multimodal instruction data (less Stage 2 needed)
    3. Only need Stage 3 (QAT) + Stage 4 (robotics SFT)

    After 1.58-bit QAT: ~800MB memory footprint.

    WHY Qwen2.5-VL-3B: Best open-source VLM at this scale, already supports
    dynamic resolution and video, and its architecture (M-RoPE, SwiGLU,
    window attention) is what we build on top of.
    """
    return HybridVLAConfig(
        model_name="hybrid-vla-base",
        pretrained=PretrainedSources(
            vision_model_id="google/siglip-large-patch16-384",
            llm_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            teacher_vision_model_id="google/siglip-large-patch16-384",
            prefer_vlm_init=True,
        ),
        img_size=384,
        vis_embed_dim=1024,
        vis_depth=24,
        vis_num_heads=16,
        llm_dim=2048,  # Qwen2.5-VL-3B
        llm_depth=36,  # Qwen2.5-VL-3B
        llm_num_heads=16,
        llm_num_kv_heads=2,
        memory_num_slots=64,
        action_chunk_size=10,
    )


def hybrid_vla_large() -> HybridVLAConfig:
    """
    ~8B parameter config for maximum performance.

    Uses Qwen2.5-VL-7B-Instruct. After 1.58-bit QAT: ~2GB memory.

    WHY this scale: 7B is the sweet spot where VLMs show strong reasoning.
    Larger than 7B gives diminishing returns for robotics while
    significantly increasing memory. 7B quantized to 1.58-bit fits
    comfortably on a single consumer GPU (RTX 3060+).
    """
    return HybridVLAConfig(
        model_name="hybrid-vla-large",
        pretrained=PretrainedSources(
            vision_model_id="google/siglip-so400m-patch14-384",
            llm_model_id="Qwen/Qwen2.5-7B-Instruct",
            vlm_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            teacher_vision_model_id="google/siglip-so400m-patch14-384",
            prefer_vlm_init=True,
        ),
        img_size=384,
        vis_embed_dim=1152,  # SigLIP-SO
        vis_depth=27,  # SigLIP-SO
        vis_num_heads=16,
        llm_dim=3584,  # Qwen2.5-7B
        llm_depth=28,  # Qwen2.5-7B
        llm_num_heads=28,
        llm_num_kv_heads=4,
        memory_num_slots=128,
        action_chunk_size=20,
    )


def hybrid_vla_v2_base() -> HybridVLAConfig:
    """
    HybridVLA v2 base (~3.6B total): Qwen2.5-VL-3B + 400M Action Expert.

    Adds MoT Action Expert, flow matching, multi-view, depth, proprioception,
    and knowledge insulation on top of the v1 base config.
    """
    return HybridVLAConfig(
        model_name="hybrid-vla-v2-base",
        pretrained=PretrainedSources(
            vision_model_id="google/siglip-large-patch16-384",
            llm_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            teacher_vision_model_id="google/siglip-large-patch16-384",
            prefer_vlm_init=True,
        ),
        img_size=384,
        vis_embed_dim=1024,
        vis_depth=24,
        vis_num_heads=16,
        llm_dim=2048,
        llm_depth=36,
        llm_num_heads=16,
        llm_num_kv_heads=2,
        memory_num_slots=64,
        # v2 features
        action_dim=14,  # bimanual: 7-DoF x 2
        action_chunk_size=50,
        action_execute_horizon=10,
        use_action_expert=True,
        action_expert_depth=24,
        action_expert_num_heads=16,
        action_expert_num_kv_heads=4,
        use_flow_matching=True,
        flow_matching_steps=10,
        num_views=3,
        view_names=["left_shoulder", "right_shoulder", "wrist"],
        use_depth=True,
        depth_model_id="depth-anything/Depth-Anything-V2-Base-hf",
        depth_token_dim=384,
        knowledge_insulation=True,
        proprio_dim=14,
        use_proprio=True,
        use_past_actions=True,
    )


def hybrid_vla_v2_large() -> HybridVLAConfig:
    """
    HybridVLA v2 large (~8.2B total): Qwen2.5-VL-7B + 800M Action Expert.
    """
    return HybridVLAConfig(
        model_name="hybrid-vla-v2-large",
        pretrained=PretrainedSources(
            vision_model_id="google/siglip-so400m-patch14-384",
            llm_model_id="Qwen/Qwen2.5-7B-Instruct",
            vlm_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            teacher_vision_model_id="google/siglip-so400m-patch14-384",
            prefer_vlm_init=True,
        ),
        img_size=384,
        vis_embed_dim=1152,
        vis_depth=27,
        vis_num_heads=16,
        llm_dim=3584,
        llm_depth=28,
        llm_num_heads=28,
        llm_num_kv_heads=4,
        memory_num_slots=128,
        # v2 features
        action_dim=14,
        action_chunk_size=50,
        action_execute_horizon=10,
        use_action_expert=True,
        action_expert_depth=28,
        action_expert_num_heads=28,
        action_expert_num_kv_heads=4,
        use_flow_matching=True,
        flow_matching_steps=10,
        num_views=3,
        view_names=["left_shoulder", "right_shoulder", "wrist"],
        use_depth=True,
        depth_model_id="depth-anything/Depth-Anything-V2-Large-hf",
        depth_token_dim=384,
        knowledge_insulation=True,
        proprio_dim=14,
        use_proprio=True,
        use_past_actions=True,
    )


CONFIG_REGISTRY = {
    "small": hybrid_vla_small,
    "base": hybrid_vla_base,
    "large": hybrid_vla_large,
    "v2-base": hybrid_vla_v2_base,
    "v2-large": hybrid_vla_v2_large,
}


def get_config(name: str = "base") -> HybridVLAConfig:
    """Get a pre-defined configuration by name."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIG_REGISTRY.keys())}")
    config = CONFIG_REGISTRY[name]()
    config.validate()
    return config
