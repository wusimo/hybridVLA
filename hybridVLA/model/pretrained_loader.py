"""
Pretrained weight loading and adaptation for HybridVLA.

DESIGN RATIONALE:
=================
We build on existing open-source models to avoid the enormous cost of
pretraining from scratch. The key insight is that most of the "knowledge"
needed for a VLA already exists in pretrained VLMs:

1. VISUAL UNDERSTANDING: SigLIP / Qwen2.5-VL's ViT already encodes rich
   visual features from billions of image-text pairs. We reuse this.

2. LANGUAGE + REASONING: Qwen2.5's LLM already understands natural language
   instructions, spatial concepts ("left of", "on top of"), and can reason
   about multi-step plans. We reuse this.

3. VISION-LANGUAGE ALIGNMENT: If we load from Qwen2.5-VL, the ViT and LLM
   are already aligned (the connector was trained). This saves Stage 1
   entirely and reduces Stage 2.

What's NEW and needs training from scratch:
- SpatiotemporalMemory (~2M params): working memory for multi-step tasks
- ChainOfPointReasoner (~1M params): spatial coordinate prediction
- ActionChunkHead (~3M params): continuous action trajectory prediction
- TokenMerger MLP (~4M params): 2x2 visual token compression

These new components total ~10M params (<1% of the model), so they converge
quickly during fine-tuning even with modest datasets.

WEIGHT LOADING STRATEGIES:
==========================
Strategy A: "Unified VLM init" (preferred for base/large)
  - Load entire Qwen2.5-VL checkpoint
  - Map its ViT weights -> our QuantizedVisionEncoder
  - Map its LLM weights -> our QuantizedLLMBackbone
  - Randomly initialize new modules (memory, CoP, action head)
  - Advantage: ViT-LLM alignment is preserved

Strategy B: "Separate init" (for small or custom combos)
  - Load SigLIP weights -> our QuantizedVisionEncoder
  - Load Qwen2.5 weights -> our QuantizedLLMBackbone
  - Randomly initialize connector + new modules
  - Requires Stage 1 (visual alignment) training

Strategy C: "From BitVLA" (for maximum quantization quality)
  - Load BitVLA checkpoint (already 1.58-bit LLM + quantized ViT)
  - Add our new modules on top
  - Skip Stages 1-3, go directly to Stage 4 (robotics SFT)
"""

import logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn

from .config import HybridVLAConfig

logger = logging.getLogger(__name__)


def _remap_key(key: str, mapping: dict[str, str]) -> str | None:
    """Apply a key remapping, returning None if the key should be skipped."""
    for src_prefix, dst_prefix in mapping.items():
        if key.startswith(src_prefix):
            return key.replace(src_prefix, dst_prefix, 1)
    return None


def load_siglip_weights(
    model: nn.Module,
    model_id: str = "google/siglip-large-patch16-384",
    device: str = "cpu",
) -> set[str]:
    """
    Load SigLIP vision encoder weights into our QuantizedVisionEncoder.

    Key mapping: SigLIP's ViT uses standard naming that maps to our blocks:
    - vision_model.embeddings.patch_embedding -> patch_embed.proj
    - vision_model.encoder.layers.N.* -> blocks.N.*
    - vision_model.post_layernorm -> norm

    Returns set of loaded parameter names for verification.
    """
    try:
        from transformers import SiglipVisionModel
    except ImportError:
        logger.error("Install transformers: pip install transformers")
        raise

    logger.info(f"Loading SigLIP weights from {model_id}")
    siglip = SiglipVisionModel.from_pretrained(model_id)
    src_state = siglip.state_dict()

    key_mapping = {
        "vision_model.embeddings.patch_embedding.weight": "vision_encoder.patch_embed.proj.weight",
        "vision_model.embeddings.patch_embedding.bias": "vision_encoder.patch_embed.proj.bias",
        "vision_model.post_layernorm.weight": "vision_encoder.norm.weight",
    }

    # Map transformer layers
    for src_key in src_state:
        if "encoder.layers." in src_key:
            # vision_model.encoder.layers.N.{self_attn,mlp,layer_norm*}.*
            # -> vision_encoder.blocks.N.{attn,mlp,norm*}.*
            layer_part = src_key.replace("vision_model.encoder.layers.", "vision_encoder.blocks.")

            # Attention mappings
            layer_part = layer_part.replace(".self_attn.q_proj.", ".attn.qkv.")  # simplified
            layer_part = layer_part.replace(".self_attn.out_proj.", ".attn.proj.")
            layer_part = layer_part.replace(".layer_norm1.", ".norm1.")
            layer_part = layer_part.replace(".layer_norm2.", ".norm2.")
            layer_part = layer_part.replace(".mlp.fc1.", ".mlp.w1.")
            layer_part = layer_part.replace(".mlp.fc2.", ".mlp.w3.")

            key_mapping[src_key] = layer_part

    loaded = set()
    dst_state = model.state_dict()

    for src_key, src_val in src_state.items():
        dst_key = key_mapping.get(src_key)
        if dst_key and dst_key in dst_state:
            if dst_state[dst_key].shape == src_val.shape:
                dst_state[dst_key] = src_val.to(device)
                loaded.add(dst_key)
            else:
                logger.warning(
                    f"Shape mismatch for {dst_key}: "
                    f"src={src_val.shape}, dst={dst_state[dst_key].shape}. "
                    f"Skipping (will use random init)."
                )

    model.load_state_dict(dst_state, strict=False)
    logger.info(f"Loaded {len(loaded)}/{len(src_state)} SigLIP parameters")
    return loaded


def load_qwen25_llm_weights(
    model: nn.Module,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cpu",
) -> set[str]:
    """
    Load Qwen2.5 LLM weights into our QuantizedLLMBackbone.

    Qwen2.5 architecture maps naturally to ours since we designed
    our backbone to mirror it:
    - model.embed_tokens -> llm.tok_emb
    - model.layers.N.self_attn.{q,k,v,o}_proj -> llm.layers.N.attn.{q,k,v,o}_proj
    - model.layers.N.mlp.{gate,up,down}_proj -> llm.layers.N.{w1,w2,w3}
    - model.layers.N.input_layernorm -> llm.layers.N.norm1
    - model.layers.N.post_attention_layernorm -> llm.layers.N.norm2
    - model.norm -> llm.norm
    - lm_head -> llm.lm_head

    Returns set of loaded parameter names.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("Install transformers: pip install transformers")
        raise

    logger.info(f"Loading Qwen2.5 LLM weights from {model_id}")
    qwen = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    src_state = qwen.state_dict()

    # Build key mapping
    key_mapping = {
        "model.embed_tokens.weight": "llm.tok_emb.weight",
        "model.norm.weight": "llm.norm.weight",
        # lm_head is tied to tok_emb in our model, but load explicitly if present
        "lm_head.weight": "llm.lm_head.weight",
    }

    for src_key in src_state:
        if "model.layers." not in src_key:
            continue
        dst_key = src_key.replace("model.layers.", "llm.layers.")
        dst_key = dst_key.replace(".self_attn.", ".attn.")
        dst_key = dst_key.replace(".mlp.gate_proj.", ".w1.")
        dst_key = dst_key.replace(".mlp.up_proj.", ".w2.")
        dst_key = dst_key.replace(".mlp.down_proj.", ".w3.")
        dst_key = dst_key.replace(".input_layernorm.", ".norm1.")
        dst_key = dst_key.replace(".post_attention_layernorm.", ".norm2.")
        key_mapping[src_key] = dst_key

    loaded = set()
    dst_state = model.state_dict()

    for src_key, src_val in src_state.items():
        dst_key = key_mapping.get(src_key)
        if dst_key and dst_key in dst_state:
            if dst_state[dst_key].shape == src_val.shape:
                dst_state[dst_key] = src_val.to(device)
                loaded.add(dst_key)
            else:
                logger.warning(
                    f"Shape mismatch for {dst_key}: "
                    f"src={src_val.shape}, dst={dst_state[dst_key].shape}. Skipping."
                )

    model.load_state_dict(dst_state, strict=False)
    logger.info(f"Loaded {len(loaded)}/{len(src_state)} Qwen2.5 LLM parameters")

    del qwen
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return loaded


def load_qwen25_vl_weights(
    model: nn.Module,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cpu",
) -> set[str]:
    """
    Load from a unified Qwen2.5-VL checkpoint (both ViT + LLM).

    WHY THIS IS PREFERRED: In Qwen2.5-VL, the vision encoder and LLM
    were jointly trained with the connector (MLP merger). This means:
    - The ViT output space is already aligned with the LLM input space
    - We skip Stage 1 (visual alignment) entirely
    - Stage 2 (instruction tuning) can be shorter since the model
      already understands multimodal instructions

    We map both the visual_model.* and model.* weight keys.
    """
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        logger.warning(
            "Qwen2VL not available in transformers. "
            "Falling back to separate ViT + LLM loading."
        )
        return set()

    logger.info(f"Loading Qwen2.5-VL weights from {model_id}")
    qwen_vl = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    src_state = qwen_vl.state_dict()

    loaded = set()
    dst_state = model.state_dict()

    # Map visual encoder keys
    vis_mapping = {}
    for src_key in src_state:
        if src_key.startswith("visual."):
            # Qwen2.5-VL uses: visual.patch_embed.*, visual.blocks.N.*, visual.merger.*
            dst_key = src_key.replace("visual.", "vision_encoder.")
            # Qwen-VL attention naming
            dst_key = dst_key.replace(".attn.qkv.", ".attn.qkv.")
            dst_key = dst_key.replace(".attn.proj.", ".attn.proj.")
            vis_mapping[src_key] = dst_key

    # Map LLM keys (same mapping as standalone Qwen2.5)
    llm_mapping = {}
    for src_key in src_state:
        if src_key.startswith("model."):
            dst_key = src_key.replace("model.embed_tokens.", "llm.tok_emb.")
            dst_key = dst_key.replace("model.norm.", "llm.norm.")
            dst_key = dst_key.replace("model.layers.", "llm.layers.")
            dst_key = dst_key.replace(".self_attn.", ".attn.")
            dst_key = dst_key.replace(".mlp.gate_proj.", ".w1.")
            dst_key = dst_key.replace(".mlp.up_proj.", ".w2.")
            dst_key = dst_key.replace(".mlp.down_proj.", ".w3.")
            dst_key = dst_key.replace(".input_layernorm.", ".norm1.")
            dst_key = dst_key.replace(".post_attention_layernorm.", ".norm2.")
            llm_mapping[src_key] = dst_key
        elif src_key == "lm_head.weight":
            llm_mapping[src_key] = "llm.lm_head.weight"

    all_mapping = {**vis_mapping, **llm_mapping}

    for src_key, src_val in src_state.items():
        dst_key = all_mapping.get(src_key)
        if dst_key and dst_key in dst_state:
            if dst_state[dst_key].shape == src_val.shape:
                dst_state[dst_key] = src_val.to(device)
                loaded.add(dst_key)
            else:
                logger.warning(
                    f"Shape mismatch for {dst_key}: "
                    f"src={src_val.shape}, dst={dst_state[dst_key].shape}. Skipping."
                )

    model.load_state_dict(dst_state, strict=False)
    logger.info(
        f"Loaded {len(loaded)}/{len(src_state)} Qwen2.5-VL parameters "
        f"(ViT: {sum(1 for k in loaded if 'vision' in k)}, "
        f"LLM: {sum(1 for k in loaded if 'llm' in k)})"
    )

    # Report what was NOT loaded (new modules that need training)
    not_loaded = set(dst_state.keys()) - loaded
    new_modules = set()
    for key in not_loaded:
        module_name = key.split(".")[0]
        new_modules.add(module_name)
    if new_modules:
        logger.info(
            f"Randomly initialized modules (need training): {sorted(new_modules)}"
        )

    del qwen_vl
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return loaded


def initialize_from_pretrained(
    model: nn.Module,
    config: HybridVLAConfig,
    device: str = "cpu",
) -> dict[str, set[str]]:
    """
    Top-level function: load pretrained weights into HybridVLA.

    Tries the preferred strategy first, falls back if needed:
    1. If prefer_vlm_init and vlm_model_id set: load unified Qwen2.5-VL
    2. Otherwise: load SigLIP + Qwen2.5 separately

    Returns dict of loaded parameter names per source for tracking.
    """
    result = {}
    pretrained = config.pretrained

    if pretrained.prefer_vlm_init and pretrained.vlm_model_id:
        # Strategy A: Unified VLM init
        loaded = load_qwen25_vl_weights(model, pretrained.vlm_model_id, device)
        if loaded:
            result["qwen_vl"] = loaded
            return result
        logger.warning("Unified VLM loading failed, falling back to separate init")

    # Strategy B: Separate init
    if pretrained.vision_model_id:
        result["siglip"] = load_siglip_weights(
            model, pretrained.vision_model_id, device
        )

    if pretrained.llm_model_id:
        result["qwen_llm"] = load_qwen25_llm_weights(
            model, pretrained.llm_model_id, device
        )

    return result


def load_teacher_for_distillation(
    model_id: str = "google/siglip-large-patch16-384",
) -> nn.Module:
    """
    Load a full-precision teacher vision encoder for Stage 3
    distillation-aware QAT.

    The teacher provides representation alignment targets so the
    quantized student retains visual quality despite 1.58-bit weights.
    The teacher is frozen during training (no gradients).
    """
    try:
        from transformers import SiglipVisionModel
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    logger.info(f"Loading teacher vision model from {model_id}")
    teacher = SiglipVisionModel.from_pretrained(model_id)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


# ============================================================================
# v2 functions
# ============================================================================

def load_depth_model(
    model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
    device: str = "cpu",
) -> nn.Module | None:
    """
    Load a pretrained monocular depth model for the v2 depth encoder.

    The depth model is frozen — it provides features for distillation
    into the VLM's visual representations.

    Compatible with HuggingFace depth estimation models:
    - depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf
    """
    try:
        from transformers import AutoModel
    except ImportError:
        logger.error("Install transformers: pip install transformers")
        raise

    logger.info(f"Loading depth model from {model_id}")
    try:
        depth_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as e:
        logger.warning(f"Failed to load depth model {model_id}: {e}. "
                       "Depth features will be zeros.")
        return None

    depth_model.eval()
    for param in depth_model.parameters():
        param.requires_grad = False
    depth_model = depth_model.to(device)

    logger.info(f"Loaded depth model: {sum(p.numel() for p in depth_model.parameters()):,} params (frozen)")
    return depth_model


def init_action_expert_from_vlm(
    model: nn.Module,
    config: HybridVLAConfig,
) -> int:
    """
    Initialize Action Expert FFN layers from the VLM backbone's FFN layers.

    Copies the VLM's SwiGLU FFN weights (w1, w2, w3) into the Action Expert's
    FFN layers. This warm-start is much better than random initialization —
    verified by both LingBot-VLA and π0.

    The Action Expert may have fewer layers than the VLM. In this case, we
    sample evenly from the VLM layers.
    """
    if not hasattr(model, "action_expert") or model.action_expert is None:
        return 0
    if not hasattr(model, "llm"):
        return 0

    ae_layers = model.action_expert.layers
    vlm_layers = model.llm.layers
    num_ae = len(ae_layers)
    num_vlm = len(vlm_layers)

    if num_vlm == 0:
        return 0

    copied = 0
    for ae_idx in range(num_ae):
        vlm_idx = int(ae_idx * num_vlm / num_ae)
        vlm_layer = vlm_layers[vlm_idx]
        ae_ffn = ae_layers[ae_idx].ffn

        try:
            if ae_ffn.w1.weight.shape == vlm_layer.w1.weight.shape:
                ae_ffn.w1.weight.data.copy_(vlm_layer.w1.weight.data)
                ae_ffn.w2.weight.data.copy_(vlm_layer.w2.weight.data)
                ae_ffn.w3.weight.data.copy_(vlm_layer.w3.weight.data)
                copied += 1
            else:
                logger.debug(
                    f"Shape mismatch AE layer {ae_idx} vs VLM layer {vlm_idx}: "
                    f"AE={ae_ffn.w1.weight.shape}, VLM={vlm_layer.w1.weight.shape}. "
                    f"Keeping random init."
                )
        except Exception as e:
            logger.debug(f"Could not copy VLM layer {vlm_idx} to AE layer {ae_idx}: {e}")

    logger.info(f"Initialized {copied}/{num_ae} Action Expert FFN layers from VLM backbone")
    return copied


def initialize_from_pretrained_v2(
    model: nn.Module,
    config: HybridVLAConfig,
    device: str = "cpu",
) -> dict[str, int]:
    """
    Top-level function: load all pretrained weights for HybridVLA v2.

    Steps:
    1. Load VLM weights (same as v1)
    2. Warm-start Action Expert from VLM FFN (if enabled)
    3. Load depth model backbone (if enabled)

    Returns dict of loaded parameter counts per source.
    """
    result = {}

    # 1. Load VLM weights (v1 path)
    vlm_loaded = initialize_from_pretrained(model, config, device)
    for source, keys in vlm_loaded.items():
        result[source] = len(keys)

    # 2. Warm-start Action Expert from VLM
    if config.use_action_expert and config.action_expert_init_from_vlm:
        copied = init_action_expert_from_vlm(model, config)
        result["action_expert_from_vlm"] = copied

    # 3. Load depth model
    if config.use_depth and hasattr(model, "depth_encoder") and model.depth_encoder is not None:
        depth_backbone = load_depth_model(config.depth_model_id, device)
        if depth_backbone is not None:
            model.depth_encoder.set_backbone(depth_backbone)
            depth_params = sum(p.numel() for p in depth_backbone.parameters())
            result["depth_model"] = depth_params

    return result
