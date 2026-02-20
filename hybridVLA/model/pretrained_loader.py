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
