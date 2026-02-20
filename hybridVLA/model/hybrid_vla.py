"""
HybridVLA: Unified Vision-Language-Action Model.

Combines the three architectures into a single end-to-end model:

From BitVLA:
  - 1.58-bit ternary quantization on all linear layers (ViT + LLM)
  - Distillation-aware QAT pipeline for the vision encoder
  - Action chunking with parallel decoding via bidirectional attention
  - Extreme memory efficiency (<2GB for the base model)

From RynnBrain:
  - Spatiotemporal memory for multi-step task continuity
  - Chain-of-Point (CoP) spatial reasoning interleaved with language
  - Hierarchical planning: high-level reasoning feeds action execution

From Qwen VLM:
  - M-RoPE for unified text/image/video position encoding
  - Dynamic resolution ViT (no fixed image size)
  - Window attention with sparse global layers in the vision encoder
  - SwiGLU activation and RMSNorm throughout
  - DeepStack multi-layer visual feature injection into the LLM
  - 2x2 token merging for efficient visual token compression

Pipeline:
  1. Vision encoder processes image(s) -> visual tokens
  2. (Optional) DeepStack extracts multi-level features for injection
  3. Visual tokens merged 2x2 and projected to LLM dimension
  4. Spatiotemporal memory updated with new visual observation
  5. Text tokens + visual tokens + memory context concatenated
  6. LLM backbone processes the sequence (with DeepStack injections)
  7. Chain-of-Point predicts spatial coordinates at reasoning steps
  8. Action head produces a chunk of future actions via parallel decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HybridVLAConfig
from .vision_encoder import QuantizedVisionEncoder
from .language_backbone import QuantizedLLMBackbone
from .action_head import ActionChunkHead
from .spatiotemporal_memory import SpatiotemporalMemory, ChainOfPointReasoner
from .mrope import MultimodalRoPE, build_multimodal_position_ids
from .quantization import DistillationLoss, compute_model_size_bits


class HybridVLA(nn.Module):
    """
    End-to-end Vision-Language-Action model for robotic manipulation.

    Supports three operational modes:
    1. VLM mode: Vision-language understanding (image captioning, VQA)
    2. Planning mode: Chain-of-Point spatial reasoning for task planning
    3. Action mode: Direct action trajectory prediction for motor control

    Training follows a 4-stage pipeline (see training/train.py):
    Stage 1: Visual alignment (freeze LLM, train connector)
    Stage 2: Visual instruction tuning (freeze ViT, train LLM + connector)
    Stage 3: Distillation-aware ViT quantization (freeze LLM, QAT on ViT)
    Stage 4: Robotics fine-tuning (train action head + fine-tune all)
    """

    def __init__(self, config: HybridVLAConfig):
        super().__init__()
        self.config = config

        # === Vision Encoder ===
        self.vision_encoder = QuantizedVisionEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.vis_embed_dim,
            llm_dim=config.llm_dim,
            depth=config.vis_depth,
            num_heads=config.vis_num_heads,
            mlp_ratio=config.vis_mlp_ratio,
            window_size=config.vis_window_size,
            global_layers=config.vis_global_layers,
            quantize=config.vis_quantize,
        )

        # === Language Backbone ===
        self.llm = QuantizedLLMBackbone(
            vocab_size=config.vocab_size,
            dim=config.llm_dim,
            depth=config.llm_depth,
            num_heads=config.llm_num_heads,
            num_kv_heads=config.llm_num_kv_heads,
            mlp_ratio=config.llm_mlp_ratio,
            max_seq_len=config.llm_max_seq_len,
            quantize=config.llm_quantize,
            deepstack_layers=config.deepstack_layers,
        )

        # === Action Head ===
        self.action_head = ActionChunkHead(
            dim=config.llm_dim,
            action_dim=config.action_dim,
            chunk_size=config.action_chunk_size,
            num_heads=config.action_num_heads,
        )

        # === Spatiotemporal Memory (RynnBrain) ===
        if config.use_memory:
            self.memory = SpatiotemporalMemory(
                dim=config.llm_dim,
                num_slots=config.memory_num_slots,
                num_heads=config.memory_num_heads,
            )
        else:
            self.memory = None

        # === Chain-of-Point (RynnBrain) ===
        if config.use_chain_of_point:
            self.cop = ChainOfPointReasoner(
                dim=config.llm_dim,
                num_points=config.cop_num_points,
            )
        else:
            self.cop = None

        # === M-RoPE for position ID building ===
        self.mrope = MultimodalRoPE(
            head_dim=config.llm_dim // config.llm_num_heads,
            interleave=True,
        )

        # === Distillation loss ===
        self.distillation_loss = DistillationLoss(gamma=config.distillation_gamma)

        # Special token embeddings for modality markers
        self.vis_start_embed = nn.Parameter(torch.randn(1, 1, config.llm_dim) * 0.02)
        self.vis_end_embed = nn.Parameter(torch.randn(1, 1, config.llm_dim) * 0.02)

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        return_deepstack: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Encode image(s) through the quantized vision encoder.

        Args:
            pixel_values: [B, C, H, W]
            return_deepstack: If True, also extract intermediate features for DeepStack.

        Returns:
            dict with:
                visual_tokens: [B, N_vis, llm_dim] merged visual tokens
                merged_h, merged_w: spatial dimensions after merging
                deepstack_features: (optional) list of multi-level features
        """
        visual_tokens, merged_h, merged_w = self.vision_encoder(pixel_values)

        result = {
            "visual_tokens": visual_tokens,
            "merged_h": merged_h,
            "merged_w": merged_w,
        }

        if return_deepstack:
            ds_features = self.vision_encoder.get_intermediate_features(pixel_values)
            # Project and merge each level's features through the token merger
            projected = []
            for feat in ds_features:
                grid_h = int(feat.shape[1] ** 0.5)
                grid_w = feat.shape[1] // grid_h
                merged, _, _ = self.vision_encoder.merger(feat, grid_h, grid_w)
                projected.append(merged)
            result["deepstack_features"] = projected

        return result

    def _build_multimodal_input(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        merged_h: int = 0,
        merged_w: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build interleaved visual + text input sequence.

        Format: [<vis_start>] [visual tokens] [<vis_end>] [text tokens]

        Returns:
            inputs_embeds: [B, total_len, D]
            position_ids: [total_len, 3]
            visual_mask: [B, total_len] binary mask for visual positions
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        parts = []

        # Visual start marker
        parts.append(self.vis_start_embed.expand(B, -1, -1))
        # Visual tokens
        parts.append(visual_tokens)
        # Visual end marker
        parts.append(self.vis_end_embed.expand(B, -1, -1))

        n_vis = 1 + visual_tokens.shape[1] + 1  # start + tokens + end

        # Text tokens
        if input_ids is not None:
            text_embeds = self.llm.tok_emb(input_ids)
            parts.append(text_embeds)
            n_text = input_ids.shape[1]
        else:
            n_text = 0

        inputs_embeds = torch.cat(parts, dim=1)  # [B, total, D]

        # Build M-RoPE position IDs
        total_len = n_vis + n_text
        pos_ids = build_multimodal_position_ids(
            text_len=n_text,
            image_grids=[(merged_h, merged_w)] if merged_h > 0 else None,
            modality_order=["image_0", "text"] if merged_h > 0 else ["text"],
        )
        # Pad for start/end markers
        if merged_h > 0:
            marker_pos = torch.zeros(2, 3, dtype=torch.long)  # zero pos for markers
            # Insert marker positions around image positions
            img_end = 1 + merged_h * merged_w
            pos_ids = torch.cat([
                marker_pos[:1],  # vis_start
                pos_ids[:merged_h * merged_w],  # image
                marker_pos[1:],  # vis_end
                pos_ids[merged_h * merged_w:],  # text
            ], dim=0)

        # Ensure position_ids matches total length
        if pos_ids.shape[0] < total_len:
            pad = pos_ids[-1:].expand(total_len - pos_ids.shape[0], -1)
            pos_ids = torch.cat([pos_ids, pad], dim=0)
        pos_ids = pos_ids[:total_len]

        # Visual mask
        visual_mask = torch.zeros(B, total_len, device=device)
        visual_mask[:, :n_vis] = 1.0

        return inputs_embeds, pos_ids, visual_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        point_targets: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        timestep: int = 0,
        use_deepstack: bool = True,
        mode: str = "action",
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through the HybridVLA model.

        Args:
            pixel_values: [B, C, H, W] input image(s)
            input_ids: [B, N_text] language instruction tokens
            action_targets: [B, chunk_size, action_dim] ground truth actions (training)
            point_mask: [B, N] mask for Chain-of-Point prediction positions
            point_targets: [B, N, 3] ground truth point coordinates (training)
            memory_state: [B, num_slots, D] previous memory state (None to initialize)
            timestep: current timestep for memory temporal encoding
            use_deepstack: whether to use multi-level feature injection
            mode: "action" | "planning" | "vlm"

        Returns:
            dict with (depending on mode):
                actions: [B, chunk_size, action_dim] predicted action chunk
                logits: [B, N, vocab_size] language model logits
                hidden_states: [B, N, D] final hidden representations
                memory_state: [B, num_slots, D] updated memory
                points: [B, N, 3] predicted spatial points (if CoP enabled)
                loss: scalar training loss (if targets provided)
        """
        B = pixel_values.shape[0]
        device = pixel_values.device
        outputs = {}

        # 1. Encode image
        vis_out = self.encode_image(pixel_values, return_deepstack=use_deepstack)
        visual_tokens = vis_out["visual_tokens"]
        merged_h = vis_out["merged_h"]
        merged_w = vis_out["merged_w"]
        deepstack_features = vis_out.get("deepstack_features")

        # 2. Update spatiotemporal memory
        if self.memory is not None:
            if memory_state is None:
                memory_state = self.memory.init_memory(B, device)
            memory_state = self.memory.write(memory_state, visual_tokens, timestep)
            # Augment visual tokens with memory context
            visual_tokens = self.memory.read(visual_tokens, memory_state)
            outputs["memory_state"] = memory_state

        # 3. Build multimodal input sequence
        inputs_embeds, position_ids, visual_mask = self._build_multimodal_input(
            visual_tokens, input_ids, merged_h, merged_w
        )

        # 4. Run through LLM backbone
        causal = mode != "action"  # bidirectional for action prediction
        logits, hidden_states = self.llm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            causal=causal,
            visual_injections=deepstack_features if use_deepstack else None,
            visual_mask=visual_mask,
        )
        outputs["logits"] = logits
        outputs["hidden_states"] = hidden_states

        # 5. Chain-of-Point reasoning (planning mode)
        if self.cop is not None and mode in ("planning", "action"):
            cop_out = self.cop(hidden_states, point_mask)
            outputs["points"] = cop_out["points"]
            outputs["point_confidence"] = cop_out["confidence"]

        # 6. Action prediction
        if mode == "action":
            actions = self.action_head(hidden_states)
            outputs["actions"] = actions

        # 7. Compute losses if targets provided
        total_loss = torch.tensor(0.0, device=device)

        if action_targets is not None and "actions" in outputs:
            action_loss = self.action_head.compute_loss(outputs["actions"], action_targets)
            outputs["action_loss"] = action_loss
            total_loss = total_loss + action_loss

        if point_targets is not None and "points" in outputs:
            if point_mask is not None:
                mask = point_mask.unsqueeze(-1)
                point_loss = F.l1_loss(
                    outputs["points"] * mask,
                    point_targets * mask,
                )
            else:
                point_loss = F.l1_loss(outputs["points"], point_targets)
            outputs["point_loss"] = point_loss
            total_loss = total_loss + point_loss

        if input_ids is not None and mode in ("vlm", "planning"):
            # Language modeling loss on text tokens (shifted)
            n_vis = 1 + visual_tokens.shape[1] + 1
            text_logits = logits[:, n_vis:-1]  # shift right
            text_targets = input_ids[:, 1:]  # shift left
            min_len = min(text_logits.shape[1], text_targets.shape[1])
            if min_len > 0:
                lm_loss = F.cross_entropy(
                    text_logits[:, :min_len].reshape(-1, self.config.vocab_size),
                    text_targets[:, :min_len].reshape(-1),
                    ignore_index=-100,
                )
                outputs["lm_loss"] = lm_loss
                total_loss = total_loss + lm_loss

        outputs["loss"] = total_loss
        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        timestep: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-time action prediction.

        Returns:
            actions: [B, chunk_size, action_dim] denormalized actions
            memory_state: updated memory for next timestep
        """
        self.eval()
        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            memory_state=memory_state,
            timestep=timestep,
            mode="action",
        )
        actions = self.action_head.denormalize_actions(outputs["actions"])
        return actions, outputs.get("memory_state")

    def get_model_stats(self) -> dict:
        """Report model size and quantization statistics."""
        return compute_model_size_bits(self)

    def freeze_vision_encoder(self):
        """Freeze vision encoder (for Stage 2 and Stage 4 initial phase)."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder (for Stage 3 QAT)."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

    def freeze_llm(self):
        """Freeze LLM backbone (for Stage 1 and Stage 3)."""
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        """Unfreeze LLM backbone (for Stage 2 and Stage 4)."""
        for param in self.llm.parameters():
            param.requires_grad = True

    def freeze_all_except_connector(self):
        """Stage 1: only the token merger MLP trains."""
        self.freeze_vision_encoder()
        self.freeze_llm()
        for param in self.action_head.parameters():
            param.requires_grad = False
        if self.memory:
            for param in self.memory.parameters():
                param.requires_grad = False
        if self.cop:
            for param in self.cop.parameters():
                param.requires_grad = False
        # Unfreeze connector (token merger)
        for param in self.vision_encoder.merger.parameters():
            param.requires_grad = True
