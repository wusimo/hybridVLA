"""
HybridVLA v2: Next-generation Vision-Language-Action Model.

Extends v1 with innovations from LingBot-VLA, π0/π0.5/π0.6, and GR00T N1:

From LingBot-VLA:
  - Mixture-of-Transformers (MoT): Action Expert with separate FFN layers
    sharing self-attention with the VLM backbone
  - Flow Matching: continuous action generation via learned velocity fields
  - Multi-view camera support (1-4 views through shared ViT)
  - Depth perception via distillation from Depth Anything V2
  - Blockwise causal attention mask

From π0/π0.5/π0.6:
  - Knowledge insulation: action expert gradients don't corrupt VLM weights
  - Proprioceptive state encoding as action expert input
  - Extended action horizon (50 steps) with receding-horizon replanning

Preserved from v1:
  - Spatiotemporal memory (RynnBrain)
  - Chain-of-Point reasoning (RynnBrain)
  - 1.58-bit BitNet quantization (BitVLA)
  - DeepStack multi-level feature injection (Qwen3-VL)
  - No pretraining from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HybridVLAConfig
from .hybrid_vla import HybridVLA
from .action_expert import ActionExpert, BlockwiseCausalMask
from .flow_matching import FlowMatchingScheduler, FlowMatchingLoss
from .depth_encoder import DepthEncoder
from .proprioception import ProprioceptionEncoder, PastActionEncoder
from .quantization import compute_model_size_bits


class HybridVLAv2(HybridVLA):
    """
    HybridVLA v2 — extends v1 with MoT Action Expert, flow matching,
    multi-view, depth, proprioception, and knowledge insulation.

    When v2 features are disabled in config, this behaves identically to v1.

    Training: 3-phase pipeline (see training/train_v2.py):
      Phase 1: Alignment warmup (action expert FFN, projectors, memory)
      Phase 2: Full training with knowledge insulation
      Phase 3: Optional 1.58-bit QAT
    """

    def __init__(self, config: HybridVLAConfig):
        super().__init__(config)
        dim = config.llm_dim
        ae_dim = config.effective_action_expert_dim

        # === Action Expert (MoT) ===
        if config.use_action_expert:
            self.action_expert = ActionExpert(
                dim=ae_dim,
                action_dim=config.action_dim,
                chunk_size=config.action_chunk_size,
                num_layers=config.action_expert_depth,
                mlp_ratio=config.action_expert_mlp_ratio,
            )
        else:
            self.action_expert = None

        # === Flow Matching ===
        if config.use_flow_matching:
            self.flow_scheduler = FlowMatchingScheduler(
                num_inference_steps=config.flow_matching_steps,
                schedule=config.flow_noise_schedule,
            )
            self.flow_loss = FlowMatchingLoss()
        else:
            self.flow_scheduler = None
            self.flow_loss = None

        # === Depth Encoder ===
        if config.use_depth:
            self.depth_encoder = DepthEncoder(
                depth_dim=config.depth_token_dim,
                output_dim=dim,
                num_tokens_per_view=16,
            )
        else:
            self.depth_encoder = None

        # === Proprioception ===
        if config.use_proprio:
            self.proprio_encoder = ProprioceptionEncoder(
                proprio_dim=config.proprio_dim,
                hidden_dim=ae_dim,
            )
        else:
            self.proprio_encoder = None

        # === Past Action Encoder ===
        if config.use_past_actions:
            self.past_action_encoder = PastActionEncoder(
                action_dim=config.action_dim,
                hidden_dim=ae_dim,
                chunk_size=config.action_chunk_size,
            )
        else:
            self.past_action_encoder = None

    def encode_image_v2(
        self,
        images: dict[str, torch.Tensor] | torch.Tensor,
        return_deepstack: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Encode single or multi-view images.

        Args:
            images: Either a single [B, C, H, W] tensor (v1 compat),
                    or a dict mapping view_name -> [B, C, H, W].
            return_deepstack: Whether to extract multi-level features.

        Returns:
            dict with visual_tokens, merged_h, merged_w, and optional
            deepstack_features and depth_tokens.
        """
        # Normalize input to list
        if isinstance(images, torch.Tensor):
            image_list = [images]
        elif isinstance(images, dict):
            image_list = list(images.values())
        else:
            image_list = images

        num_views = len(image_list)

        if num_views == 1:
            # Single view: use v1 path
            result = self.encode_image(image_list[0], return_deepstack)
        else:
            # Multi-view: encode each through shared ViT, concatenate
            result = self.vision_encoder.encode_multi_view(
                image_list, return_deepstack
            )

        # Add depth tokens if depth encoder is available
        if self.depth_encoder is not None:
            depth_tokens_list = []
            for img in image_list:
                depth_tokens_list.append(self.depth_encoder(img))
            result["depth_tokens"] = torch.cat(depth_tokens_list, dim=1)

        return result

    def forward(
        self,
        pixel_values: torch.Tensor | dict[str, torch.Tensor] | None = None,
        input_ids: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        point_targets: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        timestep: int = 0,
        use_deepstack: bool = True,
        mode: str = "action",
        # v2 inputs
        proprio_state: torch.Tensor | None = None,
        past_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through HybridVLA v2.

        When use_action_expert is False, falls back to v1 behavior.
        When use_action_expert is True, uses MoT architecture.

        Args:
            pixel_values: [B, C, H, W] or dict of view_name -> [B, C, H, W]
            input_ids: [B, N_text] language instruction tokens
            action_targets: [B, chunk_size, action_dim] ground truth actions
            point_mask: [B, N] mask for CoP prediction positions
            point_targets: [B, N, 3] ground truth points
            memory_state: [B, num_slots, D] from previous step
            timestep: current timestep
            use_deepstack: use multi-level visual feature injection
            mode: "action" | "planning" | "vlm"
            proprio_state: [B, proprio_dim] robot joint state (v2)
            past_actions: [B, chunk_size, action_dim] previous action chunk (v2)

        Returns:
            dict with actions, logits, losses, memory_state, etc.
        """
        # If no action expert, fall back to v1
        if not self.config.use_action_expert or self.action_expert is None:
            if isinstance(pixel_values, dict):
                pixel_values = list(pixel_values.values())[0]
            return super().forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                action_targets=action_targets,
                point_mask=point_mask,
                point_targets=point_targets,
                memory_state=memory_state,
                timestep=timestep,
                use_deepstack=use_deepstack,
                mode=mode,
            )

        # === v2 MoT forward pass ===
        B = self._get_batch_size(pixel_values)
        device = self._get_device(pixel_values)
        outputs = {}

        # 1. Encode images (multi-view + depth)
        vis_out = self.encode_image_v2(pixel_values, return_deepstack=use_deepstack)
        visual_tokens = vis_out["visual_tokens"]
        merged_h = vis_out["merged_h"]
        merged_w = vis_out["merged_w"]
        deepstack_features = vis_out.get("deepstack_features")
        depth_tokens = vis_out.get("depth_tokens")

        # 2. Update spatiotemporal memory
        if self.memory is not None:
            if memory_state is None:
                memory_state = self.memory.init_memory(B, device)
            memory_state = self.memory.write(memory_state, visual_tokens, timestep)
            visual_tokens = self.memory.read(visual_tokens, memory_state)
            outputs["memory_state"] = memory_state

        # 3. Build observation sequence: [vis_start, visual, depth, vis_end, text]
        obs_embeds, obs_pos_ids, visual_mask = self._build_obs_sequence(
            visual_tokens, depth_tokens, input_ids, merged_h, merged_w
        )

        # 4. Chain-of-Point (on obs tokens via normal LLM forward)
        if self.cop is not None and mode in ("planning", "action"):
            # Quick LLM pass for CoP on observation tokens only
            _, cop_hidden = self.llm(
                inputs_embeds=obs_embeds,
                position_ids=obs_pos_ids,
                causal=False,
                visual_injections=deepstack_features if use_deepstack else None,
                visual_mask=visual_mask,
            )
            cop_out = self.cop(cop_hidden, point_mask)
            outputs["points"] = cop_out["points"]
            outputs["point_confidence"] = cop_out["confidence"]

        # 5. Action prediction via MoT
        if mode == "action":
            action_outputs = self._mot_action_forward(
                obs_embeds=obs_embeds,
                obs_pos_ids=obs_pos_ids,
                deepstack_features=deepstack_features if use_deepstack else None,
                visual_mask=visual_mask,
                action_targets=action_targets,
                proprio_state=proprio_state,
                past_actions=past_actions,
            )
            outputs.update(action_outputs)

        # 6. VLM logits (for language co-training or VLM mode)
        if mode in ("vlm", "planning"):
            logits, hidden = self.llm(
                inputs_embeds=obs_embeds,
                position_ids=obs_pos_ids,
                causal=True,
                visual_injections=deepstack_features if use_deepstack else None,
                visual_mask=visual_mask,
            )
            outputs["logits"] = logits
            outputs["hidden_states"] = hidden

        # 7. Compute losses
        total_loss = torch.tensor(0.0, device=device)

        if "action_loss" in outputs:
            total_loss = total_loss + outputs["action_loss"]

        if point_targets is not None and "points" in outputs:
            if point_mask is not None:
                mask = point_mask.unsqueeze(-1)
                point_loss = F.l1_loss(
                    outputs["points"] * mask, point_targets * mask
                )
            else:
                point_loss = F.l1_loss(outputs["points"], point_targets)
            outputs["point_loss"] = point_loss
            total_loss = total_loss + point_loss

        if input_ids is not None and mode in ("vlm", "planning") and "logits" in outputs:
            n_vis = 1 + visual_tokens.shape[1]
            if depth_tokens is not None:
                n_vis += depth_tokens.shape[1]
            n_vis += 1  # vis_end
            logits = outputs["logits"]
            text_logits = logits[:, n_vis:-1]
            text_targets = input_ids[:, 1:]
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

    def _build_obs_sequence(
        self,
        visual_tokens: torch.Tensor,
        depth_tokens: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        merged_h: int,
        merged_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build observation embedding sequence including depth tokens.

        Format: [vis_start, visual_tokens, depth_tokens, vis_end, text_tokens]
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        parts = []

        parts.append(self.vis_start_embed.expand(B, -1, -1))
        parts.append(visual_tokens)

        if depth_tokens is not None:
            parts.append(depth_tokens)

        parts.append(self.vis_end_embed.expand(B, -1, -1))

        n_vis = 1 + visual_tokens.shape[1]
        if depth_tokens is not None:
            n_vis += depth_tokens.shape[1]
        n_vis += 1  # vis_end

        n_text = 0
        if input_ids is not None:
            text_embeds = self.llm.tok_emb(input_ids)
            parts.append(text_embeds)
            n_text = input_ids.shape[1]

        obs_embeds = torch.cat(parts, dim=1)
        total_len = n_vis + n_text

        # Build position IDs (simplified: sequential for obs)
        pos_ids = torch.arange(total_len, device=device).unsqueeze(-1).expand(-1, 3)

        # Visual mask
        visual_mask = torch.zeros(B, total_len, device=device)
        visual_mask[:, :n_vis] = 1.0

        return obs_embeds, pos_ids, visual_mask

    def _mot_action_forward(
        self,
        obs_embeds: torch.Tensor,
        obs_pos_ids: torch.Tensor,
        deepstack_features: list[torch.Tensor] | None,
        visual_mask: torch.Tensor,
        action_targets: torch.Tensor | None = None,
        proprio_state: torch.Tensor | None = None,
        past_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        MoT action prediction with flow matching.

        Training: sample noise level σ, add noise to gt actions, predict velocity.
        Inference: iterative denoising from Gaussian noise.
        """
        B = obs_embeds.shape[0]
        device = obs_embeds.device
        outputs = {}
        chunk_size = self.config.action_chunk_size
        action_dim = self.config.action_dim

        # Encode proprioception
        proprio_tokens = None
        if self.proprio_encoder is not None and proprio_state is not None:
            proprio_tokens = self.proprio_encoder(proprio_state)

        # Encode past actions
        past_action_tokens = None
        if self.past_action_encoder is not None and past_actions is not None:
            past_action_tokens = self.past_action_encoder(past_actions)

        if self.training and action_targets is not None:
            # === Training: flow matching ===
            actions_norm = self.action_expert.normalize_actions(action_targets)
            noise = torch.randn_like(actions_norm)
            sigma = self.flow_scheduler.sample_sigma(B, device)
            noised_actions = self.flow_scheduler.add_noise(actions_norm, noise, sigma)

            # Build action tokens
            action_embeds = self.action_expert.embed_action_tokens(
                noised_actions, sigma, proprio_tokens, past_action_tokens
            )

            # Build blockwise causal mask
            N_obs = obs_embeds.shape[1]
            N_act = action_embeds.shape[1]
            mot_mask = BlockwiseCausalMask.build(N_obs, N_act, device)

            # MoT forward through shared attention + separate FFNs
            _, obs_hidden, action_hidden = self.llm.forward_mot(
                obs_embeds=obs_embeds,
                action_embeds=action_embeds,
                obs_position_ids=obs_pos_ids,
                mot_attention_mask=mot_mask,
                visual_injections=deepstack_features,
                visual_mask=visual_mask,
                action_expert=self.action_expert,
                knowledge_insulation=self.config.knowledge_insulation,
            )

            # Predict velocity
            velocity = self.action_expert.predict_velocity(action_hidden)
            outputs["velocity"] = velocity

            # Flow matching loss
            action_loss = self.flow_loss(velocity, actions_norm, noise)
            outputs["action_loss"] = action_loss

        else:
            # === Inference: iterative denoising ===
            actions = self._denoise_actions(
                obs_embeds, obs_pos_ids, deepstack_features,
                visual_mask, proprio_tokens, past_action_tokens,
            )
            outputs["actions"] = actions

        return outputs

    @torch.no_grad()
    def _denoise_actions(
        self,
        obs_embeds: torch.Tensor,
        obs_pos_ids: torch.Tensor,
        deepstack_features: list[torch.Tensor] | None,
        visual_mask: torch.Tensor,
        proprio_tokens: torch.Tensor | None,
        past_action_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Iterative denoising for inference.

        Runs K denoising steps (default 10), each doing a full MoT forward
        pass through the action expert. The VLM backbone's KV cache could be
        reused across steps (future optimization), but for correctness we
        re-run the full model each time.
        """
        B = obs_embeds.shape[0]
        device = obs_embeds.device
        chunk_size = self.config.action_chunk_size
        action_dim = self.config.action_dim

        # Start from Gaussian noise
        a_t = torch.randn(B, chunk_size, action_dim, device=device)

        sigmas = self.flow_scheduler.get_inference_sigmas(device)
        K = len(sigmas) - 1

        for i in range(K):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            dt = sigma_next - sigma_cur

            sigma_batch = sigma_cur.expand(B)

            action_embeds = self.action_expert.embed_action_tokens(
                a_t, sigma_batch, proprio_tokens, past_action_tokens
            )

            N_obs = obs_embeds.shape[1]
            N_act = action_embeds.shape[1]
            mot_mask = BlockwiseCausalMask.build(N_obs, N_act, device)

            _, _, action_hidden = self.llm.forward_mot(
                obs_embeds=obs_embeds,
                action_embeds=action_embeds,
                obs_position_ids=obs_pos_ids,
                mot_attention_mask=mot_mask,
                visual_injections=deepstack_features,
                visual_mask=visual_mask,
                action_expert=self.action_expert,
                knowledge_insulation=False,  # no grad in inference
            )

            velocity = self.action_expert.predict_velocity(action_hidden)
            a_t = a_t + velocity * dt

        # Denormalize
        actions = self.action_expert.denormalize_actions(a_t)
        return actions

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor | dict[str, torch.Tensor] | None = None,
        input_ids: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        timestep: int = 0,
        proprio_state: torch.Tensor | None = None,
        past_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Inference-time action prediction (v2).

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
            proprio_state=proprio_state,
            past_actions=past_actions,
        )
        return outputs["actions"], outputs.get("memory_state")

    def _get_batch_size(self, pixel_values) -> int:
        if isinstance(pixel_values, torch.Tensor):
            return pixel_values.shape[0]
        elif isinstance(pixel_values, dict):
            return next(iter(pixel_values.values())).shape[0]
        return 1

    def _get_device(self, pixel_values) -> torch.device:
        if isinstance(pixel_values, torch.Tensor):
            return pixel_values.device
        elif isinstance(pixel_values, dict):
            return next(iter(pixel_values.values())).device
        return next(self.parameters()).device

    # === Freeze/unfreeze helpers for v2 training ===

    def freeze_for_phase1(self):
        """Phase 1: train only new v2 modules (action expert, projectors)."""
        self.freeze_vision_encoder()
        self.freeze_llm()
        for param in self.action_head.parameters():
            param.requires_grad = False
        if self.memory:
            for param in self.memory.parameters():
                param.requires_grad = True
        if self.cop:
            for param in self.cop.parameters():
                param.requires_grad = True
        if self.action_expert:
            for param in self.action_expert.parameters():
                param.requires_grad = True
        if self.depth_encoder:
            # Freeze backbone, train projector
            for param in self.depth_encoder.projector.parameters():
                param.requires_grad = True
        if self.proprio_encoder:
            for param in self.proprio_encoder.parameters():
                param.requires_grad = True
        if self.past_action_encoder:
            for param in self.past_action_encoder.parameters():
                param.requires_grad = True
        # Unfreeze view embeddings
        self.vision_encoder.view_embeddings.weight.requires_grad = True

    def freeze_for_phase2(self):
        """Phase 2: train action expert fully, VLM via LoRA, with insulation."""
        self.freeze_vision_encoder()
        # LLM will be wrapped with LoRA externally
        self.unfreeze_llm()
        if self.action_expert:
            for param in self.action_expert.parameters():
                param.requires_grad = True
        if self.memory:
            for param in self.memory.parameters():
                param.requires_grad = True
        if self.cop:
            for param in self.cop.parameters():
                param.requires_grad = True
        if self.depth_encoder:
            for param in self.depth_encoder.projector.parameters():
                param.requires_grad = True
        if self.proprio_encoder:
            for param in self.proprio_encoder.parameters():
                param.requires_grad = True
        if self.past_action_encoder:
            for param in self.past_action_encoder.parameters():
                param.requires_grad = True
        self.vision_encoder.view_embeddings.weight.requires_grad = True

    def get_model_stats(self) -> dict:
        """Report v2 model size and quantization statistics."""
        stats = compute_model_size_bits(self)
        if self.action_expert:
            ae_params = sum(p.numel() for p in self.action_expert.parameters())
            stats["action_expert_params"] = ae_params
        if self.depth_encoder:
            de_params = sum(p.numel() for p in self.depth_encoder.parameters())
            stats["depth_encoder_params"] = de_params
        return stats
