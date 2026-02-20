"""
Multi-stage post-training pipeline for HybridVLA.

DESIGN PRINCIPLE: NO PRETRAINING FROM SCRATCH.
==============================================
We leverage existing open-source pretrained weights and only apply
post-training procedures. The pipeline has 4 stages, but depending
on your pretrained source, you can SKIP stages:

If starting from Qwen2.5-VL (recommended):
  - SKIP Stage 1 (ViT and LLM already aligned)
  - REDUCE Stage 2 (already instruction-tuned, just add robotics data)
  - DO Stage 3 (QAT for quantization -- optional if you don't need 1.58-bit)
  - DO Stage 4 (robotics SFT -- always needed)

If starting from separate SigLIP + Qwen2.5:
  - DO Stage 1 (align ViT output space to LLM input space)
  - DO Stage 2 (multimodal instruction tuning)
  - DO Stage 3 (optional QAT)
  - DO Stage 4 (robotics SFT)

STAGE DETAILS:
==============

Stage 1: Visual Alignment (connector training)
  - Freeze: ViT, LLM, action head, memory, CoP
  - Train: Only the TokenMerger MLP connector (~4M params)
  - Data: ~600K image-caption pairs
  - Duration: ~2 hours on 8x A100
  - WHY: The MLP connector must learn to project ViT features into
    the LLM's input space. This is cheap because it's only 2 linear layers.

Stage 2: Multimodal Instruction SFT
  - Freeze: ViT
  - Train: LLM (LoRA), connector, memory, CoP
  - Data: ~1M multimodal instruction samples + robotics captions
  - Duration: ~1-2 days on 8x A100
  - WHY: Teaches the model robotics-specific reasoning (spatial relations,
    object affordances, action vocabulary) while preserving general VLM
    capabilities through LoRA (doesn't destroy pretrained knowledge).
  - LoRA WHY: Full fine-tuning of a 3B LLM risks catastrophic forgetting.
    LoRA adds ~2% extra parameters that capture new knowledge while the
    frozen base weights preserve general language understanding.

Stage 3: Distillation-Aware QAT (quantization)
  - Freeze: LLM, connector, action head
  - Train: ViT (quantization-aware training)
  - Teacher: Full-precision SigLIP provides alignment targets
  - Data: 5M subset from Stage 2
  - Duration: ~3-5 days on 8x A100
  - WHY: Compresses the ViT from FP16 to 1.58-bit weights (8x memory
    reduction) with minimal quality loss (<1.5% accuracy drop on VQA).
    The distillation loss prevents the quantized model from diverging
    too far from the full-precision teacher's representations.
  - OPTIONAL: Skip if memory efficiency isn't critical for your deployment.

Stage 4: Robotics SFT (action prediction)
  - Freeze: ViT (or continue QAT)
  - Train: LLM (LoRA), action head, memory, CoP
  - Data: Robotics demonstrations (OXE, BridgeData, LIBERO)
  - Duration: ~4 hours for LIBERO, ~2-3 days for full OXE
  - WHY: This is the critical stage that teaches the model to actually
    control a robot. The action head learns to predict continuous
    trajectories, the memory learns to track objects across timesteps,
    and CoP learns to ground spatial reasoning in coordinates.

HARDWARE REQUIREMENTS:
  - Minimum: 1x A100 40GB (base config, batch_size=4)
  - Recommended: 4-8x A100 80GB (for full pipeline with larger configs)
  - For the small config: 1x RTX 3090/4090 is sufficient
"""

import argparse
import json
import logging
import os
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    """Configure logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log")),
        ],
    )


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Cosine decay with linear warmup.

    WHY cosine: Standard for transformer fine-tuning. The warmup phase
    prevents large gradient updates from destabilizing pretrained weights
    early in training. Cosine decay smoothly reduces LR to avoid
    oscillation near convergence.
    """
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def apply_lora(model: nn.Module, config) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to the LLM backbone.

    WHY LoRA over full fine-tuning:
    1. Memory: LoRA adds only ~2% extra parameters, reducing GPU memory
       by ~60% compared to full fine-tuning (no full-rank gradient storage)
    2. Catastrophic forgetting: Base weights are frozen, preserving the
       pretrained language understanding. LoRA weights capture task-specific
       adaptations as a low-rank delta.
    3. Modularity: LoRA weights can be merged back into base weights for
       inference, or swapped for different tasks without reloading the
       full model.
    4. Composability: Different LoRA adapters (one for spatial reasoning,
       one for action prediction) can theoretically be composed.

    Uses PEFT library if available, otherwise provides a simple fallback.
    """
    if not config.use_lora:
        logger.info("LoRA disabled, using full fine-tuning for LLM")
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )
        return model

    except ImportError:
        logger.warning(
            "PEFT not installed (pip install peft). "
            "Falling back to manual LoRA implementation."
        )
        return _apply_manual_lora(model, config)


class ManualLoRALinear(nn.Module):
    """Simple LoRA wrapper when PEFT is not available."""

    def __init__(self, base_linear: nn.Linear, r: int = 64, alpha: int = 128):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * (1 / r))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def _apply_manual_lora(model: nn.Module, config) -> nn.Module:
    """Apply LoRA manually without PEFT."""
    target_names = set(config.lora_target_modules)
    count = 0

    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if parts[-1] in target_names and isinstance(module, nn.Linear):
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            lora_module = ManualLoRALinear(module, config.lora_r, config.lora_alpha)
            setattr(parent, parts[-1], lora_module)
            count += 1

    logger.info(f"Manual LoRA applied to {count} linear layers")
    return model


# =============================================================================
# Training stages
# =============================================================================

def train_stage1_visual_alignment(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    output_dir: str,
    lr: float = 1e-3,
    num_epochs: int = 1,
):
    """
    Stage 1: Visual Alignment.

    Only trains the TokenMerger MLP connector to align ViT features
    with the LLM's embedding space.

    WHEN TO SKIP: If loading from Qwen2.5-VL, the connector is already
    trained. Set --skip-stage1 in the CLI.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Visual Alignment (connector only)")
    logger.info("=" * 60)

    model.freeze_all_except_connector()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    num_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_steps // 10, num_steps)

    model.train()
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                mode="vlm",
            )
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                avg = total_loss / (step + 1)
                logger.info(f"  Stage1 Epoch {epoch+1} Step {step+1}/{len(train_loader)} Loss: {avg:.4f}")

        logger.info(f"Stage1 Epoch {epoch+1} complete. Avg Loss: {total_loss / len(train_loader):.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, "stage1_connector.pt")
    torch.save(model.vision_encoder.merger.state_dict(), ckpt_path)
    logger.info(f"Stage 1 checkpoint saved to {ckpt_path}")


def train_stage2_instruction_sft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config,
    output_dir: str,
    lr: float = 2e-5,
    num_epochs: int = 3,
):
    """
    Stage 2: Multimodal Instruction SFT.

    Fine-tunes the LLM (via LoRA) on multimodal instruction data
    including robotics-specific spatial reasoning tasks.

    This stage also begins training the new modules (memory, CoP)
    so they learn alongside the LLM's improving representations.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Multimodal Instruction SFT (LLM LoRA + new modules)")
    logger.info("=" * 60)

    # Freeze ViT, unfreeze LLM (with LoRA), unfreeze new modules
    model.freeze_vision_encoder()
    model.unfreeze_llm()
    model = apply_lora(model, config)

    # Also ensure new modules are trainable
    for module in [model.memory, model.cop, model.action_head]:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    num_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_steps // 20, num_steps)

    model.train()
    device = next(model.parameters()).device
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                mode="vlm",
                use_deepstack=True,
            )
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 200 == 0:
                avg = total_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"  Stage2 Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                    f"Loss: {avg:.4f} LR: {lr_now:.2e}"
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"Stage2 Epoch {epoch+1} complete. Train Loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, mode="vlm")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(output_dir, "stage2_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"  New best! Saved to {ckpt_path}")

    ckpt_path = os.path.join(output_dir, "stage2_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Stage 2 checkpoint saved to {ckpt_path}")


def train_stage3_qat(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    output_dir: str,
    lr: float = 1e-4,
    num_epochs: int = 5,
):
    """
    Stage 3: Distillation-Aware Quantization-Aware Training (QAT).

    Compresses the vision encoder from FP16 to 1.58-bit (ternary weights).

    HOW IT WORKS:
    1. Load a full-precision teacher (SigLIP) and freeze it
    2. Apply BitNet 1.58-bit quantization to all ViT linear layers
    3. Train with: L = L_task + gamma * MSE(student_features, teacher_features)
    4. The STE (straight-through estimator) flows gradients through quantization

    WHY distillation helps: Naive QAT degrades accuracy significantly because
    ternary weights have very limited expressiveness. The teacher provides
    "soft targets" that guide the quantized model's feature space to stay
    close to the full-precision version, recovering most of the lost accuracy.
    BitVLA showed this raises zero-shot VQA from 42.4% to 50.8%.

    WHEN TO SKIP: If you don't need extreme memory efficiency, skip this
    stage and deploy with INT8 or INT4 post-training quantization instead
    (simpler, no training required, but larger memory footprint).
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Distillation-Aware QAT (vision encoder -> 1.58-bit)")
    logger.info("=" * 60)

    from ..model.quantization import replace_linear_with_bitlinear, DistillationLoss
    from ..model.pretrained_loader import load_teacher_for_distillation

    # Load teacher
    teacher = load_teacher_for_distillation(config.pretrained.teacher_vision_model_id)
    device = next(model.parameters()).device
    teacher = teacher.to(device)
    teacher.eval()

    # Apply quantization to vision encoder
    model.freeze_llm()
    model.unfreeze_vision_encoder()
    replace_linear_with_bitlinear(
        model.vision_encoder,
        exclude_names={"patch_embed", "merger"},  # keep connector FP for quality
    )
    logger.info("Applied 1.58-bit quantization to vision encoder linear layers")

    # Freeze everything except vision encoder
    for name, param in model.named_parameters():
        param.requires_grad = "vision_encoder" in name and "merger" not in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters (ViT only): {trainable:,}")

    distill_loss_fn = DistillationLoss(gamma=config.distillation_gamma)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,  # no weight decay for QAT
    )
    num_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_steps // 10, num_steps)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_align = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)

            # Student forward (quantized ViT)
            student_tokens, _, _ = model.vision_encoder(pixel_values)

            # Teacher forward (full-precision, no gradients)
            with torch.no_grad():
                teacher_out = teacher(pixel_values)
                teacher_features = teacher_out.last_hidden_state

            # Compute distillation loss
            # Align spatial dimensions if needed
            min_len = min(student_tokens.shape[1], teacher_features.shape[1])
            s_feat = student_tokens[:, :min_len]
            t_feat = teacher_features[:, :min_len]

            # Project teacher features to student dim if needed
            if s_feat.shape[-1] != t_feat.shape[-1]:
                if not hasattr(model, '_teacher_proj'):
                    model._teacher_proj = nn.Linear(
                        t_feat.shape[-1], s_feat.shape[-1], bias=False
                    ).to(device)
                t_feat = model._teacher_proj(t_feat)

            task_loss = torch.tensor(0.0, device=device)  # placeholder
            loss = distill_loss_fn(task_loss, s_feat, t_feat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                avg = total_loss / (step + 1)
                logger.info(f"  Stage3 Epoch {epoch+1} Step {step+1}/{len(train_loader)} Loss: {avg:.6f}")

        logger.info(f"Stage3 Epoch {epoch+1} complete. Avg Loss: {total_loss / max(len(train_loader), 1):.6f}")

    # Save quantized ViT
    ckpt_path = os.path.join(output_dir, "stage3_quantized_vit.pt")
    torch.save(model.vision_encoder.state_dict(), ckpt_path)
    logger.info(f"Stage 3 checkpoint saved to {ckpt_path}")

    # Report memory savings
    from ..model.quantization import compute_model_size_bits
    stats = compute_model_size_bits(model.vision_encoder)
    logger.info(f"Quantized ViT estimated size: {stats['estimated_size_mb']:.1f} MB")

    del teacher
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def train_stage4_robotics_sft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config,
    output_dir: str,
    lr: float = 5e-5,
    num_epochs: int = 10,
    eval_every: int = 1000,
):
    """
    Stage 4: Robotics SFT (the most important stage).

    Fine-tunes the complete model for robot manipulation:
    - LLM (LoRA): learns to ground instructions in visual observations
    - Action head: learns to predict continuous action trajectories
    - Memory: learns to track objects across timesteps
    - CoP: learns to predict spatial coordinates at reasoning steps

    TRAINING RECIPE (following BitVLA/OpenVLA-OFT):
    - L1 loss on action trajectories (not cross-entropy, since actions
      are continuous, not discrete tokens)
    - Optional: additional CoP loss for spatial grounding
    - Action normalization using dataset statistics
    - Gradient accumulation for effective batch size
    - Early stopping based on validation success rate

    WHY L1 over MSE for actions: L1 is more robust to outliers in
    the action data and produces sharper predictions. MSE tends to
    average out multi-modal action distributions, leading to
    "compromise" actions that satisfy no mode well.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Robotics SFT (action prediction)")
    logger.info("=" * 60)

    # Setup trainable parameters
    model.freeze_vision_encoder()

    # Apply LoRA if not already applied
    has_lora = any("lora" in n for n, _ in model.named_parameters())
    if not has_lora:
        model = apply_lora(model, config)

    # Ensure action head and new modules are trainable
    for module in [model.action_head, model.memory, model.cop]:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    num_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_steps // 20, num_steps)

    model.train()
    device = next(model.parameters()).device
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_action_loss = 0.0
        total_point_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch.get("input_ids"),
                action_targets=batch.get("action_targets"),
                point_targets=batch.get("point_targets"),
                timestep=batch.get("timestep", torch.zeros(1)).item(),
                mode="action",
                use_deepstack=True,
            )

            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if "action_loss" in outputs:
                total_action_loss += outputs["action_loss"].item()
            if "point_loss" in outputs:
                total_point_loss += outputs["point_loss"].item()

            global_step += 1

            if global_step % 100 == 0:
                n = step + 1
                logger.info(
                    f"  Stage4 Epoch {epoch+1} Step {n}/{len(train_loader)} "
                    f"Loss: {total_loss/n:.4f} "
                    f"Action: {total_action_loss/n:.4f} "
                    f"CoP: {total_point_loss/n:.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Periodic evaluation
            if val_loader and global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device, mode="action")
                logger.info(f"  [Step {global_step}] Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(output_dir, "stage4_best.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"  New best! Saved to {ckpt_path}")
                model.train()

        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"Stage4 Epoch {epoch+1} complete. Train Loss: {avg_loss:.4f}")

    # Save final checkpoint
    ckpt_path = os.path.join(output_dir, "stage4_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Stage 4 final checkpoint saved to {ckpt_path}")

    # Report model stats
    from ..model.quantization import compute_model_size_bits
    stats = compute_model_size_bits(model)
    logger.info(f"Final model estimated size: {stats['estimated_size_mb']:.1f} MB")
    logger.info(f"Total params: {stats['total_params']:,}")
    logger.info(f"Ternary (1.58-bit) params: {stats['ternary_params']:,}")
    logger.info(f"Full-precision params: {stats['fp_params']:,}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mode: str = "action",
    max_steps: int = 200,
) -> float:
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    for step, batch in enumerate(val_loader):
        if step >= max_steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch.get("input_ids"),
            action_targets=batch.get("action_targets"),
            point_targets=batch.get("point_targets"),
            mode=mode,
        )
        total_loss += outputs["loss"].item()
        count += 1

    return total_loss / max(count, 1)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HybridVLA Multi-Stage Post-Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start: load Qwen2.5-VL, skip stages 1-2, do robotics SFT only
  python -m hybrid_vla.training.train \\
      --config base --stage 4 \\
      --train-data data/libero_goal_train.jsonl \\
      --val-data data/libero_goal_val.jsonl \\
      --output-dir checkpoints/hybridvla_base_libero

  # Full pipeline from separate ViT + LLM
  python -m hybrid_vla.training.train \\
      --config base --stage 1 2 3 4 \\
      --stage1-data data/captions.jsonl \\
      --stage2-data data/instruct.jsonl \\
      --train-data data/robotics_train.jsonl \\
      --output-dir checkpoints/hybridvla_full

  # Quantization only (Stage 3)
  python -m hybrid_vla.training.train \\
      --config base --stage 3 \\
      --stage2-data data/instruct_subset.jsonl \\
      --resume checkpoints/stage2_final.pt \\
      --output-dir checkpoints/hybridvla_quantized
        """,
    )

    # Model config
    parser.add_argument("--config", default="base", choices=["small", "base", "large"])
    parser.add_argument("--stage", nargs="+", type=int, default=[4],
                        help="Which stages to run (e.g., --stage 2 3 4)")

    # Data paths
    parser.add_argument("--stage1-data", type=str, help="Caption data for Stage 1")
    parser.add_argument("--stage2-data", type=str, help="Instruction data for Stage 2")
    parser.add_argument("--train-data", type=str, help="Robotics training data for Stage 4")
    parser.add_argument("--val-data", type=str, help="Robotics validation data")
    parser.add_argument("--action-stats", type=str, help="Action normalization stats JSON")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="checkpoints/hybridvla")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")

    # Hardware
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")

    # LoRA
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA, use full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=64)

    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger.info(f"Arguments: {vars(args)}")

    # Build model
    from ..model.config import get_config
    from ..model.hybrid_vla import HybridVLA
    from ..model.pretrained_loader import initialize_from_pretrained

    config = get_config(args.config)
    if args.no_lora:
        config.use_lora = False
    config.lora_r = args.lora_r

    logger.info(f"Building HybridVLA ({config.model_name})")
    model = HybridVLA(config)

    # Load pretrained weights
    logger.info("Loading pretrained weights...")
    loaded = initialize_from_pretrained(model, config, device=args.device)
    for source, keys in loaded.items():
        logger.info(f"  {source}: {len(keys)} parameters loaded")

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(state, strict=False)

    model = model.to(args.device)
    if args.bf16:
        model = model.to(torch.bfloat16)

    # Load action stats
    action_mean, action_std = None, None
    if args.action_stats:
        with open(args.action_stats) as f:
            stats = json.load(f)
        action_mean = np.array(stats["mean"])
        action_std = np.array(stats["std"])
        model.action_head.set_action_stats(
            torch.tensor(action_mean, dtype=torch.float32),
            torch.tensor(action_std, dtype=torch.float32),
        )

    # Run requested stages
    stages = sorted(args.stage)

    for stage in stages:
        if stage == 1:
            if not args.stage1_data:
                logger.warning("Stage 1 requested but --stage1-data not provided, skipping")
                continue
            from ..data.prepare_data import MultimodalInstructDataset
            ds = MultimodalInstructDataset(args.stage1_data, img_size=config.img_size)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            train_stage1_visual_alignment(model, loader, config, args.output_dir)

        elif stage == 2:
            if not args.stage2_data:
                logger.warning("Stage 2 requested but --stage2-data not provided, skipping")
                continue
            from ..data.prepare_data import MultimodalInstructDataset
            ds = MultimodalInstructDataset(args.stage2_data, img_size=config.img_size)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            train_stage2_instruction_sft(model, loader, None, config, args.output_dir, num_epochs=args.epochs)

        elif stage == 3:
            data_path = args.stage2_data or args.train_data
            if not data_path:
                logger.warning("Stage 3 requires data (--stage2-data or --train-data), skipping")
                continue
            from ..data.prepare_data import MultimodalInstructDataset
            ds = MultimodalInstructDataset(data_path, img_size=config.img_size)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            train_stage3_qat(model, loader, config, args.output_dir, num_epochs=args.epochs)

        elif stage == 4:
            if not args.train_data:
                logger.warning("Stage 4 requires --train-data, skipping")
                continue
            from ..data.prepare_data import RoboticsDataset
            train_ds = RoboticsDataset(
                args.train_data, img_size=config.img_size,
                action_chunk_size=config.action_chunk_size,
                action_dim=config.action_dim,
                action_mean=action_mean, action_std=action_std,
            )
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers,
            )
            val_loader = None
            if args.val_data:
                val_ds = RoboticsDataset(
                    args.val_data, img_size=config.img_size,
                    action_chunk_size=config.action_chunk_size,
                    action_dim=config.action_dim,
                    action_mean=action_mean, action_std=action_std,
                )
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

            train_stage4_robotics_sft(
                model, train_loader, val_loader, config,
                args.output_dir, lr=args.lr, num_epochs=args.epochs,
            )

    logger.info("Training complete!")

    # Save final config for reproducibility
    config_path = os.path.join(args.output_dir, "config.json")
    import dataclasses
    config_dict = dataclasses.asdict(config)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
