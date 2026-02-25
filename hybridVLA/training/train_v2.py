"""
HybridVLA v2: 3-Phase Post-Training Pipeline.

PRINCIPLE: NO pretraining from scratch. Load pretrained weights and
apply targeted post-training with the new MoT architecture.

Phase 0: Weight Loading (no training)
    ├── VLM backbone ← Qwen2.5-VL-3B/7B (pretrained)
    ├── Action Expert FFN ← copied from VLM FFN layers (warm start)
    ├── Depth encoder ← Depth Anything V2 (pretrained, frozen)
    ├── Memory, CoP ← random init (small)
    └── Proprio encoder, view embeddings ← random init (small)

Phase 1: Alignment + Action Expert Warmup
    ├── Train: Action Expert FFN, proprio encoder, view embeddings,
    │          depth projector, memory, CoP
    ├── Freeze: VLM backbone (or LoRA r=16), depth encoder backbone
    ├── Data: Robot demonstration data (multi-view, with proprio)
    ├── Loss: Flow matching on actions + CoP point loss
    └── Purpose: Align new modules to VLM representations

Phase 2: Full Fine-Tuning with Knowledge Insulation
    ├── Train: Everything (VLM via LoRA r=64, Action Expert full)
    ├── Freeze: Depth encoder backbone
    ├── Gradient: Knowledge insulation enabled
    ├── Data: Robot data + language co-training data (mixed)
    ├── Loss: Flow matching + CoP + LM loss (co-training)
    └── Purpose: Maximize action quality, preserve VLM

Phase 3: Optional Quantization (QAT)
    ├── Apply 1.58-bit QAT to VLM backbone + Action Expert
    └── Distillation from full-precision teacher
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train_v2.log")),
        ],
    )


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def apply_lora(model: nn.Module, config, lora_r: int = 64) -> nn.Module:
    """Apply LoRA to the LLM backbone for parameter-efficient fine-tuning."""
    if not config.use_lora:
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA (r={lora_r}): {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
        return model
    except ImportError:
        logger.warning("PEFT not installed. Using full fine-tuning for LLM.")
        return model


# ============================================================================
# Phase 1: Alignment + Action Expert Warmup
# ============================================================================

def train_phase1(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    output_dir: str,
    lr: float = 1e-4,
    num_epochs: int = 5,
):
    """
    Phase 1: Train new v2 modules while keeping VLM frozen.

    Trains: Action Expert FFN, proprioception encoder, past action encoder,
            depth projector, memory, CoP, view embeddings.
    Frozen: VLM backbone, depth encoder backbone.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Alignment + Action Expert Warmup")
    logger.info("=" * 60)

    model.freeze_for_phase1()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    num_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_steps // 10, num_steps)

    model.train()
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_action_loss = 0.0
        total_point_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            outputs = model(
                pixel_values=batch.get("pixel_values"),
                input_ids=batch.get("input_ids"),
                action_targets=batch.get("action_targets"),
                point_targets=batch.get("point_targets"),
                proprio_state=batch.get("proprioception"),
                past_actions=batch.get("past_actions"),
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

            if (step + 1) % 100 == 0:
                n = step + 1
                logger.info(
                    f"  Phase1 Epoch {epoch+1} Step {n}/{len(train_loader)} "
                    f"Loss: {total_loss/n:.4f} "
                    f"Action: {total_action_loss/n:.4f} "
                    f"CoP: {total_point_loss/n:.4f}"
                )

        logger.info(f"Phase1 Epoch {epoch+1} complete. Loss: {total_loss / max(len(train_loader), 1):.4f}")

    ckpt_path = os.path.join(output_dir, "phase1_checkpoint.pt")
    _save_checkpoint(model, optimizer, epoch, ckpt_path)
    logger.info(f"Phase 1 checkpoint saved to {ckpt_path}")


# ============================================================================
# Phase 2: Full Fine-Tuning with Knowledge Insulation
# ============================================================================

def train_phase2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config,
    output_dir: str,
    lr: float = 5e-5,
    num_epochs: int = 20,
    lora_r: int = 64,
    eval_every: int = 1000,
):
    """
    Phase 2: Full fine-tuning with knowledge insulation.

    The VLM backbone is fine-tuned via LoRA while the Action Expert
    trains fully. Knowledge insulation prevents action gradients from
    flowing back into the VLM backbone.

    Data: mixed batches of robot demonstrations + language co-training.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Full Fine-Tuning with Knowledge Insulation")
    logger.info("=" * 60)

    model.freeze_for_phase2()
    model = apply_lora(model, config, lora_r=lora_r)

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
        total_lm_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)
            source = batch.pop("_source", "robot")

            # Route to correct forward mode based on data source
            if source in ("language", "vlm"):
                mode = "vlm"
            else:
                mode = "action"

            outputs = model(
                pixel_values=batch.get("pixel_values"),
                input_ids=batch.get("input_ids"),
                action_targets=batch.get("action_targets") if mode == "action" else None,
                point_targets=batch.get("point_targets"),
                proprio_state=batch.get("proprioception") if mode == "action" else None,
                past_actions=batch.get("past_actions") if mode == "action" else None,
                timestep=batch.get("timestep", torch.zeros(1)).item(),
                mode=mode,
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
            if "lm_loss" in outputs:
                total_lm_loss += outputs["lm_loss"].item()

            global_step += 1

            if global_step % 100 == 0:
                n = step + 1
                logger.info(
                    f"  Phase2 Epoch {epoch+1} Step {n}/{len(train_loader)} "
                    f"Loss: {total_loss/n:.4f} "
                    f"Action: {total_action_loss/max(n,1):.4f} "
                    f"LM: {total_lm_loss/max(n,1):.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Periodic evaluation
            if val_loader and global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                logger.info(f"  [Step {global_step}] Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(output_dir, "phase2_best.pt")
                    _save_checkpoint(model, optimizer, epoch, ckpt_path)
                    logger.info(f"  New best! Saved to {ckpt_path}")
                model.train()

        logger.info(f"Phase2 Epoch {epoch+1} complete. Loss: {total_loss / max(len(train_loader), 1):.4f}")

    ckpt_path = os.path.join(output_dir, "phase2_final.pt")
    _save_checkpoint(model, optimizer, epoch, ckpt_path)
    logger.info(f"Phase 2 checkpoint saved to {ckpt_path}")

    # Report model stats
    stats = model.get_model_stats()
    logger.info(f"Model size: {stats['estimated_size_mb']:.1f} MB")
    logger.info(f"Total params: {stats['total_params']:,}")


# ============================================================================
# Phase 3: Optional Quantization (QAT)
# ============================================================================

def train_phase3_qat(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    output_dir: str,
    lr: float = 1e-4,
    num_epochs: int = 5,
):
    """
    Phase 3: Distillation-Aware QAT.

    Applies 1.58-bit quantization to both the VLM backbone and Action Expert.
    Uses a full-precision teacher for distillation alignment.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Distillation-Aware QAT (1.58-bit)")
    logger.info("=" * 60)

    from ..model.quantization import replace_linear_with_bitlinear, DistillationLoss
    from ..model.pretrained_loader import load_teacher_for_distillation

    device = next(model.parameters()).device

    # Load teacher
    teacher = load_teacher_for_distillation(config.pretrained.teacher_vision_model_id)
    teacher = teacher.to(device)

    # Apply quantization
    exclude = {"patch_embed", "merger", "tok_emb", "lm_head", "view_embeddings"}
    replace_linear_with_bitlinear(model.vision_encoder, exclude_names=exclude)
    if model.action_expert is not None:
        replace_linear_with_bitlinear(model.action_expert, exclude_names={"action_proj", "velocity_head"})
    logger.info("Applied 1.58-bit quantization")

    # Train quantized layers
    for name, param in model.named_parameters():
        param.requires_grad = "vision_encoder" in name or "action_expert" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters (QAT): {trainable:,}")

    distill_loss_fn = DistillationLoss(gamma=config.distillation_gamma)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )
    num_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_steps // 10, num_steps)

    model.train()
    teacher.eval()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            # Get first view for teacher comparison
            pv = batch.get("pixel_values")
            if isinstance(pv, dict):
                pv = next(iter(pv.values()))

            # Student
            student_tokens, _, _ = model.vision_encoder(pv)

            # Teacher
            with torch.no_grad():
                teacher_out = teacher(pv)
                teacher_features = teacher_out.last_hidden_state

            min_len = min(student_tokens.shape[1], teacher_features.shape[1])
            s_feat = student_tokens[:, :min_len]
            t_feat = teacher_features[:, :min_len]
            if s_feat.shape[-1] != t_feat.shape[-1]:
                if not hasattr(model, "_teacher_proj"):
                    model._teacher_proj = nn.Linear(
                        t_feat.shape[-1], s_feat.shape[-1], bias=False
                    ).to(device)
                t_feat = model._teacher_proj(t_feat)

            loss = distill_loss_fn(torch.tensor(0.0, device=device), s_feat, t_feat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                logger.info(f"  Phase3 Epoch {epoch+1} Step {step+1} Loss: {total_loss/(step+1):.6f}")

        logger.info(f"Phase3 Epoch {epoch+1} complete. Loss: {total_loss / max(len(train_loader), 1):.6f}")

    ckpt_path = os.path.join(output_dir, "phase3_quantized.pt")
    _save_checkpoint(model, optimizer, epoch, ckpt_path)

    from ..model.quantization import compute_model_size_bits
    stats = compute_model_size_bits(model)
    logger.info(f"Quantized model size: {stats['estimated_size_mb']:.1f} MB")

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int = 200,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    for step, batch in enumerate(val_loader):
        if step >= max_steps:
            break
        batch = _move_batch_to_device(batch, device)

        outputs = model(
            pixel_values=batch.get("pixel_values"),
            input_ids=batch.get("input_ids"),
            action_targets=batch.get("action_targets"),
            proprio_state=batch.get("proprioception"),
            past_actions=batch.get("past_actions"),
            mode="action",
        )
        total_loss += outputs["loss"].item()
        count += 1

    return total_loss / max(count, 1)


# ============================================================================
# Utilities
# ============================================================================

def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv
                         for sk, sv in v.items()}
        else:
            result[k] = v
    return result


def _save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }, path)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HybridVLA v2: 3-Phase Post-Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start: Phase 2 only (recommended for most users)
  python -m hybridVLA.training.train_v2 \\
      --config v2-base --phase 2 \\
      --train-data data/robot_train.jsonl \\
      --output-dir checkpoints/hybridvla_v2

  # Full pipeline: Phase 1 + 2
  python -m hybridVLA.training.train_v2 \\
      --config v2-base --phase 1 2 \\
      --train-data data/robot_train.jsonl \\
      --output-dir checkpoints/hybridvla_v2

  # With quantization: Phase 1 + 2 + 3
  python -m hybridVLA.training.train_v2 \\
      --config v2-base --phase 1 2 3 \\
      --train-data data/robot_train.jsonl \\
      --output-dir checkpoints/hybridvla_v2_quantized
        """,
    )

    parser.add_argument("--config", default="v2-base", choices=["v2-base", "v2-large"])
    parser.add_argument("--phase", nargs="+", type=int, default=[2],
                        help="Which phases to run (1, 2, 3)")

    parser.add_argument("--train-data", type=str, required=True, help="Robot training data (JSONL)")
    parser.add_argument("--val-data", type=str, help="Validation data (JSONL)")
    parser.add_argument("--cotrain-data", type=str, help="Language co-training data (JSONL)")
    parser.add_argument("--action-stats", type=str, help="Action normalization stats JSON")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1000)

    parser.add_argument("--output-dir", type=str, default="checkpoints/hybridvla_v2")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()
    setup_logging(args.output_dir)
    logger.info(f"Arguments: {vars(args)}")

    # Build model
    from ..model.config import get_config
    from ..model.hybrid_vla_v2 import HybridVLAv2
    from ..model.pretrained_loader import initialize_from_pretrained_v2

    config = get_config(args.config)
    config.lora_r = args.lora_r

    logger.info(f"Building HybridVLA v2 ({config.model_name})")
    model = HybridVLAv2(config)

    # Load pretrained weights
    logger.info("Loading pretrained weights...")
    loaded = initialize_from_pretrained_v2(model, config, device=args.device)
    for source, count in loaded.items():
        logger.info(f"  {source}: {count} parameters loaded")

    # Resume
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model = model.to(args.device)
    if args.bf16:
        model = model.to(torch.bfloat16)

    # Load action stats
    if args.action_stats:
        with open(args.action_stats) as f:
            stats = json.load(f)
        action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        action_std = torch.tensor(stats["std"], dtype=torch.float32)
        if model.action_expert:
            model.action_expert.set_action_stats(action_mean, action_std)
        model.action_head.set_action_stats(action_mean, action_std)

    # Build dataloaders
    from ..data.dataset_v2 import RoboticsDatasetV2, collate_v2

    train_ds = RoboticsDatasetV2(
        args.train_data,
        img_size=config.img_size,
        action_chunk_size=config.action_chunk_size,
        action_dim=config.action_dim,
        proprio_dim=config.proprio_dim,
        view_names=config.view_names,
        use_depth=config.use_depth,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_v2,
    )

    val_loader = None
    if args.val_data:
        val_ds = RoboticsDatasetV2(
            args.val_data,
            img_size=config.img_size,
            action_chunk_size=config.action_chunk_size,
            action_dim=config.action_dim,
            proprio_dim=config.proprio_dim,
            view_names=config.view_names,
            use_depth=config.use_depth,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            num_workers=args.workers, collate_fn=collate_v2,
        )

    # Run phases
    phases = sorted(args.phase)
    for phase in phases:
        if phase == 1:
            train_phase1(
                model, train_loader, config, args.output_dir,
                lr=args.lr, num_epochs=min(args.epochs, 5),
            )
        elif phase == 2:
            train_phase2(
                model, train_loader, val_loader, config, args.output_dir,
                lr=args.lr, num_epochs=args.epochs, lora_r=args.lora_r,
                eval_every=args.eval_every,
            )
        elif phase == 3:
            train_phase3_qat(
                model, train_loader, config, args.output_dir,
                num_epochs=min(args.epochs, 5),
            )

    logger.info("Training complete!")

    # Save config
    import dataclasses
    config_path = os.path.join(args.output_dir, "config_v2.json")
    config_dict = dataclasses.asdict(config)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
