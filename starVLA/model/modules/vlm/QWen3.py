# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 153716 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now. 
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning (f"action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md.")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)

    def build_qwenvl_inputs_with_memorys(
        self,
        images,
        instructions,
        memorys,  # List[List[List[Image.Image]]], [B][X_i][2]
        solutions=None,
        **kwargs
    ):
        """
        Build model inputs from raw data.
        Also preprocess memory images into visual features.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        B = len(images)

        # --- Step 1: Build chat messages (unchanged) ---
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # --- Step 2: Tokenize main inputs ---
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # --- Step 3: Process memory images into features ---
        if memorys and any(memorys):  # check non-empty
            # Flatten all memory images and collect metadata
            all_memory_images = []
            frame_counts = []  # X_i for each sample
            for mem in memorys:
                X_i = len(mem)
                frame_counts.append(X_i)
                for frame in mem:
                    assert len(frame) == 2, "Each memory frame must have exactly 2 views"
                    all_memory_images.extend(frame)  # [view0, view1]
            # Preprocess all memory images at once
            if all_memory_images:
                processed_mem = self.processor.image_processor(
                    images=all_memory_images,
                    return_tensors="pt"
                )
                pixel_values_mem = processed_mem["pixel_values"]
                # Construct image_grid_thw for memory images
                # For Qwen-VL, grid_thw can be inferred from image size, but we use processor's method
                # Alternative: use the same logic as in processor to get grid_thw
                # Since Qwen2VLImageProcessor doesn't return grid_thw directly, we compute it:
                # But actually, in Qwen3-VL, grid_thw is computed from image size and config.
                # We'll use a helper to generate it.

                # Get grid_thw for each image
                grid_thws = []
                for img in all_memory_images:
                    # Use the same logic as Qwen2VLImageProcessor to compute thw
                    # This is a simplified version — ideally reuse internal method
                    w, h = img.size
                    # From Qwen-VL: patch_size=14, max_pixels=... but we use dynamic
                    # Actually, the vision model expects grid_thw = [t, h_patches, w_patches]
                    # Since t=1 for static images:
                    patch_size = 14
                    h_patches = h // patch_size
                    w_patches = w // patch_size
                    grid_thws.append([1, h_patches, w_patches])
                image_grid_thw_mem = torch.tensor(grid_thws, dtype=torch.long)  # [N_total, 3]

                # Move to same device as model (will be moved later, but safe)
                pixel_values_mem = pixel_values_mem.to(self.model.device)
                image_grid_thw_mem = image_grid_thw_mem.to(self.model.device)

                # Extract features using self.model.visual
                with torch.no_grad():  # or not, depending on training
                    vision_output = self.model.visual(
                        pixel_values_mem, grid_thw=image_grid_thw_mem
                    )  # [N_total, D]
                # Reconstruct per-sample structure: [B] -> each [X_i, 2, D]
                # Handle different return types
                if isinstance(vision_output, tuple):
                    memory_features_flat = vision_output[0]  # usually the features
                elif hasattr(vision_output, 'last_hidden_state'):
                    memory_features_flat = vision_output.last_hidden_state
                else:
                    memory_features_flat = vision_output  # assume it's a tensor
                D = memory_features_flat.shape[-1]
                memory_features_list = []
                idx = 0
                for X_i in frame_counts:
                    feat = memory_features_flat[idx : idx + X_i * 2 * 64]  # [X_i*2, D]
                    feat = feat.view(X_i, 2, 64, D)  # [X_i, 2, D]
                    memory_features_list.append(feat)
                    idx += X_i * 2 * 64

                batch_inputs['memorys'] = memory_features_list
            else:
                # No memory images
                D = self.config.vision_config.out_hidden_size
                batch_inputs['memorys'] = [
                    torch.empty(0, 2, 64, D, device=self.model.device) for _ in memorys
                ]
        else:
            # No memory provided
            D = self.config.vision_config.out_hidden_size
            batch_inputs['memorys'] = [
                torch.empty(0, 2, 64, D, device=self.model.device) for _ in range(B)
            ]

        # --- Step 4: Handle labels (unchanged) ---
        if solutions is not None:
            action_token_min = _ACTION_TOKEN_MIN
            action_token_max = _ACTION_TOKEN_MAX
            labels = batch_inputs['input_ids'].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning("Action tokens not found in tokenizer.")
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
