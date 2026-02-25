"""
v2 Dataset for HybridVLA with multi-view, depth, and proprioception support.

Extends the v1 RoboticsDataset with:
- Multiple camera views per timestep (1-4 images)
- Optional depth images (precomputed or estimated on-the-fly)
- Proprioceptive state (joint positions, velocities, gripper)
- Past action chunk for temporal context
- Extended action horizon (50 steps default)

Data format (JSONL, one line per timestep):
{
    "episode_id": "ep_00042",
    "timestep": 15,
    "instruction": "Pick up the red block and place it on the blue plate",
    "images": {
        "left_shoulder": "path/to/left_0015.jpg",
        "right_shoulder": "path/to/right_0015.jpg",
        "wrist": "path/to/wrist_0015.jpg"
    },
    "depth_images": {  // optional
        "left_shoulder": "path/to/left_depth_0015.png",
        ...
    },
    "proprioception": [0.1, -0.3, ...],   // joint state vector
    "actions": [[x,y,z,...], ...],          // future action chunk
    "past_actions": [[x,y,z,...], ...],     // previous action chunk
    "cop_points": [{"point": [x,y,z], "text": "..."}, ...]
}

Backward compatible: if "images" is a string, treats it as single-view.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RoboticsDatasetV2(Dataset):
    """
    PyTorch dataset for HybridVLA v2 robotics training.

    Supports multi-view images, depth maps, proprioception, and
    past action context.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer=None,
        img_size: int = 384,
        action_chunk_size: int = 50,
        action_dim: int = 14,
        proprio_dim: int = 14,
        max_text_len: int = 256,
        view_names: list[str] | None = None,
        use_depth: bool = False,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.max_text_len = max_text_len
        self.view_names = view_names or ["primary"]
        self.use_depth = use_depth

        # Load manifest
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        # Action normalization
        if action_mean is not None:
            self.action_mean = torch.tensor(action_mean, dtype=torch.float32)
            self.action_std = torch.tensor(action_std, dtype=torch.float32).clamp(min=1e-6)
        else:
            self.action_mean = torch.zeros(action_dim)
            self.action_std = torch.ones(action_dim)

        logger.info(
            f"Loaded {len(self.samples)} v2 samples from {manifest_path} "
            f"(views={len(self.view_names)}, depth={use_depth})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image to [C, H, W] tensor."""
        try:
            from PIL import Image
            from torchvision import transforms
        except ImportError:
            raise ImportError("Install Pillow and torchvision")

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        try:
            img = Image.open(path).convert("RGB")
            return transform(img)
        except Exception:
            # Return zeros on load failure (don't crash training)
            return torch.zeros(3, self.img_size, self.img_size)

    def _load_depth(self, path: str) -> torch.Tensor:
        """Load depth image as a [1, H, W] tensor."""
        try:
            from PIL import Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            img = Image.open(path).convert("L")
            return transform(img)
        except Exception:
            return torch.zeros(1, self.img_size, self.img_size)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize instruction text."""
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text, max_length=self.max_text_len,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(0)
        else:
            encoded = list(text.encode("utf-8"))[:self.max_text_len]
            padded = encoded + [0] * (self.max_text_len - len(encoded))
            return torch.tensor(padded, dtype=torch.long)

    def _pad_actions(self, actions: list[list[float]]) -> torch.Tensor:
        """Pad/truncate action sequence to chunk_size and normalize."""
        if not actions:
            return torch.zeros(self.action_chunk_size, self.action_dim)

        actions_t = torch.tensor(actions, dtype=torch.float32)

        # Ensure correct action_dim
        if actions_t.shape[-1] < self.action_dim:
            pad = torch.zeros(actions_t.shape[0], self.action_dim - actions_t.shape[-1])
            actions_t = torch.cat([actions_t, pad], dim=-1)
        elif actions_t.shape[-1] > self.action_dim:
            actions_t = actions_t[:, :self.action_dim]

        # Pad/truncate to chunk_size
        if actions_t.shape[0] >= self.action_chunk_size:
            actions_t = actions_t[:self.action_chunk_size]
        else:
            pad_size = self.action_chunk_size - actions_t.shape[0]
            last_action = actions_t[-1:].expand(pad_size, -1)
            actions_t = torch.cat([actions_t, last_action], dim=0)

        # Normalize
        actions_t = (actions_t - self.action_mean) / self.action_std
        return actions_t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # === Multi-view images ===
        images_field = sample.get("images", sample.get("image", ""))
        if isinstance(images_field, str):
            # v1 compat: single image path
            view_images = {"primary": self._load_image(images_field)}
        elif isinstance(images_field, dict):
            view_images = {}
            for vname in self.view_names:
                if vname in images_field:
                    view_images[vname] = self._load_image(images_field[vname])
                else:
                    view_images[vname] = torch.zeros(3, self.img_size, self.img_size)
        else:
            view_images = {"primary": torch.zeros(3, self.img_size, self.img_size)}

        # === Depth images (optional) ===
        depth_images = {}
        if self.use_depth:
            depth_field = sample.get("depth_images", {})
            if isinstance(depth_field, dict):
                for vname in self.view_names:
                    if vname in depth_field:
                        depth_images[vname] = self._load_depth(depth_field[vname])

        # === Instruction ===
        instruction = sample.get("instruction", "")
        input_ids = self._tokenize(instruction)

        # === Actions ===
        action_targets = self._pad_actions(sample.get("actions", []))

        # === Past actions ===
        past_actions = self._pad_actions(sample.get("past_actions", []))

        # === Proprioception ===
        proprio_raw = sample.get("proprioception", [0.0] * self.proprio_dim)
        if len(proprio_raw) < self.proprio_dim:
            proprio_raw = proprio_raw + [0.0] * (self.proprio_dim - len(proprio_raw))
        proprio = torch.tensor(proprio_raw[:self.proprio_dim], dtype=torch.float32)

        # === Build result ===
        result = {
            "input_ids": input_ids,
            "action_targets": action_targets,
            "past_actions": past_actions,
            "proprioception": proprio,
            "timestep": torch.tensor(sample.get("timestep", 0), dtype=torch.long),
        }

        # Stack view images as separate keys
        for vname, img_tensor in view_images.items():
            result[f"pixel_values_{vname}"] = img_tensor

        for vname, depth_tensor in depth_images.items():
            result[f"depth_{vname}"] = depth_tensor

        # Chain-of-Point keypoints (optional)
        cop_points = sample.get("cop_points", sample.get("keypoints"))
        if cop_points:
            if isinstance(cop_points[0], dict):
                kp = [p["point"] for p in cop_points]
            else:
                kp = cop_points
            result["point_targets"] = torch.tensor(kp, dtype=torch.float32)

        return result


def collate_v2(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for v2 dataset.

    Handles variable view counts and optional depth images by
    stacking tensors per-key and collecting view images into dicts.
    """
    result = {}
    keys = batch[0].keys()
    view_keys = [k for k in keys if k.startswith("pixel_values_")]
    depth_keys = [k for k in keys if k.startswith("depth_")]
    other_keys = [k for k in keys if not k.startswith("pixel_values_") and not k.startswith("depth_")]

    # Stack standard tensors
    for key in other_keys:
        if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])

    # Stack per-view images into a dict
    images = {}
    for key in view_keys:
        view_name = key.replace("pixel_values_", "")
        images[view_name] = torch.stack([b[key] for b in batch])
    result["pixel_values"] = images

    depth = {}
    for key in depth_keys:
        view_name = key.replace("depth_", "")
        depth[view_name] = torch.stack([b[key] for b in batch])
    if depth:
        result["depth_images"] = depth

    return result
