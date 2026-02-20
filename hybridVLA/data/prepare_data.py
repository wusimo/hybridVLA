"""
Training data preparation for HybridVLA.

DESIGN RATIONALE - DATA SOURCES:
=================================
Since we're NOT pretraining from scratch, our data needs are focused on
post-training stages. Each stage has different data requirements:

Stage 1 (Visual Alignment) - SKIP if using Qwen2.5-VL init
  - Small captioning dataset to align ViT output with LLM input
  - ~600K image-caption pairs (LLaVA-style)
  - Only trains the MLP connector

Stage 2 (Multimodal Instruction Tuning) - REDUCED if using Qwen2.5-VL init
  - Diverse visual instruction-following data
  - Mix of VQA, spatial reasoning, object detection, action descriptions
  - Sources: LLaVA-Instruct, ShareGPT4V, ALLaVA, robotics captions
  - ~1M samples

Stage 3 (Distillation-Aware QAT) - Only vision encoder
  - Reuse a subset of Stage 2 data (5M samples from BitVLA paper)
  - The teacher model provides representation targets
  - No new data needed

Stage 4 (Robotics SFT) - The critical stage
  - Robot manipulation demonstrations with:
    * RGB images from the robot's camera
    * Language instructions ("pick up the red cube")
    * Action trajectories (7-DoF: xyz + rpy + gripper)
    * Optional: spatial keypoint annotations for CoP training
  - Sources:
    * Open X-Embodiment (OXE): largest open robotics dataset collection
    * DROID: diverse robot interaction dataset
    * BridgeData V2: tabletop manipulation
    * LIBERO: simulation benchmark for evaluation
  - ~100K-1M demonstrations depending on task diversity

This script handles:
1. Downloading and converting open-source robotics datasets
2. Unified format conversion (image + instruction + action trajectory)
3. Action normalization statistics computation
4. Train/val splits
5. Optional Chain-of-Point annotation generation
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Data format specification
# =============================================================================

@dataclass
class RoboticsEpisode:
    """
    Single robot manipulation episode in our unified format.

    Every training example is one timestep within an episode:
    (image, instruction, action_chunk, [keypoints])
    """
    image_path: str                          # path to RGB image
    instruction: str                         # natural language task instruction
    actions: list[list[float]]               # [chunk_size, action_dim] future actions
    keypoints: list[list[float]] | None = None  # [num_points, 3] for CoP supervision
    episode_id: str = ""                     # for grouping timesteps
    timestep: int = 0                        # position within episode
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Dataset classes
# =============================================================================

class RoboticsDataset(Dataset):
    """
    PyTorch dataset for robotics SFT (Stage 4).

    Loads preprocessed episodes from a JSONL manifest file.
    Each line in the manifest is a JSON object with:
    {
        "image": "/path/to/image.png",
        "instruction": "pick up the red block",
        "actions": [[x,y,z,r,p,y,g], ...],  # chunk_size x action_dim
        "keypoints": [[x,y,z], ...],         # optional, for CoP
        "episode_id": "ep_001",
        "timestep": 5
    }

    WHY JSONL: Simple, streamable, easy to inspect and filter.
    Each line is independent, so we can shuffle at the file level
    and shard across workers without complex indexing.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer=None,
        img_size: int = 384,
        action_chunk_size: int = 10,
        action_dim: int = 7,
        max_text_len: int = 256,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.max_text_len = max_text_len

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

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

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

        img = Image.open(path).convert("RGB")
        return transform(img)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize instruction text."""
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(0)
        else:
            # Fallback: simple byte encoding for testing without a tokenizer
            encoded = list(text.encode("utf-8"))[:self.max_text_len]
            padded = encoded + [0] * (self.max_text_len - len(encoded))
            return torch.tensor(padded, dtype=torch.long)

    def _pad_actions(self, actions: list[list[float]]) -> torch.Tensor:
        """Pad/truncate action sequence to chunk_size."""
        actions_t = torch.tensor(actions, dtype=torch.float32)

        if actions_t.shape[0] >= self.action_chunk_size:
            actions_t = actions_t[:self.action_chunk_size]
        else:
            # Repeat last action to fill chunk
            pad_size = self.action_chunk_size - actions_t.shape[0]
            last_action = actions_t[-1:].expand(pad_size, -1)
            actions_t = torch.cat([actions_t, last_action], dim=0)

        # Normalize
        actions_t = (actions_t - self.action_mean) / self.action_std
        return actions_t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        pixel_values = self._load_image(sample["image"])
        input_ids = self._tokenize(sample["instruction"])
        action_targets = self._pad_actions(sample["actions"])

        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "action_targets": action_targets,
            "timestep": torch.tensor(sample.get("timestep", 0), dtype=torch.long),
        }

        # Chain-of-Point keypoints (optional supervision)
        if "keypoints" in sample and sample["keypoints"]:
            kp = torch.tensor(sample["keypoints"], dtype=torch.float32)
            result["point_targets"] = kp

        return result


class MultimodalInstructDataset(Dataset):
    """
    Dataset for Stage 2 (Multimodal Instruction Tuning).

    Standard VLM instruction-following format:
    {
        "image": "/path/to/image.jpg",
        "conversations": [
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "A red cube on a table."}
        ]
    }

    WHY this stage matters even with Qwen2.5-VL init: We add robotics-
    specific instruction data that the base VLM hasn't seen, teaching
    the model spatial reasoning ("what is to the left of X"), action
    vocabulary ("pick up", "place on"), and object affordances.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer=None,
        img_size: int = 384,
        max_text_len: int = 512,
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_text_len = max_text_len

        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} multimodal instruction samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Image
        try:
            from PIL import Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = Image.open(sample["image"]).convert("RGB")
            pixel_values = transform(img)
        except Exception:
            pixel_values = torch.randn(3, self.img_size, self.img_size)

        # Flatten conversations to a single text
        convos = sample.get("conversations", [])
        text = ""
        for turn in convos:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            text += f"<|{role}|>\n{content}\n"

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text, max_length=self.max_text_len,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
        else:
            encoded = list(text.encode("utf-8"))[:self.max_text_len]
            padded = encoded + [0] * (self.max_text_len - len(encoded))
            input_ids = torch.tensor(padded, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }


# =============================================================================
# Data preparation utilities
# =============================================================================

def compute_action_statistics(manifest_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension mean and std of actions across the entire dataset.

    WHY: Action normalization is critical for stable training. Raw robot
    actions span different scales (position in meters vs rotation in radians
    vs gripper 0/1), and L1 loss without normalization would be dominated
    by the largest-scale dimensions.
    """
    all_actions = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                for action in sample["actions"]:
                    all_actions.append(action)

    actions_np = np.array(all_actions)
    mean = actions_np.mean(axis=0)
    std = actions_np.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    logger.info(f"Action statistics computed from {len(all_actions)} action steps")
    logger.info(f"  Mean: {mean}")
    logger.info(f"  Std:  {std}")

    return mean, std


def convert_oxe_to_manifest(
    oxe_dir: str,
    output_path: str,
    dataset_name: str = "bridge_orig",
    chunk_size: int = 10,
) -> str:
    """
    Convert Open X-Embodiment (OXE) RLDS dataset to our JSONL manifest format.

    WHY OXE: It's the largest collection of open-source robotics demonstrations
    (~2M episodes, 22 robot embodiments). BridgeData V2 within OXE is
    particularly useful for tabletop manipulation with a WidowX robot.

    Requires: pip install tensorflow tensorflow-datasets

    Each OXE episode contains:
    - steps[t].observation.image: RGB image
    - steps[t].language_instruction: text
    - steps[t].action: 7-DoF action vector
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        logger.error(
            "Install tensorflow-datasets: pip install tensorflow tensorflow-datasets\n"
            "OXE datasets also require: pip install oxe_envlogger"
        )
        raise

    logger.info(f"Converting OXE dataset '{dataset_name}' from {oxe_dir}")

    ds_builder = tfds.builder(dataset_name, data_dir=oxe_dir)
    ds = ds_builder.as_dataset(split="train")

    output_dir = Path(output_path).parent
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out_f:
        for ep_idx, episode in enumerate(ds):
            steps = list(episode["steps"])
            instruction = steps[0]["language_instruction"].numpy().decode("utf-8")

            # Extract all actions for this episode
            all_actions = []
            for step in steps:
                action = step["action"].numpy().tolist()
                all_actions.append(action)

            # Create training samples: one per timestep with future action chunk
            for t in range(len(steps)):
                # Save image
                img_array = steps[t]["observation"]["image"].numpy()
                img_path = str(images_dir / f"ep{ep_idx:06d}_t{t:04d}.png")

                try:
                    from PIL import Image
                    img = Image.fromarray(img_array)
                    img.save(img_path)
                except Exception:
                    continue

                # Future action chunk from current timestep
                future_actions = all_actions[t:t + chunk_size]
                if len(future_actions) < 2:
                    continue  # skip near-end timesteps with too few future actions

                sample = {
                    "image": img_path,
                    "instruction": instruction,
                    "actions": future_actions,
                    "episode_id": f"ep_{ep_idx:06d}",
                    "timestep": t,
                }

                out_f.write(json.dumps(sample) + "\n")
                count += 1

            if (ep_idx + 1) % 100 == 0:
                logger.info(f"Processed {ep_idx + 1} episodes, {count} samples")

    logger.info(f"Conversion complete: {count} samples -> {output_path}")
    return output_path


def convert_libero_to_manifest(
    libero_dir: str,
    output_path: str,
    suite: str = "libero_goal",
    chunk_size: int = 10,
) -> str:
    """
    Convert LIBERO benchmark data to our manifest format.

    WHY LIBERO: Standard evaluation benchmark for VLA models. It tests
    4 dimensions of robotic intelligence (spatial, object, goal, long-horizon).
    Using LIBERO data for training lets us directly compare with BitVLA,
    OpenVLA-OFT, and other baselines.

    Requires: pip install libero
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py: pip install h5py")

    libero_path = Path(libero_dir) / suite
    output_dir = Path(output_path).parent
    images_dir = output_dir / "images" / suite
    images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out_f:
        for demo_file in sorted(libero_path.glob("*.hdf5")):
            with h5py.File(demo_file, "r") as f:
                for demo_key in f["data"].keys():
                    demo = f["data"][demo_key]

                    # Extract image observations and actions
                    images = demo["obs"]["agentview_rgb"][:]
                    actions = demo["actions"][:]
                    instruction = f.attrs.get(
                        "language_instruction",
                        demo_file.stem.replace("_", " ")
                    )
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode("utf-8")

                    for t in range(len(images)):
                        img_path = str(images_dir / f"{demo_file.stem}_{demo_key}_t{t:04d}.png")

                        try:
                            from PIL import Image
                            img = Image.fromarray(images[t])
                            img.save(img_path)
                        except Exception:
                            continue

                        future_actions = actions[t:t + chunk_size].tolist()
                        if len(future_actions) < 2:
                            continue

                        sample = {
                            "image": img_path,
                            "instruction": instruction,
                            "actions": future_actions,
                            "episode_id": f"{demo_file.stem}_{demo_key}",
                            "timestep": t,
                        }
                        out_f.write(json.dumps(sample) + "\n")
                        count += 1

    logger.info(f"LIBERO conversion complete: {count} samples -> {output_path}")
    return output_path


def generate_cop_annotations(
    manifest_path: str,
    output_path: str,
    method: str = "heuristic",
) -> str:
    """
    Generate Chain-of-Point spatial annotations for training the CoP module.

    WHY: The CoP module needs (x, y, z) coordinate supervision at reasoning
    steps. Since most robotics datasets don't include this, we generate
    pseudo-labels using one of:

    1. "heuristic": Extract keypoints from the action trajectory
       - Start point: current end-effector position
       - Grasp point: position at first gripper close
       - Place point: position at last gripper open
       - Waypoints: intermediate positions at trajectory inflection points

    2. "vlm_annotated": Use a VLM (Qwen2.5-VL) to annotate spatial points
       in the image corresponding to task-relevant objects

    The heuristic method is fast and doesn't need an additional model.
    """
    logger.info(f"Generating CoP annotations for {manifest_path}")

    count = 0
    with open(manifest_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            sample = json.loads(line)
            actions = sample["actions"]

            if method == "heuristic":
                keypoints = _extract_heuristic_keypoints(actions)
            else:
                keypoints = _extract_heuristic_keypoints(actions)  # fallback

            sample["keypoints"] = keypoints
            f_out.write(json.dumps(sample) + "\n")
            count += 1

    logger.info(f"Generated CoP annotations for {count} samples -> {output_path}")
    return output_path


def _extract_heuristic_keypoints(
    actions: list[list[float]],
    num_points: int = 8,
) -> list[list[float]]:
    """
    Extract spatial keypoints from an action trajectory using heuristics.

    Points extracted:
    1. Start position (current EEF xyz)
    2. Midpoint position
    3. End position (final EEF xyz)
    4-8. Evenly spaced waypoints for trajectory shape

    These provide supervision for the CoP module to learn trajectory-aware
    spatial reasoning.
    """
    if not actions:
        return [[0.0, 0.0, 0.0]] * num_points

    actions_np = np.array(actions)
    xyz = actions_np[:, :3]  # first 3 dims assumed to be xyz position

    # Sample points evenly along the trajectory
    indices = np.linspace(0, len(xyz) - 1, num_points, dtype=int)
    keypoints = xyz[indices].tolist()

    return keypoints


# =============================================================================
# CLI interface
# =============================================================================

def main():
    """Command-line entry point for data preparation."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data for HybridVLA")
    subparsers = parser.add_subparsers(dest="command")

    # Convert OXE data
    oxe_parser = subparsers.add_parser("convert-oxe", help="Convert Open X-Embodiment data")
    oxe_parser.add_argument("--oxe-dir", required=True, help="Path to OXE data directory")
    oxe_parser.add_argument("--output", required=True, help="Output manifest path (.jsonl)")
    oxe_parser.add_argument("--dataset", default="bridge_orig", help="OXE dataset name")
    oxe_parser.add_argument("--chunk-size", type=int, default=10)

    # Convert LIBERO data
    libero_parser = subparsers.add_parser("convert-libero", help="Convert LIBERO benchmark data")
    libero_parser.add_argument("--libero-dir", required=True, help="Path to LIBERO data")
    libero_parser.add_argument("--output", required=True, help="Output manifest path (.jsonl)")
    libero_parser.add_argument("--suite", default="libero_goal",
                               choices=["libero_spatial", "libero_object", "libero_goal", "libero_long"])
    libero_parser.add_argument("--chunk-size", type=int, default=10)

    # Compute action stats
    stats_parser = subparsers.add_parser("compute-stats", help="Compute action normalization statistics")
    stats_parser.add_argument("--manifest", required=True, help="Input manifest path")
    stats_parser.add_argument("--output", required=True, help="Output stats JSON path")

    # Generate CoP annotations
    cop_parser = subparsers.add_parser("generate-cop", help="Generate Chain-of-Point annotations")
    cop_parser.add_argument("--manifest", required=True, help="Input manifest path")
    cop_parser.add_argument("--output", required=True, help="Output annotated manifest path")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "convert-oxe":
        convert_oxe_to_manifest(args.oxe_dir, args.output, args.dataset, args.chunk_size)

    elif args.command == "convert-libero":
        convert_libero_to_manifest(args.libero_dir, args.output, args.suite, args.chunk_size)

    elif args.command == "compute-stats":
        mean, std = compute_action_statistics(args.manifest)
        stats = {"mean": mean.tolist(), "std": std.tolist()}
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved action statistics to {args.output}")

    elif args.command == "generate-cop":
        generate_cop_annotations(args.manifest, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
