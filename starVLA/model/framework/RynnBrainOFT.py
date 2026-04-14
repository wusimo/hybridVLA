from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)
IGNORE_INDEX = -100


# 你之前实现的 wrapper
# from starVLA.model.modules.vlm.rynnbrain_interface import _RynnBrain_Interface


@FRAMEWORK_REGISTRY.register("RynnBrainOFT")
class RynnBrain_OFT(baseframework):
    """
    OFT on top of RynnBrain:
      - RynnBrain backbone provides hidden states
      - Inject action placeholder tokens into prompt
      - Gather their hidden states and regress continuous actions via MLP head
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # --- 1) Init RynnBrain backbone ---
        self.vlm_interface = get_vlm_model(config=self.config)

        # --- 2) Init action head ---
        self.config.framework.action_model.action_hidden_dim = self.vlm_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # --- 3) Choose action placeholder token ---
        self.action_token = "🔍"
        ids = self.vlm_interface.processor.tokenizer(
            self.action_token, add_special_tokens=False
        )["input_ids"]

        if len(ids) != 1:
            raise RuntimeError(
                f"action_token '{self.action_token}' is not a single token (got ids={ids}). "
                f"Please choose another token or add it as a special token."
            )
        self.action_token_id = ids[0]

        # --- 4) Memory mode ---
        self.memory_mode = config.framework.qwenvl.memory

        self.l1_loss = nn.L1Loss()

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        """
        Train forward: L1 regression on future actions
        examples[i] requires:
          - image: List[PIL.Image]  (multi-view)
          - lang: str
          - action: np.ndarray [T, action_dim]
        """
        batch_images = [ex["image"] for ex in examples]          # [B, [PIL,...]]
        instructions = [ex["lang"] for ex in examples]          # [B]
        actions = [ex["action"] for ex in examples]             # [B, T, A]
        if self.memory_mode:
            memorys = [ex["memory"] for ex in examples]
        steps = [ex["step"] for ex in examples]

        # step 0: append action placeholders
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [ins + prompt_suffix for ins in instructions]

        # step 1: build inputs
        if not self.memory_mode:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs(
                images=batch_images, instructions=instructions
            )
        else:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs_with_memorys(
                images=batch_images, instructions=instructions, memorys=memorys, steps=steps
            )

        # step 2: run backbone
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm_interface(
                **rb_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]

        # step 3: gather action token embeddings -> action head -> loss
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = rb_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]

            pred_actions = self.action_model.predict_action(action_queries)  # [B, chunk_len, action_dim]

            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]  # [B, chunk_len, action_dim]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> np.ndarray:
        """
        Inference: regress continuous actions
        """
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        if self.memory_mode:
            memorys = [ex["memory"] for ex in examples]
        steps = [ex["step"] for ex in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [ins + prompt_suffix for ins in instructions]

        if not self.memory_mode:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs(
                images=batch_images, instructions=instructions
            )
        else:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs_with_memorys(
                images=batch_images, instructions=instructions, memorys=memorys, steps=steps
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm_interface(
                **rb_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = rb_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,
    ) -> torch.Tensor:
        """
        Same as your Qwenvl_OFT version (vectorized gather of last chunk_len action tokens)
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)
