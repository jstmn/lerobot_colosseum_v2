#!/usr/bin/env python

# Copyright 2025 Allen Institute for AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MolmoAct2 policy implementation for LeRobot."""

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


logger = logging.getLogger(__name__)


def tensor_to_pil(tensor: Tensor, flip_180: bool = True) -> Image.Image:
    """
    Convert a tensor image to PIL Image.

    Args:
        tensor: Image tensor of shape (C, H, W) or (B, C, H, W) with values in [0, 1]
        flip_180: Whether to flip image 180 degrees (LIBERO camera convention)

    Returns:
        PIL Image in RGB format
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    # Convert from (C, H, W) to (H, W, C)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()

    # Convert from [0, 1] to [0, 255]
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = img_np.clip(0, 255).astype(np.uint8)

    # Flip 180 degrees for LIBERO camera convention
    if flip_180:
        img_np = img_np[::-1, ::-1, :].copy()

    return Image.fromarray(img_np, mode="RGB")


class MolmoAct2Policy(PreTrainedPolicy):
    """
    MolmoAct2 policy for robot manipulation.

    MolmoAct2 is a Vision-Language-Action model based on Molmo2-ER with a
    flow matching continuous action expert. It uses two camera views
    (agentview and wrist) along with robot state and task description
    to predict robot actions.

    Reference: https://huggingface.co/allenai/MolmoAct2-LIBERO
    """

    config_class = MolmoAct2Config
    name = "molmoact2"

    def __init__(self, config: MolmoAct2Config):
        """
        Initialize MolmoAct2 policy.

        Args:
            config: Configuration object for MolmoAct2
        """
        super().__init__(config)
        self.config = config

        # Lazy load model and processor
        self._model = None
        self._processor = None
        self._action_tokenizer = None

        # Action queue for action chunking
        self._action_queue: deque = deque(maxlen=config.chunk_size)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load MolmoAct2 model and processor from HuggingFace."""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for MolmoAct2. "
                "Install it with: pip install transformers"
            ) from e

        logger.info(f"Loading MolmoAct2 model from {self.config.hf_model_id}")

        # Determine dtype
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Load model
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=dtype,
        )

        # Load action tokenizer for discrete mode
        if self.config.inference_action_mode == "discrete":
            self._action_tokenizer = AutoProcessor.from_pretrained(
                "allenai/MolmoAct2-FAST-Tokenizer",
                trust_remote_code=True,
            )

        logger.info("MolmoAct2 model loaded successfully")

    def to(self, device: torch.device | str) -> "MolmoAct2Policy":
        """Move policy to specified device."""
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def eval(self) -> "MolmoAct2Policy":
        """Set policy to evaluation mode."""
        super().eval()
        if self._model is not None:
            self._model.eval()
        return self

    def train(self, mode: bool = True) -> "MolmoAct2Policy":
        """Set policy to training mode."""
        super().train(mode)
        if self._model is not None:
            self._model.train(mode)
        return self

    def reset(self):
        """Reset the action queue."""
        self._action_queue.clear()

    def _extract_images(self, batch: dict[str, Any]) -> list[Image.Image]:
        """
        Extract and convert images from batch to PIL format.

        MolmoAct2 expects images in order: [agentview, wrist]

        Args:
            batch: Input batch containing image tensors

        Returns:
            List of PIL Images [agentview, wrist]
        """
        images = []

        # Map of possible key names for each camera
        # Order matters: agentview first, then wrist
        camera_key_mappings = [
            # Agentview camera
            [
                f"{OBS_IMAGES}.agentview",
                "observation.images.agentview",
                "observation.images.agentview_image",
                f"{OBS_IMAGES}.image",
                "observation.images.image",
            ],
            # Wrist camera
            [
                f"{OBS_IMAGES}.wrist",
                "observation.images.wrist",
                "observation.images.robot0_eye_in_hand",
                "observation.images.robot0_eye_in_hand_image",
                f"{OBS_IMAGES}.image2",
                "observation.images.image2",
            ],
        ]

        for camera_keys in camera_key_mappings:
            found = False
            for key in camera_keys:
                if key in batch:
                    img_tensor = batch[key]
                    # Handle batch dimension
                    if img_tensor.dim() == 4:
                        img_tensor = img_tensor[0]  # Take first in batch
                    # Convert to PIL with 180 degree flip for LIBERO
                    images.append(tensor_to_pil(img_tensor, flip_180=True))
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Could not find camera image. Tried keys: {camera_keys}. "
                    f"Available keys: {list(batch.keys())}"
                )

        return images

    def _extract_state(self, batch: dict[str, Any]) -> np.ndarray:
        """
        Extract robot state from batch.

        LIBERO state format (8 dims):
        - eef_pos (3): end-effector position
        - axis_angle (3): end-effector rotation as axis-angle
        - gripper_qpos (2): gripper joint positions

        Args:
            batch: Input batch containing state tensor

        Returns:
            Robot state as numpy array of shape (8,)
        """
        # Try different key formats
        possible_keys = [
            OBS_STATE,
            "observation.state",
            "state",
        ]

        state = None
        for key in possible_keys:
            if key in batch:
                state = batch[key]
                break

        if state is None:
            raise ValueError(
                f"State key not found in batch. Tried: {possible_keys}. "
                f"Available keys: {list(batch.keys())}"
            )

        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # Handle batch dimension
        if state.ndim == 2:
            state = state[0]

        # Validate state dimension
        if state.shape[0] != 8:
            logger.warning(
                f"Expected 8-dim state, got {state.shape[0]}-dim. "
                f"Padding or truncating to 8 dims."
            )
            if state.shape[0] < 8:
                state = np.pad(state, (0, 8 - state.shape[0]))
            else:
                state = state[:8]

        return state.astype(np.float32)

    def _extract_task(self, batch: dict[str, Any]) -> str:
        """
        Extract task description from batch.

        Args:
            batch: Input batch containing task description

        Returns:
            Task description string
        """
        if "task" in batch:
            task = batch["task"]
            if isinstance(task, list):
                task = task[0]
            return str(task)

        # Default task if not provided
        return "complete the task"

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any]) -> Tensor:
        """
        Select action based on current observation.

        This method implements action chunking: it predicts a chunk of actions
        and returns them one at a time. New predictions are made when the
        action queue is empty.

        Args:
            batch: Input batch containing:
                - observation.images.agentview: (B, C, H, W) tensor
                - observation.images.wrist: (B, C, H, W) tensor
                - observation.state: (B, 8) tensor
                - task: str or list[str]

        Returns:
            Action tensor of shape (B, action_dim)
        """
        # Check if we need to predict new actions
        if len(self._action_queue) == 0:
            self._predict_action_chunk(batch)

        # Get next action from queue
        action = self._action_queue.popleft()

        # Add batch dimension if needed
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return action

    def _predict_action_chunk(self, batch: dict[str, Any]):
        """
        Predict a chunk of actions using MolmoAct2.

        Args:
            batch: Input batch containing observations and task
        """
        # Extract inputs
        images = self._extract_images(batch)
        state = self._extract_state(batch)
        task = self._extract_task(batch)

        # Get device
        device = next(self._model.parameters()).device

        # Prepare inference kwargs
        inference_kwargs = {
            "processor": self._processor,
            "images": images,
            "task": task,
            "state": state,
            "norm_tag": self.config.norm_tag,
            "inference_action_mode": self.config.inference_action_mode,
            "enable_depth_reasoning": self.config.enable_depth_reasoning,
            "num_steps": self.config.chunk_size,
            "normalize_language": self.config.normalize_language,
            "enable_cuda_graph": self.config.enable_cuda_graph,
        }

        # Add action tokenizer for discrete mode
        if self.config.inference_action_mode == "discrete" and self._action_tokenizer is not None:
            inference_kwargs["action_tokenizer"] = self._action_tokenizer

        # Run inference
        if self.config.dtype == "bfloat16":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = self._model.predict_action(**inference_kwargs)
        else:
            with torch.inference_mode():
                output = self._model.predict_action(**inference_kwargs)

        # Get actions from output
        actions = output.actions  # Shape: (num_steps, action_dim)

        # Convert to tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        # Move to device
        actions = actions.to(device)

        # Add actions to queue
        for i in range(min(self.config.n_action_steps, len(actions))):
            self._action_queue.append(actions[i])

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """
        Forward pass for training (not implemented for inference-only policy).

        MolmoAct2 is loaded as a pretrained model and doesn't support training
        through this interface. Use the original MolmoAct2 training code for
        fine-tuning.

        Args:
            batch: Input batch

        Returns:
            Dictionary with loss (placeholder)

        Raises:
            NotImplementedError: Training is not supported
        """
        raise NotImplementedError(
            "MolmoAct2Policy does not support training through LeRobot. "
            "Use the original MolmoAct2 training code for fine-tuning."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: MolmoAct2Config | None = None,
        **kwargs,
    ) -> "MolmoAct2Policy":
        """
        Load a pretrained MolmoAct2 policy.

        For MolmoAct2, this creates a new policy with the specified config
        and loads the model from HuggingFace Hub.

        Args:
            pretrained_name_or_path: HuggingFace model ID or local path
            config: Optional configuration override
            **kwargs: Additional arguments

        Returns:
            Loaded MolmoAct2Policy instance
        """
        if config is None:
            config = MolmoAct2Config()

        # Update model ID if a custom path is provided
        if pretrained_name_or_path != config.hf_model_id:
            config.hf_model_id = str(pretrained_name_or_path)

        # Apply any kwargs to config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return cls(config)
