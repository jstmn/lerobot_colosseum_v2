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

"""Configuration for MolmoAct2 policy."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


DEFAULT_IMAGE_SIZE = 384  # MolmoAct2 uses 384x384 images


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """
    Configuration class for MolmoAct2 policy.

    MolmoAct2 is a Vision-Language-Action model based on Molmo2-ER with a
    flow matching continuous action expert. It supports both continuous and
    discrete action prediction modes.

    Reference: https://huggingface.co/allenai/MolmoAct2-LIBERO
    """

    # Model identifier on HuggingFace Hub
    hf_model_id: str = "allenai/MolmoAct2-LIBERO"

    # Model precision
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    # Observation settings
    n_obs_steps: int = 1

    # Action settings
    chunk_size: int = 10  # Number of action steps to predict (num_steps in MolmoAct2)
    n_action_steps: int = 10  # Number of action steps to execute

    # State and action dimensions
    state_dim: int = 8  # LIBERO uses 8-dim state
    action_dim: int = 7  # LIBERO uses 7-dim action (6D EEF delta + gripper)

    # MolmoAct2 specific parameters
    inference_action_mode: str = "continuous"  # Options: "continuous", "discrete"
    norm_tag: str = "libero"  # Normalization tag for the model
    enable_depth_reasoning: bool = False  # Depth reasoning (disabled for LIBERO)
    enable_cuda_graph: bool = True  # Enable CUDA graph for faster inference
    normalize_language: bool = True  # Normalize task description (lowercase, remove punctuation)

    # Image settings
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    # Camera names (order matters: agentview first, then wrist)
    # These are the internal names used in MolmoAct2
    # The actual LIBERO keys may be different (e.g., robot0_eye_in_hand_image)
    camera_names: tuple[str, str] = ("agentview", "robot0_eye_in_hand")

    # Normalization - MolmoAct2 handles normalization internally
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,  # MolmoAct2 normalizes internally
            "ACTION": NormalizationMode.IDENTITY,  # MolmoAct2 normalizes internally
        }
    )

    # Device settings
    device: str | None = None

    # Trust remote code for HuggingFace model loading
    trust_remote_code: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.inference_action_mode not in ["continuous", "discrete"]:
            raise ValueError(f"Invalid inference_action_mode: {self.inference_action_mode}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Set up image features for both cameras
        for camera_name in self.camera_names:
            key = f"{OBS_IMAGES}.{camera_name}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.image_resolution),
                )

        # Set up state feature
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dim,),
            )

        # Set up action feature
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        """
        Return optimizer configuration.

        Note: MolmoAct2 is inference-only in LeRobot, so this returns a default config.
        """
        return AdamWConfig(
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """
        Return scheduler configuration.

        Note: MolmoAct2 is inference-only in LeRobot, so this returns None.
        """
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
