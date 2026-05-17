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

"""Processor pipeline for MolmoAct2 policy."""

from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register(name="molmoact2_language_normalizer")
class MolmoAct2LanguageNormalizer(ComplementaryDataProcessorStep):
    """
    Normalizes task description for MolmoAct2.

    This processor:
    1. Converts task description to lowercase
    2. Removes trailing punctuation
    """

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize the 'task' field.

        Args:
            complementary_data: Dictionary that may contain a 'task' key

        Returns:
            Dictionary with normalized 'task' field
        """
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        def normalize_task(t: str) -> str:
            # Convert to lowercase
            t = t.lower()
            # Remove trailing punctuation
            t = t.rstrip(".,!?;:")
            return t

        if isinstance(task, str):
            new_complementary_data["task"] = normalize_task(task)
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_complementary_data["task"] = [normalize_task(t) for t in task]

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step does not alter feature definitions."""
        return features


@ProcessorStepRegistry.register(name="molmoact2_image_processor")
class MolmoAct2ImageProcessor(ProcessorStep):
    """
    Ensures images are in the correct format for MolmoAct2.

    MolmoAct2 expects images as:
    - Shape: (B, C, H, W) or (C, H, W)
    - Values: [0, 1] float
    - Order: [agentview, wrist]
    """

    def __init__(self, camera_names: tuple[str, str] = ("agentview", "wrist")):
        """
        Initialize the image processor.

        Args:
            camera_names: Tuple of camera names in order (agentview, wrist)
        """
        self.camera_names = camera_names

    def __call__(
        self,
        data: dict[str, Any],
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]] | None = None,
        complementary_data: dict | None = None,
    ) -> tuple[dict[str, Any], dict[PipelineFeatureType, dict[str, PolicyFeature]], dict]:
        """
        Process images for MolmoAct2.

        Args:
            data: Input data dictionary containing image tensors
            features: Feature definitions
            complementary_data: Additional data

        Returns:
            Processed data, features, and complementary data
        """
        new_data = dict(data)

        # Process each camera image
        for camera_name in self.camera_names:
            # Try different key formats
            possible_keys = [
                f"observation.images.{camera_name}",
                f"observation.images.{camera_name}_image",
            ]

            for key in possible_keys:
                if key in new_data:
                    img = new_data[key]

                    # Ensure float type and [0, 1] range
                    if isinstance(img, torch.Tensor):
                        if img.dtype == torch.uint8:
                            img = img.float() / 255.0
                        elif img.max() > 1.0:
                            img = img / 255.0

                        new_data[key] = img

                    break

        return new_data, features or {}, complementary_data or {}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step does not alter feature definitions."""
        return features


def make_molmoact2_pre_post_processors(
    config: MolmoAct2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for MolmoAct2 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match expected format
    2. Processing images to correct format
    3. Normalizing language (if enabled)
    4. Adding batch dimension
    5. Moving data to device

    The post-processing pipeline handles the model's output by:
    1. Moving data to CPU

    Note: MolmoAct2 handles state/action normalization internally using
    norm_stats.json, so we don't apply external normalization.

    Args:
        config: The configuration object for MolmoAct2 policy
        dataset_stats: Dataset statistics (not used, MolmoAct2 has internal stats)

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        MolmoAct2ImageProcessor(camera_names=config.camera_names),
        AddBatchDimensionProcessorStep(),
    ]

    # Add language normalizer if enabled
    if config.normalize_language:
        input_steps.append(MolmoAct2LanguageNormalizer())

    # Add device processor
    input_steps.append(DeviceProcessorStep(device=config.device))

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
