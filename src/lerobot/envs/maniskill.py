# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
ManiSkill environment wrapper for LeRobot.

This module provides native ManiSkill environment support for LeRobot evaluation.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Optional

# Import ManiSkill to register environments
import mani_skill.envs


# =============================================================================
# Task Mappings (must match training data exactly!)
# =============================================================================

# Single Arm Tasks (16 tasks) - Colosseum v2 benchmark
# Format: env_id -> (task_index, language_description)
SINGLE_ARM_TASK_MAPPING = {
    "CookItemInPan-v1":         (0,  "Cook the item in the pan"),
    "HammerNail-v1":            (1,  "Hammer the nail into the surface"),
    "LiftPegUpright-v1":        (2,  "Lift the peg upright"),
    "OpenCabinet-v1":           (3,  "Open the cabinet door"),
    "OpenDrawer-v1":            (4,  "Open the drawer"),
    "PegInsertionSide-v2":      (5,  "Insert the peg from the side"),
    "PickDishFromRack-v1":      (6,  "Pick up the dish from the rack"),
    "PickSodaFromCabinet-v1":   (7,  "Pick up the soda can from the cabinet"),
    "PlaceBookInShelf-v1":      (8,  "Place the book on the shelf"),
    "PlaceCubeInDrawer-v1":     (9,  "Place the cube in the drawer"),
    "PlaceDishInRack-v1":       (10, "Place the dish in the rack"),
    "PlugCharger-v1":           (11, "Plug in the charger"),
    "RaiseCube-v1":             (12, "Raise the cube up from the table"),
    "RotateArrow-v1":           (13, "Rotate the arrow"),
    "ScoopBanana-v1":           (14, "Scoop the banana"),
    "StackCube-v1":             (15, "Stack one cube on top of another"),
}

# Bimanual Tasks (12 tasks) - Dual arm manipulation
# Format: env_id -> (task_index, language_description)
BIMANUAL_TASK_MAPPING = {
    "DualArmDrawerOpen-v1":     (0,  "Open the drawer with dual arms"),
    "DualArmDrawerPlace-v1":    (1,  "Place object in drawer with dual arms"),
    "DualArmLiftPot-v1":        (2,  "Lift the pot with dual arms"),
    "DualArmLiftTray-v1":       (3,  "Lift the tray with dual arms"),
    "DualArmPenCap-v1":         (4,  "Cap the pen with dual arms"),
    "DualArmPickBottle-v1":     (5,  "Pick up the bottle with dual arms"),
    "DualArmPickCube-v1":       (6,  "Pick up the cube with dual arms"),
    "DualArmPourPot-v1":        (7,  "Pour from the pot with dual arms"),
    "DualArmPushBox-v1":        (8,  "Push the box with dual arms"),
    "DualArmStack3Cube-v1":     (9,  "Stack three cubes with dual arms"),
    "DualArmStackCube-v1":      (10, "Stack cubes with dual arms"),
    "DualArmThreading-v1":      (11, "Thread the needle with dual arms"),
}

# Combined mapping for convenience
ALL_TASK_MAPPING = {**SINGLE_ARM_TASK_MAPPING, **BIMANUAL_TASK_MAPPING}


def get_task_info(env_id: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Get task_index and language description for an env_id.

    Args:
        env_id: Environment ID (e.g., "RaiseCube-v1", "DualArmLiftPot-v1")

    Returns:
        Tuple of (task_index, task_description) or (None, None) if not found
    """
    if env_id in ALL_TASK_MAPPING:
        return ALL_TASK_MAPPING[env_id]
    return None, None


def is_bimanual_task(env_id: str) -> bool:
    """Check if the task is a bimanual (dual-arm) task."""
    return env_id.startswith("DualArm") or env_id in BIMANUAL_TASK_MAPPING


class ManiSkillVectorEnvWrapper(gym.Wrapper):
    """
    Wrapper for vectorized ManiSkill environments.

    Handles batch observations and provides the interface expected by
    LeRobot's vectorized evaluation.
    """

    def __init__(
        self,
        env: gym.Env,
        task: str,
        task_description: str,
        max_episode_steps: int,
        camera_name: str = "base_camera",
        state_dim: int = 9,
    ):
        super().__init__(env)
        self._task = task
        self._task_description = task_description
        self._max_episode_steps_val = max_episode_steps
        self._camera_name = camera_name
        self._state_dim = state_dim

        # Store metadata
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        self.metadata['render_fps'] = 30

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        if hasattr(self.unwrapped, 'num_envs'):
            return self.unwrapped.num_envs
        return 1

    @property
    def envs(self):
        """
        List-like access to environments.
        LeRobot expects env.envs[i] to access individual environments.
        """
        return [self] * self.num_envs

    def _max_episode_steps(self) -> int:
        """Return maximum episode steps."""
        return self._max_episode_steps_val

    def task_description(self) -> str:
        """Return task description."""
        return self._task_description

    def task(self) -> str:
        """Alternative method for task description."""
        return self._task_description

    def call(self, method_name: str, *args, **kwargs):
        """
        Support env.call() interface for vectorized environments.
        LeRobot uses this to get task descriptions from all envs.
        """
        if method_name in ("task_description", "task"):
            return [self._task_description] * self.num_envs
        elif method_name == "_max_episode_steps":
            return [self._max_episode_steps_val] * self.num_envs
        elif hasattr(self.unwrapped, 'call'):
            return self.unwrapped.call(method_name, *args, **kwargs)
        else:
            method = getattr(self, method_name, None) or getattr(self.unwrapped, method_name)
            return [method(*args, **kwargs)] * self.num_envs

    def reset(self, **kwargs):
        """Reset and convert observation."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        """Step and convert observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure info has 'is_success' for each environment
        if 'is_success' not in info:
            if 'success' in info:
                info['is_success'] = info['success']
            else:
                info['is_success'] = np.zeros(self.num_envs, dtype=bool)

        # Convert tensors to numpy if needed
        if hasattr(info.get('is_success'), 'cpu'):
            info['is_success'] = info['is_success'].cpu().numpy()

        return self._convert_obs(obs), reward, terminated, truncated, info

    def render(self):
        """Render all environments for video."""
        return self.env.render()

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert batched ManiSkill observation to LeRobot format.

        ManiSkill format (vectorized):
            {
                'sensor_data': {'base_camera': {'rgb': (B, H, W, 3)}},
                'agent': {'qpos': (B, state_dim)}
            }

        LeRobot format:
            {
                'pixels': {'base_camera': (B, H, W, 3) uint8},
                'agent_pos': (B, state_dim) float32
            }
        """
        camera_name = self._camera_name
        rgb = obs['sensor_data'][camera_name]['rgb']

        # Convert to numpy
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Extract robot state
        qpos = obs['agent']['qpos']
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()

        state_dim = self._state_dim
        agent_pos = qpos[..., :state_dim].astype(np.float32)

        return {
            'pixels': {camera_name: rgb},
            'agent_pos': agent_pos,
        }


def create_maniskill_envs(
    task: str,
    n_envs: int,
    episode_length: int = 400,
    obs_mode: str = "rgb",
    control_mode: str = "pd_ee_delta_pose",
    render_mode: str = "rgb_array",
    sim_backend: str = "auto",
    camera_name: str = "base_camera",
    state_dim: int = 9,
    observation_height: int = 480,
    observation_width: int = 640,
    env_cls=None,
) -> Dict[str, Dict[int, gym.vector.VectorEnv]]:
    """
    Create ManiSkill environments for LeRobot evaluation.

    Args:
        task: Task name, can be "TaskName" or "TaskName::description"
        n_envs: Number of parallel environments
        episode_length: Maximum episode steps
        obs_mode: Observation mode (e.g., "rgb", "rgbd")
        control_mode: Control mode (e.g., "pd_ee_delta_pose")
        render_mode: Render mode for video recording
        sim_backend: Simulation backend ("auto", "gpu", "cpu")
        camera_name: Camera name for RGB observations
        state_dim: State dimension (qpos dimensions to use)
        observation_height: Camera image height (must match training data)
        observation_width: Camera image width (must match training data)
        env_cls: Not used, kept for API compatibility

    Returns:
        Dictionary mapping suite name to task environments:
        {"maniskill": {0: vec_env}}
    """
    # Parse task string: "TaskName" or "TaskName::description"
    if '::' in task:
        # User provided custom description
        task_name, task_description = task.split('::', 1)
        task_name = task_name.strip()
        task_description = task_description.strip()
    else:
        # Look up task description from mapping (MUST match training data!)
        task_name = task.strip()
        task_index, task_description = get_task_info(task_name)

        if task_description is None:
            # Unknown task - warn user
            print(f"  WARNING: Unknown task '{task_name}', not found in task mapping!")
            print(f"  Available single-arm tasks: {list(SINGLE_ARM_TASK_MAPPING.keys())}")
            print(f"  Available bimanual tasks: {list(BIMANUAL_TASK_MAPPING.keys())}")
            task_description = f"Complete the {task_name} task."
        else:
            print(f"  Found task in mapping: [{task_index}] \"{task_description}\"")

    # Create the ManiSkill environment with custom camera resolution
    # Use sensor_configs to override the default camera resolution
    env_kwargs = {
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "render_mode": render_mode,
        "sim_backend": sim_backend,
        "num_envs": n_envs,
        "max_episode_steps": episode_length,
        "sensor_configs": {
            camera_name: {
                "width": observation_width,
                "height": observation_height,
            }
        },
    }

    # Colosseum v2 tasks require distraction_set parameter
    # These are all tasks in SINGLE_ARM_TASK_MAPPING (Colosseum v2 benchmark)
    if task_name in SINGLE_ARM_TASK_MAPPING:
        # Use empty distraction_set to disable all distractions (match training data)
        env_kwargs["distraction_set"] = {}
        print(f"  Adding distraction_set={{}} for Colosseum v2 task")

    print(f"Creating ManiSkill environment: {task_name}")
    print(f"  n_envs: {n_envs}")
    print(f"  control_mode: {control_mode}")
    print(f"  obs_mode: {obs_mode}")
    print(f"  max_episode_steps: {episode_length}")
    print(f"  camera_resolution: {observation_width}x{observation_height}")
    print(f"  task_description: {task_description}")

    # Create base environment
    env = gym.make(task_name, **env_kwargs)

    # Wrap with LeRobot adapter
    wrapped_env = ManiSkillVectorEnvWrapper(
        env,
        task=task_name,
        task_description=task_description,
        max_episode_steps=episode_length,
        camera_name=camera_name,
        state_dim=state_dim,
    )

    # Return in LeRobot's expected format: {suite: {task_id: vec_env}}
    return {"maniskill": {0: wrapped_env}}
