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
from typing import Any, Dict

# Import ManiSkill to register environments
import mani_skill.envs


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
        env_cls: Not used, kept for API compatibility

    Returns:
        Dictionary mapping suite name to task environments:
        {"maniskill": {0: vec_env}}
    """
    # Parse task string: "TaskName" or "TaskName::description"
    if '::' in task:
        task_name, task_description = task.split('::', 1)
        task_name = task_name.strip()
        task_description = task_description.strip()
    else:
        task_name = task
        task_description = f"Complete the {task} task."

    # Create the ManiSkill environment
    env_kwargs = {
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "render_mode": render_mode,
        "sim_backend": sim_backend,
        "num_envs": n_envs,
        "max_episode_steps": episode_length,
    }

    print(f"Creating ManiSkill environment: {task_name}")
    print(f"  n_envs: {n_envs}")
    print(f"  control_mode: {control_mode}")
    print(f"  obs_mode: {obs_mode}")
    print(f"  max_episode_steps: {episode_length}")
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
