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
import os
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
    "PegInsertionSide-v1":      (5,  "Insert the peg from the side"),
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
# Note: All dual-arm tasks are v1 in ManiSkill 3.0.0b22
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
    "DualArmStackCube-v1":      (9,  "Stack cubes with dual arms"),
    "DualArmStack3Cube-v1":     (10, "Stack three cubes with dual arms"),
    "DualArmThreading-v1":      (11, "Thread with dual arms"),
}

# Combined mapping for convenience
ALL_TASK_MAPPING = {**SINGLE_ARM_TASK_MAPPING, **BIMANUAL_TASK_MAPPING}

# Per-task max episode steps: mean + 4*std of training episode lengths
# Source: jstmn/ManiSkill examples/baselines/act_clip/eval_rgbd.py
MAX_EPISODE_STEPS_BY_TASK = {
    "PickSodaFromCabinet-v1":       int(193   + 4 * 2.5),
    "PickDishFromRack-v1":          int(119   + 4 * 9.26),
    "StackCubeColosseumV2-v1":      int(107   + 4 * 8.97),
    "PlaceDishInRack-v1":           int(251   + 4 * 19.07),
    "LiftPegUprightColosseumV2-v1": int(198   + 4 * 6.51),
    "RotateArrow-v1":               int(328   + 4 * 6.94),
    "PegInsertionSideColosseumV2-v1": int(151 + 4 * 28.91),
    "PlugChargerColosseumV2-v1":    int(179   + 4 * 13.83),
    "HammerNail-v1":                int(225   + 4 * 5.6),
    "ScoopBanana-v1":               int(242   + 4 * 20.05),
    "OpenDrawer-v1":                int(118   + 4 * 3.72),
    "OpenCabinet-v1":               int(475   + 4 * 4.15),
    "PlaceCubeInDrawer-v1":         int(333   + 4 * 13.62),
    "PlaceBookInShelf-v1":          int(182   + 4 * 8.54),
    "CookItemInPan-v1":             int(473   + 4 * 15.12),
    "RaiseCube-v1":                 int(78    + 4 * 3.55),
    "DualArmPickCube-v1":           int(201.1 + 4 * 3.2),
    "DualArmPickBottle-v1":         int(130.72 + 4 * 6.23),
    "DualArmLiftPot-v1":            int(98.06 + 4 * 6.94),
    "DualArmLiftTray-v1":           int(104.72 + 4 * 4.43),
    "DualArmPushBox-v1":            int(93.04 + 4 * 9.43),
    "DualArmPourPot-v1":            int(200.72 + 4 * 3.5),
    "DualArmThreading-v1":          int(164.97 + 4 * 6.92),
    "DualArmPenCap-v1":             int(186.1 + 4 * 11.54),
    "DualArmDrawerPlace-v1":        int(186.35 + 4 * 4.0),
    "DualArmDrawerOpen-v1":         int(81.0  + 4 * 9.4),
    "DualArmStackCube-v1":          int(137.03 + 4 * 7.27),
    "DualArmStack3Cube-v1":         int(242.08 + 4 * 10.5),
}

# Tasks that support distraction_set parameter (true Colosseum v2 tasks)
# These are tasks defined in mani_skill/envs/tasks/tabletop/colosseum_v2/
# Some tasks like StackCube-v1, LiftPegUpright-v1, PegInsertionSide-v1, PlugCharger-v1
# are original versions that do NOT support distraction_set
COLOSSEUM_V2_TASKS = {
    # Single arm Colosseum v2 tasks
    "RaiseCube-v1",
    "PickSodaFromCabinet-v1",
    "PickDishFromRack-v1",
    "PlaceBookInShelf-v1",
    "PlaceDishInRack-v1",
    "RotateArrow-v1",
    "HammerNail-v1",
    "ScoopBanana-v1",
    "CookItemInPan-v1",
    "OpenDrawer-v1",
    "OpenCabinet-v1",
    "PlaceCubeInDrawer-v1",
    # ColosseumV2 specific versions (different from original)
    "StackCubeColosseumV2-v1",
    "LiftPegUprightColosseumV2-v1",
    "PegInsertionSideColosseumV2-v1",
    "PlugChargerColosseumV2-v1",
    # All bimanual tasks are Colosseum v2
    "DualArmDrawerOpen-v1",
    "DualArmDrawerPlace-v1",
    "DualArmLiftPot-v1",
    "DualArmLiftTray-v1",
    "DualArmPenCap-v1",
    "DualArmPickBottle-v1",
    "DualArmPickCube-v1",
    "DualArmPourPot-v1",
    "DualArmPushBox-v1",
    "DualArmStackCube-v1",
    "DualArmStack3Cube-v1",
    "DualArmThreading-v1",
}


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
        output_camera_name: str = None,
        state_dim: int = 9,
        camera_names: list = None,
    ):
        super().__init__(env)
        self._task = task
        self._task_description = task_description
        self._max_episode_steps_val = max_episode_steps
        self._state_dim = state_dim
        # Multi-camera support: list of (env_cam_name, output_cam_name) tuples
        # Falls back to single camera if not provided
        if camera_names is not None:
            self._camera_names = camera_names  # list of (env_name, output_name)
        else:
            _out = output_camera_name if output_camera_name else camera_name
            self._camera_names = [(camera_name, _out)]
        # Keep for backward compat
        self._camera_name = self._camera_names[0][0]
        self._output_camera_name = self._camera_names[0][1]
        self._debug = bool(int(os.getenv("LEROBOT_MANISKILL_DEBUG", "0")))

        # Store metadata
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        self.metadata['render_fps'] = 30

        # Language instructions: sampled once at env creation, fixed for all episodes
        self._current_task_descriptions = [task_description] * self.num_envs
        self._randomize_language_instructions()

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
            return list(self._current_task_descriptions)
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

    def _randomize_language_instructions(self):
        """Randomize language instructions per episode if LANGUAGE distraction set is active."""
        try:
            base_env = self.env.unwrapped
            if hasattr(base_env, "update_language_instructions"):
                updated = base_env.update_language_instructions(self._current_task_descriptions)
                if updated is not None:
                    self._current_task_descriptions = updated
        except Exception:
            pass

    def step(self, action):
        """Step and convert observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert reward to numpy if needed
        if hasattr(reward, 'cpu'):
            reward = reward.cpu().numpy()

        # Convert terminated/truncated to numpy if needed
        if hasattr(terminated, 'cpu'):
            terminated = terminated.cpu().numpy()
        if hasattr(truncated, 'cpu'):
            truncated = truncated.cpu().numpy()

        # Ensure info has 'is_success' for each environment
        if 'is_success' not in info:
            if 'success' in info:
                info['is_success'] = info['success']
            else:
                info['is_success'] = np.zeros(self.num_envs, dtype=bool)

        # Convert tensors to numpy if needed
        if hasattr(info.get('is_success'), 'cpu'):
            info['is_success'] = info['is_success'].cpu().numpy()

        # Terminate immediately on success to align with eval expectations
        if isinstance(info.get('is_success'), np.ndarray):
            terminated = np.logical_or(terminated, info['is_success'])

        # Provide final_info so LeRobot can read success at episode end
        if isinstance(terminated, np.ndarray) and isinstance(truncated, np.ndarray):
            if np.any(terminated | truncated):
                info["final_info"] = {"is_success": info["is_success"]}

        if self._debug:
            print(
                "[ManiSkillWrapper.step]",
                f"reward_mode={getattr(self.unwrapped, 'reward_mode', None)}",
                f"reward_type={type(reward)}",
                f"terminated_any={np.any(terminated) if isinstance(terminated, np.ndarray) else terminated}",
                f"truncated_any={np.any(truncated) if isinstance(truncated, np.ndarray) else truncated}",
                f"success_any={np.any(info['is_success'])}",
            )

        return self._convert_obs(obs), reward, terminated, truncated, info

    def render(self):
        """Render all environments for video."""
        frame = self.env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        if self._debug:
            if frame is None:
                print("[ManiSkillWrapper.render] frame=None")
            else:
                print("[ManiSkillWrapper.render] frame shape:", getattr(frame, "shape", None))
        return frame

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert batched ManiSkill observation to LeRobot format.

        ManiSkill format (vectorized):
            {
                'sensor_data': {'external1_camera': {'rgb': (B, H, W, 3)}, ...},
                'agent': {'qpos': (B, state_dim)}
            }

        LeRobot format:
            {
                'pixels': {'external1_camera': (B, H, W, 3) uint8, ...},
                'agent_pos': (B, state_dim) float32
            }
        """
        # Extract robot state
        qpos = obs['agent']['qpos']
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()
        agent_pos = qpos[..., :self._state_dim].astype(np.float32)

        # Read all configured cameras
        pixels = {}
        for env_cam_name, out_cam_name in self._camera_names:
            rgb = obs['sensor_data'][env_cam_name]['rgb']
            if hasattr(rgb, 'cpu'):
                rgb = rgb.cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            pixels[out_cam_name] = rgb

        return {
            'pixels': pixels,
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
    distraction_set: str = "NONE",
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
    # Use global sensor_configs to apply to ALL cameras (not camera-specific)
    env_kwargs = {
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "render_mode": render_mode,
        "sim_backend": sim_backend,
        # Ensure offscreen render cameras are configured for video capture
        "human_render_camera_configs": {
            "width": observation_width,
            "height": observation_height,
        },
        "reward_mode": "sparse",
        "num_envs": n_envs,
        "max_episode_steps": episode_length,
        "sensor_configs": {
            # Global config applied to all cameras
            "width": observation_width,
            "height": observation_height,
        },
    }

    # Camera configuration: list of (env_cam_name, output_cam_name) tuples
    # Single arm Colosseum v2: 3 cameras matching training data
    # Bimanual: TBD (fall back to single base_camera for now)
    task_camera_names = None  # will be set below

    # Only true Colosseum v2 tasks support distraction_set parameter
    # Some tasks like StackCube-v1, LiftPegUpright-v1, PegInsertionSide-v1, PlugCharger-v1
    # are original versions that do NOT support distraction_set
    if task_name in COLOSSEUM_V2_TASKS:
        from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DISTRACTION_SETS
        ds_key = distraction_set.upper()
        if ds_key not in DISTRACTION_SETS:
            raise ValueError(
                f"Unknown distraction_set '{distraction_set}'. "
                f"Valid options: {list(DISTRACTION_SETS.keys())}"
            )
        env_kwargs["distraction_set"] = DISTRACTION_SETS[ds_key]
        env_kwargs["_env_id"] = task_name
        print(f"  Adding distraction_set={ds_key} for Colosseum v2 task")

    if not is_bimanual_task(task_name):
        # Single arm: use all 3 cameras that match training data
        task_camera_names = [
            ("external1_camera", "external1_camera"),
            ("external2_camera", "external2_camera"),
            ("hand_camera",      "hand_camera"),
        ]
        print(f"  Single arm cameras: {[c[0] for c in task_camera_names]}")
    else:
        # Bimanual: fall back to single base_camera
        task_camera_names = [("base_camera", "base_camera")]
        print(f"  Bimanual task: using base_camera")

    # Auto-detect state_dim and control_mode based on task type
    # Bimanual tasks:
    #   - state_dim = 9 * 2 = 18 (two arms)
    #   - control_mode = pd_joint_pos (absolute joint positions, action_dim = 8 * 2 = 16)
    #   - Training data uses absolute positions, NOT delta!
    # Single arm tasks:
    #   - state_dim = 9
    #   - control_mode = pd_ee_delta_pose (action_dim = 7)
    if is_bimanual_task(task_name):
        actual_state_dim = 18
        actual_control_mode = "pd_joint_pos"  # Training data uses absolute joint positions
        env_kwargs["control_mode"] = actual_control_mode
        print(f"  Bimanual task detected: state_dim={actual_state_dim}, control_mode={actual_control_mode}")
    else:
        actual_state_dim = 9
        actual_control_mode = control_mode  # Use the provided control_mode
        print(f"  Single arm task: state_dim={actual_state_dim}, control_mode={actual_control_mode}")

    print(f"Creating ManiSkill environment: {task_name}")
    print(f"  n_envs: {n_envs}")
    print(f"  control_mode: {actual_control_mode}")
    print(f"  obs_mode: {obs_mode}")
    print(f"  max_episode_steps: {episode_length}")
    print(f"  camera_resolution: {observation_width}x{observation_height}")
    print(f"  cameras: {task_camera_names}")
    print(f"  state_dim: {actual_state_dim}")
    print(f"  task_description: {task_description}")
    print(f"  env_kwargs: {env_kwargs}")
    import sys
    sys.stdout.flush()  # Force flush to ensure logs are printed immediately

    # Create base environment
    print("[Step 1/4] Calling gym.make()...")
    sys.stdout.flush()
    env = gym.make(task_name, **env_kwargs)
    # gym.make() may wrap the env in a TimeLimitWrapper; unwrap to get the raw ManiSkill env
    while not hasattr(env, 'num_envs') and hasattr(env, 'env'):
        env = env.env
    print(f"[Step 2/4] gym.make() completed. env.num_envs = {env.num_envs}")
    sys.stdout.flush()

    # Wrap with LeRobot adapter
    print("[Step 3/4] Creating ManiSkillVectorEnvWrapper...")
    sys.stdout.flush()
    wrapped_env = ManiSkillVectorEnvWrapper(
        env,
        task=task_name,
        task_description=task_description,
        max_episode_steps=episode_length,
        state_dim=actual_state_dim,
        camera_names=task_camera_names,
    )
    print("[Step 4/4] ManiSkillVectorEnvWrapper created successfully.")
    sys.stdout.flush()

    # Return in LeRobot's expected format: {suite: {task_id: vec_env}}
    return {"maniskill": {0: wrapped_env}}
