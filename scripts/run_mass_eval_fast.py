#!/usr/bin/env python
"""
Mass evaluation script for LeRobot on ManiSkill Colosseum V2 tasks.

This script runs evaluation across all tasks and perturbation sets in one process, with:
- One-time policy and policy-processor initialization
- A fresh environment for every task+perturbation combination
- Checkpoint resumption (skips already completed task+perturbation combinations)
- Immediate CSV saving after each evaluation
- Error handling with failure summary at the end

Usage:
    python scripts/run_mass_eval_fast.py \
        --policy_path pythonsong/pi05_bimanual \
        --task_type bimanual \
        --batch_size 25 \
        --n_episodes 50 \
        --output_dir /path/to/outputs

    python scripts/run_mass_eval_fast.py \
        --policy_path pythonsong/pi05_single_arm \
        --task_type single_arm \
        --batch_size 25 \
        --n_episodes 50 \
        --output_dir /path/to/outputs
"""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import draccus
import pandas as pd
import torch
from mani_skill.envs.tasks.tabletop.colosseum_v2 import MAX_EPISODE_STEPS_BY_TASK
from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import PERTURBATION_SETS as _PERTURBATION_SETS

from lerobot.configs import parser as lerobot_parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.configs import ManiSkillEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

# ============================================================================
# Task Definitions
# ============================================================================

ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS = (
    "RaiseCube-v1",
    "PickSodaFromCabinet-v1",
    "PickDishFromRack-v1",
    "StackCubeColosseumV2-v1",
    "PlaceBookInShelf-v1",
    "PlaceDishInRack-v1",
    "LiftPegUprightColosseumV2-v1",
    "RotateArrow-v1",
    "PegInsertionSideColosseumV2-v1",
    "PlugChargerColosseumV2-v1",
    "HammerNail-v1",
    "ScoopBanana-v1",
    "OpenDrawer-v1",
    "OpenCabinet-v1",
    "PlaceCubeInDrawer-v1",
    "CookItemInPan-v1",
)

ALL_COLOSSEUM_V2_BIMANUAL_TASKS = (
    "DualArmPickCube-v1",
    "DualArmPickBottle-v1",
    "DualArmLiftPot-v1",
    "DualArmLiftTray-v1",
    "DualArmPushBox-v1",
    "DualArmPourPot-v1",
    "DualArmThreading-v1",
    "DualArmPenCap-v1",
    "DualArmDrawerPlace-v1",
    "DualArmDrawerOpen-v1",
    "DualArmStackCube-v1",
    "DualArmStack3Cube-v1",
)

# Keys from ManiSkill Colosseum V2 perturbation_set.PERTURBATION_SETS
PERTURBATION_SETS = tuple(_PERTURBATION_SETS.keys())

COMPLETED_MESSAGES = ("results_df", "variation_factor_disabled")

# CSV columns (matches ManiSkill eval_rgbd.py format)
CSV_COLUMNS = [
    "checkpoint_path",
    "pc_hostname",
    "now",
    "t_final",
    "duration_sec",
    "perturbation_set",
    "env_id",
    "control_mode",
    "include_depth",
    "num_eval_episodes",
    "max_episode_steps",
    "message",
    "num_sucessful_episodes",  # Note: keeping original spelling for compatibility
    "success_percent",
]

# Previous schema before t_final / duration_sec were added
_CSV_COLUMNS_WITHOUT_TIMING = [
    "checkpoint_path",
    "pc_hostname",
    "now",
    "perturbation_set",
    "env_id",
    "control_mode",
    "include_depth",
    "num_eval_episodes",
    "max_episode_steps",
    "message",
    "num_sucessful_episodes",
    "success_percent",
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_now_str() -> str:
    return datetime.now().strftime("%Y:%m:%d__%H:%M:%S")


def get_or_create_results_csv(csv_path: str) -> pd.DataFrame:
    """Load existing CSV or create a new one with proper columns.

    Migrates the previous schema (missing t_final / duration_sec) in place.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        current_columns = df.columns.tolist()

        if current_columns == CSV_COLUMNS:
            return df

        if current_columns == _CSV_COLUMNS_WITHOUT_TIMING:
            now_idx = df.columns.get_loc("now")
            df.insert(now_idx + 1, "t_final", "final-time-not-set")
            df.insert(now_idx + 2, "duration_sec", -1)
            df.to_csv(csv_path, index=False)
            print(f"Migrated results CSV to include t_final/duration_sec: {csv_path}")
            return df

        raise ValueError(
            f"CSV columns mismatch and not recognized as a migratable format!\n"
            f"Expected: {CSV_COLUMNS}\n"
            f"Got: {current_columns}"
        )
    else:
        # Create new empty DataFrame with columns
        df = pd.DataFrame(columns=CSV_COLUMNS)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Created new results CSV: {csv_path}")
        return df


def check_if_completed(df: pd.DataFrame, task: str, perturbation_set: str) -> bool:
    """Check if a task+perturbation_set combination has already been evaluated."""
    result_found = df[
        (df["env_id"] == task) &
        (df["perturbation_set"].str.upper() == perturbation_set.upper()) &
        (df["message"].isin(COMPLETED_MESSAGES))
    ]
    return len(result_found) > 0


def _is_perturbation_factor_disabled(output: str) -> bool:
    return (
        "PerturbationFactorDisabledError" in output
        or "is disabled by env" in output
        or "Variation factor disabled error" in output
    )


def save_placeholder_row(
    csv_path: str,
    checkpoint_path: str,
    pc_hostname: str,
    now: str,
    task: str,
    perturbation_set: str,
    control_mode: str,
    include_depth: bool,
    n_episodes: int,
    episode_length: int,
) -> None:
    """Save a placeholder row to indicate evaluation is in progress."""
    df = pd.read_csv(csv_path)
    row = {
        "checkpoint_path": checkpoint_path,
        "pc_hostname": pc_hostname,
        "now": now,
        "t_final": "final-time-not-set",
        "duration_sec": -1,
        "perturbation_set": perturbation_set.lower(),
        "env_id": task,
        "control_mode": control_mode,
        "include_depth": include_depth,
        "num_eval_episodes": n_episodes,
        "max_episode_steps": episode_length,
        "message": "placeholder",
        "num_sucessful_episodes": -1,
        "success_percent": -1,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def save_result_row(
    csv_path: str,
    checkpoint_path: str,
    pc_hostname: str,
    now: str,
    t_final: str,
    duration_sec: float,
    task: str,
    perturbation_set: str,
    control_mode: str,
    include_depth: bool,
    n_episodes: int,
    episode_length: int,
    message: str,
    num_successful: int,
    success_percent: float,
) -> None:
    """Save a result row after evaluation completes."""
    df = pd.read_csv(csv_path)
    row = {
        "checkpoint_path": checkpoint_path,
        "pc_hostname": pc_hostname,
        "now": now,
        "t_final": t_final,
        "duration_sec": duration_sec,
        "perturbation_set": perturbation_set.lower(),
        "env_id": task,
        "control_mode": control_mode,
        "include_depth": include_depth,
        "num_eval_episodes": n_episodes,
        "max_episode_steps": episode_length,
        "message": message,
        "num_sucessful_episodes": num_successful,
        "success_percent": f"{success_percent:.5f}" if success_percent >= 0 else success_percent,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


def make_maniskill_config(
    task: str,
    perturbation_set: str,
    episode_length: int,
    control_mode: str,
) -> ManiSkillEnv:
    """Build the configuration used to create one evaluation environment."""
    return ManiSkillEnv(
        task=task,
        episode_length=episode_length,
        control_mode=control_mode,
        perturbation_set=perturbation_set,
    )


def _normalize_config(value):
    """Convert config values into stable, directly comparable Python values."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _normalize_config(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(item) for item in value]
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    return value


def _config_differences(expected, actual, path: str) -> list[str]:
    """Return field-level differences between two normalized configurations."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(expected.keys() | actual.keys()):
            child_path = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child_path}: only in fast config ({actual[key]!r})")
            elif key not in actual:
                differences.append(f"{child_path}: missing from fast config (expected {expected[key]!r})")
            else:
                differences.extend(_config_differences(expected[key], actual[key], child_path))
        return differences
    if expected != actual:
        return [f"{path}: subprocess={expected!r}, fast={actual!r}"]
    return []


def validate_against_subprocess_config(
    *,
    policy_path: str,
    fast_policy_cfg: PreTrainedConfig,
    fast_env_cfg: ManiSkillEnv,
    task: str,
    perturbation_set: str,
    episode_length: int,
    batch_size: int,
    n_episodes: int,
    output_dir: str,
    rename_map: dict[str, str] | None,
) -> None:
    """Parse the original lerobot-eval CLI and compare its effective configs."""
    reference_args = [
        f"--policy.path={policy_path}",
        "--env.type=maniskill",
        f"--env.task={task}",
        f"--env.episode_length={episode_length}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        "--eval.max_episodes_rendered=0",
        "--trust_remote_code=true",
        f"--env.perturbation_set={perturbation_set}",
        f"--output_dir={output_dir}",
    ]
    policy_path_lower = policy_path.lower()
    if "pi" in policy_path_lower:
        reference_args.append("--policy.compile_model=false")
    if "molmoact" in policy_path_lower:
        reference_args.extend(
            [
                "--policy.inference_action_mode=continuous",
                "--policy.model_dtype=bfloat16",
                "--policy.enable_inference_cuda_graph=true",
                "--policy.device=cuda",
            ]
        )
    if rename_map:
        reference_args.append(f"--rename_map={json.dumps(rename_map)}")

    filtered_args = lerobot_parser.filter_path_args(
        EvalPipelineConfig.__get_path_fields__(),
        reference_args,
    )
    original_argv = sys.argv
    try:
        # EvalPipelineConfig.__post_init__ reads policy.path and policy overrides
        # from sys.argv, exactly as the lerobot-eval subprocess does.
        sys.argv = ["lerobot-eval", *reference_args]
        reference_cfg = draccus.parse(EvalPipelineConfig, args=filtered_args)
    finally:
        sys.argv = original_argv

    differences = [
        *_config_differences(
            _normalize_config(reference_cfg.env),
            _normalize_config(fast_env_cfg),
            "env",
        ),
        *_config_differences(
            _normalize_config(reference_cfg.policy),
            _normalize_config(fast_policy_cfg),
            "policy",
        ),
    ]
    if differences:
        formatted = "\n  - ".join(differences)
        raise AssertionError(f"Fast/subprocess config mismatch:\n  - {formatted}")
    print("CONFIG VALIDATION PASSED: policy and environment configs exactly match lerobot-eval.")


def load_policy_once(
    policy_path: str,
    env_cfg: ManiSkillEnv,
    rename_map: dict[str, str] | None = None,
):
    """Load the policy and its processors once for the entire mass-eval run."""
    policy_overrides: list[str] = []
    policy_path_lower = policy_path.lower()
    if "pi" in policy_path_lower:
        policy_overrides.append("--compile_model=false")
    if "molmoact" in policy_path_lower:
        policy_overrides.extend(
            [
                "--inference_action_mode=continuous",
                "--model_dtype=bfloat16",
                "--enable_inference_cuda_graph=true",
                "--device=cuda",
            ]
        )

    print(f"\nLoading policy once from {policy_path}...")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=policy_overrides)
    policy_cfg.pretrained_path = Path(policy_path)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": rename_map or {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    device = get_safe_torch_device(policy_cfg.device, log=True)
    print("Policy and processors loaded; they will be reused for all evaluations.")
    return policy_cfg, policy, preprocessor, postprocessor, device


def run_lerobot_eval(
    *,
    policy_cfg: PreTrainedConfig,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    env_cfg: ManiSkillEnv,
    task: str,
    perturbation_set: str,
    batch_size: int,
    n_episodes: int,
    output_dir: str,
) -> tuple[bool, int, int, str]:
    """Evaluate one fresh environment using the already-loaded policy."""
    eval_output_dir = Path(output_dir) / f"{task}_{perturbation_set}"
    print(f"\n{'='*60}")
    print(f"Running: {task} with perturbation_set={perturbation_set}")
    print(f"Creating {batch_size} fresh environment(s)")
    print(f"{'='*60}\n")

    envs = None
    try:
        # A normal lerobot-eval invocation starts with fresh processor state.
        # Reset stateful steps while keeping their loaded configuration in memory.
        preprocessor.reset()
        postprocessor.reset()
        envs = make_env(
            env_cfg,
            n_envs=batch_size,
            use_async_envs=True,
            trust_remote_code=True,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )

        autocast_context = (
            torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            eval_info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=n_episodes,
                max_episodes_rendered=0,
                return_episode_data=False,
                start_seed=1000,
                max_parallel_tasks=env_cfg.max_parallel_tasks,
            )

        eval_output_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_output_dir / "eval_info.json", "w") as f:
            json.dump(eval_info, f, indent=2)

        pc_success = eval_info["overall"]["pc_success"]
        n_successful = int(round(pc_success / 100.0 * n_episodes))
        print(f"Results from eval_info.json: pc_success={pc_success:.2f}%, n_successful={n_successful}/{n_episodes}")
        return True, n_successful, n_episodes, "results_df"
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        error_output = traceback.format_exc()
        print(error_output, flush=True)
        if _is_perturbation_factor_disabled(error_output):
            print(f"Soft-skipping {task} + {perturbation_set}: perturbation factor disabled by env")
            return False, -1, n_episodes, "variation_factor_disabled"
        return False, 0, n_episodes, f"{type(exc).__name__}: {exc}"
    finally:
        if envs is not None:
            close_envs(envs)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mass evaluation for LeRobot on ManiSkill Colosseum V2 tasks"
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to the policy checkpoint (HuggingFace repo or local path)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["single_arm", "bimanual"],
        help="Type of tasks to evaluate: single_arm or bimanual",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        required=True,
        help="Number of episodes per task+perturbation combination",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        help="Path to results CSV file",
    )
    parser.add_argument(
        "--include_depth",
        action="store_true",
        help="Include depth observations",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to evaluate (default: all tasks of the selected type)",
    )
    parser.add_argument(
        "--perturbation_sets",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific perturbation sets to evaluate (default: all {len(PERTURBATION_SETS)} sets from PERTURBATION_SETS)",
    )
    parser.add_argument(
        "--validate_config",
        action="store_true",
        help="One-time check that effective policy/env configs match run_mass_eval.py",
    )
    args = parser.parse_args()
    init_logging()
    register_third_party_plugins()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(1000)

    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_csv is None:
        args.results_csv = str(output_dir / f"results__{args.task_type.replace("_", "-")}__{args.policy_path.replace("/", "-")}.csv")
    print("Saving results to: ", args.results_csv)

    # Select task list based on task type
    if args.task_type == "bimanual":
        all_tasks = ALL_COLOSSEUM_V2_BIMANUAL_TASKS
        control_mode = "pd_joint_pos"
        eval_rename_map = None  # bimanual camera mapping TBD
        print("Evaluating BIMANUAL tasks")
    else:
        all_tasks = ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS
        control_mode = "pd_ee_delta_pose"
        # Rename env camera keys to match pi05 expected feature names
        eval_rename_map = None  # env already outputs external1/2_camera and hand_camera which match saved policy config
        print("Evaluating SINGLE ARM tasks")

    # Filter tasks if specified
    if args.tasks is not None:
        invalid_tasks = [t for t in args.tasks if t not in all_tasks]
        assert not invalid_tasks, (
            f"Invalid tasks: {invalid_tasks}. Available tasks: {all_tasks}"
        )
        tasks = tuple(args.tasks)
    else:
        tasks = all_tasks

    # Filter perturbation sets if specified
    if args.perturbation_sets is not None:
        invalid_ds = [d for d in args.perturbation_sets if d.upper() not in PERTURBATION_SETS]
        assert not invalid_ds, (
            f"Invalid perturbation sets: {invalid_ds}. Available: {PERTURBATION_SETS}"
        )
        perturbation_sets = tuple(d.upper() for d in args.perturbation_sets)
    else:
        perturbation_sets = PERTURBATION_SETS

    # Calculate total evaluations
    total_evals = len(tasks) * len(perturbation_sets)
    print(f"\nTotal evaluations: {len(tasks)} tasks x {len(perturbation_sets)} perturbation sets = {total_evals}")
    print(f"Episodes per evaluation: {args.n_episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results CSV: {args.results_csv}")

    # Initialize or load results CSV
    results_df = get_or_create_results_csv(args.results_csv)

    # Get hostname and timestamp for CSV records
    pc_hostname = socket.gethostname()
    now = get_now_str()

    # Track failures
    failed_tasks = []
    skipped_tasks = []
    completed_tasks = []

    # Remaining = task+perturbation combos still needing a run (excludes already-completed CSV rows).
    remaining = sum(
        1
        for task in tasks
        for perturbation_set in perturbation_sets
        if not check_if_completed(results_df, task, perturbation_set)
    )
    task_durations: list[float] = []
    print(f"Remaining evaluations to run: {remaining}")

    if remaining:
        # All tasks selected by one invocation have the same observation/action
        # feature layout, so any pending combination is sufficient for policy setup.
        first_task, first_perturbation_set = next(
            (task, perturbation_set)
            for task in tasks
            for perturbation_set in perturbation_sets
            if not check_if_completed(results_df, task, perturbation_set)
        )
        first_episode_length = MAX_EPISODE_STEPS_BY_TASK[first_task]
        first_env_cfg = make_maniskill_config(
            task=first_task,
            perturbation_set=first_perturbation_set,
            episode_length=first_episode_length,
            control_mode=control_mode,
        )

        policy_cfg, policy, preprocessor, postprocessor, device = load_policy_once(
            policy_path=args.policy_path,
            env_cfg=first_env_cfg,
            rename_map=eval_rename_map,
        )
        if args.validate_config:
            first_eval_batch_size = args.batch_size
            first_ds_lower = first_perturbation_set.lower()
            if args.task_type == "bimanual" and (
                "table_" in first_ds_lower or first_ds_lower == "all"
            ):
                first_eval_batch_size = max(1, args.batch_size // 4)
            validate_against_subprocess_config(
                policy_path=args.policy_path,
                fast_policy_cfg=policy_cfg,
                fast_env_cfg=first_env_cfg,
                task=first_task,
                perturbation_set=first_perturbation_set,
                episode_length=first_episode_length,
                batch_size=first_eval_batch_size,
                n_episodes=args.n_episodes,
                output_dir=str(output_dir / f"{first_task}_{first_perturbation_set}"),
                rename_map=eval_rename_map,
            )

    # Main evaluation loop
    eval_count = 0
    for task in tasks:
        for perturbation_set in perturbation_sets:
            eval_count += 1

            # Check if already completed
            if check_if_completed(results_df, task, perturbation_set):
                print(f"[{eval_count}/{total_evals}] Skipping {task} + {perturbation_set} (already completed)")
                skipped_tasks.append((task, perturbation_set))
                continue

            print(f"\n[{eval_count}/{total_evals}] Starting: {task} + {perturbation_set}")
            if task_durations:
                avg_sec = sum(task_durations) / len(task_durations)
                eta_hours = remaining * avg_sec / 3600.0
                print(
                    f"  ETA: {eta_hours:.2f} hours "
                    f"({remaining} remaining, avg {avg_sec / 60.0:.1f} min/task over {len(task_durations)} runs)"
                )
            else:
                print(f"  ETA: unknown ({remaining} remaining, no completed timings yet)")

            # Determine episode length for this task
            assert task in MAX_EPISODE_STEPS_BY_TASK, (
                f"{task} not in MAX_EPISODE_STEPS_BY_TASK. "
                f"Known tasks: {sorted(MAX_EPISODE_STEPS_BY_TASK)}"
            )
            task_episode_length = MAX_EPISODE_STEPS_BY_TASK[task]

            # Bimanual table / ALL sets use far more GPU memory (matches eval_rgbd.py).
            eval_batch_size = args.batch_size
            ds_lower = perturbation_set.lower()
            # if args.task_type == "bimanual" and ("table_" in ds_lower or ds_lower == "all"):
            #     eval_batch_size = max(1, args.batch_size // 4)
            #     print(
            #         f"  Reducing batch size {args.batch_size} -> {eval_batch_size} "
            #         f"for bimanual + {perturbation_set}"
            #     )

            # Save placeholder
            save_placeholder_row(
                csv_path=args.results_csv,
                checkpoint_path=args.policy_path,
                pc_hostname=pc_hostname,
                now=now,
                task=task,
                perturbation_set=perturbation_set,
                control_mode=control_mode,
                include_depth=args.include_depth,
                n_episodes=args.n_episodes,
                episode_length=task_episode_length,
            )

            t0 = time.time()
            env_cfg = make_maniskill_config(
                task=task,
                perturbation_set=perturbation_set,
                episode_length=task_episode_length,
                control_mode=control_mode,
            )
            success, n_successful, n_total, message = run_lerobot_eval(
                policy_cfg=policy_cfg,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                env_cfg=env_cfg,
                task=task,
                perturbation_set=perturbation_set,
                batch_size=eval_batch_size,
                n_episodes=args.n_episodes,
                output_dir=str(output_dir),
            )
            t_final = get_now_str()
            duration_sec = time.time() - t0
            task_durations.append(duration_sec)
            remaining -= 1

            if message == "variation_factor_disabled":
                save_result_row(
                    csv_path=args.results_csv,
                    checkpoint_path=args.policy_path,
                    pc_hostname=pc_hostname,
                    now=now,
                    t_final=t_final,
                    duration_sec=duration_sec,
                    task=task,
                    perturbation_set=perturbation_set,
                    control_mode=control_mode,
                    include_depth=args.include_depth,
                    n_episodes=args.n_episodes,
                    episode_length=task_episode_length,
                    message=message,
                    num_successful=-1,
                    success_percent=-1,
                )
                skipped_tasks.append((task, perturbation_set))
                print(f"Soft-skipped: {task} + {perturbation_set} (perturbation factor disabled)")
            elif success:
                assert n_total > 0, f"n_total must be > 0, got {n_total}"
                success_percent = 100.0 * n_successful / n_total
                save_result_row(
                    csv_path=args.results_csv,
                    checkpoint_path=args.policy_path,
                    pc_hostname=pc_hostname,
                    now=now,
                    t_final=t_final,
                    duration_sec=duration_sec,
                    task=task,
                    perturbation_set=perturbation_set,
                    control_mode=control_mode,
                    include_depth=args.include_depth,
                    n_episodes=args.n_episodes,
                    episode_length=task_episode_length,
                    message=message,
                    num_successful=n_successful,
                    success_percent=success_percent,
                )
                completed_tasks.append((task, perturbation_set, success_percent))
                print(f"Completed: {task} + {perturbation_set} -> {success_percent:.2f}% success")
            else:
                # Evaluation failure — record it and continue with the next combination.
                save_result_row(
                    csv_path=args.results_csv,
                    checkpoint_path=args.policy_path,
                    pc_hostname=pc_hostname,
                    now=now,
                    t_final=t_final,
                    duration_sec=duration_sec,
                    task=task,
                    perturbation_set=perturbation_set,
                    control_mode=control_mode,
                    include_depth=args.include_depth,
                    n_episodes=args.n_episodes,
                    episode_length=task_episode_length,
                    message=f"error: {message}",
                    num_successful=-1,
                    success_percent=-1,
                )
                failed_tasks.append((task, perturbation_set, message))
                print(f"FAILED: {task} + {perturbation_set} -> {message}")

            # Reload results for next iteration
            results_df = pd.read_csv(args.results_csv)

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(f"\nTotal tasks: {total_evals}")
    print(f"Completed: {len(completed_tasks)}")
    print(f"Skipped (already done): {len(skipped_tasks)}")
    print(f"Failed: {len(failed_tasks)}")

    if failed_tasks:
        print("\n" + "-" * 40)
        print("FAILED TASKS:")
        print("-" * 40)
        for task, perturbation_set, error in failed_tasks:
            print(f"  - {task} + {perturbation_set}: {error}")

    print(f"\nResults saved to: {args.results_csv}")


if __name__ == "__main__":
    main()
