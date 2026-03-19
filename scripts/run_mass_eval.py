#!/usr/bin/env python
"""
Mass evaluation script for LeRobot on ManiSkill Colosseum V2 tasks.

This script runs evaluation across all tasks and distraction sets, with:
- Checkpoint resumption (skips already completed task+distraction combinations)
- Immediate CSV saving after each evaluation
- Error handling with failure summary at the end

Usage:
    python scripts/run_mass_eval.py \
        --policy_path pythonsong/pi05_bimanual \
        --task_type bimanual \
        --batch_size 25 \
        --n_episodes 50 \
        --output_dir /path/to/outputs

    python scripts/run_mass_eval.py \
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
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from lerobot.envs.maniskill import MAX_EPISODE_STEPS_BY_TASK


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

# All distraction sets from ManiSkill Colosseum V2
DISTRACTION_SETS = (
    "NONE",
    "ALL",
    "DISTRACTOR_OBJECT",
    "MO_COLOR",
    "MO_TEXTURE",
    "MO_SIZE",
    "RO_COLOR",
    "RO_TEXTURE",
    "RO_SIZE",
    "TABLE_COLOR",
    "TABLE_TEXTURE",
    "CAMERA_POSE",
    "LIGHT_COLOR",
    "BACKGROUND_TEXTURE",
    "BACKGROUND_COLOR",
    "LANGUAGE",
    "POSE_RANDOMIZATION",
)

# CSV columns (matches ManiSkill eval_rgbd.py format)
CSV_COLUMNS = [
    "checkpoint_path",
    "pc_hostname",
    "now",
    "distraction_set",
    "env_id",
    "control_mode",
    "include_depth",
    "num_eval_episodes",
    "max_episode_steps",
    "message",
    "num_sucessful_episodes",  # Note: keeping original spelling for compatibility
    "success_percent",
]


# ============================================================================
# GPU Monitoring
# ============================================================================

def gpu_monitor(interval: float = 60.0, stop_event: threading.Event = None) -> None:
    """Background thread to periodically print GPU usage via nvidia-smi.

    This monitors actual GPU utilization and memory usage across all processes,
    including subprocess (lerobot-eval).

    Args:
        interval: Time between prints in seconds (default: 60s = 1 minute)
        stop_event: Threading event to signal when to stop monitoring
    """
    while not stop_event.is_set():
        try:
            # Use nvidia-smi to get GPU stats (works for all processes)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpu_idx, util, mem_used, mem_total, temp = parts[:5]
                        print(f"[GPU {gpu_idx}] Util: {util}% | Memory: {mem_used}/{mem_total} MB | Temp: {temp}°C")
            else:
                print(f"[GPU Monitor] nvidia-smi error: {result.stderr}")
        except FileNotFoundError:
            print("[GPU Monitor] nvidia-smi not found, skipping GPU monitoring")
            return
        except Exception as e:
            print(f"[GPU Monitor] Error: {e}")

        # Wait for interval or until stop event is set
        stop_event.wait(timeout=interval)


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_results_csv(csv_path: str) -> pd.DataFrame:
    """Load existing CSV or create a new one with proper columns.

    Automatically migrates old CSV format (10 columns) to new format (12 columns):
    - Adds pc_hostname and now columns
    - Updates message field from "results" to "results_df"
    - Removes MO_MASS distraction set rows (no longer supported)
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        current_columns = df.columns.tolist()

        # Check if migration is needed (old format has 10 columns, new has 12)
        if current_columns != CSV_COLUMNS:
            # Old format columns for reference
            old_columns = [
                "checkpoint_path", "distraction_set", "env_id", "control_mode",
                "include_depth", "num_eval_episodes", "max_episode_steps",
                "message", "num_sucessful_episodes", "success_percent",
            ]

            if current_columns == old_columns:
                print(f"Detected old CSV format. Migrating to new format...")

                # 1. Add pc_hostname and now columns after checkpoint_path
                df.insert(1, "pc_hostname", "migrated")
                df.insert(2, "now", "migrated")

                # 2. Update message field: "results" -> "results_df"
                df["message"] = df["message"].replace("results", "results_df")

                # 3. Remove MO_MASS rows (no longer supported)
                rows_before = len(df)
                df = df[df["distraction_set"].str.upper() != "MO_MASS"]
                rows_removed = rows_before - len(df)
                if rows_removed > 0:
                    print(f"Removed {rows_removed} MO_MASS rows (no longer supported)")

                # Save migrated CSV
                df.to_csv(csv_path, index=False)
                print(f"Migration complete. Saved to {csv_path}")
            else:
                raise ValueError(
                    f"CSV columns mismatch and not recognized as old format!\n"
                    f"Expected: {CSV_COLUMNS}\n"
                    f"Got: {current_columns}"
                )

        return df
    else:
        # Create new empty DataFrame with columns
        df = pd.DataFrame(columns=CSV_COLUMNS)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Created new results CSV: {csv_path}")
        return df


def check_if_completed(df: pd.DataFrame, task: str, distraction_set: str) -> bool:
    """Check if a task+distraction_set combination has already been evaluated."""
    result_found = df[
        (df["env_id"] == task) &
        (df["distraction_set"].str.upper() == distraction_set.upper()) &
        (df["message"] == "results_df")  # Only consider completed results
    ]
    return len(result_found) > 0


def save_placeholder_row(
    csv_path: str,
    checkpoint_path: str,
    pc_hostname: str,
    now: str,
    task: str,
    distraction_set: str,
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
        "distraction_set": distraction_set.lower(),
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
    task: str,
    distraction_set: str,
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
        "distraction_set": distraction_set.lower(),
        "env_id": task,
        "control_mode": control_mode,
        "include_depth": include_depth,
        "num_eval_episodes": n_episodes,
        "max_episode_steps": episode_length,
        "message": message,
        "num_sucessful_episodes": num_successful,
        "success_percent": f"{success_percent:.2f}",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


def run_lerobot_eval(
    policy_path: str,
    task: str,
    distraction_set: str,
    batch_size: int,
    n_episodes: int,
    episode_length: int,
    output_dir: str,
    rename_map: dict = None,
) -> tuple[bool, int, int, str]:
    """
    Run lerobot-eval command and parse results from eval_info.json.

    Returns:
        tuple: (success, n_successful_episodes, n_total_episodes, error_message)
    """
    # Build the output path for this specific evaluation
    eval_output_dir = Path(output_dir) / f"{task}_{distraction_set}"

    # Build the command
    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=maniskill",
        f"--env.task={task}",
        f"--env.episode_length={episode_length}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        "--eval.max_episodes_rendered=0",  # Disable video rendering for speed
        "--trust_remote_code=true",
        f"--env.distraction_set={distraction_set}",
        f"--output_dir={eval_output_dir}",
    ]

    if rename_map:
        cmd.append(f"--rename_map={json.dumps(rename_map)}")

    print(f"\n{'='*60}")
    print(f"Running: {task} with distraction_set={distraction_set}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Start GPU monitoring thread
    stop_gpu_monitor = threading.Event()
    gpu_thread = threading.Thread(
        target=gpu_monitor,
        args=(60.0, stop_gpu_monitor),  # 60 seconds = 1 minute interval
        daemon=True,
    )
    gpu_thread.start()

    try:
        # Run the command with real-time output (no capture_output)
        result = subprocess.run(
            cmd,
            timeout=3600,  # 1 hour timeout per evaluation
        )

        if result.returncode != 0:
            return False, 0, n_episodes, f"Command failed with return code {result.returncode}"

        # Read results from eval_info.json
        eval_info_path = eval_output_dir / "eval_info.json"
        if not eval_info_path.exists():
            print(f"Warning: eval_info.json not found at {eval_info_path}")
            return False, 0, n_episodes, "eval_info.json not found"

        with open(eval_info_path, "r") as f:
            eval_info = json.load(f)

        # Extract success metrics from overall aggregated results
        # The structure is: {"overall": {"pc_success": ..., ...}, "maniskill": {...}}
        overall = eval_info.get("overall", {})
        pc_success = overall.get("pc_success", 0.0)  # This is already a percentage (0-100)

        # Calculate number of successful episodes
        n_successful = int(round(pc_success / 100.0 * n_episodes))

        print(f"Results from eval_info.json: pc_success={pc_success:.2f}%, n_successful={n_successful}/{n_episodes}")

        return True, n_successful, n_episodes, "results_df"

    except subprocess.TimeoutExpired:
        return False, 0, n_episodes, "timeout"
    except json.JSONDecodeError as e:
        return False, 0, n_episodes, f"JSON parse error: {e}"
    except Exception as e:
        return False, 0, n_episodes, str(e)
    finally:
        # Stop GPU monitoring thread
        stop_gpu_monitor.set()
        gpu_thread.join(timeout=2.0)


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
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Number of episodes per task+distraction combination (default: 50)",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
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
        default=None,
        help="Path to results CSV file (default: output_dir/results_{task_type}.csv)",
    )
    parser.add_argument(
        "--include_depth",
        action="store_true",
        default=False,
        help="Include depth observations (default: False)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to evaluate (default: all tasks of the selected type)",
    )
    parser.add_argument(
        "--distraction_sets",
        type=str,
        nargs="+",
        default=None,
        help="Specific distraction sets to evaluate (default: all 16 sets)",
    )
    parser.add_argument(
        "--max_episode_steps_from_lookup",
        action="store_true",
        default=False,
        help="Use per-task episode length from MAX_EPISODE_STEPS_BY_TASK (mean+4*std of training data). "
             "Falls back to --episode_length if task not in lookup.",
    )

    args = parser.parse_args()

    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_csv is None:
        args.results_csv = str(output_dir / f"results_{args.task_type}.csv")

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
        eval_rename_map = {
            "observation.images.external1_camera": "observation.images.base_0_rgb",
            "observation.images.external2_camera": "observation.images.left_wrist_0_rgb",
            "observation.images.hand_camera":      "observation.images.right_wrist_0_rgb",
        }
        print("Evaluating SINGLE ARM tasks")

    # Filter tasks if specified
    if args.tasks is not None:
        tasks = tuple(t for t in args.tasks if t in all_tasks)
        invalid_tasks = [t for t in args.tasks if t not in all_tasks]
        if invalid_tasks:
            print(f"Warning: Invalid tasks ignored: {invalid_tasks}")
        if not tasks:
            print(f"Error: No valid tasks specified. Available tasks: {all_tasks}")
            sys.exit(1)
    else:
        tasks = all_tasks

    # Filter distraction sets if specified
    if args.distraction_sets is not None:
        distraction_sets = tuple(d.upper() for d in args.distraction_sets if d.upper() in DISTRACTION_SETS)
        invalid_ds = [d for d in args.distraction_sets if d.upper() not in DISTRACTION_SETS]
        if invalid_ds:
            print(f"Warning: Invalid distraction sets ignored: {invalid_ds}")
        if not distraction_sets:
            print(f"Error: No valid distraction sets specified. Available: {DISTRACTION_SETS}")
            sys.exit(1)
    else:
        distraction_sets = DISTRACTION_SETS

    # Calculate total evaluations
    total_evals = len(tasks) * len(distraction_sets)
    print(f"\nTotal evaluations: {len(tasks)} tasks x {len(distraction_sets)} distraction sets = {total_evals}")
    print(f"Episodes per evaluation: {args.n_episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results CSV: {args.results_csv}")

    # Initialize or load results CSV
    results_df = get_or_create_results_csv(args.results_csv)

    # Get hostname and timestamp for CSV records
    pc_hostname = socket.gethostname()
    now = datetime.now().strftime("%Y:%m:%d__%H:%M:%S")

    # Track failures
    failed_tasks = []
    skipped_tasks = []
    completed_tasks = []

    # Main evaluation loop
    eval_count = 0
    for task in tasks:
        for distraction_set in distraction_sets:
            eval_count += 1

            # Check if already completed
            if check_if_completed(results_df, task, distraction_set):
                print(f"[{eval_count}/{total_evals}] Skipping {task} + {distraction_set} (already completed)")
                skipped_tasks.append((task, distraction_set))
                continue

            print(f"\n[{eval_count}/{total_evals}] Starting: {task} + {distraction_set}")

            # Determine episode length for this task
            if args.max_episode_steps_from_lookup:
                if task in MAX_EPISODE_STEPS_BY_TASK:
                    task_episode_length = MAX_EPISODE_STEPS_BY_TASK[task]
                    print(f"  max_episode_steps_from_lookup: {task} -> {task_episode_length} steps")
                else:
                    task_episode_length = args.episode_length
                    print(f"  Warning: {task} not in MAX_EPISODE_STEPS_BY_TASK, using --episode_length={task_episode_length}")
            else:
                task_episode_length = args.episode_length

            # Save placeholder
            save_placeholder_row(
                csv_path=args.results_csv,
                checkpoint_path=args.policy_path,
                pc_hostname=pc_hostname,
                now=now,
                task=task,
                distraction_set=distraction_set,
                control_mode=control_mode,
                include_depth=args.include_depth,
                n_episodes=args.n_episodes,
                episode_length=task_episode_length,
            )

            try:
                # Run evaluation
                success, n_successful, n_total, message = run_lerobot_eval(
                    policy_path=args.policy_path,
                    task=task,
                    distraction_set=distraction_set,
                    batch_size=args.batch_size,
                    n_episodes=args.n_episodes,
                    episode_length=task_episode_length,
                    output_dir=str(output_dir),
                    rename_map=eval_rename_map,
                )

                if success:
                    success_percent = 100.0 * n_successful / n_total if n_total > 0 else 0.0
                    save_result_row(
                        csv_path=args.results_csv,
                        checkpoint_path=args.policy_path,
                        pc_hostname=pc_hostname,
                        now=now,
                        task=task,
                        distraction_set=distraction_set,
                        control_mode=control_mode,
                        include_depth=args.include_depth,
                        n_episodes=args.n_episodes,
                        episode_length=args.episode_length,
                        message=message,
                        num_successful=n_successful,
                        success_percent=success_percent,
                    )
                    completed_tasks.append((task, distraction_set, success_percent))
                    print(f"Completed: {task} + {distraction_set} -> {success_percent:.2f}% success")
                else:
                    # Save error result
                    save_result_row(
                        csv_path=args.results_csv,
                        checkpoint_path=args.policy_path,
                        pc_hostname=pc_hostname,
                        now=now,
                        task=task,
                        distraction_set=distraction_set,
                        control_mode=control_mode,
                        include_depth=args.include_depth,
                        n_episodes=args.n_episodes,
                        episode_length=args.episode_length,
                        message=f"error: {message}",
                        num_successful=-1,
                        success_percent=-1,
                    )
                    failed_tasks.append((task, distraction_set, message))
                    print(f"FAILED: {task} + {distraction_set} -> {message}")

            except Exception as e:
                # Catch-all for unexpected errors
                error_msg = str(e)
                save_result_row(
                    csv_path=args.results_csv,
                    checkpoint_path=args.policy_path,
                    pc_hostname=pc_hostname,
                    now=now,
                    task=task,
                    distraction_set=distraction_set,
                    control_mode=control_mode,
                    include_depth=args.include_depth,
                    n_episodes=args.n_episodes,
                    episode_length=args.episode_length,
                    message=f"exception: {error_msg}",
                    num_successful=-1,
                    success_percent=-1,
                )
                failed_tasks.append((task, distraction_set, error_msg))
                print(f"EXCEPTION: {task} + {distraction_set} -> {error_msg}")
                continue

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
        for task, distraction_set, error in failed_tasks:
            print(f"  - {task} + {distraction_set}: {error}")

    print(f"\nResults saved to: {args.results_csv}")


if __name__ == "__main__":
    main()
