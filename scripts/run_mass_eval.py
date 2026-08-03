#!/usr/bin/env python
"""
Mass evaluation script for LeRobot on ManiSkill Colosseum V2 tasks.

This script runs evaluation across all tasks and perturbation sets, with:
- Checkpoint resumption (skips already completed task+perturbation combinations)
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
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import PERTURBATION_SETS as _PERTURBATION_SETS
from mani_skill.envs.tasks.tabletop.colosseum_v2 import MAX_EPISODE_STEPS_BY_TASK

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


def run_lerobot_eval(
    policy_path: str,
    task: str,
    perturbation_set: str,
    batch_size: int,
    n_episodes: int,
    episode_length: int,
    output_dir: str,
    rename_map: dict = None,
) -> tuple[bool, int, int, str]:
    """
    Run lerobot-eval command and parse results from eval_info.json.

    Returns:
        tuple: (success, n_successful_episodes, n_total_episodes, message)
        message is "results_df" on success, "variation_factor_disabled" for soft skips,
        or an error string on failure.
    """
    # Build the output path for this specific evaluation
    eval_output_dir = Path(output_dir) / f"{task}_{perturbation_set}"

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
        f"--env.perturbation_set={perturbation_set}",
        f"--output_dir={eval_output_dir}",
    ]

    if "pi" in policy_path.lower():
        cmd.append("--policy.compile_model=false")   # Disable torch.compile at eval (training artifact)

    # Required for MolmoAct2 rollouts (see README single-task molmoact eval).
    if "molmoact" in policy_path.lower():
        cmd.append("--policy.inference_action_mode=continuous")

    if rename_map:
        cmd.append(f"--rename_map={json.dumps(rename_map)}")

    print(f"\n{'='*60}")
    print(f"Running: {task} with perturbation_set={perturbation_set}")
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
        # Stream output in real time while capturing it for error classification.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured_lines: list[str] = []
        assert process.stdout is not None

        def _stream_output() -> None:
            for line in process.stdout:
                print(line, end="", flush=True)
                captured_lines.append(line)

        reader = threading.Thread(target=_stream_output, daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=3600)  # 1 hour timeout per evaluation
        except subprocess.TimeoutExpired:
            process.kill()
            reader.join(timeout=2.0)
            return False, 0, n_episodes, "timeout"
        reader.join(timeout=2.0)
        captured = "".join(captured_lines)

        if returncode != 0:
            if _is_perturbation_factor_disabled(captured):
                print(f"Soft-skipping {task} + {perturbation_set}: perturbation factor disabled by env")
                return False, -1, n_episodes, "variation_factor_disabled"
            return False, 0, n_episodes, f"Command failed with return code {returncode}"

        # Read results from eval_info.json
        eval_info_path = eval_output_dir / "eval_info.json"
        assert eval_info_path.exists(), f"eval_info.json not found at {eval_info_path}"

        with open(eval_info_path, "r") as f:
            eval_info = json.load(f)

        assert "overall" in eval_info, (
            f"eval_info.json missing 'overall' key at {eval_info_path}. Got keys: {list(eval_info)}"
        )
        overall = eval_info["overall"]
        assert "pc_success" in overall, (
            f"eval_info.json missing 'overall.pc_success' at {eval_info_path}. Got keys: {list(overall)}"
        )
        pc_success = overall["pc_success"]  # percentage in [0, 100]

        n_successful = int(round(pc_success / 100.0 * n_episodes))
        print(f"Results from eval_info.json: pc_success={pc_success:.2f}%, n_successful={n_successful}/{n_episodes}")

        return True, n_successful, n_episodes, "results_df"
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
    args = parser.parse_args()

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
            if args.task_type == "bimanual" and ("table_" in ds_lower or ds_lower == "all"):
                eval_batch_size = max(1, args.batch_size // 4)
                print(
                    f"  Reducing batch size {args.batch_size} -> {eval_batch_size} "
                    f"for bimanual + {perturbation_set}"
                )

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
            success, n_successful, n_total, message = run_lerobot_eval(
                policy_path=args.policy_path,
                task=task,
                perturbation_set=perturbation_set,
                batch_size=eval_batch_size,
                n_episodes=args.n_episodes,
                episode_length=task_episode_length,
                output_dir=str(output_dir),
                rename_map=eval_rename_map,
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
                # Explicit subprocess failure (non-zero return code / timeout) — record and continue.
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
