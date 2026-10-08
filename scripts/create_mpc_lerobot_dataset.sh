#!/usr/bin/env bash
# Collect N motion-planning demos from each MPC tabletop env, replay them into
# absolute EE pose (pd_ee_pose), merge, then convert the merged file to LeRobot.
#
# Modeled on ColosseumV2/scripts/data_generation/motionplanning_colosseum_v2_single_arm.sh
#
#   bash scripts/create_mpc_lerobot_dataset.sh
#   bash scripts/create_mpc_lerobot_dataset.sh -n 100
#   bash scripts/create_mpc_lerobot_dataset.sh -n 5 --num-procs 2
#   bash scripts/create_mpc_lerobot_dataset.sh --envs "PickCube-v2-wrist PushCube-v2"
#   bash scripts/create_mpc_lerobot_dataset.sh --data-dir /media/volume/mpc_a/lerobot_data
#
# Demos and the LeRobot export go under --data-dir (default: $LEROBOT_DATA_DIR).
# That directory must already exist.

set -euo pipefail
set +o histexpand

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLOSSEUM_DIR="${ROOT}/ColosseumV2"

PERTURBATION_SET=none
INCLUDED_CAMERAS="camera_center camera_left camera_wrist"
ENVS_ARG=""
N_TRAJ=100
NUM_PROCS=""
DATA_DIR=""
LEROBOT_DIR=""
IMAGE_SIZE="378x378"
REPO_ID=""
UPLOAD=1

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-traj)
            N_TRAJ="$2"
            shift 2
            ;;
        --num-procs)
            NUM_PROCS="$2"
            shift 2
            ;;
        --included-cameras)
            INCLUDED_CAMERAS="$2"
            shift 2
            ;;
        --envs)
            ENVS_ARG="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            LEROBOT_DIR="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --no-upload)
            UPLOAD=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-n N] [--num-procs P] [--included-cameras CAMS] [--envs ENV_IDS] [--data-dir DIR] [--output-dir DIR] [--image-size WxH] [--repo-id ID] [--no-upload]"
            echo "  -n, --num-traj          Successful demos per env (default: 100)"
            echo "  --num-procs             Parallel workers (default: cpu_count/2). Must be < N."
            echo "  --included-cameras      Space-separated camera uids (default: camera_center camera_left camera_wrist)"
            echo "  --envs                  Space-separated env ids (default: the 4 MPC tabletop tasks)"
            echo "  --data-dir              Root for ManiSkill demos + default LeRobot export"
            echo "                          (default: \$LEROBOT_DATA_DIR). Must already exist."
            echo "  --output-dir            LeRobot dataset directory (default: <data-dir>/mpc_lerobot_pd_ee_pose_<N>)"
            echo "  --image-size            convert_to_lerobot image size (default: 378x378, MolmoAct2 input)"
            echo "  --repo-id               Hub dataset id (default: jstm/mpc_lerobot_pd_ee_pose_<N>)"
            echo "  --no-upload             Skip pushing the LeRobot dataset to the Hub"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! [[ "${N_TRAJ}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-traj must be a positive integer, got: ${N_TRAJ}" >&2
    exit 1
fi

if [ ! -d "${COLOSSEUM_DIR}" ]; then
    echo "ColosseumV2 not found at ${COLOSSEUM_DIR}" >&2
    exit 1
fi

if [ -z "${DATA_DIR}" ]; then
    DATA_DIR="${LEROBOT_DATA_DIR:-}"
fi
if [ -z "${DATA_DIR}" ]; then
    echo "data dir required: pass --data-dir or set LEROBOT_DATA_DIR" >&2
    exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
    echo "data dir does not exist: ${DATA_DIR}" >&2
    exit 1
fi
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
DEMOS_DIR="${DATA_DIR}/demos"

if [ -z "${LEROBOT_DIR}" ]; then
    LEROBOT_DIR="${DATA_DIR}/mpc_lerobot_pd_ee_pose_${N_TRAJ}"
fi

ENVS=(
    "PickCube-v2-wrist"
    "PushCube-v2"
    "LiftPegUpright-v2"
    "PullCubeTool-v2"
)
# --envs overrides the default list above. Pass a space-separated set of env ids.
if [ -n "$ENVS_ARG" ]; then
    read -r -a ENVS <<< "$ENVS_ARG"
fi

if [ -z "${NUM_PROCS}" ]; then
    NUM_PROCS=$(python -c 'import os; print(max(2, int(os.cpu_count() // 2)))')
fi
if ! [[ "${NUM_PROCS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-procs must be a positive integer, got: ${NUM_PROCS}" >&2
    exit 1
fi
# run.py fans out and merges to <traj-name>.h5 only when num_procs < num_traj.
# Otherwise a single worker writes <traj-name>.0.h5.
if [ "${N_TRAJ}" -le "${NUM_PROCS}" ]; then
    echo "num-traj (${N_TRAJ}) must be greater than num-procs (${NUM_PROCS})." >&2
    echo "Pass a larger -n or a smaller --num-procs (e.g. -n 5 --num-procs 2)." >&2
    exit 1
fi
# Absolute EE pose (not delta). Matches Panda pd_ee_pose (xyz + euler + gripper).
TARGET_CONTROL_MODE=pd_ee_pose
OBS_MODE=rgb
REWARD_MODE=none

INCLUDED_CAMERAS_ARG=""
if [ -n "$INCLUDED_CAMERAS" ]; then
    INCLUDED_CAMERAS_ARG="--included-cameras ${INCLUDED_CAMERAS}"
fi

cd "${COLOSSEUM_DIR}"

for ENV_ID in "${ENVS[@]}"; do

    TRAJ_PATH=${DEMOS_DIR}/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.h5
    TRANSLATED_TRAJ_PATH=${DEMOS_DIR}/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.${OBS_MODE}.${TARGET_CONTROL_MODE}.physx_cpu.h5

    if [ -f "$TRANSLATED_TRAJ_PATH" ]; then
        echo -e "\033[1;32m Converted trajectory file $TRANSLATED_TRAJ_PATH already exists\033[0m"
        continue
    else
        echo -e "\033[1;33m Converted trajectory file $TRANSLATED_TRAJ_PATH does not exist\033[0m"
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "            ---  ENV_ID: $ENV_ID ---"
    echo ""

    if [ ! -f "$TRAJ_PATH" ]; then


        python mani_skill/examples/motionplanning/panda/run.py \
            --env-id ${ENV_ID} \
            --num-traj ${N_TRAJ} \
            --perturbation-set ${PERTURBATION_SET} \
            ${INCLUDED_CAMERAS_ARG} \
            --num-procs ${NUM_PROCS} \
            --obs-mode "rgb" \
            --reward-mode ${REWARD_MODE} \
            --random-seed \
            --only-count-success \
            --record-dir "${DEMOS_DIR}" \
            --traj-name "trajectory__pd_joint_pos__${N_TRAJ}"
    else
        echo -e "\033[1;32mTrajectory file $TRAJ_PATH already exists\033[0m"
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo "            ---  REPLAYING TRAJECTORY ---"
    echo "----------------------------------------------------------------"
    echo ""

    python mani_skill/trajectory/replay_trajectory.py \
        --traj-path ${TRAJ_PATH} \
        --sim-backend physx_cpu \
        --obs-mode ${OBS_MODE} \
        --target-control-mode ${TARGET_CONTROL_MODE} \
        --no-verbose \
        --reward-mode ${REWARD_MODE} \
        --save-traj \
        --no-save-video \
        --no-discard-timeout \
        --no-allow-failure \
        --no-vis \
        --no-use-env-states \
        --use-first-env-state \
        --no-record-rewards \
        --num-envs ${NUM_PROCS}


    N_DEMOS_0=$(h5ls -r ${TRAJ_PATH} | grep -c "/actions")
    N_DEMOS_TRANSLATED=$(h5ls -r ${TRANSLATED_TRAJ_PATH} | grep -c "/actions")
    echo -e "${ENV_ID}:\tNumber of demonstrations (original, translated): (${N_DEMOS_0}, ${N_DEMOS_TRANSLATED})"
done

# Combine per-env h5 files into one multitask trajectory.
INPUT_DIRS=""
for ENV in "${ENVS[@]}"; do
    INPUT_DIRS="${INPUT_DIRS}${DEMOS_DIR}/${ENV}/motionplanning "
done
INPUT_DIRS=$(echo "${INPUT_DIRS}" | xargs)
OUTPUT_PATH=${DEMOS_DIR}/trajectory__mpc__${TARGET_CONTROL_MODE}__${N_TRAJ}.h5

echo "Input directories: ${INPUT_DIRS}"
echo ""
if [ -f "${OUTPUT_PATH}" ]; then
    echo -e "\033[1;32m Merged trajectory ${OUTPUT_PATH} already exists\033[0m"
else
    python mani_skill/trajectory/merge_multitask_trajectories.py \
        --pattern "trajectory__pd_joint_pos__${N_TRAJ}.${OBS_MODE}.${TARGET_CONTROL_MODE}*.h5" \
        --input-dirs ${INPUT_DIRS} \
        --output-path ${OUTPUT_PATH}
fi


# Done
echo "----------------------------------------------------------------"
echo "----------------------------------------------------------------"
echo "  ---  MERGED TRAJECTORY: ${OUTPUT_PATH} ---  "
echo "Saved to: ${OUTPUT_PATH}"

N_FINAL_DEMOS=$(h5ls -r ${OUTPUT_PATH} | grep -c "/actions")
echo -e "Number of demonstrations in final h5 file: ${N_FINAL_DEMOS}"

echo ""
echo "----------------------------------------------------------------"
echo "            ---  CONVERTING TO LEROBOT ---"
echo "----------------------------------------------------------------"
echo ""

if [ -f "${LEROBOT_DIR}/meta/info.json" ]; then
    echo -e "\033[1;32m LeRobot dataset ${LEROBOT_DIR} already exists\033[0m"
else
    mkdir -p "${LEROBOT_DIR}"
    python mani_skill/trajectory/convert_to_lerobot.py \
        --traj-path "${OUTPUT_PATH}" \
        --output-dir "${LEROBOT_DIR}" \
        --robot-type panda \
        --image-size "${IMAGE_SIZE}"
fi

# MolmoAct2 QUANTILES normalization needs q01/q99 on action and state.
LEROBOT_DIR="${LEROBOT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["LEROBOT_DIR"])
stats_path = root / "meta" / "stats.json"
data_path = next((root / "data").rglob("*.parquet"))
stats = json.loads(stats_path.read_text())
df = pd.read_parquet(data_path)
changed = False
for key in ("action", "observation.state"):
    if key not in df.columns or key not in stats:
        continue
    if "q01" in stats[key] and "q99" in stats[key]:
        continue
    arr = np.stack(df[key].to_numpy())
    for q, name in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")):
        stats[key][name] = np.quantile(arr, q, axis=0).tolist()
    changed = True
if changed:
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Added quantile stats to {stats_path}")
else:
    print(f"Quantile stats already present in {stats_path}")
PY

echo "----------------------------------------------------------------"
echo "  ---  LEROBOT DATASET: ${LEROBOT_DIR} ---  "
echo "Merged ManiSkill file: ${OUTPUT_PATH}"
echo "LeRobot dataset: ${LEROBOT_DIR}"

if [ "${UPLOAD}" -eq 1 ]; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "            ---  UPLOADING TO HUGGING FACE HUB ---"
    echo "----------------------------------------------------------------"
    echo ""
    if [ -z "${REPO_ID}" ]; then
        REPO_ID="jstm/mpc_lerobot_pd_ee_pose_${N_TRAJ}"
    fi
    hf repo create "${REPO_ID}" --repo-type dataset --exist-ok
    hf upload "${REPO_ID}" "${LEROBOT_DIR}" . --repo-type dataset
    echo "Uploaded dataset: https://huggingface.co/datasets/${REPO_ID}"
fi
