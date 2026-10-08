#!/usr/bin/env bash
# Sweep LeRobot training checkpoints on MPC tabletop tasks and write a markdown table.
#
#   bash scripts/eval_mpc_checkpoints.sh \
#     --train-dir outputs/train/molmoact2_mpc_pd_ee_pose_200 \
#     --batch-size 5
#
# Defaults match the MolmoAct2 MPC pd_ee_pose finetune: 50 episodes,
# batch_size=5, cameras camera_{center,left,wrist} at 378x378, control_mode=pd_ee_pose.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_DIR=""
RESULTS_MD="${ROOT}/mpc_vla_results.md"
N_EPISODES=50
BATCH_SIZE=5
OBS_SIZE=378
DEVICE=cuda
INFERENCE_ACTION_MODE=continuous

# env_id|episode_length|prompt
TASKS=(
  "PickCube-v2-wrist|100|lift the red cube"
  "PushCube-v2|80|push the cube to the goal"
  "LiftPegUpright-v2|225|lift the peg upright"
  "PullCubeTool-v2|350|grasp the red L-shaped hook and use it to pull the blue cube closer to the robot"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --results-md) RESULTS_MD="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --obs-size) OBS_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --inference-action-mode) INFERENCE_ACTION_MODE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --train-dir DIR [--batch-size B] [--results-md PATH] [--n-episodes N] [--obs-size S]"
      echo "  --train-dir     Training run dir containing checkpoints/ (required)"
      echo "  --batch-size    Parallel eval envs (default: 5)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TRAIN_DIR}" ]]; then
  echo "--train-dir is required" >&2
  exit 1
fi
if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 1
fi
if [[ "${TRAIN_DIR}" != /* ]]; then
  TRAIN_DIR="${ROOT}/${TRAIN_DIR}"
fi
CKPT_ROOT="${TRAIN_DIR}/checkpoints"
if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "No checkpoints dir at ${CKPT_ROOT}" >&2
  exit 1
fi
if ! command -v lerobot-eval >/dev/null 2>&1; then
  echo "lerobot-eval not found. Activate the lerobot_cv2 conda env first." >&2
  exit 1
fi

mapfile -t STEPS < <(
  find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n
)
if [[ ${#STEPS[@]} -eq 0 ]]; then
  echo "No numeric checkpoint dirs under ${CKPT_ROOT}" >&2
  exit 1
fi

EVAL_ROOT="${TRAIN_DIR}/checkpoint_evals"
mkdir -p "${EVAL_ROOT}"

# results[step|env_id] = pc_success
declare -A RESULTS
declare -A STATUS

run_one() {
  local step="$1"
  local env_id="$2"
  local prompt="$3"
  local episode_length="$4"

  local policy_path="${CKPT_ROOT}/${step}/pretrained_model"
  local out_dir="${EVAL_ROOT}/step_${step}/${env_id}"
  local key="${step}|${env_id}"

  if [[ -f "${out_dir}/eval_info.json" ]]; then
    echo "=== skip (exists) step=${step} ${env_id}"
    STATUS["${key}"]=ok
  else
    mkdir -p "${out_dir}"
    local -a cmd=(
      lerobot-eval
      --policy.path="${policy_path}"
      --policy.device="${DEVICE}"
      --policy.inference_action_mode="${INFERENCE_ACTION_MODE}"
      --trust_remote_code=true
      --env.type=maniskill
      --env.control_mode=pd_ee_pose
      --env.task="${env_id}::${prompt}"
      --env.episode_length="${episode_length}"
      --env.observation_height="${OBS_SIZE}"
      --env.observation_width="${OBS_SIZE}"
      --eval.n_episodes="${N_EPISODES}"
      --eval.batch_size="${BATCH_SIZE}"
      --eval.max_episodes_rendered=0
      --output_dir="${out_dir}"
    )
    if [[ "${env_id}" == "PickCube-v2-wrist" ]]; then
      cmd+=(--env.goal_height=0.1)
    fi

    echo
    echo "=== step=${step} ${env_id} n_episodes=${N_EPISODES} batch_size=${BATCH_SIZE}"
    echo "${cmd[*]}"
    if "${cmd[@]}"; then
      STATUS["${key}"]=ok
    else
      echo "FAILED: step=${step} ${env_id}" >&2
      STATUS["${key}"]=fail
      return 0
    fi
  fi

  if [[ -f "${out_dir}/eval_info.json" ]]; then
    RESULTS["${key}"]="$(python3 -c "
import json
info = json.load(open('${out_dir}/eval_info.json'))
print(f\"{info['overall']['pc_success']:.1f}\")
")"
  else
    RESULTS["${key}"]="NA"
    STATUS["${key}"]=fail
  fi
}

write_md() {
  {
    echo "# MPC VLA checkpoint results"
    echo
    echo "- Train dir: \`${TRAIN_DIR}\`"
    echo "- Episodes / task: ${N_EPISODES}"
    echo "- Eval batch size: ${BATCH_SIZE}"
    echo "- Control mode: \`pd_ee_pose\`"
    echo "- Observation size: ${OBS_SIZE}x${OBS_SIZE}"
    echo "- Inference action mode: \`${INFERENCE_ACTION_MODE}\`"
    echo
    echo "| step | $(printf '%s | ' "${TASKS[@]%%|*}" | sed 's/ | $//') | mean |"
    echo "| ---: | $(for _ in "${TASKS[@]}"; do printf ' ---: |'; done) ---: |"

    for step in "${STEPS[@]}"; do
      row="| ${step}"
      sum=0
      n=0
      for spec in "${TASKS[@]}"; do
        IFS='|' read -r env_id _ _ <<<"${spec}"
        key="${step}|${env_id}"
        val="${RESULTS[${key}]:-}"
        if [[ -z "${val}" && -f "${EVAL_ROOT}/step_${step}/${env_id}/eval_info.json" ]]; then
          val="$(python3 -c "
import json
info = json.load(open('${EVAL_ROOT}/step_${step}/${env_id}/eval_info.json'))
print(f\"{info['overall']['pc_success']:.1f}\")
")"
          RESULTS["${key}"]="${val}"
        fi
        if [[ -z "${val}" ]]; then
          val="—"
        elif [[ "${val}" != "NA" ]]; then
          sum="$(python3 -c "print(${sum}+${val})")"
          n=$((n + 1))
          val="${val}%"
        fi
        row+=" | ${val}"
      done
      if [[ "${n}" -gt 0 ]]; then
        mean="$(python3 -c "print(f'{${sum}/${n}:.1f}%')")"
      else
        mean="—"
      fi
      echo "${row} | ${mean} |"
    done
  } > "${RESULTS_MD}"
  echo "Wrote ${RESULTS_MD}"
}

for step in "${STEPS[@]}"; do
  for spec in "${TASKS[@]}"; do
    IFS='|' read -r env_id episode_length prompt <<<"${spec}"
    run_one "${step}" "${env_id}" "${prompt}" "${episode_length}"
    write_md
  done
done

echo
echo "Done. Results: ${RESULTS_MD}"
