#!/usr/bin/env bash
# 50-episode eval (batch_size=2) then 10 recorded episodes per MPC tabletop task,
# for lerobot/pi05_base and jstm/molmoact2_single_arm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v lerobot-eval >/dev/null 2>&1; then
  echo "lerobot-eval not found. Activate the lerobot_cv2 conda env first." >&2
  exit 1
fi

PI_RENAME_MAP='{"observation.images.camera_center":"observation.images.base_0_rgb","observation.images.camera_left":"observation.images.left_wrist_0_rgb","observation.images.camera_wrist":"observation.images.right_wrist_0_rgb"}'
MOLMO_RENAME_MAP='{"observation.images.camera_center":"observation.images.external1_camera","observation.images.camera_left":"observation.images.external2_camera","observation.images.camera_wrist":"observation.images.hand_camera"}'

# env_id|episode_length|prompt
TASKS=(
  "PickCube-v2-wrist|100|lift the red cube"
  "PushCube-v2|80|push the cube to the goal"
  "LiftPegUpright-v2|225|lift the peg upright"
  "PullCubeTool-v2|350|grasp the red L-shaped hook and use it to pull the blue cube closer to the robot"
)

POLICIES=(pi05 molmoact2)

run_eval() {
  local policy_name="$1"
  local env_id="$2"
  local prompt="$3"
  local episode_length="$4"
  local n_episodes="$5"
  local batch_size="$6"
  local output_dir="$7"
  local videos_dir="${8:-}"

  local -a cmd=(
    lerobot-eval
    --policy.device=cuda
    --env.type=maniskill
    --env.task="${env_id}::${prompt}"
    --env.episode_length="${episode_length}"
    --env.observation_height=224
    --env.observation_width=224
    --eval.n_episodes="${n_episodes}"
    --eval.batch_size="${batch_size}"
    --output_dir="${output_dir}"
  )

  if [[ "${env_id}" == "PickCube-v2-wrist" ]]; then
    cmd+=(--env.goal_height=0.1)
  fi

  case "${policy_name}" in
    pi05)
      cmd+=(
        --policy.path=lerobot/pi05_base
        --policy.pretrained_path=lerobot/pi05_base
        --env.control_mode=pd_joint_vel
        --env.action_dim=8
        --env.normalized_gripper_range='[1,0]'
        --rename_map="${PI_RENAME_MAP}"
      )
      ;;
    molmoact2)
      cmd+=(
        --policy.path=jstm/molmoact2_single_arm
        --policy.inference_action_mode=continuous
        --trust_remote_code=true
        --rename_map="${MOLMO_RENAME_MAP}"
      )
      ;;
    *)
      echo "unknown policy ${policy_name}" >&2
      exit 1
      ;;
  esac

  if [[ -n "${videos_dir}" ]]; then
    if [[ "${batch_size}" != "1" ]]; then
      echo "recording requires batch_size=1, got ${batch_size}" >&2
      exit 1
    fi
    cmd+=(--generate-episode-videos "${videos_dir}")
  else
    cmd+=(--eval.max_episodes_rendered=0)
  fi

  echo
  echo "=== ${policy_name} ${env_id} n_episodes=${n_episodes} batch_size=${batch_size} output_dir=${output_dir}"
  echo "${cmd[*]}"
  "${cmd[@]}"
}

for policy_name in "${POLICIES[@]}"; do
  for spec in "${TASKS[@]}"; do
    IFS='|' read -r env_id episode_length prompt <<<"${spec}"
    run_eval \
      "${policy_name}" \
      "${env_id}" \
      "${prompt}" \
      "${episode_length}" \
      50 \
      2 \
      "outputs/${policy_name}__${env_id}"
  done

  for spec in "${TASKS[@]}"; do
    IFS='|' read -r env_id episode_length prompt <<<"${spec}"
    run_eval \
      "${policy_name}" \
      "${env_id}" \
      "${prompt}" \
      "${episode_length}" \
      10 \
      1 \
      "outputs/${policy_name}__${env_id}__videos" \
      "outputs/${policy_name}__${env_id}__videos/videos"
  done
done
