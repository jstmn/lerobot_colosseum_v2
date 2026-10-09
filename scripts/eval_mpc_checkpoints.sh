#!/usr/bin/env bash
# Sweep LeRobot training checkpoints (plus the base init policy) on MPC tabletop
# tasks and upsert a per-N section into mpc_vla_results.md.
#
#   bash scripts/eval_mpc_checkpoints.sh --n 200
#   bash scripts/eval_mpc_checkpoints.sh \
#     --train-dir outputs/train/molmoact2_mpc_pd_ee_pose_200 \
#     --batch-size 20 \
#     --n-episodes 100
#
# Train dirs follow outputs/train/molmoact2_mpc_pd_ee_pose_<N>.
# On start the script syncs mpc_vla_results.md from any existing
# checkpoint_evals/**/eval_info.json, then continues missing evals.
#
# Defaults match the MolmoAct2 MPC pd_ee_pose finetune: cameras
# camera_{center,left,wrist} at 378x378, control_mode=pd_ee_pose.
# The base policy is read from the run's train_config.json
# (policy.checkpoint_path), e.g. allenai/MolmoAct2-LIBERO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_DIR=""
RESULTS_MD="${ROOT}/mpc_vla_results.md"
N=""
N_EPISODES=100
BATCH_SIZE=20
OBS_SIZE=378
DEVICE=cuda
INFERENCE_ACTION_MODE=continuous
INCLUDE_BASE=1
BASE_POLICY=""
BASE_NORM_TAG="libero"
BASE_IMAGE_KEYS='["observation.images.camera_center","observation.images.camera_left","observation.images.camera_wrist"]'
BASE_SETUP_TYPE="single franka robotic arm"
BASE_CONTROL_MODE="absolute end-effector pose"
DEFAULT_BASE_POLICY="allenai/MolmoAct2-LIBERO"

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
    --n) N="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --obs-size) OBS_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --inference-action-mode) INFERENCE_ACTION_MODE="$2"; shift 2 ;;
    --base-policy) BASE_POLICY="$2"; shift 2 ;;
    --base-norm-tag) BASE_NORM_TAG="$2"; shift 2 ;;
    --no-base) INCLUDE_BASE=0; shift ;;
    -h|--help)
      echo "Usage: $0 (--n N | --train-dir DIR) [--batch-size B] [--n-episodes E] [--base-policy ID] [--no-base]"
      echo "  --n              MPC demo count N. Train dir defaults to"
      echo "                   outputs/train/molmoact2_mpc_pd_ee_pose_<N>"
      echo "  --train-dir      Training run dir (checkpoints/ and/or checkpoint_evals/)"
      echo "  --batch-size      Parallel eval envs (default: 20)"
      echo "  --n-episodes      Eval episodes per task (default: 100)"
      echo "  --base-policy     HF/base checkpoint to eval as row 'base'"
      echo "                    (default: policy.checkpoint_path from train_config.json,"
      echo "                     else ${DEFAULT_BASE_POLICY})"
      echo "  --base-norm-tag   MolmoAct2 norm_tag for the base HF checkpoint (default: libero)"
      echo "  --no-base         Skip evaluating the base policy"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 1
fi

# Resolve N <-> train dir (convention: outputs/train/molmoact2_mpc_pd_ee_pose_<N>).
if [[ -z "${TRAIN_DIR}" && -n "${N}" ]]; then
  TRAIN_DIR="${ROOT}/outputs/train/molmoact2_mpc_pd_ee_pose_${N}"
fi
if [[ -z "${TRAIN_DIR}" ]]; then
  echo "Pass --n N or --train-dir DIR" >&2
  exit 1
fi
if [[ "${TRAIN_DIR}" != /* ]]; then
  TRAIN_DIR="${ROOT}/${TRAIN_DIR}"
fi
if [[ -z "${N}" ]]; then
  if [[ "$(basename "${TRAIN_DIR}")" =~ _([0-9]+)$ ]]; then
    N="${BASH_REMATCH[1]}"
  fi
fi
if [[ -z "${N}" ]]; then
  echo "Could not infer N from train dir name; pass --n <demo_count>" >&2
  exit 1
fi
if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--n must be a positive integer, got: ${N}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "Train dir not found: ${TRAIN_DIR}" >&2
  exit 1
fi

CKPT_ROOT="${TRAIN_DIR}/checkpoints"
EVAL_ROOT="${TRAIN_DIR}/checkpoint_evals"
mkdir -p "${EVAL_ROOT}"

if ! command -v lerobot-eval >/dev/null 2>&1; then
  echo "lerobot-eval not found. Activate the lerobot_cv2 conda env first." >&2
  exit 1
fi

# Steps from checkpoints/, falling back to existing checkpoint_evals/step_*.
mapfile -t STEPS < <(
  {
    if [[ -d "${CKPT_ROOT}" ]]; then
      find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
    fi
    if [[ -d "${EVAL_ROOT}" ]]; then
      find "${EVAL_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | sed -n 's/^step_//p'
    fi
  } | grep -E '^[0-9]+$' | sort -n | uniq
)

# Pull base policy + prompt metadata from the run's saved train config when possible.
TRAIN_CONFIG=""
if [[ -d "${CKPT_ROOT}" ]]; then
  for step in "${STEPS[@]:-}" last; do
    [[ -z "${step}" ]] && continue
    candidate="${CKPT_ROOT}/${step}/pretrained_model/train_config.json"
    if [[ -f "${candidate}" ]]; then
      TRAIN_CONFIG="${candidate}"
      break
    fi
  done
fi
DATASET_REPO_ID=""
if [[ -n "${TRAIN_CONFIG}" ]]; then
  while IFS= read -r line; do
    case "${line}" in
      BASE_POLICY=*)
        if [[ -z "${BASE_POLICY}" ]]; then
          BASE_POLICY="${line#BASE_POLICY=}"
        fi
        ;;
      BASE_IMAGE_KEYS=*) BASE_IMAGE_KEYS="${line#BASE_IMAGE_KEYS=}" ;;
      BASE_SETUP_TYPE=*) BASE_SETUP_TYPE="${line#BASE_SETUP_TYPE=}" ;;
      BASE_CONTROL_MODE=*) BASE_CONTROL_MODE="${line#BASE_CONTROL_MODE=}" ;;
      DATASET_REPO_ID=*) DATASET_REPO_ID="${line#DATASET_REPO_ID=}" ;;
    esac
  done < <(
    TRAIN_CONFIG="${TRAIN_CONFIG}" python3 - <<'PY'
import json
import os
from pathlib import Path

raw = json.loads(Path(os.environ["TRAIN_CONFIG"]).read_text())
cfg = raw["policy"]
if cfg.get("checkpoint_path"):
    print(f"BASE_POLICY={cfg['checkpoint_path']}")
if cfg.get("image_keys"):
    print(f"BASE_IMAGE_KEYS={json.dumps(cfg['image_keys'])}")
if cfg.get("setup_type"):
    print(f"BASE_SETUP_TYPE={cfg['setup_type']}")
if cfg.get("control_mode"):
    print(f"BASE_CONTROL_MODE={cfg['control_mode']}")
repo_id = (raw.get("dataset") or {}).get("repo_id") or ""
if repo_id:
    print(f"DATASET_REPO_ID={repo_id}")
PY
  )
fi

# Prefer dataset-derived N when present (overrides dir suffix if they differ).
if [[ "${DATASET_REPO_ID}" =~ _([0-9]+)$ ]]; then
  N="${BASH_REMATCH[1]}"
fi
echo "Train dir: ${TRAIN_DIR}"
echo "Results section: N=${N}"

ROWS=()
if [[ "${INCLUDE_BASE}" -eq 1 ]]; then
  if [[ -z "${BASE_POLICY}" ]]; then
    BASE_POLICY="${DEFAULT_BASE_POLICY}"
  fi
  ROWS+=("base")
  echo "Base policy: ${BASE_POLICY} (norm_tag=${BASE_NORM_TAG})"
fi
ROWS+=("${STEPS[@]:-}")

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "Nothing to eval: no base and no checkpoint/eval steps under ${TRAIN_DIR}" >&2
  exit 1
fi

# results[row|env_id] = pc_success
declare -A RESULTS
declare -A STATUS

row_out_dir() {
  local row="$1"
  local env_id="$2"
  if [[ "${row}" == "base" ]]; then
    echo "${EVAL_ROOT}/base/${env_id}"
  else
    echo "${EVAL_ROOT}/step_${row}/${env_id}"
  fi
}

load_existing_results() {
  local row env_id out_dir key
  for row in "${ROWS[@]}"; do
    for spec in "${TASKS[@]}"; do
      IFS='|' read -r env_id _ _ <<<"${spec}"
      out_dir="$(row_out_dir "${row}" "${env_id}")"
      key="${row}|${env_id}"
      if [[ -f "${out_dir}/eval_info.json" ]]; then
        RESULTS["${key}"]="$(python3 -c "
import json
info = json.load(open('''${out_dir}/eval_info.json'''))
print(f\"{info['overall']['pc_success']:.1f}\")
")"
        STATUS["${key}"]=ok
      fi
    done
  done
}

write_md() {
  local section_body
  section_body="$(
    {
      echo "## N=${N}"
      echo
      echo "- Train dir: \`${TRAIN_DIR}\`"
      if [[ -n "${DATASET_REPO_ID}" ]]; then
        echo "- Dataset: \`${DATASET_REPO_ID}\`"
      fi
      if [[ "${INCLUDE_BASE}" -eq 1 ]]; then
        echo "- Base policy: \`${BASE_POLICY}\` (norm_tag=\`${BASE_NORM_TAG}\`)"
      fi
      echo "- Episodes / task: ${N_EPISODES}"
      echo "- Eval batch size: ${BATCH_SIZE}"
      echo "- Control mode: \`pd_ee_pose\`"
      echo "- Observation size: ${OBS_SIZE}x${OBS_SIZE}"
      echo "- Inference action mode: \`${INFERENCE_ACTION_MODE}\`"
      echo
      echo "| step | $(printf '%s | ' "${TASKS[@]%%|*}" | sed 's/ | $//') | mean |"
      echo "| ---: | $(for _ in "${TASKS[@]}"; do printf ' ---: |'; done) ---: |"

      for md_row in "${ROWS[@]}"; do
        # Use md_line (not row) so we don't clobber the caller's loop variable.
        md_line="| ${md_row}"
        sum=0
        n=0
        for md_spec in "${TASKS[@]}"; do
          IFS='|' read -r env_id _ _ <<<"${md_spec}"
          key="${md_row}|${env_id}"
          val="${RESULTS[${key}]:-}"
          out_dir="$(row_out_dir "${md_row}" "${env_id}")"
          if [[ -z "${val}" && -f "${out_dir}/eval_info.json" ]]; then
            val="$(python3 -c "
import json
info = json.load(open('''${out_dir}/eval_info.json'''))
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
          md_line+=" | ${val}"
        done
        if [[ "${n}" -gt 0 ]]; then
          mean="$(python3 -c "print(f'{${sum}/${n}:.1f}%')")"
        else
          mean="—"
        fi
        echo "${md_line} | ${mean} |"
      done
    }
  )"

  RESULTS_MD="${RESULTS_MD}" SECTION_N="${N}" SECTION_BODY="${section_body}" python3 - <<'PY'
import os
import re
from pathlib import Path

path = Path(os.environ["RESULTS_MD"])
n = os.environ["SECTION_N"]
body = os.environ["SECTION_BODY"].rstrip() + "\n"
title = "# MPC VLA checkpoint results\n"

text = path.read_text() if path.exists() else ""
if not text.strip():
    text = title + "\n"

# Migrate a legacy single-table file (no ## N= headings) into one section.
if not re.search(r"(?m)^## N=\d+\s*$", text):
    legacy_n = n
    m = re.search(r"- Train dir: `([^`]+)`", text)
    if m:
        m2 = re.search(r"_([0-9]+)$", Path(m.group(1)).name)
        if m2:
            legacy_n = m2.group(1)
    rest = re.sub(r"(?m)^# MPC VLA checkpoint results\s*\n*", "", text).strip()
    text = f"{title}\n## N={legacy_n}\n\n{rest}\n" if rest else title + "\n"

pattern = re.compile(
    rf"(?ms)^## N={re.escape(n)}\s*\n.*?(?=^## N=\d+\s*$|\Z)"
)
if pattern.search(text):
    text = pattern.sub(body + "\n", text, count=1)
else:
    text = text.rstrip() + "\n\n" + body + "\n"

header_m = re.match(r"(?s)^(#[^\n]*\n\s*)", text)
header = header_m.group(1) if header_m else title + "\n"
sections = []
for part in re.split(r"(?m)(?=^## N=\d+\s*$)", text):
    m = re.match(r"(?ms)^## N=(\d+)\s*\n", part)
    if m:
        sections.append((int(m.group(1)), part.rstrip() + "\n"))
sections.sort(key=lambda x: x[0])
path.write_text(header + "\n".join(block for _, block in sections))
print(f"Wrote {path} (section N={n})")
PY
}

run_one() {
  local row="$1"
  local env_id="$2"
  local prompt="$3"
  local episode_length="$4"

  local out_dir key ckpt_path
  out_dir="$(row_out_dir "${row}" "${env_id}")"
  key="${row}|${env_id}"

  if [[ -f "${out_dir}/eval_info.json" ]]; then
    echo "=== skip (exists) row=${row} ${env_id}"
    STATUS["${key}"]=ok
  else
    if [[ "${row}" != "base" ]]; then
      ckpt_path="${CKPT_ROOT}/${row}/pretrained_model"
      if [[ ! -d "${ckpt_path}" ]]; then
        echo "=== skip (no checkpoint) row=${row} ${env_id} missing ${ckpt_path}" >&2
        RESULTS["${key}"]="${RESULTS[${key}]:-}"
        return 0
      fi
    fi

    mkdir -p "${out_dir}"
    local -a cmd=(
      lerobot-eval
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
    if [[ "${row}" == "base" ]]; then
      # LIBERO base norms use 8-D state (7 joints + 1 gripper); MPC default is 9.
      cmd+=(
        --policy.type=molmoact2
        --policy.checkpoint_path="${BASE_POLICY}"
        --policy.norm_tag="${BASE_NORM_TAG}"
        --policy.image_keys="${BASE_IMAGE_KEYS}"
        --policy.setup_type="${BASE_SETUP_TYPE}"
        --policy.control_mode="${BASE_CONTROL_MODE}"
        --env.state_dim=8
      )
    else
      cmd+=(--policy.path="${ckpt_path}")
    fi
    if [[ "${env_id}" == "PickCube-v2-wrist" ]]; then
      cmd+=(--env.goal_height=0.1)
    fi

    echo
    echo "=== row=${row} ${env_id} n_episodes=${N_EPISODES} batch_size=${BATCH_SIZE}"
    echo "${cmd[*]}"
    if "${cmd[@]}"; then
      STATUS["${key}"]=ok
    else
      echo "FAILED: row=${row} ${env_id}" >&2
      STATUS["${key}"]=fail
      return 0
    fi
  fi

  if [[ -f "${out_dir}/eval_info.json" ]]; then
    RESULTS["${key}"]="$(python3 -c "
import json
info = json.load(open('''${out_dir}/eval_info.json'''))
print(f\"{info['overall']['pc_success']:.1f}\")
")"
  else
    RESULTS["${key}"]="NA"
    STATUS["${key}"]=fail
  fi
}

echo "Syncing ${RESULTS_MD} from existing evals under ${EVAL_ROOT} ..."
load_existing_results
write_md

echo "Continuing missing evals ..."
for row in "${ROWS[@]}"; do
  for spec in "${TASKS[@]}"; do
    IFS='|' read -r env_id episode_length prompt <<<"${spec}"
    run_one "${row}" "${env_id}" "${prompt}" "${episode_length}"
    write_md
  done
done

echo
echo "Done. Results: ${RESULTS_MD}"
