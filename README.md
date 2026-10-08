![Colosseum V2 Tasks and Perturbations](https://jstmn.github.io/colosseum-v2-website/content/ColosseumV2_hero.jpg)



## Overview

This repository provides the **first native ManiSkill simulator implementation** for [LeRobot](https://github.com/huggingface/lerobot), enabling evaluation of robotic policies on the [Colosseum V2 benchmark](https://jstmn.github.io/colosseum-v2-website/).

### Key Features

- Native ManiSkill environment support for LeRobot training and evaluation
- Full Colosseum V2 benchmark support with 16 single-arm and 12 bimanual tasks
- Support for all perturbation sets (visual, physical, and language)
- Automatic per-task episode length based on training data statistics
- Mass evaluation script with checkpoint resumption and real-time CSV logging



## Supported Tasks



### Single-Arm Tasks (16)

- RaiseCube
- PickSodaFromCabinet
- PickDishFromRack
- StackCube
- PlaceBookInShelf
- PlaceDishInRack
- LiftPegUpright
- RotateArrow
- PegInsertionSide
- PlugCharger
- HammerNail
- ScoopBanana
- OpenDrawer
- OpenCabinet
- PlaceCubeInDrawer
- CookItemInPan



### Bimanual Tasks (12)

- DualArmPickCube
- DualArmPickBottle
- DualArmLiftPot
- DualArmLiftTray
- DualArmPushBox
- DualArmPourPot
- DualArmThreading
- DualArmPenCap
- DualArmDrawerPlace
- DualArmDrawerOpen
- DualArmStackCube
- DualArmStack3Cube



## Installation

```bash
# FFmpeg
sudo apt install ffmpeg -y 
# ^ or module load GCCcore/14.3.0; module load FFmpeg/7.1.2 if on a cluster

git clone https://github.com/Geeksongs/lerobot_colosseum_v2.git
cd lerobot_colosseum_v2
conda create -n lerobot_cv2 python=3.12 pip -y
conda activate lerobot_cv2
pip install -e .[dataset,training]
pip install 'numpy<2'
conda install "ffmpeg" -c conda-forge
hf auth login

# ONLY IF YOU HAVE CUDA 12.x:
# The plain PyPI torchcodec wheel is built for CUDA 13 and fails to load
# ("libnvrtc.so.13 not found") with cu12x builds of torch, so install from the cu126 index:
pip install "torchcodec==0.15" --index-url https://download.pytorch.org/whl/cu126

# If using a 5090+:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Colosseum V2
git clone https://github.com/jstmn/ColosseumV2.git
pip install -e ColosseumV2
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1; python -m ColosseumV2.mani_skill.utils.download_asset ycb; python -m ColosseumV2.mani_skill.utils.download_asset RoboCasa; python -m ColosseumV2.mani_skill.utils.download_asset partnet_mobility
```

This will install ManiSkill with the required Colosseum V2 tasks and environments.

## Perturbation Sets

Colosseum V2 supports various perturbations for robustness testing:


| Category    | Sets                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| Object      | `MO_COLOR`, `MO_TEXTURE`, `MO_SIZE`, `MO_MASS`                         |
| Robot       | `RO_COLOR`, `RO_TEXTURE`, `RO_SIZE`                                    |
| Table       | `TABLE_COLOR`, `TABLE_TEXTURE`                                         |
| Environment | `CAMERA_POSE`, `LIGHT_COLOR`, `BACKGROUND_TEXTURE`, `BACKGROUND_COLOR` |
| Distractor  | `DISTRACTOR_OBJECT`                                                    |
| Combined    | `ALL`, `NONE`                                                          |




# Evaluation

The following examples demonstrate evaluation using the **MolmoAct2** model. Our framework also supports other policy architectures including:

- **X-VLA**
- **Pi0**
- **Pi0.5**
- **Pi0-Fast**
- **SmolVLA**
- **DiT-Policy**
- **MolmoAct2**
- **ACT**
- and more...

Any policy model compatible with the LeRobot framework can be evaluated on this benchmark. Note that the Pi0.5 model trained for the ColosseumV2 paper is not usable in this repository as there were several changes to the Pi0.5 model code base made during the lerobot v0.4.3 -> 0.6.0 update. To run the model from the paper, checkout the [lerobot_0.4.3](https://github.com/jstmn/lerobot_colosseum_v2/tree/lerobot_0.4.3) branch at [jstmn/lerobot_colosseum_v2](https://github.com/jstmn/lerobot_colosseum_v2) and follow the instructions in README.md.

For MolmoAct2 models, we provide pre-trained checkpoints that can be directly loaded from HuggingFace:

- Single-Arm: `jstm/molmoact2_single_arm`
- Bimanual: `jstm/molmoact2_bimanual`

For other policy architectures, users need to train their own models.

## Single Task Evaluation

Run evaluation on a single task:

**Single-Arm Task:**

Default environment:

```bash
lerobot-eval \
  --policy.path=jstm/molmoact2_single_arm \
  --env.type=maniskill \
  --env.task=RaiseCube-v1 \
  --env.episode_length=200 \
  --eval.n_episodes=50 \
  --eval.batch_size=25 \
  --policy.inference_action_mode=continuous \
  --trust_remote_code=true \
  --output_dir=outputs/molmoact2_single_arm__$(date +%Y-%m-%d--%H-%M-%S)
```

**With a perturbation set (e.g. MO_COLOR)**:

```bash
lerobot-eval \
  --policy.path=pythonsong/pi05_single_arm \
  --env.type=maniskill \
  --env.task=PickCube-v1 \
  --env.perturbation_set=MO_COLOR \
  --eval.n_episodes=200 \
  --eval.batch_size=100 \
  --output_dir=/path/to/outputs
```

**Bimanual Task:**

```bash
lerobot-eval \
  --policy.path=TODO:TRAIN A MODEL \
  --env.type=maniskill \
  --env.task=DualArmPickCube-v1 \
  --env.episode_length=214 \
  --eval.n_episodes=200 \
  --eval.batch_size=100 \
  --policy.compile_model=false \
  --trust_remote_code=true \
  --output_dir=/path/to/outputs
```

**With a perturbation set (e.g. MO_COLOR)**:

```bash
lerobot-eval \
  --policy.path=TODO:TRAIN A MODEL \
  --env.type=maniskill \
  --env.task=DualArmPickCube-v1 \
  --env.perturbation_set=MO_COLOR \
  --eval.n_episodes=200 \
  --eval.batch_size=100 \
  --output_dir=/path/to/outputs
```



## Full ColosseumV2 Evaluation

Evaluate on all Colosseum V2 tasks and perturbation sets combinations using the `run_mass_eval_fast.py` script.

Set `POLICY` to `molmoact2` or `pi05` (Hub org is `jstm` for MolmoAct2, `pythonsong` for π0.5):

```bash
export POLICY=molmoact2  # or: pi05
if [ "$POLICY" = "pi05" ]; then ORG=pythonsong; else ORG=jstm; fi
```


| Parameter                   | Description                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--policy_path`             | HuggingFace model path                                                                                             |
| `--task_type`               | `single_arm` or `bimanual`                                                                                         |
| `--batch_size`              | Number of parallel environments                                                                                    |
| `--n_episodes`              | Episodes per task                                                                                                  |
| `--output_dir`              | Output directory for results                                                                                       |
| `--generate-episode-videos` | Directory for website episode videos (forces`batch_size=1`, `human_render_shader=rt`; writes `video_outcomes.csv`) |


**Single-Arm:**

```bash
# export CUDA_VISIBLE_DEVICES=0
python scripts/run_mass_eval_fast.py \
  --policy_path ${ORG}/${POLICY}_single_arm \
  --task_type single_arm \
  --batch_size 15 \
  --n_episodes 200 \
  --output_dir outputs/mass_eval_single_arm_${POLICY} \
  --validate_config

# Website episode videos (batch_size forced to 1, human_render_shader=rt)
python scripts/run_mass_eval_fast.py \
  --policy_path ${ORG}/${POLICY}_single_arm \
  --task_type single_arm \
  --batch_size 1 \
  --n_episodes 10 \
  --output_dir outputs/mass_eval_single_arm_${POLICY}_videos \
  --generate-episode-videos outputs/website_videos/${POLICY}_single_arm
```

**Bimanual:**

```bash
# export CUDA_VISIBLE_DEVICES=1
python scripts/run_mass_eval_fast.py \
  --policy_path ${ORG}/${POLICY}_bimanual \
  --task_type bimanual \
  --validate_config \
  --batch_size 25 \
  --n_episodes 200 \
  --output_dir outputs/mass_eval_bimanual_${POLICY}

# Website episode videos (batch_size forced to 1, human_render_shader=rt)
python scripts/run_mass_eval_fast.py \
  --policy_path ${ORG}/${POLICY}_bimanual \
  --task_type bimanual \
  --batch_size 1 \
  --n_episodes 10 \
  --output_dir outputs/mass_eval_bimanual_${POLICY}_videos \
  --generate-episode-videos outputs/website_videos/${POLICY}_bimanual
```



## Training

Train your own policy model on Colosseum V2 datasets:

**Single-Arm:**

```bash

# Pi0.5
lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-single-arm-jan27 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/pi05__single_arm/$(date +%Y-%m-%d--%H-%M-%S) \
  --job_name=pi05_training_single_arm \
  --policy.repo_id=pythonsong/pi05_single_arm \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --policy.dtype=bfloat16 \
  --steps=30000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.device=cuda \
  --batch_size=1 \
  --save_freq=1000

# MolmoAct2
lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-single-arm-jan27 \
  --dataset.revision=main \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.action_mode=continuous \
  --policy.train_action_expert_only=true \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.setup_type="single franka robotic arm in maniskill" \
  --policy.control_mode="delta end-effector pose" \
  --policy.image_keys='["observation.images.external1_camera","observation.images.external2_camera","observation.images.hand_camera"]' \
  --policy.device=cuda \
  --output_dir=PATH/TO/molmoact2_single_arm__$(date +%Y-%m-%d--%H-%M-%S) \
  --job_name=molmoact2_training_single_arm \
  --policy.repo_id=jstm/molmoact2_single_arm \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --steps=10000 \
  --batch_size=32 \
  --num_workers=32 \
  --log_freq=20 \
  --env_eval_freq=-1 \
  --save_checkpoint=true \
  --save_freq=1000
```

**Bimanual:**

```bash
lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-bimanual-jan27 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=/path/to/outputs/bimanual \
  --job_name=pi05_training_bimanual__$(date +%Y-%m-%d--%H-%M-%S) \
  --policy.repo_id=pythonsong/pi05_bimanual \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --policy.dtype=bfloat16 \
  --steps=30000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.device=cuda \
  --batch_size=8 \
  --save_freq=1000000000

# MolmoAct2
lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-bimanual-jan27 \
  --dataset.revision=main \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.action_mode=continuous \
  --policy.train_action_expert_only=true \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.setup_type="two franka robotic arms in maniskill" \
  --policy.control_mode="delta end-effector pose" \
  --policy.image_keys='["observation.images.external1_camera","observation.images.external2_camera","observation.images.panda1_hand_camera","observation.images.panda2_hand_camera"]' \
  --policy.device=cuda \
  --output_dir=PATH/TO/outputs/molmoact2_bimanual__$(date +%Y-%m-%d--%H-%M-%S) \
  --job_name=molmoact2_training_bimanual \
  --policy.repo_id=jstm/molmoact2_bimanual \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --steps=10000 \
  --batch_size=32 \
  --num_workers=32 \
  --log_freq=20 \
  --env_eval_freq=-1 \
  --save_checkpoint=true \
  --save_freq=1000
```



## Results

Results are saved to a CSV file with columns:


| Column                   | Description         |
| ------------------------ | ------------------- |
| `env_id`                 | Task name           |
| `perturbation_set`       | Perturbation type   |
| `num_eval_episodes`      | Total episodes      |
| `num_sucessful_episodes` | Successful episodes |
| `success_percent`        | Success rate (%)    |




## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face Robotics Library
- [ManiSkill](https://github.com/haosulab/ManiSkill) - GPU-parallelized robotics simulator
- [Colosseum V2](https://jstmn.github.io/colosseum-v2-website/) - Robotic manipulation benchmark



## Citation

If you use this work, please cite:

```bibtex
@misc{morgan2026colosseumv2,
  title={Colosseum V2: Benchmarking Generalization for Vision Language Action Models},
  author={Jeremy Morgan and Prajwal Vijay and Hyeonho Oh and Jincen Song and Ashvin Arora and Alina Du and Gaurav Sukhatme and Jesse Thomason and Ishika Singh},
  year={2026},
  eprint={2605.27759},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2605.27759}
}
```



## License

Apache 2.0 License


# MPC Development

### MolmoAct2 finetune dataset

Collect N motion-planning demos from each MPC tabletop env, replay them to absolute EE pose (`pd_ee_pose`), merge, then convert to LeRobot. MolmoAct2's vision encoder is **378×378**, so the exporter letterboxes frames to a square 378×378.

```bash
conda activate lerobot_cv2

# N=100 per task (default). Writes local files, then uploads to
#   https://huggingface.co/datasets/jstm/mpc_lerobot_pd_ee_pose_<N>
# Cameras: camera_center, camera_left, camera_wrist (MolmoAct2 single-arm).
# --data-dir is required (must already exist); ManiSkill demos + LeRobot
# export go under it.
bash scripts/create_mpc_lerobot_dataset.sh --data-dir "$LEROBOT_DATA_DIR" -n 50 \
  --included-cameras "camera_center camera_left camera_wrist"

# Smoke test (N must be greater than --num-procs)
bash scripts/create_mpc_lerobot_dataset.sh --data-dir "$LEROBOT_DATA_DIR" -n 6 --num-procs 2 \
  --included-cameras "camera_center camera_left camera_wrist"
```

Required: `--data-dir DIR`. Optional: `--num-procs P`, `--envs "PickCube-v2-wrist PushCube-v2"`, `--output-dir DIR`, `--image-size 378x378`, `--repo-id USER/NAME`, `--no-upload`.

Pipeline (same as `ColosseumV2/scripts/data_generation/motionplanning_colosseum_v2_single_arm.sh`, plus LeRobot export):

1. `mani_skill/examples/motionplanning/panda/run.py` — collect `pd_joint_pos` successes
2. `mani_skill/trajectory/replay_trajectory.py --target-control-mode pd_ee_pose`
3. `mani_skill/trajectory/merge_multitask_trajectories.py`
4. `mani_skill/trajectory/convert_to_lerobot.py` — square 378×378 videos + per-episode task language

Then finetune:

```bash
N=6
DATASET_REPO_ID=jstm/mpc_lerobot_pd_ee_pose_${N}
DATASET_ROOT=${LEROBOT_DATA_DIR}/mpc_lerobot_pd_ee_pose_${N}

lerobot-train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.action_mode=continuous \
  --policy.train_action_expert_only=true \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.setup_type="single franka robotic arm in maniskill" \
  --policy.control_mode="absolute end-effector pose" \
  --policy.image_keys='["observation.images.camera_center","observation.images.camera_left","observation.images.camera_wrist"]' \
  --policy.device=cuda \
  --output_dir=${LEROBOT_DATA_DIR}/molmoact2_mpc__$(date +%Y-%m-%d--%H-%M-%S) \
  --job_name=molmoact2_mpc \
  --policy.repo_id=jstm/molmoact2_mpc \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --steps=10000 \
  --batch_size=32 \
  --num_workers=32 \
  --log_freq=20 \
  --env_eval_freq=-1 \
  --save_checkpoint=true \
  --save_freq=1000
```

Use the three MPC cameras above (or remap to `external1_camera` / `external2_camera` / `hand_camera` at eval, same as below).


### pi0.5

The envs from `mpcm/envs` are registered by `ColosseumV2/mani_skill/envs/mpcm` when ManiSkill is imported:


| Env ID              | Episode length | Task language                                                                    |
| ------------------- | -------------- | -------------------------------------------------------------------------------- |
| `PickCube-v2-wrist` | 100            | lift the red cube                                                                |
| `PushCube-v2`       | 80             | push the cube to the goal                                                        |
| `LiftPegUpright-v2` | 225            | lift the peg upright                                                             |
| `PullCubeTool-v2`   | 350            | grasp the red L-shaped hook and use it to pull the blue cube closer to the robot |


`lerobot/pi05_base` reads `observation.images.base_0_rgb`, `observation.images.left_wrist_0_rgb`, and `observation.images.right_wrist_0_rgb`. `jstm/molmoact2_single_arm` reads `observation.images.external1_camera`, `observation.images.external2_camera`, and `observation.images.hand_camera`. The MPC envs expose their own cameras; remapped with `--rename_map`:

**pi0.5**
- `camera_center` → `base_0_rgb`
- `camera_left` → `left_wrist_0_rgb`
- `camera_wrist` → `right_wrist_0_rgb`

**MolmoAct2**
- `camera_center` → `external1_camera`
- `camera_left` → `external2_camera`
- `camera_wrist` → `hand_camera`



The checkpoint's default device is `mps`. On this machine pass `--policy.device=cuda`. `lerobot/pi05_base` predicts 32-D padded actions; the Franka layout is 7 joint velocities plus gripper at index 7. Pass `--env.control_mode=pd_joint_vel --env.action_dim=8` so eval keeps those 8 dims instead of slicing the first 7 into `pd_ee_delta_pose`. `panda_wristcam2` gripper targets are meters in `[-0.02, 0.04]`. `--env.normalized_gripper_range=[closed_cmd,open_cmd]` maps the last action dim (and finger qpos in state) onto that. DROID / `pi05_base` is `[1,0]` (`1`=closed, `0`=open). `PickCube-v2-wrist` success is `cube.z > goal_height` (default env value is `0.2`). Pass `--env.goal_height=0.1` to match the 10 cm lift used in MPC eval. Pointcloud observations are not supported (`is_pointcloud=True` raises). `--generate-episode-videos DIR` forces `batch_size=1` and `human_render_shader=rt`, writes videos under `DIR`, and appends `DIR/video_outcomes.csv`.

50-episode evals (`batch_size=2`) then 10 recorded episodes per task, for both policies: `scripts/eval_mpc_tabletop.sh`.

```bash
conda activate lerobot_cv2
cd /home/jstm/Projects/lerobot_colosseum_v2
pip install 'transformers>=5.4.0,<5.6.0'  # pi0.5 extra; skip if already installed

# all cameras: camera_wrist, camera_left, camera_low_neg_y, camera_center, camera_right

RENAME_MAP='{"observation.images.camera_center":"observation.images.base_0_rgb","observation.images.camera_left":"observation.images.left_wrist_0_rgb","observation.images.camera_wrist":"observation.images.right_wrist_0_rgb"}'

# Repeat with the other rows of the table (task string and --env.episode_length).
# Action: 8-D pd_joint_vel = 7 arm joint velocities, gripper at index 7.
lerobot-eval \
  --policy.path=lerobot/pi05_base \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --env.type=maniskill \
  --env.control_mode=pd_joint_vel \
  --env.action_dim=8 \
  --env.normalized_gripper_range=[1,0] \
  --env.goal_height=0.1 \
  --env.task="PickCube-v2-wrist::lift the red cube" \
  --env.episode_length=100 \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.max_episodes_rendered=0 \
  --rename_map="$RENAME_MAP" \
  --output_dir=outputs/pi05_base__PickCube-v2-wrist

# Episode videos (forces batch_size=1, human_render_shader=rt; writes video_outcomes.csv)
# Action: 8-D pd_joint_vel = 7 arm joint velocities, gripper at index 7.
lerobot-eval \
  --policy.path=lerobot/pi05_base \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --env.type=maniskill \
  --env.control_mode=pd_joint_vel \
  --env.action_dim=8 \
  --env.normalized_gripper_range=[1,0] \
  --env.goal_height=0.1 \
  --env.episode_length=100 \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --env.task="PickCube-v2-wrist::lift the red cube" \
  --eval.n_episodes=5 \
  --eval.batch_size=1 \
  --rename_map="$RENAME_MAP" \
  --output_dir=outputs/pi05_base__PickCube-v2-wrist \
  --generate-episode-videos outputs/pi05_base__PickCube-v2-wrist/videos

# MolmoAct2 (single-arm Colosseum checkpoint; remap MPC cameras onto its Colosseum keys)
MOLMO_RENAME_MAP='{"observation.images.camera_center":"observation.images.external1_camera","observation.images.camera_left":"observation.images.external2_camera","observation.images.camera_wrist":"observation.images.hand_camera"}'

lerobot-eval \
  --policy.path=jstm/molmoact2_single_arm \
  --policy.inference_action_mode=continuous \
  --policy.device=cuda \
  --trust_remote_code=true \
  --env.type=maniskill \
  --env.episode_length=100 \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --rename_map="$MOLMO_RENAME_MAP" \
  --env.goal_height=0.1 \
  --env.task="PickCube-v2-wrist::grasp and raise the red cube" \
  --output_dir=outputs/molmoact2__PickCube-v2-wrist \
  --generate-episode-videos outputs/molmoact2__PickCube-v2-wrist/videos


# --env.task="PushCube-v2::push the cube to the goal region" \
#   --output_dir=outputs/molmoact2__PushCube-v2 \
#   --generate-episode-videos outputs/molmoact2__PushCube-v2/videos
```

