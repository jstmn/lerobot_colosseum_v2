<p align="center">
  <img alt="Colosseum V2 Tasks and Perturbations" src="https://jstmn.github.io/colosseum-v2-website/content/ColosseumV2_hero.jpg" width="100%">
</p>

<div align="center">

# LeRobot Colosseum V2

**The First Native ManiSkill Support for LeRobot**

[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)

[Paper](https://jstmn.github.io/colosseum-v2-website/) | [ManiSkill](https://github.com/haosulab/ManiSkill) | [LeRobot](https://github.com/huggingface/lerobot)

</div>

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
git clone https://github.com/Geeksongs/lerobot_colosseum_v2.git
cd lerobot_colosseum_v2
conda create -n lerobot_cv2 python=3.12
conda activate lerobot_cv2
pip install -e .
pip install 'numpy<2'
conda install "ffmpeg" -c conda-forge
# Note: the plain PyPI torchcodec wheel is built for CUDA 13 and fails to load
# ("libnvrtc.so.13 not found") with cu12x builds of torch, so install from the cu126 index:
pip install "torchcodec==0.15" --index-url https://download.pytorch.org/whl/cu126
huggingface-cli login

# If using a 5090+:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Colosseum V2 Dependencies

To evaluate on Colosseum V2 benchmark, first install the Colosseum V2 repository:

```bash
git clone https://github.com/jstmn/ColosseumV2.git
pip install -e ColosseumV2
```

This will install ManiSkill with the required Colosseum V2 tasks and environments.

## Perturbation Sets

Colosseum V2 supports various perturbations for robustness testing:

| Category | Sets |
|----------|------|
| Object | `MO_COLOR`, `MO_TEXTURE`, `MO_SIZE`, `MO_MASS` |
| Robot | `RO_COLOR`, `RO_TEXTURE`, `RO_SIZE` |
| Table | `TABLE_COLOR`, `TABLE_TEXTURE` |
| Environment | `CAMERA_POSE`, `LIGHT_COLOR`, `BACKGROUND_TEXTURE`, `BACKGROUND_COLOR` |
| Distractor | `DISTRACTOR_OBJECT` |
| Combined | `ALL`, `NONE` |

## Single Task Evaluation

Run evaluation on a single task using `lerobot-eval`:

### Single-Arm Task
```bash
lerobot-eval \
  --policy.path=pythonsong/pi05_single_arm \
  --env.type=maniskill \
  --env.task=RaiseCube-v1 \
  --env.episode_length=200 \
  --eval.n_episodes=200 \
  --eval.batch_size=100 \
  --policy.compile_model=false \
  --trust_remote_code=true \
  --output_dir=/path/to/outputs
```

### Bimanual Task
```bash
lerobot-eval \
  --policy.path=pythonsong/pi05_bimanual \
  --env.type=maniskill \
  --env.task=DualArmPickCube-v1 \
  --env.episode_length=214 \
  --eval.n_episodes=200 \
  --eval.batch_size=100 \
  --policy.compile_model=false \
  --trust_remote_code=true \
  --output_dir=/path/to/outputs
```

### With Perturbation Set
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

## Mass Evaluation

The following examples demonstrate evaluation using the **Pi0.5 (pi05)** model. Our framework also supports other policy architectures including:
- **X-VLA**
- **Pi0**
- **Pi0-Fast**
- **SmolVLA**
- **DiT-Policy**
- **ACT**
- and more...

Any policy model compatible with the LeRobot framework can be evaluated on this benchmark.

For Pi0.5 models, we provide pre-trained checkpoints that can be directly loaded from HuggingFace:
- Single-Arm: `pythonsong/pi05_single_arm`
- Bimanual: `pythonsong/pi05_bimanual`

For other policy architectures, users need to train their own models.

### Single-Arm Evaluation
```bash
python scripts/run_mass_eval.py \
  --policy_path pythonsong/pi05_single_arm \
  --task_type single_arm \
  --batch_size 100 \
  --n_episodes 200 \
  --use_per_task_episode_length \
  --output_dir /path/to/outputs/mass_eval_single_arm
```

### Bimanual Evaluation
```bash
python scripts/run_mass_eval.py \
  --policy_path pythonsong/pi05_bimanual \
  --task_type bimanual \
  --batch_size 100 \
  --n_episodes 200 \
  --use_per_task_episode_length \
  --output_dir /path/to/outputs/mass_eval_bimanual
```

### Mass Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `--policy_path` | HuggingFace model path |
| `--task_type` | `single_arm` or `bimanual` |
| `--batch_size` | Number of parallel environments |
| `--n_episodes` | Episodes per task |
| `--use_per_task_episode_length` | Use optimal episode length from training data |
| `--output_dir` | Output directory for results |

## Training

Train your own policy model on Colosseum V2 datasets:

### Single-Arm Training
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
  --batch_size=8 \
  --save_freq=1000000000

# MolmoAct2
lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-single-arm-jan27 \
  --dataset.revision=main \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --output_dir=outputs/molmoact2_single_arm__$(date +%Y-%m-%d--%H-%M-%S) \
  --job_name=molmoact2_training_single_arm \
  --policy.repo_id=jstm/molmoact2_single_arm \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --steps=10000 \
  --batch_size=32 \
  --num_workers=4 \
  --log_freq=20 \
  --env_eval_freq=-1 \
  --save_checkpoint=true \
  --save_freq=2000

  # Not supported by MolmoAct2
  # --policy.compile_model=true \
  # --policy.dtype=bfloat16 \
```

### Bimanual Training
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
```

## Results

Results are saved to a CSV file with columns:

| Column | Description |
|--------|-------------|
| `env_id` | Task name |
| `perturbation_set` | Perturbation type |
| `num_eval_episodes` | Total episodes |
| `num_sucessful_episodes` | Successful episodes |
| `success_percent` | Success rate (%) |

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
