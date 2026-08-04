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
- Support for all perturbation sets (texture, lighting, camera, object variations)
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
conda create -n lerobot_pi05 python=3.12 pip -y
conda activate lerobot_pi05
pip install -e .[dataset,training]

conda install "ffmpeg" -c conda-forge
pip install 'numpy<2'
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" # see https://github.com/huggingface/lerobot/issues/2305

hf auth login
```

### Colosseum V2 Dependencies

To evaluate on Colosseum V2 benchmark, first install the Colosseum V2 repository:

```bash
git clone https://github.com/jstmn/ColosseumV2.git
pip install -e ColosseumV2/
```

This will install ManiSkill with the required Colosseum V2 tasks and environments.

## Distraction Sets

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
  --env.episode_length=150 \
  --eval.n_episodes=50 \
  --eval.batch_size=50 \
  --policy.compile_model=false \
  --trust_remote_code=true \
  --output_dir=outputs_eval
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

### With Distraction Set
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
python scripts/run_mass_eval_fast.py \
  --policy_path pythonsong/pi05_single_arm \
  --task_type single_arm \
  --batch_size 50 \
  --n_episodes 50 \
  --validate_config \
  --output_dir outputs/mass_eval_single_arm_pi05
```

### Bimanual Evaluation
```bash
python scripts/run_mass_eval_fast.py \
  --policy_path pythonsong/pi05_bimanual \
  --task_type bimanual \
  --batch_size 50 \
  --n_episodes 50 \
  --validate_config \
  --output_dir outputs/mass_eval_bimanual_pi05
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
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=pythonsong/colosseum-single-arm-jan27 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=/path/to/outputs/single_arm \
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
```

### Bimanual Training
```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=pythonsong/colosseum-bimanual-jan27 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=/path/to/outputs/bimanual \
  --job_name=pi05_training_bimanual \
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
