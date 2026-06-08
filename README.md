<p align="center">
  <img alt="Colosseum V2 Tasks and Perturbations" src="https://jstmn.github.io/colosseum-v2-website/content/ColosseumV2_hero.jpg" width="100%">
</p>

<div align="center">

# LeRobot Colosseum V2

**The First ManiSkill Simulator Integration for LeRobot**

[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)

</div>

## Overview

This repository provides the **first native ManiSkill simulator integration** for [LeRobot](https://github.com/huggingface/lerobot), enabling evaluation of robotic policies on the [Colosseum V2 benchmark](https://jstmn.github.io/colosseum-v2-website/).

### Key Features

- Native ManiSkill environment support for LeRobot evaluation
- Full Colosseum V2 benchmark with 16 single-arm and 12 bimanual tasks
- Support for all distraction sets (texture, lighting, camera, object variations)
- Automatic per-task episode length based on training data statistics
- Mass evaluation script with checkpoint resumption and real-time CSV logging

## Supported Tasks

### Single-Arm Tasks (16)
| Task | Episode Steps |
|------|---------------|
| RaiseCube-v1 | 92 |
| PickSodaFromCabinet-v1 | 203 |
| PickDishFromRack-v1 | 156 |
| StackCubeColosseumV2-v1 | 143 |
| PlaceBookInShelf-v1 | 216 |
| PlaceDishInRack-v1 | 327 |
| LiftPegUprightColosseumV2-v1 | 224 |
| RotateArrow-v1 | 356 |
| PegInsertionSideColosseumV2-v1 | 267 |
| PlugChargerColosseumV2-v1 | 234 |
| HammerNail-v1 | 247 |
| ScoopBanana-v1 | 322 |
| OpenDrawer-v1 | 133 |
| OpenCabinet-v1 | 492 |
| PlaceCubeInDrawer-v1 | 387 |
| CookItemInPan-v1 | 533 |

### Bimanual Tasks (12)
| Task | Episode Steps |
|------|---------------|
| DualArmPickCube-v1 | 214 |
| DualArmPickBottle-v1 | 156 |
| DualArmLiftPot-v1 | 126 |
| DualArmLiftTray-v1 | 122 |
| DualArmPushBox-v1 | 131 |
| DualArmPourPot-v1 | 215 |
| DualArmThreading-v1 | 193 |
| DualArmPenCap-v1 | 232 |
| DualArmDrawerPlace-v1 | 202 |
| DualArmDrawerOpen-v1 | 119 |
| DualArmStackCube-v1 | 166 |
| DualArmStack3Cube-v1 | 284 |

## Installation

```bash
git clone https://github.com/Geeksongs/lerobot_colosseum_v2.git
cd lerobot_colosseum_v2
pip install -e .
```

## Mass Evaluation

Run evaluation across all Colosseum V2 tasks with automatic per-task episode lengths:

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

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--policy_path` | HuggingFace model path |
| `--task_type` | `single_arm` or `bimanual` |
| `--batch_size` | Number of parallel environments |
| `--n_episodes` | Episodes per task |
| `--use_per_task_episode_length` | Use optimal episode length from training data |
| `--output_dir` | Output directory for results |

## Results

Results are saved to a CSV file with columns:
- `env_id`: Task name
- `distraction_set`: Perturbation type
- `num_eval_episodes`: Total episodes
- `num_sucessful_episodes`: Successful episodes
- `success_percent`: Success rate (%)

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face Robotics Library
- [ManiSkill](https://github.com/haosulab/ManiSkill) - GPU-parallelized robotics simulator
- [Colosseum V2](https://jstmn.github.io/colosseum-v2-website/) - Robotic manipulation benchmark

## License

Apache 2.0 License
