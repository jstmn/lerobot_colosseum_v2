# MolmoAct2 Policy

MolmoAct2 is a Vision-Language-Action model developed by Allen Institute for AI (AI2). It is based on Molmo2-ER with a flow matching continuous action expert for robot manipulation tasks.

## Model Information

- **Model Size**: 5B parameters
- **Base Model**: Molmo2-ER
- **Action Prediction**: Flow Matching (continuous) or Discrete tokens
- **HuggingFace**: [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO)

## Features

- Two camera inputs (agentview + wrist)
- 8-dimensional robot state input
- 7-dimensional action output (6D EEF delta + gripper)
- Language-conditioned task execution
- Built-in normalization (no external stats needed)

## Usage

### Basic Evaluation

```bash
lerobot-eval \
  --policy.path=allenai/MolmoAct2-LIBERO \
  --policy.type=molmoact2 \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

### Python API

```python
from lerobot.policies.molmoact2 import MolmoAct2Policy, MolmoAct2Config

# Create config
config = MolmoAct2Config(
    hf_model_id="allenai/MolmoAct2-LIBERO",
    dtype="bfloat16",  # Use bfloat16 for lower memory
    n_action_steps=10,
)

# Load policy
policy = MolmoAct2Policy(config)
policy.to("cuda")
policy.eval()

# Prepare observation
batch = {
    "observation.images.agentview": agentview_tensor,  # (1, 3, H, W)
    "observation.images.wrist": wrist_tensor,          # (1, 3, H, W)
    "observation.state": state_tensor,                  # (1, 8)
    "task": "pick up the red block",
}

# Get action
action = policy.select_action(batch)  # (1, 7)
```

### Async Evaluation with LIBERO

```bash
python -m libero.lifelong.evaluate_async \
    --benchmark libero_goal \
    --task_id 0 \
    --model_type lerobot \
    --lerobot_model allenai/MolmoAct2-LIBERO \
    --n_eval 20 \
    --real-time-rate 1.0
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hf_model_id` | `allenai/MolmoAct2-LIBERO` | HuggingFace model ID |
| `dtype` | `bfloat16` | Model precision (`bfloat16` or `float32`) |
| `chunk_size` | `10` | Number of action steps to predict |
| `n_action_steps` | `10` | Number of action steps to execute |
| `inference_action_mode` | `continuous` | Action mode (`continuous` or `discrete`) |
| `norm_tag` | `libero` | Normalization tag for internal stats |
| `enable_cuda_graph` | `True` | Enable CUDA graph optimization |

## Memory Requirements

| Mode | GPU Memory |
|------|------------|
| Float32 + CUDA Graph | ~26 GB |
| Float32 | ~24 GB |
| BFloat16 | <16 GB |

## Citation

```bibtex
@article{molmoact2,
  title={MolmoAct2: A Vision-Language-Action Model},
  author={Allen Institute for AI},
  year={2025}
}
```
