# MPC VLA checkpoint results

## N=50

- Train dir: `/home/exouser/lerobot_colosseum_v2/outputs/train/molmoact2_mpc_pd_ee_pose_50`
- Dataset: `jstm/mpc_lerobot_pd_ee_pose_50`
- Base policy: `allenai/MolmoAct2-LIBERO` (norm_tag=`libero`)
- Episodes / task: 100
- Eval batch size: 20
- Control mode: `pd_ee_pose`
- Observation size: 378x378
- Inference action mode: `continuous`

| step | PickCube-v2-wrist | PushCube-v2 | LiftPegUpright-v2 | PullCubeTool-v2 | mean |
| ---: |  ---: | ---: | ---: | ---: | ---: |
| base | 0.0% | — | — | — | 0.0% |
| 002000 | 5.0% | — | — | — | 5.0% |
| 004000 | 5.0% | — | — | — | 5.0% |
| 006000 | 7.0% | — | — | — | 7.0% |
| 008000 | 9.0% | — | — | — | 9.0% |
| 010000 | 6.0% | — | — | — | 6.0% |

## N=200

- Train dir: `/home/exouser/lerobot_colosseum_v2/outputs/train/molmoact2_mpc_pd_ee_pose_200`
- Base policy: `allenai/MolmoAct2-LIBERO` (norm_tag=`libero`)
- Episodes / task: 100
- Eval batch size: 20
- Control mode: `pd_ee_pose`
- Observation size: 378x378
- Inference action mode: `continuous`

| step | PickCube-v2-wrist | PushCube-v2 | LiftPegUpright-v2 | PullCubeTool-v2 | mean |
| ---: |  ---: | ---: | ---: | ---: | ---: |
| base | 0.0% | 6.0% | 0.0% | 0.0% | 1.5% |
| 002000 | 2.0% | 0.0% | 0.0% | 0.0% | 0.5% |
| 004000 | 7.0% | 5.0% | 0.0% | 0.0% | 3.0% |
| 006000 | 6.0% | 6.0% | 0.0% | 1.0% | 3.2% |
| 008000 | 6.0% | 5.0% | 0.0% | 2.0% | 3.2% |
| 010000 | 7.0% | 5.0% | 0.0% | 0.0% | 3.0% |
