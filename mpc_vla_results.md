# MPC VLA checkpoint results

- Train dir: `/home/jstm/Projects/lerobot_colosseum_v2/outputs/train/molmoact2_mpc_pd_ee_pose_200`
- Episodes / task: 100
- Eval batch size: 20
- Control mode: `pd_ee_pose`
- Observation size: 378x378
- Inference action mode: `continuous`

| step | PickCube-v2-wrist | PushCube-v2 | LiftPegUpright-v2 | PullCubeTool-v2 | mean |
| ---: |  ---: | ---: | ---: | ---: | ---: |
| 002000 | 2.0% | 0.0% | 0.0% | 0.0% | 0.5% |
| 004000 | 7.0% | 5.0% | 0.0% | 0.0% | 3.0% |
| 006000 | 6.0% | 6.0% | 0.0% | 1.0% | 3.2% |
| 008000 | 6.0% | 5.0% | 0.0% | 2.0% | 3.2% |
| 010000 | 7.0% | 5.0% | 0.0% | 0.0% | 3.0% |
